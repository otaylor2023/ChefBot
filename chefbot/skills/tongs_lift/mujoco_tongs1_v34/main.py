"""Initial sensor-only tongs geometry and grasp experiment."""
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

PADS = .103
BASE = [0.,0.,0.]

def row(F,g):
    q=Rotation.from_matrix(F[:3,:3]).as_quat()
    return [*F[:3,3],q[3],*q[:3],g]

def move(robot,F,g,n=60):
    r=robot.act([row(F,g)+BASE]*min(15,n))
    left=n-15
    prev=10.; stall=0
    while left>0:
        e=r['steps'][-1]; score=e['pos_error_m']+.1*e['rot_error_rad']
        if e['pos_error_m']<.003 and e['rot_error_rad']<.04: break
        stall=stall+1 if prev-score<.0004 else 0
        if stall>=4:break
        prev=score
        m=min(5,left); left-=m
        r=robot.act([row(F,g)+BASE]*m)
    p=r['proprio']
    robot.log('move',np.round(F[:3,3],4).tolist(),'error',r['steps'][-1],'actual',np.round(np.array(p['flange_pose_odom'])[:3,3],4).tolist(),'width',p['gripper_width_m'])
    return p

def cloud(cam):
    d=cam['depth']; v,u=np.indices(d.shape);K=cam['K']; T=cam['cam_to_odom']
    q=np.stack([(u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d],-1)
    return q@T[:3,:3].T+T[:3,3]

def get_json(a):
    t=a['text'];return json.loads(t[t.index('{'):t.rindex('}')+1])

def food_geometry(robot,cam,xyz,uv):
    u,v=np.array(uv)*[.640,.480]
    mask=robot.segment(cam['rgb'],points=[[u,v]],point_labels=[1])[0]
    robot.save_image('food_mask',mask.astype(np.uint8)*255)
    er=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    pts=xyz[er&(cam['depth']>0)&(cam['depth']<2.)]
    if len(pts)<15:pts=xyz[mask&(cam['depth']>0)&(cam['depth']<2.)]
    lo,hi=np.percentile(pts,[2,98],axis=0);center=(lo+hi)/2
    if max(hi-lo)>.20:raise ValueError('food mask too large')
    origin=pts.mean(axis=0);q=pts-origin
    fit=np.linalg.lstsq(np.column_stack([2*q,np.ones(len(q))]),np.sum(q*q,axis=1),rcond=None)[0]
    cc=origin+fit[:3];radius=np.sqrt(max(0,fit[3]+np.sum(fit[:3]**2)))
    residual=np.sqrt(np.mean((np.linalg.norm(pts-cc,axis=1)-radius)**2))
    if .008<radius<.08 and residual<.003 and np.linalg.norm(cc-center)<.05:
        center=cc
    # A positive definite ellipsoid handles nonspherical whole produce.
    qn=q/.05;x,y,z=qn.T
    D=np.column_stack([x*x,y*y,z*z,2*x*y,2*x*z,2*y*z,2*x,2*y,2*z])
    coef=np.linalg.lstsq(D,np.ones(len(qn)),rcond=None)[0]
    Q=np.array([[coef[0],coef[3],coef[4]],[coef[3],coef[1],coef[5]],[coef[4],coef[5],coef[2]]])
    vals=np.linalg.eigvalsh(Q)
    if min(vals)>0 and max(vals)<16*min(vals):
        ec=-np.linalg.solve(Q,coef[6:]);radii=.05*np.sqrt((1+ec@Q@ec)/vals);ep=origin+.05*ec
        err=.05*(D@coef-1)/np.maximum(2*np.linalg.norm(qn@Q+coef[6:],axis=1),1e-6)
        if .006<min(radii) and max(radii)<.09 and np.sqrt(np.mean(err**2))<.002 and np.linalg.norm(ep-center)<.035:
            center=ep
    robot.log('food_geometry',{'center':center.tolist(),'bounds':[lo.tolist(),hi.tolist()],'sphere_residual':float(residual)})
    return center,hi


def run(robot,object='food',where='on the counter',tool='tongs'):
    p=robot.proprio(); F=np.eye(4);F[:3,:3]=np.diag([1.,-1.,-1.]);F[:3,3]=[.19,0.,.56]
    move(robot,F,0,65)
    obs=robot.observe(['pole','wrist'])
    for name,c in obs['cameras'].items():robot.save_image('survey_'+name,c['rgb'])
    cam=obs['cameras']['pole']; xyz=cloud(cam)
    models=robot.models()['llm'];robot.log('models',models)
    model=min(models,key=lambda m:models[m]['input_per_mtok']+models[m]['output_per_mtok'])
    dets=robot.detect(cam['rgb'],['tongs'],model='grounding-dino',threshold=.20)
    robot.log('tool_detections',dets)
    ans=robot.llm(model,'Identify the kitchen tongs and the '+object+' '+where+'. Return JSON with coordinates in normalized 0..1000 image coordinates (x horizontal, y vertical): {"hinge":[x,y],"tip1":[x,y],"tip2":[x,y],"food":[x,y]}. Hinge is the joined end of the tongs, tips are centers of the TWO separate food-contact paddles at the opposite end. Food is center of the requested food, not a neighbor. Be precise.',images=[cam['rgb']],max_output_tokens=1800,temperature=0, response_schema={"type":"object","properties":{k:{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2} for k in ["hinge","tip1","tip2","food"]},"required":["hinge","tip1","tip2","food"]})
    robot.log('landmarks',ans);a=get_json(ans)
    if dets:
        options=[]
        for d in dets:
            bb=d['box']
            for swap in [False,True]:
                uv=np.array([a[k][::-1] if swap else a[k] for k in ['hinge','tip1','tip2']])*[.640,.480]
                penalty=float(np.sum(np.maximum(np.array(bb[:2])-uv,0)**2)+np.sum(np.maximum(uv-np.array(bb[2:]),0)**2))
                options.append((penalty+20*(1-d['score']),swap,d))
        _,swap,chosen=min(options,key=lambda z:z[0]);dets=[chosen]
        if swap:
            a={k:v[::-1] for k,v in a.items()};robot.log('corrected transposed landmarks')
    marked=cam['rgb'].copy()
    points={}
    for k,uv in a.items():
        u,v=np.array(uv)*[.640,.480];u=int(np.clip(u,0,639));v=int(np.clip(v,0,479))
        points[k]=xyz[v,u]
        cv2.circle(marked,(u,v),4,(255,0,0),-1);cv2.putText(marked,k,(u+5,v),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,0,0),1)
    robot.save_image('landmarks',marked);robot.log('points', {k:v.tolist() for k,v in points.items()})
    food,food_top=food_geometry(robot,cam,xyz,a['food'])
    # Segment the full tool using its semantic landmarks, then estimate its support.
    uv=np.array([a[k] for k in ['hinge','tip1','tip2']])*[.640,.480]
    box=np.array(dets[0]['box']) if dets else np.r_[uv.min(axis=0)-12,uv.max(axis=0)+12].clip([0,0,0,0],[639,479,639,479])
    mask=robot.segment(cam['rgb'],boxes=[box.tolist()])[0]
    robot.save_image('tool_mask',mask.astype(np.uint8)*255)
    valid=(cam['depth']>0)&(cam['depth']<2.)
    near=cv2.dilate(mask.astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)&~mask&valid
    z=xyz[near,2];hist,edges=np.histogram(z,bins=np.arange(.2,1.,.002))
    floor=float(edges[np.argmax(hist)]+.001)
    pts=xyz[mask&valid];pts=pts[(pts[:,2]>floor+.003)&(pts[:,2]<floor+.09)]
    hinge=points['hinge']; tip=(points['tip1']+points['tip2'])/2
    vals,axes=np.linalg.eigh(np.cov(pts[:,:2].T));axis=axes[:,-1]
    if np.dot(axis,tip[:2]-hinge[:2])<0:axis=-axis
    cross=np.array([-axis[1],axis[0]])
    lo,hi=np.percentile(pts[:,:2]@axis,[1,99])
    grip=np.r_[axis*(lo+.45*(hi-lo))+cross*np.median(pts[:,:2]@cross),floor]
    # Refine tip location from the visible end geometry, preserving model end identity.
    ends=pts[pts[:,:2]@axis>hi-.035]
    if len(ends)>10:
        tip=np.r_[axis*(hi-.014)+cross*np.mean(np.percentile(ends[:,:2]@cross,[2,98])),np.median(ends[:,2])]

    along=(pts[:,:2]-grip[:2])@axis;across=(pts[:,:2]-grip[:2])@cross
    band=pts[np.abs(along)<.015]
    if len(band)<8:return {'ok':False,'reason':'no handle geometry'}
    bounds=np.percentile((band[:,:2]-grip[:2])@cross,[2,98])
    grip[:2]+=cross*bounds.mean()
    width=float(bounds[1]-bounds[0]);grip[2]=floor+max(.010,.5*(np.percentile(band[:,2],90)-floor))
    robot.log('tool_geometry',{'floor':floor,'grip':grip.tolist(),'width':width,'axis':axis.tolist(),'tip':tip.tolist()})
    C=np.r_[cross,0.];Z=np.array([0.,0.,-1.]);D=np.cross(Z,C)
    R=np.column_stack([(C-D)/np.sqrt(2),(C+D)/np.sqrt(2),Z])
    R2=R@np.diag([-1.,-1.,1.]);initial=np.array(obs['proprio']['flange_pose_odom'])[:3,:3]
    if Rotation.from_matrix(R2@initial.T).magnitude()<Rotation.from_matrix(R@initial.T).magnitude():R=R2
    F=np.eye(4);F[:3,:3]=R;F[:3,3]=grip-PADS*Z
    global BASE
    BASE=[.075,float(grip[1]),0.]
    # Raise in front of the edge before moving out over the counter.
    S=F.copy();S[:3,3]=[.22,grip[1],F[2,3]+.04]
    move(robot,S,0,70)
    H=F.copy();H[2,3]+=.04
    move(robot,H,0,70)
    move(robot,F,0,50)
    g=1.0
    move(robot,F,g,25)
    L=F.copy();L[0,3]-=.10;L[2,3]+=.10
    move(robot,L,g,70)
    view=robot.observe(['pole','wrist'])
    for name,c in view['cameras'].items():robot.save_image('lift_'+name,c['rgb'])
    # Rotate the measured tool axis down and forward, away from the forearm.
    axis3=np.r_[axis,0.];cross3=np.r_[cross,0.]
    angle=np.deg2rad(90.)
    down=np.array([np.sin(angle),0.,-np.cos(angle)])
    side=np.array([0.,1.,0.])
    Q=np.column_stack([down,side,np.cross(down,side)])@np.column_stack([axis3,cross3,[0.,0.,1.]]).T
    U=F.copy();U[:3,:3]=Q@F[:3,:3]
    tip_offset=Q@(tip-F[:3,3])
    tip_park=np.array([.40,grip[1],floor+.11])
    U[:3,3]=tip_park-tip_offset
    move(robot,U,1.,90)
    view=robot.observe(['pole','wrist'])
    for name,c in view['cameras'].items():robot.save_image('turned_'+name,c['rgb'])
    gopen=float(np.clip((.083-width+.006)/.08,.05,.95))
    move(robot,U,gopen,25)
    view=robot.observe(['pole','wrist'])
    for name,c in view['cameras'].items():robot.save_image('opened_'+name,c['rgb'])
    # Keep the tool's measured rigid offset while approaching the target.
    BASE=[.075,float(food[1]),0.]
    V=U.copy();V[:3,3]=food-tip_offset+[0.,0.,.065]
    move(robot,V,gopen,100)
    V[:3,3]=food-tip_offset
    move(robot,V,gopen,55)
    for name,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('food_approach_'+name,c['rgb'])
    gclose=min(.98,gopen+.08)
    move(robot,V,gclose,30)
    W=V.copy();W[:3,3]+=[-.035,0.,.095]
    move(robot,W,gclose,70)
    for name,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('food_lift_'+name,c['rgb'])
    return {'ok':False,'reason':'food lift attempted; visual verification not implemented','grip_command':gclose}
