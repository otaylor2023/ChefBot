"""Initial sensor-only tongs geometry and grasp experiment."""
import json
import base64
import zlib
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

def record(robot, name, obs):
    for key, cam in obs['cameras'].items():
        depth=np.nan_to_num(cam['depth'], nan=0., posinf=0., neginf=0.)
        depth=np.rint(np.clip(depth,0.,6.)*1000).astype('<u2')
        robot.save_image(name+'_'+key+'_depth',np.stack([depth%256,depth//256,np.zeros_like(depth)],axis=-1).astype(np.uint8))
        robot.log('SENSOR', json.dumps({'name':name+'_'+key,'K':cam['K'].tolist(),'cam_to_odom':cam['cam_to_odom'].tolist(),'proprio':obs['proprio']}))

def cloud(cam):
    d=cam['depth']; v,u=np.indices(d.shape);K=cam['K']; T=cam['cam_to_odom']
    q=np.stack([(u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d],-1)
    return q@T[:3,:3].T+T[:3,3]

def get_json(a):
    t=a['text'];return json.loads(t[t.index('{'):t.rindex('}')+1])

def food_geometry(robot,cam,xyz,uv):
    u,v=np.array(uv)*[.640,.480]
    dets=robot.detect(cam['rgb'],[FOOD_NAME],model='grounding-dino',threshold=.20)
    robot.log('food_detections',dets)
    if dets:
        def match(d):
            b=np.array(d['box']);point=np.array([u,v])
            return np.sum(np.maximum(b[:2]-point,0)**2)+np.sum(np.maximum(point-b[2:],0)**2)+20*(1-d['score'])
        box=min(dets,key=match)['box']
        mask=robot.segment(cam['rgb'],boxes=[box])[0]
    else:
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


def held_heads(robot,obs,pred,axis):
    axis=np.asarray(axis);side=np.cross([0.,0.,1.],axis)
    groups=[]
    for cam in obs['cameras'].values():
        xyz=cloud(cam).reshape(-1,3);q=xyz-pred
        m=(cam['depth'].ravel()>0)&(np.abs(q@axis)<.07)&(np.abs(q@side)<.10)&(np.abs(q[:,2])<.045)
        groups.append(xyz[m])
    pts=np.concatenate(groups)
    if len(pts)<40:return None
    s=pts@axis;end=np.percentile(s,98);pts=pts[s>end-.030];b=pts@side
    means=np.percentile(b,[20,80])
    for _ in range(10):
        labels=np.argmin(abs(b[:,None]-means),axis=1)
        if min(np.sum(labels==k) for k in range(2))<12:return None
        means=np.array([np.mean(b[labels==k]) for k in range(2)])
    if not .025<means[1]-means[0]<.14:return None
    heads=[];bounds=[]
    for k in range(2):
        q=pts[labels==k]
        lo,hi=np.percentile(np.column_stack([q@axis,q@side,q[:,2]]),[5,95],axis=0)
        if hi[1]-lo[1]>.025:return None
        mid=(lo+hi)/2
        heads.append(mid[0]*axis+mid[1]*side+np.array([0.,0.,mid[2]]));bounds.append((lo,hi))
    center=np.mean(heads,axis=0);gap=bounds[1][0][1]-bounds[0][1][1]
    if np.linalg.norm(center-pred)>.05:return None
    robot.log('held_heads',{'center':center.tolist(),'gap':float(gap),'heads':np.array(heads).tolist(),'correction':(center-pred).tolist()})
    return center,gap

def run(robot,object='food',where='on the counter',tool='tongs'):
    global FOOD_NAME
    FOOD_NAME=object
    p=robot.proprio(); F=np.eye(4);F[:3,:3]=np.diag([1.,-1.,-1.]);F[:3,3]=[.19,0.,.56]
    move(robot,F,0,65)
    obs=robot.observe(['pole','wrist'])
    for name,c in obs['cameras'].items():robot.save_image('survey_'+name,c['rgb'])
    record(robot, 'survey', obs)
    cam=obs['cameras']['pole']; xyz=cloud(cam)
    models=robot.models()['llm'];robot.log('models',models)
    model=next((m for m in models if 'robotics' in m.lower()),min(models,key=lambda m:models[m]['input_per_mtok']+models[m]['output_per_mtok']))
    dets=robot.detect(cam['rgb'],['tongs'],model='grounding-dino',threshold=.20)
    robot.log('tool_detections',dets)
    ans=robot.llm(model,'Identify the kitchen tongs and the '+object+' '+where+'. Return JSON with coordinates in normalized 0..1000 image coordinates (y vertical FIRST, x horizontal SECOND): {"hinge":[y,x],"tip1":[y,x],"tip2":[y,x],"food":[y,x]}. Hinge is the joined end of the tongs, tips are centers of the TWO separate food-contact paddles at the opposite end. Food is center of the requested food, not a neighbor. Be precise.',images=[cam['rgb']],max_output_tokens=1800,temperature=0, response_schema={"type":"object","properties":{k:{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2} for k in ["hinge","tip1","tip2","food"]},"required":["hinge","tip1","tip2","food"]})
    robot.log('landmarks',ans);a={k:v[::-1] for k,v in get_json(ans).items()}
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
    # Keep the hand polarity that exposes distal paddles to the pole view.
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
    record(robot, 'lift', view)
    for name,c in view['cameras'].items():robot.save_image('lift_'+name,c['rgb'])
    # Rotate the measured tool axis down and forward, away from the forearm.
    axis3=np.r_[axis,0.];cross3=np.r_[cross,0.]
    angle=np.deg2rad(90.)
    yaw=np.deg2rad(60.)
    down=np.array([np.sin(angle)*np.cos(yaw),np.sin(angle)*np.sin(yaw),-np.cos(angle)])
    side=np.array([-np.sin(yaw),np.cos(yaw),0.])
    Q=np.column_stack([down,side,np.cross(down,side)])@np.column_stack([axis3,cross3,[0.,0.,1.]]).T
    U=F.copy();U[:3,:3]=Q@F[:3,:3]
    tip_offset=Q@(tip-F[:3,3])
    tip_park=np.array([.40,grip[1],floor+.11])
    U[:3,3]=tip_park-tip_offset
    move(robot,U,1.,90)
    view=robot.observe(['pole','wrist'])
    record(robot, 'turned', view)
    for name,c in view['cameras'].items():robot.save_image('turned_'+name,c['rgb'])
    gopen=float(np.clip((.083-width+.006)/.08,.05,.95))
    pr=move(robot,U,gopen,25)
    empty_curve=[[gopen,pr['gripper_width_m']]]
    for gg in np.r_[np.arange(gopen+.07,1.,.07),1.]:
        pr=robot.act([row(U,float(gg))+BASE]*7)['proprio']
        empty_curve.append([float(gg),pr['gripper_width_m']])
    robot.log('empty_curve',empty_curve)
    move(robot,U,gopen,25)
    view=robot.observe(['pole','wrist'])
    record(robot, 'opened', view)
    for name,c in view['cameras'].items():robot.save_image('opened_'+name,c['rgb'])
    cal=held_heads(robot,view,U[:3,3]+tip_offset,down)
    if cal is not None:
        actual=np.array(view['proprio']['flange_pose_odom'])
        tip_offset=U[:3,:3]@actual[:3,:3].T@(cal[0]-actual[:3,3])
    # Keep the measured open-head offset while approaching the target.
    BASE=[.075,float(food[1]),0.]
    V=U.copy();V[:3,3]=food-tip_offset+[0.,0.,.065]
    move(robot,V,gopen,100)
    V[:3,3]=food-tip_offset+[0.,0.,.008]
    start=np.array(robot.proprio()['flange_pose_odom'])[:3,3]
    n=int(np.ceil(np.linalg.norm(V[:3,3]-start)/.0012))
    path=[]
    for t in np.linspace(0.,1.,n+1)[1:]:
        T=V.copy();T[:3,3]=start+t*(V[:3,3]-start);path.append(row(T,gopen)+BASE)
    robot.act(path)
    move(robot,V,gopen,15)
    for name,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('food_approach_'+name,c['rgb'])
    gclose=gopen
    for gclose in np.arange(gopen+.035,1.001,.035):
        gclose=float(min(1.,gclose))
        pr=robot.act([row(V,gclose)+BASE]*6)['proprio']
        residual=pr['gripper_width_m']-(.083-.080*gclose)
        robot.log('squeeze',gclose,'width',pr['gripper_width_m'],'residual',residual)
        empty_width=float(np.interp(gclose,np.array(empty_curve)[:,0],np.array(empty_curve)[:,1]))
        extra=pr['gripper_width_m']-empty_width
        robot.log('extra_resistance',extra)
        if extra>.002 and residual>.0005:break
    W=V.copy();W[:3,3]+=[-.035,0.,.095]
    move(robot,W,gclose,70)
    for name,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('food_lift_'+name,c['rgb'])
    return {'ok':False,'reason':'food lift attempted; visual verification not implemented','grip_command':gclose}
