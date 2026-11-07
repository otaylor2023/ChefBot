"""v59: prior perception and paths with explicit Franka force calibration."""
import json
import base64
import zlib
import numpy as np
import cv2
import kinematics
from scipy.spatial.transform import Rotation

PADS = 0.118  # PORT: the real pads' middle out from the flange, the hand on its 27.25 mm holder (sim 0.091)
MOVE_CAP = 300  # PORT: control steps (20 s) a move may take while still progressing (the lab's: n+50)
BASE = [0.,0.,0.]

def row(F,g):
    q=Rotation.from_matrix(F[:3,:3]).as_quat()
    return [*F[:3,3],q[3],*q[:3],g]

def planar(x, y, yaw):
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [x, y]
    return T

def move(robot, F, g, n=30, settle=True):
    first = min(n,15) if settle else n
    r = robot.act([row(F,g)+BASE]*first)
    remaining = max(n+50, MOVE_CAP)-first  # PORT: was n+50-first
    previous = None
    stalled = 0
    while settle and remaining>0:
        e = r['steps'][-1]
        if e['pos_error_m']<.003 and e['rot_error_rad']<.06:
            break
        score = e['pos_error_m']+.1*e['rot_error_rad']
        stalled = stalled+1 if previous is not None and previous-score<.0007 else 0
        if stalled>=3:
            break
        previous = score
        count = min(5,remaining)
        r = robot.act([row(F,g)+BASE]*count)
        remaining -= count
    if settle:
        p=r['proprio'];actual=np.array(p['flange_pose_odom'])
        err=np.linalg.norm(actual[:3,3]-F[:3,3])
        predicted=kinematics.fk(p['arm_q']);measured=np.array(p['flange_pose_root'])
        if .003<err<.04 and np.linalg.norm(predicted[:3,3]-measured[:3,3])<.002:
            root=planar(*p['base_pose_odom'])
            q=kinematics.solve(np.linalg.inv(root)@F,p['arm_q'])
            if q is not None:
                r=robot.act_joints([q.tolist()+[g]]*20)
                actual=np.array(r['proprio']['flange_pose_odom'])
                r['steps'][-1]['pos_error_m']=float(np.linalg.norm(actual[:3,3]-F[:3,3]))
                r['steps'][-1]['rot_error_rad']=float(Rotation.from_matrix(actual[:3,:3]@F[:3,:3].T).magnitude())
                robot.log('kinematic refinement',err,'to',r['steps'][-1]['pos_error_m'])
    robot.log('move', np.round(F[:3, 3], 4).tolist(), 'err', r['steps'][-1],
              'actual', np.round(np.array(r['proprio']['flange_pose_odom'])[:3, 3], 4).tolist(),
              'width', r['proprio']['gripper_width_m'])
    return r['proprio']


def safe_grasp(robot,force,speed=.05):
    robot.act_base([BASE])
    return robot.gripper_grasp(force,speed)


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

def food_geometry(robot,cam,xyz,uv,dets=None):
    u,v=np.array(uv)*[.640,.480]
    if dets is None:dets=robot.detect(cam['rgb'],[FOOD_NAME],model='grounding-dino',threshold=.20)
    robot.log('food_detections',dets)
    if dets:
        def match(d):
            b=np.array(d['box']);point=np.array([u,v])
            return np.sum(np.maximum(b[:2]-point,0)**2)+np.sum(np.maximum(point-b[2:],0)**2)+20*(1-d['score'])
        box=min(dets,key=match)['box']
        mask=robot.segment(cam['rgb'],boxes=[box])[0]
    else:
        mask=robot.segment(cam['rgb'],points=[[u,v]],point_labels=[1])[0]
    global LAST_FOOD_RGB
    LAST_FOOD_RGB=np.median(cam['rgb'][mask],axis=0)
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

def semantic_heads(robot,obs,pred,axis,spread):
    cam=obs['cameras']['pole'];xyz=cloud(cam)
    models=robot.models()['llm'];model=next((n for n in models if 'robotics' in n.lower()),min(models,key=lambda n:models[n]['input_per_mtok']+models[n]['output_per_mtok']))
    a=robot.llm(model,'Locate the centers of the TWO food-contact paddles at the free ends of the kitchen tongs held by the robot. One or both may be occluded. Return p1_x,p1_y,p2_x,p2_y normalized0..1000 horizontal then vertical, and p1_visible,p2_visible. Mark visible only if the actual paddle is seen, not a robot part. Do not point at the joined hinge.',images=[cam['rgb']],max_output_tokens=2048,temperature=0,response_schema={'type':'object','properties':{**{k:{'type':'number'} for k in ['p1_x','p1_y','p2_x','p2_y']},**{k:{'type':'boolean'} for k in ['p1_visible','p2_visible']}},'required':['p1_x','p1_y','p2_x','p2_y','p1_visible','p2_visible']})
    robot.log('held paddle landmarks',a)
    try:
        a=get_json(a);heads=[];marked=cam['rgb'].copy()
        for name in ['p1','p2']:
            if not a[name+'_visible']:continue
            u,v=np.rint([a[name+'_x']*.640,a[name+'_y']*.480]).astype(int);u=int(np.clip(u,2,637));v=int(np.clip(v,2,477))
            point=np.median(xyz[v-1:v+2,u-1:u+2].reshape(-1,3),axis=0)
            if np.linalg.norm(point-pred)<.15:heads.append(point)
            cv2.circle(marked,(u,v),5,(255,0,0),-1)
        robot.save_image('held_paddles',marked)
        if len(heads)==2:center=np.mean(heads,axis=0)
        elif len(heads)==1:
            side=np.cross([0.,0.,1.],axis);side/=np.linalg.norm(side)
            sign=1 if np.dot(heads[0]-pred,side)>0 else -1
            center=heads[0]-sign*.5*spread*side
        else:return None
        center[2]=pred[2]
        if np.linalg.norm(center-pred)>.07:return None
        robot.log('semantic paddle correction',(center-pred).tolist(),'spread',spread)
        return center,spread
    except (ValueError,KeyError,IndexError):return None

def geometric_landmarks(robot,cam,xyz,dets):
    """Use joined-versus-separated end geometry only when unambiguous."""
    if not dets:return None
    chosen=max(dets,key=lambda d:d['score'])
    mask=robot.segment(cam['rgb'],boxes=[chosen['box']])[0]
    valid=(cam['depth']>0)&(cam['depth']<2.)
    near=cv2.dilate(mask.astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)&~mask&valid
    z=xyz[near,2]
    if len(z)<30:return None
    hist,edges=np.histogram(z,bins=np.arange(.2,1.,.002))
    floor=float(edges[np.argmax(hist)]+.001)
    pts=xyz[mask&valid];pts=pts[(pts[:,2]>floor+.003)&(pts[:,2]<floor+.09)]
    if len(pts)<40:return None
    _,axes=np.linalg.eigh(np.cov(pts[:,:2].T));axis=axes[:,-1]
    side=np.array([-axis[1],axis[0]])
    along=pts[:,:2]@axis;across=pts[:,:2]@side
    lo,hi=np.percentile(along,[1,99]);span=hi-lo
    if not .10<span<.50:return None
    ends=[along<lo+.15*span,along>hi-.15*span]
    if min(np.sum(m) for m in ends)<10:return None
    widths=[float(np.diff(np.percentile(across[m],[5,95]))[0]) for m in ends]
    certain=max(widths)>2*min(widths) and max(widths)-min(widths)>.02
    robot.log('end_geometry',{'widths':widths,'confident':certain})
    if not certain:return None
    food_dets=robot.detect(cam['rgb'],[FOOD_NAME],model='grounding-dino',threshold=.20)
    if len(food_dets)!=1:return None
    if widths[0]>widths[1]:axis=-axis
    side=np.array([-axis[1],axis[0]])
    along=pts[:,:2]@axis;across=pts[:,:2]@side;lo,hi=np.percentile(along,[1,99])
    far=along>hi-.15*span;near_end=along<lo+.15*span
    hinge=np.r_[axis*lo+side*np.median(across[near_end]),np.median(pts[near_end,2])]
    tips=[np.r_[axis*hi+side*b,np.median(pts[far,2])] for b in np.percentile(across[far],[20,80])]
    inv=np.linalg.inv(cam['cam_to_odom']);K=cam['K']
    def project(p):
        q=inv[:3,:3]@p+inv[:3,3];uv=(K@q)[:2]/q[2]
        return (uv/[.640,.480]).tolist()
    a={k:project(p) for k,p in zip(['hinge','tip1','tip2'],[hinge,*tips])}
    box=np.array(food_dets[0]['box']);a['food']=((box[:2]+box[2:])/2/[.640,.480]).tolist()
    robot.log('geometric_landmarks',a)
    return a,chosen,mask,food_dets

def visual_retry(robot,view,food,food_rgb,tip_offset,R,down,empty_curve,open_force):
    """Retry once only after RGB-D identifies the food still near its original support."""
    cam=view['cameras']['pole'];inv=np.linalg.inv(cam['cam_to_odom'])
    q=inv[:3,:3]@food+inv[:3,3]
    if q[2]<=0:return 'verification outside camera'
    uv=(cam['K']@q)[:2]/q[2]
    if not (0<=uv[0]<640 and 0<=uv[1]<480):return 'verification outside image'
    try:
        now,top=food_geometry(robot,cam,cloud(cam),uv/[.640,.480])
    except (ValueError,IndexError,RuntimeError,np.linalg.LinAlgError) as exc:
        robot.log('retry_verification_failed',str(exc));return 'verification unavailable'
    rise=float(now[2]-food[2]);xy=float(np.linalg.norm(now[:2]-food[:2]))
    color=float(np.linalg.norm(LAST_FOOD_RGB-food_rgb)/np.sqrt(3.))
    robot.log('visual_retry_gate',{'food':now.tolist(),'rise':rise,'xy_shift':xy,'color_rms':color})
    if xy>.04 or color>45:return 'uncertain food identity; no retry'
    if not -.02<rise<.03:return 'food not confirmed on original support; no retry'
    pr=robot.proprio()
    if pr['gripper_width_m']<.006:return 'tool not retained; no retry'
    if robot.budget().get('control_steps',0)<180:return 'insufficient time for retry'
    O=np.array(pr['flange_pose_odom'])
    predicted=O[:3,3]+O[:3,:3]@R.T@tip_offset
    safe_grasp(robot,open_force,.05)
    pr=robot.act([row(O,1.)+BASE]*12)['proprio']
    if pr['gripper_width_m']<.006:return 'tool lost while reopening'
    obs=robot.observe(['pole','wrist'])
    cal=held_heads(robot,obs,predicted,down)
    if cal is None:return 'open tips unverified; no retry'
    actual=np.array(obs['proprio']['flange_pose_odom'])
    offset=R@actual[:3,:3].T@(cal[0]-actual[:3,3])
    V=np.eye(4);V[:3,:3]=R;V[:3,3]=now-offset
    pads=V[:3,3]+PADS*R[:,2]
    if np.linalg.norm(pads-now)<max(.10,np.linalg.norm(top-now)+.04):
        return 'insufficient hand-food clearance for retry'
    robot.log('visual_retry_begin',{'target':now.tolist(),'tip_offset':offset.tolist()})
    start=actual[:3,3];n=max(1,int(np.ceil(np.linalg.norm(V[:3,3]-start)/.003)))
    path=[]
    for t in np.linspace(0.,1.,n+1)[1:]:
        T=V.copy();T[:3,3]=start+t*(V[:3,3]-start);path.append(row(T,1.)+BASE)
    robot.act(path);move(robot,V,1.,15)
    for force,empty_width in empty_curve:
        safe_grasp(robot,force,.05)
        pr=robot.act([row(V,1.)+BASE]*9)['proprio']
        extra=pr['gripper_width_m']-empty_width
        robot.log('retry_force',force,'width',pr['gripper_width_m'],'extra',extra)
        if extra>.002:
            safe_grasp(robot,min(20.,force+4.),.05)
            robot.act([row(V,1.)+BASE]*8);break
    W=np.array(robot.proprio()['flange_pose_odom']);W[:3,3]+=[-.035,0.,.095]
    move(robot,W,1.,70)
    for name,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('retry_lift_'+name,c['rgb'])
    return 'one visually triggered retry attempted; possession unverified'

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
    auto=geometric_landmarks(robot,cam,xyz,dets)
    if auto is None:
        ans=robot.llm(model,'Identify the kitchen tongs and the '+object+' '+where+'. Return JSON with coordinates in normalized 0..1000 image coordinates (y vertical FIRST, x horizontal SECOND): {"hinge":[y,x],"tip1":[y,x],"tip2":[y,x],"food":[y,x]}. Hinge is the joined end of the tongs, tips are centers of the TWO separate food-contact paddles at the opposite end. Food is center of the requested food, not a neighbor. Be precise.',images=[cam['rgb']],max_output_tokens=1800,temperature=0, response_schema={"type":"object","properties":{k:{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2} for k in ["hinge","tip1","tip2","food"]},"required":["hinge","tip1","tip2","food"]})
        robot.log('landmarks',ans);a={k:v[::-1] for k,v in get_json(ans).items()}
    else:
        a,chosen,auto_mask,food_dets=auto;dets=[chosen]
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
    food,food_top=food_geometry(robot,cam,xyz,a['food'],None if auto is None else food_dets)
    food_rgb=LAST_FOOD_RGB.copy()
    # Segment the full tool using its semantic landmarks, then estimate its support.
    uv=np.array([a[k] for k in ['hinge','tip1','tip2']])*[.640,.480]
    box=np.array(dets[0]['box']) if dets else np.r_[uv.min(axis=0)-12,uv.max(axis=0)+12].clip([0,0,0,0],[639,479,639,479])
    mask=robot.segment(cam['rgb'],boxes=[box.tolist()])[0] if auto is None else auto_mask
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
    # Use the equivalent finger polarity requiring the smaller rotation.
    if Rotation.from_matrix(R2@initial.T).magnitude()<Rotation.from_matrix(R@initial.T).magnitude():R=R2
    F=np.eye(4);F[:3,:3]=R;F[:3,3]=grip-PADS*Z
    global BASE
    BASE=[.075,float(grip[1]),0.]
    # Raise in front of the edge before moving out over the counter.
    S=np.array(obs['proprio']['flange_pose_odom']);S[2,3]=F[2,3]+.07
    move(robot,S,0,70)
    H=F.copy();H[2,3]+=.04
    move(robot,H,0,70)
    move(robot,F,0,50)
    g=1.0
    robot.act_base([BASE]);robot.gripper_grasp(20.,.05);robot.act_base([BASE]*35)
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
    # Calibrate this tool's force-width relation using only measured hand width.
    # Descend from the secure pickup force until the handles reopen enough.
    # Geometry sets the aperture objective; the measured spring curve sets force.
    pr=robot.proprio()
    empty_curve=[[20.,pr['gripper_width_m']]]
    open_force=20.
    desired_width=.80*width
    for force in [12.,8.,6.,4.,3.,2.,1.5,1.,.75,.5]:
        safe_grasp(robot,force,.05)
        pr=robot.act([row(U,1.)+BASE]*10)['proprio']
        empty_curve.append([force,pr['gripper_width_m']])
        open_force=force
        if pr['gripper_width_m']<.003:
            return {'ok':False,'reason':'tool lost during calibration'}
        if pr['gripper_width_m']>=desired_width:break
    empty_curve=sorted(empty_curve)
    robot.log('empty_curve',empty_curve,'opening_force',open_force,'desired_width',desired_width)
    gopen=1.
    view=robot.observe(['pole','wrist'])
    record(robot, 'opened', view)
    for name,c in view['cameras'].items():robot.save_image('opened_'+name,c['rgb'])
    cal=held_heads(robot,view,U[:3,3]+tip_offset,down)
    if cal is None:
        spread=float(np.linalg.norm(points['tip1'][:2]-points['tip2'][:2]))*pr['gripper_width_m']/max(width,.01)
        cal=semantic_heads(robot,view,U[:3,3]+tip_offset,down,spread)
    if cal is not None:
        actual=np.array(view['proprio']['flange_pose_odom'])
        tip_offset=U[:3,:3]@actual[:3,:3].T@(cal[0]-actual[:3,3])
    # Keep the measured open-head offset while approaching the target.
    BASE=[.075,float(food[1]),0.]
    V=U.copy();V[:3,3]=food-tip_offset+[0.,0.,.065]
    move(robot,V,gopen,100)
    V[:3,3]=food-tip_offset+[0.,0.,.008]
    start=np.array(robot.proprio()['flange_pose_odom'])[:3,3]
    n=int(np.ceil(np.linalg.norm(V[:3,3]-start)/.003))
    path=[]
    for t in np.linspace(0.,1.,n+1)[1:]:
        T=V.copy();T[:3,3]=start+t*(V[:3,3]-start);path.append(row(T,gopen)+BASE)
    robot.act(path)
    move(robot,V,gopen,15)
    for name,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('food_approach_'+name,c['rgb'])
    force=open_force
    for force,empty_width in empty_curve:
        safe_grasp(robot,force,.05)
        pr=robot.act([row(V,1.)+BASE]*9)['proprio']
        extra=pr['gripper_width_m']-empty_width
        robot.log('squeeze_force',force,'width',pr['gripper_width_m'],'extra_resistance',extra)
        if extra>.002:
            force=min(20.,force+2.)
            safe_grasp(robot,force,.05)
            robot.act([row(V,1.)+BASE]*8)
            break
    gclose=1.
    W=V.copy();W[:3,3]+=[-.035,0.,.095]
    move(robot,W,gclose,70)
    final_view=robot.observe(['pole','wrist'])
    for name,c in final_view['cameras'].items():robot.save_image('food_lift_'+name,c['rgb'])
    reason=visual_retry(robot,final_view,food,food_rgb,tip_offset,U[:3,:3],down,empty_curve,open_force)
    return {'ok':False,'reason':reason,'grip_command':gclose}
