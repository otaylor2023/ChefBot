"""Named-object pick from RGB-D using only the chefbot-v1 robot API."""
import json
import numpy as np
import cv2
import kinematics
import close_view
from scipy.spatial.transform import Rotation

PADS = 0.118  # PORT: the real pads' middle out from the flange, the hand on its 27.25 mm holder (sim 0.091)
FINGER_BAND = (.075 + .027, .115 + .027)  # PORT: where the open fingers are, out from the flange (sim .075-.115)
MOVE_CAP = 300  # PORT: control steps (20 s) a move may take while still progressing (the lab's: n+50)
BASE = [0.,0.,0.]
LOCAL_GRASP = False
GRASP_FORCE = 2.
CLOSING_ANGLE = np.pi/4  # Documented fallback; estimate from the visible open fingers.


def planar(x, y, yaw):
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [x, y]
    return T


def cloud(cam, proprio):
    d = cam['depth']
    v, u = np.indices(d.shape)
    K = cam['K']
    q = np.stack([(u-K[0, 2])*d/K[0, 0], (v-K[1, 2])*d/K[1, 1], d], -1)
    T = cam['cam_to_odom']
    return q @ T[:3, :3].T + T[:3, 3]


def calibrate_fingers(robot, xyz, cam, p):
    global CLOSING_ANGLE
    F = np.array(p['flange_pose_odom'])
    local = (xyz-F[:3,3]) @ F[:3,:3]
    radius = np.linalg.norm(local[:,:,:2], axis=-1)
    region = (cam['depth']>0) & (local[:,:,2]>FINGER_BAND[0]) & (local[:,:,2]<FINGER_BAND[1])
    region &= (radius>.025) & (radius<.075)
    pts = local[region]
    if len(pts)>40 and p['gripper_width_m']>.06:
        values, axes = np.linalg.eigh(np.cov(pts[:,:2].T))
        axis = axes[:,-1]
        if axis[0]<0: axis=-axis
        spread = np.sqrt(values[-1])
        if values[-1]>8*max(values[0],1e-8) and .025<spread<.065:
            CLOSING_ANGLE = float(np.arctan2(axis[1],axis[0]))
            robot.log('sensor-calibrated closing angle degrees',float(np.degrees(CLOSING_ANGLE)))
    return CLOSING_ANGLE


def row(F, g):
    q = Rotation.from_matrix(F[:3, :3]).as_quat()
    return [*F[:3, 3], q[3], *q[:3], g]


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


def orientation(closing, initial, approach=None):
    C = np.r_[closing, 0.]
    Z = np.array([0., 0., -1.]) if approach is None else approach
    D = np.cross(Z, C)
    R = np.column_stack([C*np.cos(CLOSING_ANGLE)-D*np.sin(CLOSING_ANGLE), C*np.sin(CLOSING_ANGLE)+D*np.cos(CLOSING_ANGLE), Z])
    R2 = R @ np.diag([-1., -1., 1.])
    if Rotation.from_matrix(R2 @ initial.T).magnitude() < Rotation.from_matrix(R @ initial.T).magnitude():
        R = R2
    return R


def perceive(robot, object, use="hold", then="", target=""):
    global LOCAL_GRASP, GRASP_FORCE, BASE
    GRASP_FORCE = 2.
    LOCAL_GRASP = False
    obs = robot.observe(['wrist','pole'])
    cam, p = obs['cameras']['wrist'], obs['proprio']
    xyz = cloud(cam, p)
    # Printed hand closing direction is documented at +45 degrees.
    robot.save_image('initial', cam['rgb'])
    dm=np.rint(np.clip(np.nan_to_num(cam['depth']),0,6)*1000).astype(np.uint16)
    robot.save_image('depth',np.stack([dm%256,dm//256,np.zeros_like(dm)],axis=-1).astype(np.uint8))
    robot.log('SENSOR',json.dumps({'K':cam['K'].tolist(),'cam_to_odom':cam['cam_to_odom'].tolist()}))
    ds = robot.detect(cam['rgb'], [object.replace('_', ' ')], model='grounding-dino', threshold=0.2)
    ds.sort(key=lambda d: d['score'], reverse=True)
    robot.log('detections', ds)
    # Always verify identity, including confident detector mistakes.
    models = robot.models()['llm']
    model = next((n for n in models if 'robotics' in n.lower()), min(models, key=lambda n: models[n]['input_per_mtok']+models[n]['output_per_mtok']))
    action = ''
    if use == 'squeeze':action = 'The intended use is squeezing this sprung mechanism twice: grasp ACROSS BOTH movable handles near their free ends so the jaws can actuate it, not the hinge. Choose local=true. '
    elif use == 'press':action = 'The intended use is pressing the tool head into a bowl: grasp its sliding handle, leaving its rod and working head free. Choose local=true. '
    elif use == 'tongs':action = 'The intended use is gripping food using the tool: grasp across its two movable handles, leaving its food-contact jaws free. Choose local=true. '
    ans = robot.llm(model, action+'Identify the '
+object.replace('_', ' ')+
        ' on the kitchen counter and choose a stable parallel-jaw grasp. The hand opens to 80 mm. '
        'Two images show the same scene: image 0 is the wrist, image 1 is the pole camera. Compare both to identify the requested object, which may be absent from one image. Return view=0 or 1 for the view where the requested object and its grasp region are most clearly visible. Return x,y for object identity; gx,gy for the CENTER of a suitable grasp region in that selected image. '
        'Use local=false when grasping the whole small body. If the body is too wide, local=true and '
        'choose a narrow load-bearing handle, protrusion, or accessible rim, with free space on both sides. '
        'For a long thin object choose its shaft rather than a wide end. Choose a visible surface point. '
        'Coordinates normalized 0..1000, x horizontal then y vertical. '
        'Also estimate the minimum practical grip force in Newtons needed to hold its weight and torque with rubber pads, accounting for visible material fragility. Prefer gentle force on delicate material. The hand supports 0.1 to 70 N. Return JSON with x,y,gx,gy,local and force_n.',
        images=[cam['rgb'],obs['cameras']['pole']['rgb']],max_output_tokens=2048,temperature=0,
        response_schema={'type':'object','properties':{**{k:{'type':'number'} for k in ['x','y','gx','gy','force_n']},
            'local':{'type':'boolean'},'view':{'type':'integer'}},'required':['x','y','gx','gy','local','force_n','view']})
    robot.log('identity', ans)
    semantic_point = None
    grasp_point = None
    local_grasp = False
    try:
        t = ans['text']; fragment=t[t.index('{'):]
        # Recover only a missing final brace; incomplete fields still fail JSON parsing.
        if '}' not in fragment:fragment+='}'
        a=json.loads(fragment[:fragment.rindex('}')+1])
        if int(a.get('view',0))==1:
            cam=obs['cameras']['pole'];xyz=cloud(cam,p)
            ds=robot.detect(cam['rgb'],[object.replace('_',' ')],model='grounding-dino',threshold=.2)
            ds.sort(key=lambda d:d['score'],reverse=True)
            robot.save_image('selected_pole',cam['rgb'])
            dm=np.rint(np.clip(np.nan_to_num(cam['depth']),0,6)*1000).astype(np.uint16)
            robot.save_image('selected_depth',np.stack([dm%256,dm//256,np.zeros_like(dm)],axis=-1).astype(np.uint8))
            robot.log('SENSOR_SELECTED',json.dumps({'K':cam['K'].tolist(),'cam_to_odom':cam['cam_to_odom'].tolist()}))
        u, v = float(a['x'])*cam['rgb'].shape[1]/1000, float(a['y'])*cam['rgb'].shape[0]/1000
        if int(a.get('view',0))==1:
            # Center a close wrist view on the selected target before measuring a grasp.
            uu=int(np.clip(round(u),0,639));vv=int(np.clip(round(v),0,479))
            target3=xyz[vv,uu]
            original=np.array(p['flange_pose_odom'])
            wrist=obs['cameras']['wrist'];extrinsic=np.linalg.inv(original)@wrist['cam_to_odom']
            camera_goal=wrist['cam_to_odom'].copy()
            camera_goal[:3,3]=target3-.30*camera_goal[:3,2]
            view_goal=camera_goal@np.linalg.inv(extrinsic)
            BASE=[.07,float(target3[1]),0.]
            state=move(robot,view_goal,0,45)
            robot.log('active wrist view target',target3.tolist())
            if np.linalg.norm(np.array(state['flange_pose_odom'])[:3,3]-view_goal[:3,3])>.04:
                return None
            geo=close_view.perceive(robot,object,use,then,target)
            LOCAL_GRASP=close_view.LOCAL_GRASP;GRASP_FORCE=close_view.GRASP_FORCE
            return geo
        semantic_point = [u,v]
        grasp_point = [float(a['gx'])*.640,float(a['gy'])*.480]
        local_grasp = bool(a['local'])
        GRASP_FORCE = float(np.clip(a.get('force_n',2.),.5,50.))
        candidates = [d for d in ds if d['box'][0] <= u <= d['box'][2] and d['box'][1] <= v <= d['box'][3]]
        if candidates:
            best_score = max(d['score'] for d in candidates)
            candidates = [d for d in candidates if d['score']>=.75*best_score]
        candidates.sort(key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]), reverse=True)
        if candidates:
            mask = robot.segment(cam['rgb'], boxes=[candidates[0]['box']])[0]
        else:
            mask = robot.segment(cam['rgb'], points=[[u, v]], point_labels=[1])[0]
    except (ValueError, KeyError, IndexError) as e:
        robot.log('identity response unusable; stop',str(e))
        return None
    valid = (cam['depth']>0) & (cam['depth']<2.5)
    def plausible(candidate):
        q = xyz[candidate & valid]
        if len(q)<20: return False
        _, ax = np.linalg.eigh(np.cov(q[:,:2].T))
        widths = np.diff(np.percentile(q[:,:2]@ax,[2,98],axis=0),axis=0)[0]
        return max(widths)<.65
    if not plausible(mask):
        robot.log('mask geometry invalid; retry semantic point')
        if semantic_point is None: return None
        mask = robot.segment(cam['rgb'],points=[semantic_point],point_labels=[1])[0]
        if not plausible(mask): return None
    robot.save_image('mask', mask.astype(np.uint8)*255)
    er = cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
    pts = xyz[er & valid] if (er & valid).sum() > 30 else xyz[mask & valid]
    if len(pts) < 20:
        return None
    lo, hi = np.percentile(pts, [2,98], axis=0)
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((51,51),np.uint8)).astype(bool) & ~mask & valid
    near = xyz[ring]
    near = near[(near[:,2] < hi[2]-.012) & (near[:,2] > lo[2]-.20)]
    floor = lo[2]
    if len(near) > 30:
        bins = np.arange(lo[2]-.20, hi[2], .003)
        hist, edges = np.histogram(near[:,2], bins)
        floor = edges[np.argmax(hist)] + .0015
    raw_pts=pts.copy()
    pts = pts[pts[:,2] > floor+.005]
    if len(pts) < 20:
        return None
    cylinder_center=None
    if hi[2]-floor>.05:
        rng=np.random.default_rng(0)
        sample=pts[rng.choice(len(pts),min(len(pts),1500),replace=False)]
        best=None
        for _ in range(300):
            three=sample[rng.choice(len(sample),3,replace=False),:2]
            A=2*(three[1:]-three[0]);b=np.sum(three[1:]**2,axis=1)-np.sum(three[0]**2)
            if abs(np.linalg.det(A))<1e-5:continue
            cc=np.linalg.solve(A,b);radius=np.linalg.norm(three[0]-cc)
            if not .018<radius<.038:continue
            inliers=abs(np.linalg.norm(sample[:,:2]-cc,axis=1)-radius)<.002
            if inliers.sum()<.4*len(sample):continue
            zspan=np.ptp(np.percentile(sample[inliers,2],[5,95]))
            if zspan<max(.035,.7*(hi[2]-floor)):continue
            if best is None or inliers.sum()>best[0]:best=(inliers.sum(),cc,radius)
        if best is not None:
            _,cc,radius=best
            body=pts[abs(np.linalg.norm(pts[:,:2]-cc,axis=1)-radius)<.004]
            origin=body[:,:2].mean(axis=0);q=body[:,:2]-origin
            fit=np.linalg.lstsq(np.column_stack([2*q,np.ones(len(q))]),np.sum(q*q,axis=1),rcond=None)[0]
            cylinder_center=origin+fit[:2]
            pts=body;local_grasp=False
            robot.log('observed vertical cylinder',cylinder_center.tolist(),float(radius),'inlier fraction',float(best[0]/len(sample)))
    if grasp_point is not None and local_grasp:
        LOCAL_GRASP = True
        uv=np.column_stack(np.where(mask&valid))[:,::-1]
        pd=np.sum((uv-np.array(grasp_point))**2,axis=1)
        nearby=uv[pd<10**2]
        if len(nearby)>4:
            heights=xyz[nearby[:,1],nearby[:,0],2]
            nearby=nearby[heights>=np.percentile(heights,85)]
            u,v=nearby[np.argmin(np.sum((nearby-np.array(grasp_point))**2,axis=1))]
        else:u,v=uv[np.argmin(pd)]
        gp=xyz[v,u]
        if gp[2]<floor+.025:
            candidates=pts[np.linalg.norm(pts[:,:2]-gp[:2],axis=1)<.06]
            if len(candidates)>15:
                peak=np.percentile(candidates[:,2],98)
                raised=candidates[candidates[:,2]>peak-.012]
                if peak-gp[2]>.025 and len(raised)>=8:
                    gp=np.median(raised,axis=0)
                    robot.log('raised local feature',gp.tolist())
        local=raw_pts[(np.linalg.norm(raw_pts[:,:2]-gp[:2],axis=1)<.035)&(raw_pts[:,2]>gp[2]-.018)&(raw_pts[:,2]>floor-.002)]
        if len(local)<8:return None
        vals,axes=np.linalg.eigh(np.cov(local[:,:2].T))
        bounds=np.percentile(local[:,:2]@axes,[2,98],axis=0)
        center=np.r_[axes@bounds.mean(axis=0),max(floor+.014,np.percentile(local[:,2],85)-.008)]
        top=float(np.percentile(local[:,2],98))
        closing=axes[:,0]
        radial=center[:2]/np.linalg.norm(center[:2]);rc=np.array([-radial[1],radial[0]])
        span=float(np.diff(np.percentile(local[:,:2]@rc,[2,98]))[0])
        robot.log('local grasp',{'point':gp.tolist(),'center':center.tolist(),'top':top,'floor':float(floor),'spans':(bounds[1]-bounds[0]).tolist(),'closing':closing.tolist()})
        marked=cam['rgb'].copy();cv2.circle(marked,(int(u),int(v)),5,(255,0,0),-1);robot.save_image('grasp_point',marked)
        return center,top,closing,p,floor,bounds[1]-bounds[0],span
    xy = pts[:,:2]
    vals, axes = np.linalg.eigh(np.cov(xy.T))
    projected = xy @ axes
    bounds = np.percentile(projected, [2,98], axis=0)
    center = np.r_[axes @ bounds.mean(axis=0), 0.]
    top = np.percentile(pts[:,2],98)
    height = top-floor
    spans = bounds[1]-bounds[0]
    if .6*max(spans)<height<1.6*max(spans) and max(spans)<1.65*min(spans):
        origin = pts.mean(axis=0)
        q = pts-origin
        fit = np.linalg.lstsq(np.column_stack([2*q,np.ones(len(q))]), np.sum(q*q,axis=1),rcond=None)[0]
        cc = origin+fit[:3]
        radius = np.sqrt(max(0.,fit[3]+np.sum(fit[:3]**2)))
        residual = np.sqrt(np.mean((np.linalg.norm(pts-cc,axis=1)-radius)**2))
        accepted = (.012<radius<.060 and residual<.003 and np.linalg.norm(cc[:2]-center[:2])<.020
                    and floor+.2*height<cc[2]<floor+.8*height)
        robot.log('sphere fit',{'center':cc.tolist(),'radius':float(radius),'residual':float(residual),'accepted':bool(accepted)})
        if accepted: center[:2]=cc[:2]
        else:
            qn = q/.05
            x,y,z = qn.T
            D = np.column_stack([x*x,y*y,z*z,2*x*y,2*x*z,2*y*z,2*x,2*y,2*z])
            coef = np.linalg.lstsq(D,np.ones(len(qn)),rcond=None)[0]
            Q = np.array([[coef[0],coef[3],coef[4]],[coef[3],coef[1],coef[5]],[coef[4],coef[5],coef[2]]])
            values = np.linalg.eigvalsh(Q)
            if min(values)>0 and max(values)<25*min(values):
                ec = -np.linalg.solve(Q,coef[6:9])
                er = .05*np.sqrt((1+ec@Q@ec)/values)
                ep = origin+.05*ec
                error = .05*(D@coef-1)/np.maximum(2*np.linalg.norm(qn@Q+coef[6:9],axis=1),1e-6)
                residual_e = np.sqrt(np.mean(error*error))
                accept_e = (.008<min(er) and max(er)<.075 and residual_e<.003
                            and np.linalg.norm(ep[:2]-center[:2])<.025
                            and floor+.2*height<ep[2]<floor+.8*height)
                robot.log('ellipsoid fit',{'center':ep.tolist(),'radii':er.tolist(),'residual':float(residual_e),'accepted':bool(accept_e)})
                if accept_e: center[:2]=ep[:2]

    if cylinder_center is not None:center[:2]=cylinder_center
    center[2] = floor + max(.016, min(.65*height, height-.008))
    robot.log('geometry', {'center':center.tolist(), 'floor':float(floor), 'top':float(top),
              'spans':(bounds[1]-bounds[0]).tolist(), 'axis':axes[:,0].tolist(),
              'initial_flange':p['flange_pose_odom']})
    closing = axes[:,0]
    widths = bounds[1]-bounds[0]
    initial_R = np.array(p['flange_pose_odom'])[:3,:3]
    natural = initial_R[:2,0]*np.cos(CLOSING_ANGLE)+initial_R[:2,1]*np.sin(CLOSING_ANGLE)
    natural /= np.linalg.norm(natural)
    natural_angle = np.arctan2(natural[1], natural[0])
    candidates = [0.] + [sign*a for a in np.deg2rad(np.arange(5,91,5)) for sign in [1.,-1.]]
    for delta in candidates if min(widths)<.050 else []:
        candidate = np.array([np.cos(natural_angle+delta), np.sin(natural_angle+delta)])
        span = np.diff(np.percentile(xy@candidate, [2,98]))[0]
        if span < .060:
            closing = candidate
            robot.log('nearest feasible closing direction',closing.tolist(),'span',float(span))
            break
    radial = center[:2]/np.linalg.norm(center[:2])
    radial_closing = np.array([-radial[1],radial[0]])
    radial_span = float(np.diff(np.percentile(xy@radial_closing,[2,98]))[0])
    return center, top, closing, p, floor, bounds[1]-bounds[0], radial_span


def tipped_pose(center, closing, reference):
    forward = np.array([-closing[1], closing[0]])
    if np.dot(forward, center[:2])<0: forward=-forward
    approach = np.r_[.5*forward, -np.sqrt(.75)]
    F = np.eye(4)
    F[:3,:3] = orientation(closing, reference, approach)
    F[:3,3] = center-PADS*approach
    H = F.copy(); H[:3,3] -= .06*approach
    return F, H, approach


def run(robot, object='object', where='on the counter', use='hold', then='', target=''):
    global BASE
    BASE = [0.,0.,0.]
    survey = np.eye(4); survey[:3,:3] = np.diag([1.,-1.,-1.]); survey[:3,3] = [.19,0.,.56]
    # Initial wrist view supplies a close RGB-D view before motion.
    geo = perceive(robot, object, use, then, target)
    if geo is None:
        return {'ok':False, 'reason':'no target geometry'}
    center, top, closing, p, floor, spans, radial_span = geo
    BASE = [.07,float(center[1]),0.]
    approach = None
    if top-floor > .030 and not LOCAL_GRASP:
        radial = center[:2]/np.linalg.norm(center[:2])
        if radial_span < .060:
            closing = np.array([-radial[1], radial[0]])
        forward = np.array([-closing[1], closing[0]])
        if np.dot(forward, radial) < 0: forward = -forward
        approach = np.r_[np.sqrt(.5)*forward, -np.sqrt(.5)]
        robot.log('tilted body grasp', approach.tolist())
    F = np.eye(4)
    F[:3,:3] = orientation(closing, np.array(p['flange_pose_odom'])[:3,:3], approach)
    F[:3,3] = center - PADS*F[:3,2]
    H = F.copy()
    if approach is None:
        H[2,3] = top + PADS + .045
    else:
        H[:3,3] -= .04*approach
    S = np.array(p['flange_pose_odom'])
    S[2,3] = max(H[2,3], top+PADS+.025)
    move(robot, S, 0, 40)
    p_hover = move(robot, H, 0, 45)
    if approach is None and np.linalg.norm(np.array(p_hover['flange_pose_odom'])[:3,3]-H[:3,3])>.025:
        F,H,approach = tipped_pose(center, closing, np.array(p_hover['flange_pose_odom'])[:3,:3])
        robot.log('tilted recovery after blocked hover')
        move(robot, H, 0, 35)
    view = robot.observe(['pole', 'wrist'])
    for name, cam in view['cameras'].items():
        robot.save_image('hover_'+name, cam['rgb'])
    p = move(robot, F, 0, 30)
    if approach is None and np.linalg.norm(np.array(p['flange_pose_odom'])[:3,3]-F[:3,3])>.04:
        F,H,approach = tipped_pose(center, closing, np.array(p['flange_pose_odom'])[:3,:3])
        robot.log('tilted recovery after blocked descent')
        move(robot, H, 0, 35)
        p = move(robot, F, 0, 30)
    if np.linalg.norm(np.array(p['flange_pose_odom'])[:3,3]-F[:3,3]) > .04:
        return {'ok':False, 'reason':'grasp pose not reached'}
    for k,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('pregrasp_'+k,c['rgb'])
    robot.act_base([BASE])
    robot.gripper_grasp(GRASP_FORCE, .05)
    p = robot.act_base([BASE]*35)["proprio"]
    robot.log('closed state', p)
    for k,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('closed_'+k,c['rgb'])
    L = F.copy(); L[2,3] += .20
    L[:2,3] -= .05*center[:2]/np.linalg.norm(center[:2])
    if LOCAL_GRASP:
        radial=center[:2]/np.linalg.norm(center[:2])
        forward=np.array([-closing[1],closing[0]])
        if np.dot(forward,radial)<0:forward=-forward
        tipped=np.r_[np.sqrt(.5)*forward,-np.sqrt(.5)]
        L[:3,:3]=orientation(closing,np.array(p['flange_pose_odom'])[:3,:3],tipped)
        goal=center+np.r_[-.08*radial,.20]
        L[:3,3]=goal-PADS*tipped
        robot.log('local lift inward and tilted',goal.tolist())
    p = move(robot, L, 1, 35)
    if p['gripper_width_m']>.004 and np.linalg.norm(np.array(p['flange_pose_odom'])[:3,3]-L[:3,3])>.04:
        robot.log('blocked lift joints',p['arm_q'])
        B = np.array(p['flange_pose_odom'])
        radial = center[:2]/np.linalg.norm(center[:2])
        B[:2,3] -= .05*radial
        robot.log('retract held object before lift')
        move(robot,B,1,25)
        L = B.copy(); L[2,3] += .13
        p = move(robot,L,1,35)
    for k,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('final_'+k,c['rgb'])
    return {'ok':p['gripper_width_m'] > .004, 'reason':'depth-based grasp completed'}
