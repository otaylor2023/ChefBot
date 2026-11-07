"""Named-object pick from RGB-D using only the chefbot-v1 robot API."""
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

PADS = 0.134  # PORT: the real pads' middle out from the flange (sim 0.091): measured on the robot 2026-11-06, by where the fingertips met the counter and a chapstick
FINGER_BAND = (.075 + .043, .115 + .043)  # PORT: where the open fingers are, out from the flange (sim .075-.115; the pads 4.3 cm further out)
MOVE_CAP = 300  # PORT: control steps (20 s) a move may take while still progressing (the lab's: n+50)
BASE = [0.,0.,0.]
LOCAL_GRASP = False
GRASP_FORCE = 2.
TOOL_HEAD = None
TOOL_RADIUS = 0.
TOOL_GRIP = None
PINCH_POINTS = None
ANCHOR_POINT = None
SUPPORT_POINT = None
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
    global LOCAL_GRASP, PINCH_POINTS, ANCHOR_POINT, SUPPORT_POINT, GRASP_FORCE, TOOL_HEAD, TOOL_RADIUS, TOOL_GRIP
    GRASP_FORCE = 2.
    PINCH_POINTS = None
    ANCHOR_POINT = None
    SUPPORT_POINT = None
    LOCAL_GRASP = False
    obs = robot.observe(['wrist'])
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
    if use != 'hold':model=min(models,key=lambda n:models[n]['input_per_mtok']+models[n]['output_per_mtok'])
    action = ''
    if use == 'squeeze':action = 'The intended use is squeezing this sprung mechanism twice: grasp ACROSS BOTH movable handles near their free ends so the jaws can actuate it, not the hinge. Choose local=true. '
    elif use == 'press':action = 'The intended use is pressing the tool head into a bowl: grasp its sliding handle, leaving its rod and working head free. Choose local=true. '
    elif use == 'tongs':action = 'The intended use is gripping food using the tool: grasp across its two movable handles, leaving its food-contact jaws free. Choose local=true. '
    ans = robot.llm(model, action+'Identify the '
+object.replace('_', ' ')+
        ' on the kitchen counter and choose a stable parallel-jaw grasp. The hand opens to 80 mm. '
        'Return x,y for object identity; gx,gy for the CENTER of a suitable grasp region. '
        'Use local=false when grasping the whole small body. If the body is too wide, local=true and '
        'choose a narrow load-bearing handle, protrusion, or accessible rim, with free space on both sides. '
        'For a long thin object choose its shaft rather than a wide end. Choose a visible surface point. '
        'Coordinates normalized 0..1000, x horizontal then y vertical. '
        'Also estimate the minimum practical gripping force in Newtons for its weight, torque and mechanism resistance with rubber pads. Account for material fragility. Return JSON with x,y,gx,gy,local and force_n.',
        images=[cam['rgb']],max_output_tokens=2048,temperature=0,
        response_schema={'type':'object','properties':{**{k:{'type':'number'} for k in ['x','y','gx','gy','force_n']},
            'local':{'type':'boolean'}},'required':['x','y','gx','gy','local','force_n']})
    robot.log('identity', ans)
    semantic_point = None
    grasp_point = None
    local_grasp = False
    try:
        t = ans['text']; a = json.loads(t[t.index('{'):t.rindex('}')+1])
        u, v = float(a['x'])*cam['rgb'].shape[1]/1000, float(a['y'])*cam['rgb'].shape[0]/1000
        semantic_point = [u,v]
        grasp_point = [float(a['gx'])*.640,float(a['gy'])*.480]
        local_grasp = bool(a['local'])
        GRASP_FORCE = float(np.clip(a.get('force_n',2.),.5,50.))
        candidates = [d for d in ds if d['box'][0] <= u <= d['box'][2] and d['box'][1] <= v <= d['box'][3]]
        if candidates:
            best_score = max(d['score'] for d in candidates)
            candidates = [d for d in candidates if d['score']>=.75*best_score]
        candidates.sort(key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]), reverse=True)
        if use != 'hold':
            # A close crop supplies enough pixels to locate thin moving members.
            if candidates: bb=np.array(candidates[0]['box'])
            else: bb=np.array([u-65,v-65,u+65,v+65])
            x0,y0=np.maximum(0,np.floor(bb[:2]-12)).astype(int)
            x1,y1=np.minimum([640,480],np.ceil(bb[2:]+12)).astype(int)
            crop=cv2.resize(cam['rgb'][y0:y1,x0:x1],(640,640))
            robot.save_image('detail',crop)
            details=action+'Locate the exact two opposing jaw CONTACT points for picking up '+object+'. '
            details+='Points must lie on the object, not the support. For two movable handles, put one contact on EACH handle at the same distance from the hinge, near their free ends. For one handle, put contacts on opposite sides of the distal half, clear of the rest ridge. '
            details+='Also mark the mechanical hinge for a squeeze task, or the working head center for a press task, as anchor. Also mark one blank exposed point on the support/rest (NOT tool) as support_x,support_y. Return JSON p1_x,p1_y,p2_x,p2_y,anchor_x,anchor_y,support_x,support_y. Each x is horizontal left-to-right, y vertical top-to-bottom, normalized 0..1000 in THIS close-up.'
            ar=robot.llm(model,details,images=[crop],max_output_tokens=2048,temperature=0,
                response_schema={'type':'object','properties':{k:{'type':'number'} for k in ['p1_x','p1_y','p2_x','p2_y','anchor_x','anchor_y','support_x','support_y']},'required':['p1_x','p1_y','p2_x','p2_y','anchor_x','anchor_y','support_x','support_y']})
            robot.log('contact_points',ar)
            tt=ar['text'];aa=json.loads(tt[tt.index('{'):tt.rindex('}')+1])
            PINCH_POINTS=np.array([[aa['p1_x'],aa['p1_y']],[aa['p2_x'],aa['p2_y']]])*[(x1-x0)/1000,(y1-y0)/1000]+[x0,y0]
            ANCHOR_POINT=np.array([aa['anchor_x'],aa['anchor_y']])*[(x1-x0)/1000,(y1-y0)/1000]+[x0,y0]
            SUPPORT_POINT=np.array([aa['support_x'],aa['support_y']])*[(x1-x0)/1000,(y1-y0)/1000]+[x0,y0]
            grasp_point=PINCH_POINTS.mean(axis=0).tolist();local_grasp=True
        if PINCH_POINTS is not None:
            mask = robot.segment(cam['rgb'],points=[*PINCH_POINTS.tolist(),ANCHOR_POINT.tolist(),SUPPORT_POINT.tolist()],point_labels=[1,1,1,0])[0]
        elif candidates:
            mask = robot.segment(cam['rgb'], boxes=[candidates[0]['box']])[0]
        else:
            mask = robot.segment(cam['rgb'], points=[[u, v]], point_labels=[1])[0]
    except (ValueError, KeyError, IndexError) as e:
        robot.log('semantic fallback', str(e))
        if not ds:
            return None
        mask = robot.segment(cam['rgb'], boxes=[ds[0]['box']])[0]
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
    pts = pts[pts[:,2] > floor+.005]
    if len(pts) < 20:
        return None
    if grasp_point is not None and local_grasp:
        LOCAL_GRASP = True
        if PINCH_POINTS is not None:
            def surface(point):
                u,v=np.rint(point).astype(int);u=int(np.clip(u,3,636));v=int(np.clip(v,3,476))
                vu=np.column_stack(np.where(mask&valid)); ds=np.sum((vu-np.array([v,u]))**2,axis=1)
                selected=vu[np.argsort(ds)[:9]]
                region=xyz[selected[:,0],selected[:,1]]
                high=region[region[:,2]>=np.percentile(region[:,2],60)]
                return np.median(high,axis=0)
            contacts=np.array([surface(point) for point in PINCH_POINTS])
            anchor=surface(ANCHOR_POINT)
            center=contacts.mean(axis=0)
            closing=contacts[1,:2]-contacts[0,:2];width=float(np.linalg.norm(closing))
            if width<.003:return None
            if use=='squeeze':
                fraction=min(1.,.040/width)
                center=anchor+fraction*(center-anchor)
                width*=fraction
            elif use=='press':
                axis=center[:2]-anchor[:2];axis/=max(np.linalg.norm(axis),1e-6)
                closing=np.array([-axis[1],axis[0]])
            closing/=np.linalg.norm(closing)
            center[2]=max(floor+.014,float(contacts[:,2].max())+.006)
            top=float(contacts[:,2].max())
            if use=='press':
                TOOL_GRIP=center.copy();TOOL_HEAD=anchor.copy()
                headpts=xyz[mask&valid];headpts=headpts[np.linalg.norm(headpts-anchor,axis=1)<.055]
                TOOL_RADIUS=.5*float(np.ptp(np.percentile(headpts[:,2],[5,95]))) if len(headpts)>10 else .015
                if len(headpts)>20:
                    origin=headpts.mean(axis=0);q=headpts-origin
                    fit=np.linalg.lstsq(np.column_stack([2*q,np.ones(len(q))]),np.sum(q*q,axis=1),rcond=None)[0]
                    cc=origin+fit[:3];radius=np.sqrt(max(0.,fit[3]+fit[:3]@fit[:3]))
                    residual=np.sqrt(np.mean((np.linalg.norm(headpts-cc,axis=1)-radius)**2))
                    if .008<radius<.06 and residual<.006 and np.linalg.norm(cc-anchor)<.06:
                        TOOL_HEAD=cc;TOOL_RADIUS=float(radius)
                robot.log('working head',TOOL_HEAD.tolist(),'radius',TOOL_RADIUS)
            robot.log('paired geometry',{'contacts':contacts.tolist(),'center':center.tolist(),'closing':closing.tolist(),'width':width})
            return center,top,closing,p,floor,np.array([width,.05]),1.
        uv=np.column_stack(np.where(mask&valid))[:,::-1]
        pd=np.sum((uv-np.array(grasp_point))**2,axis=1)
        nearby=uv[pd<10**2]
        if len(nearby)>4:
            heights=xyz[nearby[:,1],nearby[:,0],2]
            nearby=nearby[heights>=np.percentile(heights,85)]
            u,v=nearby[np.argmin(np.sum((nearby-np.array(grasp_point))**2,axis=1))]
        else:u,v=uv[np.argmin(pd)]
        gp=xyz[v,u]
        local=pts[(np.linalg.norm(pts[:,:2]-gp[:2],axis=1)<.035)&(pts[:,2]>gp[2]-.018)]
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
    # Perceive using the initial wrist camera.
    geo = perceive(robot, object, use, then, target)
    if geo is None:
        return {'ok':False, 'reason':'no target geometry'}
    center, top, closing, p, floor, spans, radial_span = geo
    bowl=locate_bowl(robot,target) if use=='press' else None
    BASE = [.07,float(center[1]),0.]
    approach = None
    if use == 'squeeze':
        forward=np.array([-closing[1],closing[0]])
        if np.dot(forward,center[:2])>0:forward=-forward
        approach=np.r_[-np.sqrt(.5)*forward,-np.sqrt(.5)]
    if (top-floor > .030 and not LOCAL_GRASP) or use == 'press':
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
    pickup=np.array(p['flange_pose_odom'])
    robot.act_base([BASE])
    robot.gripper_grasp(GRASP_FORCE, .05)
    p = robot.act_base([BASE]*35)["proprio"]
    robot.log('closed state', p)
    for k,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('closed_'+k,c['rgb'])
    L = F.copy(); L[2,3] += .13
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
    if use=='press' and bowl is not None and p['gripper_width_m']>.003:
        return press_action(robot,pickup,bowl)
    return {'ok':False if use!='hold' else p['gripper_width_m']>.004,'reason':'pickup attempted; use incomplete' if use!='hold' else 'depth-based grasp completed'}


def locate_bowl(robot,target):
    obs=robot.observe(['pole']);cam=obs['cameras']['pole'];xyz=cloud(cam,obs['proprio'])
    ds=robot.detect(cam['rgb'],[target or 'bowl'],model='grounding-dino',threshold=.2)
    if not ds:return None
    mask=robot.segment(cam['rgb'],boxes=[max(ds,key=lambda d:d['score'])['box']])[0]
    pts=xyz[mask&(cam['depth']>0)]
    if len(pts)<20:return None
    lo,hi=np.percentile(pts,[2,98],axis=0);center=(lo+hi)/2
    interior=pts[np.linalg.norm(pts[:,:2]-center[:2],axis=1)<.22*min(hi[:2]-lo[:2])]
    bottom=float(np.percentile(interior[:,2],20)) if len(interior)>10 else float(lo[2])
    center[2]=bottom
    robot.log('bowl geometry',center.tolist(),'rim',float(hi[2]))
    return center,float(hi[2])


def held_head(robot):
    obs=robot.observe(['pole']);cam=obs['cameras']['pole'];p=obs['proprio'];xyz=cloud(cam,p)
    robot.save_image('held_view',cam['rgb'])
    models=robot.models()['llm'];model=min(models,key=lambda n:models[n]['input_per_mtok']+models[n]['output_per_mtok'])
    a=robot.llm(model,'Locate the center of the wire working head of the press whisk currently held by the robot. It may have swung downward. Point to the head, not the handle or robot. Return JSON x,y (horizontal,vertical normalized 0..1000) and visible boolean.',images=[cam['rgb']],max_output_tokens=512,temperature=0,response_schema={'type':'object','properties':{'x':{'type':'number'},'y':{'type':'number'},'visible':{'type':'boolean'}},'required':['x','y','visible']})
    robot.log('held head landmark',a)
    try:
        t=a['text'];a=json.loads(t[t.index('{'):t.rindex('}')+1])
        if not a['visible']:return None
        u,v=np.rint([a['x']*.640,a['y']*.480]).astype(int)
        u=int(np.clip(u,4,635));v=int(np.clip(v,4,475))
        mask=robot.segment(cam['rgb'],points=[[u,v]],point_labels=[1])[0]
        F=np.array(p['flange_pose_odom']);grip=F[:3,3]+PADS*F[:3,2]
        pts=xyz[mask&(cam['depth']>0)]
        pts=pts[(np.linalg.norm(pts-grip,axis=1)<.3)&(pts[:,2]>grip[2]-.30)]
        seed=xyz[v,u]
        near=pts[np.linalg.norm(pts-seed,axis=1)<max(.04,1.5*TOOL_RADIUS)]
        if len(near)<8:return None
        origin=near.mean(axis=0);q=near-origin
        fit=np.linalg.lstsq(np.column_stack([2*q,np.ones(len(q))]),np.sum(q*q,axis=1),rcond=None)[0]
        head=origin+fit[:3];radius=np.sqrt(max(0.,fit[3]+np.sum(fit[:3]**2)))
        if not(.008<radius<.065 and np.linalg.norm(head-origin)<.07):head=np.median(near,axis=0)
        robot.log('held head measured',head.tolist(),'grip',grip.tolist())
        return head,grip,F
    except (ValueError,KeyError,IndexError):return None


def press_action(robot,pickup,bowl):
    global BASE
    center,rim=bowl
    observed=held_head(robot)
    if observed is None:return {'ok':False,'reason':'held head not visible'}
    head,grip,actual=observed
    axis=head-grip;length=np.linalg.norm(axis)
    if length<.045:return {'ok':False,'reason':'no extended held head'}
    axis/=length
    localhead=actual[:3,:3].T@(head-actual[:3,3])
    working_axis=np.array([-.5,0.,-np.sqrt(.75)])
    Q=Rotation.align_vectors(working_axis[None,:],axis[None,:])[0].as_matrix()
    R=Q@actual[:3,:3]
    robot.act_base([BASE]);robot.gripper_grasp(min(70.,7*GRASP_FORCE),.05);robot.act_base([BASE]*12)
    high=center.copy();high[2]=rim+TOOL_RADIUS+.06
    F=actual.copy();F[:3,:3]=R;F[:3,3]=head-R@localhead;F[2,3]+=.06
    move(robot,F,1,70)
    BASE=[.07,float(center[1]),0.]
    F[:3,3]=high-R@localhead
    p=move(robot,F,1,100)
    if np.linalg.norm(np.array(p['flange_pose_odom'])[:3,3]-F[:3,3])>.025:
        return {'ok':False,'reason':'bowl transfer unreachable'}
    contact=center.copy();contact[2]+=TOOL_RADIUS+.004
    C=F.copy();C[:3,3]=contact-R@localhead
    p=move(robot,C,1,65)
    if np.linalg.norm(np.array(p['flange_pose_odom'])[:3,3]-C[:3,3])>.025:
        move(robot,F,1,50);return {'ok':False,'reason':'head insertion obstructed'}
    excursion=float(np.clip(.85*length,.04,.15))
    robot.log('press plan',{'head_local':localhead.tolist(),'excursion':excursion,'contact':contact.tolist()})
    for cycle in range(2):
        for dz in np.arange(.004,excursion+.002,.004):
            D=C.copy();D[:3,3]+=dz*working_axis
            r=robot.act([row(D,1)+BASE]*4)
            actual=np.array(r['proprio']['flange_pose_odom'])
            robot.log('press increment',cycle,float(dz),'lag',float(np.dot(D[:3,3]-actual[:3,3],working_axis)))
            if np.dot(D[:3,3]-actual[:3,3],working_axis)>.040:break
        robot.act_base([BASE]*5)
        move(robot,C,1,55);robot.act_base([BASE]*10)
    move(robot,F,1,80)
    for k,c in robot.observe(['pole','wrist'])['cameras'].items():robot.save_image('used_'+k,c['rgb'])
    return {'ok':False,'reason':'two measured-pose press cycles attempted'}
