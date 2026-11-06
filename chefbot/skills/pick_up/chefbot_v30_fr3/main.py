"""Named-object pick from RGB-D using only the chefbot-v1 robot API.

chefbot_v30 (the robocasa_pick lab's, on the sim's panda_franka) ported to the real FR3 on ChefBot: only its hand's
numbers and a move's step cap changed (marked PORT), every other line as the lab left it. The real Franka Hand sits on a 27.25 mm camera
holder: the fingertip pads' center is 0.13065 m out from the flange (the sim hand's 0.093), the fingers 2.7 cm
further out in the wrist camera's view, and they close along +45 deg in the flange's xy (the sim's -45; the Desk
end-effector transform turns the hand +45 deg on the flange). A move may run up to MOVE_CAP steps while the arm is
still closing in (it still stops on arriving, or once progress stalls): the real arm's moves to a tilted grasp
are joint-speed bound (the wrist turns ~150 deg) and took ~10 s, where the lab's n+50 steps (~6 s) ended the hover
16-19 cm short, and the grasp then came from above and missed (2026-11-04, three runs)."""
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

PADS = 0.13065  # PORT: the real pads' center (sim 0.093)
CLOSING_ANGLE = np.pi/4  # PORT: the real hand's (sim -pi/4). Documented fallback; estimate from the visible open fingers.
FINGER_BAND = (.075 + .027, .115 + .027)  # PORT: where the open fingers are, out from the flange (sim .075-.115)
MOVE_CAP = 300  # PORT: control steps (20 s) a move may take while still progressing (the lab's: n+50)


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
    T = np.linalg.inv(planar(*proprio['base_pose_odom'])) @ cam['cam_to_odom']
    return q @ T[:3, :3].T + T[:3, 3]


def calibrate_fingers(robot, xyz, cam, p):
    global CLOSING_ANGLE
    F = np.array(p['flange_pose_root'])
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
    r = robot.act_arm([row(F,g)]*first)
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
        r = robot.act_arm([row(F,g)]*count)
        remaining -= count
    robot.log('move', np.round(F[:3, 3], 4).tolist(), 'err', r['steps'][-1],
              'actual', np.round(np.array(r['proprio']['flange_pose_root'])[:3, 3], 4).tolist(),
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


def perceive(robot, object):
    obs = robot.observe(['wrist'])
    cam, p = obs['cameras']['wrist'], obs['proprio']
    xyz = cloud(cam, p)
    calibrate_fingers(robot, xyz, cam, p)
    robot.save_image('initial', cam['rgb'])
    ds = robot.detect(cam['rgb'], [object.replace('_', ' ')], model='grounding-dino', threshold=0.2)
    ds.sort(key=lambda d: d['score'], reverse=True)
    robot.log('detections', ds)
    # Always verify identity, including confident detector mistakes.
    models = robot.models()['llm']
    model = min(models, key=lambda n: models[n]['input_per_mtok']+models[n]['output_per_mtok'])
    ans = robot.llm(model, 'Locate the '+object.replace('_', ' ')+
        ' on the kitchen counter. Return JSON only: {"x": x, "y": y} for a point inside its main body. '
        'Coordinates are integers normalized 0 to 1000, x left to right, y top to bottom. '
        'Identify the requested object carefully, not its neighbor or the counter.',
        images=[cam['rgb']], max_output_tokens=1024, temperature=0,
        response_schema={'type':'object', 'properties':{'x':{'type':'number'}, 'y':{'type':'number'}}, 'required':['x','y']})
    robot.log('identity', ans)
    semantic_point = None
    try:
        t = ans['text']; a = json.loads(t[t.index('{'):t.rindex('}')+1])
        u, v = float(a['x'])*cam['rgb'].shape[1]/1000, float(a['y'])*cam['rgb'].shape[0]/1000
        semantic_point = [u,v]
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
        return min(widths)<.12
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

    center[2] = floor + max(.016, min(.5*height, height-.01))
    robot.log('geometry', {'center':center.tolist(), 'floor':float(floor), 'top':float(top),
              'spans':(bounds[1]-bounds[0]).tolist(), 'axis':axes[:,0].tolist(),
              'initial_flange':p['flange_pose_root']})
    closing = axes[:,0]
    widths = bounds[1]-bounds[0]
    initial_R = np.array(p['flange_pose_root'])[:3,:3]
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


def run(robot, object='object'):
    geo = perceive(robot, object)
    if geo is None:
        return {'ok':False, 'reason':'no target geometry'}
    center, top, closing, p, floor, spans, radial_span = geo
    approach = None
    if top-floor > .10 and top-floor > 1.2*max(spans):
        radial = center[:2]/np.linalg.norm(center[:2])
        if radial_span < .060:
            closing = np.array([-radial[1], radial[0]])
        forward = np.array([-closing[1], closing[0]])
        if np.dot(forward, radial) < 0: forward = -forward
        approach = np.r_[np.sqrt(.5)*forward, -np.sqrt(.5)]
        robot.log('tilted body grasp', approach.tolist())
    F = np.eye(4)
    F[:3,:3] = orientation(closing, np.array(p['flange_pose_root'])[:3,:3], approach)
    F[:3,3] = center - PADS*F[:3,2]
    H = F.copy()
    if approach is None:
        H[2,3] = top + PADS + .045
    else:
        H[:3,3] -= .04*approach
    if approach is None:
        S = np.array(p['flange_pose_root'])
        S[2,3] = H[2,3]
        move(robot, S, 0, 30)
    p_hover = move(robot, H, 0, 45)
    if approach is None and np.linalg.norm(np.array(p_hover['flange_pose_root'])[:3,3]-H[:3,3])>.025:
        F,H,approach = tipped_pose(center, closing, np.array(p_hover['flange_pose_root'])[:3,:3])
        robot.log('tilted recovery after blocked hover')
        move(robot, H, 0, 35)
    view = robot.observe(['pole', 'wrist'])
    for name, cam in view['cameras'].items():
        robot.save_image('hover_'+name, cam['rgb'])
    p = move(robot, F, 0, 30)
    if approach is None and np.linalg.norm(np.array(p['flange_pose_root'])[:3,3]-F[:3,3])>.04:
        F,H,approach = tipped_pose(center, closing, np.array(p['flange_pose_root'])[:3,:3])
        robot.log('tilted recovery after blocked descent')
        move(robot, H, 0, 35)
        p = move(robot, F, 0, 30)
    if np.linalg.norm(np.array(p['flange_pose_root'])[:3,3]-F[:3,3]) > .04:
        return {'ok':False, 'reason':'grasp pose not reached'}
    p = move(robot, F, 1, 15, settle=False)
    L = F.copy(); L[2,3] += .13
    p = move(robot, L, 1, 35)
    if p['gripper_width_m']>.004 and np.linalg.norm(np.array(p['flange_pose_root'])[:3,3]-L[:3,3])>.04:
        robot.log('blocked lift joints',p['arm_q'])
        B = np.array(p['flange_pose_root'])
        radial = center[:2]/np.linalg.norm(center[:2])
        B[:2,3] -= .05*radial
        robot.log('retract held object before lift')
        move(robot,B,1,25)
        L = B.copy(); L[2,3] += .13
        p = move(robot,L,1,35)
    return {'ok':p['gripper_width_m'] > .004, 'reason':'depth-based grasp completed'}
