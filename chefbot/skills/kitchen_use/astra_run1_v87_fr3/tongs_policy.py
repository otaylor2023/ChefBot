"""Astra selects all scene-dependent actions; this harness measures and executes them."""
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw
PADS = 0.134  # PORT: the real pads' middle out from the flange (sim 0.091): measured on the robot 2026-11-06, by where the fingertips met the counter and a chapstick
MOVE_CAP = 300  # PORT: control steps (20 s) a move may take while still progressing

SYS='You are Astra, the visual decision policy controlling a physical kitchen robot. Use calibrated images and measurements to choose accurate gentle grasps and tool actions. Return concise JSON. No tools are available.'
def obj(props): return dict(type='object',additionalProperties=False,required=list(props),properties=props)
def vec(n): return dict(type='array',items={'type':'number'},minItems=n,maxItems=n)
def num(): return {'type':'number'}
def string(): return {'type':'string'}
PLAN=obj({'grasp_view':{'type':'string','enum':['full','left','right']},'center_px':vec(2),'top_px':vec(2),'support_px':vec(2),'height_fraction':num(),
          'closing_line_px':vec(4),'axis_mode':{'type':'string','enum':['closing','long_axis']},
          'tilt_deg':num(),'lift_tilt_deg':num(),'lift_tilt_direction_xy':vec(2),'tilt_direction_xy':vec(2),'force_n':num(),'open_width_m':num(),
          'base_delta_xy':vec(2),'lift_delta_xyz':vec(3),'landmarks':{'type':'array','maxItems':6,'items':obj({'label':string(),'view':{'type':'string','enum':['pole','wrist','top']},'px':vec(2),'attached':{'type':'boolean'}})},'note':string()})
PLAN['properties'].update(capture_mode={'type':'string','enum':['force','width']},outer_contacts_px=vec(4),capture_width_offset_m=num())
PLAN['required']+=['capture_mode','outer_contacts_px','capture_width_offset_m']
PLAN['properties']['probe_forces_n']={'type':'array','items':num(),'maxItems':2}
PLAN['required'].append('probe_forces_n')
PLAN['properties']['open_after_lift_fraction']=num()
PLAN['required'].append('open_after_lift_fraction')
PLAN['properties']['clear_view']={'type':'boolean'}
PLAN['required'].append('clear_view')
SCOUT=obj({'center_px':vec(2),'clear_view':{'type':'boolean'},'note':string()})
REFINE=obj({k:PLAN['properties'][k] for k in ['center_px','top_px','support_px','height_fraction','closing_line_px','axis_mode','tilt_deg','tilt_direction_xy','force_n','open_width_m','note']})
LABELS={'type':'array','items':{'type':'string'},'maxItems':2}
ALIGN=obj({'from_labels':LABELS,'to_labels':LABELS,'axis_xyz':vec(3),'approach_hint_xyz':vec(3)})
PLAN['properties']['lift_alignment']=ALIGN
PLAN['required'].append('lift_alignment')
WAY=obj({'destination_labels':LABELS,'anchor_labels':LABELS,'pad_xyz':vec(3),'approach_xyz':vec(3),'closing_xyz':vec(3),
         'gripper_kind':{'type':'string','enum':['keep','move','grasp']},'gripper_value':num(),'seconds':num()})
CHECK=obj({'action':{'type':'string','enum':['done','retry','sequence','stop']},'ok':{'type':'boolean'},'finish_width_range_m':vec(2),
           'fresh_landmarks':{'type':'array','maxItems':6,'items':obj({'label':string(),'view':{'type':'string','enum':['pole','wrist','top']},'px':vec(2),'attached':{'type':'boolean'}})},'alignment':ALIGN,'base_delta_xy':vec(2),'repeat':{'type':'integer','minimum':1,'maximum':3},'waypoints':{'type':'array','items':WAY,'maxItems':10},'note':string()})

FRAME=obj({'approach_xyz':vec(3),'closing_xyz':vec(3),'anchor_labels':LABELS,'destination_labels':LABELS})
CHECK['properties']['fresh_landmarks']['items']['properties']['view']['enum'].append('target')
CHECK['properties'].pop('waypoints')
CHECK['required'].remove('waypoints')
CHECK['properties']['frame']=FRAME
CHECK['properties']['points']={'type':'array','items':vec(6),'maxItems':10}
CHECK['required']+=['frame','points']
def expand_points(b):
    result=[]
    for p in b['points']:
        w=dict(b['frame']);w.update(pad_xyz=p[:3],gripper_kind=['keep','move','grasp'][int(np.clip(round(p[3]),0,2))],gripper_value=p[4],seconds=p[5]);result.append(w)
    return result

def point(cam,px,refine=False):
    h,w=cam['depth'].shape;u,v=np.clip(np.rint(px),[0,0],[w-1,h-1]).astype(int)
    if refine:
        ys,xs=np.mgrid[max(0,v-3):min(h,v+4),max(0,u-3):min(w,u+4)]
        dep=cam['depth'][ys,xs];good=np.isfinite(dep)&(dep>0)&((xs-u)**2+(ys-v)**2<=9)
        if good.any():
            j,i=np.unravel_index(np.argmin(np.where(good,dep,np.inf)),good.shape);u,v=xs[j,i],ys[j,i]
    z=float(cam['depth'][v,u]);K=cam['K']
    if not np.isfinite(z) or z<=0: raise ValueError('no depth at selected pixel')
    return (cam['cam_to_odom']@np.array([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z,1.]))[:3]

def overhead(cams,mark=True):
    xyzs=[];colors=[]
    for cam in cams.values():
        d=cam['depth'];vv,uu=np.indices(d.shape);K=cam['K']
        xyz=np.stack([(uu-K[0,2])*d/K[0,0],(vv-K[1,2])*d/K[1,1],d],-1).reshape(-1,3)
        C=cam['cam_to_odom'];xyz=xyz@C[:3,:3].T+C[:3,3]
        good=(d.ravel()>0)&np.all(np.isfinite(xyz),axis=1)
        good&=(xyz[:,0]>=0.1)&(xyz[:,0]<1.0)&(xyz[:,1]>-.5)&(xyz[:,1]<=.5)&(xyz[:,2]>0)&(xyz[:,2]<1.2)
        xyzs.append(xyz[good]);colors.append(cam['rgb'].reshape(-1,3)[good])
    xyz=np.concatenate(xyzs);rgb=np.concatenate(colors)
    u=np.clip(np.rint((.5-xyz[:,1])*640),0,639).astype(int);v=np.clip(np.rint((1.-xyz[:,0])*640),0,575).astype(int)
    flat=v*640+u;order=np.lexsort((-xyz[:,2],flat));f=flat[order];ids=order[np.r_[True,f[1:]!=f[:-1]]]
    z=np.full((576,640),np.nan);im=np.zeros((576,640,3),dtype=np.uint8);im[:]=[170,35,170]
    z[v[ids],u[ids]]=xyz[ids,2];im[v[ids],u[ids]]=rgb[ids]
    dist,ij=distance_transform_edt(~np.isfinite(z),return_indices=True);fill=(~np.isfinite(z))&(dist<=2)
    z[fill]=z[ij[0][fill],ij[1][fill]];im[fill]=im[ij[0][fill],ij[1][fill]]
    pic=Image.fromarray(im);dr=ImageDraw.Draw(pic)
    for x in range(0,640,64):
        dr.line((x,0,x,575),fill=(95,95,95));dr.text((x+2,18),str(x),fill='yellow')
    for y in range(0,576,64):
        dr.line((0,y,639,y),fill=(95,95,95));dr.text((2,y+2),str(y),fill='yellow')
    dr.text((4,2),'METRIC TOP VIEW: 64 px = 10 cm, forward UP',fill='yellow')
    return (np.array(pic) if mark else im),z

def xy(px):
    u,v=px;return np.array([1-v/640,.5-u/640])
def selected(z,px):
    u,v=np.clip(np.rint(px),[0,0],[639,575]).astype(int);h=z[v,u]
    if not np.isfinite(h):
        ys,xs=np.mgrid[max(0,v-3):min(576,v+4),max(0,u-3):min(640,u+4)]
        good=np.isfinite(z[ys,xs])&((xs-u)**2+(ys-v)**2<=9)
        if not good.any():raise ValueError('selected top-view pixel has unknown depth')
        k=np.argmin(np.where(good,(xs-u)**2+(ys-v)**2,np.inf));j,i=np.unravel_index(k,good.shape)
        u,v=xs[j,i],ys[j,i];h=z[v,u]
    return np.r_[xy([u,v]),h]

def selected_top(z,px):
    # Bounded local depth-edge refinement, never object localization.
    u,v=np.clip(np.rint(px),[0,0],[639,575]).astype(int)
    ys,xs=np.mgrid[max(0,v-3):min(576,v+4),max(0,u-3):min(640,u+4)]
    valid=np.isfinite(z[ys,xs])&((xs-u)**2+(ys-v)**2<=9)
    if not valid.any():return selected(z,px)
    k=np.argmax(np.where(valid,z[ys,xs],-np.inf));j,i=np.unravel_index(k,valid.shape)
    return np.r_[xy([xs[j,i],ys[j,i]]),z[ys[j,i],xs[j,i]]]

def pose(pad,approach,closing):
    a=np.asarray(approach,dtype=float);a/=np.linalg.norm(a)
    c=np.asarray(closing,dtype=float);c-=a*np.dot(a,c);c/=np.linalg.norm(c)
    b=np.cross(a,c)
    # Pad closing axis is (flange x + flange y)/sqrt(2); flange z approaches.
    F=np.eye(4);F[:3,:3]=np.column_stack([(c-b)/np.sqrt(2),(c+b)/np.sqrt(2),a]);F[:3,3]=np.asarray(pad)-PADS*a
    return F

def row(F,g,base):
    x,y,z,w=Rotation.from_matrix(F[:3,:3]).as_quat();return [*F[:3,3],w,x,y,z,g,*base]
def move(robot,F,g,base,seconds=1.,settle=5):
    # PORT: the real host holds the base in an arm-only run, so arrival is the arm's alone (the lab's also wanted
    # base_error_m < .01, which a held base never meets), and a move may take up to MOVE_CAP more steps while still
    # progressing (the lab's: `settle` 6-step chunks, or 20 here; the real arm's tilted moves are joint-speed bound)
    r=robot.act([row(F,g,base)]*max(1,int(seconds*15)))
    last_score=None;plateau=0
    for _ in range(MOVE_CAP//6 if settle else 0):
        e=r['steps'][-1]
        if e['pos_error_m']<.008 and e['rot_error_rad']<.10:break
        score=e['pos_error_m']+.1*e['rot_error_rad']
        if last_score is not None and last_score-score<.0008:plateau+=1
        else:plateau=0
        if plateau>=3:break
        last_score=score
        r=robot.act([row(F,g,base)]*6)
    robot.log('motion',r['steps'][-1],'width',r['proprio']['gripper_width_m'])
    return r

def descend(robot,F,g,base):
    F0=np.array(robot.proprio()['flange_pose_odom'])
    distance=np.linalg.norm(F[:3,3]-F0[:3,3])
    n=max(15,int(np.ceil(distance/.025*15)))
    slerp=Slerp([0,1],Rotation.from_matrix([F0[:3,:3],F[:3,:3]]))
    rows=[]
    for t in np.linspace(0,1,n+1)[1:]:
        Q=np.eye(4);Q[:3,3]=(1-t)*F0[:3,3]+t*F[:3,3];Q[:3,:3]=slerp(t).as_matrix()
        rows.append(row(Q,g,base))
    r=robot.act(rows)
    robot.log('descent',r['steps'][-1])
    return move(robot,F,g,base,.4,2)

def depth_detail(zmap,u0,v0):
    z=zmap[v0:v0+128,u0:u0+128];valid=np.isfinite(z)
    lo=float(np.floor(np.nanmin(z)*100)/100);hi=max(lo+.05,float(np.ceil(np.nanmax(z)*100)/100))
    t=np.clip((np.nan_to_num(z,nan=lo)-lo)/(hi-lo),0,1)
    stops=np.array([[25,45,210],[0,210,240],[240,230,30],[230,35,20]],float)
    rgb=np.stack([np.interp(t,[0,1/3,2/3,1],stops[:,i]) for i in range(3)],-1).astype(np.uint8);rgb[~valid]=[170,35,170]
    pic=Image.fromarray(rgb).resize((640,640));dr=ImageDraw.Draw(pic)
    for a in range(0,640,80):
        dr.line((a,0,a,639),fill=(150,150,150));dr.text((a+2,18),str(a),fill='white')
        dr.line((0,a,639,a),fill=(150,150,150));dr.text((2,a+2),str(a),fill='white')
    dr.text((4,2),f'HEIGHT z meters: blue={lo:.3f},cyan={lo+(hi-lo)/3:.3f},yellow={lo+2*(hi-lo)/3:.3f},red={hi:.3f}',fill='white')
    for v in range(24,128,24):
        for u in range(24,128,24):
            if valid[v,u]:dr.text((u*5,v*5),f'{z[v,u]:.3f}',fill='white',stroke_width=1,stroke_fill='black')
    return np.asarray(pic)

MODEL_LATENCIES=[]
def transport(robot,F,g,base):
    F0=np.array(robot.proprio()['flange_pose_odom']);P0=F0[:3,3]+PADS*F0[:3,2];P1=F[:3,3]+PADS*F[:3,2]
    angle=np.linalg.norm(Rotation.from_matrix(F[:3,:3]@F0[:3,:3].T).as_rotvec())
    n=max(15,int(np.ceil(max(np.linalg.norm(P1-P0)/.10,angle/.7)*15)))
    turns=Slerp([0,1],Rotation.from_matrix([F0[:3,:3],F[:3,:3]]));rows=[]
    for t in np.linspace(0,1,n+1)[1:]:
        Q=np.eye(4);Q[:3,:3]=turns(t).as_matrix();Q[:3,3]=(1-t)*P0+t*P1-PADS*Q[:3,2];rows.append(row(Q,g,base))
    robot.act(rows)
    return move(robot,F,g,base,.5,6)

def hand_crop(cam,pad):
    C=np.linalg.inv(cam['cam_to_odom']);K=cam['K']
    pts=np.asarray([np.asarray(pad)+[dx,dy,dz] for dx in [-.12,.12] for dy in [-.12,.12] for dz in [-.22,.09]])
    pc=pts@C[:3,:3].T+C[:3,3];uv=pc[:,:2]/pc[:,2,None]
    uv=uv*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    x0,y0=np.maximum(np.floor(uv.min(0)).astype(int),[0,0]);x1,y1=np.minimum(np.ceil(uv.max(0)).astype(int),[640,480])
    if x1<=x0 or y1<=y0:return cam['rgb']
    pic=Image.fromarray(cam['rgb'][y0:y1,x0:x1]);factor=min(480/pic.width,480/pic.height)
    return np.asarray(pic.resize((max(1,int(pic.width*factor)),max(1,int(pic.height*factor)))))

def reference_landmarks(raw,landmarks,grasp_px):
    entries=[('grasp',np.asarray(grasp_px,dtype=float))]
    for lm in landmarks:
        if lm['attached'] and lm['xyz'] is not None:
            x,y,_=lm['xyz'];entries.append((lm['label'],np.array([(0.5-y)*640,(1-x)*640])))
    uv=np.asarray([e[1] for e in entries]);lo=np.maximum(np.floor(uv.min(0)-20),[0,0]).astype(int);hi=np.minimum(np.ceil(uv.max(0)+20),[640,576]).astype(int)
    if np.any(hi-lo<8):return raw
    scale=min(640/(hi[0]-lo[0]),640/(hi[1]-lo[1]))
    pic=Image.fromarray(raw[lo[1]:hi[1],lo[0]:hi[0]]).resize(tuple(np.maximum(1,((hi-lo)*scale).astype(int))))
    dr=ImageDraw.Draw(pic)
    for label,pt in entries:
        x,y=(pt-lo)*scale;dr.ellipse((x-5,y-5,x+5,y+5),outline='cyan',width=2)
        dr.text((min(x+7,pic.width-100),max(14,y-14)),label,fill='yellow',stroke_width=1,stroke_fill='black')
    dr.text((3,2),'BEFORE: your selected tool landmarks; crop pixels NOT native',fill='white',stroke_width=1,stroke_fill='black')
    return np.asarray(pic)

def predicted_overlay(cam,landmarks,Fn,Fc):
    pic=Image.fromarray(cam['rgb']);dr=ImageDraw.Draw(pic);C=np.linalg.inv(cam['cam_to_odom']);T=Fn@np.linalg.inv(Fc);K=cam['K']
    for lm in landmarks:
        if not lm['attached'] or lm['xyz'] is None:continue
        pc=C@T@np.r_[lm['xyz'],1]
        if pc[2]<=0:continue
        u,v=(K@pc[:3])[:2]/pc[2]
        if not(0<=u<640 and 0<=v<480):continue
        dr.ellipse((u-6,v-6,u+6,v+6),outline='cyan',width=1)
        dr.text((min(u+8,530),max(15,v-16)),'P:'+lm['label'],fill='cyan',stroke_width=1,stroke_fill='black')
    dr.text((3,2),'CYAN P = rigid PREDICTION, not observed position',fill='cyan',stroke_width=1,stroke_fill='black')
    return np.asarray(pic)

def full_height(z):
    valid=np.isfinite(z);lo,hi=np.nanpercentile(z,[5,95]);hi=max(lo+.05,hi)
    t=np.clip((np.nan_to_num(z,nan=lo)-lo)/(hi-lo),0,1)
    stops=np.array([[25,45,210],[0,210,240],[240,230,30],[230,35,20]],float)
    rgb=np.stack([np.interp(t,[0,1/3,2/3,1],stops[:,i]) for i in range(3)],-1).astype(np.uint8);rgb[~valid]=[170,35,170]
    pic=Image.fromarray(rgb);dr=ImageDraw.Draw(pic)
    dr.text((3,2),f'FULL HEIGHT MAP: blue={lo:.3f}, red={hi:.3f}m (5th-95th percentiles; clipped)',fill='white',stroke_width=1,stroke_fill='black')
    for v in range(32,576,48):
        for u in range(32,640,48):
            if valid[v,u]:dr.text((u,v),f'{z[v,u]:.3f}',fill='white',stroke_width=1,stroke_fill='black')
    return np.asarray(pic)

def fixed_tiles(raw):
    result=[]
    for u0,name in [(0,'LEFT'),(320,'RIGHT')]:
        pic=Image.fromarray(raw[128:448,u0:u0+320]).resize((640,640));dr=ImageDraw.Draw(pic)
        for t in range(0,640,80):
            dr.line((t,0,t,639),fill=(80,150,170));dr.text((t+2,17),str(t),fill='yellow')
            dr.line((0,t,639,t),fill=(80,150,170));dr.text((2,t+2),str(t),fill='yellow')
        dr.text((3,2),name+' DETAIL; 12.8px=1cm; local coordinates',fill='yellow')
        result.append(np.asarray(pic))
    return result

def ask(robot,prompt,images,schema):
    now=robot.proprio()['sim_time_s']
    typical=float(np.median(MODEL_LATENCIES)) if MODEL_LATENCIES else 10.0
    prompt+=f'\nTIME: world elapsed={now:.1f}s, HARD remaining={max(0,90-now):.1f}s. Each model call costs about{typical:.1f}s while physics continues. Aim to finish in20-40s. Batch a complete safe tool-use program (alignment, placement, two cycles, final lift) when geometry is sufficiently known; avoid an inspection after every small translation. Reserve time for final verification. If setup/finish share sequence with cycles, repeat=1 and write two cycles explicitly. Decisions remain yours.'
    r=robot.astra(prompt,images=images,schema=schema,effort='low',tier='fast',instructions=SYS)
    robot.log('astra',r.get('latency_s'),r.get('json') or r.get('failure'))
    MODEL_LATENCIES.append(r.get('latency_s',10.0))
    if not r.get('ok'):raise ValueError('Astra failed')
    return r['json']

def landmark_mean(landmarks,labels):
    if not labels:return None
    d={lm['label']:lm['xyz'] for lm in landmarks if lm['xyz'] is not None}
    if any(k not in d for k in labels):raise ValueError('Unknown landmark label requested')
    return np.mean([d[k] for k in labels],axis=0)

def aligned_rotation(Fn,Fc,landmarks,align):
    p1=landmark_mean(landmarks,align['from_labels']);p2=landmark_mean(landmarks,align['to_labels'])
    if p1 is None or p2 is None:return None
    a=Fn[:3,:3]@Fc[:3,:3].T@(p2-p1)
    b=np.asarray(align['axis_xyz'],dtype=float)
    if np.linalg.norm(a)<1e-8 or np.linalg.norm(b)<1e-8:raise ValueError('Degenerate selected alignment axis')
    a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    axis=np.cross(a,b);sn=np.linalg.norm(axis);cs=np.clip(np.dot(a,b),-1,1)
    if sn<1e-8:
        if cs>0:Q=Fn[:3,:3].copy()
        else:
            basis=np.eye(3)[np.argmin(np.abs(a))];axis=np.cross(a,basis);axis/=np.linalg.norm(axis)
            Q=Rotation.from_rotvec(np.pi*axis).as_matrix()@Fn[:3,:3]
    else:Q=Rotation.from_rotvec(axis/sn*np.arctan2(sn,cs)).as_matrix()@Fn[:3,:3]
    hint=np.asarray(align['approach_hint_xyz'],dtype=float)
    h=hint-b*np.dot(hint,b);v=Q[:,2]-b*np.dot(Q[:,2],b)
    if np.linalg.norm(h)>1e-6 and np.linalg.norm(v)>1e-6:
        h/=np.linalg.norm(h);v/=np.linalg.norm(v)
        angle=np.arctan2(np.dot(b,np.cross(v,h)),np.dot(v,h))
        Q=Rotation.from_rotvec(b*angle).as_matrix()@Q
    return Q

def run(robot,object='object',where='on the counter',use='hold',then='',target=''):
    task=f'Pick up {object.replace("_"," ")} {where}. Use={use}. Then: {then}. Target: {target}.'
    previous='';g=0;history=[]
    mechanics='Capture BOTH handles of tongs, not hinge, so the hand controls jaw closure. open_after_lift_fraction selects a partial opening AFTER secure lift and BEFORE the first inspection: width=measured closed width + fraction*(your measured initial outer-handle span - closed width). Choose0..1 from visible food width versus initial jaw gap, leaving preload for retention; -1 keeps capture unchanged. The purpose is to observe the ACTUAL OPEN jaw geometry before food placement, rather than predict its change during opening. Avoid fraction near1 that removes spring preload; choose enough gap to surround the food with clearance. The compiler executes only your selected fraction and waits for measured width to reach it or plateau before taking the use-planning image (at most4s extra). Compare actual width with requested width; spring-loaded motion may lag commanded speed. Firm force may close EMPTY robust jaws for secure transport; then reopen with spring contact before surrounding food. capture_mode force uses force_n; width uses the projected separation of your outer_contacts_px plus capture_width_offset_m (bounded+-8mm). Outer contacts use SAME grasp_view as center, at the chosen grip station. A barely engaged open grip can slip. probe_forces_n optionally requests up to2 supported force/width probes, each1.2s plus2s final recapture; choose a useful intermediate test if spring resistance is uncertain. Tool capture force is NOT food squeeze force: choose food force sufficient to overcome spring plus gentle jaw pressure, using measured force/width response and lever geometry. Reopen only enough for the FOOD, maintaining preload; original rest handle span is an upper reference, not a mandatory opening target. Infer jaw-gap gain from hinge-to-grip versus hinge-to-tip distances. For initial pickup alignment, choose two CLEARLY VISIBLE solid points along the handle bisector, with from_labels toward the hinge and to_labels toward the jaws. These may be hinge and grip-station midpoint or two visible handle stations; they need NOT be the jaw tips. This axis suffices to rotate the working end downward. If the hand hides the jaws, do not invent their pixels or sample floor/counter behind them: omit hidden jaw landmarks and locate actual jaws after lifting. Prefer a visible native pole or wrist surface when top projection has a hole. Check height-map colors: a tool supported on the counter cannot have an attached point down on the floor. After lifting, identify both visible jaw pads, food and support. Position open jaws on opposite food sides, close gently, lift food>=5cm. In the post-lift image, inspect whether the observed jaw gap already surrounds the food: if sufficient, PRESERVE the current opening during placement instead of adding a fresh wider move. The observed current opening is known to retain this tool; a few extra millimeters can remove spring preload and drop it. Only widen further when the visible gap is actually insufficient, and choose the minimum change consistent with retaining both handles. Do not reopen merely as a ritual setup step. If a secure gap cannot fit the food, decide on a new grip station or stop rather than assume a near-rest opening holds. Hand must NEVER touch food; at finish tool held unsupported and food held in tongs. The jaws ARTICULATE relative to the hand when width changes: initial open-jaw rigid predictions become stale after strong closure, even without slip. Do not invent two separate jaws from predicted labels when the real closed jaws overlap. If closed overlapping jaw pads are visible, select one current CLOSED-JAW CENTER and hinge, align their bisector down, and use this center as the tool-axis reference above the FOOD CENTER before reopening symmetrically. If jaws remain visibly open, measure both pad centers and use their midpoint. Two points at different heights along ONE visible jaw are not two opposing jaws. An edge-on view with overlapping pads cannot measure the opening gap: choose a revealing oblique rotation while preserving grip if necessary, or explicitly use a closed-center reference only when actually closed. A single OPEN jaw needs a side offset; a closed overlapping pair center does not. Compare actual food top and support heights, choose pad contact around the food sides, and distinguish a distal jaw tip from its contact-pad center.'
    for attempt in range(2):
        obs=robot.observe(['pole','wrist'])
        prompt='Choose first grasp and lift. Robot: hand opens80mm; contact pads span27mm along approach, middle91mm beyond flange. Closing axis is your selected direction. Grasp BOTH opposing surfaces at one station, clear supports/ridges with the entire finger bodies. Use adequate engagement and support clearance. force_n .1..70N; fragile food requires gentle force; robust tool capture needs torque-resistant grip.\nGRASP pixels all use grasp_view: full=image3, left=image4, or right=image5. center_px is horizontal midpoint of contacts. top_px is a visible upper grip surface, support_px nearby bare support; height_fraction sets pad z=support+fraction*(top-support), minimum16mm above support. Highest depth within3 full-map pixels refines top. closing_line_px=[u1,v1,u2,v2]; axis_mode closing uses this line, long_axis closes perpendicular to it. tilt_deg0 is downward; 0..65 tilts toward tilt_direction_xy after projection perpendicular to closing axis. open_width_m0..08.\nbase_delta_xy requests a safe base shift (forward bound supplied, lateral<=.35m). Low arm root: far/high or far-SIDEWAYS poses often stall. Use the holonomic base: choose base y near the intended FLANGE y, accounting for the91mm flange-to-pad offset and tool anchor offset. A bowl on the opposite side often needs a substantial lateral base move before pressing. base_delta_xy is a DELTA, so subtract current base from desired base. Do not leave base y at0 by default when working far to one side. At relative pad x=.45m, downward approach reaches roughly z=.75m, tilt25 z=.85m, tilt60 z=.975m. Large backward tilts are difficult at far grasps. lift_delta_xyz: retract up to.30m, rise.06..35m; allow the whole hanging span and rotation clearance. lift_tilt_deg and lift_tilt_direction_xy set final approach, independently of capture. Or choose lift_alignment: measured attached from_labels→to_labels axis (pairs mean midpoint) is rotated onto axis_xyz. approach_hint_xyz selects remaining twist around that axis by desired approach projection; zero=minimal rotation. Choose this twist for VISIBILITY as well as reach: the pole camera looks forward from behind the arm, so a forward(+x) approach often puts the flange directly in front of the hanging head/jaws. For TONGS the inspection must show BOTH jaws separated in the image, not merely the tool beside the flange. Pure sideways approach can make the jaw-opening direction run into camera depth, so the two pads overlap even while open. Choose an oblique twist balancing hand clearance and a jaw-opening direction projected visibly across the pole image, with reachable posture. Do not default to either pure forward or pure sideways. Use the measured camera views to choose the twist. Fixed hand/tool angle cannot change without regrasp/slip. Empty label lists use ordinary lift tilt.\nlandmarks: up to6 labelled points, view pole/wrist native640x480 or top FULL map640x576; attached true only on tool, false on stationary target. Choose visible SOLID surfaces, not holes/background. Attached top points refine to highest surface within3px; native attached points to nearest depth within3px. These are visible surface points, not automatically hidden geometric centers. Invalid depth pauses alignment and uses your ordinary lift. Note<=10words.'
        for perception in range(2):
            cams=obs['cameras'];p=obs['proprio'];im,zmap=overhead(cams);raw,_=overhead(cams,False)
            left,right=fixed_tiles(raw);heights=full_height(zmap)
            for label,pic in [('pole',cams['pole']['rgb']),('wrist',cams['wrist']['rgb']),('top',im),('left',left),('right',right),('height',heights)]:robot.save_image(f'plan{attempt}_{perception}_{label}',pic)
            acquisition_prompt=f'''{task} {previous}
{mechanics}
Images1=pole,2=wrist native640x480,3=FULL metric RGB top640x576,4=LEFT enlarged workspace tile640x640,5=RIGHT tile640x640,6=FULL height map same pixels as image3. All points are calibrated RGB-D, magenta unknown. Full top: x=1-v/640,y=.5-u/640 m; x forward/up,y left/left,z up. Tiles are FIXED workspace crops, not detected objects: left full=(u/2,128+v/2), right full=(320+u/2,128+v/2). Choose grasp_view for ALL grasp pixels including closing line and outer_contacts when applicable. Landmarks view=top ALWAYS uses full image3, never tile pixels. Use visible solid surfaces and height evidence; do not sample a lower background behind a narrow tool.
Current base={p['base_pose_odom']}. The initial bumper gap is about8..16cm; absolute forward base x is capped at.06m.
clear_view=true requests the prevalidated empty-hand camera-clear posture, recording new calibrated views and returning to the original arm pose (about7s), then another plan. Use only if useful grip geometry is occluded; allowed once before first attempt. Otherwise false and this plan executes directly. A magenta boundary is not necessarily a physical handle end.
{prompt}'''
            a=ask(robot,acquisition_prompt,[cams['pole']['rgb'],cams['wrist']['rgb'],im,left,right,heights],PLAN)
            if a['clear_view'] and attempt==0 and perception==0:
                view_start=np.asarray(p['flange_pose_odom']);view_base=np.asarray(p['base_pose_odom'])
                parked=pose([.20,0,.45],[0,0,-1],[0,-1,0]);robot.act_arm([row(parked,0,[0,0,0])[:8]]*60)
                obs=robot.observe(['pole','wrist']);move(robot,view_start,0,view_base,3.,8);obs['proprio']=robot.proprio();continue
            break
        if a['grasp_view']!='full':
            origin=[0 if a['grasp_view']=='left' else 320,128]
            for k in ['center_px','top_px','support_px']:a[k]=(np.asarray(a[k])/2+origin).tolist()
            a['closing_line_px']=(np.asarray(a['closing_line_px'])/2+np.tile(origin,2)).tolist()
            a['outer_contacts_px']=(np.asarray(a['outer_contacts_px'])/2+np.tile(origin,2)).tolist()
            a['grasp_view']='full'
        reference_rgb=cams['pole']['rgb']
        try:top=selected_top(zmap,a['top_px']);support=selected(zmap,a['support_px'])
        except ValueError as e:return {'ok':False,'reason':'Invalid selected grasp depth: '+str(e)}
        landmarks=[]
        for lm in a['landmarks']:
            try:pt=(selected_top(zmap,lm['px']) if lm['attached'] else selected(zmap,lm['px'])) if lm['view']=='top' else point(cams[lm['view']],lm['px'],refine=lm['attached'])
            except ValueError:pt=None
            landmarks.append(dict(label=lm['label'],attached=lm['attached'],xyz=None if pt is None else pt.tolist()))
        reference_rgb=reference_landmarks(raw,landmarks,a['center_px'])
        robot.save_image('selected_tool_reference',reference_rgb)
        G=np.r_[xy(a['center_px']),support[2]+np.clip(a['height_fraction'],0,1.2)*(top[2]-support[2])]
        G[2]=max(G[2],support[2]+.016)
        line=a['closing_line_px'];d=xy(line[2:])-xy(line[:2]);c=np.r_[d,0.];c/=np.linalg.norm(c)
        if a['axis_mode']=='long_axis':c=np.array([-c[1],c[0],0.])
        F0=np.array(p['flange_pose_odom']);current_c=(F0[:3,0]+F0[:3,1])/np.sqrt(2)
        if np.dot(c,current_c)<0:c=-c
        h=np.r_[a['tilt_direction_xy'],0.];h-=c*np.dot(h,c)
        if np.linalg.norm(h)<1e-6:h=np.array([-c[1],c[0],0.])
        h/=np.linalg.norm(h);theta=np.deg2rad(np.clip(a['tilt_deg'],0,65));approach=np.sin(theta)*h+[0,0,-np.cos(theta)]
        F=pose(G,approach,c);H=F.copy();H[:3,3]-=approach*max(.075,(top[2]-G[2]+.04)/max(.4,-approach[2]))
        base=np.array(p['base_pose_odom']);base[:2]+=np.clip(a['base_delta_xy'],[-.08,-.35],[.06,.35]);base[0]=min(.06,base[0])
        robot.log('geometry',dict(top=top.tolist(),support=support.tolist(),pad=G.tolist(),approach=approach.tolist(),closing=c.tolist(),base=base.tolist()))
        robot.gripper_move(float(np.clip(a['open_width_m'],0,.08)),.1);g=0
        move(robot,H,g,base,1.5,12)
        descend(robot,F,g,base)
        contacts=a['outer_contacts_px'];span=abs(np.dot(xy(contacts[2:])-xy(contacts[:2]),c[:2]))
        capture_width=float(np.clip(span+np.clip(a['capture_width_offset_m'],-.008,.008),0,.08))
        robot.log('selected_handle_span',float(span),'capture_width',capture_width,'capture_mode',a['capture_mode'])
        if a['capture_mode']=='width':robot.gripper_move(capture_width,.05);g=0
        else:robot.gripper_grasp(float(np.clip(a['force_n'],.1,70)),.05);g=1
        closed=move(robot,F,g,base,2.0,0)['proprio']
        force_width_probe=[dict(force_n=a['force_n'] if a['capture_mode']=='force' else None,width_m=closed['gripper_width_m'])]
        for probe_force in a['probe_forces_n']:
            robot.gripper_grasp(float(np.clip(probe_force,.1,70)),.05);g=1
            measured=move(robot,F,g,base,1.2,0)['proprio']
            force_width_probe.append(dict(force_n=probe_force,width_m=measured['gripper_width_m']))
        if a['probe_forces_n']:
            if a['capture_mode']=='width':robot.gripper_move(capture_width,.05);g=0
            else:robot.gripper_grasp(float(np.clip(a['force_n'],.1,70)),.05);g=1
            closed=move(robot,F,g,base,2.0,0)['proprio']
        robot.log('force_width_probe',force_width_probe)
        Fc=np.array(closed['flange_pose_odom']);pad_close=Fc[:3,3]+PADS*Fc[:3,2]
        d=np.clip(a['lift_delta_xyz'],[-.30,-.20,.06],[.10,.20,.35])
        theta2=np.deg2rad(np.clip(a['lift_tilt_deg'],-65,65))
        h2=np.r_[a['lift_tilt_direction_xy'],0.]
        if np.linalg.norm(h2)<1e-8:
            if abs(theta2)>1e-8:return {'ok':False,'reason':'Nonzero lift tilt requires a nonzero selected azimuth'}
            h2=np.array([1.,0.,0.]) # Azimuth has no effect when the selected tilt is zero.
        else:h2/=np.linalg.norm(h2)
        a2=np.sin(theta2)*h2+[0,0,-np.cos(theta2)]
        axis=np.cross(Fc[:3,2],a2);sn=np.linalg.norm(axis);cs=np.clip(np.dot(Fc[:3,2],a2),-1,1)
        delta_R=np.eye(3) if sn<1e-8 else Rotation.from_rotvec(axis/sn*np.arctan2(sn,cs)).as_matrix()
        Rl=delta_R@Fc[:3,:3]
        try:tool_R=aligned_rotation(Fc,Fc,landmarks,a['lift_alignment'])
        except ValueError as err:
            tool_R=None;history.append({'initial_alignment_paused':str(err),'reason':'Selected landmark has no valid depth; ordinary model-selected lift retained.'})
        if tool_R is not None:Rl=tool_R
        L=pose(pad_close+d,Rl[:,2],(Rl[:,0]+Rl[:,1])/np.sqrt(2))
        lift_result=transport(robot,L,g,base)
        post_lift_width=None
        if a['open_after_lift_fraction']>=0:
            held_width=robot.proprio()['gripper_width_m']
            post_lift_width=float(np.clip(held_width+np.clip(a['open_after_lift_fraction'],0,1)*(span-held_width),0,.08))
            robot.gripper_move(post_lift_width,.025);g=0
            hold_F=np.asarray(robot.proprio()['flange_pose_odom'])
            hold_seconds=max(1.0,abs(post_lift_width-held_width)/.025+.3)
            opened=move(robot,hold_F,g,base,hold_seconds,0)
            previous_width=opened['proprio']['gripper_width_m'];stable=0
            for _ in range(10):
                if abs(previous_width-post_lift_width)<.001:break
                opened=robot.act([row(hold_F,g,base)]*6)
                width=opened['proprio']['gripper_width_m']
                stable=stable+1 if abs(width-previous_width)<.0003 else 0
                previous_width=width
                if stable>=3:break
            robot.log('opening_motion_completed_or_plateau',dict(requested_width_m=post_lift_width,actual_width_m=previous_width))
            robot.log('selected_post_lift_opening',a['open_after_lift_fraction'],post_lift_width)
        for check in range(5):
            now=robot.observe(['pole','wrist']);pn=now['proprio'];Fn=np.array(pn['flange_pose_odom']);pad=Fn[:3,3]+PADS*Fn[:3,2]
            topim,zn=overhead(now['cameras']);detail=hand_crop(now['cameras']['pole'],pad)
            stationary=[lm['xyz'] for lm in landmarks if not lm['attached'] and lm['xyz'] is not None]
            center=np.mean(stationary,axis=0) if stationary else pad
            target_u0=int(np.clip(round((.5-center[1])*640)-64,0,512));target_v0=int(np.clip(round((1-center[0])*640)-64,0,448))
            target_height=depth_detail(zn,target_u0,target_v0)
            robot.save_image(f'check{attempt}_{check}_target_height',target_height)
            robot.save_image(f'check{attempt}_{check}_hand_detail',detail)
            for name,cam in now['cameras'].items():robot.save_image(f'check{attempt}_{check}_{name}',cam['rgb'])
            robot.save_image(f'check{attempt}_{check}_top',topim)
            measurements={'selected_post_lift_width_m':post_lift_width,'lift_requested_pad_xyz':(L[:3,3]+PADS*L[:3,2]).round(4).tolist(),'lift_tracking_error':lift_result['steps'][-1],'force_width_probe_supported':force_width_probe,'pad_xyz':pad.round(4).tolist(),'approach_xyz':Fn[:3,2].round(4).tolist(),'closing_xyz':((Fn[:3,0]+Fn[:3,1])/np.sqrt(2)).round(4).tolist(),'capture_width_m':closed['gripper_width_m'],'initial_outer_handle_span_m':float(span),'selected_capture_width_m':capture_width,'width_m':pn['gripper_width_m'],'last_command':pn.get('gripper_command'),'pad_delta_since_close':(pad-pad_close).round(4).tolist(),'grasp_surface_z':float(top[2]),'support_z':float(support[2]),'base_pose_odom':pn['base_pose_odom'],'landmarks_initial':landmarks,'landmarks_predicted_rigid':[dict(label=lm['label'],xyz=(Fn@np.linalg.inv(Fc)@np.r_[lm['xyz'],1])[:3].round(4).tolist()) for lm in landmarks if lm['attached'] and lm['xyz'] is not None]}
            annotated_pole=predicted_overlay(now['cameras']['pole'],landmarks,Fn,Fc)
            robot.save_image(f'check{attempt}_{check}_predictions',annotated_pole)
            b=ask(robot,f'''{task}
{mechanics}
Images1=BEFORE selected-tool crop (not native pixels). 2=NOW pole with cyan P=rigid PREDICTIONS, not detections;3=raw wrist;4=full metric top;5=hand-detail crop. Compare predicted and visible tool locations, repair if mismatched. Image6=HEIGHT DETAIL centered on your selected stationary target points, same color/height legend as initial height map. Choose fresh_landmarks view=target for local image6 pixels0..640; code maps them to full-map pixels [{target_u0},{target_v0}]+px/5 before depth measurement. This reveals target shape, rim and bottom heights; choose actual food/interior surfaces, not hidden depth holes. Current-pixel selections use images2/3/4 or target=image6. Full top x=1-v/640,y=.5-u/640 m.
Measurements:{json.dumps(measurements)}
Actions/execution so far:{json.dumps(history)}
Choose action: done only when actual task success is confirmed; retry for failed capture; stop if unsafe; sequence to act. Occlusion is uncertainty, not proof of a drop. Nonzero width alone is not proof of holding. Preserve grip while revealing if needed; compare reference identity. Do not mistake predicted circles for observed tool features. Consider reported actual pose and tracking residual; repeating an unreachable pose wastes time.
GEOMETRY: fresh_landmarks selects points in CURRENT pole/wrist native640x480 or FULL top map; attached tool points replace rigid offsets, stationary points locate target. Same3px local edge refinement as initial. Correct stale geometry after slip. alignment rotates attached from_labels→to_labels onto axis_xyz; approach_hint_xyz chooses twist as before; a sideways approach may reveal the tool beside the flange to the rear pole camera, whereas a forward approach often occludes it. Empty label lists disable alignment. With alignment active, frame approach/closing vectors are ignored.
PROGRAM: frame contains approach_xyz (flange toward pads), closing_xyz, anchor_labels, destination_labels. points are [x,y,z,command,value,seconds], seconds .3..3, command0 keep,1 move(width m),2 grasp(force N). All points share frame. Empty anchor means PAD target; attached anchor label(s) mean their point/midpoint is the target. Empty destination means absolute odom xyz; stationary destination label(s) mean xyz is OFFSET from their mean. Thus choose alignment, anchor and target to place tool geometry without guessing hand transforms. Articulated jaws require state-aware anchors: a CLOSED overlapping pair center or midpoint of both OPEN jaws can target food center; one OPEN jaw needs a side offset. Opening changes individual jaw points relative to the hand, so their rigid P circles are not current observations. Gripper move can push70N when blocked; fragile food requires force control. base_delta_xy shifts base safely before/during sequence. repeat repeats ALL points, so use1 when setup/lift surround explicitly written two cycles. Before use, align the base laterally with the working FLANGE position: choose base_delta_y=desired_base_y-current_base_y from the measured target/tool offsets. This matters especially when moving from tool side to bowl/food side; holding base y fixed can make otherwise easy poses unreachable. Use only enough tool clearance above the target to avoid the rim/food, not an unnecessarily high overhead waypoint. Batch known-safe placement/use/lift to save latency. No hidden contact or force sensor is available.
For done select finish_width_range_m as a plausible fresh-width condition excluding empty width if appropriate; code checks it after response latency. Other actions use[0,.08]. Nonsequence: points=[],fresh_landmarks=[],alignment with empty label lists,base_delta_xy=[0,0],repeat1,neutral frame. ok true only for verified success. Note<=15words.''',[reference_rgb,annotated_pole,now['cameras']['wrist']['rgb'],topim,detail,target_height],CHECK)
            history.append(b)
            if b['action']=='stop':return {'ok':b['ok'],'reason':b['note']}
            if b['action']=='done':
                latest=robot.proprio();w=latest['gripper_width_m'];lo,hi=b['finish_width_range_m']
                if lo<=w<=hi:return {'ok':b['ok'],'reason':b['note']}
                history.append({'completion_condition_not_met':{'latest_width_m':w,'model_requested_range':[lo,hi]}})
                continue
            if b['action']=='retry':
                previous='Previous attempt: '+b['note'];break
            measurement_errors=[]
            for lm in b['fresh_landmarks']:
                if lm['view']=='target':
                    lm=dict(lm,view='top',px=(np.asarray(lm['px'])/5+[target_u0,target_v0]).tolist())
                try:pt=(selected_top(zn,lm['px']) if lm['attached'] else selected(zn,lm['px'])) if lm['view']=='top' else point(now['cameras'][lm['view']],lm['px'],refine=lm['attached'])
                except ValueError:
                    measurement_errors.append(lm['label']);continue
                if lm['attached']:pt=(Fc@np.linalg.inv(Fn)@np.r_[pt,1])[:3]
                landmarks=[old for old in landmarks if old['label']!=lm['label']]
                landmarks.append(dict(label=lm['label'],attached=lm['attached'],xyz=pt.tolist()))
            if measurement_errors:
                history.append({'sequence_paused_invalid_depth':measurement_errors});continue
            try:
                for labels in [b['alignment']['from_labels'],b['alignment']['to_labels'],b['frame']['anchor_labels'],b['frame']['destination_labels']]:
                    landmark_mean(landmarks,labels)
            except ValueError as err:
                history.append({'sequence_paused_invalid_landmark':str(err)});continue
            base[:2]+=np.clip(b['base_delta_xy'],[-.15,-.4],[.06,.4]);base[0]=min(.06,base[0]);base[1]=np.clip(base[1],-.6,.6)
            executed=[]
            target_R=aligned_rotation(Fn,Fc,landmarks,b['alignment'])
            for wp in expand_points(b)*int(np.clip(b['repeat'],1,3)):
                goal=np.asarray(wp['pad_xyz'],dtype=float)
                destination=landmark_mean(landmarks,wp['destination_labels'])
                if destination is not None:goal=goal+destination
                if target_R is not None:Q=pose(goal,target_R[:,2],(target_R[:,0]+target_R[:,1])/np.sqrt(2))
                else:Q=pose(goal,wp['approach_xyz'],wp['closing_xyz'])
                anchor=landmark_mean(landmarks,wp['anchor_labels'])
                if anchor is not None:Q[:3,3]-=Q[:3,:3]@Fc[:3,:3].T@(anchor-pad_close)
                if np.linalg.norm(Q[:3,3]-Fn[:3,3])>.65 or not .25<Q[2,3]<1.15:return {'ok':False,'reason':'Waypoint outside safety bounds'}
                if wp['gripper_kind']=='move':robot.gripper_move(float(np.clip(wp['gripper_value'],0,.08)),.05);g=0
                elif wp['gripper_kind']=='grasp':robot.gripper_grasp(float(np.clip(wp['gripper_value'],.1,70)),.05);g=1
                rr=move(robot,Q,g,base,float(np.clip(wp['seconds'],.3,3)),3)
                executed.append({'width_m':rr['proprio']['gripper_width_m'],'tracking_error':rr['steps'][-1]})
            history.append({'executed_measurements':executed})
        else:return {'ok':False,'reason':'Action iteration limit'}
    return {'ok':False,'reason':'Attempt limit'}
