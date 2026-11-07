"""Astra selects all scene-dependent actions; this harness measures and executes them."""
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw
PADS = 0.130  # PORT: the real pads' middle out from the flange (sim 0.091): the fingertips 0.148 by touch (calibration/tcp.py, 2026-11-06), the pads' middle 18.3 mm above them
MOVE_CAP = 300  # PORT: control steps (20 s) a move may take while still progressing

SYS='You are Astra, the visual decision policy controlling a physical kitchen robot. Use calibrated images and measurements to choose accurate gentle grasps and tool actions. Return concise JSON. No tools are available.'
def obj(props): return dict(type='object',additionalProperties=False,required=list(props),properties=props)
def vec(n): return dict(type='array',items={'type':'number'},minItems=n,maxItems=n)
def num(): return {'type':'number'}
def string(): return {'type':'string'}
PLAN=obj({'grasp_view':{'type':'string','enum':['detail','full']},'center_px':vec(2),'top_px':vec(2),'support_px':vec(2),'height_fraction':num(),
          'closing_line_px':vec(4),'axis_mode':{'type':'string','enum':['closing','long_axis']},
          'tilt_deg':num(),'lift_tilt_deg':num(),'lift_tilt_direction_xy':vec(2),'tilt_direction_xy':vec(2),'force_n':num(),'open_width_m':num(),
          'base_delta_xy':vec(2),'lift_delta_xyz':vec(3),'landmarks':{'type':'array','maxItems':6,'items':obj({'label':string(),'view':{'type':'string','enum':['pole','wrist','top']},'px':vec(2),'attached':{'type':'boolean'}})},'note':string()})
SCOUT=obj({'center_px':vec(2),'clear_view':{'type':'boolean'},'note':string()})
REFINE=obj({k:PLAN['properties'][k] for k in ['center_px','top_px','support_px','height_fraction','closing_line_px','axis_mode','tilt_deg','tilt_direction_xy','force_n','open_width_m','note']})
WAY=obj({'pad_xyz':vec(3),'approach_xyz':vec(3),'closing_xyz':vec(3),
         'gripper_kind':{'type':'string','enum':['keep','move','grasp']},'gripper_value':num(),'seconds':num()})
CHECK=obj({'action':{'type':'string','enum':['done','retry','sequence','stop']},'ok':{'type':'boolean'},'finish_width_range_m':vec(2),
           'base_delta_xy':vec(2),'repeat':{'type':'integer','minimum':1,'maximum':3},'waypoints':{'type':'array','items':WAY,'maxItems':10},'note':string()})

def point(cam,px):
    h,w=cam['depth'].shape;u,v=np.clip(np.rint(px),[0,0],[w-1,h-1]).astype(int)
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

def ask(robot,prompt,images,schema):
    now=robot.proprio()['sim_time_s']
    typical=float(np.median(MODEL_LATENCIES)) if MODEL_LATENCIES else 10.0
    prompt+=f'\nTIME: world elapsed={now:.1f}s, HARD remaining={max(0,90-now):.1f}s. Each model call costs about{typical:.1f}s while physics continues. Aim to finish in20-40s. Batch a complete safe tool-use program (alignment, placement, two cycles, final lift) when geometry is sufficiently known; avoid an inspection after every small translation. Reserve time for final verification. If setup/finish share sequence with cycles, repeat=1 and write two cycles explicitly. Decisions remain yours.'
    r=robot.astra(prompt,images=images,schema=schema,effort='low',tier='fast',instructions=SYS)
    robot.log('astra',r.get('latency_s'),r.get('json') or r.get('failure'))
    MODEL_LATENCIES.append(r.get('latency_s',10.0))
    if not r.get('ok'):raise ValueError('Astra failed')
    return r['json']

def run(robot,object='object',where='on the counter',use='hold',then='',target=''):
    task=f'Pick up {object.replace("_"," ")} {where}. Use={use}. Then: {then}. Target: {target}.'
    previous='';g=0;history=[]
    mechanics={
      'hold':'',
      'squeeze':'Identify hinge, two handle ends, and raised rest ridge. Grasp BOTH handles at a station whose entire finger bodies clear the ridge. Consider BOTH sides of the ridge, including narrower handle portions near hinge when free-end spread is too large. Avoid the hinge joint itself. A station only1-3cm from ridge can still collide because fingers have bulk. Do not automatically prefer the free ends. Do not collide with ridge along tilted descent; vertical often clears it. Initial force must resist tool weight AND twisting moment while handles remain mostly open; do not use a fragile-food grip for a robust spring tool. After lifting, a long tool grasped near one end can rotate and hang down: use measured endpoint span to choose enough lift AND retraction so its LOWEST end clears counter. A small10cm lift can leave20cm-long hanging handles on the counter. Retract substantially toward robot for high-lift reach, rather than lifting far forward. Before squeeze cycles, verify the entire tool is free of rest AND counter. If it sags during opening, lift/retract while retaining it before further cycles. After lift use position widths to squeeze/reopen twice, keeping contact with outward-springing handles. Width ratio at same grasp station approximates handle closing fraction; return to initial spread between pulses. Select needed forces/widths from actual geometry, no assumed tool size.',
      'press':'Identify sliding handle and wire head. Grasp the HANDLE, away from rod/head and rest ridge. Tool must be turned upright with head DOWN before approaching bowl. Select attached landmark at head center and handle end; fixed landmark at bowl interior center/bottom. Use rigid geometry to calculate hand pose for head alignment. Move base laterally to align with bowl rather than overreach sideways. Press DOWN on handle while head rests inside bowl; retract enough for spring reset, twice, then lift clear.',
      'tongs':'Identify spring/hinge, squeeze handles and food-contact jaws. Grasp BOTH handles so hand closure operates jaws, not the hinge itself. Need contact on opposing outer handles; open hand initially must enclose both. Select attached landmarks at both jaw tips and hinge; fixed landmark at target food top and support. Orient tips DOWN around target food, with hand well above food; choose jaw opening and gentle squeezing force from geometry. Move base laterally to align with target. Lift food IN jaws without any hand contact.'
    }.get(use,'')
    for attempt in range(2):
        obs=robot.observe(['pole','wrist']);cams=obs['cameras'];p=obs['proprio'];im,zmap=overhead(cams)
        for name,c in cams.items():robot.save_image(f'initial{attempt}_{name}',c['rgb'])
        robot.save_image(f'top{attempt}',im)
        prompt=f'''{task} {previous} {mechanics}
Image1=pole RGB;2=wrist RGB;3=metric top view from calibrated RGB-D,640x576. ALL requested pixels in image3. MAGENTA regions have NO depth data; never choose height pixels there. 64 pixels=10cm. World odom axes x forward(up image),y left(left image),z up. Pixel(u,v) maps to x=1-v/640,y=0.5-u/640 meters. Current base={p['base_pose_odom']}.
Choose first grasp. Parallel hand opens80mm, pads 27mm long along approach, soft leaves/lips. Pad middle is91mm past flange. Fingers close along chosen line. Never push a delicate object with position-controlled closure: choose low grasp FORCE just sufficient for friction. force_n=.1..70. Choose force for WEIGHT AND TORQUE, not just grip-feature width. A large or long robust object held by a small off-center rim/handle can need tens of newtons to resist twisting/slip; a thin grip feature does not imply a fragile object. Fragile or soft items still need low force. open_width_m usually.08.
center_px=horizontal CENTER between intended opposed contacts;top_px=measured visible upper surface at grasp region (harness refines to highest depth surface within3 full-map pixels,4.7mm, to tolerate narrow-edge pixel error);support_px=nearby bare counter or support surface. height_fraction chooses contact center between support0 and top1. Finger extends14mm below pad center, clear support. Avoid grasping above object or on unstable tapered tips. Choose accessible narrow part with substantial opening clearance: avoid using nearly all80mm. Prefer outer span<=60mm where shape permits, leaving room for protruding compliant leaves/lips and pixel/pose error. A narrow protrusion or an edge pinch may be safer than forcing jaws over a wide body. Finger bodies extend beyond the selected point: clear raised supports, adjacent rims and rest ridges by several centimeters along the whole approach, not merely at the pad center.
closing_line_px=[u1,v1,u2,v2] either jaw closing axis(axis_mode=closing), or object's longitudinal axis(axis_mode=long_axis) with jaws perpendicular in horizontal world plane. tilt_deg=0 vertical down or0..65 angled; tilt_direction_xy specifies forward horizontal direction of approach (e.g.[1,0]); this is projected perpendicular to closing axis. Tilting toward the target helps high counter reach. Arm base is low,0.335m above floor; far/high overhead poses can be unreachable. base_delta_xy in meters chooses safe base repositioning; initial forward bumper gap8..16cm, so at most0.06m forward without more evidence. A small lateral move can align to object. lift_tilt_deg and lift_tilt_direction_xy choose the FINAL approach direction INDEPENDENTLY of initial closing axis. Code rotates the entire held hand by the minimal rotation from current approach to this chosen direction while translating the PAD along lift_delta_xyz. The closing axis tilts with the hand. Choose forward tilt25..60 plus retraction when high lift is needed; vertical-down pads have poor high reach. At relative pad x=.45m, vertical-down max pad z is about.75m versus.85 at tilt25 and.975 at tilt60 (robot geometry, approximate). Preserve orientation if adequate. lift_delta_xyz in meters chooses enough upward/retraction movement for object to end>=5cm higher, allowing slip/rotation. If grasp is far from object center, the farthest part can drop by almost the FULL distance from grasp to farthest extremity when hanging. Choose lift from that visible span plus clearance, not just center displacement. Maximum allowed lift50cm/retraction35cm, final lift tilt up to100degrees; large lifts need retraction and substantial forward hand tilt for reach. landmarks: for tool use choose up to6 significant pixels in indicated view for calibrated3D measurement; attached=true only for points on picked tool, false for target/support. For hold use[]. Note<=12words. Astra chooses all these from this scene.'''
        scout=ask(robot,f"{task} Images1=pole,2=wrist,3=metric overhead640x576,64pixels=10cm. Choose one pixel in image3 at the intended grasp FEATURE for a closer look. Identify actual target, not distractor. For a tool, choose its usable grip region clear of rest ridge; account for later use. Do not guess hidden grip features. clear_view=true requests the prevalidated empty-hand camera-clearance posture (flange x=.20,z=.541 in root, hand down), taking about7s including return to original arm pose after recording the clear view; choose it when arm occludes a useful grip feature. False if sufficient view already. This option is allowed only before first attempt. Note<=8words.",[cams['pole']['rgb'],cams['wrist']['rgb'],im],SCOUT)
        cleared=bool(scout['clear_view']) and attempt==0
        if cleared:
            view_start=np.array(p['flange_pose_odom']);view_base=np.array(p['base_pose_odom'])
            parked=pose([.20,0,.45],[0,0,-1],[0,-1,0]);robot.act_arm([row(parked,0,[0,0,0])[:8]]*60)
            obs=robot.observe(['pole','wrist']);cams=obs['cameras'];p=obs['proprio'];im,zmap=overhead(cams)
            robot.save_image('clear_pole',cams['pole']['rgb']);robot.save_image('clear_top',im)
            move(robot,view_start,0,view_base,3.,8)
            p=robot.proprio()
        reference_rgb=cams['pole']['rgb'] if cleared else cams['wrist']['rgb']
        raw,_=overhead(cams,False)
        u0=int(np.clip(round(scout['center_px'][0])-64,0,512));v0=int(np.clip(round(scout['center_px'][1])-64,0,448))
        pic=Image.fromarray(raw[v0:v0+128,u0:u0+128]).resize((640,640));draw=ImageDraw.Draw(pic)
        for t in range(0,640,80):
            draw.line((t,0,t,639),fill=(80,150,170));draw.text((t+2,16),str(t),fill='yellow')
            draw.line((0,t,639,t),fill=(80,150,170));draw.text((2,t+2),str(t),fill='yellow')
        draw.text((4,2),'GRASP DETAIL640x640. 32pixels=1cm. Magenta=unknown.',fill='yellow')
        zoom=np.asarray(pic);robot.save_image(f'grasp_detail{attempt}',zoom)
        heights=depth_detail(zmap,u0,v0);robot.save_image(f'height_detail{attempt}',heights)
        fine_prompt=f"""{task} {previous} {mechanics}
Images1=pole,2=wrist,3=MAGNIFIED GRASP DETAIL640x640,4=full metric overhead640x576,5=HEIGHT MAP of the same detail crop as image3 (same pixels). Image5 shows measured world z: its legend and numeric samples distinguish raised grip surfaces from lower supports. Use height evidence to avoid sampling behind a thin handle or choosing a pad level that hits neighboring ridges. Choose grasp_view=detail for ALL GRASP pixels (center,top,support,closing_line) in IMAGE3, or grasp_view=full for all of them in IMAGE4 if a better grip feature lies outside the crop. Full map remains64pixels=10cm with world x=1-v/640,y=.5-u/640. This can happen after clearing the camera view. Detail map: It has32pixels=1cm; magenta=NO depth data. Image3 pixel(u,v) maps world x=1-({v0}+v/5)/640,y=.5-({u0}+u/5)/640 meters. +x forward(up),+y left(left),+z up. Current base={p['base_pose_odom']}.
Only landmarks with view=top use full metric image4 coordinates, not crop. Landmark pole/wrist coordinates remain their native640x480.
Choose precise upper-surface pixels on the intended GRIP FEATURE, not a lower surface behind it. Thin handles need pixel accuracy. Center between opposing contact sites. You may choose a different grasp station within this detail. Select higher contact on tapering/rounded bulky bodies if lower width exceeds opening; never force fingers over wide food.
"""+prompt[prompt.index('Choose first grasp.'):]
        a=ask(robot,fine_prompt,[cams['pole']['rgb'],cams['wrist']['rgb'],zoom,im,heights],PLAN)
        if a['grasp_view']=='detail':
            for k in ['center_px','top_px','support_px']:a[k]=(np.asarray(a[k])/5+[u0,v0]).tolist()
            a['closing_line_px']=(np.asarray(a['closing_line_px'])/5+[u0,v0,u0,v0]).tolist()
        for repair in range(2):
            try:
                top=selected_top(zmap,a['top_px']);support=selected(zmap,a['support_px']);break
            except ValueError as e:
                if repair:return {'ok':False,'reason':str(e)}
                a=ask(robot,'For this repair, grasp_view=full and all coordinates refer to full metric image3. Your chosen top/support pixel is in a depth hole. Choose different visibly COLORED valid pixels on corresponding surfaces; magenta missing areas cannot be measured. '+prompt,[cams['pole']['rgb'],cams['wrist']['rgb'],im],PLAN)
        landmarks=[]
        for lm in a['landmarks']:
            try:pt=selected(zmap,lm['px']) if lm['view']=='top' else point(cams[lm['view']],lm['px'])
            except ValueError:pt=None
            landmarks.append(dict(label=lm['label'],attached=lm['attached'],xyz=None if pt is None else pt.tolist()))
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
        robot.gripper_grasp(float(np.clip(a['force_n'],.1,70)),.05);g=1
        closed=move(robot,F,g,base,1.0,0)['proprio']
        Fc=np.array(closed['flange_pose_odom']);pad_close=Fc[:3,3]+PADS*Fc[:3,2]
        d=np.clip(a['lift_delta_xyz'],[-.35,-.20,.06],[.10,.20,.50])
        h2=np.r_[a['lift_tilt_direction_xy'],0.];h2/=np.linalg.norm(h2)
        theta2=np.deg2rad(np.clip(a['lift_tilt_deg'],-65,100));a2=np.sin(theta2)*h2+[0,0,-np.cos(theta2)]
        axis=np.cross(Fc[:3,2],a2);sn=np.linalg.norm(axis);cs=np.clip(np.dot(Fc[:3,2],a2),-1,1)
        delta_R=np.eye(3) if sn<1e-8 else Rotation.from_rotvec(axis/sn*np.arctan2(sn,cs)).as_matrix()
        Rl=delta_R@Fc[:3,:3];L=pose(pad_close+d,Rl[:,2],(Rl[:,0]+Rl[:,1])/np.sqrt(2))
        transport(robot,L,g,base)
        for check in range(5):
            now=robot.observe(['pole','wrist']);pn=now['proprio'];Fn=np.array(pn['flange_pose_odom']);pad=Fn[:3,3]+PADS*Fn[:3,2]
            topim,zn=overhead(now['cameras']);detail=hand_crop(now['cameras']['pole'],pad)
            robot.save_image(f'check{attempt}_{check}_hand_detail',detail)
            for name,cam in now['cameras'].items():robot.save_image(f'check{attempt}_{check}_{name}',cam['rgb'])
            robot.save_image(f'check{attempt}_{check}_top',topim)
            measurements={'pad_xyz':pad.round(4).tolist(),'approach_xyz':Fn[:3,2].round(4).tolist(),'closing_xyz':((Fn[:3,0]+Fn[:3,1])/np.sqrt(2)).round(4).tolist(),'width_m':pn['gripper_width_m'],'last_command':pn.get('gripper_command'),'pad_delta_since_close':(pad-pad_close).round(4).tolist(),'grasp_surface_z':float(top[2]),'support_z':float(support[2]),'base_pose_odom':pn['base_pose_odom'],'landmarks_initial':landmarks,'landmarks_predicted_rigid':[dict(label=lm['label'],xyz=(Fn@np.linalg.inv(Fc)@np.r_[lm['xyz'],1])[:3].round(4).tolist()) for lm in landmarks if lm['attached'] and lm['xyz'] is not None]}
            b=ask(robot,f'''{task} {mechanics}
Verify and choose next action. Images:1=BEFORE reference showing target,2=NOW pole,3=NOW wrist,4=NOW metric top view;5=zoom around measured hand. Inspect partial visible identity/colour carefully; no need to move merely for a larger image. Metric map (64px=10cm;world x=1-v/640,y=.5-u/640). Measurements(meters,odom):{json.dumps(measurements)}. Initial grasp plan:{json.dumps(a)}. Actions already taken:{json.dumps(history)}.
Success: named object held touching fingers and no support, unbroken, >=5cm above starting position. Pad displacement alone does not prove object displacement. Check the LOWEST part and farthest extremity: a long object can hang vertically with its bottom still touching the counter even when the handle is high. Require clear air under the entire object, conservatively at least5cm above original support, and fingers as its only support. If the lower edge is hidden or near the counter, choose a further reachable lift/retraction before claiming success. Compare initial target appearance carefully; abrupt colour change can indicate breakage. The hand often occludes the held object; occlusion alone is NOT evidence of a failed grasp. If gripper width is nonzero and original object has disappeared from counter, first reveal by rotating/retracting while retaining grip, not retry/open. Do not mistake a differently coloured/shaped distractor for target. For hold finish when visually confirmed. For tongs, hand must never touch food; use held tongs to lift named food>=5cm. For squeeze, while held off rest: open <=25% closed, squeeze >=55% closed TWICE with reopening<=25% between; finish<=40% closed. For press, orient whisk head down into bowl, push sliding handle >=70% stroke TWICE, spring back<=30% between; keep bowl upright; finish whisk held up.
Choose done if ALL requirements met, retry for failed grasp, stop if unsafe, sequence to move held object or operate it. sequence waypoints: absolute PAD xyz in odom(m), approach_xyz=unit direction from flange toward fingertips, closing_xyz=unit jaw closing direction perpendicular to approach. Preserve current vectors for translation; rotate them to reorient a held tool. Gripper keep, grasp(value=force N .1..70), or move(value=width m0..08). move holds up to70N if blocked; use force control for fragile food. Hand commands take time (up to1.6s across full opening). seconds(.3..3) is hold/motion duration at each waypoint. Coordinates must come from these images and measurements. base_delta_xy chooses safe whole-body shift in odom before/during sequence (maxforward x=.06 absolute); align base y with working area to avoid lateral overreach. repeat=1 normally, or2 for two identical squeeze/release or press/reset cycles. For done/retry/stop base_delta_xy=[0,0],repeat=1. Predicted landmark coordinates assume rigid grasp and are NOT evidence against slip. You may provide a deliberate short sequence of open/close or press/retract waypoints then inspect. Avoid dropping the tool while opening; spring tools can follow the pads. General motion: retract(-x) while lifting if far overhead unreachable. Note<=20words. ok only true for confirmed success. finish_width_range_m=[min,max] is YOUR condition for retaining completion after your response latency: choose a tight plausible interval around observed held-object width, excluding empty width if appropriate. The harness reads fresh width after a done response; if outside your interval it reobserves for you instead of finishing. For other actions use[0,.08]. Do not infer current retention just from an earlier image; world continues during reasoning.''',[reference_rgb,now['cameras']['pole']['rgb'],now['cameras']['wrist']['rgb'],topim,detail],CHECK)
            history.append(b)
            if b['action']=='stop':return {'ok':b['ok'],'reason':b['note']}
            if b['action']=='done':
                latest=robot.proprio();w=latest['gripper_width_m'];lo,hi=b['finish_width_range_m']
                if lo<=w<=hi:return {'ok':b['ok'],'reason':b['note']}
                history.append({'completion_condition_not_met':{'latest_width_m':w,'model_requested_range':[lo,hi]}})
                continue
            if b['action']=='retry':
                previous='Previous attempt: '+b['note'];break
            base[:2]+=np.clip(b['base_delta_xy'],[-.15,-.4],[.06,.4]);base[0]=min(.06,base[0]);base[1]=np.clip(base[1],-.6,.6)
            executed=[]
            for wp in b['waypoints']*int(np.clip(b['repeat'],1,3)):
                Q=pose(wp['pad_xyz'],wp['approach_xyz'],wp['closing_xyz'])
                if np.linalg.norm(Q[:3,3]-Fn[:3,3])>.65 or not .25<Q[2,3]<1.15:return {'ok':False,'reason':'Waypoint outside safety bounds'}
                if wp['gripper_kind']=='move':robot.gripper_move(float(np.clip(wp['gripper_value'],0,.08)),.05);g=0
                elif wp['gripper_kind']=='grasp':robot.gripper_grasp(float(np.clip(wp['gripper_value'],.1,70)),.05);g=1
                rr=move(robot,Q,g,base,float(np.clip(wp['seconds'],.3,3)),3)
                executed.append({'width_m':rr['proprio']['gripper_width_m'],'tracking_error':rr['steps'][-1]})
            history.append({'executed_measurements':executed})
        else:return {'ok':False,'reason':'Action iteration limit'}
    return {'ok':False,'reason':'Attempt limit'}
