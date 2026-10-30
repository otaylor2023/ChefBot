"""Astra chooses pixels, grasp geometry, and termination; calibrated geometry executes it."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw

SYS = 'You are the visual decision-making policy for a real robot. Choose safe accurate grasps from calibrated images. Be concise. Return only the requested JSON.'

def obj_schema(props):
    return dict(type='object', additionalProperties=False, required=list(props), properties=props)
def vec(n, typ='number'):
    return dict(type='array',items={'type':typ},minItems=n,maxItems=n)
PLAN = obj_schema({
    'center_px':vec(2), 'top_px':vec(2), 'support_px':vec(2),
    'height_fraction':{'type':'number'}, 'approach_tilt_deg':{'type':'number','minimum':0,'maximum':45},
    'axis_mode':{'type':'string','enum':['closing','long_axis']},
    'closing_line_px':vec(4), 'lift_cm':{'type':'number'}, 'note':{'type':'string'}})

CHECK = obj_schema({'action':{'type':'string','enum':['done','lift_more','retry','stop']},'ok':{'type':'boolean'},'delta_cm':vec(3),'note':{'type':'string'}})

def root_T(p):
    x,y,a=p['base_pose_odom']; c,s=np.cos(a),np.sin(a)
    T=np.eye(4);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=[x,y];return T

def point(cam,px,T):
    u,v=np.clip(np.rint(px),[0,0],[639,359]).astype(int)
    z=float(cam['depth'][v,u])
    if not np.isfinite(z) or z<=0: raise ValueError('invalid selected depth')
    K=cam['K'];return (np.linalg.inv(T)@cam['cam_to_odom']@np.array([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z,1]))[:3]

def ray_plane(cam,px,T,z):
    C=np.linalg.inv(T)@cam['cam_to_odom']; K=cam['K'];u,v=px
    d=C[:3,:3]@np.array([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],1.])
    if abs(d[2])<0.01: raise ValueError('horizontal selection ray')
    return C[:3,3]+d*((z-C[2,3])/d[2])

def overhead(cams,T):
    # Pure calibrated point-cloud rasterization. No object masks or image recognition.
    xyzs=[];colors=[]
    for cam in cams.values():
        d=cam['depth'];vv,uu=np.indices(d.shape);K=cam['K']
        xyz=np.stack([(uu-K[0,2])*d/K[0,0],(vv-K[1,2])*d/K[1,1],d],axis=-1).reshape(-1,3)
        C=np.linalg.inv(T)@cam['cam_to_odom'];xyz=xyz@C[:3,:3].T+C[:3,3]
        good=(d.ravel()>0)&np.all(np.isfinite(xyz),axis=1)
        good&=(xyz[:,0]>=0.10)&(xyz[:,0]<0.75)&(xyz[:,1]>-0.40)&(xyz[:,1]<=0.40)&(xyz[:,2]>-0.20)&(xyz[:,2]<0.65)
        xyzs.append(xyz[good]);colors.append(cam['rgb'].reshape(-1,3)[good])
    xyz=np.concatenate(xyzs);rgb=np.concatenate(colors)
    u=np.clip(np.rint((0.40-xyz[:,1])/0.00125),0,639).astype(int)
    v=np.clip(np.rint((0.75-xyz[:,0])/0.00125),0,519).astype(int)
    flat=v*640+u
    order=np.lexsort((-xyz[:,2],flat));f=flat[order]
    keep=np.r_[True,f[1:]!=f[:-1]];ids=order[keep]
    z=np.full((520,640),np.nan);im=np.full((520,640,3),35,dtype=np.uint8)
    z[v[ids],u[ids]]=xyz[ids,2];im[v[ids],u[ids]]=rgb[ids]
    # Splat nearest samples over at most two raster pixels (2.5mm); unknown stays unknown.
    valid=np.isfinite(z);dist,ij=distance_transform_edt(~valid,return_indices=True)
    fill=(~valid)&(dist<=2)
    z[fill]=z[ij[0][fill],ij[1][fill]];im[fill]=im[ij[0][fill],ij[1][fill]]
    pic=Image.fromarray(im);draw=ImageDraw.Draw(pic)
    for x in range(0,640,80):
        draw.line((x,0,x,519),fill=(110,110,110));draw.text((x+2,18),str(x),fill='yellow')
    for y in range(0,520,80):
        draw.line((0,y,639,y),fill=(110,110,110));draw.text((2,y+2),str(y),fill='yellow')
    draw.text((4,2),'METRIC TOP VIEW 640x520: 80 pixels = 10 cm. Forward UP.',fill='yellow')
    return np.array(pic),z

def overhead_xy(px):
    u,v=px;return np.array([0.75-v*0.00125,0.40-u*0.00125])

def overhead_point(z,px):
    u,v=np.clip(np.rint(px),[0,0],[639,519]).astype(int)
    h=float(z[v,u])
    if not np.isfinite(h):
        # Local refinement of Astra's selected pixel only, at most three pixels.
        ys,xs=np.mgrid[max(0,v-3):min(z.shape[0],v+4),max(0,u-3):min(z.shape[1],u+4)]
        valid=np.isfinite(z[ys,xs])&((xs-u)**2+(ys-v)**2<=9)
        if not valid.any():raise ValueError('Astra selected unknown overhead depth')
        k=np.argmin(np.where(valid,(xs-u)**2+(ys-v)**2,np.inf));j,i=np.unravel_index(k,valid.shape)
        u,v=int(xs[j,i]),int(ys[j,i]);h=float(z[v,u])
    return np.r_[overhead_xy([u,v]),h]

def marked(cam,name):
    im=Image.fromarray(cam['rgb']);d=ImageDraw.Draw(im)
    for u in range(0,640,80):
        d.line((u,0,u,359),fill=(130,130,130),width=1);d.text((u+2,17),str(u),fill='yellow')
    for v in range(0,360,60):
        d.line((0,v,639,v),fill=(130,130,130),width=1);d.text((2,v+2),str(v),fill='yellow')
    d.text((4,2),name+' 640x360, pixel x right, y down',fill='yellow')
    return np.array(im)

def row(F,g):
    x,y,z,w=Rotation.from_matrix(F[:3,:3]).as_quat();return [*F[:3,3],w,x,y,z,g]

def move(robot,F,g,seconds,settle_rounds=3):
    # The API servo already bounds Cartesian and angular speed.
    r=robot.act_arm([row(F,g)]*max(1,int(seconds*15)))
    for _ in range(settle_rounds):
        e=r['steps'][-1]
        if e['pos_error_m']<0.008 and e['rot_error_rad']<0.12: break
        r=robot.act_arm([row(F,g)]*6)
    robot.log('motion',seconds,'error',r['steps'][-1], 'width',r['proprio']['gripper_width_m'])
    return r

def ask(robot,prompt,images,schema):
    r=robot.astra(prompt,images=images,schema=schema,effort='low',tier='fast',instructions=SYS)
    robot.log('astra',r.get('latency_s'),r.get('json') or r.get('failure'))
    if not r.get('ok'): raise ValueError('Astra failed')
    return r['json']

def run(robot,object='object'):
    previous=''
    for attempt in range(2):
        obs=robot.observe(['pole','base']);cams=obs['cameras'];p=obs['proprio'];T=root_T(p)
        for name,c in cams.items():robot.save_image(f'initial_{attempt}_{name}',c['rgb'])
        top_image,zmap=overhead(cams,T)
        robot.save_image(f'overhead_{attempt}',top_image)
        plan_prompt=f'''Pick up the {object.replace('_',' ')}. {previous}
Images1/2 are pole/base RGB for identifying the object. Image3 is a calibrated metric overhead RGB point cloud,640x520 pixels,80 pixels=10cm,forward UP. Select ALL coordinates in IMAGE3. Gray holes are unknown, not objects.
Jaws open94mm, pads27mm along fingers/28mm wide. approach_tilt_deg chooses0=vertical down, or up to45deg angled down-and-forward (toward the object from the robot), perpendicular to the chosen closing axis. For upright or high bodies,30–45deg improves arm reach and lift clearance; use0 where flat geometry or clutter calls for vertical entry. The small arm struggles to lift with a vertical-down flange above root z~0.33m. Grasp low on stable bodies. Compare physical width with the scale; fit the object between OPEN fingers without pushing it.
center_px: horizontal CENTER of the intended grasp in the metric view, centered between opposite body sides. top_px: a valid colored pixel on the UPPER surface of that grip region for measuring height. These can differ. support_px: nearby BARE STOOL TOP.
height_fraction: pad-center height between support=0 and top=1; fingertips extend13.5mm lower. Choose stable body contact, clear of stool, avoiding caps/rims/stalks/tapered tips and nearby objects.
axis_mode=long_axis: closing_line_px=[x1,y1,x2,y2] marks longitudinal centerline; close perpendicular to it. Or axis_mode=closing: mark the desired jaw closing direction directly. In this metric view, lengths and angles are true horizontal lengths/angles. Choose the narrow side, not a diagonal.
lift_cm: choose enough for the OBJECT to finish at least5cm higher. Upright objects normally need10–12cm; long objects lying down may rotate down and need16–20cm. Respect reach; avoid an unnecessarily high lift. note: at most5 words.'''
        a=ask(robot,plan_prompt,[cams['pole']['rgb'],cams['base']['rgb'],top_image],PLAN)
        for repair in range(2):
            try:
                top=overhead_point(zmap,a['top_px']);support=overhead_point(zmap,a['support_px']);break
            except ValueError:
                if repair: return {'ok':False,'reason':'Selected height pixels have no measured depth'}
                a=ask(robot, 'Your selected top or support pixel has no measured depth within3pixels. Choose visibly colored measured surfaces, avoiding holes. Revise the plan.\n'+plan_prompt, [cams['pole']['rgb'],cams['base']['rgb'],top_image], PLAN)
        G=top.copy();G[:2]=overhead_xy(a['center_px'])
        G[2]=support[2]+float(np.clip(a['height_fraction'],0.05,0.95))*(top[2]-support[2])
        G[2]=max(G[2],support[2]+0.016)
        line=a['closing_line_px'];d=overhead_xy(line[2:])-overhead_xy(line[:2]);v=np.r_[d,0.];v/=np.linalg.norm(v)
        if a['axis_mode']=='long_axis':v=np.array([-v[1],v[0],0.])
        F0=np.array(p['flange_pose_root'])
        if np.dot(v,F0[:3,1])<0:v=-v # identical physical finger axis, shorter rotation
        h=np.array([-v[1],v[0],0.])
        if h[0]<-1e-6 or (abs(h[0])<=1e-6 and h[1]<0):h=-h
        theta=np.deg2rad(np.clip(a['approach_tilt_deg'],0,45))
        approach=np.sin(theta)*h+np.array([0.,0.,-np.cos(theta)]);F=np.eye(4);F[:3,:3]=np.column_stack([approach,v,np.cross(approach,v)]);F[:3,3]=G-0.13*approach
        H=F.copy()
        approach_distance=max(0.10,(top[2]-G[2]+0.05)/max(0.5,-approach[2]))
        H[:3,3]-=approach*approach_distance
        if np.linalg.norm(F[:3,3])>0.85: return {'ok':False,'reason':'outside safety reach'}
        robot.log('geometry',dict(top=top.tolist(),support=support.tolist(),pad=G.tolist(),closing=v.tolist(),flange=F.tolist()))
        move(robot,H,0,1.6,settle_rounds=15)
        move(robot,F,0,0.8)
        closed=robot.act_arm([row(F,1)]*10)['proprio']
        Fc=np.array(closed['flange_pose_root']);pad_at_close=Fc[:3,3]+0.13*Fc[:3,0]
        L=F.copy();L[2,3]+=np.clip(a['lift_cm']/100,0.08,0.20)
        move(robot,L,1,0.8)
        for check_index in range(3):
            obs2=robot.observe(['pole','base','wrist']);p2=obs2['proprio']
            Fnow=np.array(p2['flange_pose_root']);pad_now=Fnow[:3,3]+0.13*Fnow[:3,0]
            actual_delta=(pad_now-pad_at_close)*100
            for name,c in obs2['cameras'].items():robot.save_image(f'check_{attempt}_{check_index}_{name}',c['rgb'])
            b=ask(robot,f'''Independently verify the {object.replace('_',' ')} was picked up, not a distractor.
Images:1 BEFORE base;2 NOW pole;3 NOW base;4 NOW wrist. Compare object identity and actual elevation. Do not assume the previous target choice or commanded movement was correct.
Measured gripper width={p2['gripper_width_m']*1000:.1f}mm. Measured PAD movement since closing={actual_delta.round(1).tolist()}cm in root(+x forward,+y left,+z up). Current pad={pad_now.round(3).tolist()}m. Pad movement is not object movement: allow slip/rotation.
Choose done only if the named object is held and at least5cm above its initial resting position. lift_more if correctly held but too low: choose delta_cm=[dx,dy,dz], at most10cm horizontally and15cm upward. If vertical reach is limited, retract toward robot(negative x) while lifting. retry for a missed/wrong/slipped grasp needing a new plan; stop if unsafe. For non-lift actions delta_cm=[0,0,0]. ok=true only for confirmed success. note<=12 words.''', [cams['base']['rgb']]+[obs2['cameras'][n]['rgb'] for n in ['pole','base','wrist']], CHECK)
            if b['action']!='lift_more':break
            if check_index==2:return {'ok':False,'reason':'Additional lift safety limit'}
            d=np.clip(np.asarray(b['delta_cm'])/100,[-0.10,-0.10,0],[0.10,0.10,0.15])
            L=Fnow.copy();L[:3,3]+=d
            move(robot,L,1,0.8)
        if b['action']!='retry':return {'ok':b['ok'],'reason':b['note']}
        previous='Previous attempt failed: '+b['note']+' Choose corrected grasp.'
        move(robot,H,0,0.8)
    return {'ok':False,'reason':'Astra requested retry; attempt safety limit'}
