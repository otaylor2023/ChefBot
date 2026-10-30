"""Perception-driven, arm-only pick. All geometry comes from calibrated depth."""
import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation, Slerp


def planar(x,y,yaw):
    T=np.eye(4); c,s=np.cos(yaw),np.sin(yaw)
    T[:2,:2]=[[c,-s],[s,c]]; T[:2,3]=[x,y]
    return T


def cloud(cam, proprio):
    d=cam['depth']; v,u=np.indices(d.shape); K=cam['K']
    p=np.stack([(u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d],-1)
    T=np.linalg.inv(planar(*proprio['base_pose_odom']))@cam['cam_to_odom']
    return p@T[:3,:3].T+T[:3,3]


def row(F,g):
    q=Rotation.from_matrix(F[:3,:3]).as_quat()
    return [*F[:3,3],q[3],*q[:3],g]


def move(robot,F,g,n=30,settle=True,allow_stall=False):
    r=robot.act_arm([row(F,g)]*n)
    previous=None
    stalled=0
    for _ in range(12 if settle else 0):
        e=r['steps'][-1]
        if e['pos_error_m']<0.025 and e['rot_error_rad']<0.18: break
        score=e['pos_error_m']+0.13*e['rot_error_rad']
        stalled=stalled+1 if previous is not None and previous-score<0.001 else 0
        if stalled>=2 and (allow_stall or (e['pos_error_m']<0.04 and e['rot_error_rad']<0.4)):break
        previous=score
        r=robot.act_arm([row(F,g)]*15)
    robot.log('move',np.round(F[:3,3],3).tolist(),'err',r['steps'][-1], 'width',r['proprio']['gripper_width_m'])
    return r['proprio']


def pick(robot,object='object',retry=False):
    obs=robot.observe(['pole']); cam=obs['cameras']['pole']; p=obs['proprio']
    robot.save_image('initial',cam['rgb'])
    ds=robot.detect(cam['rgb'],[object.replace('_',' ')],model='grounding-dino',threshold=0.2)
    robot.log('detections',ds)
    ds.sort(key=lambda d:d['score'],reverse=True)
    ambiguous=not ds or ds[0]['score']<0.5 or (len(ds)>1 and ds[0]['score']-ds[1]['score']<0.15)
    semantic_point=None
    if True:  # Semantic identity also checks confident detector mistakes.
        models=robot.models()['llm']
        robot.log('models',models)
        names=list(models)
        model=min(names,key=lambda n:models[n]['input_per_mtok']+models[n]['output_per_mtok'])
        ans=robot.llm(model, 'Locate the '+object.replace('_',' ')+' among the objects on the low stool. Return JSON only: {"x": x, "y": y} for a point inside its main body. Coordinates are integers normalized 0 to 1000, x left to right, y top to bottom. Identify the requested object carefully, not another object or the stool.', images=[cam['rgb']], max_output_tokens=1024, temperature=0, response_schema={"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"}},"required":["x","y"]})
        robot.log('identity',ans)
        try:
            t=ans['text']; data=json.loads(t[t.index('{'):t.rindex('}')+1]); u,v=data['x'],data['y']
            u=float(u)*cam['rgb'].shape[1]/1000; v=float(v)*cam['rgb'].shape[0]/1000
            semantic_point=[u,v]
            candidates=[d for d in ds if d['box'][0]<=u<=d['box'][2] and d['box'][1]<=v<=d['box'][3]]
            candidates.sort(key=lambda d:(d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]))
            if candidates:
                robot.log('semantic box',candidates[0])
                mask=robot.segment(cam['rgb'],boxes=[candidates[0]['box']])[0]
            else:
                mask=robot.segment(cam['rgb'],points=[[u,v]],point_labels=[1])[0]
        except Exception as e:
            robot.log('identity fallback',str(e))
            if not ds:return {'ok':False,'reason':'no detection'}
            mask=robot.segment(cam['rgb'],boxes=[ds[0]['box']])[0]
    else:
        mask=robot.segment(cam['rgb'],boxes=[ds[0]['box']])[0]
    robot.save_image('mask',mask.astype(np.uint8)*255)
    xyz=cloud(cam,p); valid=(cam['depth']>0)&(cam['depth']<2)
    pts=xyz[mask&valid]
    if len(pts)<20:return {'ok':False,'reason':'insufficient depth'}
    if semantic_point is not None:
        # Objects in this task fit a 94 mm gripper. A much wider mask is background.
        _,axes=np.linalg.eigh(np.cov(pts[:,:2].T))
        projected=pts[:,:2]@axes
        widths=np.diff(np.percentile(projected,[2,98],axis=0),axis=0)[0]
        if min(widths)>0.11:
            robot.log('box mask too broad; segment semantic point',widths.tolist())
            mask=robot.segment(cam['rgb'],points=[semantic_point],point_labels=[1])[0]
            pts=xyz[mask&valid]
            if len(pts)<20:return {'ok':False,'reason':'insufficient target depth'}
    # Reject background masks outside the arm workspace before any motion.
    # 0.8 m conservatively covers the 0.6 m arm plus the fingertip offset.
    if np.linalg.norm(np.median(pts,axis=0))>0.8:
        robot.log('mask outside reach; trying detector boxes')
        found=False
        for d in ds:
            candidate=robot.segment(cam['rgb'],boxes=[d['box']])[0]
            q=xyz[candidate&valid]
            if len(q)>=20 and np.linalg.norm(np.median(q,axis=0))<0.8:
                mask=candidate;pts=q;found=True
                robot.log('reachable detector fallback',d)
                break
        if not found:return {'ok':False,'reason':'no reachable target mask'}
    # Robust bounds, with a slight erosion to reject mask boundary depth bleed.
    er=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    if (er&valid).sum()>30:pts=xyz[er&valid]
    lo,hi=np.percentile(pts,[2,98],axis=0)
    center=(lo+hi)/2
    # Recover the support plane from the ring around the object.
    ring=cv2.dilate(mask.astype(np.uint8),np.ones((41,41),np.uint8)).astype(bool)&~mask&valid
    near=xyz[ring]; near=near[(near[:,2]<hi[2]-0.015)&(near[:,2]>lo[2]-0.12)]
    floor=lo[2]
    if len(near)>30:
        bins=np.arange(lo[2]-0.12,hi[2],0.004); hist,edges=np.histogram(near[:,2],bins)
        floor=edges[np.argmax(hist)]+0.002
    pts=pts[pts[:,2]>floor+0.006]
    if len(pts)<20:return {'ok':False,'reason':'no elevated object'}
    xy=pts[:,:2]; vals,vecs=np.linalg.eigh(np.cov(xy.T)); closing=vecs[:,0]
    # Resolve 180-degree symmetry toward a natural wrist pose.
    if retry and np.ptp(xy@vecs[:,1])<0.082: closing=vecs[:,1]
    if closing[1]>0:closing=-closing
    Y=np.r_[closing,0.]; X=np.array([0.,0.,-1.]); Z=np.cross(X,Y)
    R=np.column_stack([X,Y,Z])
    a=xy@closing; long=np.array([-closing[1],closing[0]]); b=xy@long
    center[:2]=closing*np.mean(np.percentile(a,[2,98]))+long*np.mean(np.percentile(b,[2,98]))
    top=np.percentile(pts[:,2],98); height=top-floor
    # A visible cylindrical surface has a biased bounding-box center.
    # Accept a circle only when the body itself supports a small, accurate fit.
    spans=np.diff(np.percentile(xy@vecs,[2,98],axis=0),axis=0)[0]
    body=pts[(pts[:,2]>floor+0.25*height)&(pts[:,2]<floor+0.75*height),:2]
    if height>1.4*max(spans) and len(body)>30:
        origin=np.mean(body,axis=0); q=body-origin
        A=np.column_stack([2*q,np.ones(len(q))])
        fit=np.linalg.lstsq(A,np.sum(q*q,axis=1),rcond=None)[0]
        radius=np.sqrt(max(0.,fit[2]+np.sum(fit[:2]**2)))
        cc=origin+fit[:2]
        residual=np.sqrt(np.mean((np.linalg.norm(body-cc,axis=1)-radius)**2))
        if (0.008<radius<0.055 and residual<min(0.003,0.12*radius)
                and np.max(np.ptp(body@vecs,axis=0))>radius
                and np.linalg.norm(cc-center[:2])<0.035):
            robot.log('circular body center',cc.tolist(),'radius',radius,'residual',residual)
            center[:2]=cc
    center[2]=floor+max(0.018,min(height*0.5,height-0.012))
    if retry: center[2]+=0.008
    robot.log('geometry',{'center':center.tolist(),'floor':float(floor),'top':float(top),'width':float(np.ptp(a)),'axis':closing.tolist()})
    F=np.eye(4);F[:3,:3]=R;F[:3,3]=center-X*0.13
    H=F.copy();H[2,3]=top+0.13+0.09
    move(robot,H,0,45,allow_stall=True)
    p=move(robot,F,0,35,allow_stall=True)
    if np.linalg.norm(np.array(p['flange_pose_root'])[:3,3]-F[:3,3])>0.05:
        # Recover a missed grasp pose, keeping the open fingers above the object.
        move(robot,H,0,35,allow_stall=True)
        Y=F[:3,1]; horizontal=np.cross(Y,np.array([0.,0.,1.]))
        if np.dot(horizontal[:2],center[:2])<0:horizontal=-horizontal
        X=(horizontal+np.array([0.,0.,-1.]))/np.sqrt(2)
        F[:3,:3]=np.column_stack([X,Y,np.cross(X,Y)])
        F[:3,3]=center-0.13*X
        H=F.copy();H[2,3]+=top-center[2]+0.06
        robot.log('tilted grasp recovery after missed descent')
        move(robot,H,0,45,allow_stall=True)
        move(robot,F,0,35,allow_stall=True)
    p=move(robot,F,1,15,settle=False)
    L=F.copy();L[2,3]+=0.17
    p=move(robot,L,1,35,settle=False)
    if np.linalg.norm(np.array(p['flange_pose_root'])[:3,3]-L[:3,3])>0.025:
        # Pitch around the closing axis while preserving the requested pad midpoint.
        # Choose the equivalent pitch that brings the flange toward the arm base.
        Y=L[:3,1]; horizontal=np.cross(Y,np.array([0.,0.,1.]))
        if np.dot(horizontal[:2],center[:2])<0:horizontal=-horizontal
        X=0.70710678*horizontal+np.array([0.,0.,-0.70710678])
        target=L[:3,3]+0.13*L[:3,0]
        L[:3,:3]=np.column_stack([X,Y,np.cross(X,Y)])
        L[:3,3]=target-0.13*X
        robot.log('tilted lift recovery')
        p=move(robot,L,1,60,allow_stall=True)
    return {'ok':p['gripper_width_m']>0.004,'reason':'completed depth-based grasp'}


def run(robot,object='object'):
    initial=np.array(robot.proprio()['flange_pose_root'])
    result=pick(robot,object)
    if not result['ok']:
        robot.log('retry empty grasp')
        move(robot,initial,0,45)
        result=pick(robot,object,retry=True)
    return result
