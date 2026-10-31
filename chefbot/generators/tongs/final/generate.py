"""Procedural kitchen tongs. SI units; no external assets or downloaded data.

Pivot/channel, bent spring band, pasta comb, wire/strap scissors, bamboo and
molded-plastic flexure families. Mirror-coordinate springs represent a coil or equivalent flexure;
band and bamboo arms are rigid approximations of distributed elastic bending.
Convex patches tile the hollow surfaces and open slots. Lock parts are visual.
"""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import PchipInterpolator
import mujoco


def fmt(x):
    if isinstance(x, (tuple, list, np.ndarray)):
        return " ".join(f"{float(v):.9g}" for v in x)
    return str(x)


def contact_metadata(material, density, grip_mu):
    return dict(tip_contact_priority=1,contact_margin_m=.0003,
                tip_torsional_friction_m=.003,tip_rolling_friction_m=.00035,
                tip_contact_timeconst_s=.004 if material in ('silicone','nylon') else .003,
                head_density_kg_m3=density,grip_friction=grip_mu)


def element(parent, tag, **kw):
    return ET.SubElement(parent, tag, {k: fmt(v) for k, v in kw.items()})


def shell_grid(points, thickness):
    """Closed shell from an (nx,nz,3) inner surface; thickness is a vector."""
    p = np.asarray(points)
    n, m, _ = p.shape
    vertices = np.concatenate([p.reshape(-1, 3), (p + thickness).reshape(-1, 3)])
    faces = []
    k = n * m
    for i in range(n - 1):
        for j in range(m - 1):
            a = i * m + j
            for off, reverse in ((0, False), (k, True)):
                tris = [(a+off, a+m+off, a+1+off), (a+1+off, a+m+off, a+m+1+off)]
                faces.extend([tuple(reversed(t)) if reverse else t for t in tris])
    edges = [(j, j+1) for j in range(m-1)]
    edges += [((n-1)*m+j+1, (n-1)*m+j) for j in range(m-1)]
    edges += [(i*m+m-1, (i+1)*m+m-1) for i in range(n-1)]
    edges += [((i+1)*m, i*m) for i in range(n-1)]
    for a, b in edges:
        faces.extend([(a,b,b+k),(a,b+k,a+k)])
    faces = np.asarray(faces, int)
    # Consistent positive signed volume, even for mirrored arm coordinates.
    vol = np.sum(np.einsum('ij,ij->i', vertices[faces[:,0]],
                  np.cross(vertices[faces[:,1]], vertices[faces[:,2]]))) / 6
    if vol < 0:
        faces = faces[:, ::-1]
    return vertices, faces, abs(vol)


class Builder:
    def __init__(self, seed, out):
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.root = ET.Element('mujoco', model=f'kitchen_tongs_{seed}')
        element(self.root, 'compiler', angle='radian')
        self.asset = element(self.root, 'asset')
        self.obj = element(element(self.root, 'worldbody'), 'body', name='object')
        self.count = 0
        self.mass = 0.0

    def material(self, name, rgba, specular, shininess):
        element(self.asset, 'material', name=name, rgba=rgba, specular=specular, shininess=shininess)

    def mesh(self, v, f=None, smooth=False, uv=None):
        name = f'mesh_{self.count}'
        self.count += 1
        if f is None:
            hull = ConvexHull(v)
            f = hull.simplices.copy()
            for i, tri in enumerate(f):
                a,b,c = v[tri]
                if np.dot(np.cross(b-a,c-a), hull.equations[i,:3]) < 0:
                    f[i] = tri[::-1]
        path = self.out / (name+'.obj')
        with path.open('w') as fp:
            for p in v:
                fp.write('v '+fmt(p)+'\n')
            if uv is not None:
                for p in uv:fp.write('vt '+fmt(p)+'\n')
            for face in f:
                fp.write('f '+' '.join(str(int(i)+1) if uv is None else f'{int(i)+1}/{int(i)+1}' for i in face)+'\n')
        element(self.asset, 'mesh', name=name, file=path.name, smoothnormal='true' if smooth else 'false')
        return name

    def geom(self, body, material, mass=0, collision=False, friction=.4, soft=False, **kw):
        self.mass += mass
        attrs = dict(material=material, mass=mass, group=3 if collision else 1,
                     contype=1 if collision else 0, conaffinity=1 if collision else 0)
        if collision:
            attrs.update(friction=[friction, .003, .00035], condim=6,
                         solref=[.004 if soft else .003, 1], solimp=[.95,.99,.001])
            if material in ('head','wood'):
                attrs['priority']=1
            if material=='head':
                attrs['margin']=.0003
        attrs.update(kw)
        return element(body, 'geom', **attrs)

    def solid(self, body, v, f, material, mass, friction=.4, soft=False, visual=True):
        name = self.mesh(v, f)
        self.geom(body, material, mass, True, friction, soft, type='mesh', mesh=name)
        if visual:
            self.geom(body, material, type='mesh', mesh=name)

    def primitive(self, body, material, density, friction=.4, soft=False, **kw):
        typ, size = kw['type'], kw['size']
        if typ == 'box':
            volume = 8*np.prod(size)
        elif typ == 'cylinder':
            volume = math.pi*size[0]**2*2*size[1]
        elif typ == 'sphere':
            volume = 4/3*math.pi*size[0]**3
        elif typ == 'capsule':
            ends = np.asarray(kw['fromto']).reshape(2,3)
            volume = math.pi*size[0]**2*np.linalg.norm(ends[1]-ends[0])+4/3*math.pi*size[0]**3
        else:
            raise ValueError(typ)
        self.geom(body, material, volume*density, True, friction, soft, **kw)
        self.geom(body, material, **kw)

    def tube(self, body, points, radius, material):
        # Smooth connected visual wire, from its analytic centreline.
        p = np.asarray(points)
        tang = np.gradient(p, axis=0)
        tang /= np.linalg.norm(tang, axis=1)[:,None]
        vertices = []
        # Parallel transport avoids an abrupt cross-section twist when a
        # curved rim turns from the long edge around its broad front.
        ref=np.eye(3)[np.argmin(abs(tang[0]))]
        a=np.cross(tang[0],ref);a/=np.linalg.norm(a)
        for x,t in zip(p,tang):
            a=a-np.dot(a,t)*t
            a/=np.linalg.norm(a)
            b = np.cross(t,a)
            vertices.extend([x+radius*(a*math.cos(phi)+b*math.sin(phi)) for phi in np.linspace(0,2*math.pi,9)[:-1]])
        faces = []
        for i in range(len(p)-1):
            for j in range(8):
                a,b = i*8+j,i*8+(j+1)%8
                faces.extend([(a,b,a+8),(b,b+8,a+8)])
        for j in range(1,7):
            faces.extend([(0,j+1,j),((len(p)-1)*8,(len(p)-1)*8+j,(len(p)-1)*8+j+1)])
        name = self.mesh(np.array(vertices), np.array(faces), True)
        self.geom(body,material,type='mesh',mesh=name)


def certify(builder):
    """Physical metadata from our actual model, including finite-rate closure.

    This does not import a lab test or alter any geometry or spring. Length is
    the model's bounding extent and mass its explicit material mass. Closure
    is recorded under a documented 20 N/s squeeze ramp, rather than pretending
    the static spring equation describes a moving, massive tong exactly.
    """
    root = builder.root
    tmp = ET.fromstring(ET.tostring(root))
    element(tmp,'option',timestep=.002,gravity=[0,0,0],cone='elliptic',impratio=20,noslip_iterations=4)
    # Only this temporary in-memory model uses absolute paths. Output assets
    # stay portable, and generate() works from any caller working directory.
    for asset in list(tmp.findall('asset/mesh')) + list(tmp.findall('asset/texture')):
        if asset.get('file'):
            asset.set('file', str(builder.out / asset.get('file')))
    model = mujoco.MjModel.from_xml_string(ET.tostring(tmp,encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    signs = np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    pts = []
    for g in range(model.ngeom):
        aabb = model.geom_aabb[g]
        pts.extend(data.geom_xpos[g] + (aabb[:3]+signs*aabb[3:]) @ data.geom_xmat[g].reshape(3,3).T)
    lo,hi = np.min(pts,axis=0),np.max(pts,axis=0)
    centre = (hi+lo)/2
    # Move top-level fixed geoms and arm frames together to the actual centre.
    for child in builder.obj:
        if child.tag in ('body','geom','site','inertial'):
            pos = np.fromstring(child.get('pos','0 0 0'), sep=' ')
            child.set('pos',fmt(pos-centre))
    ids = {n:model.site(n).id for n in ['grip_a','grip_b','tip_a','tip_b']}
    arms = [model.body('arm_a').id, model.body('arm_b').id]
    mujoco.mj_step(model,data,nstep=500)
    gap = np.linalg.norm(data.site_xpos[ids['tip_b']]-data.site_xpos[ids['tip_a']])
    closing = None
    for tick in range(300):
        force = tick * .2
        ga,gb = [data.site_xpos[ids[n]].copy() for n in ['grip_a','grip_b']]
        direction = (gb-ga)/np.linalg.norm(gb-ga)
        data.xfrc_applied[:] = 0
        for arm,pt,f in [(arms[0],ga,direction*force),(arms[1],gb,-direction*force)]:
            data.xfrc_applied[arm,:3] = f
            data.xfrc_applied[arm,3:] = np.cross(pt-data.xipos[arm],f)
        mujoco.mj_step(model,data,nstep=5)
        mid = (data.site_xpos[ids['tip_a']]+data.site_xpos[ids['tip_b']])/2
        touching = False
        for contact in data.contact:
            b1,b2 = model.geom_bodyid[contact.geom1],model.geom_bodyid[contact.geom2]
            if set([int(b1),int(b2)])==set(arms) and np.linalg.norm(contact.pos-mid)<.045:
                touching=True
                break
        if touching:
            closing = force
            break
    if closing is None:
        raise ValueError('Generated tips could not close at 60 N')
    return float(hi[0]-lo[0]),float(gap),float(closing)


def generate(seed: int, out_dir: str) -> dict:
    if np.random.default_rng(seed ^ 0xBABA).random()<.10:
        return generate_bamboo(seed,out_dir)
    if np.random.default_rng(seed ^ 0xB710).random()<.10:
        return generate_molded(seed,out_dir)
    rng = np.random.default_rng(seed)
    design_rng = np.random.default_rng(seed ^ 0x5A17B3)
    # Overall nominal classes refer to the kitchen working part, not packaging.
    nominal = float(rng.choice([.23,.27,.30,.35]))
    L = nominal + float(rng.uniform(-.008,.008))
    head_len = float(rng.uniform(.055,.070))*math.sqrt(L/.28)
    head_width = float(rng.uniform(.032,.044))
    arm_width = float(rng.uniform(.015,.021))
    sheet = float(rng.uniform(.00065,.0010))
    channel_depth = float(rng.uniform(.003,.005))
    gap_target = float(rng.uniform(.065,.105))
    grip_x = L * float(rng.uniform(.32,.40))
    head_t = float(rng.uniform(.0035,.0050))
    cup = float(rng.uniform(.0030,.0045))
    scallop = float(rng.uniform(.07,.13))
    lobes = int(rng.integers(4,7))
    k = float(rng.uniform(1.5,3.4))
    preload = float(rng.uniform(.20,.36))
    mu = float(rng.uniform(.85,1.15))
    color_idx = int(rng.integers(0,5))
    colors = [[.20,.21,.22,1],[.65,.055,.035,1],[.045,.33,.36,1],[.16,.30,.12,1],[.22,.11,.32,1]]
    family = str(design_rng.choice(['silicone','steel','nylon'],p=[.45,.35,.20]))
    head_profile = 'scalloped paddle'
    lock = 'pull ring'
    grips = bool(design_rng.random() < .85)
    head_density=1100
    head_rgba=colors[color_idx]
    bead_radius=.0009
    if family=='steel':
        L=float(design_rng.choice([.18,.235,.30,.395]))+float(design_rng.uniform(-.006,.006))
        grip_x=L*float(design_rng.uniform(.32,.40))
        head_len=float(design_rng.uniform(.047,.069))*math.sqrt(L/.28)
        head_width=float(design_rng.uniform(.030,.044))
        head_t=sheet
        cup=float(design_rng.uniform(.003,.005))
        mu=float(design_rng.uniform(.30,.55))
        head_density=7900
        head_rgba=[.70,.73,.76,1]
        bead_radius=sheet*.65
        lock=str(design_rng.choice(['none','sliding collar','pull ring'],p=[.50,.30,.20]))
        if L>.36 and lock=='pull ring':lock='none'
        grips=bool(design_rng.random()<.25)
        if design_rng.random()<.22:
            head_profile='smooth serving spoon'
            scallop=0.
        else:head_profile='scalloped steel cup'
    elif family=='nylon':
        head_t=float(design_rng.uniform(.0028,.0040))
        cup=float(design_rng.uniform(.0025,.0035))
        mu=float(design_rng.uniform(.50,.75))
        head_density=1150
        head_rgba=[.10,.105,.11,1] if design_rng.random()<.7 else [.83,.80,.65,1]
        color_idx=5 if head_rgba[0]<.5 else 6
        head_profile='scalloped nylon paddle'
        grips=True
    elif design_rng.random()<.28:
        head_profile='smooth silicone leaf'
        scallop=0.
    is_pasta=bool(np.random.default_rng(seed ^ 0xCB617).random()<.14)
    construction = 'spring band' if design_rng.random()<.20 or is_pasta else 'pivot'
    band_radius=0.
    band_thickness=0.
    if construction=='spring band':
        family='steel'
        L=float(design_rng.choice([.18,.23,.30]))+float(design_rng.uniform(-.006,.006))
        grip_x=L*float(design_rng.uniform(.35,.42))
        arm_width=float(design_rng.uniform(.013,.020))
        head_len=float(design_rng.uniform(.040,.060))*math.sqrt(L/.23)
        head_width=float(design_rng.uniform(.027,.040))
        head_t=sheet
        cup=float(design_rng.uniform(.0025,.004))
        mu=float(design_rng.uniform(.35,.55))
        head_density=7900
        head_rgba=[.70,.73,.76,1]
        bead_radius=sheet*.65
        lock='none'
        grips=False
        head_profile='scalloped serving cup'
        band_radius=float(design_rng.uniform(.006,.010))
        k=float(design_rng.uniform(2.5,5.5))*L/.23
        preload=0.
        # A split equivalent-coordinate model of the elastic U bend. The
        # physical relative-angle rate is half the per-arm joint rate.
        band_thickness=(k*.5*12*math.pi*band_radius/(186e9*arm_width))**(1/3)
    if is_pasta:
        L=float(design_rng.choice([.20,.23,.29]))+float(design_rng.uniform(-.005,.005))
        grip_x=L*float(design_rng.uniform(.35,.42))
        head_len=float(design_rng.uniform(.055,.070))
        head_width=float(design_rng.uniform(.035,.045))
        head_t=float(design_rng.uniform(.0011,.0015))
        k=float(design_rng.uniform(2.5,5.5))*L/.23
        band_thickness=(k*.5*12*math.pi*band_radius/(186e9*arm_width))**(1/3)
        head_profile='five-tooth pasta comb'
        scallop=0.
    is_scissor=bool(not is_pasta and np.random.default_rng(seed ^ 0xA3129).random()<.16)
    flat_scissor=bool(is_scissor and np.random.default_rng(seed ^ 0x5C155).random()<.65)
    if is_scissor:
        construction='scissor'
        family='steel'
        L=float(design_rng.uniform(.165,.210))
        # Robot pinch point on the exposed shank just ahead of the loop.
        # Surface normal of the diagonal wire (25 mm x 20 mm run).
        grip_x=-.0345+.0025*20/math.hypot(25,20)
        if flat_scissor:grip_x=-.0345+.007*20/math.hypot(25,20)
        arm_width=.016
        head_len=float(design_rng.uniform(.048,.060))
        head_width=float(design_rng.uniform(.032,.042))
        head_t=sheet
        head_density=7900
        cup=float(design_rng.uniform(.0025,.004))
        mu=float(design_rng.uniform(.35,.55))
        head_rgba=[.70,.73,.76,1]
        bead_radius=sheet*.8
        k=float(design_rng.uniform(.9,1.6))
        preload=float(design_rng.uniform(.12,.22))
        band_radius=0.
        band_thickness=0.
        grips=True
        lock='none'
        head_profile='scalloped steel cup'
    if scallop==0:
        head_profile=head_profile.replace('scalloped','smooth')
    # Mini tongs have a smaller natural mouth than long grill utensils.
    if L<.27:gap_target*=math.sqrt(L/.27)
    head_width=min(head_width,head_len*.75)
    detail_rng=np.random.default_rng(seed ^ 0x44A11)
    detail='smooth' if family=='steel' else str(detail_rng.choice(['smooth','ribs','studs'],p=[.3,.4,.3]))
    relief_height=0. if detail=='smooth' else float(detail_rng.uniform(.00035,.00065))
    def head_relief(u,v):
        if detail=='smooth' or abs(v)>=.78:return 0.
        if detail=='ribs':
            return relief_height*sum(math.exp(-((u-a)/.020)**2) for a in np.linspace(.25,.82,6))*(1-(v/.78)**6)
        return relief_height*sum(math.exp(-((u-a)/.025)**2-((v-b)/.115)**2) for a in [.30,.46,.62,.78] for b in [-.48,0,.48])
    tip_x = L-head_len*.48
    base_y = -.004
    longcup = .0015
    tip_y = base_y+cup+longcup*(1-(2*.52-1)**2)+.004*(1-.52)**4
    tip_y-=head_relief(.52,0)
    if is_pasta:
        tip_x=L-head_len*.5+.0045
        spine_z=head_width*.5-.006
        tv=spine_z/(spine_z+head_width*.5)
        tip_y=.003-.010*tv*tv
    half = math.asin(gap_target/(2*math.hypot(tip_x,tip_y)))-math.atan2(tip_y,tip_x)
    if construction=='spring band':
        for _ in range(8):
            half=math.asin((gap_target*.5-band_radius*math.cos(half))/math.hypot(tip_x,tip_y))-math.atan2(tip_y,tip_x)
    if construction!='scissor':
        # Choose an accessible point on the actual handle surface. Long/wide
        # tongs are squeezed closer to their hinge; the pad follows that point.
        outer_bound=(.003 if construction=='spring band' else .010)+sheet+(.0031 if grips else 0.)
        grip_x=min(grip_x,(.0325-(band_radius+outer_bound)*math.cos(half))/math.sin(half))
    # Rolled stainless rim / rounded molded edge, with physical tube volume.
    bead_radius=sheet*1.4 if family=='steel' else head_t*.50
    hinge_damping=.05*(L/.23)**2
    B = Builder(seed,out_dir)
    B.material('steel',[.65,.68,.71,1],.85,.65)
    B.material('polished',[.52,.55,.58,1],.95,.85)
    B.material('spring',[.23,.25,.28,1],.65,.55)
    B.material('head',head_rgba,.8 if family=='steel' else .18,.65 if family=='steel' else .15)
    B.material('grip',[.105,.115,.12,1],.17,.1)
    B.material('loop_grip',[.60,.36,.055,1] if color_idx%2 else [.085,.10,.11,1],.15,.12)
    start_head = L-head_len
    neck_ratio=arm_width*.60/(head_width*(1-scallop))
    widthfun=PchipInterpolator([0,.15,.34,.55,.75,.91,1],[neck_ratio,.65,.91,1,.93,.62,.025])
    blunt_nose=bool(family=='steel' and not is_pasta and np.random.default_rng(seed ^ 0xC411).random()<.55)
    nose_fraction=.025
    if blunt_nose:
        nose_fraction=.36
        widthfun=PchipInterpolator([0,.15,.34,.55,.75,.91,1],[neck_ratio,.65,.91,1,.96,.76,nose_fraction])
    if family=='nylon':
        nose_fraction=.30
        widthfun=PchipInterpolator([0,.15,.34,.55,.75,.91,1],[neck_ratio,.65,.95,1,.96,.76,.30])
    front_rim=blunt_nose or family=='nylon'
    def hw(u):
        return head_width*.5*widthfun(u)*(1-scallop*(.5+.5*np.cos(2*math.pi*lobes*np.asarray(u))))
    def headpoints(us,vs,side):
        points=[]
        for u in us:
            row=[]
            envelope=head_width*.5*widthfun(u)
            blend=min(1.,u/.15);blend=blend*blend*(3-2*blend)
            for v in vs:
                z=v*hw(u)
                # Scallops trim a smooth spoon surface. They do not corrugate
                # its entire interior, as scaling curvature by scalloped width
                # would do. Blend the narrow neck into the handle's section.
                yy=base_y+cup*(1-(z/envelope)**2)+longcup*(1-(2*u-1)**2)+.004*(1-u)**4
                neck=.003 if construction=='spring band' else .004-channel_depth*(1-max(0,1-v**4)**.25)
                yy=blend*yy+(1-blend)*neck
                yy-=head_relief(u,z/envelope)
                # A broad serving nose has a slightly crowned front edge.
                nose_blend=max(0.,(u-.85)/.15)**2 if blunt_nose else 0.
                row.append([start_head+u*head_len-head_len*.025*v*v*nose_blend,side*yy,z])
            points.append(row)
        return np.array(points)
    def head_shell(us,vs,side):
        pts=headpoints(us,vs,side)
        offset=np.zeros_like(pts)
        for i,u in enumerate(us):
            envelope=head_width*.5*widthfun(u)
            for j in range(len(vs)):
                offset[i,j,1]=side*(head_t+head_relief(u,pts[i,j,2]/envelope))
        return shell_grid(pts,offset)
    xstations=np.array([-.004,.045,grip_x+.035,start_head])
    ystations=np.array([channel_depth+.002,.010,.009,.004])
    wstations=np.array([arm_width*.78,arm_width*1.14,arm_width*.92,arm_width*.60])
    if construction=='spring band':
        xstations[0]=0.
        ystations=np.array([0.,.002,.001,.003])
        wstations=np.array([arm_width,arm_width*.98,arm_width*.85,arm_width*.65])
    elif construction=='scissor':
        xstations=np.array([-.035,.015,.070,start_head])
    else:
        xstations[0]=.016
    yfun=PchipInterpolator(xstations,ystations)
    wfun=PchipInterpolator(xstations,wstations)
    def channel_points(xs,ss,side,offset=0):
        p,n=[],[]
        for x in xs:
            row,nrow=[],[]
            w=float(wfun(x));y=float(yfun(x))
            if construction=='pivot' and x<.045:
                root_half=arm_width*.5+(.0018 if side<0 else -.0005)
                blend=np.clip((x-.022)/(.045-.022),0.,1.)
                blend=blend*blend*(3-2*blend)
                w=2*root_half*(1-blend)+w*blend
            for s in ss:
                if construction=='spring band':
                    normal=np.array([0,side,0])
                    row.append(np.array([x,side*y,w*.5*s/(math.pi/2)])+normal*offset)
                    nrow.append(normal)
                    continue
                # A stamped U section has a broad web and rounded corners,
                # rather than a half-elliptical, tube-like outer surface.
                cs=max(0.,math.cos(s));sn=math.sin(s)
                normal=np.array([0,side*w*.5*cs**1.5,channel_depth*np.sign(sn)*abs(sn)**1.5])
                normal/=np.linalg.norm(normal)
                row.append(np.array([x,side*(y-channel_depth*(1-cs**.5)),w*.5*np.sign(sn)*abs(sn)**.5])+normal*offset)
                nrow.append(normal)
            p.append(row);n.append(nrow)
        return np.array(p),np.array(n)
    for letter,side in [('a',-1),('b',1)]:
        body_pos=[-band_radius*math.sin(half),side*band_radius*math.cos(half),0]
        arm=element(B.obj,'body',name='arm_'+letter,pos=body_pos,quat=[math.cos(half/2),0,0,side*math.sin(half/2)])
        element(arm,'joint',name='hinge_'+letter,type='hinge',axis=[0,0,-side],range=[0,half+.10],stiffness=k,springref=-preload,damping=hinge_damping,armature=.000002,solreflimit=[.002,1],solimplimit=[.99,.999,.0001])
        if construction=='scissor':
            # Continuous wire shank passes across the rivet, with clearance in
            # z between the two arms. Bent ends terminate at a coated oval loop.
            rod=.0025
            scissor_z=.004 if flat_scissor else .006
            shaft=np.array([[-.045,-side*.020,side*scissor_z],[-.020,0,side*scissor_z],[.026,side*.001,side*scissor_z],[start_head,side*.004,0.]])
            if flat_scissor:
                widths=[.014,.014,.011,.006]
                sy=PchipInterpolator(shaft[:,0],shaft[:,1])
                sz=PchipInterpolator(shaft[:,0],shaft[:,2])
                sw=PchipInterpolator(shaft[:,0],widths)
                def strap(xs):
                    ps=[]
                    for x in xs:
                        normal=np.array([-float(sy.derivative()(x)),1.,0.]);normal/=np.linalg.norm(normal)
                        center=np.array([x,float(sy(x)),float(sz(x))-.0012])
                        ps.append([center+normal*v*float(sw(x))/2 for v in [-1,1]])
                    return shell_grid(np.array(ps),[0,0,.0024])
                sv,sf,svol=strap(np.linspace(shaft[0,0],shaft[-1,0],81))
                B.geom(arm,'polished',type='mesh',mesh=B.mesh(sv,sf,True))
                for xa,xb in zip(shaft[:-1,0],shaft[1:,0]):
                    sv,sf,svol=strap(np.linspace(xa,xb,17))
                    B.solid(arm,sv,None,'polished',svol*7900,.4,False,False)
                gx=-.0345
                normal=side*np.array([float(sy.derivative()(gx)),-1.,0.]);normal/=np.linalg.norm(normal)
                strap_grip=np.array([gx,float(sy(gx)),float(sz(gx))])+normal*float(sw(gx))/2
                grip_x=float(strap_grip[0])
            else:
                B.tube(arm,shaft,rod,'polished')
                for a,b in zip(shaft[:-1],shaft[1:]):
                    volume=math.pi*rod**2*np.linalg.norm(b-a)+4/3*math.pi*rod**3
                    B.geom(arm,'polished',volume*7900,True,.4,type='capsule',size=[rod],fromto=np.r_[a,b])
            # The oval's frontmost point joins the bent shank at x=-45 mm.
            ts=np.linspace(0,2*math.pi,65)
            oval=np.column_stack([-.070+.025*np.cos(ts),-side*.020+.013*np.sin(ts),np.full(len(ts),side*.006)])
            if flat_scissor:
                def loop_strip(angles):
                    ps=np.array([[[-.070+rx*math.cos(t),-side*.020+ry*math.sin(t),side*scissor_z-.0012] for rx,ry in [(.023,.014),(.028,.019)]] for t in angles])
                    return shell_grid(ps,[0,0,.0024])
                lv,lf,lvol=loop_strip(ts)
                B.geom(arm,'polished',type='mesh',mesh=B.mesh(lv,lf,True))
                for ta,tb in zip(np.linspace(0,2*math.pi,9)[:-1],np.linspace(0,2*math.pi,9)[1:]):
                    lv,lf,lvol=loop_strip(np.linspace(ta,tb,9))
                    B.solid(arm,lv,None,'polished',lvol*7900,.4,False,False)
            else:
                B.tube(arm,oval,.0042,'loop_grip')
                coarse=oval[::8]
                for a,b in zip(coarse[:-1],coarse[1:]):
                    length_seg=np.linalg.norm(b-a)
                    volume=math.pi*.0042**2*length_seg+4/3*math.pi*.0042**3
                    mass=math.pi*rod**2*length_seg*7900+(volume-math.pi*rod**2*length_seg)*1100
                    B.geom(arm,'loop_grip',mass,True,.9,True,type='capsule',size=[.0042],fromto=np.r_[a,b])
        else:
            # Rounded formed channel: one continuous visual, 3x3 convex patches.
            pts,norm=channel_points(np.linspace(xstations[0],xstations[-1],49),np.linspace(-math.pi/2,math.pi/2,25),side)
            v,f,vol=shell_grid(pts,norm*sheet)
            B.geom(arm,'steel',type='mesh',mesh=B.mesh(v,f,True))
            for xa,xb in zip(xstations[:-1],xstations[1:]):
                bands=[(-math.pi/2,math.pi/2)] if construction=='spring band' else [(-math.pi/2,-.70),(-.70,.70),(.70,math.pi/2)]
                for sa,sb in bands:
                    pts,norm=channel_points(np.linspace(xa,xb,7),np.linspace(sa,sb,7),side)
                    v,f,vol=shell_grid(pts,norm*sheet)
                    B.solid(arm,v,None,'steel',vol*7900,visual=False)
            # Two flanges on each channel form a nested clevis around the pin.
            # The channel starts beyond the circular bearing, leaving clearance
            # for the opposite arm's inner ears throughout the motion.
            theta=np.linspace(0,2*math.pi,33)[:-1]
            ear_radius=.008 if side<0 else .006
            outline=np.column_stack([ear_radius*np.cos(theta),ear_radius*np.sin(theta)])
            outline=np.vstack([outline,[[.023,side*.003],[.022,side*.009]]])
            outline=outline[ConvexHull(outline).vertices]
            if construction=='pivot':
                extent=arm_width*.5+(.0018 if side<0 else -.0005)
                zabs=np.full(len(outline),extent)
                for sign in [-1,1]:
                    z=sign*zabs
                    verts=np.vstack([np.column_stack([outline,z-.00065]),np.column_stack([outline,z+.00065])])
                    mass=ConvexHull(outline).volume*.0013*7900
                    B.solid(arm,verts,None,'steel',mass)
                if side<0:
                    # Rear fold of the outer channel joins its two cheeks.
                    # Three convex arc sectors leave clearance around the
                    # inner bearing circle, even at the full opening angle.
                    def hinge_fold(angles):
                        pts=np.array([[[.0082*math.cos(t),.0082*math.sin(t),z] for z in [-extent,extent]] for t in angles])
                        normals=np.array([[[math.cos(t),math.sin(t),0]]*2 for t in angles])
                        return shell_grid(pts,normals*.00065)
                    hv,hf,hvol=hinge_fold(np.linspace(math.pi/2,3*math.pi/2,49))
                    B.geom(arm,'steel',type='mesh',mesh=B.mesh(hv,hf,True))
                    for ta,tb in zip(np.linspace(math.pi/2,3*math.pi/2,4)[:-1],np.linspace(math.pi/2,3*math.pi/2,4)[1:]):
                        hv,hf,hvol=hinge_fold(np.linspace(ta,tb,9))
                        B.solid(arm,hv,None,'steel',hvol*7900,visual=False)
        if is_pasta:
            # One-piece serving comb: a continuous spine and five formed,
            # blunt-ended fingers. Every slot is an actual collision opening.
            spine_z=head_width*.5-.006
            sp=np.array([[[x,side*.003,z] for z in [spine_z-.002,head_width*.5]] for x in [start_head+.009,L]])
            v,f,vol=shell_grid(sp,[0,side*head_t,0])
            B.solid(arm,v,f,'head',vol*7900,mu,False,False)
            def neck_shape(xs):
                out=[]
                for x in xs:
                    u=(x-start_head)/.014
                    blend=u*u*(3-2*u)
                    lo=-arm_width*.325*(1-blend)+(spine_z-.002)*blend
                    hi=arm_width*.325*(1-blend)+head_width*.5*blend
                    out.append([[x,side*.003,lo],[x,side*.003,hi]])
                return np.array(out)
            for xa,xb in [(start_head,start_head+.007),(start_head+.007,start_head+.014)]:
                v,f,vol=shell_grid(neck_shape(np.linspace(xa,xb,9)),[0,side*head_t,0])
                B.solid(arm,v,None,'head',vol*7900,mu,False,False)
            centers=np.linspace(start_head+.014,L-.005,5)
            tooth_width=(centers[1]-centers[0])*.70
            # Form a single smooth visual shell: no coplanar overlap seams at
            # the prong roots, and rounded slots like a stamped comb.
            from shapely.geometry import Polygon, box
            from shapely.ops import unary_union
            import trimesh
            neck=neck_shape(np.linspace(start_head,start_head+.014,25))
            polygons=[Polygon(np.vstack([neck[:,0,[0,2]],neck[::-1,1,:][:,[0,2]]])),
                      box(start_head+.009,spine_z-.002,L,head_width*.5)]
            for center in centers:
                def finger(ts,rs):
                    return np.array([[[center+r*tooth_width*.5*(1-.20*t**3),side*(.003-.010*t*t),spine_z-t*(spine_z+head_width*.5)] for r in rs] for t in ts])
                outline=finger(np.linspace(0,1,33),[-1,1])
                polygons.append(Polygon(np.vstack([outline[:,0,:][:,[0,2]],outline[::-1,1,:][:,[0,2]]])))
                for ta,tb in [(0,.35),(.35,.70),(.70,1.)]:
                    v,f,vol=shell_grid(finger(np.linspace(ta,tb,7),[-1,1]),[0,side*head_t,0])
                    B.solid(arm,v,None,'head',vol*7900,mu,False,False)
                end=finger([1.],[-1.,1.]).reshape(2,3)
                end[:,1]+=side*head_t*.5
                B.primitive(arm,'head',7900,mu,type='capsule',size=[head_t*.60],fromto=end.reshape(-1))
            outline=unary_union(polygons).buffer(-.0006,quad_segs=4).buffer(.0012,quad_segs=4).buffer(-.0006,quad_segs=4)
            mesh=trimesh.creation.extrude_polygon(outline,height=head_t,engine='earcut')
            flat,faces=trimesh.remesh.subdivide_to_size(mesh.vertices,mesh.faces,max_edge=.004,max_iter=5)
            t=np.clip((spine_z-flat[:,1])/(spine_z+head_width*.5),0,1)
            blend=np.clip((flat[:,0]-start_head)/.014,0,1);blend=blend*blend*(3-2*blend)
            verts=np.column_stack([flat[:,0],side*(.003-.010*t*t*blend+flat[:,2]),flat[:,1]])
            if side>0:faces=faces[:,::-1]
            B.geom(arm,'head',type='mesh',mesh=B.mesh(verts,faces,True))
        else:
            # Continuous scalloped head visual, nine convex surface tiles, and
            # rounded overmold edge beads. Primitive beads also avoid single-point
            # mesh contact degeneracy when two perfectly mirrored rims meet.
            v,f,vol=head_shell(np.linspace(0,1,61),np.linspace(-1,1,25),side)
            name=B.mesh(v,f,True)
            B.geom(arm,'head',type='mesh',mesh=name)
            for u0,u1 in zip(np.linspace(0,1,4)[:-1],np.linspace(0,1,4)[1:]):
                for v0,v1 in [(-1,-.45),(-.45,.45),(.45,1)]:
                    tv,tf,tvol=head_shell(np.linspace(u0,u1,9),np.linspace(v0,v1,7),side)
                    B.solid(arm,tv,None,'head',tvol*head_density,mu,family!='steel',False)
            for vside in [-1,1]:
                edge_stations=np.linspace(0,1,4 if front_rim else 5)
                for u0,u1 in zip(edge_stations[:-1],edge_stations[1:]):
                    ends=headpoints([u0,u1],[vside],side).reshape(2,3)
                    ends[:,1]+=side*.0004
                    rad=bead_radius
                    volume=math.pi*rad**2*np.linalg.norm(ends[1]-ends[0])+4/3*math.pi*rad**3
                    B.geom(arm,'head',volume*head_density,True,mu,family!='steel',type='capsule',size=[rad],fromto=ends.reshape(-1))
            if front_rim:
                for va,vb in [(-1.,0.),(0.,1.)]:
                    ends=headpoints([1.],[va,vb],side).reshape(2,3)
                    ends[:,1]+=side*.0004
                    volume=math.pi*bead_radius**2*np.linalg.norm(ends[1]-ends[0])+4/3*math.pi*bead_radius**3
                    B.geom(arm,'head',volume*head_density,True,mu,family!='steel',type='capsule',size=[bead_radius],fromto=ends.reshape(-1))
            # One continuous rim follows both sides and the front. There are
            # no separate capped tube ends at the two front corners.
            rim=np.vstack([headpoints(np.linspace(0,1,81),[-1],side).reshape(-1,3),
                           headpoints([1.],np.linspace(-1,1,25)[1:],side).reshape(-1,3),
                           headpoints(np.linspace(1,0,81)[1:],[1],side).reshape(-1,3)])
            rim[:,1]+=side*.0004
            B.tube(arm,rim,bead_radius,'head')
            # Merge the molded/rolled edge into the cup shell, eliminating
            # intersecting visual surfaces and their jagged shadow seams.
            import trimesh
            visuals=[g for g in arm.findall('geom') if g.get('group')=='1' and g.get('material')=='head' and g.get('type')=='mesh']
            parts=[]
            for g in visuals:
                entry=B.asset.find(f"mesh[@name='{g.get('mesh')}']")
                mesh=trimesh.load(B.out/entry.get('file'),force='mesh')
                mesh.fix_normals()
                parts.append(mesh)
            merged=trimesh.boolean.union(parts,engine='manifold')
            # The shell and rolled edge are one solid material. Count the
            # overlap once using that actual union, not full intersecting
            # capsule volumes. Preserve shell masses and distribute only the
            # remaining material around the edge.
            head_geoms=[g for g in arm.findall('geom') if g.get('group')=='3' and g.get('material')=='head']
            shell_mass=sum(float(g.get('mass')) for g in head_geoms if g.get('type')=='mesh')
            rims=[g for g in head_geoms if g.get('type')=='capsule']
            old_rim_mass=sum(float(g.get('mass')) for g in rims)
            actual_head_volume=float(merged.volume)
            new_rim_mass=actual_head_volume*head_density-shell_mass
            if not merged.is_watertight or new_rim_mass<=0:
                raise ValueError('Cup union does not provide a valid material volume')
            for g in rims:g.set('mass',fmt(float(g.get('mass'))*new_rim_mass/old_rim_mass))
            B.mass+=new_rim_mass-old_rim_mass
            for g in visuals:arm.remove(g)
            B.geom(arm,'head',type='mesh',mesh=B.mesh(merged.vertices,merged.faces,True))
        # Curved steel reinforcement sits within the molded shell. The core's
        # incremental mass replaces the polymer already counted in this volume.
        if family!='steel':
            us=np.linspace(.015,.56,23)
            ps=headpoints(us,np.linspace(-.24,.24,7),side)
            for i,u in enumerate(us):
                for j,vcoord in enumerate(np.linspace(-.24,.24,7)):
                    ps[i,j,1]+=side*(head_t*.65-sheet*.5+head_relief(u,vcoord*hw(u)/(head_width*.5*widthfun(u))))
            cv,cf,cvol=shell_grid(ps,[0,side*sheet,0])
            B.solid(arm,cv,cf,'steel',cvol*(7900-head_density),.4,False,True)
        if construction=='scissor':
            grip_y=.0116+(.007 if flat_scissor else .0025)*25/math.hypot(25,20)
            element(arm,'site',name='grip_'+letter,pos=strap_grip if flat_scissor else [grip_x,-side*grip_y,side*.006],size=.002,rgba=[0,0,0,0])
        else:
            # Rounded tapered soft grip follows the formed web, not a floating block.
            gx0,gx1=.038,min(start_head-.012,grip_x+.052)
            def grip_patch(xs):
                ps,ns=[],[]
                for x in xs:
                    u=(x-gx0)/(gx1-gx0)
                    smax=.30+1.17*math.sin(math.pi*u)**.30
                    pts,norm=channel_points([x],np.linspace(-smax,smax,17),side,sheet+.00015)
                    thickness=.0005+.0024*math.sin(math.pi*u)**.45
                    ps.append(pts[0]);ns.append(norm[0]*thickness)
                return shell_grid(np.array(ps),np.array(ns))
            if grips:
                gv,gf,gvol=grip_patch(np.linspace(gx0,gx1,41))
                B.geom(arm,'grip',type='mesh',mesh=B.mesh(gv,gf,True))
                for g0,g1 in [(gx0,gx1)]:
                    gv,gf,gvol=grip_patch(np.linspace(g0,g1,13))
                    B.solid(arm,gv,None,'grip',gvol*1100,.95,True,False)
            grip_u=(grip_x-gx0)/(gx1-gx0)
            grip_y=float(yfun(grip_x))+sheet
            if grips:grip_y+=.00015+.0005+.0024*math.sin(math.pi*grip_u)**.45
            element(arm,'site',name='grip_'+letter,pos=[grip_x,side*grip_y,0],size=.002,rgba=[0,0,0,0])
        element(arm,'site',name='tip_'+letter,pos=[tip_x,side*tip_y,0],size=.002,rgba=[0,0,0,0])
        # Coil legs terminate against each arm; wire is visual only.
        if construction=='pivot':
            leg_wire=(k*.5*64*4*.009/186e9)**.25
            leg_span=4*leg_wire*1.1
            coil_end=np.array([.0045*math.cos(half),-side*.0045*math.sin(half),side*leg_span/2])
            B.tube(arm,[coil_end,[.015,side*.004,side*leg_span/2],[.026,side*.004,0]],leg_wire/2,'spring')
        elif construction=='scissor':
            leg_turns=2 if flat_scissor else 4
            leg_wire=(k*.5*64*leg_turns*.009/186e9)**.25
            leg_span=leg_turns*leg_wire*1.1
            coil_end=np.array([.0045*math.cos(half),-side*.0045*math.sin(half),side*leg_span/2])
            seat_z=.0028 if flat_scissor else .0035
            B.tube(arm,[coil_end,[.013,side*.001,side*leg_span/2],[.025,side*.001,side*seat_z]],leg_wire/2,'spring')
    wire=0.
    if construction!='spring band':
        # Hinge pin with domed heads; functional mass resides in the pin collision.
        pin_half=arm_width*.5+(.0033 if construction=='pivot' else .0025)
        if flat_scissor:pin_half=.0065
        B.primitive(B.obj,'polished',7900,type='cylinder',size=[.002,pin_half],pos=[0,0,0])
        for z in [-pin_half,pin_half]:
            B.geom(B.obj,'polished',type='ellipsoid',size=[.0034,.0034,.0008],pos=[0,0,z])
        turns=2 if flat_scissor else 4
        theta=np.linspace(0,turns*2*math.pi,129)
        wire=(k*.5*64*turns*.009/186e9)**.25
        span=turns*wire*1.1
        coil_points=np.column_stack([.0045*np.cos(theta),.0045*np.sin(theta),np.linspace(-span/2,span/2,len(theta))])
        B.tube(B.obj,coil_points,wire/2,'spring')
        # Explicit root inertia includes enclosed coil wire and rivet heads.
        # Visual geoms stay mass=0. Only the pin needs an exposed collision
        # surface; the coil sits inside the moving bearing housing.
        pin_mass=math.pi*.002**2*(2*pin_half)*7900
        pin_I=np.diag([pin_mass*(3*.002**2+4*pin_half**2)/12]*2+[pin_mass*.002**2/2])
        parts=[(pin_mass,np.zeros(3),pin_I)]
        head_mass=4/3*math.pi*.0034**2*.0008*7900
        head_I=np.diag([head_mass*(.0034**2+.0008**2)/5]*2+[head_mass*2*.0034**2/5])
        for z in [-pin_half,pin_half]:parts.append((head_mass,np.array([0.,0.,z]),head_I))
        for pa,pb in zip(coil_points[:-1],coil_points[1:]):
            direction=pb-pa;ell=np.linalg.norm(direction);direction/=ell
            mass=math.pi*(wire/2)**2*ell*7900
            transverse=mass*(3*(wire/2)**2+ell**2)/12
            tensor=np.eye(3)*transverse+(mass*(wire/2)**2/2-transverse)*np.outer(direction,direction)
            parts.append((mass,(pa+pb)/2,tensor))
        root_mass=sum(m for m,p,i in parts)
        centroid=sum(m*p for m,p,i in parts)/root_mass
        tensor=sum(i+m*(np.dot(p-centroid,p-centroid)*np.eye(3)-np.outer(p-centroid,p-centroid)) for m,p,i in parts)
        element(B.obj,'inertial',mass=root_mass,pos=centroid,fullinertia=[tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]])
        B.mass+=root_mass-pin_mass
        # The nested moving arm cheeks form the housing around the rivet.
    else:
        def band_patch(angles):
            points=np.array([[[-band_radius*math.cos(t),band_radius*math.sin(t),z] for z in [-arm_width/2,arm_width/2]] for t in angles])
            normals=np.array([[[-math.cos(t),math.sin(t),0]]*2 for t in angles])
            return shell_grid(points,normals*band_thickness)
        a0,a1=-math.pi/2+half,math.pi/2-half
        v,f,vol=band_patch(np.linspace(a0,a1,65))
        B.geom(B.obj,'steel',type='mesh',mesh=B.mesh(v,f,True))
        for ta,tb in zip(np.linspace(a0,a1,9)[:-1],np.linspace(a0,a1,9)[1:]):
            v,f,vol=band_patch(np.linspace(ta,tb,7))
            B.solid(B.obj,v,None,'steel',vol*7900,visual=False)
    # Slender sliding lock stem and open pull ring; non-colliding by contract.
    if lock=='pull ring':
        B.geom(B.obj,'steel',type='box',size=[.022,.0038,.0007],pos=[-.007,0,0])
        B.geom(B.obj,'grip',type='box',size=[.005,.004,.0017],pos=[-.023,0,0])
        theta=np.linspace(0,2*math.pi,65)
        ring=np.array([[[-.035+rx*math.cos(th),ry*math.sin(th),-.0017] for rx,ry in [(.008,.0058),(.013,.010)]] for th in theta])
        rv,rf,_=shell_grid(ring,[0,0,.0034])
        B.geom(B.obj,'grip',type='mesh',mesh=B.mesh(rv,rf,True))
    elif lock=='sliding collar':
        y=.016;z=arm_width*.5+.0025
        B.tube(B.obj,[[.020,-y,-z],[.020,y,-z],[.020,y,z],[.020,-y,z],[.020,-y,-z]],.00085,'polished')
    eq=element(B.root,'equality')
    element(eq,'joint',joint1='hinge_a',joint2='hinge_b',polycoef=[0,1,0,0,0],solref=[.004,1])
    contacts=element(B.root,'contact')
    # Bearing/shaft interfaces belong to the hinge constraint, not surface
    # contact. Explicitly exclude these even when the root is fixed to a stand.
    for letter in ['a','b']:
        element(contacts,'exclude',body1='object',body2='arm_'+letter)
    length,gap,closing=certify(B)
    (B.out/'model.xml').write_text(ET.tostring(B.root,encoding='unicode'))
    style=f'{head_profile}, stainless channels, '+('soft grips, ' if grips else 'bare handles, ')+('no lock' if lock=='none' else lock+' lock')
    if construction=='spring band':style=f'{head_profile}, one-piece stainless spring band, flat strips, no lock'
    if construction=='scissor':style=f'scissor serving tongs, '+('flat polished steel straps and finger loops, ' if flat_scissor else 'chrome steel rods and coated finger loops, ')+head_profile
    if detail!='smooth':style+=f', molded {detail}'
    params=dict(length_m=round(length,5),mass_kg=round(B.mass,5),rest_tip_gap_m=round(gap,5),closing_force_N=round(closing,2),tip_friction=round(mu,4),style=style,head_relief=detail,head_relief_height_m=relief_height,
                working_length_m=L,head_length_m=head_len,head_width_m=head_width,arm_width_m=arm_width,steel_thickness_m=sheet,channel_depth_m=channel_depth if construction=='pivot' else 0.,grip_distance_m=abs(grip_x),
                head_thickness_m=head_t,cup_depth_m=.010 if is_pasta else cup,scallop_fraction=scallop,scallop_count=lobes,spring_stiffness_Nm_per_rad=k,spring_preload_rad=preload,head_color_index=color_idx,
                spring_wire_diameter_m=wire,rest_half_angle_rad=half,head_material=family,head_profile=head_profile,lock_type=lock,soft_grips=grips,construction=construction,spring_band_radius_m=band_radius,spring_band_thickness_m=band_thickness,pasta_teeth=5 if is_pasta else 0,edge_radius_m=0. if is_pasta else bead_radius,hinge_damping_Nms_per_rad=hinge_damping)
    params['head_nose_fraction']=0. if is_pasta else nose_fraction
    params['scissor_arm_form']='flat strap' if flat_scissor else ('round rod' if is_scissor else 'not scissor')
    params['channel_section_power']=4 if construction=='pivot' else 0
    if is_scissor:
        params['arm_width_m']=.014 if flat_scissor else .005
        params['scissor_rod_diameter_m']=0. if flat_scissor else .005
        params['scissor_strap_thickness_m']=.0024 if flat_scissor else 0.
    params['spring_coil_turns']=turns if construction!='spring band' else 0
    if flat_scissor:params['soft_grips']=False
    params['longitudinal_cup_depth_m']=0. if is_pasta else longcup
    params.update(contact_metadata(family,head_density,.95 if grips and construction!='scissor' else .4))
    print(json.dumps(params))
    return params


def generate_bamboo(seed, out_dir):
    """Solid laminated strips; hinges approximate distributed arm flexure."""
    from PIL import Image
    rng=np.random.default_rng(seed ^ 0xD012)
    L=float(rng.choice([.22,.24,.30]))+float(rng.uniform(-.005,.005))
    width=float(rng.uniform(.018,.024))
    thick=float(rng.uniform(.0038,.0046))
    root_thick=float(np.random.default_rng(seed ^ 0xAA401).uniform(.0022,.0027))
    def thickness(x):
        return root_thick+(thick-root_thick)*np.asarray(x)/L
    density=float(rng.uniform(650,780))
    modulus=float(rng.uniform(8e9,12e9))
    target=float(rng.uniform(.045,.075))*math.sqrt(L/.24)
    mu=float(rng.uniform(.50,.80))
    shade=float(rng.uniform(.90,1.08))
    gripx=L*float(rng.uniform(.40,.48))
    tipx=L-.016;tipy=-.004;root_y=.003
    half=math.asin((target*.5-root_y)/math.hypot(tipx,tipy))-math.atan2(tipy,tipx)
    damping=.03*(L/.23)**2
    B=Builder(seed,out_dir)
    # Procedural longitudinal grain, never a downloaded photograph.
    tx=np.linspace(0,1,512)[:,None];tz=np.linspace(0,1,128)[None,:]
    grain=np.zeros((512,128))
    for frequency,amplitude in [(14,.035),(37,.022),(83,.012)]:
        phase=float(rng.uniform(0,2*math.pi))
        grain+=amplitude*np.sin(2*math.pi*(frequency*tz+.15*np.sin(2*math.pi*tx+phase))+phase)
    grain+=rng.normal(0,.006,grain.shape)
    rgb=np.clip(np.array([.76,.55,.30])[None,None,:]*shade*(1+grain[:,:,None]),0,1)
    Image.fromarray((rgb*255).astype('uint8')).save(B.out/'bamboo_grain.png')
    element(B.asset,'texture',name='bamboo_grain',type='2d',file='bamboo_grain.png')
    element(B.asset,'material',name='wood',texture='bamboo_grain',rgba=[1,1,1,1],specular=.08,shininess=.08)
    ys=PchipInterpolator([0,.055,L-.055,L-.025,L],[0,.002,.002,-.004,-.004])
    ws=PchipInterpolator([0,.055,L-.055,L-.025,L-.002,L],np.array([.75,1,.80,1.15,.90,.72])*width)
    # Match linear cantilever tip deflection under load at the actual grip.
    # delta_tip/F = integral (a-x)*(L-x)/(E*I(x)) dx; q=delta_tip/L.
    beam_x=np.linspace(0,gripx,257)
    inertia=ws(beam_x)*thickness(beam_x)**3/12
    integrand=(gripx-beam_x)*(L-beam_x)/(modulus*inertia)
    compliance=np.sum((integrand[:-1]+integrand[1:])*.5*np.diff(beam_x))
    k=gripx*L/compliance
    def strip(xs, side, heel=False):
        phi=np.linspace(0,2*math.pi,33)[:-1]
        c=np.sign(np.cos(phi))*np.abs(np.cos(phi))**.4
        s=np.sign(np.sin(phi))*np.abs(np.sin(phi))**.4
        vertices=[];uv=[]
        for x in xs:
            if heel:
                cy=0.;ry=.006;w=width*.75
            else:
                local_thick=float(thickness(x))
                cy=side*(float(ys(x))+local_thick*.5);ry=local_thick*.5;w=float(ws(x))
                if x>L-.002:ry*=1-.25*(x-L+.002)/.002
            for j in range(len(phi)):
                vertices.append([x,cy+ry*c[j],w*.5*s[j]])
                uv.append([(j/len(phi)),(x+.018)/(L+.018)])
        faces=[];n=len(phi)
        for i in range(len(xs)-1):
            for j in range(n):
                a=i*n+j;b=i*n+(j+1)%n
                faces.extend([(a,b,a+n),(b,b+n,a+n)])
        for j in range(1,n-1):
            faces.extend([(0,j+1,j),((len(xs)-1)*n,(len(xs)-1)*n+j,(len(xs)-1)*n+j+1)])
        v=np.array(vertices);f=np.array(faces)
        vol=np.sum(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])))/6
        if vol<0:f=f[:,::-1]
        return v,f,abs(vol),np.array(uv)
    # A glued solid heel joins the flexible strips; the joints carry the
    # reduced-order bending compliance and there is no rendered metal pin.
    v,f,vol,uv=strip(np.linspace(-.018,0,5),1,True)
    B.geom(B.obj,'wood',vol*density,True,mu,type='mesh',mesh=B.mesh(v,f))
    B.geom(B.obj,'wood',type='mesh',mesh=B.mesh(v,f,True,uv))
    for letter,side in [('a',-1),('b',1)]:
        arm=element(B.obj,'body',name='arm_'+letter,pos=[0,side*root_y,0],quat=[math.cos(half/2),0,0,side*math.sin(half/2)])
        element(arm,'joint',name='hinge_'+letter,type='hinge',axis=[0,0,-side],range=[0,half+.08],stiffness=k,springref=0,damping=damping,armature=.000002,solreflimit=[.002,1],solimplimit=[.99,.999,.0001])
        v,f,vol,uv=strip(np.r_[np.linspace(0,L-.002,79),L],side)
        B.geom(arm,'wood',type='mesh',mesh=B.mesh(v,f,True,uv))
        stations=np.unique(np.r_[np.linspace(0,L-.055,6),L-.04,L-.025,L-.012,L-.002,L])
        for xa,xb in zip(stations[:-1],stations[1:]):
            v,f,vol,uv=strip(np.linspace(xa,xb,5),side)
            B.geom(arm,'wood',vol*density,True,mu,type='mesh',mesh=B.mesh(v),margin=.0003)
        element(arm,'site',name='grip_'+letter,pos=[gripx,side*(float(ys(gripx))+float(thickness(gripx))),0],size=.002,rgba=[0,0,0,0])
        element(arm,'site',name='tip_'+letter,pos=[tipx,side*tipy,0],size=.002,rgba=[0,0,0,0])
    eq=element(B.root,'equality');element(eq,'joint',joint1='hinge_a',joint2='hinge_b',polycoef=[0,1,0,0,0],solref=[.004,1])
    contact=element(B.root,'contact')
    for letter in ['a','b']:element(contact,'exclude',body1='object',body2='arm_'+letter)
    length,gap,closing=certify(B)
    (B.out/'model.xml').write_text(ET.tostring(B.root,encoding='unicode'))
    params=dict(length_m=round(length,5),mass_kg=round(B.mass,5),rest_tip_gap_m=round(gap,5),closing_force_N=closing,tip_friction=mu,
                style='bamboo serving tongs, tapered solid strips, joined heel, flat rounded tips',head_material='bamboo',head_profile='flat bamboo paddle',construction='bamboo flexure',lock_type='none',soft_grips=False,
                working_length_m=L,head_length_m=.04,head_width_m=width*1.15,arm_width_m=width,grip_distance_m=gripx,rest_half_angle_rad=half,
                wood_thickness_m=thick,wood_root_thickness_m=root_thick,wood_density_kg_m3=density,wood_modulus_Pa=modulus,wood_tone=shade,spring_stiffness_Nm_per_rad=k,spring_preload_rad=0.,hinge_damping_Nms_per_rad=damping)
    params.update(contact_metadata('bamboo',density,mu))
    params['wood_grain']='seeded longitudinal waves and fine noise'
    print(json.dumps(params));return params


def generate_molded(seed, out_dir):
    """PBT loop flexure with overmolded silicone paddles.

    Equivalent spring combines root-arc rotation and arm bending compliance.
    It is a room-temperature elastic approximation, not a creep model.
    """
    rng=np.random.default_rng(seed ^ 0x171AF)
    L=float(rng.uniform(.250,.290))
    width=float(rng.uniform(.019,.023))
    headw=float(rng.uniform(.029,.036))
    headlen=float(rng.uniform(.052,.065))
    radius=float(rng.uniform(.012,.016))
    heel_t=float(rng.uniform(.0025,.0031))
    arm_t=float(rng.uniform(.0045,.0055))
    head_t=float(rng.uniform(.0040,.0050))
    E=float(rng.uniform(2.2e9,2.7e9))
    density=1300.;mu=float(rng.uniform(.85,1.15))
    target=float(rng.uniform(.055,.078))
    gx=L*float(rng.uniform(.39,.46))
    palette=int(rng.integers(0,4))
    colors=[[.12,.135,.14,1],[.77,.74,.61,1],[.12,.34,.16,1],[.19,.20,.21,1]]
    headcolors=[[.24,.46,.14,1],[.77,.74,.61,1],[.12,.34,.16,1],[.62,.055,.035,1]]
    start=L-headlen
    tx=start+.54*headlen
    ys=PchipInterpolator([0,.035,L*.52,start,L],[0,.002,.002,0,-.003])
    ws=PchipInterpolator([0,.035,L*.52,start,L-.015,L],
                         [width,width*1.06,width*.90,headw*.78,headw,headw*.72])
    ts=PchipInterpolator([0,.035,start,L],[heel_t,arm_t,head_t,head_t*.72])
    relief=.00065
    ty=float(ys(tx))-relief
    half=.10
    for _ in range(10):
        half=math.asin((target*.5-radius*math.cos(half))/math.hypot(tx,ty))-math.atan2(ty,tx)
    # Use an accessible squeeze point on the real outer handle, including
    # the wide heel offset and the molded section thickness. Moving the load
    # point also changes the equivalent beam rate computed below.
    def outer_grip_y(x):
        return radius*math.cos(half)+x*math.sin(half)+(float(ys(x))+float(ts(x)))*math.cos(half)
    if outer_grip_y(gx)>.0325:
        lo,hi=0.,gx
        for _ in range(32):
            mid=(lo+hi)/2
            if outer_grip_y(mid)>.0325:hi=mid
            else:lo=mid
        gx=lo
    # Two mirror-coordinate hinges each rotate q, so the loop's relative
    # rotation is 2q. Its rate per arm is 2EI / arc_length.
    arc_length=radius*(math.pi-2*half)
    loop_k=2*E*width*heel_t**3/(12*arc_length)
    xs=np.linspace(0,gx,257)
    integrand=(gx-xs)*(L-xs)/(E*ws(xs)*ts(xs)**3/12)
    compliance=np.sum((integrand[:-1]+integrand[1:])*.5*np.diff(xs))
    arm_k=gx*L/compliance
    k=1/(1/loop_k+1/arm_k)
    damping=.035*(L/.27)**2
    B=Builder(seed,out_dir)
    B.material('plastic',colors[palette],.23,.20)
    B.material('head',headcolors[palette],.13,.12)

    def section_mesh(xs,side):
        phi=np.linspace(0,2*math.pi,33)[:-1]
        cy=np.sign(np.cos(phi))*abs(np.cos(phi))**.35
        cz=np.sign(np.sin(phi))*abs(np.sin(phi))**.35
        v=[]
        for x in xs:
            thick=float(ts(x));w=float(ws(x));y=float(ys(x))
            for a,b in zip(cy,cz):
                v.append([x,side*(y+thick*.5)+thick*.5*a,w*.5*b])
        v=np.array(v);f=[];n=len(phi)
        for i in range(len(xs)-1):
            for j in range(n):
                a=i*n+j;b=i*n+(j+1)%n
                f.extend([(a,b,a+n),(b,b+n,a+n)])
        for j in range(1,n-1):
            f.extend([(0,j+1,j),((len(xs)-1)*n,(len(xs)-1)*n+j,(len(xs)-1)*n+j+1)])
        f=np.array(f);vol=np.sum(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])))/6
        if vol<0:f=f[:,::-1]
        return v,f,abs(vol)

    for letter,side in [('a',-1),('b',1)]:
        arm=element(B.obj,'body',name='arm_'+letter,pos=[-radius*math.sin(half),side*radius*math.cos(half),0],
                    quat=[math.cos(half/2),0,0,side*math.sin(half/2)])
        element(arm,'joint',name='hinge_'+letter,type='hinge',axis=[0,0,-side],range=[0,half+.14],
                stiffness=k,springref=0,damping=damping,armature=.000002,
                solreflimit=[.002,1],solimplimit=[.99,.999,.0001])
        for lo,hi,mat,rho in [(0,start,'plastic',density),(start,L,'head',1100.)]:
            v,f,vol=section_mesh(np.linspace(lo,hi,65),side)
            B.geom(arm,mat,type='mesh',mesh=B.mesh(v,f,True))
            for xa,xb in zip(np.linspace(lo,hi,7)[:-1],np.linspace(lo,hi,7)[1:]):
                v,f,vol=section_mesh(np.linspace(xa,xb,7),side)
                B.geom(arm,mat,vol*rho,True,mu if mat=='head' else .55,mat=='head',
                       type='mesh',mesh=B.mesh(v),margin=.0003)
        # Broad, rounded silicone gripping ribs are actual contact geometry.
        for u in [.15,.28,.41,.54,.67,.80]:
            x=start+u*headlen;y=side*float(ys(x));z=float(ws(x))*.40
            # Half of each round rib is embedded in the already-counted head.
            ribvol=(math.pi*relief**2*(2*z)+4/3*math.pi*relief**3)*.5
            B.geom(arm,'head',ribvol*1100.,True,mu,True,type='capsule',size=[relief],fromto=[x,y,-z,x,y,z])
            B.geom(arm,'head',type='capsule',size=[relief],fromto=[x,y,-z,x,y,z])
        # The molded body continues inside the soft paddles as a PBT core.
        # Add only replacement density: the silicone volume already exists.
        for lo,hi in [(start,(start+L-.008)/2),((start+L-.008)/2,L-.008)]:
            p=np.array([[[x,side*(float(ys(x))+.0012),z*float(ws(x))*.30] for z in [-1,1]] for x in np.linspace(lo,hi,13)])
            v,f,vol=shell_grid(p,[0,side*.0015,0])
            B.solid(arm,v,None,'plastic',vol*(density-1100.),.55,False,False)
            B.geom(arm,'plastic',type='mesh',mesh=B.mesh(v,f,True))
        element(arm,'site',name='grip_'+letter,pos=[gx,side*(float(ys(gx))+float(ts(gx))),0],size=.002,rgba=[0,0,0,0])
        element(arm,'site',name='tip_'+letter,pos=[tx,side*ty,0],size=.002,rgba=[0,0,0,0])

    def heel(angles):
        p=np.array([[[-radius*math.cos(t),radius*math.sin(t),z] for z in [-width/2,width/2]] for t in angles])
        n=np.array([[[-math.cos(t),math.sin(t),0]]*2 for t in angles])
        return shell_grid(p,n*heel_t)
    a0,a1=-math.pi/2+half,math.pi/2-half
    v,f,vol=heel(np.linspace(a0,a1,65))
    B.geom(B.obj,'plastic',type='mesh',mesh=B.mesh(v,f,True))
    cuts=np.linspace(a0,a1,9)
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        v,f,vol=heel(np.linspace(lo,hi,9))
        B.solid(B.obj,v,None,'plastic',vol*density,.55,False,False)
    eq=element(B.root,'equality')
    element(eq,'joint',joint1='hinge_a',joint2='hinge_b',polycoef=[0,1,0,0,0],solref=[.004,1])
    contact=element(B.root,'contact')
    for letter in ['a','b']:element(contact,'exclude',body1='object',body2='arm_'+letter)
    length,gap,closing=certify(B)
    (B.out/'model.xml').write_text(ET.tostring(B.root,encoding='unicode'))
    params=dict(length_m=round(length,5),mass_kg=round(B.mass,5),rest_tip_gap_m=round(gap,5),closing_force_N=closing,tip_friction=mu,
                style='molded PBT loop-flexure tongs, rounded silicone paddle tips with transverse ribs',
                head_material='silicone',head_profile='ribbed molded paddle',construction='molded flexure',lock_type='none',soft_grips=False,
                working_length_m=L,head_length_m=headlen,head_width_m=headw,arm_width_m=width,grip_distance_m=gx,rest_half_angle_rad=half,
                plastic_heel_radius_m=radius,plastic_heel_thickness_m=heel_t,plastic_arm_thickness_m=arm_t,head_thickness_m=head_t,
                plastic_modulus_Pa=E,plastic_density_kg_m3=density,plastic_palette=palette,plastic_core_thickness_m=.0015,head_relief='ribs',head_relief_height_m=relief,
                spring_stiffness_Nm_per_rad=k,spring_preload_rad=0.,hinge_damping_Nms_per_rad=damping)
    params.update(contact_metadata('silicone',1100.,.55))
    print(json.dumps(params));return params
