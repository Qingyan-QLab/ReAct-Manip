"""Conservative 2-D object-path screen, not a robot reachability guarantee."""
from functools import lru_cache
import numpy as np
from scipy.ndimage import label


@lru_cache(maxsize=128)
def screen_insertion(g, allow_tilt=True, return_path=False):
    """Connected collision-free pose lattice at full opening.

    Includes laptop feet in a bounding rectangle; drawer front/back/floor,
    cabinet back and tabletop roof are obstacles. Y is aligned at zero.
    Discrete screening does not replace continuous MuJoCo/robot path tests.
    """
    f = g.front+.008-g.opening
    h = g.thickness+g.foot_height
    xs = np.arange(f-.18, g.front+g.depth+.05, .004)
    zs = np.arange(g.floor+h/2+.001, g.top+.32, .004)
    angles = np.deg2rad(np.arange(-80,81,4)) if allow_tilt else np.array([0.])
    x,z,a = np.meshgrid(xs,zs,angles,indexing='ij')
    c,s = np.cos(a),np.sin(a)
    hx,hz = g.laptop_size[0]/2,h/2
    free = np.ones(x.shape,dtype=bool)
    obstacles = [(f+g.depth/2,g.floor-.004,g.depth/2+.008,.004),
        (f-.004,g.floor+g.height/2-.003,.004,g.height/2-.003),
        (f+g.depth+.004,g.floor+g.height/2-.003,.004,g.height/2-.003),
        (.30,.78,.35,.02),
        (g.front+g.depth+.03,g.floor+g.height/2,.008,g.height/2)]
    for ox,oz,bx,bz in obstacles:
        dx,dz=x-ox,z-oz
        overlap=(np.abs(dx)<hx*np.abs(c)+hz*np.abs(s)+bx+.0005)&(np.abs(dz)<hx*np.abs(s)+hz*np.abs(c)+bz+.0005)
        overlap &= (np.abs(dx*c+dz*s)<hx+bx*np.abs(c)+bz*np.abs(s)+.0005)
        overlap &= (np.abs(-dx*s+dz*c)<hz+bx*np.abs(s)+bz*np.abs(c)+.0005)
        free &= ~overlap
    components,_ = label(free)
    target=(int(np.argmin(abs(xs-(f+g.depth/2)))),0,int(np.argmin(abs(angles))))
    component=int(components[target])
    feasible = bool(component and np.any(components[:,-1,:]==component))
    if not return_path:
        return feasible
    if not feasible:
        return None
    from collections import deque
    start = np.ravel_multi_index(target,free.shape)
    parents = np.full(free.size,-1,dtype=np.int32)
    parents[start] = start
    queue = deque([start])
    nz,na = len(zs),len(angles)
    valid = (components==component).ravel()
    while queue:
        node=queue.popleft()
        ix, rem=divmod(node,nz*na); iz,ia=divmod(rem,na)
        if iz==nz-1:
            path=[]
            while True:
                i,j,k=np.unravel_index(node,free.shape)
                path.append((xs[i],zs[j],angles[k]))
                if parents[node]==node: return np.asarray(path)
                node=int(parents[node])
        neighbors=[]
        if ix>0: neighbors.append(node-nz*na)
        if ix<len(xs)-1: neighbors.append(node+nz*na)
        if iz>0: neighbors.append(node-na)
        if iz<nz-1: neighbors.append(node+na)
        if ia>0: neighbors.append(node-1)
        if ia<na-1: neighbors.append(node+1)
        for nxt in neighbors:
            if valid[nxt] and parents[nxt]<0:
                parents[nxt]=node; queue.append(nxt)
