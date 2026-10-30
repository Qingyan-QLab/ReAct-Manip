"""Coupled dimensions for the constrained-opening laptop task (metres)."""
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np


@dataclass(frozen=True)
class DrawerGeometry:
    width: float = .425
    depth: float = .365
    height: float = .11
    width_gap: float = .060
    depth_gap: float = .060
    thickness: float = .030
    mass: float = .5
    opening_ratio: float = .96
    max_open: float | None = None
    front: float = -.05
    top: float = .80
    foot_height: float = .010
    slide_friction_n: float = .05

    def __post_init__(self):
        if not np.isfinite(self.slide_friction_n) or self.slide_friction_n < 0:
            raise ValueError('drawer slide friction must be nonnegative and finite')
        values = [self.width, self.depth, self.height, self.width_gap, self.depth_gap,
                  self.thickness, self.mass, self.opening_ratio]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('drawer/laptop dimensions must be positive and finite')
        # Keep saved tight-fit scenarios valid; new defaults enlarge the drawer
        # by 35 mm on each axis without changing the derived laptop size.
        if not (.34 <= self.width <= .475 and .28 <= self.depth <= .415 and .07 <= self.height <= .13):
            raise ValueError('drawer dimensions outside supported envelope')
        if not (.015 <= self.width_gap <= .070 and .015 <= self.depth_gap <= .070):
            raise ValueError('laptop edge clearances must be 0.015–0.070 m total')
        # Retain support for saved heavier scenarios; new profiles use a
        # lightweight laptop so grasping strength is not the main difficulty.
        if not (.012 <= self.thickness <= .036 and .35 <= self.mass <= 1.8):
            raise ValueError('invalid laptop thickness or mass')
        if not np.isfinite(self.opening) or not .85 * self.laptop_size[0] <= self.opening <= self.depth:
            raise ValueError('opening outside conservative v2 envelope; do not reuse v1 max_open blindly')

    @property
    def laptop_size(self):
        return (self.depth - self.depth_gap, self.width - self.width_gap, self.thickness)

    @property
    def opening(self):
        return float(self.max_open if self.max_open is not None else min(self.depth, self.opening_ratio * self.laptop_size[0]))

    @property
    def floor(self):
        return self.top - .04 - self.height


def geometry_from_scenario(scenario):
    p = scenario.parameters if scenario is not None else {}
    d, l = p.get('drawer', {}), p.get('laptop', {})
    g = DrawerGeometry(width=float(d.get('width_m', .425)), depth=float(d.get('depth_m', .365)),
        height=float(d.get('height_m', .11)), opening_ratio=float(d.get('opening_ratio', .96)),
        max_open=None if 'max_open_m' not in d else float(d['max_open_m']),
        width_gap=float(l.get('width_gap_m', .060)), depth_gap=float(l.get('depth_gap_m', .060)),
        thickness=float(l.get('thickness_m', .030)), mass=float(l.get('mass_kg', .5)),
        slide_friction_n=float(d.get('slide_friction_n', .05)))
    from .feasibility import screen_insertion
    if not screen_insertion(g):
        raise ValueError('drawer/laptop configuration failed object insertion path screen')
    return g


def vec(v): return ' '.join(str(float(x)) for x in v)


def build_drawer(arena, g):
    """True inner dimensions; closed front is flush with the desk edge."""
    old = arena.worldbody.find("body[@name='drawer']")
    if old is not None: arena.worldbody.remove(old)
    def solid(parent, name, pos, size, material='metal_mat'):
        a = dict(type='box', pos=vec(pos), size=vec(size), material=material, friction='.6 .01 .001', solref='.005 1')
        ET.SubElement(parent, 'geom', name=name, group='0', **a)
        ET.SubElement(parent, 'geom', name=name+'_vis', group='1', contype='0', conaffinity='0', density='0', **a)
    # The tabletop is the fixed roof: insertion cannot bypass it from above.
    for name in ('table_top', 'table_top_vis'):
        geom = arena.worldbody.find(f"geom[@name='{name}']")
        geom.set('pos', '.30 0 .78'); geom.set('size', '.35 .55 .02')
    for sign in (-1, 1):
        solid(arena.worldbody, 'cabinet_side_'+str(sign), (g.front+g.depth/2+.008, sign*(g.width/2+.019), g.floor+g.height/2), (g.depth/2+.018,.007,g.height/2))
    solid(arena.worldbody, 'cabinet_back', (g.front+g.depth+.03,0,g.floor+g.height/2), (.008,g.width/2+.026,g.height/2))
    drawer = ET.SubElement(arena.worldbody, 'body', name='drawer', pos=vec((g.front+.008,0,g.floor)))
    ET.SubElement(drawer, 'joint', name='drawer_slide', type='slide', axis='1 0 0', range=f'{-g.opening} 0', damping='.4', frictionloss=str(g.slide_friction_n), armature='.03')
    solid(drawer,'drawer_floor',(g.depth/2,0,-.004),(g.depth/2+.008,g.width/2+.008,.004))
    solid(drawer,'drawer_front',(-.004,0,g.height/2-.003),(.004,g.width/2+.008,g.height/2-.003))
    solid(drawer,'drawer_back',(g.depth+.004,0,g.height/2-.003),(.004,g.width/2+.008,g.height/2-.003))
    for sign, name in ((-1,'neg'),(1,'pos')):
        solid(drawer,'drawer_side_'+name,(g.depth/2,sign*(g.width/2+.004),g.height/2-.003),(g.depth/2,.004,g.height/2-.003))
    solid(drawer,'drawer_handle',(-.065,0,g.height*.55),(.01,.11,.01))
    for sign in (-1,1):
        solid(drawer,'drawer_handle_post_'+str(sign),(-.035,sign*.09,g.height*.55),(.027,.006,.006))
    ET.SubElement(drawer,'site',name='drawer_handle_site',pos=vec((-.065,0,g.height*.55)),size='.005',rgba='0 0 0 0')


def bevel_laptop(obj, g):
    """Rounded lower edge and corners, shared collision/visual convex mesh."""
    hx, hy, hz = np.array(g.laptop_size)/2
    vertices = []
    radius = min(.012, g.thickness*.45)
    rings = [(-hz+radius*(1-np.cos(a)), radius*(1-np.sin(a)))
             for a in np.linspace(0,np.pi/2,9)] + [(hz,0.)]
    corner = .006
    for z, inset in rings:
        for sx,sy,start in ((1,1,0),(-1,1,90),(-1,-1,180),(1,-1,270)):
            for a in np.deg2rad(np.linspace(start,start+90,5)):
                vertices.append((sx*(hx-inset-corner)+corner*np.cos(a),
                                 sy*(hy-inset-corner)+corner*np.sin(a),z))
    mesh = obj.naming_prefix+'beveled_shell'
    ET.SubElement(obj.asset,'mesh',name=mesh,vertex=vec(np.asarray(vertices).ravel()))
    body = obj.get_obj()
    for geom in body.iter('geom'):
        geom.set('type','mesh'); geom.set('mesh',mesh); geom.attrib.pop('size',None)
        if geom.get('group','0') == '0':
            geom.attrib.pop('density',None); geom.set('mass',str(g.mass-.02))
    for i,(x,y) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
        attrs=dict(type='cylinder',pos=vec((x*(hx-.025),y*(hy-.025),-hz-g.foot_height/2)),size=f'.009 {g.foot_height/2}',rgba='.05 .05 .05 1',friction='.8 .01 .001')
        ET.SubElement(body,'geom',name=obj.naming_prefix+f'foot{i}',group='0',mass='.005',**attrs)
        ET.SubElement(body,'geom',name=obj.naming_prefix+f'foot{i}_vis',group='1',contype='0',conaffinity='0',mass='0',**attrs)
