"""Articulated metal rods; the legacy elastic cable remains loadable for replay."""
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np

BAR = np.array([.72, -.25, 1.0])
BAR_RADIUS = .035
BAR_HALF_LENGTH = .30
TABLE_Z = .65
TABLE_CENTER = np.array([.55, .27, .63])
TABLE_HALF_SIZE = np.array([.43, .22, .02])
START = np.array([.20, .22, TABLE_Z + .013])


def vec(v):
    return ' '.join(f'{x:.10g}' for x in v)


@dataclass(frozen=True)
class Rope:
    elastic = True
    cap_inset = 0.
    color = '.85 .30 .10 1'
    length: float = .70
    radius: float = .012
    segments: int = 48
    mass: float = .24
    heavy_center: float = .30
    heavy_width: float = .25
    heavy_ratio: float = 2.5
    friction: float = .15
    bend: float = 20000.
    twist: float = 10000.
    damping: float = .001

    def __post_init__(self):
        values = [self.length, self.radius, self.mass, self.heavy_center,
                  self.heavy_width, self.heavy_ratio, self.friction, self.bend,
                  self.twist, self.damping]
        if not np.isfinite(values).all() or not isinstance(self.segments, int):
            raise ValueError('Rope parameters must be finite; segments must be an integer')
        if not (24 <= self.segments <= 160 and .5 <= self.length <= 1.6 and
                .008 <= self.length/self.segments <= .030 and
                .008 <= self.radius <= .016 and .1 <= self.mass <= .5 and
                .2 <= self.heavy_center <= .8 and .15 <= self.heavy_width <= .35 and
                1 <= self.heavy_ratio <= 3 and .05 <= self.friction <= 1 and
                0 < self.bend <= 1e5 and 0 < self.twist <= 1e5 and 0 <= self.damping <= .01):
            raise ValueError('Rope parameters outside supported construction bounds')

    @property
    def ds(self):
        return self.length / self.segments

    @property
    def masses(self):
        s = (np.arange(self.segments) + .5) / self.segments
        weights = 1 + (self.heavy_ratio - 1) * np.exp(-.5*((s-self.heavy_center)/(self.heavy_width/2))**2)
        return self.mass * weights / weights.sum()

    @property
    def median_index(self):
        return int(np.searchsorted(np.cumsum(self.masses), self.mass/2))

    def add_to(self, root, start=START):
        if self.elastic:
            extension = root.find('extension')
            if extension is None:
                extension = ET.SubElement(root, 'extension')
            plugin = ET.SubElement(extension, 'plugin', plugin='mujoco.elasticity.cable')
            instance = ET.SubElement(plugin, 'instance', name='rope_elasticity')
            for key, value in [('bend', self.bend), ('twist', self.twist), ('flat', 'true')]:
                ET.SubElement(instance, 'config', key=key, value=str(value))
        contact = root.find('contact')
        if contact is None:
            contact = ET.SubElement(root, 'contact')
        parent = root.find('worldbody')
        for i, mass in enumerate(self.masses):
            body = ET.SubElement(parent, 'body', name=f'rope_{i}',
                                 pos=vec(start if i == 0 else [self.ds, 0, 0]))
            shaft=self.ds-2*self.cap_inset
            cylinder_mass=mass*shaft/(shaft+4*self.radius/3)
            sphere_mass=mass-cylinder_mass
            axial=(.5*cylinder_mass+.4*sphere_mass)*self.radius**2
            transverse=cylinder_mass*(3*self.radius**2+shaft**2)/12 + sphere_mass*(.4*self.radius**2+.25*shaft**2+.375*shaft*self.radius)
            ET.SubElement(body, 'inertial', pos=vec([self.ds/2,0,0]), mass=str(mass),
                          diaginertia=vec([axial,transverse,transverse]))
            if i == 0:
                ET.SubElement(body, 'freejoint', name='rope_root')
            else:
                ET.SubElement(body, 'joint', name=f'rope_joint_{i}', type='ball', damping=str(self.damping))
            color=self.color
            if not self.elastic:
                if i==0:
                    color='.9 .08 .06 1'
                elif i==self.segments-1:
                    color='.06 .25 .95 1'
            ET.SubElement(body, 'geom', name=f'rope_geom_{i}', type='capsule',
                          fromto=vec([self.cap_inset,0,0,self.ds-self.cap_inset,0,0]), size=str(self.radius), mass=str(mass),
                          rgba=color, group='1', friction=f'{self.friction} .001 .0001',
                          condim='3', solref='.008 1', solimp='.95 .99 .001')
            ET.SubElement(body, 'site', name=f'rope_node_{i}', size='.002', rgba='0 0 0 0')
            if self.elastic:
                ET.SubElement(body, 'plugin', instance='rope_elasticity')
            # Capsules overlap near joints. Exclude local neighbours only;
            # nonlocal self contact remains physical.
            neighbours=int(np.ceil(2*self.radius/self.ds))+1 if self.elastic else 1
            for j in range(max(0, i-neighbours), i):
                ET.SubElement(contact, 'exclude', body1=f'rope_{j}', body2=f'rope_{i}')
            parent = body
        ET.SubElement(parent, 'site', name=f'rope_node_{self.segments}', pos=vec([self.ds,0,0]),
                      size='.002', rgba='0 0 0 0')


@dataclass(frozen=True)
class RodChain:
    """Rigid weighted rods joined at endpoints by invisible, free ball joints."""
    length: float = 1.
    radius: float = .012
    segments: int = 10
    link_masses: tuple = (.012,.016,.018,.020,.022,.026,.028,.030,.032,.036)
    friction: float = .05
    damping: float = .001
    elastic = False
    color = '.55 .58 .62 1'

    def __post_init__(self):
        masses=np.asarray(self.link_masses,dtype=float)
        if (not isinstance(self.segments,int) or isinstance(self.segments,bool)
                or not 3<=self.segments<=160 or masses.shape!=(self.segments,)):
            raise ValueError('chain.link_masses must contain one mass per rod (3–160 rods)')
        if not np.isfinite(masses).all() or np.any(masses<=0) or not .05<=masses.sum()<=1.5:
            raise ValueError('Rod masses must be finite and positive; total must be 0.05–1.5 kg')
        values=[self.length,self.radius,self.friction,self.damping]
        if (not np.isfinite(values).all() or not .3<=self.length<=2.
                or not .004<=self.radius<=.016 or not .0005<=self.friction<=1.
                or not 0<=self.damping<=.01 or not 2*self.radius<self.ds<=.25):
            raise ValueError('Rod chain geometry/contact parameters outside bounds; each capsule must be longer than its diameter')
        object.__setattr__(self,'link_masses',tuple(float(x) for x in masses))

    @property
    def ds(self):
        return self.length/self.segments

    @property
    def cap_inset(self):
        # Rounded rod ends reach the virtual pivots: each rod is exactly ds long.
        return self.radius

    @property
    def masses(self):
        return np.asarray(self.link_masses)

    @property
    def mass(self):
        return float(sum(self.link_masses))

    median_index=Rope.median_index

    def add_to(self, root, start=START):
        # Harder tangential contact constraints reduce slow numerical grip slip.
        # This leaves friction coefficients, contact geometry and forces unchanged.
        root.find('option').set('impratio', '100')
        Rope.add_to(self, root, start)


def add_fixture(root):
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='bar', type='cylinder', pos=vec(BAR),
                  quat='.7071067812 .7071067812 0 0', size=f'{BAR_RADIUS} {BAR_HALF_LENGTH}',
                  rgba='.32 .37 .42 1', group='1', friction='.15 .001 .0001', condim='3')
    for i, y in enumerate((BAR[1]-BAR_HALF_LENGTH-.025, BAR[1]+BAR_HALF_LENGTH+.025)):
        ET.SubElement(world, 'geom', name=f'stand_{i}', type='box',
                      pos=vec([BAR[0],y,(BAR[2]+.04)/2]), size=vec([.025,.025,(BAR[2]+.04)/2]),
                      rgba='.25 .28 .32 1', group='1')
    ET.SubElement(world, 'geom', name='table_top', type='box', pos=vec(TABLE_CENTER),
                  size=vec(TABLE_HALF_SIZE), material='wood_mat', group='1')
    for i, x in enumerate((.2,.9)):
        ET.SubElement(world, 'geom', name=f'table_leg_{i}', type='box', pos=f'{x} .27 .31',
                      size='.025 .14 .31', rgba='.3 .3 .3 1', group='1')
