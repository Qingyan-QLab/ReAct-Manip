"""Shared procedural collision geometry for calibration and the robot task."""
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np

BALL_RADIUS = 0.025
BALL_MASS = 0.10
RAIL_RADIUS = 0.006
HALF_GAP = 0.018
TRACK_Y = -0.18
CUP_CENTER = np.array([1.10, TRACK_Y, 0.43])
CUP_BASE_HEIGHT = float(CUP_CENTER[2])
CUP_RADIUS = 0.05
CUP_HEIGHT = 0.17
RELEASE_RANGE = (0.32, 0.57)


def values(a):
    return ' '.join(f'{v:.9g}' for v in a)


@dataclass(frozen=True)
class Cup:
    """Resolved cup geometry for one task instance.

    The instance variable is the cup base elevation.  Horizontal position,
    radius, and wall height remain part of the fixed task geometry.
    """

    base_height_m: float = CUP_BASE_HEIGHT
    center_x: float = float(CUP_CENTER[0])
    center_y: float = float(CUP_CENTER[1])
    radius: float = CUP_RADIUS
    height: float = CUP_HEIGHT

    def __post_init__(self):
        if not 0.39 <= self.base_height_m <= 0.47:
            raise ValueError('Cup base height outside calibrated family: [0.39,0.47] m')
        if self.radius != CUP_RADIUS or self.height != CUP_HEIGHT:
            raise ValueError('This task uses fixed cup radius and wall height')

    @property
    def center(self):
        return np.array([self.center_x, self.center_y, self.base_height_m], dtype=float)

    @property
    def mouth_height(self):
        return self.base_height_m + self.height


@dataclass(frozen=True)
class Track:
    start_height: float = 0.97
    exit_angle_deg: float = 25.0
    segments: int = 160

    def __post_init__(self):
        if not (0.91 <= self.start_height <= 1.03 and 18 <= self.exit_angle_deg <= 32):
            raise ValueError('Track outside calibrated family: height [0.91,1.03], angle [18,32]')
        if self.segments < 40:
            raise ValueError('At least 40 rail segments required')

    def height(self, x):
        # Cubic Hermite segments, matching zero slope at the valley.
        x = np.asarray(x)
        left = x <= .68
        a, b = np.where(left, .28, .68), np.where(left, .68, .84)
        z0, z1 = np.where(left, self.start_height, .64), np.where(left, .64, .69)
        m0 = np.where(left, -1.0, 0.0)
        m1 = np.where(left, 0., np.tan(np.deg2rad(self.exit_angle_deg)))
        t = (x-a)/(b-a)
        return (2*t**3-3*t**2+1)*z0 + (t**3-2*t**2+t)*(b-a)*m0 + (-2*t**3+3*t**2)*z1 + (t**3-t**2)*(b-a)*m1

    def slope(self, x):
        h = 1e-5
        return float((self.height(x+h)-self.height(x-h))/(2*h))

    def ball_position(self, x, clearance=0.001):
        if not RELEASE_RANGE[0] <= x <= RELEASE_RANGE[1]:
            raise ValueError(f'Release x must be in {RELEASE_RANGE}')
        slope = self.slope(x)
        normal = np.array([-slope, 0, 1])/np.sqrt(1+slope*slope)
        offset = np.sqrt((BALL_RADIUS+RAIL_RADIUS)**2-HALF_GAP**2)
        return np.array([x, TRACK_Y, self.height(x)]) + normal*(offset+clearance)

    def add_to(self, root, cup=None):
        cup = cup or Cup()
        world = root.find('worldbody')
        contact = root.find('contact')
        if contact is None:
            contact = ET.SubElement(root, 'contact')
        xs = np.linspace(.28, .84, self.segments+1)
        body = ET.SubElement(world, 'body', name='track')
        for side, y in enumerate((TRACK_Y-HALF_GAP, TRACK_Y+HALF_GAP)):
            for i in range(self.segments):
                name = f'rail_{side}_{i}'
                ends = [xs[i], y, self.height(xs[i]), xs[i+1], y, self.height(xs[i+1])]
                ET.SubElement(body, 'geom', name=name, type='capsule', fromto=values(ends),
                              size=str(RAIL_RADIUS), rgba='.25 .48 .65 1', friction='.35 .001 .00002',
                              condim='6', solref='.006 1')
                # Explicit pair isolates rail physics from gripper and cup friction.
                ET.SubElement(contact, 'pair', geom1='ball_g0', geom2=name,
                              condim='6', friction='.35 .35 .001 .00002 .00002', solref='.006 1')
        for x in (.29, .48, .68, .83):
            z = float(self.height(x))-.018
            for y in (TRACK_Y-.055, TRACK_Y+.055):
                ET.SubElement(world, 'geom', type='box', pos=values([x,y,z/2]),
                              size=values([.009,.009,z/2]), rgba='.3 .32 .35 1')
            ET.SubElement(world, 'geom', type='box', pos=values([x,TRACK_Y,z-.012]),
                          size='.009 .065 .006', rgba='.3 .32 .35 1')
        cup_body = ET.SubElement(world, 'body', name='pot', pos=values(cup.center))
        ET.SubElement(cup_body, 'geom', name='cup_bottom', type='cylinder', size=f'{cup.radius+.006} .008',
                      rgba='.85 .65 .18 1', condim='6', friction='.6 .01 .003', solref='.008 1')
        n = 40
        for i in range(n):
            a = 2*np.pi*i/n
            ET.SubElement(cup_body, 'geom', name=f'cup_wall_{i}', type='box',
                          pos=values([(cup.radius+.004)*np.cos(a), (cup.radius+.004)*np.sin(a), cup.height/2]),
                          size=values([.004, (cup.radius+.008)*np.tan(np.pi/n), cup.height/2]),
                          euler=values([0,0,a]), rgba='.85 .65 .18 1', solref='.008 1')
        ET.SubElement(world, 'geom', name='cup_support', type='box',
                      pos=values([cup.center_x,cup.center_y,(cup.base_height_m-.01)/2]),
                      size=values([.085,.085,(cup.base_height_m-.01)/2]), rgba='.3 .32 .35 1')

        # Shared renderer hides group 0 collision proxies; these primitives also
        # provide the visible task surfaces (no duplicate collision geometry).
        for geom in world.iter('geom'):
            geom.set('group', '1')


def arena_root():
    return ET.fromstring('''<mujoco model="rail_ball_to_cup">
      <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
      <asset><material name="wood_mat" rgba=".5 .4 .3 1"/></asset>
      <worldbody><light pos="0 -1 3" dir="0 0 -1"/>
      <geom name="floor" type="plane" size="4 4 .05" rgba=".8 .8 .8 1"/>
      <camera name="agentview" pos="1 -2 1.6" xyaxes="1 0 0 0 .4 .9165"/>
      </worldbody></mujoco>''')
