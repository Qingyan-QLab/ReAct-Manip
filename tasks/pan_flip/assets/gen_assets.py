"""Original procedural assets: a convex-segment pan and one two-sided rigid disk.

All dimensions are metres. Visual geoms carry no mass or collision; explicit
inertias describe the assembled tools. No downloaded assets are required.
"""
import math
import numpy as np
import xml.etree.ElementTree as ET
from tasks.pan_flip.mass_properties import pie_mass_properties


def vec(values):
    return ' '.join(str(float(x)) for x in values)


def geom(body, name, **attributes):
    common = dict(name=name, solref='.008 1', solimp='.95 .99 .001',
                  friction='.45 .002 .0001', condim='4')
    common.update(attributes)
    return ET.SubElement(body, 'geom', **common)


def pan_mass_properties(mass, radius, handle_length):
    """Calibrated head plus a uniform box handle (10% of total mass).

    Recover the fixed head's first/second moments from the original reference
    tool, then replace the handle moments. This preserves reference dynamics.
    """
    fraction = .1
    reference_center = -.215
    handle_center = -(radius - .02 + handle_length/2)
    com_x = -.025 + fraction*(handle_center-reference_center)
    # Moments about the pan origin, normalized by total mass.
    origin = np.array([.008, .014+.025**2, .021+.025**2])
    origin[1:] += fraction*((handle_length**2-.17**2)/12
                           +handle_center**2-reference_center**2)
    origin[1:] -= com_x**2
    return np.array([com_x,0.,0.]), mass*origin


def add_pan(eef, radius=.15, mass=.35, friction=.45,
            handle_length=.17, grip_fraction=11/17):
    # Grip fraction runs from the handle root toward its free end.
    grip_distance = radius-.02 + handle_length*grip_fraction
    body = ET.SubElement(eef, 'body', name='pan', pos=vec([0,grip_distance,0]),
                         quat='0 .7071067811865476 .7071067811865476 0')
    com, inertia = pan_mass_properties(mass, radius, handle_length)
    ET.SubElement(body, 'inertial', pos=vec(com), mass=str(mass),
                  diaginertia=vec(inertia))
    geom(body, 'pan_bottom', type='cylinder', size=vec([radius, .004]),
         pos='0 0 -.004', rgba='.13 .15 .17 1', group='1', friction=f'{friction} .002 .0001')
    # Overlapping tangential boxes form an outward-sloping, open bowl.
    for i in range(32):
        angle = 2*math.pi*i/32
        wall = ET.SubElement(body, 'body', name=f'pan_wall_body_{i}',
                            pos=vec([radius*math.cos(angle),radius*math.sin(angle),0]),
                            euler=vec([0,0,angle]))
        geom(wall, f'pan_wall_{i}', type='box', pos='.003 0 .012',
             size=vec([.004, radius*math.tan(math.pi/32)+.001,.018]),
             euler='0 .35 0', rgba='.22 .24 .27 1', group='1',
             friction=f'{friction} .002 .0001', mass='0')
    geom(body, 'pan_handle', type='box', pos=vec([-(radius-.02+handle_length/2),0,0]),
         size=vec([handle_length/2,.012,.009]),
         rgba='.12 .08 .055 1', group='1')
    ET.SubElement(body, 'site', name='pan_center', pos='0 0 0', size='.003', rgba='0 0 0 0')
    return body


def add_pie(world, asset, radius=.065, half_thickness=.009, mass=.08, friction=.45,
            mass_uniformity=1.0, density_direction_deg=0.0):
    vertices=[]
    for r,z in [(radius-.002,-half_thickness),(radius,-half_thickness+.002),
                (radius,half_thickness-.002),(radius-.002,half_thickness)]:
        for i in range(32):
            a=2*math.pi*i/32
            vertices.extend([r*math.cos(a),r*math.sin(a),z])
    ET.SubElement(asset,'mesh',name='pie_hull',vertex=vec(vertices))
    body = ET.SubElement(world, 'body', name='pie', pos='0 0 1')
    ET.SubElement(body, 'freejoint', name='pie_joint')
    com, inertia = pie_mass_properties(radius, half_thickness, mass,
                                       mass_uniformity, density_direction_deg)
    inertial = dict(pos=vec(com), mass=str(mass))
    if mass_uniformity == 1.0:
        # Preserve the original uniform-disk reference exactly.
        inertial['diaginertia'] = vec(inertia.diagonal())
    else:
        inertial['fullinertia'] = vec([inertia[0,0], inertia[1,1], inertia[2,2],
                                      inertia[0,1], inertia[0,2], inertia[1,2]])
    ET.SubElement(body, 'inertial', **inertial)
    geom(body, 'pie_collision', type='mesh', mesh='pie_hull',
         rgba='.8 .55 .23 1', group='0', friction=f'{friction} .002 .0001')
    # Two non-overlapping visual halves, permanently attached to ONE body.
    for label, sign, color in [('A',1,'.95 .68 .22 1'),('B',-1,'.32 .10 .035 1')]:
        ET.SubElement(body, 'geom', name=f'pie_face_{label}', type='cylinder',
                      size=vec([radius,half_thickness/2]),
                      pos=vec([0,0,sign*half_thickness/2]), rgba=color,
                      contype='0', conaffinity='0', group='1', mass='0')
    return body
