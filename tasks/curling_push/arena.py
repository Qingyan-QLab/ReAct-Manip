"""Procedural flat lane, visual-only target and free rigid stone."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from react_manip.cameras import look_at


def vec(values):
    return ' '.join(map(str, values))


def camera_poses(p):
    """Fixed views based on lane dimensions, never on the hidden target position.

    The long axis runs horizontally in the image. Framing allows the default
    4:3 viewport as well as the wider recording viewport.
    """
    ice = p['ice']
    length, z = ice['length_m'], ice['height_m']
    center = np.array([length/2-.15, 0., z])
    span = length+.8  # include the Panda base behind the launch end
    top_center = center + [-.3,0.,0.]
    top = top_center + [0., 0., max(1.8, span*.8)]
    side = center + [0., -span*.57, span*.57]
    room = center + [span*.25, -span*.8, span*.65]
    close_target = np.array([.43,0.,z+.06])
    close = close_target + [.35,-1.25,.85]
    return {
        'workspace_top': (top, look_at(top,top_center,up=(0,1,0)), 55.),
        'workspace_oblique': (side, look_at(side,center), 55.),
        'room_overview': (room, look_at(room,center+[0,0,.15]), 60.),
        'launch_closeup': (close, look_at(close,close_target), 55.),
    }


def build_xml(p):
    ice, stone, target = (p[k] for k in ('ice', 'stone', 'target'))
    length, width, z = (ice[k] for k in ('length_m', 'width_m', 'height_m'))
    root = ET.Element('mujoco', model='curling_push')
    ET.SubElement(root, 'option', timestep='.002', integrator='implicitfast')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'material', name='wood_mat', rgba='.35 .4 .45 1')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='2 -2 7', dir='0 0 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'geom', name='floor', type='plane', size=vec([(length+2)/2,2.2,.1]), pos=vec([length/2-.3,0,0]), rgba='.25 .28 .3 1')
    for i, (pos,size) in enumerate((([length/2-.1,1.6,1.2],[length/2+.8,.05,1.2]),
                                   ([length/2-.1,-1.6,.12],[length/2+.8,.05,.12]),
                                   ([length+.65,0,1.2],[.05,1.6,1.2]))):
        ET.SubElement(world,'geom',name=f'room_wall_{i}',type='box',pos=vec(pos),size=vec(size),
                      contype='0',conaffinity='0',group='1')
    ET.SubElement(world, 'geom', name='ice', type='box', pos=vec([(length+.2)/2,0,z-.04]),
                  size=vec([(length-.2)/2,width/2,.04]), rgba='.78 .9 .96 1',group='1',
                  friction=vec([ice['friction'], .001, .0001]), condim='4', contype='2', conaffinity='4')
    for x in (.4, length/2, length-.3):
        for y in (-width*.4, width*.4):
            ET.SubElement(world,'geom',name=f'lane_support_{x}_{y}',type='box',
                          pos=vec([x,y,(z-.08)/2]),size=vec([.06,.06,(z-.08)/2]),material='wood_mat',group='1')
    ring = ET.SubElement(world, 'body', name='target', pos=vec([target['distance_m'],0,z+.001]))
    for i in range(96):
        a, b = 2*math.pi*i/96, 2*math.pi*(i+1)/96
        radius = target['radius_m']
        ET.SubElement(ring,'geom',name=f'target_ring_{i}',type='capsule',size='.004',
                      fromto=vec([radius*math.cos(a),radius*math.sin(a),0,radius*math.cos(b),radius*math.sin(b),0]),
                      rgba='.9 .12 .13 1',contype='0',conaffinity='0',group='1')
    ET.SubElement(world,'geom',name='push_boundary',type='box',pos=vec([p['rules']['push_end_x_m'],0,z+.001]),
                  size=vec([.005,width/2,.001]),rgba='.1 .3 .8 1',group='1',contype='0',conaffinity='0')
    body = ET.SubElement(world,'body',name='stone',pos=vec([stone['start_x_m'],0,z+stone['height_m']/2+.001]))
    ET.SubElement(body,'freejoint',name='stone_free')
    mass, radius, height = stone['mass_kg'], stone['radius_m'], stone['height_m']
    inertia_xy = mass*(3*radius**2+height**2)/12
    ET.SubElement(body,'inertial',pos='0 0 0',mass=str(mass),
                  diaginertia=vec([inertia_xy,inertia_xy,mass*radius**2/2]))
    ET.SubElement(body,'geom',name='stone_collision',type='cylinder',size=vec([stone['radius_m'],stone['height_m']/2]),
                  mass=str(stone['mass_kg']),group='1',rgba='.25 .29 .33 1',friction=vec([ice['friction'],.001,.0001]),condim='4')
    ET.SubElement(body,'geom',name='stone_handle',type='capsule',size='.012',
                  fromto=vec([-.04,0,stone['height_m']/2+.022,.04,0,stone['height_m']/2+.022]),
                  mass='0',rgba='.95 .25 .08 1',contype='0',conaffinity='0',group='1')
    for x in (-.04,.04):
        ET.SubElement(body,'geom',name=f'handle_support_{x}',type='capsule',size='.009',
                      fromto=vec([x,0,stone['height_m']/2,x,0,stone['height_m']/2+.022]),
                      mass='0',rgba='.95 .25 .08 1',contype='0',conaffinity='0',group='1')
    # Four small bottom pads approximate the running band and avoid a cylinder's
    # single rocking contact on an extremely slippery surface. They carry no mass.
    for i, (x,y) in enumerate(((.06,0),(-.06,0),(0,.06),(0,-.06))):
        ET.SubElement(body,'geom',name=f'stone_pad_{i}',type='sphere',size='.004',
                      pos=vec([x*stone['radius_m']/.09,y*stone['radius_m']/.09,-stone['height_m']/2+.004]),
                      mass='0',rgba='.1 .1 .1 1',contype='4',conaffinity='2')
    # Explicit ice pair prevents robot/material defaults changing slide friction.
    contact = ET.SubElement(root,'contact')
    mu = ice['friction']
    for i in range(4):
        ET.SubElement(contact,'pair',geom1='ice',geom2=f'stone_pad_{i}',condim='4',
                      friction=vec([mu,mu,.001,.0001,.0001]),solref='.008 1')
    cameras = camera_poses(p)
    for name, (pos, rotation, fovy) in cameras.items():
        ET.SubElement(world,'camera',name=name,pos=vec(pos),xyaxes=vec(list(rotation[:,0])+list(rotation[:,1])),fovy=str(fovy))
    return ET.tostring(root,encoding='unicode')
