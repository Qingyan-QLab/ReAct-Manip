"""Curved and easy-to-topple support catalog for phone_lean.

Except for the deliberately thin upright book, support collision geometry uses
only cylinders, spheres, capsules, or tapered stacks. No support receives a
hidden stabilizing base or feasibility-specific geometry.
"""
import math
import xml.etree.ElementTree as ET

LEGACY_SUPPORTS = ('cup', 'bowl', 'pyramid')
SUPPORT_POOL = (*LEGACY_SUPPORTS, 'alarm_clock', 'book', 'bottle',
                'can', 'vase', 'cone', 'dumbbell')

# Half-width, half-depth, and total height. The book is 22 cm wide, 8 mm
# thick, and 30 cm high before the configured 0.85-1.15 scale. Thus its
# realized height is always at least 25.5 cm and thickness at most 9.2 mm.
DIMENSIONS = {
    'cup': (.055, .055, .14),
    'bowl': (.106, .106, .081),
    'pyramid': (.093, .082, .163),
    'alarm_clock': (.075, .035, .16),
    'book': (.110, .004, .30),
    'bottle': (.050, .050, .18),
    'can': (.055, .055, .15),
    'vase': (.065, .065, .18),
    'cone': (.075, .075, .18),
    'dumbbell': (.145, .045, .09),
}
RADII = {name: math.hypot(x, y) for name, (x, y, _) in DIMENSIONS.items()}
BOTTOM_OFFSETS = {name: 0. for name in SUPPORT_POOL}
TOP_DEPTH = {name: depth for name, (_, depth, _) in DIMENSIONS.items()}


def add_support(world, name):
    """Build a selected procedural support with matching visual geometry."""
    if name in LEGACY_SUPPORTS or name not in DIMENSIONS:
        raise ValueError(f'Unsupported procedural support {name!r}')
    body = ET.SubElement(world, 'body', name=name, pos='0 0 .68')
    ET.SubElement(body, 'freejoint', name=f'{name}_free')

    def pair(part_name, geom_type, *, mass, rgba, **attributes):
        for visual in (False, True):
            values = dict(attributes)
            values.update({
                'name': part_name + ('_vis' if visual else ''),
                'type': geom_type,
                'mass': '0' if visual else str(mass),
                'group': '1' if visual else '0',
                'contype': '0' if visual else '1',
                'conaffinity': '0' if visual else '1',
                'rgba': rgba,
            })
            if not visual:
                values.update(friction='.45 .003 .0001', condim='6')
            ET.SubElement(body, 'geom', **values)

    if name == 'book':
        # local -Y is the visible front cover. Identity orientation therefore
        # makes the upright front face the robot at the table's -Y edge.
        x, y, height = DIMENSIONS[name]
        cover = .00025
        pair('book_pages', 'box', size=f'{x-.004} {y-cover} {height/2-.003}',
             pos=f'0 0 {height/2}', mass=.018, rgba='.92 .88 .72 1')
        pair('book_front_cover', 'box', size=f'{x} {cover} {height/2}',
             pos=f'0 {-y+cover} {height/2}', mass=.003, rgba='.55 .12 .10 1')
        pair('book_back_cover', 'box', size=f'{x} {cover} {height/2}',
             pos=f'0 {y-cover} {height/2}', mass=.003, rgba='.32 .07 .07 1')
        pair('book_spine', 'box', size=f'.0015 {y} {height/2}',
             pos=f'{-x+.0015} 0 {height/2}', mass=.001, rgba='.45 .09 .08 1')
        return body

    if name == 'alarm_clock':
        pair('alarm_clock_face', 'cylinder', size='.062 .025', pos='0 0 .087',
             euler=f'{math.pi/2} 0 0', mass=.10, rgba='.82 .26 .20 1')
        pair('alarm_clock_dial', 'cylinder', size='.053 .002', pos='0 -.027 .087',
             euler=f'{math.pi/2} 0 0', mass=.003, rgba='.90 .88 .78 1')
        for side in (-1, 1):
            pair(f'alarm_clock_bell_{side}', 'sphere', size='.023',
                 pos=f'{side*.045} 0 .145', mass=.012, rgba='.60 .64 .68 1')
            pair(f'alarm_clock_leg_{side}', 'capsule', size='.008',
                 fromto=f'{side*.034} 0 .008 {side*.041} 0 .038',
                 mass=.004, rgba='.18 .18 .20 1')
        return body

    if name == 'bottle':
        pair('bottle_body', 'cylinder', size='.050 .060', pos='0 0 .060',
             mass=.045, rgba='.25 .65 .40 1')
        pair('bottle_shoulder', 'capsule', size='.038', fromto='0 0 .090 0 0 .125',
             mass=.012, rgba='.25 .65 .40 1')
        pair('bottle_neck', 'cylinder', size='.022 .024', pos='0 0 .148',
             mass=.006, rgba='.25 .65 .40 1')
        pair('bottle_cap', 'cylinder', size='.024 .006', pos='0 0 .178',
             mass=.002, rgba='.88 .88 .86 1')
        return body

    if name == 'can':
        pair('can_body', 'cylinder', size='.055 .071', pos='0 0 .075',
             mass=.055, rgba='.72 .16 .12 1')
        pair('can_top_rim', 'cylinder', size='.056 .002', pos='0 0 .148',
             mass=.002, rgba='.72 .74 .76 1')
        return body

    if name == 'vase':
        pair('vase_base', 'cylinder', size='.052 .006', pos='0 0 .006',
             mass=.012, rgba='.35 .55 .82 1')
        pair('vase_body', 'capsule', size='.045', fromto='0 0 .050 0 0 .120',
             mass=.045, rgba='.35 .55 .82 1')
        pair('vase_neck', 'cylinder', size='.026 .020', pos='0 0 .157',
             mass=.008, rgba='.35 .55 .82 1')
        pair('vase_rim', 'cylinder', size='.032 .003', pos='0 0 .179',
             mass=.002, rgba='.28 .46 .72 1')
        return body

    if name == 'cone':
        # A six-tier round taper avoids broad planar faces while remaining
        # compatible with the pinned MuJoCo build, which lacks cone geoms.
        radii = (.075, .064, .053, .042, .031, .020)
        for index, radius in enumerate(radii):
            pair(f'cone_tier_{index}', 'cylinder', size=f'{radius} .015',
                 pos=f'0 0 {(index+.5)*.03}', mass=.010,
                 rgba=f'{.88-index*.035} {.48-index*.025} .10 1')
        return body

    if name == 'dumbbell':
        pair('dumbbell_handle', 'capsule', size='.017',
             fromto='-.095 0 .045 .095 0 .045', mass=.020, rgba='.18 .18 .20 1')
        for side in (-1, 1):
            pair(f'dumbbell_end_{side}', 'sphere', size='.042',
                 pos=f'{side*.105} 0 .045', mass=.030, rgba='.28 .30 .32 1')
        return body

    raise ValueError(f'No geometry builder for {name!r}')
