"""Robot-owned fixed stands. Never add these as task arena geometry."""
import xml.etree.ElementTree as ET


def use_existing_mount(robot):
    """Remove an automatic floor stand for tasks with a table/car mounting point.

    Call before task-specific set_base_xpos. Preserve all original arm bodies,
    sites and cameras; only undo the wrapper installed by install_fixed_stand.
    """
    root = robot.worldbody.find('body')
    mount = root.find(f"body[@name='{robot.naming_prefix}fixed_stand_mount']")
    if mount is None:
        return
    root.remove(mount)
    for child in list(mount):
        root.append(child)
    for suffix in ('fixed_stand_collision', 'fixed_stand_visual'):
        geom = root.find(f"geom[@name='{robot.naming_prefix}{suffix}']")
        if geom is not None:
            root.remove(geom)


def install_fixed_stand(robot, *, height, half_width, half_depth, center_xy=(0., 0.)):
    """Raise the complete fixed model once, keeping the root on the floor.

    All imported bodies/sites/cameras move together. A collision geom and a
    non-colliding visual geom share the same dimensions. Mobile models must
    not use this helper: their chassis already defines the mounting height.
    """
    if min(height, half_width, half_depth) <= 0:
        raise ValueError('stand dimensions must be positive')
    root = robot.worldbody.find('body')
    if root is None:
        raise ValueError('robot has no root body')
    if root.find('.//freejoint') is not None or root.find(".//joint[@type='free']") is not None:
        raise ValueError('cannot add a fixed stand to a free-root robot')
    name = robot.naming_prefix + 'fixed_stand_mount'
    if root.find(f"body[@name='{name}']") is not None:
        return
    mount = ET.Element('body', name=name, pos=f'0 0 {height}')
    for child in list(root):
        root.remove(child)
        mount.append(child)
    root.append(mount)
    attrs = dict(type='box', pos=f'{center_xy[0]} {center_xy[1]} {height / 2}',
                 size=f'{half_depth} {half_width} {height / 2}', rgba='.25 .28 .32 1',
                 friction='.8 .01 .001', solref='.005 1', condim='6')
    ET.SubElement(root, 'geom', name=robot.naming_prefix+'fixed_stand_collision', group='0', **attrs)
    ET.SubElement(root, 'geom', name=robot.naming_prefix+'fixed_stand_visual', group='1', contype='0', conaffinity='0', density='0', **attrs)


def set_fixed_mount_pose(robot, *, position, yaw, height):
    """Place a complete floor stand using its arm-mount frame, not its floor root.

    position is the world position of the imported asset origin at mounting
    height. Only yaw is allowed: the stand stays vertical and reaches z=0.
    Both arms, sites and cameras keep their asset-relative transforms.
    """
    import math
    import numpy as np
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all() or not math.isfinite(yaw):
        raise ValueError('mount pose must be finite')
    if not math.isfinite(height) or height <= 0 or abs(position[2] - height) > 1e-12:
        raise ValueError('mount height must equal its world z above the floor')
    root = robot.worldbody.find('body')
    mount = root.find(f"body[@name='{robot.naming_prefix}fixed_stand_mount']")
    if mount is None:
        raise ValueError('fixed mount pose requires an installed floor stand')
    mount.set('pos', f'0 0 {height}')
    for suffix in ('fixed_stand_collision', 'fixed_stand_visual'):
        geom = root.find(f"geom[@name='{robot.naming_prefix}{suffix}']")
        pos = np.fromstring(geom.get('pos'), sep=' ')
        size = np.fromstring(geom.get('size'), sep=' ')
        pos[2] = size[2] = height / 2
        geom.set('pos', ' '.join(map(str, pos)))
        geom.set('size', ' '.join(map(str, size)))
    robot.set_base_xpos([position[0], position[1], 0.])
    robot.set_base_ori(np.array([0., 0., yaw]))
