"""Shared camera contract. World cameras are fixed; robot cameras are body mounted.

No rendering is performed unless explicitly requested. Legacy MJCF names stay
available, while canonical names have the same meaning in every task.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np

FIXED_CAMERAS = ('workspace_oblique', 'workspace_top', 'room_overview')
LABELS = {'workspace_oblique': 'Workspace oblique', 'workspace_top': 'Workspace top',
          'room_overview': 'Room overview', 'basket_top': 'Basket top'}


@dataclass(frozen=True)
class Workspace:
    center: tuple = (0., 0., .75)
    half_size: tuple = (.6, .8)
    forward: tuple = (1., 0.)
    room_half_size: tuple = (3., 3.)


# Task-level declarations, not robot geometry or hidden object state. The
# table envelope is refined from the compiled static support where available.
WORKSPACES = {
    'find_object_in_drawer': Workspace((.10, 0., .80), (.65, .65)),
    'find_object_in_drawer1': Workspace((.10, 0., .80), (.65, .65)),
    'find_object_in_drawer2': Workspace((-.10, 0., .48), (.80, .75)),
    'rope_hang': Workspace((.55, 0., .85), (.55, .55)),
    'pan_flip': Workspace((.6, 0., 1.0), (.45, .4)),
    'rail_ball_to_cup': Workspace((.7, 0., .65), (.65, .5)),
    'conveyor_grasp': Workspace((2., 0., .9), (.9, .3), (0., -1.), (3.3, 2.5)),
    'laptop_into_drawer': Workspace((.1, 0., .8), (.65, .65)),
    'box_insertion': Workspace((.25, 0., .78), (.45, .8)),
    'button_indicator_matching': Workspace((.25, 0., .78), (.45, .8)),
    'pen_strike': Workspace((0., 0., .75), (.5, .5)),
    'pitch_pot': Workspace((-.25, 0., .18), (.65, .65)),
    'paddle_juggle': Workspace((.15, 0., .8), (.7, .7)),
    'phone_power_on': Workspace((0., 0., .15), (.5, .5)),
    'turntable_gamble': Workspace((-.05, 0., .54), (.65, .65)),
    'ball_to_target_hole': Workspace((0., 0., .75), (.7, .7)),
    'cup_lid_matching': Workspace(), 'leveling_board': Workspace(),
    'ring_toss': Workspace(), 'phone_cup': Workspace(),
    'car_drive': Workspace((1.25, 0., .07), (1.25, 1.), (1., 0.), (5., 5.)),
    # Local target view; the room view covers the entire curling lane.
    'curling_sweep': Workspace((6., 0., .68), (1., .8), (1., 0.), (7., 3.)),
    'curling_push': Workspace((2.85, 0., .65), (2.65, .55), (1., 0.), (4., 2.)),
}


def _vec(x):
    return ' '.join(f'{float(v):.9g}' for v in x)


def look_at(position, target, up=(0., 0., 1.)):
    """MuJoCo basis: -Z optical forward, +Y image-up, +X image-right."""
    z = np.asarray(position, float) - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        raise ValueError('camera up is parallel to viewing direction')
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def fixed_poses(workspace):
    c = np.array(workspace.center, float)
    forward = np.r_[workspace.forward, 0.]
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, [0, 0, 1])
    # Prefer 1.5 m above the surface; enlarge for broad native workspaces
    # instead of silently cropping them or using extreme fisheye-like FOVs.
    height = max(1.5, max(workspace.half_size) / np.tan(np.deg2rad(37.5)) * 1.12)
    top = c + [0, 0, height]
    distance = max(1.6, max(workspace.half_size) * 1.7)
    oblique = c + forward * distance + side * distance * .25 + [0, 0, distance]
    room_center = c.copy()
    if workspace.center[0] == 2. and workspace.forward == (0., -1.):
        room_center[1] = 1.0
    rx, ry = workspace.room_half_size
    room = room_center + np.array([rx * .8, ry * .8, max(2.4, .9 * max(rx, ry))])
    return {'workspace_top': (top, look_at(top, c, forward), 75.),
            'workspace_oblique': (oblique, look_at(oblique, c), 65.),
            'room_overview': (room, look_at(room, room_center), 85.)}


def ensure_scene_cameras(arena, task_id=None):
    """Insert canonical fixed cameras before compilation; retain old names."""
    world = arena.worldbody
    task_id = task_id or arena.root.get('model', '')
    workspace = WORKSPACES.get(task_id, Workspace())
    poses = fixed_poses(workspace)
    # Always reserve the optional regional view for this multi-zone task.
    if 'conveyor' in task_id:
        poses['basket_top'] = (np.array([2., 1.3, 2.25]), np.eye(3), 65.)
    existing = {c.get('name') for c in world.iter('camera')}
    for name, (position, rotation, fovy) in poses.items():
        if name not in existing:
            ET.SubElement(world, 'camera', name=name, pos=_vec(position),
                          xyaxes=_vec(np.r_[rotation[:, 0], rotation[:, 1]]), fovy=str(fovy))


def ensure_robot_cameras(robot):
    """Normalize imported names using embodiment-owned sensor parameters."""
    from copy import deepcopy
    from .robots.sensors import sensor_profile, install_declared_cameras
    install_declared_cameras(robot)
    profile = sensor_profile(robot)
    prefix = robot.naming_prefix
    entries = [(b, c) for b in robot.worldbody.iter('body') for c in b.findall('camera')]
    existing = {c.get('name') for _, c in entries}
    for body, camera in entries:
        name = camera.get('name', '')
        canonical = None
        if name.endswith('_head_camera'):
            canonical = prefix + 'head'
        elif name.endswith('eye_in_hand'):
            arm = 'left' if 'left' in name else 'right'
            canonical = prefix + 'wrist_' + arm
        elif name.endswith('overhead_camera'):
            canonical = prefix + 'overhead_debug'
        if canonical and canonical not in existing:
            clone = deepcopy(camera)
            clone.set('name', canonical)
            if canonical.endswith('_head') and profile.head_xyaxes is not None:
                # Fixed mount pitch, not target tracking. The legacy head
                # name retains its original horizontal optical axis.
                for attr in ('quat', 'euler', 'axisangle', 'zaxis'):
                    clone.attrib.pop(attr, None)
                clone.set('xyaxes', profile.head_xyaxes)
                # The imported mast visual encloses the old x=.16 sensor.
                # A small forward bracket puts the optical centre outside it.
                position = np.fromstring(clone.get('pos', '0 0 0'), sep=' ')
                if profile.head_min_x is not None:
                    position[0] = max(profile.head_min_x, position[0])
                clone.set('pos', _vec(position))
                camera.set('pos', _vec(position))
            elif '_wrist_' in canonical and profile.wrist_target is not None:
                # Mount the sensor on the eef rear/top face.  The imported
                # asset used a +Y side offset, which made the view look
                # rolled by roughly 90 degrees.  Use an explicit local
                # position when the embodiment declares one, then aim at a
                # point 6 cm beyond TCP with a tool-frame up vector.
                position = (np.asarray(profile.wrist_position, dtype=float)
                            if profile.wrist_position is not None else
                            np.fromstring(clone.get('pos', '0 0 0'), sep=' '))
                rotation = look_at(position, np.array(profile.wrist_target), profile.wrist_up)
                for attr in ('quat', 'euler', 'axisangle', 'zaxis'):
                    clone.attrib.pop(attr, None)
                    camera.attrib.pop(attr, None)
                camera.set('pos', _vec(position))
                clone.set('pos', _vec(position))
                clone.set('xyaxes', _vec(np.r_[rotation[:, 0], rotation[:, 1]]))
                camera.set('xyaxes', clone.get('xyaxes'))
            body.append(clone)
            existing.add(canonical)


def raw_sim(env):
    sim = getattr(env, 'sim', env)
    model = getattr(sim.model, '_model', sim.model)
    data = getattr(sim.data, '_data', sim.data)
    return sim, model, data


def configure_fixed_cameras(env):
    """Called after reset, never during step: no tracking or hidden-state input."""
    import mujoco
    _, model, data = raw_sim(env)
    module = type(env).__module__.split('.')
    task_id = getattr(env, 'task_id', module[1] if len(module) > 1 else '')
    w = getattr(env, 'camera_workspace', WORKSPACES.get(task_id, Workspace()))
    if isinstance(w, dict):
        w = Workspace(**w)
    center, half = np.array(w.center), np.array(w.half_size)
    # Read only static table supports, not the randomly placed target objects.
    if task_id != 'conveyor_grasp':
        for name in ('table', 'table_top', 'table_top_col', 'table_collision'):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0 and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                extent = np.abs(data.geom_xmat[gid].reshape(3, 3)) @ model.geom_size[gid]
                center = data.geom_xpos[gid].copy() + [0, 0, extent[2]]
                half = extent[:2] * 1.08
                break
    if task_id == 'pitch_pot':
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pot')
        if bid >= 0:
            center = data.xpos[bid].copy() + [0, 0, .18]
            half = np.array([.65, .65])
    forward = np.array(w.forward)
    robots = getattr(env, 'robots', ())
    if robots:
        root = robots[0].robot_model.root_body
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
        if bid >= 0:
            direction = center[:2] - data.xpos[bid, :2]
            if np.linalg.norm(direction) > .1:
                forward = direction / np.linalg.norm(direction)
    poses = fixed_poses(Workspace(tuple(center), tuple(half), tuple(forward), w.room_half_size))
    if robots and bid >= 0:
        anchors = [center[:2] - half, center[:2] + half, data.xpos[bid, :2]]
        if hasattr(env, 'bin_centers'):
            anchors.extend(np.asarray(env.bin_centers)[:, :2])
        anchors = np.asarray(anchors)
        room_center = np.r_[(anchors.min(axis=0) + anchors.max(axis=0)) / 2, center[2]]
        rx, ry = w.room_half_size
        room_pos = room_center + [rx * .8, ry * .8, max(2.4, .9 * max(rx, ry))]
        poses['room_overview'] = (room_pos, look_at(room_pos, room_center), 85.)
    if task_id == 'car_drive':
        ids = [i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith('road_')]
        if ids:
            route_center = data.geom_xpos[ids].mean(axis=0)
            room_pos = route_center + [4, -5, 5]
            poses['room_overview'] = (room_pos, look_at(room_pos, route_center), 85.)
    if hasattr(env, 'bin_centers'):
        bins = np.asarray(env.bin_centers)
        c = bins.mean(axis=0)
        p = c + [0, 0, 1.5]
        poses['basket_top'] = (p, look_at(p, c, np.r_[forward, 0]), 65.)
    # Optional task-owned static framing, reconstructed from scene dimensions.
    # Preserve it through both factory setup and hard resets.
    poses.update(getattr(env, 'fixed_camera_poses', {}))
    for name, (position, rotation, fovy) in poses.items():
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise ValueError(f'{task_id}: missing canonical camera {name}')
        model.cam_pos[cid] = position
        mujoco.mju_mat2Quat(model.cam_quat[cid], rotation.flatten())
        model.cam_fovy[cid] = fovy
    mujoco.mj_forward(model, data)


def camera_specs(env):
    import mujoco
    _, model, _ = raw_sim(env)
    result = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        canonical = name in FIXED_CAMERAS or name == 'basket_top' or name.endswith(('_head', '_wrist_right', '_wrist_left', '_overhead_debug'))
        role = 'fixed' if model.cam_bodyid[i] == 0 else 'mounted'
        default_policy = name in ('workspace_top', 'workspace_oblique') or name.endswith(('_head', '_wrist_right', '_wrist_left'))
        result.append(dict(id=name, label=LABELS.get(name, name.replace('_head', ' head').replace('_wrist_right', ' right wrist').replace('_wrist_left', ' left wrist')),
                           canonical=canonical, display_default=canonical and not name.endswith('_overhead_debug'),
                           mounting=role, parent_body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.cam_bodyid[i])),
                           width=640, height=480, fovy=float(model.cam_fovy[i]), policy_default=default_policy,
                           calibrated=False, image_format='RGB uint8 HWC, top-left origin'))
    def order(s):
        n = s['id']
        if n in FIXED_CAMERAS:
            return (FIXED_CAMERAS.index(n), n)
        return (3 if n.endswith('_head') else 4 if n.endswith('_wrist_right') else 5 if n.endswith('_wrist_left') else 6, n)
    return sorted(result, key=order)


def display_camera_names(env, *, include_legacy=False):
    """Standard operator views; retain legacy/debug cameras only by opt-in.

    An unmigrated environment with no standard views keeps its original views.
    This selection does not change model cameras or policy observations.
    """
    specs = camera_specs(env)
    selected = [s['id'] for s in specs if s['display_default']]
    if include_legacy or not selected:
        return [s['id'] for s in specs]
    return selected


# Fixed mounts, expressed in the robot mount frame (+X forward, +Z up).
# Task profiles describe static workspace framing, never target tracking.
POLICY_CAMERA = 'robot0_head_camera'
EGO_PROFILES = {
    'car_drive2': (50., 100.),
    'curling_push': (40., 80.),
    'conveyor_grasp': (70., 100.),
    'tilte_maze': (65., 60.),
    # These two tasks use a tall vertical workspace.  The generic 60-degree
    # downward pitch puts the pan's toss arc and the ring-toss far row against
    # (or beyond) the top edge of the 4:3 policy image.  Lower pitch means the
    # optical axis is tilted upward while preserving the robot-mounted pose.
    'pan_flip': (40., 85.),
    'ring_toss1': (50., 80.),
}


def configure_policy_camera(env):
    """Pitch the robot main sensor; preserve its body attachment."""
    import mujoco
    _, model, data = raw_sim(env)
    module = type(env).__module__.split('.')
    task_id = getattr(env, 'task_id', module[1] if len(module) > 1 else '')
    pitch, fovy = EGO_PROFILES.get(task_id, (60., 80.))
    angle = np.deg2rad(pitch)
    rotation = look_at(np.zeros(3), np.array([np.cos(angle), 0., -np.sin(angle)]))
    for name in (POLICY_CAMERA, 'robot0_head'):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            continue
        if model.cam_bodyid[cid] == 0:
            raise ValueError(f'{name} must be mounted on the robot')
        mujoco.mju_mat2Quat(model.cam_quat[cid], rotation.flatten())
        model.cam_fovy[cid] = fovy
