"""Embodiment-owned sensor profiles; no task or scene dependencies.

Third-party robot models may expose ``sensor_profile`` or register their
exact model class name. Unknown models retain imported camera extrinsics.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class MountedCamera:
    """Body and role names are unprefixed, in the robot model namespace."""
    role: str
    body: str
    position: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float = 65.

    def __post_init__(self):
        if self.role not in ('head', 'wrist_right', 'wrist_left', 'overhead_debug'):
            raise ValueError(f'unknown mounted camera role: {self.role}')
        if not 0 < self.fovy < 180:
            raise ValueError('fovy must be between 0 and 180 degrees')


@dataclass(frozen=True)
class SensorProfile:
    overhead_height: float = 1.8
    head_min_x: float | None = None
    head_xyaxes: str | None = None
    # Wrist camera mount in the gripper/eef local frame.  ``None`` preserves
    # an imported asset offset for models that do not have a canonical mount.
    wrist_position: tuple[float, float, float] | None = None
    wrist_target: tuple[float, float, float] | None = None
    wrist_up: tuple[float, float, float] = (0., 1., 0.)
    cameras: tuple[MountedCamera, ...] = ()


_PROFILES = {
    'Panda': SensorProfile(overhead_height=.65, cameras=(MountedCamera('head', 'base', (-.12, -.20, .80), (0., -1., 0., .866025403784, 0., .5), 80.),)),
    'CLSawyer': SensorProfile(cameras=(MountedCamera('head', 'base', (-.12, -.20, 1.70), (0., -1., 0., .866025403784, 0., .5), 80.),)),
    # First physical-fit trial: lift the wrist sensor 10 cm in the eef local
    # ``-X`` direction (the neutral image-up direction) to reduce gripper-root
    # occlusion while keeping the same TCP aiming point.
    'PiperX': SensorProfile(wrist_position=(-.1, 0., -.08), wrist_target=(0., 0., .06), wrist_up=(-1., 0., 0.), cameras=(
        MountedCamera('head', 'right_base', (-.12, -.20, .80),
                      (0., -1., 0., .866025403784, 0., .5), 80.),
    )),
    'DualPiperX': SensorProfile(head_min_x=.24, head_xyaxes='0 -1 0 .5 0 .866025403784', wrist_position=(-.1, 0., -.08), wrist_target=(0., 0., .06), wrist_up=(-1., 0., 0.)),
    'MobileDualPiperX': SensorProfile(head_min_x=.24, head_xyaxes='0 -1 0 .5 0 .866025403784', wrist_position=(-.1, 0., -.08), wrist_target=(0., 0., .06), wrist_up=(-1., 0., 0.)),
}


_PROFILES['AgileXPiper'] = _PROFILES['PiperX']
_PROFILES['AgileXAloha'] = _PROFILES['DualPiperX']
_PROFILES['AgileXMobileAloha'] = _PROFILES['MobileDualPiperX']


def register_sensor_profile(model_class_name, profile):
    if not isinstance(profile, SensorProfile):
        raise TypeError('profile must be a SensorProfile')
    if model_class_name in _PROFILES:
        raise ValueError(f'sensor profile already registered: {model_class_name}')
    _PROFILES[model_class_name] = profile


def sensor_profile(robot):
    profile = getattr(robot, 'sensor_profile', None)
    if profile is None:
        profile = _PROFILES.get(type(robot).__name__, SensorProfile())
    if not isinstance(profile, SensorProfile):
        raise TypeError('robot.sensor_profile must be a SensorProfile')
    return profile


def install_declared_cameras(robot):
    """New models specify mounts explicitly, without legacy name inference."""
    profile = sensor_profile(robot)
    roles = [c.role for c in profile.cameras]
    if len(set(roles)) != len(roles):
        raise ValueError('duplicate mounted camera roles')
    for spec in profile.cameras:
        name = robot.naming_prefix + spec.role
        body_name = robot.correct_naming(spec.body)
        body = next((b for b in robot.worldbody.iter('body') if b.get('name') == body_name), None)
        if body is None:
            raise ValueError(f'{name}: camera mount body not found: {body_name}')
        matches = [(b, c) for b in robot.worldbody.iter('body') for c in b.findall('camera') if c.get('name') == name]
        if len(matches) > 1 or (matches and matches[0][0] is not body):
            raise ValueError(f'{name}: conflicting camera mount')
        camera = matches[0][1] if matches else ET.SubElement(body, 'camera', name=name)
        for attr in ('quat', 'euler', 'axisangle', 'zaxis', 'target', 'mode'):
            camera.attrib.pop(attr, None)
        camera.set('pos', ' '.join(map(str, spec.position)))
        camera.set('xyaxes', ' '.join(map(str, spec.xyaxes)))
        camera.set('fovy', str(spec.fovy))
        if spec.role == 'head':
            # Keep the existing public main-camera ID for all three models.
            from copy import deepcopy
            alias = robot.naming_prefix + 'head_camera'
            if not any(c.get('name') == alias for c in robot.worldbody.iter('camera')):
                clone = deepcopy(camera); clone.set('name', alias); body.append(clone)


def ensure_robot_overhead_camera(robot_model, *, suffix='overhead_camera'):
    """Attach virtual debug camera to moving root, independent of scene."""
    world = getattr(robot_model, 'worldbody', None)
    if world is None:
        return None
    naming = getattr(robot_model, 'correct_naming', None)
    name = naming(suffix) if callable(naming) else suffix
    if any(c.get('name') == name for c in world.iter('camera')):
        return name
    root = world.find('body')
    if root is None:
        return None
    for body in root.iter('body'):
        if body.find('freejoint') is not None or body.find("joint[@type='free']") is not None:
            root = body
            break
    height = sensor_profile(robot_model).overhead_height
    ET.SubElement(root, 'camera', name=name, pos=f'0.20 0 {height}',
                  xyaxes='0 1 0 -1 0 0', fovy='65')
    return name


PIPERX_ROBOTS = ('piperx', 'dual_piperx', 'mobile_dual_piperx')
CAMERA_CONTRACT_VERSION = '3.0'


def policy_camera_ids(robot_id):
    """Ordered public views; left/right names follow robot arms, not screen sides."""
    main = ('robot0_head_camera',)
    if robot_id in PIPERX_ROBOTS:
        return main + ('robot0_wrist_right',) + (() if robot_id == 'piperx' else ('robot0_wrist_left',))
    from ..optional_integrations import import_optional_integration
    catalog = import_optional_integration('integrations.rpent_embodiments.catalog')
    if catalog is not None and robot_id in catalog.EMBODIMENTS:
        return catalog.camera_ids(robot_id)
    return main
