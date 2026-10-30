"""Lazy robosuite 1.5 ports of the four explicitly selected RPent embodiments."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np

from .catalog import EMBODIMENTS


def asset_root():
    return Path(os.environ.get('RPENT_ASSET_DIR', Path(__file__).resolve().parents[2] / 'assets/robots/rpent'))


def validate_bundle(robot_id):
    directory = asset_root() / robot_id
    manifest = directory / 'manifest.json'
    if not manifest.exists():
        raise FileNotFoundError(f'Missing {robot_id} assets; run integrations/rpent_embodiments/tools/build_assets.py')
    data = json.loads(manifest.read_text())
    if data['robot_id'] != robot_id or not data.get('files'):
        raise ValueError('Invalid embodiment manifest')
    root=Path(__file__).resolve().parents[2]
    for key,path in (('source_lock_sha256',Path(__file__).with_name('assets.lock.json')),
                     ('generator_sha256',Path(__file__).with_name('tools')/'build_assets.py')):
        if data.get(key)!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Stale robot bundle; rerun integrations/rpent_embodiments/tools/build_assets.py --offline')
    for name, checksum in data['files'].items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Changed or missing model resource: {path}')
    return data


def controller_config(robot_id):
    from robosuite.controllers import load_composite_controller_config
    spec = EMBODIMENTS[robot_id]
    if spec.mobile:
        return load_composite_controller_config(robot='PandaOmron')
    parts = {}
    for arm in spec.arms:
        parts[arm] = dict(type='OSC_POSE', input_max=1, input_min=-1,
                          output_max=[.05]*3 + [.5]*3, output_min=[-.05]*3 + [-.5]*3,
                          kp=150, damping_ratio=1, impedance_mode='fixed',
                          input_type='delta', input_ref_frame='world',
                          uncouple_pos_ori=True, interpolation=None, gripper={'type': 'GRIP'})
        if spec.source == 'robodojo':
            # Native qpos targets are rescaled by the evaluator wrapper, not
            # treated as Cartesian deltas or sent to a different robot's IK.
            parts[arm] = dict(type='JOINT_POSITION', input_type='absolute',
                              kp=150, damping_ratio=1, interpolation=None, gripper={'type': 'GRIP'})
    return dict(type='BASIC', body_parts=parts)


def register_models():
    from robosuite.models.robots import Panda, PandaOmron
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.models.grippers import GRIPPER_MAPPING
    from robosuite.models.grippers.gripper_model import GripperModel
    from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot, WheeledRobot
    from react_manip.robots.sensors import SensorProfile, MountedCamera

    if 'LiberoPanda' in REGISTERED_ROBOTS:
        return

    class LiberoPanda(Panda):
        @property
        def init_qpos(self):
            return np.array([0, -.161037389, 0, -2.44459747, 0, 2.22675220, np.pi/4])

        @property
        def sensor_profile(self):
            return SensorProfile(cameras=(MountedCamera('head', 'base', (-.12, -.20, .80),
                (0., -1., 0., .866025403784, 0., .5), 80.),))

    class RoboCasaPandaOmron(PandaOmron):
        def set_base_xpos(self, pos):
            # These floor-standing bases must not inherit a negative mounting
            # height from a task's legacy table-mounted Panda fallback.
            pos=np.array(pos,dtype=float);pos[2]=max(0.,pos[2])
            super().set_base_xpos(pos)

        def add_base(self, base):
            super().add_base(base)
            import xml.etree.ElementTree as ET
            parent=self.worldbody.find(f".//body[@name='{base.correct_naming('support')}']")
            if parent is None:raise ValueError('PandaOmron support frame is missing')
            # RoboCasa CAM_CONFIGS.DEFAULT, attached to the moving lift support.
            for side,pos,quat in (
                ('left',[-.5,.35,1.05],[.55623853,.29935253,-.37678665,-.6775092]),
                ('right',[-.5,-.35,1.05],[.6775091886520386,.3767866790294647,-.2993525564670563,-.55623859167099])):
                ET.SubElement(parent,'camera',name=self.naming_prefix+'agentview_'+side,
                    pos=' '.join(map(str,pos)),quat=' '.join(map(str,quat)),fovy='60')
            self.cameras=self.get_element_names(self.worldbody,'camera')

    class SourceGripper(GripperModel):
        source_id = ''

        def __init__(self, idn=0):
            data = validate_bundle(self.source_id)
            self._config = data['gripper']
            super().__init__(str(asset_root()/self.source_id/'gripper.xml'), idn=idn)

        @property
        def dof(self):
            return 1

        @property
        def init_qpos(self):
            return np.array(self._config['open_qpos'])

        def format_action(self, action):
            value = np.asarray(action, dtype=float)
            if value.shape != (1,) or not np.isfinite(value).all():
                raise ValueError('Expected one finite gripper command')
            self.current_action = np.clip(value, -1, 1)
            return np.repeat(-self.current_action[0], 2)

        @property
        def _important_geoms(self):
            return {key: self._config['geoms'][side] for key, side in (
                ('left_finger', 'left'), ('right_finger', 'right'),
                ('left_fingerpad', 'left'), ('right_fingerpad', 'right'))}

    class RoboTwinGripper(SourceGripper):
        source_id = 'aloha_agilex_robotwin'

    class RoboDojoGripper(SourceGripper):
        source_id = 'dual_arx_x5_robodojo'

    class SourceDualArm(ManipulatorModel):
        source_id = ''
        arms = ['right', 'left']

        def __init__(self, idn=0):
            self._source_config = validate_bundle(self.source_id)
            super().__init__(str(asset_root()/self.source_id/'robot.xml'), idn=idn)
            # Adjacent joint housings overlap at the bearing. In particular,
            # the fixed base is welded to world, so MuJoCo's automatic parent
            # filter does not suppress its contact with joint 1.
            import xml.etree.ElementTree as ET
            contact=self.root.find('contact')
            for parent in self.worldbody.iter('body'):
                for child in parent.findall('body'):
                    if child.find('joint') is not None:
                        ET.SubElement(contact,'exclude',body1=parent.get('name'),body2=child.get('name'))

        def set_base_xpos(self, pos):
            pos=np.array(pos,dtype=float);pos[2]=max(0.,pos[2])
            super().set_base_xpos(pos)

        @property
        def default_base(self): return 'NullMount'
        @property
        def default_gripper(self):
            name = 'RoboTwinGripper' if self.source_id == 'aloha_agilex_robotwin' else 'RoboDojoGripper'
            return {arm: name for arm in self.arms}
        @property
        def default_controller_config(self): return {arm: 'osc_pose' for arm in self.arms}
        @property
        def init_qpos(self): return np.array(self._source_config['init_qpos'], dtype=float)
        @property
        def base_xpos_offset(self):
            return {'empty': (-.6, 0, 0), 'bins': (-.6, 0, 0), 'table': lambda length: (-.16-length/2, 0, 0)}
        @property
        def top_offset(self): return np.array([0, 0, 1.5])
        @property
        def _horizontal_radius(self): return .8
        @property
        def arm_type(self): return 'bimanual'
        @property
        def _eef_name(self): return {arm: arm+'_hand' for arm in self.arms}

    class RoboTwinAloha(SourceDualArm):
        source_id = 'aloha_agilex_robotwin'

    class RoboDojoX5(SourceDualArm):
        source_id = 'dual_arx_x5_robodojo'

    for cls in (RoboTwinGripper, RoboDojoGripper):
        GRIPPER_MAPPING[cls.__name__] = cls
    for cls in (LiberoPanda, RoboTwinAloha, RoboDojoX5):
        ROBOT_CLASS_MAPPING[cls.__name__] = FixedBaseRobot
    ROBOT_CLASS_MAPPING['RoboCasaPandaOmron'] = WheeledRobot


def resolve(robot_id, *, include_controller=True):
    from react_manip.robots.resolution import RobosuiteResolution
    register_models()
    spec = EMBODIMENTS[robot_id]
    return RobosuiteResolution(robot_id, spec.model,
        controller_configs=controller_config(robot_id) if include_controller else None,
        gripper_types='PandaGripper' if spec.dof == 7 else 'default',
        initialization_noise=None, arms=spec.arms, mobile_base=spec.mobile)


class EmbodimentAdapter:
    def __init__(self, robot_id):
        from react_manip.robots.base import RobotSpec
        s = EMBODIMENTS[robot_id]
        self.spec = RobotSpec(robot_id=robot_id, arm_names=s.arms,
            arm_dof=(s.dof,)*len(s.arms), gripper_type='parallel',
            capabilities=frozenset({'parallel_gripper', 'bimanual' if len(s.arms)==2 else 'single_arm'} |
                                   ({'mobile_base'} if s.mobile else set())),
            metadata={'source': s.source, 'native_action': s.native_action})

    def make(self, **kwargs):
        return resolve(self.spec.robot_id).robot_name

    def validate_assets(self):
        if len(self.spec.arm_names) == 2:
            validate_bundle(self.spec.robot_id)

    def load_mujoco_model(self):
        from robosuite.models.robots import create_robot
        from robosuite.models.grippers import gripper_factory
        model = create_robot(self.make())
        for arm in model.arms:
            model.add_gripper(gripper_factory(model.default_gripper[arm], idn=f'0_{arm}'), model.eef_name[arm])
        if EMBODIMENTS[self.spec.robot_id].mobile:
            from robosuite.models.bases import BASE_MAPPING
            model.add_base(BASE_MAPPING[model.default_base](idn=0))
        return model.get_model()


def preserve_native_action_units(config, robot_id):
    """LIBERO's unit action means 5 cm / 0.5 rad, even on a 50 Hz task.

    Keep task servo gains and physics; temporal adaptation happens in the VLA
    adapter and cannot be implemented by changing the benchmark clock.
    """
    if robot_id != 'panda_libero':return
    for part in config['body_parts'].values():
        part.update(output_max=[.05]*3+[.5]*3,output_min=[-.05]*3+[-.5]*3,
                    input_ref_frame='world',input_type='delta')
