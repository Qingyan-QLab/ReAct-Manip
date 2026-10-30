"""Lazy robosuite integration of pinned official AgileX follower models."""
from pathlib import Path
import os
import numpy as np


def asset_dir():
    return Path(os.environ.get("LEGACY_AGILEX_ASSET_DIR", Path(__file__).resolve().parents[2] / "assets/robots/agilex/legacy"))


def validate_bundle():
    import hashlib, json
    directory = asset_dir()
    manifest = directory / 'provenance.json'
    if not manifest.exists():
        raise FileNotFoundError(
            'Run python -m react_manip.robots.legacy_assets_builder')
    data = json.loads(manifest.read_text())
    lock = Path(__file__).with_name('legacy_agilex_assets.lock.json')
    if data.get('source_lock_sha256') != hashlib.sha256(lock.read_bytes()).hexdigest():
        raise ValueError('Legacy asset lock mismatch; rebuild assets')
    for name, expected in data['generated_sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Modified legacy asset: {name}')


def controller_config(arms=("right",)):
    return {"type": "BASIC", "body_parts": {arm: {
        "type": "OSC_POSE", "input_max": 1, "input_min": -1,
        "output_max": [0.015]*3 + [0.08]*3, "output_min": [-0.015]*3 + [-0.08]*3,
        "kp": 60, "damping_ratio": 1.5, "impedance_mode": "fixed",
        "input_type": "delta", "input_ref_frame": "world", "uncouple_pos_ori": True,
        "goal_update_mode": "achieved", "interpolation": None,
        "gripper": {"type": "GRIP"},
    } for arm in arms}}


def register_agilex():
    """Register classes without importing robosuite during registry discovery."""
    validate_bundle()
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    if "AgileXPiper" in REGISTERED_ROBOTS:
        return
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.models.grippers.gripper_model import GripperModel
    from robosuite.models.grippers import GRIPPER_MAPPING
    from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot

    class AgileXPiperGripper(GripperModel):
        def __init__(self, idn=0):
            super().__init__(str(asset_dir() / "gripper.xml"), idn=idn)

        @property
        def dof(self):
            return 1

        @property
        def init_qpos(self):
            return np.array([0.035, -0.035])

        def format_action(self, action):
            # +1 closes both opposing slides; -1 opens.  The keyboard demo
            # sends a persistent toggle state, so map that state directly to
            # the normalized gripper target instead of accumulating only a
            # small increment on every simulation step.  The downstream
            # position actuators still provide the physical smoothing.
            self.current_action = np.clip(np.asarray(action, dtype=float), -1, 1)
            return np.array([-self.current_action[0], self.current_action[0]])

        @property
        def _important_geoms(self):
            import json
            names = json.loads((asset_dir() / 'provenance.json').read_text())['gripper_collision_names']
            return {'left_finger': names['finger7'], 'right_finger': names['finger8'],
                    'left_fingerpad': names['finger7'], 'right_fingerpad': names['finger8']}

    class AgileXPiper(ManipulatorModel):
        arms = ["right"]
        asset_name = "legacy_piperx.xml"
        fixed_stand = dict(height=.80, half_depth=.09, half_width=.10)

        def __init__(self, idn=0):
            super().__init__(str(asset_dir() / self.asset_name), idn=idn)
            if self.fixed_stand is not None:
                from .mounts import install_fixed_stand
                install_fixed_stand(self, **self.fixed_stand)

        def add_gripper(self, gripper, arm_name=None):
            super().add_gripper(gripper, arm_name)
            from ..cameras import ensure_robot_cameras
            ensure_robot_cameras(self)
            self.cameras = self.get_element_names(self.worldbody, 'camera')

        @property
        def default_base(self):
            return "NullMount"

        @property
        def default_gripper(self):
            return {arm: "AgileXPiperGripper" for arm in self.arms}

        @property
        def default_controller_config(self):
            return {arm: "osc_pose" for arm in self.arms}

        @property
        def init_qpos(self):
            # Calibrated tabletop starting posture used by the physical
            # manipulation tasks.  A zero posture places the wrist high and
            # causes the controller's first goal update to sweep through
            # nearby objects.
            return np.tile([0, 0.8, -1.1, 0, 0.3, 0], len(self.arms))

        @property
        def base_xpos_offset(self):
            return {"empty": (-0.45, 0, 0), "bins": (-0.45, 0, 0),
                    "table": lambda length: (-0.1-length/2, 0, 0)}

        @property
        def top_offset(self):
            return np.array([0, 0, 0.7])

        @property
        def _horizontal_radius(self):
            return 0.7

        @property
        def arm_type(self):
            return "single" if len(self.arms) == 1 else "bimanual"

        @property
        def _eef_name(self):
            return {arm: arm + "_hand" for arm in self.arms}

    class AgileXAloha(AgileXPiper):
        arms = ["right", "left"]
        asset_name = "aloha.xml"
        # Imported dual-arm bases are at x=.23875, y=+/- .30.
        # Support their mounting plane, not the empty asset origin.
        fixed_stand = dict(height=.65, half_depth=.09, half_width=.375, center_xy=(.23875, 0.))

    class AgileXMobileAloha(AgileXAloha):
        asset_name = "aloha_mobile.xml"
        fixed_stand = None

        # FixedBaseRobot is used only for the two arm controllers. The XML
        # still contains the physical free root and ten wheel/caster joints;
        # they must not enter robosuite's scalar arm-qpos reset/indexing.
        @property
        def joints(self):
            return [name for name in super().joints if "mobile_" not in name]

        @property
        def actuators(self):
            return [name for name in super().actuators if "mobile_" not in name]

    GRIPPER_MAPPING["AgileXPiperGripper"] = AgileXPiperGripper
    for cls in (AgileXPiper, AgileXAloha, AgileXMobileAloha):
        ROBOT_CLASS_MAPPING[cls.__name__] = FixedBaseRobot


def resolve_robot(robot_id):
    if robot_id not in ("legacy_piperx", "aloha", "aloha_mobile"):
        raise ValueError(f"not an AgileX robot: {robot_id}")
    register_agilex()
    if robot_id == "aloha_mobile":
        import xml.etree.ElementTree as ET
        if ET.parse(asset_dir() / "aloha_mobile.xml").find(".//freejoint[@name='mobile_free']") is None:
            raise ValueError("Stale planar ALOHA assets: run python scripts/build_agilex_assets.py")
    return "AgileXPiper" if robot_id == "legacy_piperx" else ("AgileXMobileAloha" if robot_id == "aloha_mobile" else "AgileXAloha")
