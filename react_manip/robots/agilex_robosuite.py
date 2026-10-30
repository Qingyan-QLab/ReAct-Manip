"""Lazy robosuite integration of pinned official AgileX follower models."""
from pathlib import Path
import os
import numpy as np


def asset_dir():
    override = os.environ.get("AGILEX_ASSET_DIR")
    if override:
        return Path(override)
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "scripts/build_agilex_assets.py").is_file():
        return root / "assets/robots/agilex"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "react-manip/robots/agilex"


def validate_bundle():
    """Reject stale or modified generated bundles, including old grasp aids."""
    import hashlib
    import json
    import xml.etree.ElementTree as ET
    directory = asset_dir()
    manifest = directory / "provenance.json"
    if not manifest.exists():
        raise FileNotFoundError("Run react-manip-assets prepare to build PiPER-X assets")
    data = json.loads(manifest.read_text())
    if data.get("schema") != "piperx-family.v1":
        raise ValueError("Legacy AgileX assets: rebuild with scripts/build_agilex_assets.py")
    lock = Path(__file__).resolve().parents[2] / "scripts/piperx_assets.lock.json"
    if data.get("source_lock_sha256") != hashlib.sha256(lock.read_bytes()).hexdigest():
        raise ValueError("PiPER-X source lock changed; rebuild the asset bundle")
    for name in ("piperx.xml", "dual_piperx.xml", "mobile_dual_piperx.xml", "gripper.xml"):
        if name not in data.get("generated_sha256", {}):
            raise ValueError(f"Incomplete PiPER-X manifest: {name}")
    for name, expected in data["generated_sha256"].items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Stale or modified PiPER-X asset {path}; rebuild the bundle")
    gripper = ET.parse(directory / "gripper.xml")
    for geom in gripper.iter("geom"):
        if geom.get("name", "").endswith(("_pad", "_flat_pad")):
            raise ValueError("Unexpected legacy grasp-helper geometry")
        if "_collision" in geom.get("name", "") and (geom.get("contype", "1") == "0" or geom.get("conaffinity", "1") == "0"):
            raise ValueError("Disabled physical gripper surface")


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
    if "PiperX" in REGISTERED_ROBOTS:
        return
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.models.grippers.gripper_model import GripperModel
    from robosuite.models.grippers import GRIPPER_MAPPING
    from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot

    class PiperXGripper(GripperModel):
        def __init__(self, idn=0):
            super().__init__(str(asset_dir() / "gripper.xml"), idn=idn)
            # Simulation tuning, independent of the checksum-locked geometry.
            # The generated 1 N s/m damping lets a full-stroke position command
            # overshoot the slide stops and interlock the opposing finger meshes.
            # Dampen the physical joints, without clipping positions or removing
            # contact surfaces. Tasks can still apply their validated settings.
            for joint in self.worldbody.iter("joint"):
                if joint.get("type") == "slide":
                    joint.set("damping", "8")

        @property
        def dof(self):
            return 1

        @property
        def init_qpos(self):
            return np.array([0.04, -0.04])

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
            # Every convex part participates in robosuite contact detection.
            import json
            names = json.loads((asset_dir() / "provenance.json").read_text())["gripper_collision_names"]
            return {"left_finger": names["finger7"], "right_finger": names["finger8"],
                    "left_fingerpad": names["finger7"], "right_fingerpad": names["finger8"]}


    class PiperX(ManipulatorModel):
        arms = ["right"]
        asset_name = "piperx.xml"
        fixed_stand = dict(height=.80, half_depth=.09, half_width=.10)

        def __init__(self, idn=0):
            super().__init__(str(asset_dir() / self.asset_name), idn=idn)
            if self.fixed_stand is not None:
                from .mounts import install_fixed_stand
                install_fixed_stand(self, **self.fixed_stand)

        def add_gripper(self, gripper, arm_name=None):
            super().add_gripper(gripper, arm_name)
            # Install cameras on the embodiment before any task merges XML.
            # Covers tasks that use robosuite's own ManipulationTask as well.
            from ..cameras import ensure_robot_cameras
            ensure_robot_cameras(self)
            self.cameras = self.get_element_names(self.worldbody, 'camera')

        @property
        def default_base(self):
            return "NullMount"

        @property
        def default_gripper(self):
            return {arm: "PiperXGripper" for arm in self.arms}

        @property
        def default_controller_config(self):
            return {arm: "osc_pose" for arm in self.arms}

        @property
        def init_qpos(self):
            # Official PiPER-X joint frames differ from the old extracted arm.
            # Start with the fingers pointing toward the tabletop, within the
            # official limits and with margin to the wrist limit.
            return np.tile([0, 1.2, -1.1, 1.2, 0, 0], len(self.arms))

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

    class DualPiperX(PiperX):
        arms = ["right", "left"]
        asset_name = "dual_piperx.xml"
        # Imported dual-arm bases are at x=.23875, y=+/- .30.
        # Support their mounting plane, not the empty asset origin.
        fixed_stand = dict(height=.65, half_depth=.09, half_width=.375, center_xy=(.23875, 0.))

    class MobileDualPiperX(DualPiperX):
        asset_name = "mobile_dual_piperx.xml"
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

    GRIPPER_MAPPING["PiperXGripper"] = PiperXGripper
    for cls in (PiperX, DualPiperX, MobileDualPiperX):
        ROBOT_CLASS_MAPPING[cls.__name__] = FixedBaseRobot


def resolve_robot(robot_id):
    if robot_id in ('legacy_piperx', 'aloha', 'aloha_mobile'):
        from .legacy_agilex_robosuite import resolve_robot as resolve_legacy
        return resolve_legacy(robot_id)
    if robot_id not in ("piperx", "dual_piperx", "mobile_dual_piperx"):
        raise ValueError(f"not an AgileX robot: {robot_id}")
    register_agilex()
    if robot_id == "mobile_dual_piperx":
        import xml.etree.ElementTree as ET
        if ET.parse(asset_dir() / "mobile_dual_piperx.xml").find(".//freejoint[@name='mobile_free']") is None:
            raise ValueError("Stale planar ALOHA assets: run python scripts/build_agilex_assets.py")
    return "PiperX" if robot_id == "piperx" else ("MobileDualPiperX" if robot_id == "mobile_dual_piperx" else "DualPiperX")


def validate_compiled_grippers(env):
    """Task composition may tune materials, but cannot replace contact shapes."""
    import json
    raw = getattr(env, 'env', env)
    robots = getattr(raw, 'robots', ())
    if not any(getattr(robot, 'name', '') in ('PiperX','DualPiperX','MobileDualPiperX','AgileXPiper','AgileXAloha','AgileXMobileAloha') for robot in robots):
        return
    model = raw.sim.model
    for robot in robots:
        if getattr(robot, 'name', '') not in ('PiperX','DualPiperX','MobileDualPiperX','AgileXPiper','AgileXAloha','AgileXMobileAloha'):
            continue
        legacy = getattr(robot, 'name', '').startswith('AgileX')
        directory = asset_dir()
        if legacy:
            from .legacy_agilex_robosuite import asset_dir as legacy_asset_dir
            directory = legacy_asset_dir()
        manifest = json.loads((directory / 'provenance.json').read_text())
        for gripper in robot.gripper.values():
            if type(gripper).__name__ != ('AgileXPiperGripper' if legacy else 'PiperXGripper'):
                raise ValueError('PiPER-X must use the central PiperXGripper asset')
            prefix = gripper.naming_prefix
            expected = {prefix + name for parts in manifest['gripper_collision_names'].values() for name in parts}
            finger_bodies = {model.body_name2id(prefix + name) for name in ('finger7', 'finger8')}
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] in finger_bodies and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
                    if model.geom_id2name(geom_id) not in expected:
                        raise ValueError('Task added an unapproved PiPER-X finger collision shape')
            for parts in manifest['gripper_collision_names'].values():
                for name in parts:
                    c = model.geom_name2id(prefix + name)
                    v = model.geom_name2id(prefix + name.replace('_collision','_visual'))
                    if (model.geom_type[c] != model.geom_type[v]
                        or not np.allclose(model.geom_size[c], model.geom_size[v], atol=1e-10)
                        or model.geom_dataid[c] != model.geom_dataid[v]
                        or not np.allclose(model.geom_pos[c],model.geom_pos[v],atol=1e-10)
                        or not np.allclose(model.geom_quat[c],model.geom_quat[v],atol=1e-10)
                        or not model.geom_contype[c] or not model.geom_conaffinity[c]
                        or model.geom_contype[v] or model.geom_conaffinity[v]
                        or model.geom_margin[c] != 0):
                        raise ValueError(f'Task changed PiPER-X contact surface: {prefix + name}')
