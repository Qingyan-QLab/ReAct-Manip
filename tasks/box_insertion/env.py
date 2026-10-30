"""Physics-based robot box insertion task.

The block is a passive free rigid body. Its position changes only when contact
forces from the robot end-effector are integrated by MuJoCo; the environment
never writes the block joint ``qpos`` / ``qvel`` during ``step``.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import mujoco
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from react_manip.visual_task import ManipulationTask

from react_manip.core import SuccessRule, TaskSpec
from react_manip.core.scenarios import sample_scenario
from react_manip.core.visual_randomization import apply_visual_scenario, resolve_scenario

ROOT = Path(__file__).resolve().parent
SCENE_XML = ROOT / "assets" / "scene.xml"
BLOCK_FRICTION_PATCHES = ("left", "right")
MIN_SLOT_CLEARANCE_M = 0.05  # total opening margin; 2.5 cm nominal per side
MIN_INITIAL_GRIPPER_CLEARANCE_M = 0.05

SPEC = TaskSpec(
    task_id="box_insertion", task_version="1.7", task_type="one_shot",
    required_arms=1, required_gripper="parallel", required_eef_dof=6,
    supported_robots=("dual_piperx", "panda", "sawyer", "piperx", "mobile_dual_piperx"),
    horizon_sec=60.0, control_hz=20.0, done_behavior="stop",
    success=SuccessRule(min_occupancy=0.0, min_events=1, max_failures=0,
                        require_terminal_success=True),
)


class BoxInsertionArena(Arena):
    def __init__(self, scenario=None):
        super().__init__(str(SCENE_XML))
        self.set_origin([0, 0, 0])
        resolved = resolve_scenario(scenario)
        parameters = resolved.parameters if resolved is not None else {}
        sampled = parameters.get("block_friction", {})
        geoms = {geom.get("name"): geom for geom in self.worldbody.iter("geom")}
        physics = parameters.get("physics", {})
        # Priority 1 makes the wall coefficient the actual contact coefficient,
        # independent of either block patch's table-contact friction.
        for name, key, fallback in (
            ("table", "table_friction", 0.005),
            ("box_left", "slot_left_wall_friction", physics.get("slot_side_wall_friction", 3.0)),
            ("box_right", "slot_right_wall_friction", physics.get("slot_side_wall_friction", 3.0)),
            ("box_back", "slot_back_wall_friction", 0.02),
        ):
            coefficient = float(physics.get(key, fallback))
            if not np.isfinite(coefficient) or coefficient <= 0.0:
                raise ValueError(f"invalid {key}: {coefficient}")
            friction = geoms[name].get("friction").split()
            friction[0] = f"{coefficient:.12g}"
            geoms[name].set("friction", " ".join(friction))
        density = parameters.get("block_density")
        patch_densities = dict.fromkeys(BLOCK_FRICTION_PATCHES, 650.0)
        if density is not None:
            light = float(density["light_kg_m3"])
            heavy = float(density["heavy_kg_m3"])
            heavy_side = density["heavy_side"]
            if (not np.isfinite([light, heavy]).all() or not 0 < light < heavy
                    or heavy_side not in BLOCK_FRICTION_PATCHES):
                raise ValueError(f"invalid block_density: {density}")
            patch_densities = {
                patch: heavy if patch == heavy_side else light
                for patch in BLOCK_FRICTION_PATCHES
            }
        variation = resolved.metadata.get('episode_variation', {}) if resolved else {}
        mass_scale = float(variation.get('mass_scale', 1.))
        if not np.isfinite(mass_scale) or not .7 <= mass_scale <= 1.3:
            raise ValueError('invalid Episode block mass scale')
        # Uniform density scaling changes mass and inertia together, preserving COM.
        patch_densities = {patch: value * mass_scale for patch, value in patch_densities.items()}
        self.block_patch_density = np.array(list(patch_densities.values()))
        block_size = np.asarray(geoms["block_visual"].get("size").split(), dtype=float)
        block_params = parameters.get("wood_block", {})
        width = float(block_params.get("width_m", 2.0 * block_size[1]))
        if not np.isfinite(width) or not 0.60 <= width <= 1.00:
            raise ValueError(f"wood block width must be in [0.60, 1.00] m, got {width}")
        block_size[1] = width / 2.0
        self.block_width = width

        # The slot is derived from the sampled block width. Clearance is a
        # total lateral margin (not per-side). Enforce the
        # contract at construction time as well as in the Profile, so older
        # saved Scenarios with the former 0.0002 m value cannot create a slot
        # that is narrower than the required 5 cm margin.
        box_params = parameters.get("box", {})
        clearance = float(box_params.get("clearance_m", MIN_SLOT_CLEARANCE_M))
        if not np.isfinite(clearance) or clearance <= 0.0:
            raise ValueError(f"slot clearance must be positive, got {clearance}")
        clearance = max(clearance, MIN_SLOT_CLEARANCE_M)
        wall = float(box_params.get("wall", 0.02))
        if not np.isfinite(wall) or wall <= 0.0:
            raise ValueError(f"slot wall thickness must be positive, got {wall}")
        opening = width + clearance
        outer_width = opening + 2.0 * wall
        self.slot_opening_width = opening
        self.slot_outer_width = outer_width
        self.slot_clearance = clearance
        side_half_thickness = wall / 2.0
        side_center = opening / 2.0 + side_half_thickness
        geoms["box_left"].set("pos", f"0 {-side_center:.9g} 0.035")
        geoms["box_right"].set("pos", f"0 {side_center:.9g} 0.035")
        geoms["box_left"].set("size", f"0.16 {side_half_thickness:.9g} 0.035")
        geoms["box_right"].set("size", f"0.16 {side_half_thickness:.9g} 0.035")
        back = geoms["box_back"]
        back_size = np.asarray(back.get("size").split(), dtype=float)
        back_size[1] = outer_width / 2.0
        back.set("size", " ".join(f"{value:.9g}" for value in back_size))
        for site in self.worldbody.iter("site"):
            if site.get("name") == "box_entry":
                site_size = np.asarray(site.get("size").split(), dtype=float)
                site_size[1] = opening / 2.0
                site.set("size", " ".join(f"{value:.9g}" for value in site_size))

        half_width = float(block_size[1])
        full_width = 2.0 * half_width
        split_from_left = sampled.get("split_from_left_m")
        if split_from_left is None:
            # New profiles sample a width-independent ratio.  This keeps the
            # historical 20--80% split rule valid for both 0.60 m and 1.00 m
            # blocks.  Scenarios saved before this change may still provide a
            # ratio or an absolute split distance.
            split_from_left = full_width * float(sampled.get("split_ratio", 0.5))
        split_from_left = float(split_from_left)
        if (not np.isfinite(split_from_left)
                or not 0.0 < split_from_left < full_width):
            raise ValueError(
                "invalid block friction split distance from left edge: "
                f"{split_from_left} m for width {full_width} m"
            )
        split_y = -half_width + split_from_left
        bounds = ((-half_width, split_y), (split_y, half_width))
        for patch, (lower, upper) in zip(BLOCK_FRICTION_PATCHES, bounds):
            coefficient = float(sampled.get(patch, 0.10))
            if not np.isfinite(coefficient) or coefficient <= 0.0:
                raise ValueError(f"invalid block friction for {patch}: {coefficient}")
            geom = geoms[f"block_contact_{patch}"]
            geom.set("pos", f"0 {(lower + upper) / 2.0:.9g} 0")
            geom.set(
                "size",
                f"{block_size[0]:.9g} {(upper - lower) / 2.0:.9g} {block_size[2]:.9g}",
            )
            geom.set("friction", f"{coefficient:.9g} 0.01 0.001")
            # MuJoCo derives one body's mass, offset COM and inertia from the
            # two disjoint solid regions; the visual geom has zero mass.
            geom.set("density", f"{patch_densities[patch]:.12g}")
        geoms["block_visual"].set("size", " ".join(f"{value:.9g}" for value in block_size))
        apply_visual_scenario(self, scenario)


def _default_controller():
    config = load_composite_controller_config(controller="BASIC")
    config["body_parts"]["right"] = {
        "type": "OSC_POSE", "input_max": 1, "input_min": -1,
        "output_max": [0.015, 0.015, 0.015, 0.25, 0.25, 0.25],
        "output_min": [-0.015, -0.015, -0.015, -0.25, -0.25, -0.25],
        "kp": 190, "damping_ratio": 1.5, "impedance_mode": "fixed",
        "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
        "position_limits": None, "orientation_limits": None,
        "uncouple_pos_ori": True, "input_type": "delta",
        "input_ref_frame": "world", "interpolation": None,
        "ramp_ratio": 0.1, "gripper": {"type": "GRIP"},
    }
    return config


class BoxInsertion(ManipulationEnv):
    privileged_info_keys = (
        'task_seed', 'block_patch_friction', 'block_patch_density_kg_m3',
        'block_mass_kg', 'block_com_local_m', 'slot_wall_friction',
        'block_patch_width_m', 'block_width_m', 'slot_opening_width_m',
        'slot_clearance_m', 'block_friction_split_y_m', 'block_friction_split_from_left_m',
    )
    privileged_observation_keys = privileged_info_keys + (
        'block_pose', 'block_velocity', 'block_angular_velocity', 'eef_to_push_face',
    )
    """Single-arm EEF controller that physically pushes a block into a box."""

    @property
    def action_dim(self):
        """Expose robosuite's compiled layout (7 single-arm / 14 bimanual)."""
        return int(getattr(self, "_action_dim", 7))
    # Pygame must not inject an additional _pre_action call: robosuite's base
    # step loop already applies this action at every physics substep.
    step_handles_robot_action = True
    task_space_active_kp = 180.0
    task_space_passive_kp = 20.0
    task_space_rotation_kp = 8.0
    _TARGET_DEPTH = 0.36
    _BOTTOM_TOLERANCE = 0.005
    # MuJoCo's resting contact permits sub-millimetre penetration into the
    # tabletop. Keep this separate from the much stricter bottom-depth rule.
    _CONTAINMENT_TOLERANCE = 0.0005
    _STABLE_SPEED = 0.025
    _STABLE_ANGULAR_SPEED = 0.08
    _STABLE_TICKS = 10
    # Default geometry (1.00 m block + 0.05 m total clearance). Each Scenario
    # replaces this with its sampled width-derived bounds at model load.
    _SLOT_BOUNDS = (0.10, 0.40, -0.5001, 0.5001)
    _SLOT_Z_BOUNDS = (0.78, 0.85)

    def reward(self, action=None):
        """Robosuite hook; task scoring is computed in :meth:`step`.

        Returning a finite scalar keeps the base environment's stepping loop
        intact while avoiding any hidden state writes or legacy object-state
        reward assumptions.
        """
        del action
        return 0.0

    def _task_instruction(self):
        return (
            "Push the free wooden block into the open box slot. "
            "Block contact friction and overall mass vary between Episodes but remain fixed within each Episode. "
            "The block and slot dimensions and the center-of-mass bias pattern stay the same within a Task Instance. "
            "Use observed motion to adjust pushes and correct rotation. "
            "The block must remain inside the slot, reach the required insertion "
            "depth near the back wall, and settle without crossing the side walls. "
            f"Keep it stable for {self._STABLE_TICKS / self.control_freq:g} seconds. "
            "Final-position points are at most 50; success earns up to 50 time points: "
            "full through 10 seconds, linearly decreasing to zero at 40 seconds. "
            f"The deadline is {self.horizon / self.control_freq:g} seconds. "
            "Position points use the final pose, not the maximum earlier insertion. "
            "Invalid side, back or vertical containment gives zero position points; partial entry is allowed for partial credit. "
            "Falling below the table ends the episode with zero. Success ends immediately; timeout retains final-position points."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='box_insertion',
            task_instruction=self._task_instruction(),
            success_condition=(f"All eight corners contained, insertion depth at least {self._TARGET_DEPTH - self._BOTTOM_TOLERANCE:g} m, "
                               f"linear speed <={self._STABLE_SPEED:g} m/s and angular speed <={self._STABLE_ANGULAR_SPEED:g} rad/s "
                               f"for {self._STABLE_TICKS} ticks"),
            constraints=(),
        )

    def __init__(self, robots="dual_piperx", env_configuration="default",
                 controller_configs=None, gripper_types="default",
                 base_types="default", initialization_noise=None,
                 use_camera_obs=False, use_object_obs=False,
                 reward_scale=1.0, reward_shaping=False, has_renderer=False,
                 has_offscreen_renderer=False, render_camera="overview",
                 render_collision_mesh=False, render_visual_mesh=True,
                 render_gpu_device_id=-1, control_freq=20, lite_physics=False,
                 horizon=1200, ignore_done=False, hard_reset=True,
                 camera_names="overview", camera_heights=256,
                 camera_widths=256, camera_depths=False,
                 camera_segmentations=None, renderer="mjviewer",
                 renderer_config=None, seed=None, scenario=None,
                 **kwargs: Any):
        del kwargs
        if robots in ('dual_piperx', 'piperx', 'mobile_dual_piperx', 'panda', 'sawyer', 'legacy_piperx', 'aloha', 'aloha_mobile'):
            from react_manip.robots.resolution import resolve_robosuite_robot
            resolved = resolve_robosuite_robot(robots)
            robots = resolved.robot_name
            if controller_configs is None:
                controller_configs = resolved.controller_configs
        if scenario is None:
            profile = json.loads((ROOT.parents[1] / "configs/box_insertion/profile_default.json").read_text())
            scenario = sample_scenario(profile, seed=0 if seed is None else seed)
        self.scenario = resolve_scenario(scenario)
        self.seed = seed
        self.use_object_obs = use_object_obs
        self.reward_scale = float(reward_scale)
        self.reward_shaping = bool(reward_shaping)
        self._stable_ticks = 0
        self._max_depth_seen = 0.0
        self._events: list[str] = []
        self._failure_reason: str | None = None
        if controller_configs is None:
            controller_configs = _default_controller()
        super().__init__(
            robots=robots, env_configuration=env_configuration,
            controller_configs=controller_configs, base_types=base_types,
            gripper_types=gripper_types, initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera, render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id, control_freq=control_freq,
            lite_physics=lite_physics, horizon=horizon, ignore_done=ignore_done,
            hard_reset=hard_reset, camera_names=camera_names,
            camera_heights=camera_heights, camera_widths=camera_widths,
            camera_depths=camera_depths, camera_segmentations=camera_segmentations,
            renderer=renderer, renderer_config=renderer_config, seed=seed,
        )

    def _load_model(self):
        super()._load_model()
        robot_model = self.robots[0].robot_model
        if robot_model.__class__.__name__ in ("MobileDualPiperX", "AgileXMobileAloha"):
            robot_model.set_base_xpos([-0.72, 0.0, 0.0])
        elif robot_model.__class__.__name__ == "Panda":
            robot_model.set_base_xpos([-0.5, 0.0, -0.1])
        elif robot_model.__class__.__name__ in ("PiperX", "DualPiperX", "AgileXPiper", "AgileXAloha"):
            from react_manip.robots.mounts import set_fixed_mount_pose
            piper = robot_model.__class__.__name__ in ("PiperX", "AgileXPiper")
            height = .62 if piper else .55
            # Preserve calibrated arm world poses without sinking the stand
            # root below the floor to lower the mounting plane.
            set_fixed_mount_pose(robot_model, position=[-.38 if piper else -.62, 0., height],
                                 yaw=0., height=height)
        else:
            robot_model.set_base_xpos([-0.62, 0.0, -0.1])
        if robot_model.__class__.__name__ == "PiperX":
            # The shared start pose places the fingertips inside this task's
            # wide block. Raise/retract the shoulder before controller reset;
            # leave the block, fixture, and robot mounting pose unchanged.
            self.robots[0].init_qpos = np.array([0., .8, -1.1, 1.2, 0., 0.])
        arena = BoxInsertionArena(self.scenario)
        self._block_patch_density = arena.block_patch_density.copy()
        self._block_width_m = arena.block_width
        self._slot_opening_width_m = arena.slot_opening_width
        self._slot_clearance_m = arena.slot_clearance
        self._slot_bounds = (0.10, 0.40, -arena.slot_opening_width / 2.0,
                             arena.slot_opening_width / 2.0)
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[],
        )

    def _setup_references(self):
        super()._setup_references()
        model = self.sim.model
        self.block_body_id = model.body_name2id("wood_block")
        self._block_mass_kg = float(model.body_mass[self.block_body_id])
        self._block_com_local_m = np.asarray(model.body_ipos[self.block_body_id]).copy()
        self._slot_wall_friction = np.array([
            model.geom_friction[model.geom_name2id(name), 0]
            for name in ("box_left", "box_right", "box_back")
        ])
        self._block_geom_ids = {
            model.geom_name2id(f"block_contact_{patch}")
            for patch in BLOCK_FRICTION_PATCHES
        }
        # Kept for compatibility with external diagnostics that previously
        # expected one representative collision geom.
        self.block_geom_id = min(self._block_geom_ids)
        visual_geom_id = model.geom_name2id("block_visual")
        self.block_joint_id = model.joint_name2id("block_free")
        self.block_qpos_addr = int(model.jnt_qposadr[self.block_joint_id])
        self.block_qvel_addr = int(model.jnt_dofadr[self.block_joint_id])
        self._block_start_pos = np.asarray(model.body_pos[self.block_body_id], dtype=float).copy()
        self._block_half_length = float(model.geom_size[visual_geom_id, 0])
        self._block_half_width = float(model.geom_size[visual_geom_id, 1])
        self._block_half_height = float(model.geom_size[visual_geom_id, 2])
        self._block_patch_friction = np.asarray([
            model.geom_friction[model.geom_name2id(f"block_contact_{patch}"), 0]
            for patch in BLOCK_FRICTION_PATCHES
        ], dtype=float)
        patch_geom_ids = [
            model.geom_name2id(f"block_contact_{patch}")
            for patch in BLOCK_FRICTION_PATCHES
        ]
        self._block_patch_width_m = np.asarray(
            [2.0 * model.geom_size[geom_id, 1] for geom_id in patch_geom_ids],
            dtype=float,
        )
        left_geom_id = patch_geom_ids[0]
        self._block_friction_split_y_m = float(
            model.geom_pos[left_geom_id, 1] + model.geom_size[left_geom_id, 1]
        )
        self._block_friction_split_from_left_m = float(
            self._block_friction_split_y_m + self._block_half_width
        )
        self._gripper_geom_ids = set()
        for gripper in self.robots[0].gripper.values():
            for name in gripper.contact_geoms:
                try:
                    self._gripper_geom_ids.add(model.geom_name2id(name))
                except (ValueError, KeyError):
                    pass
        sites = self.robots[0].eef_site_id
        self.eef_site_id = sites["right"] if isinstance(sites, dict) else int(sites)
        gripper_indexes = self.robots[0]._ref_gripper_joint_pos_indexes
        if isinstance(gripper_indexes, dict):
            gripper_indexes = next(iter(gripper_indexes.values()), [])
        self._gripper_qpos = np.asarray(gripper_indexes, dtype=int)

    def _block_depth(self):
        displacement = self._block_pose()[0] - self._block_start_pos[0]
        return float(np.clip(displacement, 0.0, self._TARGET_DEPTH))

    def _block_pose(self):
        return np.asarray(self.sim.data.xpos[self.block_body_id], dtype=float).copy()

    def _block_velocity(self):
        # cvel is angular then linear in the body frame; this is sufficient
        # for the low-speed stability criterion after converting to world.
        return np.asarray(self.sim.data.cvel[self.block_body_id, 3:6], dtype=float).copy()

    def _block_angular_velocity(self):
        return np.asarray(self.sim.data.cvel[self.block_body_id, :3], dtype=float).copy()

    def _block_yaw(self):
        rotation = np.asarray(self.sim.data.xmat[self.block_body_id], dtype=float).reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _block_corners(self):
        """World corners of the rotated visual bounding box."""
        center = self._block_pose()
        rotation = np.asarray(self.sim.data.xmat[self.block_body_id], dtype=float).reshape(3, 3)
        local = np.array([
            [sx * self._block_half_length, sy * self._block_half_width, sz * self._block_half_height]
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
        ])
        return center + local @ rotation.T

    def _block_contained(self):
        """Whether every rotated 3D corner of the block is inside the slot."""
        corners = self._block_corners()
        xmin, xmax, ymin, ymax = self._slot_bounds
        zmin, zmax = self._SLOT_Z_BOUNDS
        tolerance = self._CONTAINMENT_TOLERANCE
        return bool(
            np.all(corners[:, 0] >= xmin - tolerance)
            and np.all(corners[:, 0] <= xmax + tolerance)
            and np.all(corners[:, 1] >= ymin - tolerance)
            and np.all(corners[:, 1] <= ymax + tolerance)
            and np.all(corners[:, 2] >= zmin - tolerance)
            and np.all(corners[:, 2] <= zmax + tolerance)
        )

    def _bottom_reached(self):
        """Whether the block center is within tolerance of the rear wall."""
        return bool(
            self._block_depth() >= self._TARGET_DEPTH - self._BOTTOM_TOLERANCE
        )

    def _success_blockers(self, *, speed=None, angular_speed=None):
        """Return explicit reasons why the current state is not terminal success.

        A block can touch the rear wall while still being yawed or pressed
        against a side wall.  Reporting each independent condition makes that
        physically meaningful distinction visible to interactive users and
        benchmark logs instead of reducing it to a mysterious false result.
        """
        if speed is None:
            speed = float(np.linalg.norm(self._block_velocity()))
        if angular_speed is None:
            angular_speed = float(np.linalg.norm(self._block_angular_velocity()))
        blockers = []
        if not self._block_contained():
            blockers.append("block_not_contained")
        if not self._bottom_reached():
            blockers.append("slot_bottom_not_reached")
        if speed > self._STABLE_SPEED:
            blockers.append("linear_motion_not_stable")
        if angular_speed > self._STABLE_ANGULAR_SPEED:
            blockers.append("angular_motion_not_stable")
        return blockers

    def _contact_force(self):
        peak = 0.0
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 not in self._block_geom_ids and geom2 not in self._block_geom_ids:
                continue
            other = geom2 if geom1 in self._block_geom_ids else geom1
            if other not in self._gripper_geom_ids:
                continue
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.sim.model._model, self.sim.data._data, index, force)
            peak = max(peak, float(np.linalg.norm(force[:3])))
        return peak

    def _observation(self):
        data = self.sim.data
        eef_pos = np.asarray(data.site_xpos[self.eef_site_id], dtype=np.float32).copy()
        eef_mat = np.asarray(data.site_xmat[self.eef_site_id], dtype=np.float32).copy()
        block_pose = self._block_pose().astype(np.float32)
        return {
            "robot_qpos": data.qpos[self.robots[0]._ref_joint_pos_indexes].astype(np.float32).copy(),
            "robot_qvel": data.qvel[self.robots[0]._ref_joint_vel_indexes].astype(np.float32).copy(),
            "eef_pos": eef_pos, "eef_mat": eef_mat,
            "gripper_qpos": data.qpos[self._gripper_qpos].astype(np.float32).copy(),
            "block_pose": block_pose,
            "block_velocity": self._block_velocity().astype(np.float32),
            "block_angular_velocity": self._block_angular_velocity().astype(np.float32),
            "block_patch_friction": self._block_patch_friction.astype(np.float32).copy(),
            "block_patch_density_kg_m3": self._block_patch_density.astype(np.float32).copy(),
            "block_mass_kg": np.array([self._block_mass_kg], dtype=np.float32),
            "block_com_local_m": self._block_com_local_m.astype(np.float32).copy(),
            "slot_wall_friction": self._slot_wall_friction.astype(np.float32).copy(),
            "block_patch_width_m": self._block_patch_width_m.astype(np.float32).copy(),
            "block_width_m": np.array([self._block_width_m], dtype=np.float32),
            "slot_opening_width_m": np.array([self._slot_opening_width_m], dtype=np.float32),
            "slot_clearance_m": np.array([self._slot_clearance_m], dtype=np.float32),
            "block_friction_split_y_m": np.array(
                [self._block_friction_split_y_m], dtype=np.float32
            ),
            "block_friction_split_from_left_m": np.array(
                [self._block_friction_split_from_left_m], dtype=np.float32
            ),
            "block_lateral_m": np.array([block_pose[1] - self._block_start_pos[1]], dtype=np.float32),
            "block_yaw_rad": np.array([self._block_yaw()], dtype=np.float32),
            "block_contained": np.array([self._block_contained()], dtype=np.bool_),
            "bottom_reached": np.array([self._bottom_reached()], dtype=np.bool_),
            "insertion_depth_m": np.array([self._block_depth()], dtype=np.float32),
            "insertion_fraction": np.array([self._block_depth() / self._TARGET_DEPTH], dtype=np.float32),
            "eef_to_push_face": block_pose - eef_pos + np.array([-self._block_half_length, 0, 0], dtype=np.float32),
            "contact_force_N": np.array([self._contact_force()], dtype=np.float32),
            "time_remaining_sec": np.array([max(0.0, SPEC.horizon_sec - self.timestep / self.control_freq)], dtype=np.float32),
        }

    def _score_info(self):
        from .scoring import score_state
        depth = self._block_depth()
        score = score_state(center=self._block_pose(), corners=self._block_corners(),
                            bounds=self._slot_bounds, z_bounds=self._SLOT_Z_BOUNDS,
                            half_length=self._block_half_length, elapsed=self.timestep/self.control_freq,
                            success=bool(self._events), failed=self._failure_reason is not None)
        return {
            **score,
            "elapsed_sec": float(self.timestep / self.control_freq),
            "insertion_depth_m": depth,
            "max_insertion_depth_m": self._max_depth_seen,
            "insertion_fraction": float(depth / self._TARGET_DEPTH),
            "contact_peak_force_N": self._contact_force(),
            "stable_ticks": float(self._stable_ticks),
            "block_contained": self._block_contained(),
            "bottom_reached": self._bottom_reached(),
            "remaining_insertion_distance_m": float(
                max(0.0, self._TARGET_DEPTH - depth)
            ),
            "block_lateral_m": float(self._block_pose()[1] - self._block_start_pos[1]),
            "block_yaw_rad": self._block_yaw(),
            "block_patch_friction": self._block_patch_friction.tolist(),
            "task_seed": self.scenario.seed,
            "block_patch_density_kg_m3": self._block_patch_density.tolist(),
            "block_mass_kg": self._block_mass_kg,
            "block_com_local_m": self._block_com_local_m.tolist(),
            "slot_wall_friction": self._slot_wall_friction.tolist(),
            "block_patch_width_m": self._block_patch_width_m.tolist(),
            "block_width_m": self._block_width_m,
            "slot_opening_width_m": self._slot_opening_width_m,
            "slot_clearance_m": self._slot_clearance_m,
            "block_friction_split_y_m": self._block_friction_split_y_m,
            "block_friction_split_from_left_m": self._block_friction_split_from_left_m,
        }

    def _reset_internal(self):
        super()._reset_internal()
        self._stable_ticks = 0
        self._max_depth_seen = self._block_depth()
        self._events = []
        self._failure_reason = None

    def _initial_gripper_clearance(self):
        """Minimum collision-surface gap to the block, slot, and tabletop."""
        model, data = self.sim.model._model, self.sim.data._data
        obstacles = [self.sim.model.geom_name2id(name) for name in (
            "block_contact_left", "block_contact_right", "box_left",
            "box_right", "box_back", "table",
        )]
        gripper_geoms = [self.sim.model.geom_name2id(name)
                         for gripper in self.robots[0].gripper.values()
                         for name in gripper.contact_geoms]
        return min(mujoco.mj_geomDistance(model, data, hand, obstacle, 1., None)
                   for hand in gripper_geoms for obstacle in obstacles)

    def reset(self, *, seed=None, **kwargs):
        del kwargs
        if seed is not None:
            self.seed = seed
        super().reset()
        if self.robots[0].robot_model.__class__.__name__ == "PiperX":
            clearance = self._initial_gripper_clearance()
            if clearance < MIN_INITIAL_GRIPPER_CLEARANCE_M:
                raise ValueError(f"Initial gripper clearance {clearance:.6f} m is below "
                                 f"{MIN_INITIAL_GRIPPER_CLEARANCE_M:.3f} m")
        return self._observation(), {
            "success_state": False,
            "success_blockers": self._success_blockers(),
            "events": [],
            **self._score_info(),
        }

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"expected finite action shape {(self.action_dim,)}")
        if self._failure_reason is not None or self._events:
            raise RuntimeError("episode finished; call reset")
        previous = self._score_info()
        super().step(np.clip(action, -1.0, 1.0))
        depth = self._block_depth()
        self._max_depth_seen = max(self._max_depth_seen, depth)
        speed = float(np.linalg.norm(self._block_velocity()))
        angular_speed = float(np.linalg.norm(self._block_angular_velocity()))
        completed = (
            self._block_contained()
            and self._bottom_reached()
            and speed <= self._STABLE_SPEED
            and angular_speed <= self._STABLE_ANGULAR_SPEED
        )
        self._stable_ticks = self._stable_ticks + 1 if completed else 0
        tick_events: list[str] = []
        if self._stable_ticks >= self._STABLE_TICKS:
            self._events.append("block_inserted")
            tick_events.append("block_inserted")
        finite = bool(np.isfinite(self.sim.data.qpos).all() and np.isfinite(self.sim.data.qvel).all())
        if not finite:
            self._failure_reason = "non_finite_state"
        elif self._block_pose()[2] < 0.65:
            self._failure_reason = "block_fell_from_table"
        if self._failure_reason is not None:
            self._events.clear()
            tick_events.clear()
        current = self._score_info()
        current_total = current["time_score"] + current["distance_score"]
        previous_total = previous["time_score"] + previous["distance_score"]
        current["total_score"] = float(current_total)
        info = {
            "success_state": bool(tick_events), "terminal_success": bool(tick_events),
            "events": tick_events,
            "failure": {"recoverable": False, "reason": self._failure_reason} if self._failure_reason else None,
            "success_blockers": self._success_blockers(speed=speed, angular_speed=angular_speed),
            "score_delta": float(current_total - previous_total), **current,
        }
        terminated = bool(tick_events or self._failure_reason)
        truncated = bool(self.timestep >= self.horizon and not terminated)
        return self._observation(), float(info["score_delta"]), terminated, truncated, info

    def close(self):
        super().close()
