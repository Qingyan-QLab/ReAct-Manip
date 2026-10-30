"""Physical single-ball pick-and-throw task. Only reset assigns ball state."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from robosuite.models.objects import BallObject

from react_manip.core.visual_randomization import apply_visual_scenario, resolve_scenario
from react_manip.robots.resolution import default_controller_config
from react_manip.visual_task import ManipulationTask

ROOT = Path(__file__).resolve().parent
BALL_R = 0.025
TRAY_POS = np.array([0.60, 0.22, 0.65])
RELEASE_POS = np.array([0.5, 0.0, 1.0])  # reference for the legacy distance convention
POT_MOUTH_Z = 0.30
POT_CAVITY_R = 0.09


class PitchPot(ManipulationEnv):
    """Per-arm OSC action: world xyz / axis-angle deltas, gripper (-1 open).

    A step advances one control period, including during flight. The ball is
    never attached to the hand or assigned a launch velocity. One ball per
    episode; reset is the only way to replenish it.
    """
    task_id = "pitch_pot"
    robot_control_mode = "eef_delta"
    step_handles_robot_action = True

    def _task_instruction(self):
        return (
            "Grasp the ball from the tray, swing the arm, and release it into the pot. "
            "The ball must enter the pot through a physical throw and remain inside "
            "for at least 0.5 seconds. "
            "Do not place the ball directly into the pot. First grasp for 0.1 seconds and lift "
            "at least 0.06 m above the initial ball height. Open the gripper deliberately; release speed must be >=0.2 m/s. "
            "Inside the pot, remain free of gripper contact with speed <0.08 m/s for 0.5 seconds. "
            "Slipping, releasing too slowly, contacting the tray after opening the gripper, falling to the floor, "
            "or resting outside the pot for 0.5 seconds ends the attempt with zero points. "
            f"Success earns 100 and ends the episode; timeout at {self.horizon / self.control_freq:g} seconds earns zero."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='pitch_pot',
            task_instruction=self._task_instruction(),
            success_condition='The thrown ball stays inside the pot for at least 0.5 seconds.',
            constraints=(),
        )
    privileged_observation_keys = ("object-state", "obj_pos", "obj_vel", "pot_pos")

    def __init__(self, robots="Panda", *, seed=0, scenario=None,
                 controller_configs=None, gripper_types="default",
                 initialization_noise=None, control_freq=50, horizon=3000,
                 has_renderer=False, has_offscreen_renderer=False,
                 use_camera_obs=False, use_object_obs=False, object_type="ball",
                 n_balls=1, **kwargs):
        from react_manip.robots.resolution import robot_id_from_model, resolve_robosuite_robot
        resolved = resolve_robosuite_robot(robot_id_from_model(robots))
        self.robot_id = resolved.robot_id
        robots = resolved.robot_name
        if gripper_types == "default":
            gripper_types = resolved.gripper_types
        if object_type != "ball" or n_balls != 1:
            raise ValueError("Physical pitch_pot uses one ball per episode")
        self.scenario = resolve_scenario(scenario)
        if self.scenario is None:
            import json
            from react_manip.core.scenarios import sample_scenario
            profile = ROOT.parents[1] / "configs/pitch_pot/profile_default.json"
            self.scenario = sample_scenario(json.loads(profile.read_text()), seed=seed or 0)
        if self.scenario.task_id != self.task_id:
            raise ValueError("Scenario task_id must be pitch_pot")
        p = self.scenario.parameters
        self.embodiment = p.get("embodiment", {})
        self.active_arm = self.embodiment.get("active_arm", "right")
        if self.active_arm not in resolved.arms:
            raise ValueError("active_arm is unavailable on this robot")
        self.tray_pos = np.asarray(p.get("tray", {}).get("position_m", TRAY_POS), float)
        if self.tray_pos.shape != (3,) or not np.isfinite(self.tray_pos).all() or self.tray_pos[2] <= .1:
            raise ValueError("Invalid tray.position_m")
        self.pickup_height = float(p.get("tray", {}).get("pickup_height_m", 0.0))
        if not np.isfinite(self.pickup_height) or not 0 <= self.pickup_height <= .2:
            raise ValueError("Invalid tray.pickup_height_m")
        self.ball_radius = float(p.get("ball", {}).get("radius_m", BALL_R))
        self.ball_mass = float(p.get("ball", {}).get("mass_kg", 0.10))
        self.ball_friction = float(p.get("ball", {}).get("friction", 0.8))
        pot = p.get("pot", {})
        self.pot_y = float(pot.get("y_m", 0.0))
        if "base_distance_m" in pot:
            distance = float(pot["base_distance_m"])
            if not np.isfinite([distance, self.pot_y]).all() or not 0.5 <= distance <= 10 or abs(self.pot_y) > distance:
                raise ValueError("pot.base_distance_m must be in [0.5, 10] metres")
            self.pot_x = float(np.sqrt(distance**2 - self.pot_y**2))
        else:
            # Explicit legacy / near-calibration Scenarios keep their meaning.
            self.pot_x = RELEASE_POS[0] + float(pot.get("distance_m", 0.28))
        self.flight_bounds = max(8.0, self.pot_x + 2.0)
        self.ball_xy = np.asarray(p.get("ball", {}).get("start_xy_m", self.tray_pos[:2]), float)
        if (not np.isfinite([self.ball_radius, self.ball_mass, self.ball_friction,
                             self.pot_x, self.pot_y]).all()
                or not 0.01 <= self.ball_radius <= 0.03 or self.ball_mass <= 0
                or self.ball_friction <= 0 or self.ball_xy.shape != (2,)
                or not np.isfinite(self.ball_xy).all()
                or np.any(np.abs(self.ball_xy - self.tray_pos[:2]) > [0.06, 0.09])):
            raise ValueError("Invalid ball/pot configuration")
        kwargs.setdefault("render_camera", "agentview")
        kwargs.setdefault("camera_names", "agentview")
        self.use_object_obs = use_object_obs
        self._reset_tracking()
        cfg = deepcopy(controller_configs or default_controller_config(robots))
        parts = cfg["body_parts"]
        self.position_scale = 1.0 if "base_distance_m" in pot else 0.20
        self.rotation_scale = 0.20
        for arm in parts.get("arms", parts).values():
            if arm["type"] != "OSC_POSE" or arm.get("impedance_mode", "fixed") != "fixed":
                raise ValueError("pitch_pot requires fixed-impedance OSC_POSE")
            arm.update(input_ref_frame="world", input_type="delta", kp=300, damping_ratio=1,
                       output_max=[self.position_scale] * 3 + [self.rotation_scale] * 3,
                       output_min=[-self.position_scale] * 3 + [-self.rotation_scale] * 3)
        from react_manip.optional_integrations import preserve_native_action_units
        preserve_native_action_units(cfg, self.robot_id)
        super().__init__(robots=robots, controller_configs=cfg,
                         gripper_types=gripper_types, initialization_noise=initialization_noise,
                         control_freq=control_freq, horizon=horizon, lite_physics=False,
                         has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer,
                         use_camera_obs=use_camera_obs, seed=seed, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(self.embodiment.get("base_pos", [0, 0, 0]))
        from .model_geometry import adapt_robot_geometry
        adapt_robot_geometry(self.robots[0])
        arena = Arena(str(ROOT / "assets/pitch_pot_arena.xml"))
        # Move only the pot, not the tray, lights or robot.
        arena.worldbody.find("body[@name='pot']").set("pos", f"{self.pot_x} {self.pot_y} 0")
        # The tray visual must not duplicate its collision surface.
        tray = arena.worldbody.find("body[@name='tray']")
        tray.set("pos", " ".join(map(str, self.tray_pos - [0.6, 0, .65])))
        tray.find("geom[@name='tray_floor']").set("contype", "0")
        tray.find("geom[@name='tray_floor']").set("conaffinity", "0")
        # Compact pickup tray leaves a clear forward flight corridor.
        for name in ("tray_floor", "tray_floor_g0"):
            tray.find(f"geom[@name='{name}']").set("size", "0.10 0.14 0.01")
        for name, x in (("tray_rim_px", .70), ("tray_rim_nx", .50)):
            tray.find(f"geom[@name='{name}']").set("pos", f"{x} 0 0.655")
        for name in ("tray_rim_py", "tray_rim_ny"):
            tray.find(f"geom[@name='{name}']").set("size", "0.10 0.005 0.015")
        half_width = float(self.scenario.parameters.get("tray", {}).get("half_width_m", .14))
        if not np.isfinite(half_width) or not .03 <= half_width <= .3:
            raise ValueError("Invalid tray.half_width_m")
        for name in ("tray_floor", "tray_floor_g0"):
            tray.find(f"geom[@name='{name}']").set("size", f"0.10 {half_width} 0.01")
        for name in ("tray_rim_px", "tray_rim_nx"):
            tray.find(f"geom[@name='{name}']").set("size", f"0.005 {half_width} 0.015")
        for name, sign in (("tray_rim_py", 1), ("tray_rim_ny", -1)):
            tray.find(f"geom[@name='{name}']").set("pos", f"0.6 {sign*half_width} .655")
        if self.scenario.parameters.get("tray", {}).get("rimless", False):
            for geom in list(tray.findall("geom")):
                if geom.get("name", "").startswith("tray_rim_"):
                    tray.remove(geom)
        # Visible support instead of a floating tray; does not obstruct grasping.
        support_half = (self.tray_pos[2] - .02) / 2
        ET.SubElement(tray, "geom", name="tray_support", type="box",
                      pos=f"0.60 0 {support_half - (self.tray_pos[2] - .65)}",
                      size=f"0.08 {min(.09,half_width)} {support_half}",
                      rgba="0.3 0.3 0.32 1", group="1")
        if self.pickup_height > 0:
            ET.SubElement(arena.worldbody, "geom", name="tray_ball_pedestal", type="cylinder",
                          pos=" ".join(map(str, [*self.ball_xy, self.tray_pos[2] + self.pickup_height/2])),
                          size=f"0.014 {self.pickup_height/2}", group="1", rgba="0.45 0.45 0.48 1")
        from react_manip.cameras import look_at
        span = max(1.0, self.pot_x)
        position = np.array([span * .5, -max(2.1, span * 1.35), max(1.45, span * .55)])
        target = np.array([span * .5, 0, .55])
        rotation = look_at(position, target)
        ET.SubElement(arena.worldbody, "camera", name="pitch_side",
                      pos=" ".join(map(str, position)),
                      xyaxes=" ".join(map(str, np.r_[rotation[:, 0], rotation[:, 1]])), fovy="50")
        apply_visual_scenario(arena, self.scenario)
        # Keep the entire 10 m lane inside the visible room. The shared
        # visual randomizer defaults to walls at +/-8 m.
        for i, (pos, size) in enumerate((
                ("0 -16 3", "16 .05 3"), ("0 16 3", "16 .05 3"),
                ("-16 0 3", ".05 16 3"), ("16 0 3", ".05 16 3"))):
            wall = arena.worldbody.find(f"geom[@name='room_wall_visual_{i}']")
            if wall is not None:
                wall.set("pos", pos)
                wall.set("size", size)
        arena.worldbody.find("geom[@name='floor']").set("size", "16 16 .05")
        self.obj = BallObject("ball", size=[self.ball_radius],
                              density=self.ball_mass / (4 / 3 * np.pi * self.ball_radius**3),
                              friction=[self.ball_friction, 0.005, 0.0001],
                              solref=[0.01, 1], rgba=[0.9, 0.25, 0.2, 1])
        self.model = ManipulationTask(mujoco_arena=arena,
                                     mujoco_robots=[r.robot_model for r in self.robots],
                                     mujoco_objects=[self.obj])
        self.camera_workspace = dict(center=(0.65, 0, 0.65), half_size=(0.60, 0.45))

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.obj_body_id = m.body_name2id(self.obj.root_body)
        self.obj_jnt_addr = m.get_joint_qpos_addr(self.obj.joints[0])[0]
        self.obj_qvel_addr = m.get_joint_qvel_addr(self.obj.joints[0])[0]
        self.obj_geom_ids = {m.geom_name2id(n) for n in self.obj.contact_geoms}
        self.pot_body_id = m.body_name2id("pot")
        self.tray_geom_ids = {i for i in range(m.ngeom)
                              if (m.geom_id2name(i) or "").startswith("tray_")}
        gripper = self.robots[0].gripper[self.active_arm]
        self.finger_groups = [
            {m.geom_name2id(n) for n in gripper.important_geoms[key]}
            for key in ("left_finger", "right_finger")
        ]
        self.gripper_geom_ids = {m.geom_name2id(n) for n in gripper.contact_geoms}

    def _reset_tracking(self):
        self.phase = "ready"
        self._grasp_time = 0.0
        self._settle_time = 0.0
        self._lost_contact_time = 0.0
        self._rest_outside_time = 0.0
        self._release_sample = None
        self._grasped = self._lifted = self._released = self._entered = False
        self._success = self._finished = False
        self._failure = None
        self._open_requested = False
        self.attempts = 0
        self.flight_platform_contacts = 0
        self.release_position = self.release_velocity = self.release_sample_time = None

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_tracking()
        # Reset-only initialization; all subsequent movement is dynamics.
        self.sim.data.set_joint_qpos(self.obj.joints[0],
            np.r_[self.ball_xy, self.tray_pos[2] + self.pickup_height + self.ball_radius + 0.001, 1, 0, 0, 0])
        self.sim.data.qvel[self.obj_qvel_addr:self.obj_qvel_addr + 6] = 0
        self.sim.forward()

    def ball_pos(self):
        return self.sim.data.body_xpos[self.obj_body_id].copy()

    def ball_velocity(self):
        return self.sim.data.qvel[self.obj_qvel_addr:self.obj_qvel_addr + 3].copy()

    def eef_pos(self, arm=None):
        return self.sim.data.site_xpos[self.robots[0].eef_site_id[arm or self.active_arm]].copy()

    def eef_mat(self, arm=None):
        return self.sim.data.site_xmat[self.robots[0].eef_site_id[arm or self.active_arm]].reshape(3, 3).copy()

    def _contacts(self):
        touched = set()
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if contact.geom1 in self.obj_geom_ids:
                touched.add(contact.geom2)
            if contact.geom2 in self.obj_geom_ids:
                touched.add(contact.geom1)
        return all(touched & group for group in self.finger_groups), bool(touched & self.gripper_geom_ids)

    def _update_observables(self, force=False):
        super()._update_observables(force=force)
        # robosuite calls this after every 2 ms physics substep. Checking only
        # at 50 Hz can miss a short bounce between control ticks.
        if force or not self._lifted or not self._open_requested:
            return
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            pair = {contact.geom1, contact.geom2}
            if pair & self.obj_geom_ids and pair & self.tray_geom_ids:
                self.flight_platform_contacts += 1
                self._failure = "platform_contact_after_release"

    def _obj_in_pot(self):
        p = self.ball_pos() - self.sim.data.body_xpos[self.pot_body_id]
        return (-0.02 < p[2] < POT_MOUTH_Z - self.ball_radius
                and np.linalg.norm(p[:2]) < POT_CAVITY_R - self.ball_radius)

    def _check_success(self):
        return self._success

    @property
    def score(self):
        return 100.0 if self._success else 0.0

    def reward(self, action=None):
        return float(self._success)

    def step(self, action):
        if self._finished:
            raise RuntimeError("Episode ended; reset required")
        a = np.asarray(action, float)
        if a.shape != (self.action_dim,) or not np.isfinite(a).all():
            raise ValueError(f"Expected finite action shape {(self.action_dim,)}")
        before = float(self.sim.data.time)
        obs, _, done, _ = super().step(np.clip(a, -1, 1))
        dt = float(self.sim.data.time) - before
        events, milestones = [], []
        both, any_contact = self._contacts()
        self._grasp_time = self._grasp_time + dt if both else 0.0
        if not self._grasped and self._grasp_time >= 0.1:
            self._grasped = True
            self.phase = "grasped"
            milestones.append("ball_grasped")
        if self._grasped and both and not self._lifted and self.ball_pos()[2] > self.tray_pos[2] + self.pickup_height + self.ball_radius + 0.06:
            self._lifted = True
            self.phase = "lifted"
            milestones.append("ball_lifted")
        grip_start, _ = self.robots[0].composite_controller._action_split_indexes[self.active_arm + "_gripper"]
        if self._lifted and a[grip_start] < -0.5:
            self._open_requested = True
        if self._lifted and not self._released:
            if any_contact:
                self._release_sample = None
            elif self._release_sample is None:
                self._release_sample = (self.ball_pos(), self.ball_velocity(), float(self.sim.data.time))
            self._lost_contact_time = 0 if any_contact else self._lost_contact_time + dt
            if not any_contact and self._lost_contact_time >= 0.04:
                self._released = True
                self.attempts = 1
                self.release_position, self.release_velocity, self.release_sample_time = self._release_sample
                self.phase = "flight"
                milestones.append("ball_released")
                if not self._open_requested:
                    self._failure = "ball_slipped"
                elif np.linalg.norm(self.release_velocity) < 0.2:
                    self._failure = "released_without_throw"
        in_pot = self._obj_in_pot()
        if self._released and in_pot and not self._entered:
            self._entered = True
            milestones.append("ball_entered")
        settled = self._released and in_pot and not any_contact and np.linalg.norm(self.ball_velocity()) < 0.08
        self._settle_time = self._settle_time + dt if settled else 0.0
        if self._settle_time >= 0.5 and not self._failure:
            self._success = True
            self.phase = "success"
            events.append("ball_settled_in_pot")
        p = self.ball_pos()
        if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
            self._failure = "non_finite_state"
        elif p[2] < self.ball_radius + 0.01 and not in_pot:
            self._failure = "bounced_out" if self._entered else ("missed_pot" if self._released else "ball_dropped")
        elif np.max(np.abs(p)) > self.flight_bounds:
            self._failure = "out_of_bounds"
        rest_outside = self._released and not in_pot and not any_contact and np.linalg.norm(self.ball_velocity()) < 0.04
        self._rest_outside_time = self._rest_outside_time + dt if rest_outside else 0
        if self._rest_outside_time >= 0.5:
            self._failure = "settled_outside_pot"
        if self._failure:
            self.phase = "failed"
        terminated = self._success or self._failure is not None
        truncated = bool(done and not terminated)
        self._finished = bool(terminated or truncated)
        info = dict(success_state=self._success, terminal_success=self._success,
                    total_score=self.score, events=events, milestones=milestones,
                    phase=self.phase, grasp_contact=both, ball_lifted=self._lifted,
                    ball_released=self._released, flight_platform_contacts=self.flight_platform_contacts, elapsed_sec=float(self.sim.data.time),
                    failure={"recoverable": False, "reason": self._failure} if self._failure else None)
        return obs, float(self._success), bool(terminated), truncated, info
