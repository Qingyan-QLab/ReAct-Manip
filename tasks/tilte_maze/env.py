"""Physical single-arm tilt maze and its compact two-axis prototype."""

from __future__ import annotations

import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

from react_manip.core import Scenario, SuccessRule, TaskSpec
from react_manip.core.scenarios import sample_scenario
from react_manip.core.visual_randomization import apply_visual_scenario_xml
from .assets.gen_assets import MAZE_GRID, config_from_scenario, make_xml
from .scoring import MazeScore, YELLOW_ENTRY_PENALTY, MAX_PENALIZED_ENTRIES

ROOT = Path(__file__).resolve().parent

SPEC = TaskSpec(
    task_id="tilte_maze", task_version="0.6", task_type="one_shot",
    required_arms=1, required_gripper="parallel", supported_robots=("dual_piperx", "piperx"),
    horizon_sec=600.0, control_hz=20.0, done_behavior="fail_and_stop",
    success=SuccessRule(min_occupancy=0.0, min_events=1, max_failures=0,
                        require_terminal_success=True),
)


def _quat(roll: float, pitch: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, sr * sp], dtype=float)


def _tilt_from_joystick(quat, config):
    """Map the stick tip direction to the same downhill direction on the tray."""
    w, x, y, z = quat
    tip_x = 2 * (x * z + w * y)
    tip_y = 2 * (y * z - w * x)
    tip_z = 1 - 2 * (x * x + y * y)
    gain = config["joystick"]["board_tilt_per_10_deg"] / 10.0
    # A tip displacement toward +X lowers the tray's +X edge; +Y does the
    # same for its +Y edge. The sign of roll differs from pitch by geometry.
    roll = -math.atan2(tip_y, tip_z) * gain
    pitch = math.atan2(tip_x, tip_z) * gain
    limit = math.radians(config["task"]["max_tilt_deg"])
    length = math.hypot(roll, pitch)
    if length > limit:
        roll *= limit / length
        pitch *= limit / length
    return np.array([roll, pitch], dtype=float)


class TilteMaze:
    """Maze board controlled by roll/pitch actions in ``[-1, 1]``."""

    action_dim = 2
    task_id = "tilte_maze"

    def __init__(self, *, seed: int | None = 0,
                 scenario: Scenario | str | Path | None = None,
                 has_offscreen_renderer: bool = False, **kwargs):
        del kwargs
        if isinstance(scenario, (str, Path)):
            scenario = Scenario.load(scenario)
        if scenario is None:
            profile = json.loads((ROOT.parents[1] / "configs/tilte_maze/profile_default.json").read_text())
            scenario = sample_scenario(profile, seed=0 if seed is None else int(seed))
        if scenario.task_id != self.task_id:
            raise ValueError(f"scenario task {scenario.task_id!r} does not match {self.task_id!r}")
        self.scenario = scenario
        self.seed = scenario.seed if seed is None else int(seed)
        self.config = config_from_scenario(scenario.parameters)
        self.has_offscreen_renderer = bool(has_offscreen_renderer)
        self.model = None
        self.data = None
        self.renderer = None
        self._board_qpos = self._ball_qpos = self._board_qvel = self._ball_qvel = 0
        self._joystick_qpos = self._joystick_qvel = 0
        self._step_count = 0
        self._phase = "search"
        self._events: list[str] = []
        self._score = None
        self._tilt = np.zeros(2, dtype=float)
        self._camera_observation_ids: tuple[str, ...] = ()
        self.reset(seed=self.seed)

    def _build(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        xml = apply_visual_scenario_xml(make_xml(self.config), self.scenario)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        board = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "board_ball")
        ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        joystick = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joystick_ball")
        self._board_qpos = int(self.model.jnt_qposadr[board])
        self._ball_qpos = int(self.model.jnt_qposadr[ball])
        self._board_qvel = int(self.model.jnt_dofadr[board])
        self._ball_qvel = int(self.model.jnt_dofadr[ball])
        self._joystick_qpos = int(self.model.jnt_qposadr[joystick])
        self._joystick_qvel = int(self.model.jnt_dofadr[joystick])
        if self.has_offscreen_renderer:
            self.renderer = mujoco.Renderer(self.model, width=720, height=540)

    def reset(self, *, seed: int | None = None, **kwargs):
        del kwargs
        if seed is not None and int(seed) != self.seed:
            self.seed = int(seed)
            profile = json.loads((ROOT.parents[1] / "configs/tilte_maze/profile_default.json").read_text())
            self.scenario = sample_scenario(profile, seed=self.seed)
            self.config = config_from_scenario(self.scenario.parameters)
        self._build()
        b = self.config["board"]
        self.data.qpos[self._board_qpos:self._board_qpos + 4] = [1, 0, 0, 0]
        self.data.qpos[self._joystick_qpos:self._joystick_qpos + 4] = [1, 0, 0, 0]
        start = self.config["ball"]["start_xy_m"]
        center_x, center_y = self.config["board"]["center_xy_m"]
        self.data.qpos[self._ball_qpos:self._ball_qpos + 7] = [
            center_x + start[0], center_y + start[1],
            b["center_height_m"] + b["floor_height_m"] +
            self.config["ball"]["radius_m"] + .003, 1, 0, 0, 0,
        ]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._step_count, self._phase, self._events = 0, "search", []
        self._score = MazeScore(b["half_size_m"], b["rim_width_m"],
                                self.config["ball"]["radius_m"])
        self._tilt[:] = 0
        return self._observation(), self._info()

    def _task_instruction(self):
        return (
            "Guide the ball from the green start cell through the 32-by-32 maze "
            "to the blue target area. Red cells are solid walls, grey cells mark "
            "the intended route, white cells are safe floor, and yellow cells are "
            f"penalty regions that remain level floor; each yellow-region entry costs "
            f"{YELLOW_ENTRY_PENALTY:g} points for at most {MAX_PENALIZED_ENTRIES} entries (80 points total), "
            "while reaching the blue area succeeds. Keep the ball inside "
            "the black outer rim. Route progress earns up to 100 points and the score is clamped to 0-100."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='tilte_maze',
            task_instruction=self._task_instruction(),
            success_condition='The ball enters a blue goal cell before the time limit. Score alone does not finish the task.',
            constraints=(),
        )

    def _local_ball(self):
        bid = self.model.body("maze_board").id
        R = self.data.xmat[bid].reshape(3, 3)
        board_pos = self.data.xpos[bid]
        ball_pos = self.data.xpos[self.model.body("ball").id]
        local_pos = R.T @ (ball_pos - board_pos)
        local_vel = R.T @ self.data.cvel[self.model.body("ball").id, 3:]
        return local_pos, local_vel

    def _observation(self):
        pos, vel = self._local_ball()
        return {
            "ball_board_pos": pos.astype(np.float32),
            "ball_board_vel": vel.astype(np.float32),
            "phase": np.array([{"search": 0, "trap": 1, "success": 2}.get(self._phase, 3)], dtype=np.int8),
            "tilt": self._tilt.astype(np.float32).copy(),
        }

    def _info(self):
        return {"success_state": self._phase == "success", "terminal_success": self._phase == "success",
                "events": list(self._events), "failure": "time_limit" if self._phase == "timeout" else None,
                "score": self._score.score, "route_progress_pct": self._score.progress_pct,
                "yellow_entries": self._score.yellow_entries,
                "penalty": self._score.penalty_points,
                "board_tilt_per_10_stick_deg": self.config["joystick"]["board_tilt_per_10_deg"],
                "surface_friction": self.config["board"]["surface_friction"],
                "surface_rolling_friction": self.config["board"]["surface_rolling_friction"],
                "wall_friction": self.config["maze"]["wall_friction"],
                "wall_rolling_friction": self.config["maze"]["wall_rolling_friction"],
                "wall_bounce": self.config["maze"]["wall_bounce"],
                "scenario_id": self.scenario.scenario_id, "scenario_digest": self.scenario.digest}

    def step(self, action):
        if self._phase in ("success", "timeout"):
            return self._observation(), 0.0, True, False, self._info()
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size < 2:
            raise ValueError("action must contain roll and pitch")
        mapping_gain = self.config["joystick"]["board_tilt_per_10_deg"] / 10.0
        reachable_tilt_deg = min(
            self.config["task"]["max_tilt_deg"],
            self.config["joystick"]["max_angle_deg"] * mapping_gain,
        )
        max_tilt = math.radians(reachable_tilt_deg)
        roll, pitch = np.clip(a[:2], -1, 1) * max_tilt
        self._tilt[:] = (roll, pitch)
        # The native two-axis prototype tilts the physical stick. The board
        # reads that joint through the same reduction used by the robot scene.
        stick_gain = 1.0 / mapping_gain
        self.data.qpos[self._joystick_qpos:self._joystick_qpos + 4] = _quat(float(roll) * stick_gain,
                                                                           float(pitch) * stick_gain)
        self.data.qvel[self._joystick_qvel:self._joystick_qvel + 3] = 0
        self._tilt[:] = _tilt_from_joystick(self.data.qpos[self._joystick_qpos:self._joystick_qpos + 4],
                                           self.config)
        self.data.qpos[self._board_qpos:self._board_qpos + 4] = _quat(*self._tilt)
        self.data.qvel[self._board_qvel:self._board_qvel + 3] = 0
        for _ in range(self.config["task"]["substeps_per_control"]):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        pos, _ = self._local_ball()
        reward, self._events = self._score.update(pos[:2])
        if self._score.success:
            self._phase = "success"
        elif self._step_count >= int(self.config["task"]["horizon_sec"] * SPEC.control_hz):
            self._phase = "timeout"
            self._events.append("time_limit")
        return self._observation(), reward, self._phase in ("success", "timeout"), False, self._info()

    def render(self, camera="maze_closeup"):
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def set_camera_observations(self, camera_ids=(), *, width=640, height=480):
        """Match the benchmark camera hook used by the legacy adapter."""
        self._camera_observation_ids = tuple(camera_ids or ())
        if self.has_offscreen_renderer and self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=int(width), height=int(height))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class _MazeArena:
    """Robosuite Arena wrapper around the generated Scenario XML."""

    def __new__(cls, config, robot_id="piperx", finger_geoms=None):
        # Arena parses a filename, while the native prototype intentionally
        # generates XML in memory.  Keep this bridge temporary and remove the
        # file immediately after robosuite has parsed it.
        from robosuite.models.arenas import Arena
        from react_manip.core.visual_randomization import apply_visual_scenario_xml
        xml = make_xml(config)
        handle = tempfile.NamedTemporaryFile(prefix="tilte_maze_", suffix=".xml",
                                             dir=ROOT, delete=False)
        try:
            handle.write(xml.encode("utf-8")); handle.close()
            arena = Arena(handle.name)
            # Arena recolors group-0 collision geoms for diagnostics.  Keep a
            # separate group-1 visual copy so the physical scene retains the
            # authored wood/metal colors while collision meshes remain exact.
            for parent in arena.worldbody.iter():
                for geom in list(parent.findall("geom")):
                    if (geom.get("group") == "1" or geom.get("name", "").startswith("robot")
                            or geom.get("name") == "board_floor_collision"):
                        continue
                    visual = deepcopy(geom)
                    visual.set("name", geom.get("name", "") + "_visual")
                    visual.set("group", "1")
                    visual.set("contype", "0")
                    visual.set("conaffinity", "0")
                    name = geom.get("name", "")
                    if name == "table":
                        visual.set("rgba", "0.48 0.27 0.11 1")
                    elif name.startswith("arm_mount"):
                        visual.set("rgba", "0.16 0.18 0.20 1")
                    elif name == "maze_socket":
                        visual.set("rgba", "0.28 0.29 0.31 1")
                    elif name == "maze_ball_pivot":
                        visual.set("rgba", "0.75 0.77 0.79 1")
                    elif name == "joystick_base":
                        visual.set("rgba", "0.19 0.21 0.24 1")
                    elif name == "joystick_stem":
                        visual.set("rgba", "0.72 0.74 0.77 1")
                    elif name == "joystick_grip":
                        visual.set("rgba", "0.13 0.34 0.75 1")
                    elif name == "ball_geom":
                        visual.set("rgba", "1 0.08 0.75 1")
                    elif name == "room_floor":
                        visual.set("rgba", "0.70 0.74 0.78 1")
                    elif name.startswith("room_"):
                        visual.set("rgba", "0.88 0.89 0.91 1")
                    elif name.startswith("board_floor"):
                        visual.attrib.pop("material", None)
                        # Preserve the spreadsheet's floor coding on the
                        # physical robot visual copy (the original collision
                        # geom remains unchanged).
                        try:
                            ix, iy = (int(v) for v in name.rsplit("_", 2)[-2:])
                            color = {"g": "0.62 0.62 0.64 1", "t": "0.95 0.78 0.16 1",
                                     "s": "0.18 0.64 0.20 1", "G": "0.05 0.38 0.82 1"}.get(
                                         MAZE_GRID[iy][ix],
                                         "0.93 0.93 0.93 1")
                        except (ValueError, IndexError):
                            color = "0.93 0.93 0.93 1"
                        visual.set("rgba", color)
                    elif name.startswith("maze_wall"):
                        visual.attrib.pop("material", None)
                        # Keep the workbook's black outer ring distinct from
                        # the red internal partitions in the physical arena.
                        visual.set("rgba", "0.015 0.018 0.022 1" if geom.get("material") == "maze_wall_black" else "0.78 0.035 0.035 1")
                    elif name.startswith("rim_"):
                        visual.attrib.pop("material", None)
                        visual.set("rgba", "0.015 0.018 0.022 1")
                    visual.attrib.pop("mass", None)
                    visual.attrib.pop("density", None)
                    parent.append(visual)
            # Imported finger meshes need explicit pairs with the lever grip;
            # otherwise robosuite's compiled contact exclusions can skip the
            # small hand-held sphere even when the visible meshes overlap.
            contact = arena.root.find("contact")
            if contact is None:
                contact = ET.SubElement(arena.root, "contact")
            if finger_geoms is None:
                from react_manip.robots.agilex_robosuite import asset_dir
                if robot_id in ("legacy_piperx", "aloha"):
                    from react_manip.robots.legacy_agilex_robosuite import asset_dir
                import json
                names=json.loads((asset_dir()/"provenance.json").read_text())["gripper_collision_names"]
                finger_geoms=[f"gripper0_right_{name}" for group in names.values() for name in group]
            for geom_name in dict.fromkeys(finger_geoms):
                ET.SubElement(contact,"pair",geom1="joystick_grip",geom2=geom_name,condim="3",
                    friction=f"{config['joystick']['contact_friction']} {config['joystick']['contact_friction']} 0.03 0.003 0.003")

        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
        return arena


class TilteMazePhysical:
    """Single-arm robosuite interface for pushing a physical joystick.

    The 7-D action is the standard PiPER OSC pose and gripper command. Finger
    contact moves the spring-centered ball-jointed stick; its measured pose
    drives the tray's central ball joint through a bounded reduction. The ball
    stays free and moves under gravity and maze contacts.
    """

    task_id = "tilte_maze"

    def __new__(cls, *args, **kwargs):
        # Keep imports lazy so Scenario sampling and the native keyboard demo
        # remain usable without robosuite installed.
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

        class _Physical(ManipulationEnv):
            task_id = "tilte_maze"

            def __init__(self, robots="dual_piperx", controller_configs=None,
                         gripper_types="default", seed=0, scenario=None, **kw):
                from react_manip.robots.agilex_robosuite import controller_config, resolve_robot
                from react_manip.robots.resolution import robot_id_from_model, resolve_robosuite_robot
                resolved = resolve_robosuite_robot(robot_id_from_model(robots))
                self.robot_id = resolved.robot_id
                if self.robot_id not in ('dual_piperx', 'piperx', 'aloha', 'legacy_piperx', 'panda_libero'):
                    raise ValueError('tilte_maze supports dual_piperx and piperx')
                robots = resolved.robot_name
                if scenario is None:
                    profile = json.loads((ROOT.parents[1] / "configs/tilte_maze/profile_default.json").read_text())
                    scenario = sample_scenario(profile, seed=0 if seed is None else int(seed))
                if isinstance(scenario, (str, Path)):
                    scenario = Scenario.load(scenario)
                if scenario.task_id != "tilte_maze":
                    raise ValueError("Scenario task_id must be tilte_maze")
                self.scenario = scenario
                self.config = config_from_scenario(scenario.parameters)
                self.seed = scenario.seed
                kw.pop('initialization_noise', None)
                kw.pop('control_freq', None)
                super().__init__(
                    robots=robots,
                    controller_configs=controller_configs or resolved.controller_configs or controller_config(resolved.arms),
                    gripper_types=resolved.gripper_types if self.robot_id=="panda_libero" else gripper_types,
                    has_renderer=kw.pop("has_renderer", False),
                    has_offscreen_renderer=kw.pop("has_offscreen_renderer", True),
                    use_camera_obs=kw.pop("use_camera_obs", False),
                    render_camera=kw.pop("render_camera", "workspace_oblique"),
                    camera_names=kw.pop("camera_names", "workspace_oblique"),
                    horizon=kw.pop("horizon", int(self.config["task"]["horizon_sec"] * 20)),
                    control_freq=20, initialization_noise=None, seed=self.seed, **kw,
                )

            def _load_model(self):
                from react_manip.robots.mounts import use_existing_mount
                from react_manip.visual_task import ManipulationTask

                super()._load_model()
                robot = self.robots[0].robot_model
                if self.robot_id in ('dual_piperx', 'aloha'):
                    from react_manip.robots.mounts import set_fixed_mount_pose
                    set_fixed_mount_pose(robot, position=[-.99, .10, .82], yaw=0., height=.82)
                else:
                    use_existing_mount(robot)
                    robot.set_base_xpos([-.75, -.20, 0. if self.robot_id=="panda_libero" else .82])
                gripper=self.robots[0].gripper['right']
                fingers=[name for key in ('left_finger','right_finger','left_fingerpad','right_fingerpad')
                         for name in gripper.important_geoms.get(key,[])]
                arena = _MazeArena(self.config, self.robot_id, fingers)
                self.model = ManipulationTask(
                    mujoco_arena=arena, mujoco_robots=[self.robots[0].robot_model], mujoco_objects=[])

            def _setup_references(self):
                super()._setup_references()
                # Only the visible finger meshes can push the stick. Broad
                # arm-link hulls and hidden grasp aids cannot move it.
                for geom_id, name in enumerate(self.sim.model.geom_names):
                    if name.startswith("robot0_") and "_collision" in name:
                        # Arm links still collide with the tabletop (bit 2).
                        self.sim.model.geom_contype[geom_id] = 2
                        self.sim.model.geom_conaffinity[geom_id] = 2
                    elif name.startswith("gripper0_") and "_finger" in name and "_collision" in name:
                        self.sim.model.geom_contype[geom_id] = 3
                        self.sim.model.geom_conaffinity[geom_id] = 3
                        self.sim.model.geom_friction[geom_id] = [
                            self.config["joystick"]["contact_friction"], 0.03, 0.003]
                        self.sim.model.geom_margin[geom_id] = 0.0
                        self.sim.model.geom_solref[geom_id] = [0.02, 1.0]
                        self.sim.model.geom_solimp[geom_id] = [0.95, 0.99, 0.001, 0.5, 2.0]
                def first(value):
                    return int(value[0]) if isinstance(value, tuple) else int(value)
                self._board_qpos = first(self.sim.model.get_joint_qpos_addr("board_ball"))
                self._board_qvel = first(self.sim.model.get_joint_qvel_addr("board_ball"))
                self._joystick_qpos = first(self.sim.model.get_joint_qpos_addr("joystick_ball"))
                self._joystick_qvel = first(self.sim.model.get_joint_qvel_addr("joystick_ball"))
                self._ball_qpos = first(self.sim.model.get_joint_qpos_addr("ball_free"))
                self._board_target = np.zeros(2, dtype=float)
                self._score = MazeScore(self.config["board"]["half_size_m"],
                                        self.config["board"]["rim_width_m"],
                                        self.config["ball"]["radius_m"])
                self._maze_board_body = self.sim.model.body_name2id("maze_board")
                self._maze_ball_body = self.sim.model.body_name2id("ball")

            def _reset_internal(self):
                super()._reset_internal()
                home_overrides=() if self.robot_id=='panda_libero' else (("robot0_right_joint2",1.45),("robot0_right_joint3",-.80))
                for joint_name, value in home_overrides:
                    jid = self.sim.model.joint_name2id(joint_name)
                    self.sim.data.qpos[self.sim.model.jnt_qposadr[jid]] = value
                b = self.config["board"]; start = self.config["ball"]["start_xy_m"]
                self.sim.data.qpos[self._board_qpos:self._board_qpos + 4] = [1, 0, 0, 0]
                self.sim.data.qpos[self._joystick_qpos:self._joystick_qpos + 4] = [1, 0, 0, 0]
                self.sim.data.qpos[self._ball_qpos:self._ball_qpos + 7] = [
                    b["center_xy_m"][0] + start[0], b["center_xy_m"][1] + start[1],
                    b["center_height_m"] + b["floor_height_m"] +
                    self.config["ball"]["radius_m"] + .003, 1, 0, 0, 0,
                ]
                self.sim.data.qvel[:] = 0
                self._board_target[:] = 0
                self._score.reset()
                self.sim.forward()

            def _task_instruction(self):
                return ("Use the single arm to push the joystick in any direction. "
                        "The maze board tilts the same way about its central ball joint. "
                        f"Earn route progress along grey cells, lose "
                        f"{YELLOW_ENTRY_PENALTY:g} points per yellow-zone "
                        f"entry for at most {MAX_PENALIZED_ENTRIES} entries (80 points total), "
                        "and reach the blue area to finish successfully. "
                        "Route progress earns up to 100 points and the score is clamped to 0-100. "
                        "Progress uses the farthest grey-route position reached; moving backward does not remove progress. "
                        "Each exit and re-entry into yellow incurs another penalty until the cap. "
                        "Yellow cells are level floor, red cells are walls, and white cells are safe floor. "
                        f"Time limit: {self.horizon / self.control_freq:g} seconds. Timeout keeps the current score; "
                        "entering blue ends successfully even if penalties make the score less than 100.")

            def get_episode_language_prompt(self):
                from react_manip.core.language import task_language_prompt
                return task_language_prompt(
                    self, task_id='tilte_maze',
                    task_instruction=self._task_instruction(),
                    success_condition='The ball enters a blue goal cell before the time limit. Score alone does not finish the task.',
                    constraints=(),
                )

            def _pre_action(self, action, policy_step=False):
                super()._pre_action(action, policy_step=policy_step)
                stick_quat = self.sim.data.qpos[self._joystick_qpos:self._joystick_qpos + 4]
                target = _tilt_from_joystick(stick_quat, self.config)
                # A modest rate limit avoids abrupt tray motion if the stick
                # is struck. The ball itself is never steered or repositioned.
                dt = float(self.sim.model.opt.timestep)
                delta = np.clip(target - self._board_target,
                                -math.radians(30) * dt, math.radians(30) * dt)
                self._board_target += delta
                self.sim.data.qpos[self._board_qpos:self._board_qpos + 4] = _quat(*self._board_target)
                self.sim.data.qvel[self._board_qvel:self._board_qvel + 3] = [delta[0] / dt, delta[1] / dt, 0]

            def reward(self, action=None):
                del action
                return 0.0

            def _post_action(self, action):
                reward, done, info = super()._post_action(action)
                board_rotation = self.sim.data.body_xmat[self._maze_board_body].reshape(3, 3)
                ball_local = board_rotation.T @ (
                    self.sim.data.body_xpos[self._maze_ball_body]
                    - self.sim.data.body_xpos[self._maze_board_body])
                reward, events = self._score.update(ball_local[:2])
                if self._score.success:
                    done = True
                    self.done = True
                quat = self.sim.data.qpos[self._board_qpos:self._board_qpos + 4]
                roll = math.atan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
                pitch = math.asin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1))
                stick_quat = self.sim.data.qpos[self._joystick_qpos:self._joystick_qpos + 4]
                mapping_gain = self.config["joystick"]["board_tilt_per_10_deg"] / 10.0
                stick_roll, stick_pitch = _tilt_from_joystick(
                    stick_quat, self.config) / mapping_gain
                info = dict(info or {})
                info.update(board_roll_deg=math.degrees(roll), board_pitch_deg=math.degrees(pitch),
                            joystick_side_deg=-math.degrees(stick_roll),
                            joystick_forward_deg=math.degrees(stick_pitch),
                            total_score=self._score.score, score=self._score.score,
                            route_progress_pct=self._score.progress_pct,
                            yellow_entries=self._score.yellow_entries,
                            penalty=self._score.penalty_points,
                            board_tilt_per_10_stick_deg=(
                                self.config["joystick"]["board_tilt_per_10_deg"]),
                            surface_friction=self.config["board"]["surface_friction"],
                            surface_rolling_friction=(
                                self.config["board"]["surface_rolling_friction"]),
                            wall_friction=self.config["maze"]["wall_friction"],
                            wall_rolling_friction=(
                                self.config["maze"]["wall_rolling_friction"]),
                            wall_bounce=self.config["maze"]["wall_bounce"],
                            events=events,
                            success_state=self._score.success,
                            terminal_success=self._score.success,
                            failure="time_limit" if done and not self._score.success else None)
                info['TimeLimit.truncated'] = bool(done and not self._score.success)
                return reward, done, info

        return _Physical(*args, **kwargs)
