"""Bimanual, contact-driven leveling-board environment.

This is independent from :mod:`tasks.leveling_board`: the source task-level
shim-selection actions are replaced by real Cartesian arm and gripper control.
"""
from __future__ import annotations

import math
from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from robosuite.models.tasks import ManipulationTask

from react_manip.robots.agilex_robosuite import controller_config, resolve_robot

from .instances import (SLOTS, sample_task_instance, sample_episode_instance,
                        validate_task_instance, validate_episode_instance)
from .scoring import LevelScore, MIN_CONTACT_FORCE_N

ROOT = Path(__file__).resolve().parent
SHIM_NAMES = tuple(f"shim_{i:02d}" for i in range(1, 7))
SOLUTION_SHIMS = SHIM_NAMES[:3]


class LevelingBoard1Arena(Arena):
    def __init__(self, dimensions):
        super().__init__(str(ROOT / "assets" / "scene.xml"))
        for name in ("shim_07", "shim_08"):
            self.worldbody.remove(self.worldbody.find(f"./body[@name='{name}']"))
        self.worldbody.find("./geom[@name='support_block']").set("pos", "-0.425 -0.10 0.74")
        self.worldbody.find("./body[@name='board']").set("pos", "-0.40 -0.10 0.81")
        # Compile randomized collision sizes and inertia together. Runtime
        # geom_size edits leave collision bounds based on the old dimensions.
        for name, dims in dimensions.items():
            self.worldbody.find(f"./body[@name='{name}']/geom").set(
                "size", " ".join(str(v / 2) for v in dims))
        texture_dir = ROOT.parent / "laptop_into_drawer" / "assets" / "textures"

        def material(name, filename, repeat="3 3", **attrs):
            ET.SubElement(self.asset, "texture", name=name + "_texture", type="2d",
                          file=str(texture_dir / filename))
            ET.SubElement(self.asset, "material", name=name + "_material",
                          texture=name + "_texture", texuniform="true", texrepeat=repeat, **attrs)

        material("level_table", "light-wood.png", "4 3", specular="0.18", shininess="0.25")
        material("level_board", "dark-wood.png", "5 1", specular="0.22", shininess="0.30")
        material("level_support", "gray-plaster.png", "2 2", specular="0.12", shininess="0.18")
        material("level_shim", "wood-tiles.png", "2 2", specular="0.16", shininess="0.22")
        material("level_room_floor", "light-gray-floor-tile.png", "6 6", specular="0.12", shininess="0.15")
        material("level_room_wall", "white-plaster.png", "3 2", specular="0.05")
        ET.SubElement(self.asset, "material", name="level_room_trim", rgba="0.23 0.28 0.32 1")
        # Room shell is outside all existing cameras and the robot workspace.
        # The floor can catch fallen objects; the task geometry is untouched.
        for name, pos, size, mat in (
            ("floor", "0 0 -0.05", "3 3 0.05", "level_room_floor_material"),
            ("back", "0 3 1.5", "3 0.06 1.5", "level_room_wall_material"),
            ("front", "0 -3 1.5", "3 0.06 1.5", "level_room_wall_material"),
            ("left", "-3 0 1.5", "0.06 3 1.5", "level_room_wall_material"),
            ("right", "3 0 1.5", "0.06 3 1.5", "level_room_wall_material"),
        ):
            ET.SubElement(self.worldbody, "geom", name="level_room_" + name,
                          type="box", pos=pos, size=size, material=mat, group="1",
                          friction="0.8 0.005 0.0001")
        # Visual table supports fill the former empty space under the tabletop.
        for i, (x, y) in enumerate(((-0.60, -0.40), (-0.60, 0.40), (0.60, -0.40), (0.60, 0.40))):
            ET.SubElement(self.worldbody, "geom", name=f"level_room_table_leg_{i}",
                          type="box", pos=f"{x} {y} 0.315", size="0.035 0.035 0.315",
                          material="level_room_trim", group="1", contype="0", conaffinity="0")
        ET.SubElement(self.worldbody, "light", name="level_room_ceiling_light",
                      pos="0 0 2.8", dir="0 0 -1", diffuse="0.35 0.35 0.35",
                      ambient="0.12 0.12 0.12", castshadow="false")

        def visual_copy(parent, collision_name, visual_name, mat):
            source = parent.find(f"./geom[@name='{collision_name}']")
            source.set("group", "0")
            attrs = {k: v for k, v in source.attrib.items()
                     if k not in {"name", "mass", "density", "friction", "solref", "solimp",
                                  "contype", "conaffinity", "group", "rgba", "material"}}
            ET.SubElement(parent, "geom", name=visual_name, group="1", material=mat,
                          contype="0", conaffinity="0", mass="0", **attrs)

        table = self.worldbody.find("./geom[@name='table']")
        table.set("group", "1"); table.set("material", "level_table_material")
        support = self.worldbody.find("./geom[@name='support_block']")
        support.set("group", "1"); support.set("material", "level_support_material"); support.set("rgba", "0.18 0.62 0.22 1")
        board = self.worldbody.find("./body[@name='board']")
        visual_copy(board, "long_board_collision", "long_board_visual", "level_board_material")
        for name in SHIM_NAMES:
            body = self.worldbody.find(f"./body[@name='{name}']")
            visual_copy(body, name + "_collision", name + "_visual", "level_shim_material")

        # The robot bases are mounted directly at the table edge.  Do not add
        # a decorative plate here: it was easily mistaken for a task object
        # and appeared as an unexplained white board in the overview camera.
        ET.SubElement(self.worldbody, "camera", name="level_overview",
                      pos="-1.25 -1.55 1.45", xyaxes="0.78 -0.63 0 0.34 0.42 0.84", fovy="56")
        ET.SubElement(self.worldbody, "camera", name="level_topview",
                      pos="0 0 2.25", xyaxes="1 0 0 0 1 0", fovy="48")
        ET.SubElement(self.worldbody, "camera", name="level_sideview",
                      pos="0 1.60 1.05", xyaxes="-1 0 0 0 -0.25 0.97", fovy="52")
        ET.SubElement(self.worldbody, "camera", name="level_board_end",
                      pos="0.76 -0.55 0.92", xyaxes="0.65 0.76 0 -0.25 0.21 0.95", fovy="50")
        ET.SubElement(self.worldbody, "light", name="level_key", pos="-0.8 -1.0 2.3",
                      dir="0.25 0.25 -1", diffuse="0.95 0.92 0.86", ambient="0.28 0.28 0.30")
        ET.SubElement(self.worldbody, "light", name="level_fill", pos="0.9 0.7 1.8",
                      dir="-0.3 -0.2 -1", diffuse="0.62 0.70 0.86", ambient="0.16 0.17 0.20")


class LevelingBoard1(ManipulationEnv):
    def __init__(self, robots="dual_piperx", controller_configs=None, gripper_types="default", seed=None,
                 task_instance=None, episode_instance=None, **kwargs):
        self.task_instance = deepcopy(task_instance if task_instance is not None else sample_task_instance(0 if seed is None else seed))
        validate_task_instance(self.task_instance)
        self.episode_instance = deepcopy(episode_instance if episode_instance is not None else sample_episode_instance(self.task_instance, 0 if seed is None else seed))
        validate_episode_instance(self.task_instance, self.episode_instance)
        from react_manip.robots.resolution import resolve_robosuite_robot, robot_id_from_model
        if isinstance(robots,str) and robot_id_from_model(robots) in ('panda_libero','aloha_agilex_robotwin'):
            resolved=resolve_robosuite_robot(robot_id_from_model(robots))
            robots=resolved.robot_name
            controller_configs=resolved.controller_configs
            gripper_types=resolved.gripper_types
        if isinstance(robots, str):
            if robots not in ("dual_piperx", "DualPiperX", "aloha", "AgileXAloha", "aloha_agilex_robotwin", "RoboTwinAloha"):
                raise ValueError("leveling_board1 uses the fixed dual PiPER-X; start it with --robot dual_piperx")
            if robots in ("dual_piperx", "aloha"):
                robots = resolve_robot(robots)
        super().__init__(robots=robots, controller_configs=controller_configs or controller_config(("right", "left")),
                         gripper_types=gripper_types, has_renderer=kwargs.pop("has_renderer", False),
                         has_offscreen_renderer=kwargs.pop("has_offscreen_renderer", True),
                         use_camera_obs=kwargs.pop("use_camera_obs", False),
                         render_camera=kwargs.pop("render_camera", "level_overview"),
                         camera_names=kwargs.pop("camera_names", "level_overview"),
                         horizon=kwargs.pop("horizon", 3600), control_freq=20,
                         initialization_noise=kwargs.pop("initialization_noise", None), seed=seed, **kwargs)

    def _load_model(self):
        super()._load_model()
        from react_manip.robots.mounts import use_existing_mount
        use_existing_mount(self.robots[0].robot_model)
        robot_model = self.robots[0].robot_model
        # Contact geometry is supplied by the central PiPER-X asset.
        # Shift the bimanual base toward the board's working span.  In the
        # original pose the left wrist sat outside the board's left edge,
        # while the right wrist was directly over the shim pile.  This offset
        # puts the left arm at the board's near end and leaves the right arm
        # just outside the shims for a clear grasp approach.
        robot_model.set_base_xpos([0.89, 0.0, 0. if robot_model.__class__.__name__=="RoboTwinAloha" else 0.68])
        # Face the table's short (Y) edge. The left arm works over the
        # board's +Y side; the right arm works over the shim side (-Y).
        robot_model.set_base_ori(np.array([0.0, 0.0, math.pi]))
        self._randomize_shims()
        arena = LevelingBoard1Arena(self.shim_dimensions); arena.set_origin([0.0, 0.0, 0.0])
        from react_manip.cameras import ensure_scene_cameras, ensure_robot_cameras
        ensure_scene_cameras(arena, 'leveling_board1')
        ensure_robot_cameras(robot_model)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[robot_model], mujoco_objects=[])

    def _free_qpos_address(self, body_name):
        body_id = self.sim.model.body_name2id(body_name)
        joint_id = int(self.sim.model.body_jntadr[body_id])
        return body_id, int(self.sim.model.jnt_qposadr[joint_id])

    def _setup_references(self):
        super()._setup_references()
        # Mesh collision margins on the imported arms are visually generous;
        # they can otherwise push a nearby free shim before the gripper is
        # actually touching it. Keep a tiny numerical margin only.
        for geom_id in range(self.sim.model.ngeom):
            body_id = int(self.sim.model.geom_bodyid[geom_id])
            body_name = self.sim.model.body_names[body_id]
            if body_name.startswith("robot0_"):
                self.sim.model.geom_margin[geom_id] = 0.0001
        self.board_body_id = self.sim.model.body_name2id("board")
        hinge_id = self.sim.model.joint_name2id("board_hinge")
        self.board_hinge_qpos = int(self.sim.model.jnt_qposadr[hinge_id])
        self.board_hinge_dof = int(self.sim.model.jnt_dofadr[hinge_id])
        self.shim_body_ids, self.shim_qpos = {}, {}
        self.shim_geom_ids, self.shim_visual_geom_ids = {}, {}
        for name in SHIM_NAMES:
            body_id, qpos = self._free_qpos_address(name)
            self.shim_body_ids[name] = body_id; self.shim_qpos[name] = qpos
            self.shim_geom_ids[name] = self.sim.model.geom_name2id(name + "_collision")
            self.shim_visual_geom_ids[name] = self.sim.model.geom_name2id(name + "_visual")
            self.sim.model.geom_margin[self.shim_geom_ids[name]] = 0.0001

    def _set_free_pose(self, qpos, position, quat=(1.0, 0.0, 0.0, 0.0)):
        self.sim.data.qpos[qpos:qpos + 3] = position
        self.sim.data.qpos[qpos + 3:qpos + 7] = quat

    def _randomize_shims(self):
        task = getattr(self, 'task_instance', None)
        if task is None:
            task = sample_task_instance(0 if self.seed is None else int(self.seed))
        self.shim_dimensions = {n:np.asarray(v) for n,v in task['shim_dimensions'].items()}
        self.solution_axes = dict(task['solution_axes'])

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.data.qpos[self.board_hinge_qpos] = math.radians(8.0)
        validate_episode_instance(self.task_instance, self.episode_instance)
        for name, slot in zip(SHIM_NAMES, self.episode_instance['slot_indices']):
            xy = SLOTS[slot]
            self._set_free_pose(self.shim_qpos[name],
                               [*xy, .6815 + float(self.shim_dimensions[name][2])/2])
        self.sim.data.qvel[:] = 0.0; self.sim.forward()
        # Hold both arms with their normal controllers while loose scene
        # objects settle. A zero actuator target would close the grippers,
        # and stepping without arm torques lets the arms fall under gravity.
        hold_actions = []
        for robot in self.robots:
            action = np.zeros(robot.action_dim)
            for arm, gripper in robot.gripper.items():
                actuator = robot._ref_joint_gripper_actuator_indexes[arm][0]
                low, high = self.sim.model.actuator_ctrlrange[actuator]
                # PiPER-X maps +1 to closed and -1 to open. Match the actual
                # initial opening instead of commanding either end stop.
                normalized = 2 * (gripper.init_qpos[0] - low) / (high - low) - 1
                start, end = robot.composite_controller._action_split_indexes[robot.get_gripper_name(arm)]
                action[start:end] = -normalized
            hold_actions.append(action)
        for index in range(300):
            for robot, action in zip(self.robots, hold_actions):
                robot.control(action, policy_step=(index == 0))
            self.sim.step()
        # Keep the bodies fully dynamic after settling, but discard residual
        # micro-velocities from the initial drop so they do not visibly buzz.
        self.sim.data.qvel[:] = 0.0
        self.sim.forward()
        self._validate_robot_initial_state()
        self.sim.data.time = 0.
        self.cur_time = 0.
        self._level_score = LevelScore(self.control_freq)
        self._score_report = {'score':0., 'terminal_success':False}

    def _validate_robot_initial_state(self):
        """Reject invalid reset states; never clamp joints or alter live steps."""
        model, data = self.sim.model, self.sim.data
        robot_geoms = {i for i, name in enumerate(model.geom_names)
                       if name and name.startswith(("robot", "gripper"))}
        for joint, name in enumerate(model.joint_names):
            if (not name or not name.startswith(("robot", "gripper"))
                    or not model.jnt_limited[joint]):
                continue
            value = data.qpos[model.jnt_qposadr[joint]]
            low, high = model.jnt_range[joint]
            if not np.isfinite(value) or value < low - 1e-6 or value > high + 1e-6:
                raise ValueError(f"Initial robot joint outside limits: {name}={value}, range=[{low}, {high}]")
        for contact in data.contact[:data.ncon]:
            if (contact.geom1 in robot_geoms or contact.geom2 in robot_geoms) and contact.dist < -1e-4:
                raise ValueError(f"Initial robot penetration: {model.geom_id2name(contact.geom1)} / "
                                 f"{model.geom_id2name(contact.geom2)}, depth={-contact.dist:.6f} m")

    @property
    def board_angle_deg(self):
        return float(np.degrees(self.sim.data.qpos[self.board_hinge_qpos]))

    def _task_instruction(self):
        return ("Use both PiPER-X arms and the six shims to level the hinged board. "
                "Shim dimensions, which shims have useful thin sides, and their positions vary between Episodes. "
                "Inspect their geometry and choose suitable orientations and combinations; sizes stay fixed during an Episode. "
                "Release the board and all shims. Shims must independently support the board from the table. "
                "Hold an absolute board angle at most 0.5 degrees, released and physically stable, "
                f"for 5 continuous seconds to finish with 100 points. Time limit: {self.horizon / self.control_freq:g} seconds. "
                "Stability requires board angular speed <=0.01 rad/s, angle range <=0.1 degrees, "
                "supporting-shim linear speed <=0.005 m/s, angular speed <=0.05 rad/s and drift <=0.005 m. "
                "Angle score is clip(100 - 10*max(0,abs(angle)-0.5),0,100). "
                "No release or no independent shim support gives zero; released but not stable gives 60% of angle score. "
                "Released, independently supported and stable gives the full angle score. Timeout retains the current score; "
                "100 points alone does not finish the task before the five-second success hold.")

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='leveling_board1',
            task_instruction=self._task_instruction(),
            success_condition='Hold the released board level and independently supported by shims for five continuous seconds.',
            constraints=(),
        )

    def reward(self, action=None):
        return float(np.exp(-0.45 * abs(self.board_angle_deg)))

    @property
    def score(self):
        return float(self._score_report['score'])

    def _check_success(self):
        return self._level_score.success

    def _support_state(self):
        """Upward loaded contacts form a table -> shim(s) -> board path."""
        import mujoco
        model, data = self.sim.model, self.sim.data
        labels = {model.geom_name2id('table'):'table',
                  model.geom_name2id('long_board_collision'):'board',
                  **{gid:name for name,gid in self.shim_geom_ids.items()}}
        robot = {i for i,name in enumerate(model.geom_names)
                 if name and name.startswith(('robot','gripper'))}
        edges = []; released = True
        for index,contact in enumerate(data.contact[:data.ncon]):
            g1,g2 = int(contact.geom1),int(contact.geom2)
            force = np.zeros(6)
            mujoco.mj_contactForce(model._model,data._data,index,force)
            if force[0] < MIN_CONTACT_FORCE_N:
                continue
            if (g1 in robot and labels.get(g2) not in (None,'table') or
                    g2 in robot and labels.get(g1) not in (None,'table')):
                released = False
            a,b = labels.get(g1),labels.get(g2)
            if a is None or b is None or a==b:
                continue
            if 'board' in (a,b):
                local = data.body_xmat[self.board_body_id].reshape(3,3).T @ (contact.pos-data.body_xpos[self.board_body_id])
                if local[2] > -.005:
                    continue
            # MuJoCo's contact frame points from geom1 towards geom2.
            normal_z = float(contact.frame[2])
            if abs(normal_z) < .5 or abs(normal_z*force[0]) < MIN_CONTACT_FORCE_N:
                continue
            lower,upper = (a,b) if normal_z>0 else (b,a)
            if lower!='board' and upper!='table':
                edges.append((lower,upper))
        reachable = {'table'}
        for _ in range(8):
            reachable |= {b for a,b in edges if a in reachable and not (a=='table' and b=='board')}
        ancestors = {'board'}
        for _ in range(8):
            ancestors |= {a for a,b in edges if b in ancestors and not (a=='table' and b=='board')}
        shims = (reachable & ancestors) - {'table','board'}
        return released, 'board' in reachable and bool(shims), sorted(shims)

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
            raise ValueError('non-finite leveling state')
        released,supported,shims = self._support_state()
        data = self.sim.data
        positions = {n:data.body_xpos[self.shim_body_ids[n]].copy() for n in shims}
        before = self._level_score.success
        report = self._level_score.update(angle=self.board_angle_deg, released=released,
            supported=supported, board_speed=float(data.qvel[self.board_hinge_dof]), positions=positions,
            linear_speed=max((np.linalg.norm(data.get_body_xvelp(n)) for n in shims),default=0.),
            angular_speed=max((np.linalg.norm(data.get_body_xvelr(n)) for n in shims),default=0.))
        self._score_report = report
        success = report['terminal_success']
        info = dict(info or {}, **report, board_angle_deg=self.board_angle_deg,
                    level_error_deg=abs(self.board_angle_deg), total_score=report['score'],
                    support_shims=shims, success_state=success,
                    events=['board_stably_leveled'] if success and not before else [])
        return reward, bool(done or success), info


def make_env(seed=0, robot_id="dual_piperx", **kwargs):
    return LevelingBoard1(robots=robot_id, seed=seed, **kwargs)
