"""Open a drawer and place the named object, released, inside the desktop bin.

Only reset initializes free-body poses. Runtime motion goes through the
standard robosuite controller and MuJoCo contacts; no attachment or teleport.
"""
from __future__ import annotations

from copy import deepcopy
import math
import os
import json
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mujoco
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena

from react_manip.cameras import Workspace
from react_manip.core.geometry import GeometryFeatureAdapter, _pose
from react_manip.core.language import EpisodeLanguagePrompt
from react_manip.core.visual_randomization import apply_visual_scenario, resolve_scenario
from react_manip.visual_task import ManipulationTask
from .objects import OBJECT_NAMES, LABELS, add_object, vec

DRAWER_FLOOR_Z = .658
TABLE_TOP_Z = .44
BOX_CENTER = np.array([-.40, -.55])
BOX_INNER_HALF = np.array([.12, .11])
BOX_FLOOR_Z = TABLE_TOP_Z + .008
BOX_RIM_Z = BOX_FLOOR_Z + .10
STABLE_SECONDS = .5
CONTAINMENT_TOLERANCE = .0005  # allow soft-contact penetration at an inner wall
SLOTS = np.array([(x, y) for x in (.030, .140) for y in (-.085, 0, .085)])


class FindObjectArena(Arena):
    def __init__(self, names, scenario, offsets):
        # Own the physical model: changes to the laptop task must not resize
        # this drawer underneath its fixed cabinet or change handle contacts.
        super().__init__(str(Path(__file__).parent / "assets/drawer_arena.xml"))
        self.root.set("model", "find_object_in_drawer2")
        self.worldbody.find("./body[@name='drawer']/joint").set("range", "-0.32 0")
        template = self.worldbody.find("./body[@name='drawer']")
        self.worldbody.remove(template)
        for index, offset in enumerate(offsets):
            body = deepcopy(template)
            body.set("pos", vec(np.fromstring(body.get("pos"), sep=" ") + offset))
            for node in body.iter():
                if node.get("name"):
                    node.set("name", node.get("name") + f"_{index}")
            self.worldbody.append(body)
        # Two continuous floor-standing cabinets, separated by a 26 mm gap.
        def fixed_box(name, pos, size, material="wood_mat"):
            for visual in (False, True):
                ET.SubElement(self.worldbody, "geom", name=name + ("_vis" if visual else ""),
                              type="box", pos=vec(pos), size=vec(size), material=material,
                              group="1" if visual else "0", contype="0" if visual else "1",
                              conaffinity="0" if visual else "1")
        for cabinet, y in enumerate((-.20, .20)):
            fixed_box(f"cabinet_{cabinet}_back", (.379,y,.39), (.010,.187,.39))
            for side in (-1,1):
                fixed_box(f"cabinet_{cabinet}_side_{side}", (.160,y+side*.180,.39), (.209,.007,.39))
            fixed_box(f"cabinet_{cabinet}_roof", (.160,y,.774), (.209,.187,.006))
            fixed_box(f"cabinet_{cabinet}_plinth", (.160,y,.095), (.209,.187,.095))
            for row in (1, 2):
                fixed_box(f"cabinet_{cabinet}_fascia_{row}", (-.046,y,.8145-.22*row), (.004,.173,.0495))
            for row in range(3):
                fixed_box(f"cabinet_{cabinet}_shelf_{row}", (.160,y,.637-.22*row), (.209,.187,.006))
        # A separate low table in front, clear of all six slide sweeps.
        fixed_box("table_top", (-.40,-.55,TABLE_TOP_Z-.015), (.19,.17,.015))
        for i, (x,y) in enumerate(((-.55,-.68),(-.25,-.68),(-.55,-.42),(-.25,-.42))):
            fixed_box(f"table_leg{i}", (x,y,(TABLE_TOP_Z-.03)/2), (.02,.02,(TABLE_TOP_Z-.03)/2))
        # Fixed desktop bin: bottom + four walls, no lid or collision proxy
        # across the opening. All dimensions are shared with adjudication.
        cx, cy = BOX_CENTER
        hx, hy = BOX_INNER_HALF
        floor_half = (BOX_FLOOR_Z - TABLE_TOP_Z) / 2
        wall_half = (BOX_RIM_Z - BOX_FLOOR_Z) / 2
        for name, pos, size in (
            ("floor", (cx, cy, TABLE_TOP_Z + floor_half), (hx + .006, hy + .006, floor_half)),
            ("left", (cx - hx - .003, cy, BOX_FLOOR_Z + wall_half), (.003, hy + .006, wall_half)),
            ("right", (cx + hx + .003, cy, BOX_FLOOR_Z + wall_half), (.003, hy + .006, wall_half)),
            ("front", (cx, cy - hy - .003, BOX_FLOOR_Z + wall_half), (hx, .003, wall_half)),
            ("back", (cx, cy + hy + .003, BOX_FLOOR_Z + wall_half), (hx, .003, wall_half)),
        ):
            for visual in (False, True):
                ET.SubElement(self.worldbody, "geom", name=f"destination_box_{name}" + ("_vis" if visual else ""),
                              type="box", pos=vec(pos), size=vec(size), rgba=".24 .52 .56 1",
                              friction=".8 .01 .001", group="1" if visual else "0",
                              contype="0" if visual else "1", conaffinity="0" if visual else "1")
        if scenario is not None:
            apply_visual_scenario(self, scenario)
        self.heights = [add_object(self.worldbody, name) for name in names]
        # Robot controllers call forward before reset places the stacks. Avoid
        # starting 72 free bodies at an identical temporary compiler pose.
        for i, name in enumerate(names):
            self.worldbody.find(f"./body[@name='{name}']").set("pos", vec((2.+.12*(i%9),2.+.10*(i//9),.1)))


class FindObjectInDrawer2(ManipulationEnv):
    task_id = "find_object_in_drawer2"
    step_handles_robot_action = True
    camera_workspace = Workspace((-.10, 0, .48), (.80, .75))
    privileged_observation_keys = ("target_index", "object_positions", "drawer_qpos")

    def __init__(self, robots="DualPiperX", *, seed=None, scenario=None, drawer_count=None, objects_per_drawer=None, clutter_layers=None,
                 controller_configs=None, gripper_types="default",
                 use_object_obs=False, use_camera_obs=False, has_renderer=False,
                 has_offscreen_renderer=False, initialization_noise=None,
                 control_freq=20, horizon=12000, **kwargs):
        from react_manip.robots.resolution import robot_id_from_model, resolve_robosuite_robot
        resolved = resolve_robosuite_robot(robot_id_from_model(robots))
        self.robot_id = resolved.robot_id
        self.scenario = resolve_scenario(scenario)
        if self.scenario is None:
            from react_manip.core.scenarios import sample_scenario
            profile = Path(__file__).parents[2] / "configs/find_object_in_drawer2/profile_default.json"
            self.scenario = sample_scenario(json.loads(profile.read_text()), seed=0 if seed is None else seed)
        params = self.scenario.parameters if self.scenario is not None else {}
        self.drawer_count = params.get("drawer_count", 6) if drawer_count is None else drawer_count
        if type(self.drawer_count) is not int or self.drawer_count != 6:
            raise ValueError("drawer_count must be 6 (two cabinets, three drawers each)")
        self.clutter_layers = params.get("clutter_layers", 1) if clutter_layers is None else clutter_layers
        if type(self.clutter_layers) is not int or self.clutter_layers not in (1, 2, 3):
            raise ValueError("clutter_layers must be 1, 2, or 3")
        counts = params.get("objects_per_drawer", 6) if objects_per_drawer is None else objects_per_drawer
        if type(counts) is int:
            counts = [counts] * self.drawer_count
        if (not isinstance(counts, (list, tuple)) or len(counts) != self.drawer_count
                or any(type(n) is not int or not 2*self.clutter_layers <= n <= 6*self.clutter_layers for n in counts)):
            raise ValueError("objects_per_drawer must be an integer or one integer per drawer; each must be between 2*clutter_layers and 6*clutter_layers")
        self.objects_per_drawer = tuple(counts)
        self.base_counts = [math.ceil(n / self.clutter_layers) for n in counts]
        self.base_count = sum(self.base_counts)
        self.target_count = sum(counts)
        if self.target_count > len(OBJECT_NAMES):
            raise ValueError("The unique object pool contains 36 kinds; total objects_per_drawer must not exceed 36")
        # Select once per scenario; subsequent resets reshuffle this unique set.
        # Keep catalog ordering so geometry IDs do not depend on drawer order.
        selection = np.random.default_rng(self.scenario.seed).choice(
            len(OBJECT_NAMES), self.target_count, replace=False)
        self.object_names = tuple(OBJECT_NAMES[i] for i in sorted(selection))
        self.drawer_offsets = np.array([[0., y, z] for z in (0., -.22, -.44) for y in (-.20, .20)])
        self.object_drawers = np.zeros(len(self.object_names), dtype=int)
        self.active_drawer = 0
        self.target_index = 0
        self._episode_index = -1
        self._targets = []
        self._round_index = 0
        self.total_score = 0
        self._stable_ticks = 0
        self._opened = False
        self._success = False
        from react_manip.robots.agilex_robosuite import controller_config
        robots = resolved.robot_name
        gripper_types = resolved.gripper_types
        if controller_configs is None:
            controller_configs = controller_config(resolved.arms)
        controller_configs = deepcopy(controller_configs)
        controller_configs["body_parts"]["right"]["kp"] = ([240,240,240,60,60,60]
            if self.robot_id in ("panda", "sawyer") else [180,180,180,12,12,12])
        # Exact object positions never enter policy observations.
        del use_object_obs
        # MuJoCo step1/step2 avoids repeating full forward dynamics once
        # per arm and per substep in this 72-object scene.
        kwargs.setdefault("lite_physics", True)
        kwargs.setdefault("base_types", "NullMount")
        super().__init__(robots=robots, seed=seed, controller_configs=controller_configs,
                         gripper_types=gripper_types, initialization_noise=initialization_noise,
                         use_camera_obs=use_camera_obs, has_renderer=has_renderer,
                         has_offscreen_renderer=has_offscreen_renderer, control_freq=control_freq,
                         horizon=horizon, **kwargs)

    def _load_model(self):
        super()._load_model()
        from react_manip.robots.mounts import use_existing_mount, install_fixed_stand
        robot = self.robots[0].robot_model
        use_existing_mount(robot)
        if self.robot_id in ("dual_piperx", "aloha"):
            install_fixed_stand(robot, height=.50, half_depth=.08, half_width=.375)
            for suffix in ("fixed_stand_collision", "fixed_stand_visual"):
                support = robot.worldbody.find(f".//geom[@name='{robot.naming_prefix}{suffix}']")
                support.set("pos", ".23875 0 .25")
            robot.set_base_xpos([-.90, .30, 0])
        elif self.robot_id in ("mobile_dual_piperx", "aloha_mobile"):
            # Preserve the free chassis, wheels, and original arm mount height.
            robot.set_base_xpos([-.88, .30, 0])
        elif self.robot_id == "dual_arx_x5_robodojo":
            # Source X5 pair already includes its .765 m fixed stands.
            robot.set_base_xpos([-.66125, 0, 0])
        elif self.robot_id == "sawyer":
            robot.set_base_xpos([-.97, .10, 0])
            # Elbow-up initial branch clears the integrated Sawyer pedestal.
            self.robots[0].init_qpos = np.array([-.558505, -.335591, .147074, 2.166391, .030685, -1.302726, -1.547614])
        else:
            install_fixed_stand(robot, height=.50, half_depth=.08, half_width=.10)
            robot.set_base_xpos([-.66125, 0, 0])
        arena = FindObjectArena(self.object_names, self.scenario, self.drawer_offsets)
        self.object_heights = arena.heights
        self.model = ManipulationTask(mujoco_arena=arena,
                                     mujoco_robots=[r.robot_model for r in self.robots],
                                     mujoco_objects=[])
        if self.robot_id == "sawyer":
            # These directly connected bearing housings are tangent in the
            # upstream asset. The floor-fixed base is welded to world, so
            # MuJoCo's usual parent-body filtering does not exclude them.
            # Treat the bearing as an adjacent-link pair, not a friction brake.
            ET.SubElement(self.model.contact, "exclude", {
                "body1": "robot0_right_arm_base_link", "body2": "robot0_right_l0"})
        self.model.root.find("option").set("integrator", "implicitfast")
        self.model.root.find("option").set("impratio", "10")
        flag = self.model.root.find("option/flag")
        if flag is None:
            flag = ET.SubElement(self.model.root.find("option"), "flag")
        # MuJoCo 3.1 uses libccd by default and predates the nativeccd flag.
        if hasattr(mujoco.mjtDisableBit, "mjDSBL_NATIVECCD"):
            flag.set("nativeccd", "disable")
        flag.set("multiccd", "enable")

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.object_body_ids = [m.body_name2id(name) for name in self.object_names]
        self.object_joint_ids = [m.joint_name2id(f"{name}_free") for name in self.object_names]
        self.object_geom_ids = [set(np.flatnonzero((m.geom_bodyid == body) & (m.geom_contype != 0)))
                                for body in self.object_body_ids]
        self.robot_geom_ids = {i for i in range(m.ngeom)
                               if (m.geom_id2name(i) or "").startswith(("robot0_", "gripper0_"))}
        self.drawer_qpos_addrs = [int(m.jnt_qposadr[m.joint_name2id(f"drawer_slide_{i}")]) for i in range(self.drawer_count)]
        self.drawer_qvel_addrs = [int(m.jnt_dofadr[m.joint_name2id(f"drawer_slide_{i}")]) for i in range(self.drawer_count)]
        self.handle_site_ids = [m.site_name2id(f"drawer_handle_site_{i}") for i in range(self.drawer_count)]
        self.table_geom_id = m.geom_name2id("table_top")
        self.box_floor_geom_id = m.geom_name2id("destination_box_floor")

    @property
    def drawer_qpos_addr(self): return self.drawer_qpos_addrs[self.active_drawer]
    @property
    def drawer_qvel_addr(self): return self.drawer_qvel_addrs[self.active_drawer]
    @property
    def handle_site_id(self): return self.handle_site_ids[self.active_drawer]

    def _reset_internal(self):
        super()._reset_internal()
        self._stable_ticks = 0
        self._round_index = 0
        self.total_score = 0
        self._opened = self._success = False
        self._episode_index += 1
        self.active_drawer = 0
        self.opened_drawers = np.zeros(self.drawer_count, dtype=bool)
        if not self.deterministic_reset:
            # Task Instance owns the layout; reset must never reshuffle it.
            layout_rng = np.random.default_rng(self.scenario.seed)
            self.target_index = int(layout_rng.integers(self.target_count))
            self.sim.data.qpos[self.drawer_qpos_addrs] = 0
            self.sim.data.qvel[self.drawer_qvel_addrs] = 0
            order = layout_rng.permutation(self.target_count)
            cursor = 0
            for drawer, (count, base_count) in enumerate(zip(self.objects_per_drawer, self.base_counts)):
                indices = order[cursor:cursor + count]
                cursor += count
                slots = SLOTS[layout_rng.permutation(len(SLOTS))[:base_count]]
                occupied = []
                stack_tops = [DRAWER_FLOOR_Z + self.drawer_offsets[drawer,2]] * base_count
                for k, index in enumerate(indices):
                    column = k % base_count
                    low, high = self.object_bounds(index, local=True)
                    for attempt in range(128):
                        yaw = float(layout_rng.uniform(-.15, .15))
                        c, sn = math.cos(yaw), math.sin(yaw)
                        rotation = np.array([[c,-sn],[sn,c]])
                        half = np.abs(rotation) @ ((high[:2]-low[:2])/2)
                        xy = slots[column] + layout_rng.uniform(-.003, .003, 2)
                        center = xy + rotation @ ((low[:2]+high[:2])/2)
                        lo, hi = center-half, center+half
                        inside = np.all(lo >= [-.042,-.164]) and np.all(hi <= [.346,.164])
                        overlap = any(np.all(hi > other_lo) and np.all(lo < other_hi)
                                      for other_lo, other_hi, level in occupied if level == k//base_count)
                        if inside and not overlap:
                            break
                    else:
                        raise ValueError(f"No collision-free slot for {self.object_names[index]} in drawer {drawer+1}")
                    occupied.append((lo, hi, k//base_count))
                    z = stack_tops[column] + .001 - low[2]
                    stack_tops[column] = z + high[2]
                    # Closed drawers have a 110 mm clear height. Reject a
                    # requested stack that would intersect the cabinet above.
                    if stack_tops[column] > DRAWER_FLOOR_Z + self.drawer_offsets[drawer,2] + .110:
                        raise ValueError("Object stack exceeds drawer clearance; use clutter_layers=1")
                    position = np.r_[xy + self.drawer_offsets[drawer,:2], z]
                    joint = self.object_joint_ids[index]
                    q, v = int(self.sim.model.jnt_qposadr[joint]), int(self.sim.model.jnt_dofadr[joint])
                    self.sim.data.qpos[q:q+7] = [*position, math.cos(yaw/2),0,0,math.sin(yaw/2)]
                    self.sim.data.qvel[v:v+6] = 0
                    self.object_drawers[index] = drawer
        inventory = self.scenario.parameters.get('inventory')
        if inventory is not None:
            from .instances import validate_inventory
            validate_inventory(inventory)
            if list(self.object_names) != inventory['names']:
                raise ValueError('inventory identities differ from compiled model')
            for joint, pose in zip(self.object_joint_ids, inventory['poses']):
                q = int(self.sim.model.jnt_qposadr[joint])
                self.sim.data.qpos[q:q+7] = pose
            self.object_drawers[:] = inventory['drawers']
            targets = inventory['target_set']
        else:
            targets = np.random.default_rng(np.random.SeedSequence([self.scenario.seed, 15005])).choice(self.target_count, 5, replace=False).tolist()
        self.set_episode_targets(targets)
        self.sim.forward()
        if inventory is not None:
            for i, drawer in enumerate(self.object_drawers):
                lo, hi = self.object_bounds(i)
                offset = self.drawer_offsets[drawer]
                if (np.any(lo[:2]-offset[:2] < [-.042,-.164]) or np.any(hi[:2]-offset[:2] > [.346,.164])
                        or abs(lo[2]-(DRAWER_FLOOR_Z+offset[2]+.001))>1e-6
                        or hi[2]>DRAWER_FLOOR_Z+offset[2]+.110):
                    raise ValueError('inventory violates drawer containment')
            object_geoms = set().union(*self.object_geom_ids)
            for c in self.sim.data.contact[:self.sim.data.ncon]:
                if (c.geom1 in object_geoms or c.geom2 in object_geoms) and c.dist < -1e-7:
                    raise ValueError('inventory starts in collision')

    def set_episode_targets(self, targets):
        if len(targets) != 5 or len(set(targets)) != 5 or any(type(t) is not int or not 0 <= t < self.target_count for t in targets):
            raise ValueError('exactly five distinct target indices required')
        self._targets = list(targets)
        self._round_index = 0
        self.total_score = 0
        self._stable_ticks = 0
        self._success = False
        self.target_index = targets[0]
        self.active_drawer = int(self.object_drawers[self.target_index])

    def _settle_instruction(self, ready):
        self._stable_ticks = self._stable_ticks + 1 if ready else 0
        if self._stable_ticks < math.ceil(STABLE_SECONDS * self.control_freq):
            return False
        self.total_score += 20
        self._round_index += 1
        self._stable_ticks = 0
        self._success = self._round_index == 5
        if not self._success:
            self.target_index = self._targets[self._round_index]
            self.active_drawer = int(self.object_drawers[self.target_index])
            self._opened = bool(self.opened_drawers[self.active_drawer])
        return True

    def object_bounds(self, index, *, local=False):
        """Collision AABB in world space, or body-local space for placement."""
        m, d = self.sim.model, self.sim.data
        lower, upper = [], []
        for geom in self.object_geom_ids[index]:
            if local:
                matrix = np.zeros(9)
                mujoco.mju_quat2Mat(matrix, m.geom_quat[geom])
                r = matrix.reshape(3, 3)
            else:
                r = np.asarray(d.geom_xmat[geom]).reshape(3, 3)
            center = m.geom_pos[geom] if local else d.geom_xpos[geom]
            size = m.geom_size[geom]
            kind = m.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = np.full(3, size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                radius = np.sqrt(np.sum((r * size) ** 2, axis=1))
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                radius = size[0] * np.sqrt(np.maximum(0, 1 - r[:, 2] ** 2)) + size[1] * np.abs(r[:, 2])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius = size[0] + size[1] * np.abs(r[:, 2])
            else:  # all remaining object parts are boxes
                radius = np.abs(r) @ size
            lower.append(center - radius)
            upper.append(center + radius)
        return np.min(lower, axis=0), np.max(upper, axis=0)

    def _placement_state(self):
        ids = self.object_geom_ids[self.target_index]
        box_contact, robot_contact = False, False
        force = np.zeros(6)
        for contact_index in range(self.sim.data.ncon):
            c = self.sim.data.contact[contact_index]
            other = c.geom2 if c.geom1 in ids else c.geom1 if c.geom2 in ids else None
            if other is None:
                continue
            if other in self.robot_geom_ids and c.dist <= .001:
                robot_contact = True
            if other == self.box_floor_geom_id:
                mujoco.mj_contactForce(self.sim.model._model, self.sim.data._data, contact_index, force)
                box_contact |= bool(force[0] > 1e-4)
        low, high = self.object_bounds(self.target_index)
        contained = bool(np.all(low[:2] >= BOX_CENTER - BOX_INNER_HALF - CONTAINMENT_TOLERANCE)
                         and np.all(high[:2] <= BOX_CENTER + BOX_INNER_HALF + CONTAINMENT_TOLERANCE)
                         and abs(low[2] - BOX_FLOOR_Z) <= .006
                         and high[2] <= BOX_RIM_Z + .002)
        joint = self.object_joint_ids[self.target_index]
        v = int(self.sim.model.jnt_dofadr[joint])
        vel = self.sim.data.qvel[v:v + 6]
        stable = bool(np.linalg.norm(vel[:3]) < .025 and np.linalg.norm(vel[3:]) < .20)
        return dict(box_contact=box_contact, released=not robot_contact,
                    contained=contained, stable=stable)

    def _check_success(self):
        # Read-only: reward, logs and geometry queries cannot advance dwell time.
        return bool(self._success)

    def reward(self, action=None):
        return float(self._success)

    def step(self, action):
        if self.done:
            raise RuntimeError('episode already finished; reset before stepping')
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"expected finite action shape {(self.action_dim,)}")
        action = np.clip(action, -1, 1).copy()
        if self.robot_id in ("mobile_dual_piperx", "aloha_mobile"):
            from react_manip.robots.mobile_base import set_wheel_command
            set_wheel_command(self, 0., 0.)
        observation, _, timeout, info = super().step(action)
        self.opened_drawers |= self.sim.data.qpos[self.drawer_qpos_addrs] <= -.08
        self._opened = bool(self.opened_drawers[self.object_drawers[self.target_index]])
        state = self._placement_state()
        ready = self._opened and all(state.values())
        completed = self._settle_instruction(ready)
        terminated = bool(self._success)
        self.done = bool(terminated or timeout)
        prompt = self.get_episode_language_prompt()
        info.update(success_state=terminated, terminal_success=terminated,
                    events=(["instruction_completed"] if completed else []) + (["task_success"] if terminated else []),
                    total_score=self.total_score, score_delta=20 if completed else 0,
                    completed_rounds=self._round_index,
                    language_prompt=prompt["text"], language_prompt_structured=prompt)
        return observation, float(completed), terminated, bool(timeout and not terminated), info

    def _task_instruction(self):
        return (
            "Use either arm or both arms together to complete five search-and-place instructions in order, "
            f"sharing a {self.horizon / self.control_freq:g}-second time limit. Open drawers to find the currently named object, "
            "place it fully inside the open desktop storage box, supported by its bottom, "
            "release it and keep it stable for 0.5 seconds. Each completed instruction earns "
            "20 points; completing all five earns 100 and ends the episode. On timeout, keep "
            "earned points; unfinished instructions earn zero. Wrong objects do not count; "
            "retries are allowed. Closing drawers is not required. Do not reset between instructions. "
            "Open the target drawer at least 0.08 m at some point. The placed object must have linear speed <0.025 m/s "
            "and angular speed <0.20 rad/s, with no robot contact; its bottom must be within 0.006 m of the box floor "
            "and its top no more than 0.002 m above the rim."
        )

    def _round_instruction(self):
        if self._success or getattr(self, 'done', False):
            return ""
        en = LABELS[self.object_names[self.target_index]][0]
        return (f"Instruction {self._round_index + 1} of 5: find the {en}, place it inside "
                "the open storage box on the desktop, and release it.")

    def get_episode_language_prompt(self):
        task_instruction = self._task_instruction()
        round_instruction = self._round_instruction()
        active = bool(round_instruction)
        round_index = self._round_index if active else None
        status = ("All five instructions completed." if self._success else
                  "Time limit reached; episode ended." if not active else "in_progress")
        context = {"round_index": round_index, "round_count": 5,
                   "completed_rounds": self._round_index, "status": status}
        if active:
            context["target_name"] = LABELS[self.object_names[self.target_index]][0]
        text = " ".join(part for part in (task_instruction, round_instruction,
                                          status if not active else "") if part)
        return EpisodeLanguagePrompt(
            task_id=self.task_id, instruction=text, robot_id=self.robot_id,
            task_instruction=task_instruction, round_instruction=round_instruction,
            round_index=round_index,
            episode_index=max(0, self._episode_index),
            scenario_id=getattr(self.scenario, "scenario_id", None),
            success_condition=f"Complete all five instructions within the shared {self.horizon / self.control_freq:g}-second limit. Each named object must rest fully inside the storage box, supported by its bottom, released and stable for 0.5 seconds",
            constraints=("Either arm or both arms may be used. Release the target from both grippers before it counts. Search the drawers; the instruction does not identify a drawer. Closing the drawer is not required. Wrong objects do not count; retries are allowed.",),
            public_context=context,
        ).to_dict()


@GeometryFeatureAdapter.register("find_object_in_drawer2")
class FindObjectInDrawer2Geometry(GeometryFeatureAdapter):
    def extract(self):
        e = self.env
        objects = [{"name": name, "pose": _pose(e, body), "drawer_index": int(e.object_drawers[i]), "bounds": np.array(e.object_bounds(i))}
                   for i, (name, body) in enumerate(zip(e.object_names, e.object_body_ids))]
        return self._result(
            entities={"objects": objects,
                      "drawers": [{"index": i, "slide_position": float(e.sim.data.qpos[q]), "offset": e.drawer_offsets[i].copy()} for i,q in enumerate(e.drawer_qpos_addrs)],
                      "handle": {"position": e.sim.data.site_xpos[e.handle_site_id].copy()},
                      "table": {"top_z": TABLE_TOP_Z},
                      "destination_box": {"inner_min": np.r_[BOX_CENTER - BOX_INNER_HALF, BOX_FLOOR_Z],
                                          "inner_max": np.r_[BOX_CENTER + BOX_INNER_HALF, BOX_RIM_Z]}},
            relations={"target_index": e.target_index, "target_placement": e._placement_state()},
            task_state={"opened": e._opened, "stable_ticks": e._stable_ticks, "success": e._check_success()},
        )
