"""Fixed-base turntable: collision-driven spin, passive coast, settled score."""
from pathlib import Path
import math
import json
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from react_manip.visual_task import ManipulationTask
from react_manip.robots.agilex_robosuite import controller_config, resolve_robot

ROOT = Path(__file__).resolve().parent
VALUES = (0, 8, 12, 20, 32, 48, 80, 100, 80, 48, 32, 20, 12, 8, 8, 4)
SCORE_MAX = max(VALUES)
SECTOR_WIDTH = 2 * math.pi / len(VALUES)
SECTOR_EDGES = np.arange(len(VALUES) + 1) * SECTOR_WIDTH


class TurntableArena(Arena):
    def __init__(self, inertia, resistance, damping, scale=1., ring_yaw=0., center_y=0.):
        super().__init__(str(ROOT / "assets/turntable_gamble_arena.xml"))
        # All passive resistance is specified at the bearing, without an
        # additional implicit aerodynamic drag from the source arena.
        self.root.find("option").set("density", "0")
        sun = self.worldbody.find("light[@name='sun']")
        sun.set("diffuse", ".45 .45 .45"); sun.set("ambient", ".10 .10 .10")
        sun.set("specular", "0 0 0")
        self.worldbody.find("body[@name='pedestal']/geom").set("pos", "0 0 0.69")
        self.worldbody.find("body[@name='pedestal']/geom").set("size", "0.20 0.01")
        self.worldbody.find("body[@name='table']/geom").set("pos", "0 0 0.706")
        ring = self.worldbody.find("body[@name='ring']")
        ring.set("pos", "0 0 0.716")
        ring.remove(ring.find("joint"))
        ring.remove(ring.find("geom[@name='ring_mass']"))

        # Copy the original turntable's geometry, including its radial handle.
        source = ET.fromstring((ROOT / "assets/turntable.xml").read_text().replace("__DENSITY__", "1000"))
        for asset in source.find("asset"):
            self.asset.append(asset)
        disk_material = self.asset.find("material[@name='m_disk']")
        disk_material.set("rgba", "0.10 0.18 0.24 1")
        disk_material.set("specular", "0.30")
        disk_material.set("shininess", "0.35")
        disk = source.find("worldbody/body/body")
        disk.set("name", "turntable")
        disk.set("pos", "0 0 0.74")
        hinge = disk.find("joint")
        hinge.set("damping", str(damping))
        hinge.set("frictionloss", str(resistance))
        # Explicit inertia includes the complete moving assembly exactly once.
        ET.SubElement(disk, "inertial", pos="0 0 0", mass="2.5",
                      diaginertia=f"{inertia * .6} {inertia * .6} {inertia}")
        disk.remove(disk.find("geom[@name='wall_smooth']"))
        for geom in list(disk.findall("geom")):
            if geom.get("name", "").startswith("rim_wall"):
                disk.remove(geom)
                continue
            geom.set("group", "1")
            geom.set("margin", "0")
        disk.find("geom[@name='disk']").set("rgba", "0.10 0.18 0.24 1")
        self._round_rim(disk, "rim", .295, .305, .010, .040, collision=True)
        self._round_rim(ring, "score_border", .439, .446, -.004, .006)
        ET.SubElement(disk, "geom", name="handle_support", type="cylinder",
                      size="0.014 0.025", pos="0.10 0 0.035", group="1", rgba="0.3 0.32 0.35 1")
        # Visible radial marker terminates over the fixed score ring.
        ET.SubElement(disk, "geom", name="pointer_line", type="box",
                      size="0.17 0.003 0.001", pos="0.19 0 0.043", group="1",
                      rgba="1 0.22 0.08 1", contype="0", conaffinity="0")
        self.worldbody.append(disk)
        self._labels(ring)
        # A broad, low paddle at the rim is the physical pushing surface.
        # The inner pointer is paint/visual geometry, not an inner grab handle.
        for name in ('stick', 'stick_tip', 'arrow_board', 'arrow_tip', 'handle_support'):
            geom = disk.find(f"geom[@name='{name}']")
            disk.remove(geom)
        # The pointer shaft is a painted line, not a non-colliding solid rod.
        disk.find("geom[@name='pointer_line']").set('pos', '.19 0 .0112')
        disk.find("geom[@name='pointer_line']").set('size', '.17 .003 .0003')
        paddle = disk.find("geom[@name='stick_plate']")
        paddle.set('pos', '.35 0 .025')
        paddle.set('size', '.045 .040 .015')
        paddle.set('rgba', '1 .35 .08 1')
        # All horizontal dimensions, mesh vertices and labels scale together.
        # Keep the near edge at its original y coordinate, clear of the fixed
        # robot base, while the enlarged far side moves out of arm reach.
        for body in (disk, ring, self.worldbody.find("body[@name='table']"),
                     self.worldbody.find("body[@name='pedestal']")):
            for element in body.iter():
                if element is not body and 'pos' in element.attrib:
                    pos = np.fromstring(element.get('pos'), sep=' '); pos[:2] *= scale
                    element.set('pos', ' '.join(map(str, pos)))
                if element.tag == 'geom':
                    if 'fromto' in element.attrib:
                        pts = np.fromstring(element.get('fromto'), sep=' ').reshape(2,3)
                        pts[:,:2] *= scale
                        element.set('fromto', ' '.join(map(str, pts.ravel())))
                    if 'size' in element.attrib:
                        size = np.fromstring(element.get('size'), sep=' ')
                        size[:2 if element.get('type') == 'box' else 1] *= scale
                        element.set('size', ' '.join(map(str, size)))
            pos = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
            pos[1] += center_y
            body.set('pos', ' '.join(map(str, pos)))
        for mesh in self.asset.findall('mesh'):
            mesh.set('scale', f'{scale} {scale} 1')
        ring.set('quat', f'{math.cos(ring_yaw/2)} 0 0 {math.sin(ring_yaw/2)}')
        self._room()
        from react_manip.cameras import ensure_scene_cameras
        ensure_scene_cameras(self, "turntable_gamble1")
        self.worldbody.find("geom[@name='desk']").set('size', '.85 .90 .025')
        self.worldbody.find("geom[@name='desk']").set('pos', '0 .10 .655')
        self.worldbody.find("camera[@name='agentview']").set("pos", "1.25 -1.60 1.85")
        self.worldbody.find("camera[@name='topview']").set("pos", f"0 {center_y} 2.9")
        self.worldbody.find("camera[@name='topview']").set("xyaxes", "1 0 0 0 1 0")
        ET.SubElement(self.worldbody, "camera", name="handleview", pos="0.8 -0.9 1.2",
                      xyaxes="0.747 0.664 0 -0.28 0.315 0.907", fovy="55")

    def _round_rim(self, parent, name, inner, outer, bottom, top, collision=False):
        """Closed annular visual mesh; local convex pieces preserve the hollow."""
        n = 128
        vertices = []
        for z in (bottom, top):
            for radius in (inner, outer):
                for i in range(n):
                    angle = 2 * math.pi * i / n
                    vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
        faces = []
        for i in range(n):
            j = (i + 1) % n
            for a, b, c, d in ((i, j, n+j, n+i), (2*n+i, 3*n+i, 3*n+j, 2*n+j),
                               (i, 2*n+i, 2*n+j, j), (n+i, n+j, 3*n+j, 3*n+i)):
                faces.extend(((a, b, c), (a, c, d)))
        ET.SubElement(self.asset, "mesh", name=name + "_mesh",
                      vertex=" ".join(str(v) for p in vertices for v in p),
                      face=" ".join(str(v) for f in faces for v in f))
        ET.SubElement(parent, "geom", name=name + "_visual", type="mesh", mesh=name + "_mesh",
                      group="1", contype="0", conaffinity="0", mass="0", rgba=".62 .69 .73 1")
        if collision:
            radius = (inner + outer) / 2
            for i in range(n):
                angle = 2 * math.pi * (i + .5) / n
                ET.SubElement(parent, "geom", name=f"{name}_contact_{i}", type="box", group="0",
                              pos=f"{radius*math.cos(angle)} {radius*math.sin(angle)} {(bottom+top)/2}",
                              size=f"{(outer-inner)/2} {outer*math.tan(math.pi/n)} {(top-bottom)/2}",
                              euler=f"0 0 {angle}", margin="0", friction="1.0 .005 .0001")

    def _labels(self, ring):
        # Seven-segment geometry keeps numbers readable without texture files.
        segments = {"a": (-1, 2, 1, 2), "b": (1, 2, 1, 0), "c": (1, 0, 1, -2),
                    "d": (-1, -2, 1, -2), "e": (-1, 0, -1, -2), "f": (-1, 2, -1, 0),
                    "g": (-1, 0, 1, 0)}
        digits = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
                  "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg"}
        for index, value in enumerate(VALUES):
            theta = (index + .5) * SECTOR_WIDTH
            label = ET.SubElement(ring, "body", name=f"score_label_{index}",
                                 pos=f"{.404 * math.cos(theta)} {.404 * math.sin(theta)} .006",
                                 euler=f"0 0 {theta - math.pi / 2}")
            text = str(value)
            for j, char in enumerate(text):
                offset = (j - (len(text) - 1) / 2) * .024
                for segment in digits[char]:
                    x1, y1, x2, y2 = segments[segment]
                    ET.SubElement(label, "geom", type="capsule", size="0.0015",
                                  fromto=f"{offset+x1*.007} {y1*.007} 0 {offset+x2*.007} {y2*.007} 0",
                                  group="1", rgba="0.03 0.03 0.03 1", contype="0", conaffinity="0")

    def _room(self):
        textures = ROOT.parent / "laptop_into_drawer/assets/textures"
        for name, filename in (("tt_wood", "light-wood.png"), ("tt_floor", "light-gray-floor-tile.png"),
                               ("tt_wall", "white-plaster.png")):
            ET.SubElement(self.asset, "texture", name=name, type="2d", file=str(textures / filename))
            ET.SubElement(self.asset, "material", name=name, texture=name, texrepeat="3 3", specular="0.08")
        self.worldbody.find("geom[@name='floor']").set("material", "tt_floor")
        ET.SubElement(self.worldbody, "geom", name="desk", type="box", pos="0 -0.08 0.655",
                      size="0.72 0.63 0.025", material="tt_wood", group="1")
        for i, (x, y) in enumerate(((-.62, -.61), (-.62, .45), (.62, -.61), (.62, .45))):
            ET.SubElement(self.worldbody, "geom", name=f"desk_leg_{i}", type="box",
                          pos=f"{x} {y} .315", size=".035 .035 .315", group="1", rgba=".2 .23 .27 1")
        for name, pos, size in (("back", "0 3 1.5", "3 .05 1.5"), ("front", "0 -3 1.5", "3 .05 1.5"),
                                ("left", "-3 0 1.5", ".05 3 1.5"), ("right", "3 0 1.5", ".05 3 1.5")):
            ET.SubElement(self.worldbody, "geom", name="room_" + name, type="box", pos=pos,
                          size=size, material="tt_wall", group="1")
        ET.SubElement(self.worldbody, "light", name="fill", pos="1 0 2.5", dir="-.3 0 -1",
                      diffuse=".25 .25 .25", ambient=".05 .05 .05", castshadow="false")


class TurntableGamble1(ManipulationEnv):
    def __init__(self, seed=0, robot_id='dual_piperx', scenario=None, **kwargs):
        from react_manip.core.scenarios import Scenario
        from react_manip.instances.records import validate_parameters
        from .instances import sample_task_instance
        if robot_id not in ('dual_piperx', 'piperx', 'aloha', 'legacy_piperx', 'panda_libero'):
            raise ValueError('turntable_gamble1 supports dual_piperx and piperx')
        self.robot_id = robot_id
        self.profile = json.loads((ROOT.parents[1]/'configs/turntable_gamble1/profile_default.json').read_text())
        if scenario is None:
            scenario = Scenario('1.0', 'turntable_gamble1', f'turntable_gamble1_{seed}', seed,
                                sample_task_instance(seed, self.profile), self.profile['assets'])
        if scenario.task_id != 'turntable_gamble1':
            raise ValueError('turntable scenario task mismatch')
        validate_parameters(scenario.parameters, {'task_id': 'turntable_gamble1', 'sampling_profile': self.profile})
        self.scenario = scenario
        self.scene_seed = scenario.seed
        t = scenario.parameters['turntable']
        init = scenario.parameters['initialization']
        resolved = scenario.parameters['derived']
        self.base_inertia = t['inertia']
        self.resistance = t['frictionloss']
        self.damping = t['damping']
        self.radius = t['radius_m']
        self.scale = resolved['scale']
        self.inertia = resolved['inertia_diagonal'][2]
        self.center_y = resolved['center_y_m']
        self.initial_angle = self.profile['initialization']['pointer_angle_rad']
        self.initial_score = init['score']
        self.initial_sector = init['sector']
        self.initial_fraction = init['fraction']
        self.ring_yaw = resolved['ring_yaw_rad']
        self.minimum_coast_angle = math.radians(self.profile['fixed']['task']['minimum_free_coast_deg'])
        self.minimum_coast_time = self.profile['fixed']['task']['minimum_free_coast_sec']
        config = controller_config(('right', 'left') if robot_id in ('dual_piperx', 'aloha') else ('right',))
        # A flick needs more tangential speed than the generic fine-positioning
        # delta cap permits at the enlarged radius. Joint/torque limits remain.
        limits = self.profile['fixed']['controller']
        delta_limits = [limits['translation_delta_max_m']]*3 + [limits['rotation_delta_max_rad']]*3
        for part in config['body_parts'].values():
            part['output_max'] = delta_limits
            part['output_min'] = [-v for v in delta_limits]
        if kwargs.pop('initialization_noise', None) is not None:
            raise ValueError('turntable initialization noise is disabled')
        if robot_id=='panda_libero':
            from react_manip.robots.resolution import resolve_robosuite_robot
            resolved=resolve_robosuite_robot(robot_id)
            config=resolved.controller_configs
            kwargs['gripper_types']=resolved.gripper_types
            model_name=resolved.robot_name
        else:model_name=resolve_robot(robot_id)
        super().__init__(robots=model_name, controller_configs=config,
                         has_renderer=kwargs.pop("has_renderer", False), has_offscreen_renderer=kwargs.pop("has_offscreen_renderer", False),
                         use_camera_obs=kwargs.pop("use_camera_obs", False), camera_names="agentview",
                         render_camera="agentview", control_freq=20, horizon=4000, seed=scenario.seed,
                         initialization_noise=None, **kwargs)

    def _load_model(self):
        super()._load_model()
        from react_manip.robots.mounts import use_existing_mount, set_fixed_mount_pose
        robot = self.robots[0].robot_model
        if self.robot_id in ('dual_piperx', 'aloha'):
            # Keep the original right-arm mounting point. The entire dual-arm
            # asset and its aligned stand move together, with the root on z=0.
            set_fixed_mount_pose(robot, position=[-.30, -.61-.23875, .68],
                                 yaw=math.pi/2, height=.68)
        else:
            # Legacy PiPER is mounted on the existing tabletop (top z=.68).
            use_existing_mount(robot)
            robot.set_base_xpos([0, -.61, 0. if self.robot_id=='panda_libero' else .68])
            robot.set_base_ori(np.array([0, 0, math.pi / 2]))
        # Use real finger collision meshes, without hidden extended grasp aids.
        # Contact geometry is supplied by the central PiPER-X asset.
        self.model = ManipulationTask(mujoco_arena=TurntableArena(self.inertia, self.resistance, self.damping,
                                                               self.scale, self.ring_yaw, self.center_y),
                                      mujoco_robots=[robot], mujoco_objects=[])

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        # robosuite overwrites XML fluid defaults while constructing the sim.
        m.opt.density = 0.0
        m.opt.viscosity = 0.0
        jid = m.joint_name2id("turntable_hinge")
        self.angle_addr = int(m.jnt_qposadr[jid]); self.speed_addr = int(m.jnt_dofadr[jid])
        self.disk_id = m.body_name2id("turntable")

    def _reset_internal(self):
        super()._reset_internal()
        # Handle initially faces the arm. No disk state is written during play.
        self.sim.data.qpos[self.angle_addr] = self.initial_angle
        self.sim.data.qvel[self.speed_addr] = 0
        self.spinning = False; self.settle_time = 0.0
        self.last_score = None
        self.episode_finished = False; self.termination_reason = None
        self.release_angle = None; self.free_coast_time = 0.; self.free_coast_angle = 0.
        self.last_coast_angle = 0.; self.last_round_valid = False
        self._control_tick_contact = False
        self.sim.forward()

    @property
    def angle(self):
        return float(self.sim.data.qpos[self.angle_addr])

    @property
    def speed(self):
        return float(self.sim.data.qvel[self.speed_addr])

    def sector_at(self, angle):
        # Exact stored boundaries belong to the following sector. Dividing
        # again can round 11*width / width to 10.999999999999998.
        return (int(np.searchsorted(SECTOR_EDGES, (angle-self.ring_yaw) % (2 * math.pi), side="right")) - 1) % len(VALUES)

    def scene_manifest(self):
        return dict(seed=self.scene_seed, profile_id=self.profile['profile_id'], radius_m=self.radius,
                    scale=self.scale, center_xyz_m=[0,self.center_y,.74], base_inertia=self.base_inertia,
                    inertia=self.inertia, frictionloss=self.resistance, damping=self.damping,
                    initial_angle=self.initial_angle, ring_yaw=self.ring_yaw, initial_score=self.initial_score,
                    initial_sector=self.initial_sector, initial_fraction=self.initial_fraction,
                    minimum_free_coast_deg=math.degrees(self.minimum_coast_angle),
                    minimum_free_coast_sec=self.minimum_coast_time,
                    controller_delta_limits=self.profile['fixed']['controller'])

    def _task_instruction(self):
        return (
            "Use the robot arms to push the turntable handle and make the "
            "turntable spin. "
            "Release the handle and allow the turntable to slow down naturally "
            "under physical friction and damping. "
            "The goal is to stop with the pointer on the highest-value sector. "
            f"Push the broad rim paddle, release, and coast freely for at least "
            f"{math.degrees(self.minimum_coast_angle):g} degrees and {self.minimum_coast_time:g} seconds. "
            f"A qualified stop on the {max(VALUES)}-point sector is successful; dragging to it is not. "
            "The first settlement ends this episode with its 0-100 score. "
            "Settlement requires angular speed <0.01 rad/s without robot contact for one second. "
            "Touching again restarts the free-coast measurement. Invalid coasting scores zero. "
            f"Timeout at {self.horizon / self.control_freq:g} seconds before settlement scores zero; scores are not accumulated."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='turntable_gamble1',
            task_instruction=self._task_instruction(),
            success_condition='valid free coast followed by settlement in the 100-point sector',
            constraints=(),
        )

    def arm_contact(self):
        m = self.sim.model
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            b1, b2 = (int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            if self.disk_id in (b1, b2):
                other = b2 if b1 == self.disk_id else b1
                if m.body_names[other].startswith(("robot0_", "gripper0_")):
                    return True
        return False

    def _check_success(self):
        return self.episode_finished and self.last_score == SCORE_MAX

    def reward(self, action=None):
        return float(self.last_score or 0) / max(VALUES) if not self.spinning else 0.0

    def _pre_action(self, action, policy_step=False):
        if policy_step:
            self._control_tick_contact = False
        self._control_tick_contact = self._control_tick_contact or self.arm_contact()
        super()._pre_action(action, policy_step)

    def _post_action(self, action):
        _, done, info = super()._post_action(action)
        contact = self._control_tick_contact or self.arm_contact()
        if not self.episode_finished and not self.spinning and contact and abs(self.speed) > .10:
            self.spinning = True; self.settle_time = 0.0
            self.release_angle = None; self.free_coast_time = 0.; self.free_coast_angle = 0.
        settled_now = False
        if self.spinning and not self.episode_finished:
            if contact:
                self.release_angle = None; self.free_coast_time = 0.; self.free_coast_angle = 0.
            else:
                if self.release_angle is None:
                    self.release_angle = self.angle
                self.free_coast_time += self.control_timestep
                self.free_coast_angle = abs(self.angle-self.release_angle)
            self.settle_time = self.settle_time + self.control_timestep if abs(self.speed) < .01 and not contact else 0.0
            if self.settle_time >= 1.0:
                self.last_round_valid = (self.free_coast_angle >= self.minimum_coast_angle
                                         and self.free_coast_time >= self.minimum_coast_time)
                self.last_coast_angle = self.free_coast_angle
                self.last_score = VALUES[self.sector_at(self.angle)] if self.last_round_valid else 0
                self.episode_finished = True; self.termination_reason = "settled"
                self.spinning = False; settled_now = True
        if done and not self.episode_finished:
            self.last_score = 0
            self.episode_finished = True; self.termination_reason = "timeout"
            self.spinning = False
        self.done = bool(done or self.episode_finished)
        current = VALUES[self.sector_at(self.angle)]
        info.update(angle_deg=math.degrees(self.angle) % 360, angular_speed=self.speed,
                    current_score=current, score=self.last_score if self.last_score is not None else 0,
                    episode_finished=self.episode_finished, termination_reason=self.termination_reason,
                    phase=self.termination_reason or ("spinning" if self.spinning else "ready"),
                    arm_contact=contact, settled_now=settled_now,
                    ring_yaw_deg=math.degrees(self.ring_yaw)%360, initial_score=self.initial_score,
                    free_coast_deg=math.degrees(self.free_coast_angle), free_coast_time=self.free_coast_time,
                    last_round_valid=self.last_round_valid, radius_m=self.radius,
                    **{'TimeLimit.truncated': self.termination_reason == 'timeout'},
                    success_state=self._check_success(),
                    terminal_success=self._check_success())
        # Reward is awarded once per settled spin, never while dragging.
        return (self.last_score / max(VALUES) if settled_now else 0.0), self.done, info
