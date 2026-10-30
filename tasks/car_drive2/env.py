"""Straight-road obstacle driving with two physical joystick manipulators."""
from __future__ import annotations
import math
import random
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from react_manip.visual_task import ManipulationTask

from react_manip.robots.agilex_robosuite import controller_config
from react_manip.core.scenarios import sample_scenario
from react_manip.core.visual_randomization import resolve_scenario

ROOT = Path(__file__).resolve().parent
ROAD_HALF = 4.2
ROAD_LENGTH = 18.0


class VehicleTask(ManipulationTask):
    def generate_id_mappings(self, sim):
        # Mounted robots are mapped individually by robosuite. Do not map the
        # entire vehicle subtree a second time as an arena object.
        world = self.mujoco_arena.worldbody
        car = world.find("body[@name='car']")
        world.remove(car)
        try:
            super().generate_id_mappings(sim)
        finally:
            world.append(car)


class StraightRoadArena(Arena):
    def __init__(self, seed=0, *, road_half=ROAD_HALF, road_length=ROAD_LENGTH,
                 obstacle_positions=(4.5, 10.0, 15.0), extension_range=(.25, .40),
                 half_width_range=(.32, .42), road_friction=(.8, .01, .001),
                 joystick_damping=8.0, joystick_stiffness=3.0):
        super().__init__(str(ROOT / "assets" / "scene.xml"))
        rng = random.Random(seed)
        # MuJoCo-native procedural textures: no external downloads required.
        for name, c1, c2 in (
            ('car_paint', '.055 .26 .43', '.058 .27 .44'),
            ('rail_metal', '.60 .65 .70', '.63 .68 .73'),
        ):
            ET.SubElement(self.asset,'texture',name=name+'_tex',type='2d',builtin='checker',
                          width='256',height='256',rgb1=c1,rgb2=c2)
            ET.SubElement(self.asset,'material',name=name,texture=name+'_tex',
                          texrepeat='64 64',texuniform='true',specular='.55',shininess='.65')
        for geom in list(self.worldbody.findall("geom")):
            name = geom.get("name", "")
            if name.startswith(("road_", "lane_", "rail_")) or name == "terrain":
                self.worldbody.remove(geom)
        # Wide, continuous asphalt strip and shoulders.
        road_friction_text = " ".join(f"{float(v):.6g}" for v in road_friction)
        ET.SubElement(self.worldbody, "geom", name="straight_road", type="box",
                      pos=f"0 {road_length/2:.3f} .035", size=f"{road_half} {road_length/2:.3f} .035",
                      rgba=".16 .18 .21 1", friction=road_friction_text)
        for x in (-road_half - .12, road_half + .12):
            ET.SubElement(self.worldbody, "geom", name=f"road_shoulder_{x}", type="box",
                          pos=f"{x} {road_length/2:.3f} .02", size=f".12 {road_length/2:.3f} .02",
                          rgba=".30 .32 .27 1")
        # Clear dashed center line to make the corridor readable.
        # Inner fence face is x=+/-road_half: existing conservative vehicle
        # bounds stop the mocap chassis before it can pass these physical rails.
        for side in (-1,1):
            x=side*(road_half+.07)
            for z in (.30,.62):
                ET.SubElement(self.worldbody,'geom',name=f'fence_{side}_{z}',type='box',
                              pos=f'{x} {road_length/2} {z}',size=f'.07 {road_length/2} .10',
                              material='rail_metal',friction='.8 .01 .001')
            for i,y in enumerate(np.linspace(.12,road_length-.12,13)):
                ET.SubElement(self.worldbody,'geom',name=f'fence_post_{side}_{i}',type='box',
                              pos=f'{side*(road_half+.16)} {y} .37',size='.06 .065 .37',
                              material='rail_metal')
                ET.SubElement(self.worldbody,'geom',name=f'fence_reflector_{side}_{i}',type='box',
                              pos=f'{side*(road_half-.004)} {y} .62',size='.004 .055 .045',
                              rgba='1 .64 .08 1',contype='0',conaffinity='0')
        for y in np.arange(.55, road_length, 1.0):
            ET.SubElement(self.worldbody, "geom", name=f"center_line_{y:.2f}", type="box",
                          pos=f"0 {y:.3f} .076", size=".035 .25 .004", rgba=".95 .82 .16 1",
                          contype="0", conaffinity="0")
        # Obstacles are static physical boxes. At every longitudinal section
        # at least one side remains open; a small lateral gap forms the route.
        self.obstacle_specs = []
        for i, y in enumerate(tuple(float(v) for v in obstacle_positions)):
            # Three staggered, broad barriers leave alternating left/right
            # openings: the drivable corridor is visibly S-shaped while still
            # leaving generous clearance for the vehicle.
            side = (-1, 1, -1)[i]
            # Full-width side barriers: each starts just inside its fence and
            # reaches across the centre, leaving only the opposite lane open.
            # Derive both ends together: overlap the fence by 5 cm and
            # extend 25--40 cm beyond the road centre. No outside shortcut
            # and no common straight corridor between alternating barriers.
            outer = road_half + .05
            beyond_center = rng.uniform(*extension_range)
            x = side * (outer - beyond_center) / 2
            sx = (outer + beyond_center) / 2
            sy = rng.uniform(*half_width_range)
            self.obstacle_specs.append((x, y, sx, sy))
            ET.SubElement(self.worldbody, "geom", name=f"road_obstacle_{i}", type="box",
                          pos=f"{x:.3f} {y:.3f} .16", size=f"{sx:.3f} {sy:.3f} .16",
                          rgba=(".82 .32 .10 1" if i % 2 else ".16 .35 .72 1"),
                          friction=".8 .02 .002")
        # Cameras and room lighting.
        ET.SubElement(self.worldbody, "camera", name="road_overview", pos="5 -3 8",
                      xyaxes=".86 .51 0 -.25 .42 .87", fovy="55")
        ET.SubElement(self.worldbody, "camera", name="road_top", pos="0 8 24",
                      xyaxes="1 0 0 0 1 0", fovy="50")
        ET.SubElement(self.worldbody, "camera", name="driver_view", pos="0 -1.4 1.6",
                      xyaxes="1 0 0 0 .45 .89", fovy="65")
        ET.SubElement(self.worldbody, "light", name="road_fill", pos="-2 3 5",
                      dir=".1 0 -1", diffuse=".8 .8 .8", ambient=".25 .25 .25")
        # Two passive spring-centered slides, moved by physical finger contact.
        car = self.worldbody.find("body[@name='car']")
        for child in list(car):
            car.remove(child)
        ET.SubElement(car, 'geom', name='car_chassis', type='box', size='.65 .65 .12',
                      pos='0 0 .20', material='car_paint', group='1')
        # Thin decorative trim lies on the body, outside neither arm workspace
        # nor the existing chassis footprint. It adds no grasp collision.
        for y in (-.52,.52):
            ET.SubElement(car,'geom',name=f'car_stripe_{y}',type='box',
                          pos=f'0 {y} .321',size='.60 .035 .001',rgba='.88 .92 .94 1',
                          group='1',contype='0',conaffinity='0')
        for x in (-.651,.651):
            ET.SubElement(car,'geom',name=f'car_trim_{x}',type='box',
                          pos=f'{x} 0 .15',size='.001 .59 .035',material='rail_metal',
                          group='1',contype='0',conaffinity='0')
        for y in (-.42,.42):
            ET.SubElement(car,'geom',name=f'headlamp_{y}',type='box',
                          pos=f'.652 {y} .25',size='.002 .10 .035',rgba='.95 .96 .8 1',
                          group='1',contype='0',conaffinity='0')
        for x in (-.45,.45):
            for y in (-.59,.59):
                ET.SubElement(car,'geom',type='cylinder',pos=f'{x} {y} 0',size='.19 .07',
                              quat='.7071068 .7071068 0 0',rgba='.08 .08 .09 1',group='1')
        ET.SubElement(car,'camera',name='cockpit2',pos='-.9 0 1.6',
                      xyaxes='0 -1 0 .7 0 .7',fovy='65')
        for side, y in (("left", .3), ("right", -.3)):
            ET.SubElement(car,'geom',name=f'console_{side}',type='box',pos=f'.10 {y} .58',
                          size='.18 .10 .08',rgba='.25 .27 .29 1',group='1')
            # Raised to the robot's neutral wrist height so the fingers can
            # grasp the shaft without a deep downward reach.
            body = ET.SubElement(car, "body", name=f"joystick_{side}", pos=f".10 {y} .80")
            ET.SubElement(body,'inertial',pos='0 0 .04',mass='.12',diaginertia='.001 .001 .0001')
            ET.SubElement(body,'joint',name=f'joystick_{side}_slide',type='slide',axis='1 0 0',
                          limited='true',range='-.18 .18',damping=str(joystick_damping),
                          stiffness=str(joystick_stiffness),springref='0')
            ET.SubElement(body, "geom", name=f"joystick_{side}_stem", type="cylinder",
                          pos='0 0 -.07', size=".012 .07", rgba=".12 .12 .14 1",group='1',
                          contype='1',conaffinity='1',friction='.4 .003 .0001',condim='3',
                          solref='.02 1',solimp='.90 .95 .001',margin='0')
            ET.SubElement(body, "geom", name=f"joystick_{side}_grip", type="cylinder",
                          size=".022 .035", rgba=".18 .20 .23 1",group='1',
                          friction='1.2 .003 .0001',condim='3',solref='.02 1',solimp='.90 .95 .001',margin='0')
            for z in (-.044,.044):
                ET.SubElement(body,'geom',name=f'joystick_{side}_collar_{z}',type='cylinder',
                              pos=f'0 0 {z}',size='.030 .004',rgba='.95 .55 .08 1',group='1',
                              friction='.6 .003 .0001',solref='.02 1',solimp='.90 .95 .001')
        for geom in self.worldbody.findall('geom'):
            geom.set('group','1')
        ET.SubElement(self.worldbody,'geom',type='plane',size='30 30 .1',pos='0 0 -.01',
                      rgba='.3 .38 .26 1',group='1')

        from react_manip.cameras import ensure_scene_cameras
        ensure_scene_cameras(self, "car_drive2")


class CarDrive2(ManipulationEnv):

    def __init__(self, robots, controller_configs=None, gripper_types="default", seed=0, **kwargs):
        seed = 0 if seed is None else int(seed)
        scenario = kwargs.pop("scenario", None)
        self.scenario = resolve_scenario(scenario)
        if self.scenario is None:
            profile = ROOT.parents[1] / "configs" / "car_drive2" / "profile_default.json"
            if profile.exists():
                self.scenario = sample_scenario(json.loads(profile.read_text()), seed=seed)
        if self.scenario is not None and self.scenario.task_id != "car_drive2":
            raise ValueError("Scenario task_id must be car_drive2")
        params = self.scenario.parameters if self.scenario is not None else {}
        fixed_road = params.get("road", {})
        fixed_vehicle = params.get("vehicle", {})
        fixed_obstacles = params.get("obstacles", {})
        fixed_joysticks = params.get("joysticks", {})
        self._road_half = float(fixed_road.get("half_width_m", ROAD_HALF))
        self._road_length = float(fixed_road.get("length_m", ROAD_LENGTH))
        self._start_y = float(fixed_vehicle.get("start_y_m", 1.5))
        self._road_friction = tuple(fixed_road.get("surface_friction", (.8, .01, .001)))
        self._obstacle_positions = tuple(fixed_obstacles.get("longitudinal_positions_m", (4.5, 10.0, 15.0)))
        sampled_obstacle = params.get("obstacle", {})
        self._obstacle_extension = tuple(fixed_obstacles.get("extension_range_m", (.25, .40)))
        self._obstacle_half_width = tuple(fixed_obstacles.get("half_width_range_m", (.32, .42)))
        if "extension_m" in sampled_obstacle:
            self._obstacle_extension = (float(sampled_obstacle["extension_m"]),) * 2
        if "half_width_m" in sampled_obstacle:
            self._obstacle_half_width = (float(sampled_obstacle["half_width_m"]),) * 2
        if "longitudinal_positions_m" in sampled_obstacle:
            self._obstacle_positions = tuple(sampled_obstacle["longitudinal_positions_m"])
        self._nominal_speed = float(fixed_vehicle.get("nominal_speed_m_s", 1.05))
        self._steering_gain = float(fixed_vehicle.get("steering_gain", .08))
        self._joystick_damping = float(fixed_joysticks.get("damping", 8.0))
        self._joystick_stiffness = float(fixed_joysticks.get("stiffness", 3.0))
        if "damping" in params.get("joystick", {}):
            self._joystick_damping = float(params["joystick"]["damping"])
        self.seed = int(self.scenario.seed) if self.scenario is not None else int(seed)
        self._scenario_seed = self.seed
        super().__init__(robots=robots, controller_configs=controller_configs or controller_config(("right", "left")),
                         gripper_types=gripper_types, has_renderer=kwargs.pop("has_renderer", False),
                         has_offscreen_renderer=kwargs.pop("has_offscreen_renderer", True),
                         use_camera_obs=False, render_camera=kwargs.pop("render_camera", "road_overview"),
                         camera_names=kwargs.pop("camera_names", "road_overview"), horizon=kwargs.pop("horizon", 3600),
                         control_freq=20, initialization_noise=None, seed=self.seed, **kwargs)

    def _load_model(self):
        super()._load_model()
        from react_manip.robots.mounts import use_existing_mount
        use_existing_mount(self.robots[0].robot_model)
        robot = self.robots[0].robot_model
        # The custom dual PiPER-X assembly is bolted to the vehicle roof, with one arm on each side of
        # the two forward/back joystick handles.
        if robot.__class__.__name__=='RoboTwinAloha':
            root=robot.worldbody.find('body')
            for geom in list(root.findall('geom')):
                if geom.get('name','').endswith('_stand'):
                    pos=np.fromstring(geom.get('pos'),sep=' ');size=np.fromstring(geom.get('size'),sep=' ')
                    top=pos[2]+size[2];pos[2]=(top+.32)/2;size[2]=(top-.32)/2
                    geom.set('pos',' '.join(map(str,pos)));geom.set('size',' '.join(map(str,size)))
            # The source arm bases are already ~.78 m above its root.
            robot.set_base_xpos([-.35, 0., .0])
        else:robot.set_base_xpos([-.35, 0., .32])
        robot.set_base_ori(np.zeros(3))
        if robot.__class__.__name__ == "DualPiperX":
            # The shared tabletop pose intersects both vehicle consoles. Hold
            # the hands above them before allowing full-stroke gripper control.
            self.robots[0].init_qpos = np.tile([0., .2, -1.1, 1.2, 0., 0.], 2)
        # Contact geometry is supplied by the central PiPER-X asset.
        arena = StraightRoadArena(
            self._scenario_seed,
            road_half=self._road_half,
            road_length=self._road_length,
            obstacle_positions=self._obstacle_positions,
            extension_range=self._obstacle_extension,
            half_width_range=self._obstacle_half_width,
            road_friction=self._road_friction,
            joystick_damping=self._joystick_damping,
            joystick_stiffness=self._joystick_stiffness,
        )
        self._obstacle_specs = arena.obstacle_specs
        self.model = VehicleTask(mujoco_arena=arena, mujoco_robots=[robot], mujoco_objects=[])
        root = self.model.worldbody.find(f"body[@name='{robot.root_body}']")
        self.model.worldbody.remove(root)
        self.model.worldbody.find("body[@name='car']").append(root)

    def _setup_references(self):
        super()._setup_references()
        model = self.sim.model
        body = model.body_name2id("car")
        self._car_mocap = int(model.body_mocapid[body])
        self._joystick_q = {}
        for side in ("left", "right"):
            bid = int(model.get_joint_qpos_addr(f'joystick_{side}_slide'))
            self._joystick_q[side] = bid

    def _reset_internal(self):
        super()._reset_internal()
        self._car_xy = np.array([0., self._start_y], dtype=float)
        self._car_yaw = math.pi / 2
        self._left_drive = self._right_drive = 0.
        self._steps = 0
        self._crashed = False
        self._set_car_pose()
        if self.robots[0].robot_model.__class__.__name__ == "DualPiperX":
            clearance = self._initial_gripper_clearance()
            if clearance < .04:
                raise ValueError(f"Initial vehicle gripper clearance {clearance:.6f} m is below 0.04 m")

    def _initial_gripper_clearance(self):
        import mujoco
        model, data = self.sim.model, self.sim.data
        hands = [model.geom_name2id(name) for gripper in self.robots[0].gripper.values()
                 for name in gripper.contact_geoms]
        obstacles = [i for i in range(model.ngeom)
                     if not (model.geom_id2name(i) or '').startswith(('robot', 'gripper'))
                     and (model.geom_contype[i] or model.geom_conaffinity[i])]
        return min(mujoco.mj_geomDistance(model._model, data._data, hand, obstacle, 1., None)
                   for hand in hands for obstacle in obstacles)

    def _task_instruction(self):
        return (
            "Use the two arms to control the left and right vehicle joysticks. "
            "The speed produced by a given throttle position varies between Episodes and stays fixed within each Episode. "
            "Observe vehicle motion and adjust throttle and steering; the road and obstacles stay the same within a Task Instance. "
            "Push both joysticks forward to drive forward, pull both backward "
            "to reverse, and center both joysticks to stop. "
            "Use differential joystick positions to steer around the obstacles "
            "and reach the far end of the road. "
            f"Reach vehicle Y >= {self._road_length - 1.2:g} meters within "
            f"{self.horizon / self.control_freq:g} seconds to earn 100 points. "
            "Otherwise the episode times out with 0 points and failure. "
            "There is no partial credit or collision penalty; blocked moves "
            "can be recovered by reversing or steering."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='car_drive2',
            task_instruction=self._task_instruction(),
            success_condition=f"Reach Y >= {self._road_length - 1.2:g} m by the last allowed control step.",
            constraints=(),
        )

    def _set_car_pose(self):
        self.sim.data.mocap_pos[self._car_mocap] = [self._car_xy[0], self._car_xy[1], .20]
        self.sim.data.mocap_quat[self._car_mocap] = [math.cos(self._car_yaw/2), 0, 0, math.sin(self._car_yaw/2)]
        self.sim.forward()
        # OSC base-frame goals use each arm's mounting site, not the car
        # body's origin. Match CompositeController.run_controller so a goal
        # captured after reset / vehicle motion keeps the same reference frame.
        composite = getattr(self.robots[0], 'composite_controller', None)
        if composite is not None:
            composite.update_state()
            # The mocap move invalidates cached end-effector poses. Refresh
            # before the next achieved-relative policy goal is computed.
            for arm in self.robots[0].arms:
                composite.part_controllers[arm].update(force=True)

    def _observation(self):
        return {"vehicle_xy": self._car_xy.astype(np.float32).copy(),
                "vehicle_yaw": np.array([self._car_yaw], dtype=np.float32),
                "left_joystick": np.array([self._left_drive], dtype=np.float32),
                "right_joystick": np.array([self._right_drive], dtype=np.float32),
                "distance": np.array([self._car_xy[1]-self._start_y], dtype=np.float32),
                "score": np.array([self.score], dtype=np.float32)}

    def _check_success(self):
        """Reaching the far-end zone is the sole success condition."""
        return bool(self._car_xy[1] >= self._road_length - 1.2)

    @property
    def score(self):
        return 100.0 if self._check_success() else 0.0

    @property
    def total_score(self):
        return self.score

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        # Vehicle commands come ONLY from measured physical slide displacement.
        def lever(side):
            value = float(np.clip(self.sim.data.qpos[self._joystick_q[side]]/.18,-1,1))
            # A small neutral band prevents solver settling/contact noise from
            # driving the vehicle when both operators have released the sticks.
            return math.copysign(max(0.,abs(value)-.10)/.90,value)
        self._left_drive = lever('left')
        self._right_drive = lever('right')
        dt = 1 / 20
        speed = self._nominal_speed * (self._left_drive + self._right_drive) / 2
        # Gentle differential steering keeps a hand-held lever controllable;
        # the earlier gain made a small contact asymmetry spin the car into a
        # fence before the operator could correct it.
        # Keep the differential steering gentle: the two physical arms have
        # slightly different contact leverage, so a high gain amplifies tiny
        # solver asymmetries into an unwanted spin.
        yaw_rate = self._steering_gain * (self._right_drive - self._left_drive)
        old_xy, old_yaw = self._car_xy.copy(), self._car_yaw
        self._crashed = False  # A blocked move is recoverable: reverse or recenter.
        self._car_yaw += yaw_rate * dt
        self._car_xy += speed * dt * np.array([math.cos(self._car_yaw), math.sin(self._car_yaw)])
        if abs(self._car_xy[0]) > self._road_half - .95 or not .95 <= self._car_xy[1] <= self._road_length-.95:
            self._crashed = True
        for x, y, sx, sy in getattr(self, "_obstacle_specs", ()):
            if abs(self._car_xy[0] - x) < sx + .95 and abs(self._car_xy[1] - y) < sy + .95:
                self._crashed = True
        if self._crashed:
            self._car_xy, self._car_yaw = old_xy, old_yaw
            speed=0.
        self._set_car_pose(); self._steps += 1
        finished = self._check_success()
        timed_out = bool(done and not finished)
        self.done = bool(done or finished)
        if finished: info["events"] = ["drive_finished"]
        info.update(success_state=finished, terminal_success=finished,
                    score=self.score, total_score=self.total_score, score_max=100.0,
                    episode_finished=self.done, timed_out=timed_out,
                    elapsed_time_s=self.timestep / self.control_freq,
                    time_limit_s=self.horizon / self.control_freq,
                    blocked=bool(self._crashed),
                    failure="timeout" if timed_out else "obstacle_or_road_edge" if self._crashed else None,
                    vehicle_xy=self._car_xy.copy(), speed_m_s=speed,
                    joystick_left=self._left_drive, joystick_right=self._right_drive,
                    distance_m=float(self._car_xy[1]-self._start_y))
        return self.score, self.done, info

    def reward(self, action=None):
        return self.score
