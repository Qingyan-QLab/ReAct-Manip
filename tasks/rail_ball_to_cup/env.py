"""Physical multi-embodiment pick/place/roll/flight task. No runtime state injection."""
from copy import deepcopy
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from robosuite.models.objects import BallObject
from react_manip.core.visual_randomization import apply_visual_scenario, resolve_scenario
from react_manip.core.scenarios import sample_scenario
from react_manip.visual_task import ManipulationTask
TRAY_POS = np.array([0.60, 0.22, 0.65])
from .track import Track, Cup, BALL_RADIUS, BALL_MASS, RELEASE_RANGE, values
from . import geometry  # noqa: F401 -- task-local adapter registration

ROOT = Path(__file__).resolve().parent

# The cup is a physical collision body, so its measured ball center can sit a
# few millimetres outside the ideal mathematical cavity at rest (wall/bottom
# thickness and solver contact compliance).  Keep the tolerance small enough
# to reject a ball beside the cup while avoiding false negatives for a ball
# that is visibly in it.
CUP_POSITION_TOLERANCE = 0.008
CUP_SETTLE_SPEED = 0.12
CUP_SETTLE_SECONDS = 0.30


class RailBallToCup(ManipulationEnv):
    """Task-local robot setup, physical grasp helpers, and rail task logic.

    Each arm uses seven world EEF translation/rotation and gripper actions.
    reset replenishes the ball; the exact same Scenario is kept across retries.
    """
    task_id = 'rail_ball_to_cup'
    robot_control_mode = 'eef_delta'
    step_handles_robot_action = True
    privileged_observation_keys = ('object-state', 'obj_pos', 'obj_vel', 'pot_pos')

    def _task_instruction(self):
        return (
            "Grasp the ball and gently place it on the descending dual rails. "
            "Choose the release position so that the ball rolls through the valley, "
            "launches upward from the rail end, and lands in the cup. "
            "Do not throw the ball directly or touch it after release. "
            "The task succeeds when the released ball remains inside the cup "
            "for at least 0.3 seconds, without gripper contact and with speed <0.12 m/s. "
            "First grasp for 0.1 seconds and lift the ball at least 0.06 m clear of its initial tray height. "
            "Open the gripper to release near the descending rail, within 0.035 m of the allowed release path "
            "and with speed <=0.45 m/s. The ball must contact the valley and rail exit, then launch forward and upward. "
            "Slipping, invalid release, touching after release, rail contact after launch, falling to the floor, "
            "or resting outside the cup for one second ends the attempt with zero points. "
            f"Success earns 100 and ends the episode; the time limit is {self.horizon / self.control_freq:g} seconds, with zero on timeout."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='rail_ball_to_cup',
            task_instruction=self._task_instruction(),
            success_condition='The released ball enters the cup and stays inside for at least 0.3 seconds.',
            constraints=('Do not throw the ball directly or touch it after release. Rail geometry is fixed across retries in the same scenario.',),
        )

    def __init__(self, robots='Panda', *, scenario=None, seed=0,
                 controller_configs=None, gripper_types='default', initialization_noise=None,
                 control_freq=50, horizon=3000, has_renderer=False,
                 has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=False,
                 embodiment_config=None, **kwargs):
        from react_manip.robots.resolution import robot_id_from_model, resolve_robosuite_robot, default_controller_config
        resolved = resolve_robosuite_robot(robot_id_from_model(robots))
        self.robot_id = resolved.robot_id
        robots = resolved.robot_name
        if gripper_types == 'default':
            gripper_types = resolved.gripper_types
        self.embodiment_config = deepcopy(embodiment_config if embodiment_config is not None else
                                         json.loads((ROOT/'embodiments.json').read_text())[self.robot_id])
        self.active_arm = self.embodiment_config.get('active_arm', 'right')
        if self.active_arm not in resolved.arms:
            raise ValueError('Active arm is unavailable on this robot')
        scenario = resolve_scenario(scenario)
        if scenario is None:
            profile = json.loads((ROOT.parents[1]/'configs/rail_ball_to_cup/profile_default.json').read_text())
            scenario = sample_scenario(profile, seed=seed or 0)
        if scenario.task_id != self.task_id:
            raise ValueError('Scenario task_id must be rail_ball_to_cup')
        self.track = Track(**scenario.parameters.get('track', {}))
        self.cup = Cup(**scenario.parameters.get('cup', {}))
        from .probe import require_feasible
        require_feasible(self.track, self.cup)
        # This version calibrates one ball only; don't silently accept unsupported physics.
        ball = scenario.parameters.get('ball', {})
        if any(ball.get(key, value) != value for key, value in
               [('radius_m', BALL_RADIUS), ('mass_kg', BALL_MASS), ('friction', .8)]):
            raise ValueError('This task uses a fixed 50 mm, 100 g ball with friction 0.8')
        self.scenario = scenario
        self.ball_radius, self.ball_mass, self.ball_friction = BALL_RADIUS, BALL_MASS, .8
        self.ball_xy = np.asarray(ball.get('start_xy_m', TRAY_POS[:2]), float)
        if self.ball_xy.shape != (2,) or not np.isfinite(self.ball_xy).all() or np.any(np.abs(self.ball_xy-TRAY_POS[:2]) > .012):
            raise ValueError('Ball must start on its pickup platform')
        self.use_object_obs = use_object_obs
        self.position_scale, self.rotation_scale = .20, .20
        self._reset_tracking()
        cfg = deepcopy(controller_configs or default_controller_config(robots))
        parts = cfg['body_parts']
        for name in resolved.arms:
            arm = parts.get('arms', parts)[name]
            if arm['type'] != 'OSC_POSE' or arm.get('impedance_mode', 'fixed') != 'fixed':
                raise ValueError('rail_ball_to_cup requires fixed-impedance OSC_POSE')
            arm.update(input_ref_frame='world', input_type='delta', kp=300, damping_ratio=1,
                       output_max=[self.position_scale]*3+[self.rotation_scale]*3,
                       output_min=[-self.position_scale]*3+[-self.rotation_scale]*3)
        kwargs.setdefault('render_camera', 'agentview')
        kwargs.setdefault('camera_names', 'agentview')
        from react_manip.optional_integrations import preserve_native_action_units
        preserve_native_action_units(cfg, self.robot_id)
        ManipulationEnv.__init__(self, robots=robots, controller_configs=cfg,
                         gripper_types=gripper_types, initialization_noise=initialization_noise,
                         control_freq=control_freq, horizon=horizon, lite_physics=False,
                         has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer,
                         use_camera_obs=use_camera_obs, seed=seed, **kwargs)

    def _load_model(self):
        ManipulationEnv._load_model(self)
        self.robots[0].robot_model.set_base_xpos(self.embodiment_config.get('base_pos', [0,0,0]))
        if 'init_qpos' in self.embodiment_config:
            self.robots[0].init_qpos = np.asarray(self.embodiment_config['init_qpos'], float)
        if self.embodiment_config.get('grasp_condim'):
            for gripper in self.robots[0].gripper.values():
                for geom in self.robots[0].robot_model.worldbody.iter('geom'):
                    if geom.get('name') in gripper.contact_geoms:
                        geom.set('condim', str(self.embodiment_config['grasp_condim']))
        arena = Arena(str(ROOT/'assets/arena.xml'))
        self.track.add_to(arena.root, cup=self.cup)
        world = arena.worldbody
        ET.SubElement(world, 'geom', name='tray_floor', type='box', pos=values([.6,.22,.635]),
                      size='.018 .018 .015', material='wood_mat', group='1')
        ET.SubElement(world, 'geom', name='tray_support', type='box', pos='.6 .22 .31',
                      size='.015 .015 .31', rgba='.3 .32 .35 1', group='1')
        from react_manip.cameras import look_at
        position, target = np.array([1.3,-2.1,1.6]), np.array([.65,-.08,.65])
        rotation = look_at(position,target)
        ET.SubElement(world, 'camera', name='rail_side', pos=values(position),
                      xyaxes=values(np.r_[rotation[:,0],rotation[:,1]]), fovy='48')
        apply_visual_scenario(arena,self.scenario)
        self.obj = BallObject('ball',size=[BALL_RADIUS],density=BALL_MASS/(4/3*np.pi*BALL_RADIUS**3),
                              friction=[.8,.005,.0001],solref=[.01,1],rgba=[.9,.2,.15,1])
        self.model = ManipulationTask(mujoco_arena=arena,mujoco_robots=[r.robot_model for r in self.robots],
                                     mujoco_objects=[self.obj])
        self.model.root.find('option').attrib.update(arena.root.find('option').attrib)
        self.camera_workspace = dict(center=(.7,0,.65), half_size=(.65,.5))

    def _setup_references(self):
        ManipulationEnv._setup_references(self)
        m = self.sim.model
        self.obj_body_id = m.body_name2id(self.obj.root_body)
        self.obj_jnt_addr = m.get_joint_qpos_addr(self.obj.joints[0])[0]
        self.obj_qvel_addr = m.get_joint_qvel_addr(self.obj.joints[0])[0]
        self.obj_geom_ids = {m.geom_name2id(n) for n in self.obj.contact_geoms}
        self.pot_body_id = m.body_name2id('pot')
        self.tray_geom_ids = {i for i in range(m.ngeom) if (m.geom_id2name(i) or '').startswith('tray_')}
        gripper = self.robots[0].gripper[self.active_arm]
        self.finger_groups = [{m.geom_name2id(n) for n in gripper.important_geoms[key]}
                              for key in ('left_finger', 'right_finger')]
        self.gripper_geom_ids = {m.geom_name2id(n) for g in self.robots[0].gripper.values() for n in g.contact_geoms}
        m=self.sim.model
        self.rail_ids = {i for i in range(m.ngeom) if (m.geom_id2name(i) or '').startswith('rail_')}

    def _reset_internal(self):
        ManipulationEnv._reset_internal(self)
        self._reset_tracking()
        self.sim.data.set_joint_qpos(self.obj.joints[0],
            np.r_[self.ball_xy, TRAY_POS[2]+BALL_RADIUS+.001, 1,0,0,0])
        self.sim.data.qvel[self.obj_qvel_addr:self.obj_qvel_addr+6] = 0
        self.sim.forward()

    def ball_pos(self):
        return self.sim.data.body_xpos[self.obj_body_id].copy()

    def ball_velocity(self):
        return self.sim.data.qvel[self.obj_qvel_addr:self.obj_qvel_addr+3].copy()

    def _check_success(self):
        return self._success

    def reward(self, action=None):
        return float(self._success)

    def _contacts(self):
        touched = set()
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            if c.geom1 in self.obj_geom_ids:
                touched.add(c.geom2)
            if c.geom2 in self.obj_geom_ids:
                touched.add(c.geom1)
        return all(touched & g for g in self.finger_groups), bool(touched & self.gripper_geom_ids)

    def eef_pos(self, arm=None):
        return self.sim.data.site_xpos[self.robots[0].eef_site_id[arm or self.active_arm]].copy()

    def eef_mat(self, arm=None):
        return self.sim.data.site_xmat[self.robots[0].eef_site_id[arm or self.active_arm]].reshape(3,3).copy()

    def _reset_tracking(self):
        self.phase = 'ready'
        self._grasp_time = self._settle_time = self._rest_outside_time = 0.
        self._release_sample = None
        self._grasped = self._lifted = self._released = self._entered = False
        self._success = self._finished = self._open_requested = False
        self._failure = None
        self.attempts = 0
        self.release_position = self.release_velocity = None
        self._rail_seen = self._valley_seen = self._exit_seen = self._flight = False
        self._release_time = None
        self._launch_sample = None
        self.launch_position = self.launch_velocity = self.mouth_crossing = None
        self._previous_ball = None
        self._pending_milestones = []

    def _obj_in_pot(self):
        p = self.ball_pos()-self.cup.center
        # Use a small contact/measurement tolerance.  The old strict bounds
        # rejected valid placements when MuJoCo left the sphere a few mm into
        # the cup wall or bottom.  This still requires the ball centre to be
        # in the cup cavity, rather than merely near the cup.
        return bool(
            -CUP_POSITION_TOLERANCE < p[2] < self.cup.height - BALL_RADIUS + CUP_POSITION_TOLERANCE
            and np.linalg.norm(p[:2]) < self.cup.radius - BALL_RADIUS + CUP_POSITION_TOLERANCE
        )

    @property
    def score(self):
        return 100.0 if self._success else 0.0

    def _update_observables(self, force=False):
        # Inspect each physics substep, bypass PitchPot's prohibition on rail contact.
        ManipulationEnv._update_observables(self,force=force)
        if force or not self._lifted:
            return
        p, v = self.ball_pos(), self.ball_velocity()
        both, grip = self._contacts()
        rail = False
        exit_contact = False
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            if ({c.geom1,c.geom2} & self.obj_geom_ids) and ({c.geom1,c.geom2} & self.rail_ids):
                rail = True
                rail_id = next(iter({c.geom1, c.geom2} & self.rail_ids))
                exit_contact |= int(self.sim.model.geom_id2name(rail_id).rsplit('_', 1)[1]) >= self.track.segments-3
        # A jaw-opening contact gap is not yet a clean release. Confirm 60 ms
        # of separation, retaining the first sample of that uninterrupted gap.
        if not self._released:
            if grip:
                self._release_sample = None
            elif self._release_sample is None:
                self._release_sample = (p.copy(), v.copy(), float(self.sim.data.time))
            if self._release_sample is not None and float(self.sim.data.time)-self._release_sample[2] >= .06:
                self._released = True
                self.attempts = 1
                self.release_position, self.release_velocity, self._release_time = self._release_sample
                self._pending_milestones.append('ball_released')
                xs = np.linspace(*RELEASE_RANGE, 250)
                centers = np.array([self.track.ball_position(x, 0) for x in xs])
                distance = float(np.min(np.linalg.norm(centers-self.release_position, axis=1)))
                if not self._open_requested:
                    self._failure = 'ball_slipped'
                elif distance > .035 or np.linalg.norm(self.release_velocity) > .45:
                    self._failure = 'release_outside_track_or_thrown'
                self.phase = 'rolling'
        if not self._released:
            return
        if grip:
            self._failure = 'ball_touched_after_release'
        if rail:
            self._launch_sample = None
            self._rail_seen = True
            if .64 < p[0] < .73:
                self._valley_seen = True
            if exit_contact and self._valley_seen:
                self._exit_seen = True
        if not rail and not self._flight and self._exit_seen:
            if self._launch_sample is None:
                self._launch_sample = (p.copy(), v.copy(), float(self.sim.data.time))
            if float(self.sim.data.time)-self._launch_sample[2] >= .008 and p[0] > .875:
                launch_p, launch_v, _ = self._launch_sample
                if launch_v[0] > 0 and launch_v[2] > 0:
                    self._flight = True
                    self.launch_position, self.launch_velocity = launch_p, launch_v
                    self.phase = 'flight'
                    self._pending_milestones.append('ball_launched_from_rail')
                else:
                    self._failure = 'invalid_exit'
        if self._flight and rail:
            self._failure='rail_contact_after_launch'
        prev=self._previous_ball
        mouth=self.cup.mouth_height
        if self._flight and self.mouth_crossing is None and prev is not None and prev[2] > mouth >= p[2] and v[2] < 0:
            t=(mouth-prev[2])/(p[2]-prev[2])
            self.mouth_crossing=prev+t*(p-prev)
        self._previous_ball=p.copy()

    def step(self, action):
        if self._finished:
            raise RuntimeError('Episode ended; reset required')
        a=np.asarray(action,float)
        if a.shape != (self.action_dim,) or not np.isfinite(a).all():
            raise ValueError(f'Expected finite action shape {(self.action_dim,)}')
        if self._lifted and a[self.robots[0].composite_controller._action_split_indexes[self.active_arm+'_gripper'][0]] < -.5:
            self._open_requested=True
        before=float(self.sim.data.time)
        obs,_,done,_=ManipulationEnv.step(self,np.clip(a,-1,1))
        dt=float(self.sim.data.time)-before
        milestones=self._pending_milestones
        self._pending_milestones=[]
        both,grip=self._contacts()
        self._grasp_time=self._grasp_time+dt if both else 0.
        if not self._grasped and self._grasp_time >= .1:
            self._grasped=True
            self.phase='grasped'
            milestones.append('ball_grasped')
        if self._grasped and both and not self._lifted and self.ball_pos()[2] > TRAY_POS[2]+BALL_RADIUS+.06:
            self._lifted=True
            self.phase='lifted'
            milestones.append('ball_lifted')
        inside=self._obj_in_pot()
        # A valid release and physical cup occupancy are sufficient for the
        # terminal check.  `_flight` is a useful route milestone, but making it
        # mandatory caused false negatives when a fast rail exit was sampled
        # between control steps.
        if self._released and inside and not self._entered:
            self._entered=True
            milestones.append('ball_entered_cup')
        settled=self._released and inside and not grip and np.linalg.norm(self.ball_velocity()) < CUP_SETTLE_SPEED
        self._settle_time=self._settle_time+dt if settled else 0.
        p=self.ball_pos()
        if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
            self._failure='non_finite_state'
        elif p[2] < BALL_RADIUS+.01:
            self._failure='bounced_out' if self._entered else 'missed_cup'
        elif np.max(np.abs(p)) > 4:
            self._failure='out_of_bounds'
        rest=self._released and not inside and not grip and np.linalg.norm(self.ball_velocity()) < .04
        self._rest_outside_time=self._rest_outside_time+dt if rest else 0.
        if self._rest_outside_time >= 1.:
            self._failure='stalled_on_rail' if not self._flight else 'settled_outside_cup'
        events=[]
        if self._settle_time >= CUP_SETTLE_SECONDS and not self._failure:
            self._success=True
            self.phase='success'
            events.append('ball_settled_in_cup')
        if self._failure:
            self.phase='failed'
        terminated=bool(self._success or self._failure)
        truncated=bool(done and not terminated)
        self._finished=terminated or truncated
        info=dict(success_state=self._success,terminal_success=self._success,total_score=self.score,
                  events=events,milestones=milestones,phase=self.phase,elapsed_sec=float(self.sim.data.time),
                  failure={'recoverable':False,'reason':self._failure} if self._failure else None)
        return obs,float(self._success),terminated,truncated,info
