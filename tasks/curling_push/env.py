"""A robot pushes a free stone through actuator/contact dynamics only."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from react_manip.core.visual_randomization import apply_visual_scenario
from react_manip.robots.resolution import default_controller_config
from react_manip.visual_task import ManipulationTask
from .config import load_scenario
from .arena import build_xml, camera_poses
from .rules import PushState
from . import geometry  # register internal read-only features


class CurlingPush(ManipulationEnv):
    task_id = 'curling_push'
    robot_control_mode = 'eef_delta'
    step_handles_robot_action = True
    privileged_observation_keys = ('object-state', 'stone_position', 'stone_velocity', 'target_position')

    def __init__(self, robots='Panda', *, seed=0, scenario=None, profile_path=None,
                 controller_configs=None, gripper_types='default', initialization_noise=None,
                 has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                 use_object_obs=False, **kwargs):
        from react_manip.robots.resolution import robot_id_from_model, resolve_robosuite_robot
        resolved = resolve_robosuite_robot(robot_id_from_model(robots))
        self.robot_id = resolved.robot_id
        robots = resolved.robot_name
        if gripper_types == 'default':
            gripper_types = resolved.gripper_types
        if use_object_obs:
            raise ValueError('Use GeometryFeatureAdapter for internal geometry; policy sees proprio + images')
        self.scenario = load_scenario(scenario, seed=seed, profile_path=profile_path)
        self._parameters = deepcopy(self.scenario.parameters)
        self._state = PushState(self._parameters)
        self._tmp = TemporaryDirectory(prefix='curling-push-')
        self._attempt = 0
        c = self._parameters['control']
        self.position_scale, self.rotation_scale = c['position_scale'], c['rotation_scale']
        self.camera_workspace = dict(center=((self._parameters['ice']['length_m']+.2)/2,0,self._parameters['ice']['height_m']),
                                     half_size=((self._parameters['ice']['length_m']-.2)/2,.55),room_half_size=(4,2))
        self.fixed_camera_poses = camera_poses(self._parameters)
        controller = deepcopy(controller_configs or default_controller_config(robots))
        parts = controller['body_parts']
        for arm in parts.get('arms', parts).values():
            if arm['type'] != 'OSC_POSE' or arm.get('impedance_mode','fixed') != 'fixed':
                raise ValueError('curling_push requires fixed-impedance OSC_POSE')
            arm.update(input_ref_frame='world',input_type='delta',kp=300,damping_ratio=1,
                       output_max=[self.position_scale]*3+[self.rotation_scale]*3,
                       output_min=[-self.position_scale]*3+[-self.rotation_scale]*3)
        kwargs.setdefault('render_camera','workspace_oblique')
        kwargs.setdefault('camera_names',['workspace_oblique','launch_closeup'])
        # Control timing is owned by the Profile, not a second set of defaults.
        for key, expected in [('control_freq',c['hz']),('horizon',round(c['hz']*c['horizon_sec']))]:
            if key in kwargs and kwargs.pop(key) != expected:
                raise ValueError(f'{key} must match Scenario control settings')
        from react_manip.optional_integrations import preserve_native_action_units
        preserve_native_action_units(controller, self.robot_id)
        super().__init__(robots=robots, controller_configs=controller,gripper_types=gripper_types,
                         initialization_noise=initialization_noise, control_freq=c['hz'],
                         horizon=round(c['hz']*c['horizon_sec']),lite_physics=False,
                         has_renderer=has_renderer,has_offscreen_renderer=has_offscreen_renderer,
                         use_camera_obs=use_camera_obs,seed=seed,**kwargs)

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos([0,0,0])
        path = Path(self._tmp.name)/'arena.xml'
        path.write_text(build_xml(self._parameters))
        arena = Arena(str(path))
        apply_visual_scenario(arena,self.scenario)
        self.model = ManipulationTask(mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],mujoco_objects=[])
        self.model.root.find('option').set('timestep','.002')
        self.model.root.find('option').set('integrator','implicitfast')

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.stone_body_id = m.body_name2id('stone')
        self.target_body_id = m.body_name2id('target')
        self._qadr = m.get_joint_qpos_addr('stone_free')[0]
        self._vadr = m.get_joint_qvel_addr('stone_free')[0]
        self._stone_gid = m.geom_name2id('stone_collision')
        self._robot_gids = {i for i,n in enumerate(m.geom_names) if n and n.startswith(('robot0_','gripper0_'))}

    def _reset_internal(self):
        super()._reset_internal()
        self._state = PushState(self._parameters)
        p = self._parameters
        self.sim.data.qpos[self._qadr:self._qadr+7] = [p['stone']['start_x_m'],0,
            p['ice']['height_m']+p['stone']['height_m']/2+.001,1,0,0,0]
        self.sim.data.qvel[self._vadr:self._vadr+6] = 0
        self.sim.forward()
        self._finished = False

    def reset(self, *, seed=None, options=None):
        # A retry never resamples physics, even if the runner supplies a reset seed.
        # To switch hidden parameters, construct a new environment with a Scenario.
        if options and options.get('resample'):
            raise ValueError('Create a new environment with a new Scenario to resample')
        obs = super().reset()
        self._attempt += 1
        return obs, self._info()

    def stone_position(self):
        return self.sim.data.body_xpos[self.stone_body_id].copy()

    def stone_velocity(self):
        return self.sim.data.qvel[self._vadr:self._vadr+6].copy()

    def eef_pos(self, arm='right'):
        return self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]].copy()

    def eef_mat(self, arm='right'):
        return self.sim.data.site_xmat[self.robots[0].eef_site_id[arm]].reshape(3,3).copy()

    def _update_observables(self, force=False):
        super()._update_observables(force=force)
        if force or not hasattr(self,'_stone_gid'):
            return
        contact = any((c.geom1 == self._stone_gid and c.geom2 in self._robot_gids)
                      or (c.geom2 == self._stone_gid and c.geom1 in self._robot_gids)
                      for c in self.sim.data.contact[:self.sim.data.ncon])
        self._state.update(float(self.sim.data.time),self.stone_position(),self.stone_velocity(),contact)

    def _check_success(self):
        return self._state.success

    def _task_instruction(self):
        r = self._parameters['rules']
        return (
            "Push the curling stone once, release before its center crosses the blue line, "
            "then let it slide freely and stop entirely inside the red target circle. "
            f"The push may last at most {r['max_push_sec']:g} seconds; separation for {r['release_gap_sec']:g} seconds ends it. "
            f"Rest means linear speed <{r['stop_speed_mps']:g} m/s and angular speed <{r['stop_angular_speed_rps']:g} rad/s "
            f"for {r['stop_hold_sec']:g} seconds. Do not lift the stone more than {r['max_lift_m']:g} m above its resting height. "
            "A second touch after release, pushing past the blue line, pushing too long, leaving the ice, "
            "or stopping outside the target ends the attempt with zero points. "
            f"Success earns 100 points and ends the episode; timeout at {self.horizon / self.control_freq:g} seconds earns zero."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='curling_push',
            task_instruction=self._task_instruction(),
            success_condition='The released stone rests entirely inside the circle for the required hold, without violating the push, lift or boundary limits.',
            constraints=(),
        )

    def reward(self, action=None):
        return float(self._state.success)

    def _info(self, events=()):
        s = self._state
        return dict(success_state=s.success,terminal_success=s.success,events=list(events),
                    phase=s.phase,attempt=self._attempt,task_score=100. if s.success else 0.,
                    failure={'recoverable':False,'reason':s.failure} if s.failure else None)

    def step(self, action):
        if self._finished:
            raise RuntimeError('Episode ended; call reset to retry the same Scenario')
        a = np.asarray(action,dtype=float)
        if a.shape != (self.action_dim,) or not np.isfinite(a).all():
            raise ValueError(f'Expected finite action shape {(self.action_dim,)}')
        obs, _, _, _ = super().step(np.clip(a,-1,1))
        terminal = self._state.finished
        timeout = self.timestep >= self.horizon and not terminal
        if timeout:
            self._state.failure = 'timeout'
            self._state.finished = True
        self._finished = terminal or timeout
        events = ['stone_settled_in_target'] if self._state.success else []
        return obs,float(self._state.success),terminal,timeout,self._info(events)

    def close(self):
        try:
            super().close()
        finally:
            self._tmp.cleanup()
