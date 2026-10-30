"""Versioned physical adapters; all runtime sampling is confined to generation.

Initial state uses MuJoCo's version-bound integration vector. Controllers start
from reset targets at that state; this is an initial-state format, not a generic
mid-trajectory checkpoint. No simulator state is assigned during env.step().
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

from .records import VERSION, SUPPORTED_TASKS, digest, seal, validate_pair, validate_record, validate_parameters

ROOT = Path(__file__).resolve().parents[2]
from react_manip.robots.sensors import policy_camera_ids

_SOURCE_DRIFT_WARNED: set[str] = set()


def _check_runtime_identity(actual, expected, task_id):
    """Validate compiled runtime fields while tolerating evaluator-only drift.

    Saved instances contain a broad source digest. Changes to the viewer or
    evaluator control flow must not invalidate already-recorded physical
    scenes, so a source-digest-only difference is reported as a warning. Model,
    controller, camera, and physics identity changes remain hard failures.
    """

    mismatches = [key for key in expected if actual.get(key) != expected.get(key)]
    mismatches.extend(key for key in actual if key not in expected)
    if not mismatches:
        return
    if set(mismatches) == {'source_digest'}:
        if task_id not in _SOURCE_DRIFT_WARNED:
            warnings.warn(
                f'{task_id}: saved source digest differs from the current evaluator; '
                'continuing because compiled runtime identity matches',
                RuntimeWarning,
                stacklevel=3,
            )
            _SOURCE_DRIFT_WARNED.add(task_id)
        return
    raise ValueError('compiled model/controller/runtime mismatch')


def source_digest(task_id):
    paths = list((ROOT / 'react_manip').rglob('*.py')) + list((ROOT / 'tasks' / task_id).rglob('*.py'))
    paths += list((ROOT / 'tasks' / task_id).rglob('*.xml'))
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _model_digest(env):
    # Hash compiled geometry/dynamics/assets, not XML temporary file names.
    model = env.sim.model._model
    names = ('qpos0 body_pos body_quat body_mass body_inertia body_ipos body_iquat '
             'jnt_type jnt_axis jnt_pos jnt_range dof_damping dof_frictionloss dof_armature '
             'geom_type geom_bodyid geom_pos geom_quat geom_size geom_friction geom_solref geom_solimp '
             'geom_contype geom_conaffinity geom_condim pair_friction pair_solref pair_solimp '
             'actuator_gainprm actuator_biasprm actuator_dynprm actuator_ctrlrange actuator_forcerange '
             'eq_data eq_type eq_obj1id eq_obj2id mesh_vert mesh_face mesh_normal '
             'cam_bodyid cam_pos cam_quat cam_fovy light_pos light_dir light_diffuse light_specular light_ambient '
             'geom_rgba geom_matid mat_rgba mat_texid mat_texrepeat mat_emission mat_specular mat_shininess mat_reflectance tex_data').split()
    h = hashlib.sha256()
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            array = np.ascontiguousarray(value)
            h.update(name.encode()); h.update(str((array.shape, str(array.dtype))).encode()); h.update(array.tobytes())
    h.update(bytes(model.names))
    h.update(str((model.nq, model.nv, model.na, model.nu, model.opt.timestep,
                  model.opt.integrator, model.opt.solver, model.opt.iterations,
                  model.opt.tolerance, tuple(model.opt.gravity))).encode())
    return h.hexdigest()


def spec_for(task_id):
    from react_manip.core.specs import load_task_spec
    if task_id not in SUPPORTED_TASKS:
        raise ValueError('task has not passed the instance adapter gate')
    return load_task_spec(ROOT / 'tasks' / task_id / 'task_spec.json')


def contract_for(task_id):
    return json.loads((ROOT / 'tasks' / task_id / 'instance_contract.json').read_text())


def _construct(task_id, scenario, robot_id=None, episode=None, *, render=False):
    from react_manip.builtin import discover_task_definitions
    definition = next((s, f) for s, f in discover_task_definitions(task_ids=(task_id,)) if s.task_id == task_id)
    if episode is None:
        episode = deepcopy(scenario.metadata.get('initial_episode_scene'))
    extra = {'episode_instance': episode} if task_id in ('phone_lean','leveling_board1','rope_hang','laptop_into_drawer') else {}
    env = definition[1](**extra, scenario=scenario, robot_id=robot_id, seed=scenario.seed,
                        initialization_noise=None, has_renderer=render, has_offscreen_renderer=False,
                        use_camera_obs=False)
    return env


def runtime_identity(env, task_id):
    import mujoco
    import robosuite
    raw = getattr(env, 'env', env)
    spec = spec_for(task_id)
    controllers = [r.composite_controller.composite_controller_specific_config
                   if hasattr(r.composite_controller, 'composite_controller_specific_config') else {}
                   for r in raw.robots]
    # Per-arm config is the concrete config supplied to controller construction.
    controllers += [getattr(r, 'controller_config', {}) for r in raw.robots]
    for robot in raw.robots:
        controllers.append({name: {
            'class': type(part).__module__ + '.' + type(part).__name__,
            **{key: _json(getattr(part, key)) for key in (
                'input_min', 'input_max', 'output_min', 'output_max', 'kp', 'kd',
                'damping_ratio', 'control_dim', 'input_ref_frame', 'input_type', 'impedance_mode')
               if hasattr(part, key)}} for name, part in robot.composite_controller.part_controllers.items()})
    return dict(python='.'.join(map(str, sys.version_info[:2])), numpy=np.__version__,
                mujoco=mujoco.__version__, robosuite=robosuite.__version__,
                source_digest=source_digest(task_id), control_hz=spec.control_hz,
                horizon_sec=spec.horizon_sec, physics_timestep=float(raw.sim.model.opt.timestep),
                model_digest=_model_digest(raw),
                action_layout=_json(raw.robots[0].composite_controller._action_split_indexes),
                controller_digest=digest(_json(controllers)), cameras=list(policy_camera_ids(raw.robot_id if hasattr(raw, 'robot_id') else spec.supported_robots[0])))


def capture_state(env):
    import mujoco
    raw = getattr(env, 'env', env)
    model, data = raw.sim.model._model, raw.sim.data._data
    state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    vector = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, vector, state_spec)
    return dict(format='mujoco.integration.v1', state_spec=state_spec, integration_state=vector.tolist())






class ReplayEnv:
    """Bind explicit episodes to an instance; never forward their files to policy."""
    def __init__(self, instance, *, cameras=True, native_actions=False):
        self._native_actions = native_actions
        validate_record(instance)
        spec = spec_for(instance['task_id'])
        if digest(spec.to_dict()) != instance['task_spec_digest'] or digest(contract_for(spec.task_id)) != instance['contract_digest']:
            raise ValueError('TaskSpec/instance contract version mismatch')
        validate_parameters(instance['scenario']['parameters'], contract_for(spec.task_id))
        from react_manip.core.scenarios import Scenario
        data = instance['scenario']
        scenario = Scenario(**{k: deepcopy(v) for k, v in data.items() if k != 'digest'})
        self._instance = deepcopy(instance)
        self._cameras = cameras
        self._episode = None
        self._wrapped = _construct(spec.task_id, scenario, instance['robot_id'])
        self.env = self._wrapped.env  # policy geometry filter expects the task here
        try:
            self._wrapped.reset()
            _check_runtime_identity(runtime_identity(self._wrapped, spec.task_id),
                                    instance['runtime'], spec.task_id)
            if cameras:
                self._wrapped.set_camera_observations(instance['runtime']['cameras'])
        except BaseException:
            self.close()
            raise

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def bind(self, episode):
        validate_pair(self._instance, episode)
        if 'variation' in episode or (self._episode and 'variation' in self._episode):
            from react_manip.core.scenarios import Scenario
            from .variation import episode_scenario
            data = self._instance['scenario']
            scenario = Scenario(**{k: deepcopy(v) for k,v in data.items() if k != 'digest'})
            if 'variation' in episode:
                scenario = episode_scenario(scenario, episode['variation'])
            candidate = _construct(self._instance['task_id'], scenario,
                                   self._instance['robot_id'], episode=episode.get('scene'))
            try:
                candidate.reset()
                expected = episode.get('runtime', self._instance['runtime'])
                _check_runtime_identity(runtime_identity(candidate, self._instance['task_id']),
                                        expected, self._instance['task_id'])
                if self._cameras:
                    candidate.set_camera_observations(expected['cameras'])
            except BaseException:
                candidate.close()
                raise
            self._wrapped.close()
            self._wrapped, self.env = candidate, candidate.env
        if self._instance['task_id'] == 'phone_lean':
            # Poses reference the explicit Episode scene.
            self.env.episode_instance = deepcopy(episode['scene'])
            self.env.episode_seed = episode['scene']['episode_seed']
            self.env.layout_seed = self.env.episode_seed
        if self._instance['task_id'] in ('leveling_board1','rope_hang','laptop_into_drawer'):
            self.env.episode_instance = deepcopy(episode['scene'])
        self._episode = deepcopy(episode)

    def reset(self, *, seed=None, **kwargs):
        if seed is not None or kwargs:
            raise ValueError('explicit episode replay forbids seed/options overrides')
        if self._episode is None:
            raise ValueError('bind an Episode Instance before reset')
        import mujoco
        self._wrapped.reset()
        self._wrapped._episode_index = 0
        raw = self.env
        saved = self._episode['initial_state']
        if self._instance['task_id'] in ('rail_ball_to_cup', 'pitch_pot', 'conveyor_grasp', 'turntable_gamble1', 'ring_toss1', 'box_insertion', 'car_drive2', 'rope_hang', 'pan_flip', 'button_indicator_matching', 'laptop_into_drawer', 'find_object_in_drawer2', 'tilte_maze'):
            # State must match the resolved model's reset, including Episode
            # parameters; arbitrary resealed robot/object pose edits are forbidden.
            if saved != capture_state(self._wrapped):
                raise ValueError(f"{self._instance['task_id']} forbids Episode initial-state variation")
        model, data = raw.sim.model._model, raw.sim.data._data
        if saved['state_spec'] != int(mujoco.mjtState.mjSTATE_INTEGRATION):
            raise ValueError('initial state mask mismatch')
        vector = np.asarray(saved['integration_state'], dtype=float)
        if vector.shape != (mujoco.mj_stateSize(model, saved['state_spec']),) or not np.isfinite(vector).all():
            raise ValueError('initial state shape/value mismatch')
        # This v1 format supports reset states only, not arbitrary task checkpoints.
        if vector[0] != 0:
            raise ValueError('initial state time must be zero')
        mujoco.mj_setState(model, data, vector, saved['state_spec'])
        raw.sim.forward()
        raw.timestep = 0
        raw.cur_time = 0
        raw.done = False
        for robot in raw.robots:
            robot.composite_controller.update_state()
            robot.composite_controller.reset()
            for controller in robot.composite_controller.part_controllers.values():
                if hasattr(controller, 'reset_goal'):
                    controller.reset_goal()
        if self._instance['task_id'] == 'car_drive2':
            # Native reset restores Python vehicle state; sync moving controller origins.
            raw._set_car_pose()
        # Resetting targets may touch controls; restore the exact integration input.
        mujoco.mj_setState(model, data, vector, saved['state_spec'])
        raw.sim.forward()
        if self._instance['task_id'] in ('button_indicator_matching', 'find_object_in_drawer2'):
            raw.set_episode_targets(self._episode['round_plan']['targets'])
            raw._episode_index = 0
        else:
            raw._attempt = 1
        raw._update_observables(force=True)
        obs = raw._get_observations(force_update=True)
        if self._instance['task_id'] in ('button_indicator_matching', 'conveyor_grasp'):
            obs = raw._observation(obs)
        elif self._instance['task_id'] == 'box_insertion':
            obs = raw._observation()
        prompt = self._wrapped.get_language_prompt()
        self._wrapped.language_prompt = prompt
        # mj_forward refreshes warm-start acceleration by tiny roundoff amounts.
        # Restore the recorded integration input after all derived observations.
        obs = self._wrapped._with_camera_observations(obs)
        mujoco.mj_setState(model, data, vector, saved['state_spec'])
        return obs, {
            'success_state': False, 'terminal_success': False, 'events': [],
            'language_prompt': prompt['text'], 'language_prompt_structured': prompt}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        low,high=-np.ones(self._wrapped.action_dim),np.ones(self._wrapped.action_dim)
        if self._native_actions:
            robot=self.env.robots[0]
            for arm in robot.arms:
                part=robot.part_controllers[arm]
                if part.name=='JOINT_POSITION' and getattr(part,'input_type',None)=='absolute':
                    s,e=robot.composite_controller._action_split_indexes[arm]
                    limits=self.env.sim.model.jnt_range[part.joint_indexes['joints']]
                    low[s:e],high[s:e]=limits[:,0],limits[:,1]
        if (action.shape != low.shape or not np.isfinite(action).all()
                or np.any(action<low-1e-8) or np.any(action>high+1e-8)):
            raise ValueError('action must be finite and inside the recorded controller bounds')
        result = self._wrapped.step(action)
        if not np.isfinite(self.env.sim.data.qpos).all() or not np.isfinite(self.env.sim.data.qvel).all():
            raise ValueError('non-finite simulation state; evaluation is invalid')
        # Refresh language after dynamic round transitions; do not send the future plan.
        prompt = self._wrapped.get_language_prompt()
        self._wrapped.language_prompt = prompt
        obs, reward, terminated, truncated, info = result
        if self._instance['task_id'] == 'rope_hang' and (info.get('failure') or {}).get('reason') in ('physics_instability', 'non_finite_state'):
            raise ValueError('numerical rope simulation failure; evaluation is invalid')
        info = dict(info, language_prompt=prompt['text'], language_prompt_structured=prompt)
        return obs, reward, terminated, truncated, info

    def close(self):
        self._wrapped.close()
