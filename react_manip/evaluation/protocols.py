"""Reusable evaluation schedules selected by TaskSpec.evaluation.

Custom protocol factories receive config and return an object with run(context).
The context supplies the generic runner, task factory and policy factory.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import importlib
import time
from typing import Any, Callable

from react_manip.core.aggregate import aggregate_results
from react_manip.core.runner import BenchmarkRunner, _call_with_supported_kwargs
from react_manip.core.timeouts import (
    USE_DEFAULT_TASK_WALL_TIMEOUT,
    resolve_task_wall_timeout_sec,
)
from .metrics import pass_metrics


def _positive(value, name, *, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be an integer >= {0 if allow_zero else 1}')
    return value


@dataclass
class EvaluationContext:
    spec: Any
    factory: Callable
    policy_factory: Callable
    robot_id: str | None = None
    seed: int = 0
    scenario: Any = None
    groups: int = 1
    episodes: int | None = None
    runner: Any = field(default_factory=BenchmarkRunner)
    runner_options: dict = field(default_factory=dict)
    on_result: Callable | None = None
    viewer: Any = None

    def __post_init__(self):
        from react_manip.core.compatibility import resolve_task_robot
        self.robot_id = resolve_task_robot(self.spec, self.robot_id)

    def final_view(self, env, step, result):
        if self.viewer is None:
            return
        hz = self.runner_options.get('control_hz') or self.spec.control_hz
        horizon = self.runner_options.get('horizon_sec') or self.spec.horizon_sec
        if result.done or step >= max(1, round(hz*horizon)):
            from react_manip.core.observation import policy_observation
            self.viewer.observe(env, policy_observation(result.observation, self.spec, env), result.info)

    def emit(self, row):
        if self.on_result:
            self.on_result(row)

    def runner_options_for_episode(self, task_deadline: float | None = None) -> dict:
        """Return runner options without the protocol-only task budget."""

        options = {k: v for k, v in self.runner_options.items()
                   if k != 'task_wall_timeout_sec'}
        if task_deadline is not None:
            remaining = max(1e-6, task_deadline - time.perf_counter())
            configured = options.get('episode_wall_timeout_sec')
            if configured is None:
                configured = getattr(self.runner, 'episode_wall_timeout_sec', None)
            if configured is None:
                configured = remaining
            options['episode_wall_timeout_sec'] = min(float(configured), remaining)
        return options


class IndependentProtocol:
    def __init__(self, config):
        self.config = dict(config)
        _positive(config.get('episodes', 3), 'episodes')

    def run(self, context):
        if context.groups != 1:
            raise ValueError('--groups is only supported by grouped protocols')
        n = context.episodes if context.episodes is not None else self.config.get('episodes', 3)
        _positive(n, 'episodes')
        rows = []
        task_timeout = resolve_task_wall_timeout_sec(
            context.spec.task_id,
            context.runner_options.get('task_wall_timeout_sec', USE_DEFAULT_TASK_WALL_TIMEOUT),
        )
        task_started = time.perf_counter()
        task_deadline = None if task_timeout is None else task_started + task_timeout
        timed_out = False
        for i in range(n):
            if task_deadline is not None and time.perf_counter() >= task_deadline:
                timed_out = True
                break
            if context.viewer:
                context.viewer.begin_episode(context.spec.task_id, f'{context.spec.task_id}_episode_{i:04d}',
                                             'evaluation', i, context.runner_options.get('control_hz') or context.spec.control_hz)
            try:
                result = context.runner.evaluate(
                    context.spec, context.factory, context.policy_factory(),
                    robot_id=context.robot_id, seed=context.seed+i,
                    env_kwargs={'scenario': deepcopy(context.scenario)} if context.scenario is not None else None,
                    observation_callback=context.viewer.observe if context.viewer else None,
                    step_callback=context.final_view if context.viewer else None,
                    **context.runner_options_for_episode(task_deadline))
            finally:
                if context.viewer:
                    context.viewer.end_episode()
            row = result.to_dict()
            rows.append(row)
            context.emit(row)
            if row.get('wall_clock_timeout'):
                timed_out = True
                break
        return {'protocol': {**self.config, 'episodes': n}, 'episodes': rows,
                'summary': aggregate_results(rows) if rows else None,
                'valid': not timed_out and len(rows) == n and not any('error' in row for row in rows),
                'wall_clock_timeout': timed_out,
                'timeout_scope': 'task' if timed_out else None,
                'task_wall_time_sec': time.perf_counter() - task_started,
                'completed_episodes': len(rows),
                'skipped_episodes': n - len(rows)}


class _EpisodePolicy:
    """Separate episode initialization from the legacy policy.reset contract."""
    def __init__(self, policy, phase, index, seed):
        self.policy, self.phase, self.index, self.seed = policy, phase, index, seed
        self.policy_id = getattr(policy, 'policy_id', type(policy).__name__)
        self.requires_geometry_truth = getattr(policy, 'requires_geometry_truth', False)

    def set_action_dim(self, action_dim):
        if hasattr(self.policy, 'set_action_dim'):
            self.policy.set_action_dim(action_dim)

    def reset(self, task_spec=None, **kwargs):
        _call_with_supported_kwargs(self.policy.begin_episode, {
            'phase': self.phase, 'episode_index': self.index,
            'seed': self.seed, 'task_spec': task_spec})

    def act(self, observation, info):
        return _call_with_supported_kwargs(self.policy.act, {'observation': observation, 'info': info})

    def end_episode(self, **feedback):
        _call_with_supported_kwargs(self.policy.end_episode, {
            **feedback, 'phase': self.phase, 'episode_index': self.index})


class PracticeExamProtocol:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.practice = _positive(config.get('practice_episodes'), 'practice_episodes', allow_zero=True)
        self.exam = _positive(config.get('exam_episodes'), 'exam_episodes')
        self.online = config.get('exam_adaptation') == 'online'
        if config.get('exam_adaptation') not in ('frozen', 'online'):
            raise ValueError('unsupported exam_adaptation')
        scoring = config.get('scoring', {})
        self.score_key = scoring.get('score_key') if self.online else None
        if self.online:
            if scoring.get('primary') != 'mean_exam_task_score' or not isinstance(self.score_key, str) or not self.score_key:
                raise ValueError('online evaluation requires mean_exam_task_score and score_key')
        elif scoring.get('primary') != 'exam_success_rate' or scoring.get('scale') != 100:
            raise ValueError('practice_exam requires exam_success_rate scoring with scale=100')
        self.ks = scoring.get('pass_k', [1])
        if not isinstance(self.ks, list) or not self.ks or len(set(self.ks)) != len(self.ks):
            raise ValueError('pass_k must be a nonempty list of distinct integers')
        pass_metrics([False]*self.exam, self.ks)

    def run(self, context):
        if context.episodes is not None:
            raise ValueError('--episodes cannot override task-owned practice/exam counts')
        _positive(context.groups, 'groups')
        if context.scenario is not None and context.groups != 1:
            raise ValueError('a single --scenario requires --groups 1')
        groups, all_rows = [], []
        task_timeout = resolve_task_wall_timeout_sec(
            context.spec.task_id,
            context.runner_options.get('task_wall_timeout_sec', USE_DEFAULT_TASK_WALL_TIMEOUT),
        )
        task_started = time.perf_counter()
        task_deadline = None if task_timeout is None else task_started + task_timeout
        timed_out = False
        for group_index in range(context.groups):
            if task_deadline is not None and time.perf_counter() >= task_deadline:
                timed_out = True
                break
            # A factory must return a fresh policy, so adaptation never crosses groups.
            policy = context.policy_factory()
            required = ('begin_episode', 'end_episode')
            if not self.online:
                required += ('get_adaptation_state', 'set_adaptation_state')
            missing = [name for name in required if not callable(getattr(policy, name, None))]
            if missing:
                raise TypeError('practice_exam policy is missing: ' + ', '.join(missing))
            group_seed = context.seed + group_index
            # RGB is managed by the evaluator adapter. Native default offscreen
            # contexts can open GLFW even for state-only/headless evaluation.
            kwargs = {'seed': group_seed, 'robot_id': context.robot_id,
                      'has_offscreen_renderer': False, 'use_camera_obs': False}
            if context.scenario is not None:
                kwargs['scenario'] = deepcopy(context.scenario)
            if context.runner_options.get('render'):
                kwargs.update(has_renderer=True, has_offscreen_renderer=False, use_camera_obs=False)
            env = _call_with_supported_kwargs(context.factory, kwargs)
            rows = []
            try:
                scenario = getattr(env, 'scenario', None)
                if scenario is None:
                    raise ValueError('practice_exam requires an environment exposing a fixed Scenario')
                digest = scenario.digest
                scenario_record = deepcopy(scenario.to_dict())
                if scenario.task_id != context.spec.task_id:
                    raise ValueError('environment Scenario belongs to a different task')
                if context.scenario is not None and digest != context.scenario.digest:
                    raise ValueError('environment did not preserve the requested Scenario')
                snapshot = None
                from react_manip.instances.variation import TASKS as VARIED_TASKS, construct_sampled_episode
                varied = context.spec.task_id in VARIED_TASKS
                if varied and (self.practice, self.exam) != (10,5):
                    raise ValueError('Episode variation requires the released 10/5 schedule')
                for phase, count in (('practice', self.practice), ('exam', self.exam)):
                    if phase == 'exam' and not self.online:
                        snapshot = deepcopy(policy.get_adaptation_state())
                    for index in range(count):
                        if task_deadline is not None and time.perf_counter() >= task_deadline:
                            timed_out = True
                            break
                        if phase == 'exam' and not self.online:
                            policy.set_adaptation_state(deepcopy(snapshot))
                        # Distinct seeds for trial randomness, independent of scene sampling.
                        episode_seed = group_seed*1000003 + (0 if phase == 'practice' else self.practice) + index
                        variation = None
                        if varied:
                            next_env, variation = construct_sampled_episode(scenario, context.robot_id,
                                index + (0 if phase == 'practice' else self.practice), episode_seed,
                                render=bool(context.runner_options.get('render')))
                            env.close()
                            env = next_env
                        expected_digest = env.scenario.digest if varied else digest
                        session = _EpisodePolicy(policy, phase, index, episode_seed)
                        def check_scenario(current_env, step, result):
                            if current_env.scenario.digest != expected_digest:
                                raise ValueError('Scenario changed within a practice/exam group')
                            context.final_view(current_env, step, result)
                        if context.viewer:
                            context.viewer.begin_episode(context.spec.task_id,
                                f'{context.spec.task_id}_group_{group_index:04d}_{phase}_{index:04d}',
                                phase, index, context.runner_options.get('control_hz') or context.spec.control_hz)
                        try:
                            result = context.runner.evaluate(
                                context.spec, context.factory, session, env=env, stop_on_done=True,
                                robot_id=context.robot_id, seed=None if varied else episode_seed,
                                observation_callback=context.viewer.observe if context.viewer else None,
                                step_callback=check_scenario, score_key=self.score_key,
                                **context.runner_options_for_episode(task_deadline))
                        finally:
                            if context.viewer:
                                context.viewer.end_episode()
                        if env.scenario.digest != expected_digest:
                            raise ValueError('Scenario changed within a practice/exam group')
                        row = {**result.to_dict(), 'group_index': group_index,
                               'phase': phase, 'episode_index': index}
                        if varied:
                            row['episode_variation'] = variation
                        rows.append(row)
                        all_rows.append(row)
                        context.emit(row)
                        if row.get('wall_clock_timeout'):
                            timed_out = True
                            break
                        if self.online and "error" in row:
                            raise ValueError(row["error"])
                    if timed_out:
                        break
                if timed_out:
                    break
                exam_rows = [row for row in rows if row['phase'] == 'exam']
                metrics = pass_metrics([row['passed'] for row in exam_rows], self.ks)
                metrics['exam_scores'] = [row['episode_score'] for row in exam_rows] if self.online else None
                metrics['score'] = sum(metrics['exam_scores'])/self.exam if self.online else 100*metrics['pass_rate']
                groups.append({'group_index': group_index, 'seed': group_seed,
                               'scenario': scenario_record, 'episodes': rows, 'summary': metrics})
            finally:
                env.close()
            if timed_out:
                break
        if timed_out:
            return {
                'protocol': self.config,
                'groups': groups,
                'episodes': all_rows,
                'summary': None,
                'valid': False,
                'wall_clock_timeout': True,
                'timeout_scope': 'task',
                'task_wall_time_sec': time.perf_counter() - task_started,
                'completed_episodes': len(all_rows),
                'skipped_episodes': context.groups * (self.practice + self.exam) - len(all_rows),
            }
        count = len(groups)
        summary = {'groups': count, 'exam_episodes': count*self.exam,
                   'pass_count': sum(g['summary']['pass_count'] for g in groups),
                   'pass_rate': sum(g['summary']['pass_rate'] for g in groups)/count,
                   'score': sum(g['summary']['score'] for g in groups)/count}
        for key in ('pass_at_k', 'pass_all_k'):
            summary[key] = {str(k): sum(g['summary'][key][str(k)] for g in groups)/count for k in self.ks}
        return {'protocol': self.config, 'groups': groups, 'episodes': all_rows, 'summary': summary,
                'valid': not any('error' in row for row in all_rows)}


def load_protocol(spec):
    config = dict(spec.evaluation) or {'protocol': 'independent', 'episodes': 3}
    name = config.get('protocol')
    if name == 'independent':
        return IndependentProtocol(config)
    if name == 'practice_exam':
        return PracticeExamProtocol(config)
    if name == 'custom':
        module, name = config['factory'].split(':', 1)
        protocol = getattr(importlib.import_module(module), name)(config=config)
        if not callable(getattr(protocol, 'run', None)):
            raise TypeError('custom protocol must expose run(context)')
        return protocol
    raise ValueError(f'unknown evaluation protocol: {name!r}')
