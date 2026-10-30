"""Manifest execution. Files stay on the evaluator side of the policy boundary."""
import json
import math
import time
from pathlib import Path

from .records import VERSION, SUPPORTED_TASKS, load_suite, save_record, seal
from react_manip.core.timeouts import (
    DEFAULT_TASK_WALL_TIMEOUT_SEC,
    USE_DEFAULT_TASK_WALL_TIMEOUT,
    resolve_task_wall_timeout_sec,
    validate_timeout,
)

PROTOCOL = dict(name='practice_exam', practice_episodes=10, exam_episodes=5,
                exam_adaptation='online', invalid='abort_group',
                clock='synchronous_simulation', score='mean_exam_task_score', policy_track='public')






def evaluate(path, policy_factory, *, cameras=True, runner=None, env_factory=None, on_result=None,
              trace_dir=None, visualize=False, video_dir=None, policy_factory_for_robot=None,
              close_policy=False, task_wall_timeout_sec=USE_DEFAULT_TASK_WALL_TIMEOUT,
              episode_wall_timeout_sec=None):
    if (visualize or video_dir is not None) and not cameras:
        raise ValueError('visualization and video require policy images')
    from react_manip.evaluation.viewer import EvaluationViewer
    with EvaluationViewer(enabled=visualize, video_dir=video_dir) as viewer:
        return _evaluate(path, policy_factory, cameras=cameras, runner=runner,
                         env_factory=env_factory, on_result=on_result,
                         policy_factory_for_robot=policy_factory_for_robot, close_policy=close_policy,
                         trace_dir=trace_dir, viewer=viewer if viewer.active else None,
                         task_wall_timeout_sec=task_wall_timeout_sec,
                         episode_wall_timeout_sec=episode_wall_timeout_sec)


def _evaluate(path, policy_factory, *, cameras=True, runner=None, env_factory=None,
              on_result=None, trace_dir=None, viewer=None, policy_factory_for_robot=None, close_policy=False,
              task_wall_timeout_sec=USE_DEFAULT_TASK_WALL_TIMEOUT, episode_wall_timeout_sec=None):
    """Fresh policy per instance; all fifteen episodes retain learned state."""
    from react_manip.core.runner import BenchmarkRunner
    from react_manip.evaluation.protocols import _EpisodePolicy
    from react_manip.evaluation.viewer import SkipEpisode
    from .runtime import ReplayEnv, spec_for, contract_for
    manifest, groups = load_suite(path)
    if trace_dir is not None:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=False)
    runner = runner or BenchmarkRunner()
    env_factory = env_factory or ReplayEnv
    explicit_episode_timeout = validate_timeout(episode_wall_timeout_sec, 'episode_wall_timeout_sec')
    task_started = {}
    task_timeouts = {}
    timed_out_tasks = set()
    rows, outcomes = [], []
    for instance, episodes in groups:
        task_id = instance['task_id']
        task_started.setdefault(task_id, time.perf_counter())
        task_timeouts.setdefault(task_id, resolve_task_wall_timeout_sec(task_id, task_wall_timeout_sec))
        task_timeout = task_timeouts[task_id]
        task_deadline = None if task_timeout is None else task_started[task_id] + task_timeout
        if task_id in timed_out_tasks or (task_deadline is not None and time.perf_counter() >= task_deadline):
            timed_out_tasks.add(task_id)
            outcomes.append({'task_instance_id': instance['task_instance_id'], 'valid': False,
                             'exam_scores': [], 'error': 'wall_clock_timeout',
                             'completed_episodes': 0, 'skipped_episodes': len(episodes), 'score': None,
                             'timeout_scope': 'task'})
            continue
        group_rows, error, env, policy = [], None, None, None
        try:
            policy = policy_factory_for_robot(instance['robot_id']) if policy_factory_for_robot else policy_factory()
            for name in ('begin_episode', 'end_episode'):
                if not callable(getattr(policy, name, None)):
                    raise TypeError(f'policy requires {name}')
            if getattr(policy, 'requires_geometry_truth', False):
                raise ValueError('public track rejects privileged geometry policies')
            env = env_factory(instance, cameras=cameras)
            spec = spec_for(instance['task_id'])
            score_key = contract_for(instance['task_id'])['evaluation']['score_key']
            for index, episode in enumerate(episodes):
                if task_deadline is not None and time.perf_counter() >= task_deadline:
                    timed_out_tasks.add(task_id)
                    error = 'wall_clock_timeout'
                    break
                phase = 'practice' if index < 10 else 'exam'
                env.bind(episode)
                # Policy RNG seed is protocol-owned and is not the private environment seed.
                session = _EpisodePolicy(policy, phase, index if index < 10 else index-10, index)
                writer = None
                if viewer:
                    viewer.begin_episode(instance['task_id'], episode['episode_id'], phase,
                                         index if index < 10 else index-10, spec.control_hz)
                callback = None
                if trace_dir is not None:
                    from .trajectory import TrajectoryWriter, RecordingPolicy
                    writer = TrajectoryWriter(trace_dir / episode['episode_id'], {
                        'manifest_digest': manifest['content_digest'], 'task_instance_digest': instance['content_digest'],
                        'episode_digest': episode['content_digest'], 'runtime': episode.get('runtime', instance['runtime']),
                        'phase': phase, 'policy_id': session.policy_id,
                        'policy_provenance': getattr(policy, 'provenance', None),
                        'policy_track': 'public', 'cameras_enabled': cameras,
                        'action_application': 'reject invalid; no clipping',
                        'control_hz': spec.control_hz})
                    session = RecordingPolicy(session, writer)
                    def callback(current_env, step, result):
                        from react_manip.core.observation import policy_observation
                        writer.write({'type': 'transition', 'tick': step-1,
                                      'sim_time_sec': float(current_env.env.sim.data.time),
                                      'next_observation': policy_observation(result.observation, spec, current_env),
                                      'reward': result.reward, 'terminated': result.terminated,
                                      'truncated': result.truncated, 'info': result.info})
                trace_callback = callback
                def callback(current_env, step, result):
                    if trace_callback:
                        trace_callback(current_env, step, result)
                    if viewer and (result.done or step >= round(spec.horizon_sec * spec.control_hz)):
                        from react_manip.core.observation import policy_observation
                        viewer.observe(current_env, policy_observation(result.observation, spec, current_env), result.info)
                skipped = False
                try:
                    remaining = None if task_deadline is None else max(1e-6, task_deadline - time.perf_counter())
                    runner_episode_timeout = explicit_episode_timeout
                    if runner_episode_timeout is None:
                        runner_episode_timeout = getattr(runner, 'episode_wall_timeout_sec', None)
                    if remaining is not None:
                        runner_episode_timeout = remaining if runner_episode_timeout is None else min(runner_episode_timeout, remaining)
                    result = runner.evaluate(spec, lambda **kwargs: env, session, env=env, seed=None,
                                             robot_id=instance['robot_id'], camera_observations=cameras, stop_on_done=True, step_callback=callback,
                                             observation_callback=viewer.observe if viewer else None,
                                             horizon_sec=instance['runtime']['horizon_sec'],
                                             control_hz=instance['runtime']['control_hz'], score_key=score_key,
                                             episode_wall_timeout_sec=runner_episode_timeout)
                    if writer:
                        writer.write({'type': 'episode_result', **result.to_dict()})
                except SkipEpisode:
                    # A skipped episode is recorded and the next saved
                    # episode is evaluated. It intentionally makes the group
                    # invalid for protocol scoring, but must not abort the
                    # remaining episodes or groups.
                    skipped = True
                    if writer:
                        writer.write({'type': 'skipped', 'reason': 'viewer_skip'})
                    try:
                        session.end_episode(skipped=True, passed=False)
                    except Exception:
                        # The skip action itself is still honored; policy
                        # cleanup errors are represented by the invalid row.
                        pass
                except Exception as exc:
                    if writer:
                        writer.write({'type': 'invalid', 'reason': f'{type(exc).__name__}: {exc}'})
                    raise
                finally:
                    if viewer:
                        viewer.end_episode()
                    if writer:
                        writer.close()
                if skipped:
                    row = {'task_id': instance['task_id'], 'task_instance_id': instance['task_instance_id'],
                           'task_instance_digest': instance['content_digest'],
                           'episode_id': episode['episode_id'], 'episode_digest': episode['content_digest'],
                           'phase': phase, 'episode_index': index, 'skipped': True,
                           'error': 'skipped_by_viewer', 'passed': False, 'episode_score': None}
                    rows.append(row)
                    group_rows.append(row)
                    if on_result:
                        on_result(row)
                    continue
                row = {**result.to_dict(), 'phase': phase, 'episode_index': index,
                       'task_instance_id': instance['task_instance_id'],
                       'task_instance_digest': instance['content_digest'],
                       'episode_id': episode['episode_id'], 'episode_digest': episode['content_digest']}
                if 'error' not in row:
                    value = row.get('episode_score')
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        row['error'] = 'missing or non-finite episode_score'
                rows.append(row); group_rows.append(row)
                if on_result:
                    on_result(row)
                if 'error' in row:
                    if row.get('error') == 'wall_clock_timeout':
                        timed_out_tasks.add(task_id)
                    raise ValueError(row['error'])
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if env is not None:env.close()
            finally:
                if close_policy and policy is not None:
                    try:getattr(policy,'shutdown',policy.close)()
                    except Exception as exc:error=f'Policy cleanup failed: {type(exc).__name__}: {exc}'
        valid = (error is None and len(group_rows) == 15 and
                 not any('error' in row or row.get('skipped') for row in group_rows))
        exam_scores = [r.get('episode_score') for r in group_rows[10:]]
        score = sum(exam_scores)/5 if valid else None
        outcomes.append({'task_instance_id': instance['task_instance_id'], 'valid': valid,
                         'exam_scores': exam_scores, 'error': error, 'completed_episodes': len(group_rows), 'score': score,
                         'skipped_episodes': sum(bool(r.get('skipped')) for r in group_rows),
                         'timeout_scope': 'task' if task_id in timed_out_tasks else None})
        if viewer and viewer.aborted:
            break
    valid = len(outcomes) == len(groups) and all(g['valid'] for g in outcomes)
    skipped_episodes = sum(len(episodes) for instance, episodes in groups
                           if instance['task_id'] in timed_out_tasks)
    task_wall_time_sec = {task_id: time.perf_counter() - started
                          for task_id, started in task_started.items()}
    return {'schema_version': VERSION, 'suite_id': manifest['suite_id'], 'manifest_digest': manifest['content_digest'],
            'protocol': manifest['protocol'], 'valid': valid,
            'mode': 'public_visual' if cameras else 'state_only_smoke',
            'protocol_score': sum(g['score'] for g in outcomes)/len(outcomes) if valid and cameras else None,
            'official_score': None,  # Pilot contracts have not passed publication gates.
            'groups': outcomes, 'episodes': rows,
            'limitations': ['in-process policy state isolation relies on adapter contract',
                            'pilot suite; task solvability and publication gates are separate'],
            'wall_clock_timeout': bool(timed_out_tasks),
            'timeout_scope': 'task' if timed_out_tasks else None,
            'task_wall_timeout_sec': (DEFAULT_TASK_WALL_TIMEOUT_SEC
                                      if task_wall_timeout_sec is USE_DEFAULT_TASK_WALL_TIMEOUT
                                      else validate_timeout(task_wall_timeout_sec, 'task_wall_timeout_sec')),
            'task_wall_timeout_sec_by_task': task_timeouts,
            'task_wall_time_sec': task_wall_time_sec,
            'completed_episodes': len(rows),
            'skipped_episodes': sum(bool(row.get('skipped')) for row in rows) + max(0, skipped_episodes - sum(
                1 for row in rows if row.get('task_instance_id') in {
                    instance['task_instance_id'] for instance, episodes in groups
                    if instance['task_id'] in timed_out_tasks
                }))}
