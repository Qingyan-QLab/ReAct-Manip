"""Small, explicit API for one development episode of box_insertion.

The returned records are compatible with the existing instance validators and
replay adapter. Scores obtained here are not official benchmark results.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from react_manip.core.compatibility import resolve_task_robot
from react_manip.core.scenarios import sample_scenario
from .records import VERSION, digest, load_record, save_record, seal, validate_pair
from .runtime import ReplayEnv, _construct, capture_state, contract_for, runtime_identity, spec_for
from .variation import episode_scenario, sample_variation


@dataclass(frozen=True)
class InstanceBundle:
    task: dict
    episode: dict

    def save(self, directory):
        directory = Path(directory)
        save_record(self.task, directory / 'task.json')
        save_record(self.episode, directory / 'episode.json')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        bundle = cls(load_record(directory / 'task.json'), load_record(directory / 'episode.json'))
        validate_pair(bundle.task, bundle.episode)
        return bundle


def create_instance(task_id: str = 'box_insertion', *, seed: int = 0,
                    episode_index: int = 0) -> InstanceBundle:
    """Generate one Task/Episode pair; currently supports box_insertion only."""
    if task_id != 'box_insertion':
        raise ValueError('the minimal package currently supports only box_insertion')
    if type(seed) is not int or seed < 0 or type(episode_index) is not int or not 0 <= episode_index < 15:
        raise ValueError('seed must be nonnegative and episode_index must be in [0, 14]')
    spec, contract = spec_for(task_id), contract_for(task_id)
    scenario = sample_scenario(contract['sampling_profile'], seed=seed)
    scenario.metadata['episode_variation_version'] = 1
    robot_id = resolve_task_robot(spec)
    base = _construct(task_id, scenario, robot_id)
    try:
        base.reset()
        task = seal('task_instance', task_id=task_id, task_instance_id=scenario.scenario_id,
                    task_spec_digest=digest(spec.to_dict()), contract_digest=digest(contract),
                    robot_id=robot_id, scenario=scenario.to_dict(),
                    runtime=runtime_identity(base, task_id),
                    provenance={'sampler': VERSION, 'seed': seed})
    finally:
        base.close()
    variation = sample_variation(task_id, scenario.parameters, seed, episode_index)
    resolved = episode_scenario(scenario, variation)
    env = _construct(task_id, resolved, robot_id)
    try:
        env.reset()
        episode = seal('episode_instance', task_id=task_id,
                       task_instance_id=task['task_instance_id'],
                       task_instance_digest=task['content_digest'],
                       episode_id=f"{task['task_instance_id']}_e{episode_index:03d}",
                       initial_state=capture_state(env), round_plan=None,
                       variation=variation, runtime=runtime_identity(env, task_id),
                       provenance={'sampler': VERSION, 'seed': episode_index})
    finally:
        env.close()
    validate_pair(task, episode)
    return InstanceBundle(task, episode)


@contextmanager
def open_instance(bundle: InstanceBundle, *, cameras: bool = False):
    """Replay one explicit Episode with reset/step, closing it on exit."""
    validate_pair(bundle.task, bundle.episode)
    env = ReplayEnv(bundle.task, cameras=cameras)
    try:
        env.bind(bundle.episode)
        yield env
    finally:
        env.close()


def task_score(info: dict, task_id: str = 'box_insertion') -> float:
    """Read the task's current score from step info (development use only)."""
    import math
    key = contract_for(task_id)['evaluation']['score_key']
    value = info.get(key)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'missing or non-finite info.{key}')
    return float(value)
