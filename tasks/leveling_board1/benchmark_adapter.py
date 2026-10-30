"""Construct one resolved Episode; the evaluator resolves geometry before this bridge."""
from copy import deepcopy
from react_manip.core.scenarios import Scenario
from react_manip.legacy import LegacyTaskAdapter, attach_robot_context
from .instances import sample_task_instance, sample_episode_instance
from .env import make_env


class LevelingAdapter(LegacyTaskAdapter):
    def __init__(self, scenario, robot_id='dual_piperx', episode_instance=None, **kwargs):
        self._robot = robot_id
        self._construction = kwargs
        self._explicit_episode = deepcopy(episode_instance)
        raw = make_env(robot_id=robot_id, task_instance=scenario.parameters,
                       episode_instance=episode_instance or sample_episode_instance(scenario.parameters,0), **kwargs)
        attach_robot_context(raw, robot_id)
        super().__init__(raw)
        self.scenario = scenario

    def reset(self, *, seed=None):
        if seed is not None:
            if self._explicit_episode is not None:
                raise ValueError('explicit leveling Episode forbids seed overrides')
            self.env.episode_instance = sample_episode_instance(self.scenario.parameters,seed)
        return super().reset()


def make_benchmark_env(*, seed=0, robot_id=None, scenario=None, episode_instance=None, **kwargs):
    if scenario is None:
        seed = 0 if seed is None else seed
        scenario = Scenario('1.0','leveling_board1',f'leveling_board1_{seed}',seed,sample_task_instance(seed))
    kwargs.setdefault('has_offscreen_renderer',False)
    return LevelingAdapter(scenario,robot_id or 'dual_piperx',episode_instance,**kwargs)
