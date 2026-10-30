"""Bridge for one resolved phone Episode, with immutable geometry during reset/step."""
from copy import deepcopy

from react_manip.core.scenarios import Scenario
from react_manip.legacy import LegacyTaskAdapter, attach_robot_context
from .instances import sample_task_instance, sample_episode_instance
from .env import make_env


class PhoneAdapter(LegacyTaskAdapter):
    def __init__(self, scenario, robot_id, episode_instance=None, **kwargs):
        self.scenario = scenario
        self._robot = robot_id
        self._construction = kwargs
        task = scenario.parameters
        self._explicit_episode = deepcopy(episode_instance)
        episode = episode_instance or sample_episode_instance(task, 0)
        raw = make_env(robot_id=robot_id, task_instance=task,
                       episode_instance=episode, **kwargs)
        attach_robot_context(raw, robot_id)
        super().__init__(raw)
        self.scenario = scenario

    def reset(self, *, seed=None):
        if seed is not None:
            if self._explicit_episode is not None:
                raise ValueError('explicit phone Episode forbids seed overrides')
            episode = sample_episode_instance(self.scenario.parameters, seed)
            raw = make_env(robot_id=self._robot, task_instance=self.scenario.parameters,
                           episode_instance=episode, **self._construction)
            attach_robot_context(raw, self._robot)
            self.close()
            self.env = raw
            return super().reset()
        if self._explicit_episode is not None:
            return super().reset()
        return super().reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        private = {'layout_positions', 'layout_seed', 'task_seed', 'episode_seed',
                   'task_instance_id', 'episode_instance_id'}
        return obs, reward, terminated, truncated, {k:v for k,v in info.items() if k not in private}


def make_benchmark_env(*, seed=0, robot_id=None, scenario=None, episode_instance=None, **kwargs):
    if scenario is None:
        task = sample_task_instance(0 if seed is None else seed)
        scenario = Scenario('1.0', 'phone_lean', task['task_instance_id'], task['task_seed'], task)
    kwargs.setdefault('has_offscreen_renderer', False)
    return PhoneAdapter(scenario, robot_id or 'dual_piperx', episode_instance, **kwargs)
