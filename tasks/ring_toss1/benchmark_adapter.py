"""Standard evaluation entry point for the single-ring toss."""
from react_manip.legacy import LegacyTaskAdapter
from .env import RingToss1


class RingTossAdapter(LegacyTaskAdapter):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # LegacyTaskAdapter synthesizes a 'score' event every positive-score
        # tick. This task latches points until timeout; keep only actual awards.
        info['events'] = [event for event in info.get('events', []) if event != 'score']
        if info.get('termination_reason') == 'timeout':
            terminated, truncated = False, True
        return obs, reward, terminated, truncated, info


def make_benchmark_env(*, seed=0, robot_id=None, scenario=None, **kwargs):
    raw = RingToss1(seed=0 if seed is None else seed,
                         robot_id=robot_id or 'dual_piperx', scenario=scenario, **kwargs)
    adapted = RingTossAdapter(raw)
    adapted.scenario = raw.scenario
    return adapted
