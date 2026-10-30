"""Standard evaluation entry point for the single-settlement turntable."""
from react_manip.legacy import LegacyTaskAdapter
from .env import TurntableGamble1


class TurntableAdapter(LegacyTaskAdapter):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if info.get('termination_reason') == 'timeout':
            terminated, truncated = False, True
        return obs, reward, terminated, truncated, info


def make_benchmark_env(*, seed=0, robot_id=None, scenario=None, **kwargs):
    raw = TurntableGamble1(seed=0 if seed is None else seed,
                         robot_id=robot_id or 'dual_piperx', scenario=scenario, **kwargs)
    adapted = TurntableAdapter(raw)
    adapted.scenario = raw.scenario
    return adapted
