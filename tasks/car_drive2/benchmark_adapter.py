"""Physical joystick evaluation with recoverable blocks and true timeout semantics."""
import numpy as np
from react_manip.legacy import LegacyTaskAdapter
from .construction import make_env


class CarDriveAdapter(LegacyTaskAdapter):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        raw = self.env
        if not (np.isfinite(raw._car_xy).all() and np.isfinite(raw._car_yaw)):
            raise ValueError('non-finite vehicle state; evaluation is invalid')
        info['vehicle_xy'] = raw._car_xy.tolist()
        info['events'] = [event for event in info.get('events', []) if event != 'score']
        if info.get('timed_out'):
            terminated, truncated = False, True
            info['failure'] = {'recoverable': False, 'reason': 'timeout'}
        elif info.get('blocked'):
            # A rejected movement is a diagnostic, not a task failure.
            info['failure'] = None
        return obs, reward, terminated, truncated, info


def make_benchmark_env(*, seed=0, robot_id=None, **kwargs):
    for key in ('initialization_noise', 'use_camera_obs'):
        kwargs.pop(key, None)  # Fixed initialization and explicit evaluator RGB.
    raw = make_env(seed=0 if seed is None else seed, robot_id=robot_id or 'dual_piperx', **kwargs)
    raw.task_id = 'car_drive2'
    adapted = CarDriveAdapter(raw)
    adapted.scenario = raw.scenario
    return adapted
