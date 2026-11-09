"""HarnessVLA additions to the shared evaluator-owned RGB-D contract."""
from __future__ import annotations

import numpy as np

from . import OBSERVATION_VERSION
from react_manip.rgbd import RGBDSensor, back_project


def controller_contract(raw):
    """Publish robot control calibration only, without task/object parameters."""
    robot = raw.robots[0]
    controller = robot.composite_controller
    layout = {k: list(v) for k, v in controller._action_split_indexes.items()}
    arms = {}
    for name, part in controller.part_controllers.items():
        if name.endswith('_gripper'):
            continue
        if name not in ('right', 'left') or part.control_dim != 6:
            raise ValueError('Unsupported PiPER-X controller layout')
        if part.input_type != 'delta' or part.input_ref_frame not in ('world', 'base'):
            raise ValueError('Integration requires world/base-frame delta OSC')
        arms[name] = {key: np.asarray(getattr(part, key)).tolist() for key in
                      ('input_min', 'input_max', 'output_min', 'output_max')}
        rotation = np.eye(3) if part.input_ref_frame == 'world' else np.asarray(part.origin_ori)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError('Controller reference-frame calibration is unavailable')
        arms[name].update(input_type=part.input_type, input_ref_frame=part.input_ref_frame,
                          world_from_control_rotation=rotation.tolist())
    return dict(schema_version='react-manip.osc.v1', action_layout=layout, arms=arms,
                gripper_open=-1.0, gripper_close=1.0)


class RGBDReplayEnv:
    """Add HarnessVLA control metadata to the shared RGB-D replay environment."""

    def __init__(self, instance, *, cameras=True):
        if not cameras:
            raise ValueError('RGB-D evaluation requires camera observations')
        from react_manip.instances.runtime import ReplayEnv
        self._replay = ReplayEnv(instance, cameras=True)
        self._camera_ids = tuple(instance['runtime']['cameras'])

    @property
    def env(self):
        # Used exclusively by benchmark-side observation visibility filtering.
        return self._replay.env

    def __getattr__(self, name):
        return getattr(self._replay, name)

    def bind(self, episode):
        self._replay.bind(episode)

    def set_camera_observations(self, camera_ids=(), *, width=640, height=480):
        if tuple(camera_ids) != self._camera_ids or (width, height) != (640, 480):
            raise ValueError('RGB-D v1 uses the recorded policy cameras at 640x480')
        self._replay.set_camera_observations(camera_ids, width=width, height=height)

    def _enrich(self, observation):
        required = ('images', 'depth_m', 'depth_valid', 'camera_calibration')
        if any(set(observation.get(key, ())) != set(self._camera_ids) for key in required):
            raise ValueError('Replay environment did not provide the recorded RGB-D cameras')
        return dict(observation, control_contract=controller_contract(self.env),
                    observation_schema=OBSERVATION_VERSION)

    def reset(self, **kwargs):
        observation, info = self._replay.reset(**kwargs)
        return self._enrich(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._replay.step(action)
        return self._enrich(observation), reward, terminated, truncated, info

    def close(self):
        self._replay.close()
