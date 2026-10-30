"""Fixed HarnessVLA primitives implemented against public PiPER-X observations."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class Primitives:
    def __init__(self, env, vla=None):
        self.env, self.vla = env, vla
        self.grippers = {arm: -1.0 for arm in self.contract['arms']}

    @property
    def contract(self):
        return self.env.current.observation['control_contract']

    def _step(self, action):
        result = self.env.step(action)
        observe = getattr(self.vla, 'observe', None)
        if observe is not None:
            observe(self.env.current.observation)
        return result

    def pose(self, arm):
        obs = self.env.current.observation
        if arm not in self.contract['arms']:
            raise ValueError(f'Unavailable arm: {arm}')
        if 'eef_pos' in obs and len(self.contract['arms']) == 1:
            return np.asarray(obs['eef_pos']), np.asarray(obs['eef_mat']).reshape(3, 3)
        prefix = 'robot0_' if len(self.contract['arms']) == 1 else f'robot0_{arm}_'
        return np.asarray(obs[prefix + 'eef_pos']), Rotation.from_quat(obs[prefix + 'eef_quat_site']).as_matrix()

    def action(self, arm, delta):
        config = self.contract['arms'][arm]
        low, high = np.asarray(config['output_min']), np.asarray(config['output_max'])
        if np.any(high <= low):
            raise ValueError('Invalid OSC output range')
        delta = np.asarray(delta, dtype=float)
        rotation = np.asarray(config.get('world_from_control_rotation', np.eye(3)))
        delta = np.r_[rotation.T @ delta[:3], rotation.T @ delta[3:]]
        delta = np.clip(delta, low, high)
        normalized = ((delta - low) / (high - low) *
                      (np.asarray(config['input_max']) - np.asarray(config['input_min'])) +
                      np.asarray(config['input_min']))
        action = np.zeros(self.env.action_dim)
        start, end = self.contract['action_layout'][arm]
        action[start:end] = normalized
        for name, value in self.grippers.items():
            start, end = self.contract['action_layout'][name + '_gripper']
            action[start:end] = value
        return action

    def move_pose(self, xyz, quaternion_xyzw=None, *, arm='right', max_steps=100, tolerance_m=.01,
                  tolerance_rad=.05):
        target = np.asarray(xyz, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('xyz must contain three finite world coordinates')
        if not 1 <= max_steps <= 1000 or not (0 < tolerance_m <= .1 and 0 < tolerance_rad <= .5):
            raise ValueError('Invalid motion budget or tolerance')
        _, initial_rotation = self.pose(arm)
        desired = initial_rotation if quaternion_xyzw is None else Rotation.from_quat(quaternion_xyzw).as_matrix()
        for step in range(max_steps):
            position, rotation = self.pose(arm)
            error = Rotation.from_matrix(desired @ rotation.T).as_rotvec()
            if np.linalg.norm(target - position) <= tolerance_m and np.linalg.norm(error) <= tolerance_rad:
                return dict(reached=True, executed_steps=step)
            _, _, term, trunc, _ = self._step(self.action(arm, np.r_[target - position, error]))
            if term or trunc:
                return dict(reached=False, episode_ended=True, executed_steps=step + 1)
        return dict(reached=False, executed_steps=max_steps)

    def move_to(self, xyz, *, arm='right', max_steps=100):
        return self.move_pose(xyz, arm=arm, max_steps=max_steps)

    def rotate(self, radians, axis, *, arm='right', max_steps=100):
        if not np.isfinite(radians) or abs(radians) > np.pi:
            raise ValueError('Rotation must be finite and within [-pi, pi]')
        position, rotation = self.pose(arm)
        # Local tool-axis rotation, explicitly declared rather than inheriting
        # LIBERO's yaw/pitch conventions or assuming identical home frames.
        desired = Rotation.from_matrix(rotation) * Rotation.from_rotvec(np.eye(3)[axis] * radians)
        return self.move_pose(position, desired.as_quat(), arm=arm, max_steps=max_steps)

    def set_gripper(self, closed, *, arm='right', steps=10):
        if arm not in self.grippers or not 1 <= steps <= 200:
            raise ValueError('Invalid arm or gripper step budget')
        self.grippers[arm] = 1.0 if closed else -1.0
        for step in range(steps):
            _, _, term, trunc, _ = self._step(self.action(arm, np.zeros(6)))
            if term or trunc:
                return dict(executed_steps=step + 1, episode_ended=True)
        return dict(executed_steps=steps)

    def vla_act(self, prompt, *, max_chunks=4):
        if self.vla is None:
            raise RuntimeError('No calibrated frozen VLA backend is configured')
        if not prompt.strip() or not 1 <= max_chunks <= 100:
            raise ValueError('Invalid VLA prompt or chunk budget')
        count = 0
        for _ in range(max_chunks):
            actions = np.asarray(self.vla.predict(self.env.current.observation, prompt), dtype=float)
            if (actions.ndim != 2 or actions.shape[1] != self.env.action_dim or len(actions) == 0
                    or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
                raise ValueError('VLA adapter must return finite normalized [steps, action_dim] actions')
            for action in actions:
                _, _, term, trunc, _ = self._step(action)
                count += 1
                for arm in self.grippers:
                    start, _ = self.contract['action_layout'][arm + '_gripper']
                    self.grippers[arm] = float(action[start])
                if term or trunc:
                    return dict(executed_steps=count, episode_ended=True)
        return dict(executed_steps=count, episode_ended=False)
