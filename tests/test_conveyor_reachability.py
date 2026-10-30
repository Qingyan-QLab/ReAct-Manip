"""Sampled conveyor defaults must start clear and remain reachable."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from react_manip.core.scenarios import sample_scenario
from react_manip.instances.runtime import _construct
from tasks.conveyor_grasp.env import DUAL_PIPER_BASE_Y, DUAL_PIPER_READY_QPOS


ROOT = Path(__file__).resolve().parents[1]
PIPER_BASE_ORIENTATION = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])


def _scenario(seed):
    profile = json.loads(
        (ROOT / 'configs/conveyor_grasp/profile_default.json').read_text())
    return sample_scenario(profile, seed=seed)


def _robot_penetrations(raw):
    model, data = raw.sim.model, raw.sim.data
    rows = []
    for contact in data.contact[:data.ncon]:
        if contact.dist >= 0:
            continue
        names = (model.geom_id2name(contact.geom1) or '',
                 model.geom_id2name(contact.geom2) or '')
        if any(name.startswith(('robot', 'gripper', 'fixed_mount'))
               for name in names):
            rows.append((float(contact.dist), *names))
    return rows


@pytest.mark.parametrize('seed', range(3))
def test_conveyor_sampled_scenarios_start_clear_and_stable(seed):
    env = _construct('conveyor_grasp', _scenario(seed), 'dual_piperx')
    try:
        env.reset()
        raw = env.env
        robot = raw.robots[0]
        root = raw.sim.model.body_name2id(robot.robot_model.root_body)
        np.testing.assert_allclose(raw.sim.data.body_xpos[root],
                                   [2.0, DUAL_PIPER_BASE_Y, 0.0], atol=1e-10)
        np.testing.assert_allclose(
            raw.sim.data.qpos[robot._ref_joint_pos_indexes],
            DUAL_PIPER_READY_QPOS, atol=1e-10,
        )
        assert not _robot_penetrations(raw)
        for _ in range(20):
            env.step(np.zeros(env.action_dim))
        assert np.isfinite(raw.sim.data.qpos).all()
        assert np.isfinite(raw.sim.data.qvel).all()
    finally:
        env.close()


def _orientation(yaw_degrees):
    return (Rotation.from_euler('z', yaw_degrees, degrees=True).as_matrix()
            @ Rotation.from_euler('x', 90.0, degrees=True).as_matrix()
            @ PIPER_BASE_ORIENTATION)


def _reach(env, arm, goal, yaw_degrees):
    raw = env.env
    robot = raw.robots[0]
    layout = robot.composite_controller._action_split_indexes
    site = robot.eef_site_id[arm]
    target = _orientation(yaw_degrees)
    other = 'left' if arm == 'right' else 'right'
    best = np.inf
    for _ in range(240):
        position = raw.sim.data.site_xpos[site]
        current = raw.sim.data.site_xmat[site].reshape(3, 3)
        distance = float(np.linalg.norm(position - goal))
        angle = Rotation.from_matrix(target @ current.T).magnitude()
        best = min(best, distance)
        if distance < .04 and angle < np.deg2rad(10):
            return best
        action = np.zeros(env.action_dim)
        start = layout[arm][0]
        action[start:start + 3] = np.clip((goal - position) / .025, -1, 1)
        action[start + 3:start + 6] = np.clip(
            Rotation.from_matrix(target @ current.T).as_rotvec() / .12, -1, 1)
        action[layout[arm + '_gripper'][0]] = -1.0
        action[layout[other + '_gripper'][0]] = -1.0
        env.step(action)
    pytest.fail(f'{arm} arm stopped {best:.3f} m from reachability witness')


def test_conveyor_public_actions_reach_belt_and_all_four_baskets():
    env = _construct('conveyor_grasp', _scenario(2), 'dual_piperx')
    checks = (
        ('right', (1.55, .13, .98), -90),
        ('left', (2.45, .13, .98), 45),
        ('right', (1.43, .40, 1.10), -135),
        ('left', (2.57, .40, 1.10), 90),
        ('right', (1.55, .72, 1.10), -90),
        ('left', (2.45, .72, 1.10), 90),
    )
    try:
        for arm, goal, yaw in checks:
            env.reset()
            assert _reach(env, arm, np.asarray(goal, dtype=float), yaw) < .04
    finally:
        env.close()
