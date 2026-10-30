"""Sampled box task defaults must start clear of the robot."""
import json
from pathlib import Path
import numpy as np
import pytest
from react_manip.core.scenarios import sample_scenario
from react_manip.instances.runtime import _construct

ROOT = Path(__file__).resolve().parents[1]

def _scenario(seed):
    profile = json.loads(
        (ROOT / 'configs/box_insertion/profile_default.json').read_text())
    return sample_scenario(profile, seed=seed)


@pytest.mark.parametrize('seed', range(3))
def test_box_reset_has_gap_and_zero_action_does_not_push(seed):
    env = _construct('box_insertion', _scenario(seed), 'piperx')
    try:
        for _ in range(2):
            env.reset()
            raw = env.env
            assert raw._initial_gripper_clearance() >= .05
            start = raw._block_pose().copy()
            for _ in range(20):
                env.step(np.zeros(env.action_dim))
                assert raw._initial_gripper_clearance() >= .05
            # Gravity may settle the block vertically, but idle fingers must not push it.
            np.testing.assert_allclose(raw._block_pose()[:2], start[:2], atol=1e-5)
    finally:
        env.close()


def test_seed_42_task_default_starts_clear():
    env = _construct('box_insertion', _scenario(42), 'piperx')
    try:
        env.reset()
        assert env.env._initial_gripper_clearance() >= .05
    finally:
        env.close()
