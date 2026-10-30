from types import SimpleNamespace
import numpy as np
import pytest
from integrations.harness_vla.primitives import Primitives


def environment(dual=True):
    arms = ['right', 'left'] if dual else ['right']
    layout = {'right': [0, 6], 'right_gripper': [12, 13] if dual else [6, 7]}
    if dual:
        layout.update(left=[6, 12], left_gripper=[13, 14])
    contract = dict(arms={arm: dict(output_min=[-.015]*3+[-.08]*3,
                                    output_max=[.015]*3+[.08]*3,
                                    input_min=[-1]*6, input_max=[1]*6) for arm in arms}, action_layout=layout)
    obs = dict(control_contract=contract)
    for arm in arms:
        prefix = f'robot0_{arm}_' if dual else 'robot0_'
        obs[prefix+'eef_pos'] = np.zeros(3)
        obs[prefix+'eef_quat_site'] = [0, 0, 0, 1]
    env = SimpleNamespace(current=SimpleNamespace(observation=obs), action_dim=14 if dual else 7, actions=[])
    def step(action):
        env.actions.append(action.copy())
        return obs, None, False, False, {}
    env.step = step
    return env


def test_dual_arm_layout_and_controller_scaling():
    env = environment()
    primitives = Primitives(env)
    primitives.grippers.update(right=1, left=-1)
    action = primitives.action('left', [.0075, 0, 0, 0, 0, .04])
    assert np.allclose(action[:6], 0)
    assert action[6] == pytest.approx(.5)
    assert action[11] == pytest.approx(.5)
    assert list(action[12:]) == [1, -1]


def test_pose_primitive_obeys_budget_without_teleportation():
    env = environment(False)
    result = Primitives(env).move_to([.1, 0, 0], max_steps=3)
    assert result == dict(reached=False, executed_steps=3)
    assert len(env.actions) == 3
    assert all(action[0] == 1 for action in env.actions)
    assert np.allclose(env.current.observation['robot0_eef_pos'], 0)


def test_box_pose_schema_and_terminal_gripper():
    env = environment(False)
    env.current.observation.update(eef_pos=np.array([.1, .2, .3]), eef_mat=np.eye(3))
    env.step = lambda action: ({}, 100, True, False, {})
    primitive = Primitives(env)
    assert np.allclose(primitive.pose('right')[0], [.1, .2, .3])
    assert primitive.set_gripper(False, steps=10)['executed_steps'] == 1


def test_vla_uses_live_observation_and_preserves_gripper_command():
    env = environment(False)
    calls = []
    class VLA:
        def predict(self, observation, prompt):
            calls.append((observation, prompt))
            return [[0, 0, 0, 0, 0, 0, 1]]
    primitive = Primitives(env, VLA())
    assert primitive.vla_act('grasp', max_chunks=2)['executed_steps'] == 2
    assert len(calls) == 2 and calls[0][0] is env.current.observation
    assert primitive.action('right', np.zeros(6))[-1] == 1


def test_missing_vla_fails_instead_of_zero_policy():
    with pytest.raises(RuntimeError, match='No calibrated frozen VLA'):
        Primitives(environment()).vla_act('pick')


def test_invalid_vla_chunk_rejected_before_execution():
    env = environment()
    vla = SimpleNamespace(predict=lambda *_: [[2]*14])
    with pytest.raises(ValueError, match='finite normalized'):
        Primitives(env, vla).vla_act('pick')
    assert not env.actions


def test_world_error_is_rotated_into_moving_base_frame():
    env = environment(False)
    config = env.current.observation['control_contract']['arms']['right']
    config['input_ref_frame'] = 'base'
    config['world_from_control_rotation'] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    action = Primitives(env).action('right', [.0075, 0, 0, .04, 0, 0])
    assert np.allclose(action[:6], [0, -.5, 0, 0, -.5, 0])
