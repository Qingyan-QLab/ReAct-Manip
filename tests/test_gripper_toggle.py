"""Full-stroke keyboard commands must not trap the physical finger meshes."""
import numpy as np
import pytest

from react_manip.builtin import discover_task_definitions
from react_manip.robots.arm_actions import arm_action


@pytest.mark.parametrize('task_id', ['leveling_board1', 'box_insertion', 'rail_ball_to_cup', 'car_drive2'])
def test_close_reopen_and_reverse_within_limits(task_id, monkeypatch):
    from robosuite.utils.binding_utils import MjSim
    _, factory = next((s, f) for s, f in discover_task_definitions() if s.task_id == task_id)
    env = factory(seed=0, has_offscreen_renderer=False, use_camera_obs=False)
    try:
        env.reset()
        robot = env.env.robots[0]
        model, data = env.sim.model._model, env.sim.data._data
        joints = [model.joint(name).id for gripper in robot.gripper.values() for name in gripper.joints]
        addresses = model.jnt_qposadr[joints]
        bounds = model.jnt_range[joints]
        samples = []
        # Inclined car fingers load MuJoCo's soft stops during arm settling;
        # permit at most 0.1 mm of transient constraint compliance there.
        tolerance = 1e-4 if task_id == 'car_drive2' else 1e-6
        if task_id == 'car_drive2':
            assert env.env._initial_gripper_clearance() >= .04

        def instrument(original):
            def step(sim, *args, **kwargs):
                result = original(sim, *args, **kwargs)
                if sim is env.sim:
                    values = data.qpos[addresses].copy()
                    # Check each 2 ms physics step, not just the final state.
                    assert np.all(values >= bounds[:, 0] - tolerance)
                    assert np.all(values <= bounds[:, 1] + tolerance)
                    samples.append(values)
                return result
            return step

        for name in ('step', 'step2'):
            monkeypatch.setattr(MjSim, name, instrument(getattr(MjSim, name)))

        def command(active, states, ticks):
            action = arm_action(env.action_dim, robot.composite_controller._action_split_indexes,
                                active, np.zeros(6), states)
            for _ in range(ticks):
                env.step(action)

        states = dict.fromkeys(robot.arms, -1.)
        command(robot.arms[0], states, 10)
        for arm in robot.arms:
            # Match Space on the selected arm, leaving other arms open.
            for target in (1., -1., 1., -1.):
                states[arm] = target
                command(arm, states, 10)
                for other in robot.arms:
                    expected = [0., 0.] if states[other] == 1 else [.05, -.05]
                    np.testing.assert_allclose(
                        data.qpos[robot._ref_gripper_joint_pos_indexes[other]], expected, atol=3e-4)
            # Reverse mid-stroke, then verify a full reopening.
            states[arm] = 1.
            command(arm, states, 1)
            states[arm] = -1.
            command(arm, states, 10)
            np.testing.assert_allclose(data.qpos[robot._ref_gripper_joint_pos_indexes[arm]],
                                       [.05, -.05], atol=3e-4)
        assert len(samples) >= 600
    finally:
        env.close()
