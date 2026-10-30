"""Vehicle-mounted arms must retain one reference frame at rest and after yaw."""
import numpy as np
import pytest
from react_manip.builtin import discover_task_definitions
from scripts.pygame_demo import KeyboardPoseControl


@pytest.mark.parametrize('manual_mode', [False, True])
def test_car_idle_release_and_moving_mount(manual_mode):
    _, factory = next((s, f) for s, f in discover_task_definitions() if s.task_id == 'car_drive2')
    env = factory(seed=0, has_offscreen_renderer=False)
    try:
        for reset_index in range(2):
            env.reset()
            raw = env.env
            robot = raw.robots[0]
            layout = robot.composite_controller._action_split_indexes
            manual = KeyboardPoseControl(robot, raw.control_timestep) if manual_mode else None
            grips = dict.fromkeys(robot.arms, -1)

            def pose(arm):
                c = robot.part_controllers[arm]
                c.update(force=True)
                return c.world_to_origin_frame(c.ref_pos).copy()

            def step(direction=np.zeros(6)):
                if manual:
                    action = manual.action(env.action_dim, layout, robot.arms[0], direction, grips)
                else:
                    from react_manip.robots.arm_actions import arm_action
                    action = arm_action(env.action_dim, layout, robot.arms[0], direction, grips)
                env.step(action)

            initial = {a: pose(a) for a in robot.arms}
            for _ in range(200):
                step()
                for arm in robot.arms:
                    c = robot.part_controllers[arm]
                    origin, orientation = robot.composite_controller.get_controller_base_pose(arm)
                    np.testing.assert_allclose(c.origin_pos, origin, atol=1e-10)
                    np.testing.assert_allclose(c.origin_ori, orientation, atol=1e-10)
                    assert np.linalg.norm(pose(arm) - initial[arm]) < .001
            np.testing.assert_allclose(raw._car_xy, [0, raw._start_y], atol=1e-10)

            # Change only the mount pose as a frame-covariance fixture: no
            # robot joint state or controller gains are edited by the test.
            raw._car_xy += [.1, .2]
            raw._car_yaw += .35
            raw._set_car_pose()
            for _ in range(80):
                step()
                for arm in robot.arms:
                    assert np.linalg.norm(pose(arm) - initial[arm]) < .001

            if manual:
                for _ in range(8):
                    step(np.array([1., 0, 0, 0, 0, 0]))
                released = pose(robot.arms[0])
                assert np.linalg.norm(released - initial[robot.arms[0]]) > .001
                for _ in range(100):
                    step()
                    assert np.linalg.norm(pose(robot.arms[0]) - released) < .005
                assert np.linalg.norm(robot.part_controllers[robot.arms[0]].ref_pos_vel) < .003
    finally:
        env.close()
