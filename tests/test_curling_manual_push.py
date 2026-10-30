"""A real PiPER-X contact push can reach the seed-zero target at manual speed."""
import numpy as np
from react_manip.builtin import discover_task_definitions
from scripts.pygame_demo import KeyboardPoseControl


def test_single_physical_push_reaches_target():
    _, factory = next((s, f) for s, f in discover_task_definitions() if s.task_id == 'curling_push')
    env = factory(seed=0, has_offscreen_renderer=False, use_camera_obs=False)
    try:
        env.reset()
        raw = env.env
        robot = raw.robots[0]
        arm = robot.arms[0]
        controller = robot.part_controllers[arm]
        layout = robot.composite_controller._action_split_indexes
        manual = KeyboardPoseControl(robot, raw.control_timestep, fast_speed=.65)
        peak_velocity = 0.

        def step(direction, fast=False):
            nonlocal peak_velocity
            action = manual.action(env.action_dim, layout, arm, direction, {arm: 1.}, fast)
            result = env.step(action)
            controller.update(force=True)
            peak_velocity = max(peak_velocity, raw.stone_velocity()[0])
            return result[2] or result[3]

        # All movement is through the same Cartesian controls as Pygame;
        # no writes to stone position, velocity, applied force, or scoring state.
        for goal in ([.32, 0, .98], [.32, 0, .71]):
            for _ in range(700):
                controller.update(force=True)
                delta = np.array(goal) - controller.ref_pos
                if np.linalg.norm(delta) < .003:
                    break
                assert not step(np.r_[np.clip(delta/.01, -1, 1), [0, 0, 0]])
            assert np.linalg.norm(np.array(goal) - controller.ref_pos) < .003
            manual.stop()
        for _ in range(100):
            assert not step([1, 0, 0, 0, 0, 0], fast=True)
            if raw.stone_position()[0] > .78:
                break
        for _ in range(2200):
            if step(np.zeros(6)):
                break
        assert peak_velocity > .5
        assert raw._info()['task_score'] == 100.
        assert raw._state.failure is None
    finally:
        env.close()
