"""Manual input should brake on release and remain independent of task scaling."""
import numpy as np
import pytest
from react_manip.builtin import discover_task_definitions
from scripts.pygame_demo import KeyboardPoseControl


@pytest.mark.parametrize('fast', [False, True])
def test_pitch_release_holds_target_and_brakes(fast):
    _, factory = next((s, f) for s, f in discover_task_definitions() if s.task_id == 'pitch_pot')
    env = factory(seed=0, has_offscreen_renderer=False, use_camera_obs=False)
    try:
        env.reset()
        robot = env.env.robots[0]
        arm = robot.arms[0]
        controller = robot.part_controllers[arm]
        manual = KeyboardPoseControl(robot, env.env.control_timestep)
        layout = robot.composite_controller._action_split_indexes
        grips = dict.fromkeys(robot.arms, -1)
        for _ in range(10):
            action = manual.action(env.action_dim, layout, arm, [0, 1, 0, 0, 0, 0], grips, fast)
            env.step(action)
        controller.update(force=True)
        release = controller.ref_pos.copy()
        if fast:
            assert np.linalg.norm(controller.ref_pos_vel) > .2  # retains throwing-speed capability
        positions = []
        goal = None
        for _ in range(100):
            action = manual.action(env.action_dim, layout, arm, np.zeros(6), grips, fast)
            np.testing.assert_array_equal(action[:6], 0)
            env.step(action)
            controller.update(force=True)
            if goal is None:
                goal = controller.goal_pos.copy()
                np.testing.assert_allclose(goal[1], release[1], atol=1e-10)
            np.testing.assert_array_equal(controller.goal_pos, goal)
            positions.append(controller.ref_pos.copy())
        assert np.max(np.linalg.norm(np.array(positions) - release, axis=1)) < (.04 if fast else .003)
        assert np.linalg.norm(controller.ref_pos_vel) < .003
    finally:
        env.close()


def test_speed_scaling_and_stop_both_arms():
    _, factory = next((s, f) for s, f in discover_task_definitions() if s.task_id == 'leveling_board1')
    env = factory(seed=0, has_offscreen_renderer=False, use_camera_obs=False)
    try:
        env.reset()
        r = env.env.robots[0]
        manual = KeyboardPoseControl(r, env.env.control_timestep)
        grips = dict.fromkeys(r.arms, -1)
        layout = r.composite_controller._action_split_indexes
        for arm in r.arms:
            a = manual.action(env.action_dim, layout, arm, [1, 1, 0, 0, 0, 0], grips)
            lo, hi = layout[arm]
            delta = r.part_controllers[arm].scale_action(a[lo:hi])
            assert np.linalg.norm(delta[:3]) == pytest.approx(.06 * env.env.control_timestep)
            env.step(a)
        manual.stop()
        for arm in r.arms:
            c = r.part_controllers[arm]
            np.testing.assert_allclose(c.goal_pos, c.ref_pos)
            np.testing.assert_allclose(c.goal_ori, c.ref_ori_mat)
            np.testing.assert_array_equal(manual.previous[arm], 0)
    finally:
        env.close()


@pytest.mark.parametrize('toggle', [False, True])
def test_window_focus_loss_clears_held_keys_and_fast_mode(monkeypatch, toggle):
    import sys
    from types import SimpleNamespace as NS
    from scripts import pygame_demo
    from react_manip import rendering
    pg = NS()
    names = ('QUIT KEYDOWN KEYUP WINDOWFOCUSLOST K_ESCAPE K_BACKSPACE K_TAB K_SPACE '
             'K_LEFTBRACKET K_RIGHTBRACKET K_RIGHT K_LEFT K_UP K_DOWN K_PAGEUP K_PAGEDOWN '
             'K_d K_a K_w K_s K_e K_q K_LSHIFT K_RSHIFT K_f K_EQUALS K_KP_PLUS K_MINUS K_KP_MINUS').split()
    for code, name in enumerate(names):
        setattr(pg, name, code)
    pg.K_1, pg.K_9 = 100, 108
    frames = iter([
        [NS(type=pg.KEYDOWN, key=pg.K_UP), NS(type=pg.KEYDOWN, key=pg.K_f if toggle else pg.K_LSHIFT)],
        [NS(type=pg.KEYDOWN, key=pg.K_EQUALS)], [NS(type=pg.WINDOWFOCUSLOST)], [], [NS(type=pg.QUIT)],
    ])
    pg.event = NS(get=lambda: next(frames))
    surface = NS(fill=lambda *a: None, blit=lambda *a: None)
    pg.display = NS(set_mode=lambda *a: surface, set_caption=lambda *a: None, flip=lambda: None)
    pg.font = NS(Font=lambda *a: NS(render=lambda *a: None, size=lambda s: (len(s)*8, 20)))
    pg.time = NS(Clock=lambda: NS(tick=lambda *a: None))
    pg.surfarray = NS(make_surface=lambda *a: surface)
    pg.init = pg.quit = lambda: None
    monkeypatch.setitem(sys.modules, 'pygame', pg)
    monkeypatch.setattr(rendering, 'NativeRGBRenderer', lambda: NS(
        render=lambda *a: np.zeros((10, 10, 3), dtype=np.uint8), close=lambda: None))
    calls = []
    original = KeyboardPoseControl.action
    def action(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append((result.copy(), kwargs['fast'], self.robot.part_controllers['right'].goal_pos.copy()))
        return result
    monkeypatch.setattr(KeyboardPoseControl, 'action', action)
    assert pygame_demo.main(['--task', 'pitch_pot', '--seed', '0']) == 0
    assert len(calls) == 4
    assert calls[0][0][1] == pytest.approx(.02)
    assert calls[1][0][1] == pytest.approx(.021)
    assert [x[1] for x in calls] == [True, True, False, False]
    for result, _, _ in calls[2:]:
        np.testing.assert_array_equal(result[:6], 0)
    np.testing.assert_array_equal(calls[2][2], calls[3][2])
