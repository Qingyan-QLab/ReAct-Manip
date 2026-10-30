#!/usr/bin/env python3
"""Run an unscored task-default scene with keyboard pose control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



class KeyboardPoseControl:
    """Manual-only physical speed scaling and persistent stop targets."""

    def __init__(self, robot, dt, fast_speed=1.0):
        self.robot, self.dt = robot, float(dt)
        self.fast_speed = float(fast_speed)
        if not np.isfinite(self.fast_speed) or not 0 < self.fast_speed <= 2:
            raise ValueError("fast speed must be in (0, 2] m/s")
        self.previous = {arm: np.zeros(6) for arm in robot.arms}
        for arm in robot.arms:
            controller = robot.part_controllers[arm]
            if (controller.input_type != 'delta'
                    or controller.impedance_mode != 'fixed'):
                raise ValueError(
                    'Keyboard control requires fixed-impedance delta OSC_POSE')
            controller.update(force=True)
            controller.reset_goal(goal_update_mode='desired')
            controller.goal_pos, controller.goal_ori = self.current_pose(controller)

    @staticmethod
    def current_pose(controller):
        if controller.input_ref_frame == 'world':
            return controller.ref_pos.copy(), controller.ref_ori_mat.copy()
        return (controller.world_to_origin_frame(controller.ref_pos),
                controller.goal_origin_to_eef_pose()[:3, :3])

    def stop(self):
        for arm in self.robot.arms:
            controller = self.robot.part_controllers[arm]
            controller.update(force=True)
            controller.goal_pos, controller.goal_ori = self.current_pose(controller)
            self.previous[arm].fill(0)

    def action(self, action_dim, layout, active, direction, grippers, fast=False):
        from scipy.spatial.transform import Rotation
        from react_manip.robots.arm_actions import arm_action

        direction = np.asarray(direction, dtype=float).copy()
        for part in (slice(0, 3), slice(3, 6)):
            direction[part] /= max(1., np.linalg.norm(direction[part]))
        controller = self.robot.part_controllers[active]
        controller.update(force=True)
        position, orientation = self.current_pose(controller)
        previous = self.previous[active]
        released = (direction[:3] == 0) & (previous[:3] != 0)
        controller.goal_pos[released] = position[released]
        if not np.any(direction[3:]) and np.any(previous[3:]):
            controller.goal_ori = orientation.copy()
        moving = direction[:3] != 0
        limit = .10 if fast else .02
        lead = controller.goal_pos[moving] - position[moving]
        controller.goal_pos[moving] = position[moving] + np.clip(
            lead, -limit, limit)
        if np.any(direction[3:]):
            rotation = Rotation.from_matrix(
                controller.goal_ori @ orientation.T).as_rotvec()
            rotation *= min(1., .3 / max(np.linalg.norm(rotation), 1e-12))
            controller.goal_ori = (
                Rotation.from_rotvec(rotation).as_matrix() @ orientation)
        speeds = [self.fast_speed if fast else .06] * 3
        speeds += [1.2 if fast else .3] * 3
        delta = direction * np.asarray(speeds) * self.dt
        output_mid = (controller.output_max + controller.output_min) / 2
        output_half = (controller.output_max - controller.output_min) / 2
        pose = (delta - output_mid) / output_half
        pose = (pose * (controller.input_max - controller.input_min) / 2
                + (controller.input_max + controller.input_min) / 2)
        self.previous[active] = direction.copy()
        return arm_action(action_dim, layout, active, pose, grippers)


def _parser():
    from react_manip.builtin import discover_task_definitions

    available_tasks = tuple(
        spec.task_id for spec, _ in discover_task_definitions())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=available_tasks,
                        default=('box_insertion' if 'box_insertion' in available_tasks
                                 else available_tasks[0]))
    parser.add_argument('--robot', default=None,
                        help='robot override; defaults to the task-configured embodiment')
    parser.add_argument('--seed', type=int, default=0,
                        help='nonnegative task-default construction seed')
    parser.add_argument('--scenario', type=Path,
                        help='resolved Scenario JSON for the selected task')
    parser.add_argument('--hz', type=float, default=20,
                        help='display pacing; task physics timing stays unchanged')
    parser.add_argument('--camera', help='initial camera name')
    parser.add_argument('--fast-speed', type=float, default=1.0,
                        help='fast target speed in m/s, (0, 2]; default 1.0')
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error('seed must be nonnegative')
    if args.hz <= 0 or not np.isfinite(args.hz):
        parser.error('hz must be positive')
    if not np.isfinite(args.fast_speed) or not 0 < args.fast_speed <= 2:
        parser.error('fast-speed must be in (0, 2] m/s')

    try:
        import pygame
    except ImportError:
        parser.error('install dependencies: python -m pip install -r requirements.txt')

    from react_manip.builtin import register_builtin_tasks
    from react_manip.cameras import display_camera_names
    from react_manip.core.compatibility import resolve_task_robot
    from react_manip.core.registry import get_task_registry
    from react_manip.core.scenarios import Scenario, sample_scenario
    from react_manip.rendering import NativeRGBRenderer
    from react_manip.robot_registry import register_default_robots

    register_default_robots()
    register_builtin_tasks()
    definition = get_task_registry().get(args.task)
    try:
        args.robot = resolve_task_robot(definition.spec, args.robot)
    except ValueError as exc:
        parser.error(str(exc))

    if args.scenario:
        scenario = Scenario.load(args.scenario)
    elif args.task == 'button_indicator_matching':
        profile = ROOT / 'configs' / args.task / 'profile_default.json'
        scenario = sample_scenario(json.loads(profile.read_text()), seed=args.seed)
    else:
        scenario = None
    if scenario is not None and scenario.task_id != args.task:
        parser.error('scenario task does not match --task')

    seed = scenario.seed if scenario is not None else args.seed
    env = renderer = None
    try:
        kwargs = dict(seed=seed, robot_id=args.robot,
                      has_offscreen_renderer=False)
        if scenario is not None:
            kwargs['scenario'] = scenario
        env = definition.factory(**kwargs)

        pygame.init()
        screen = pygame.display.set_mode((1120, 720))
        pygame.display.set_caption('ReAct-Manip task debug: ' + args.task)
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()
        renderer = NativeRGBRenderer()
        keys = set()
        active = 0
        camera_index = 0
        fast_latched = False
        raw = arms = layout = grippers = manual = cameras = None
        done, info = False, {}

        def reset():
            nonlocal raw, arms, layout, grippers, manual, done, info
            nonlocal cameras, active, camera_index, fast_latched
            reset_result = env.reset()
            info = (reset_result[1] if isinstance(reset_result, tuple)
                    and len(reset_result) == 2 else {})
            raw = getattr(env, 'env', env)
            robot = raw.robots[0]
            arms = tuple(robot.arms)
            layout = robot.composite_controller._action_split_indexes
            grippers = {arm: -1.0 for arm in arms}
            manual = KeyboardPoseControl(robot, raw.control_timestep,
                                         args.fast_speed)
            fast_latched = False
            done = False
            active = 0
            keys.clear()
            cameras = list(display_camera_names(raw, include_legacy=True))
            if not cameras:
                cameras = ['__free__']
            camera_index = (cameras.index(args.camera)
                            if args.camera in cameras else 0)

        reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_BACKSPACE:
                        reset()
                    elif event.key == pygame.K_TAB:
                        manual.stop()
                        fast_latched = False
                        active = (active + 1) % len(arms)
                        keys.clear()
                    elif event.key == pygame.K_f:
                        fast_latched = not fast_latched
                    elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS,
                                       pygame.K_MINUS, pygame.K_KP_MINUS):
                        sign = (1 if event.key in
                                (pygame.K_EQUALS, pygame.K_KP_PLUS) else -1)
                        manual.fast_speed = float(np.clip(
                            round(manual.fast_speed + sign * .05, 2), .05, 2.))
                    elif event.key == pygame.K_SPACE:
                        arm = arms[active]
                        grippers[arm] = -grippers[arm]
                    elif pygame.K_1 <= event.key <= pygame.K_9:
                        index = event.key - pygame.K_1
                        if index < len(cameras):
                            camera_index = index
                    elif event.key in (pygame.K_LEFTBRACKET,
                                       pygame.K_RIGHTBRACKET):
                        step = 1 if event.key == pygame.K_RIGHTBRACKET else -1
                        camera_index = (camera_index + step) % len(cameras)
                    else:
                        keys.add(event.key)
                elif event.type == pygame.WINDOWFOCUSLOST:
                    fast_latched = False
                    keys.clear()
                    manual.stop()
                elif event.type == pygame.KEYUP:
                    keys.discard(event.key)
            if not running:
                break

            def axis(positive, negative):
                return float((positive in keys) - (negative in keys))

            target = np.array([
                axis(pygame.K_RIGHT, pygame.K_LEFT),
                axis(pygame.K_UP, pygame.K_DOWN),
                axis(pygame.K_PAGEUP, pygame.K_PAGEDOWN),
                axis(pygame.K_d, pygame.K_a),
                axis(pygame.K_w, pygame.K_s),
                axis(pygame.K_e, pygame.K_q),
            ])
            fast = fast_latched or bool(
                keys & {pygame.K_LSHIFT, pygame.K_RSHIFT})
            if not done:
                action = manual.action(env.action_dim, layout, arms[active],
                                       target, grippers, fast=fast)
                _, _, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

            frame = renderer.render(raw.sim, cameras[camera_index])
            screen.fill((20, 24, 30))
            screen.blit(pygame.surfarray.make_surface(
                frame.transpose(1, 0, 2)), (0, 0))
            score = next((info[key] for key in
                          ('task_score', 'total_score', 'score', 'S')
                          if key in info), '--')
            lines = [
                f'Task debug: {args.task}',
                f'Robot: {args.robot}  Seed: {seed}  Arm: {arms[active]}',
                f'Camera: {cameras[camera_index]}', f'Score: {score}',
                f"Mode: {'FAST' if fast else 'FINE'}  "
                f"Target: {manual.fast_speed if fast else .06:.2f} m/s",
                'Arrows / PgUp PgDn: translation',
                'A D / W S / Q E: rotation',
                'F: toggle fast; Shift: hold fast',
                f'+/-: fast target {manual.fast_speed:.2f} m/s',
                'Space: gripper    Tab: switch arm',
                '1-9 / [ ]: cameras',
                'Backspace: reset    Esc: quit',
                ('Episode ended. Backspace to reset.'
                 if done else 'Unscored task-default debug'),
            ]
            for index, line in enumerate(lines):
                screen.blit(font.render(line, True, (235, 235, 235)),
                            (650, 20 + index * 32))
            prompt = str(info.get('language_prompt', ''))
            line, rows = '', []
            for word in prompt.split():
                candidate = (line + ' ' + word).strip()
                if font.size(candidate)[0] > 1080 and line:
                    rows.append(line)
                    line = word
                else:
                    line = candidate
            if line:
                rows.append(line)
            for index, line in enumerate(rows[:6]):
                screen.blit(font.render(line, True, (235, 235, 235)),
                            (20, 510 + index * 28))
            pygame.display.flip()
            clock.tick(args.hz)
        return 0
    finally:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()
        pygame.quit()


if __name__ == '__main__':
    raise SystemExit(main())
