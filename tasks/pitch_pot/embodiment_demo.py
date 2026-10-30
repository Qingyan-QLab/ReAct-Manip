"""Calibrated physical reference episodes; no launch injection or moving targets."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from react_manip.core.scenarios import sample_scenario
from tasks.pitch_pot.runtime import make_env
from tasks.pitch_pot.operation import PickThrowController, ThrowMotion

ROOT = Path(__file__).resolve().parent


def make_reference_env(robot_id, *, seed=0):
    config = json.loads((ROOT / 'embodiments.json').read_text())[robot_id]
    profile = json.loads((ROOT.parents[1] / 'configs/pitch_pot/profile_near_calibration.json').read_text())
    profile['profile_id'] = f'pitch_pot_{robot_id}_reference'
    # Merge nested task parameters without changing the user's default Profile.
    for group, values in config['parameters'].items():
        profile['fixed'].setdefault(group, {}).update(deepcopy(values))
    scenario = sample_scenario(profile, seed=seed)
    spec, env = make_env(robot_id=robot_id, scenario=scenario, hard_reset=False,
                         initialization_noise=None)
    return spec, env, ThrowMotion(**config['motion'])


def main(robot_id, argv=None):
    parser = argparse.ArgumentParser(description=f'{robot_id}: calibrated physical pick-and-throw demo')
    parser.add_argument('--seed', type=int, default=0, help='Appearance seed; calibrated physics stays fixed')
    parser.add_argument('--headless', action='store_true', help='Validate without opening a window')
    parser.add_argument('--exit-on-done', action='store_true')
    parser.add_argument('--camera', default='pitch_side')
    parser.add_argument('--out', type=Path, default=Path(f'outputs/pitch_pot/{robot_id}/result.json'))
    parser.add_argument('--hz', type=float, default=50, help='Display clock only; physics stays at 50 Hz')
    args = parser.parse_args(argv)
    if not np.isfinite(args.hz) or args.hz <= 0:
        parser.error('--hz must be positive and finite')
    spec, env, motion = make_reference_env(robot_id, seed=args.seed)
    pygame = renderer = recorder = None
    result = None
    try:
        if not args.headless:
            import pygame
            from react_manip.rendering import NativeRGBRenderer
            pygame.init()
            screen = pygame.display.set_mode((960, 810))
            pygame.display.set_caption(f'Pitch pot - {robot_id} physical reference')
            font = pygame.font.Font(None, 24)
            clock = pygame.time.Clock()
            renderer = NativeRGBRenderer(960, 720)
        cameras = list(dict.fromkeys([args.camera, 'pitch_side', 'workspace_top', 'room_overview']))
        camera_index = 0
        def start():
            if renderer:
                renderer.close()
            env.reset()
            return PickThrowController(env, motion=motion)
        controller = start()
        paused = finished = False
        running = True
        steps = 0
        milestones, events = [], []
        while running:
            for event in ([] if pygame is None else pygame.event.get()):
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_c:
                        camera_index = (camera_index + 1) % len(cameras)
                    elif event.key == pygame.K_BACKSPACE:
                        controller = start()
                        paused = finished = False
                        steps, milestones, events, result = 0, [], [], None
            if not running:
                break
            if not paused and not finished:
                _, _, terminated, truncated, info = env.step(controller.act())
                steps += 1
                milestones.extend(info['milestones'])
                events.extend(info['events'])
                finished = terminated or truncated
                if finished:
                    result = dict(task_id=spec.task_id, task_version=spec.task_version, robot_id=robot_id, robot_model=env.robots[0].robot_model.__class__.__name__,
                                  action_dim=env.action_dim, success=bool(env._check_success()), score=env.score,
                                  steps=steps, elapsed_sec=float(env.sim.data.time), phase=env.phase,
                                  failure=info.get('failure'), milestones=milestones, events=events,
                                  flight_platform_contacts=env.flight_platform_contacts,
                                  release_position=None if env.release_position is None else env.release_position.tolist(),
                                  release_velocity=None if env.release_velocity is None else env.release_velocity.tolist(),
                                  scenario=env.scenario.to_dict(), motion=asdict(motion))
                    try:
                        args.out.parent.mkdir(parents=True, exist_ok=True)
                        args.out.write_text(json.dumps(result, indent=2) + '\n')
                    except OSError as exc:
                        print(f'[pitch_pot] warning: could not save result: {exc}', flush=True)
                    print(json.dumps({k: result[k] for k in ('robot_id','success','score','steps','failure','flight_platform_contacts')}), flush=True)
            if renderer:
                rgb = renderer.render(env.sim, cameras[camera_index])
                screen.fill((24, 26, 30))
                screen.blit(pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1)), (0, 0))
                lines = [f'{robot_id} | {controller.stage} | {env.phase} | score={env.score:.0f} | t={env.sim.data.time:.2f}s',
                         'P pause | Backspace retry | C camera | Esc quit',
                         f'Fixed reference scene | {"PAUSED" if paused else "FINISHED" if finished else "RUNNING"} | platform contacts={env.flight_platform_contacts}']
                for i, line in enumerate(lines):
                    screen.blit(font.render(line, True, (235,235,235)), (12, 730+i*25))
                pygame.display.flip()
                clock.tick(args.hz)
            if finished and (args.headless or args.exit_on_done):
                break
        return 0 if result and result['success'] else 1
    finally:
        if recorder:
            recorder.close()
        if renderer:
            renderer.close()
        env.close()
        if pygame:
            pygame.quit()
