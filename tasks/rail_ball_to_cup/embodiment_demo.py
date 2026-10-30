"""Embodiment-specific action packing for physical grasp and rail placement."""
import numpy as np
from scipy.spatial.transform import Rotation
from .operation import PickPlaceController


class EmbodimentPickPlaceController(PickPlaceController):
    def __init__(self, env, release_x=None, **kwargs):
        if release_x is None:
            release_x = env.embodiment_config.get('release_x', .35)
        super().__init__(env, release_x=release_x, **kwargs)

    def reset(self):
        super().reset()
        self.layout = self.env.robots[0].composite_controller._action_split_indexes
        self.park = {arm: (self.env.eef_pos(arm), self.env.eef_mat(arm))
                     for arm in self.env.robots[0].arms if arm != self.env.active_arm}

    def act(self, observation=None, info=None):
        primary = super().act(observation, info)
        e = self.env
        action = np.zeros(e.action_dim)
        action[slice(*self.layout[e.active_arm])] = primary[:6]
        action[slice(*self.layout[e.active_arm+'_gripper'])] = primary[6]
        for arm, (position, orientation) in self.park.items():
            delta = (position-e.eef_pos(arm))/e.position_scale
            rot = Rotation.from_matrix(orientation@e.eef_mat(arm).T).as_rotvec()/e.rotation_scale
            action[slice(*self.layout[arm])] = np.clip(np.r_[delta,rot],-1,1)
            action[slice(*self.layout[arm+'_gripper'])] = -1
        return action


def main(robot_id, argv=None):
    """Live by default, same deterministic action sequence in --headless mode."""
    import argparse
    import json
    from pathlib import Path
    from .runtime import make_env

    parser = argparse.ArgumentParser(description=f'{robot_id}: physical pick, place and rail launch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scenario', type=Path)
    parser.add_argument('--release-x', type=float)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--exit-on-done', action='store_true')
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--hz', type=float, default=50.)
    parser.add_argument('--camera', default='rail_side')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--out', type=Path, default=Path(f'outputs/rail_ball_to_cup/{robot_id}/result.json'))
    args = parser.parse_args(argv)
    if not np.isfinite([args.seconds,args.hz]).all() or min(args.seconds,args.hz) <= 0:
        parser.error('--seconds and --hz must be positive and finite')
    spec, env = make_env(robot_id=robot_id, seed=args.seed, scenario=args.scenario,
                         hard_reset=False, initialization_noise=None)
    renderer = pygame = writer = None
    result = None
    try:
        env.reset()
        release_x = args.release_x if args.release_x is not None else env.embodiment_config.get('release_x', .35)
        controller = EmbodimentPickPlaceController(env, release_x=release_x)
        cameras = list(dict.fromkeys([args.camera,'workspace_oblique','workspace_top','room_overview']))
        camera_index = 0
        if not args.headless:
            import pygame
            from react_manip.rendering import NativeRGBRenderer
            pygame.init()
            screen = pygame.display.set_mode((960,810))
            pygame.display.set_caption(f'Rail ball to cup - {robot_id}')
            font = pygame.font.Font(None,24)
            clock = pygame.time.Clock()
            renderer = NativeRGBRenderer(960,720)
        if args.video:
            import cv2
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.video),cv2.VideoWriter_fourcc(*'mp4v'),env.control_freq,(960,720))
            if not writer.isOpened():
                raise RuntimeError('Cannot open video writer')
        running = True
        paused = finished = False
        steps = 0
        info = {}
        milestones = []
        limit = min(env.horizon, int(np.ceil(args.seconds*env.control_freq)))
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
                        camera_index = (camera_index+1)%len(cameras)
                    elif event.key == pygame.K_BACKSPACE:
                        env.reset()
                        controller.reset()
                        steps, milestones, info = 0, [], {}
                        paused = finished = False
                        result = None
            if not running:
                break
            if not paused and not finished:
                _, reward, terminated, truncated, info = env.step(controller.act())
                milestones.extend(info.get('milestones', []))
                steps += 1
                if writer is not None:
                    rgb = env.render_camera(args.camera,width=960,height=720)
                    writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                finished = terminated or truncated or steps >= limit
                if finished:
                    result = dict(success=bool(info.get('terminal_success')), score=info.get('total_score', 0),
                                  steps=steps, elapsed_sec=float(env.sim.data.time),
                                  robot_id=robot_id, active_arm=env.active_arm,
                                  robot_model=env.robots[0].robot_model.name, action_dim=env.action_dim,
                                  release_x=release_x, stage=controller.stage, milestones=milestones,
                                  failure=info.get('failure'), scenario=env.scenario.to_dict(),
                                  embodiment=env.embodiment_config,
                                  launch_velocity=None if env.launch_velocity is None else env.launch_velocity.tolist())
                    args.out.parent.mkdir(parents=True,exist_ok=True)
                    args.out.write_text(json.dumps(result,indent=2)+'\n')
                    print(json.dumps(result),flush=True)
            if renderer is not None:
                rgb = renderer.render(env.sim,cameras[camera_index])
                screen.fill((24,26,30))
                screen.blit(pygame.surfarray.make_surface(np.swapaxes(rgb,0,1)),(0,0))
                lines = [f'{robot_id} | {controller.stage} | {env.phase} | score={env.score:g}/{spec.metadata["score_max"]}',
                         'P pause | Backspace retry | C camera | Esc quit',
                         f'Release x={release_x:.3f} m | t={env.sim.data.time:.2f}s | physical contact only']
                for i,line in enumerate(lines):
                    screen.blit(font.render(line,True,(235,235,235)),(12,735+i*24))
                pygame.display.flip()
                clock.tick(args.hz)
            if finished and (args.headless or args.exit_on_done):
                break
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
        env.close()
        if pygame is not None:
            pygame.quit()
    return 0 if result and result['success'] else 1
