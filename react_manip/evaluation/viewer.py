"""Optional evaluator UI. Displays the policy's existing RGB, never steps the env."""
from pathlib import Path
import json
import re
import numpy as np


class SkipEpisode(Exception):
    """Raised by the interactive viewer when the current episode is skipped."""


def compose_policy_images(images):
    """Main image on the left, up to two wrist views on the right; no overlays."""
    from PIL import Image
    if not 1 <= len(images) <= 3:
        raise ValueError('evaluation viewer requires one to three policy cameras')
    names = list(images)
    for name, frame in images.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('invalid policy camera ID')
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError('policy cameras must provide uint8 HWC RGB')
    if len(names) == 1:
        frame = images[names[0]]
        return frame, {names[0]: (0, 0, frame.shape[1], frame.shape[0])}
    width, height = 960, 480
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    boxes = [(0, 0, 640, 480), (640, 0, 320, 240), (640, 240, 320, 240)]
    layout = {}
    for name, (x, y, w, h) in zip(names, boxes):
        source = Image.fromarray(images[name])
        scale = min(w/source.width, h/source.height)
        source = source.resize((max(1, round(source.width*scale)), max(1, round(source.height*scale))), Image.Resampling.LANCZOS)
        ox, oy = x + (w-source.width)//2, y + (h-source.height)//2
        canvas[oy:oy+source.height, ox:ox+source.width] = np.asarray(source)
        layout[name] = (ox, oy, source.width, source.height)
    return canvas, layout


def check_dependencies(display, video):
    try:
        if display:
            import pygame
        if video:
            import imageio.v2
            import imageio_ffmpeg
            imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise ImportError('Install dependencies: python -m pip install -r requirements.txt') from exc


class EvaluationViewer:
    def __init__(self, *, enabled=False, video_dir=None):
        self.enabled = enabled
        self.video_dir = Path(video_dir) if video_dir is not None else None
        self.active = enabled or self.video_dir is not None
        self.aborted = False
        self.skipped = False
        self.paused = False
        self.debug = False
        self.writer = None
        self.camera_writers = {}
        self.pg = None
        self.frames = 0

    def __enter__(self):
        check_dependencies(self.enabled, self.video_dir is not None)
        if self.video_dir is not None:
            self.video_dir.mkdir(parents=True, exist_ok=False)
        if self.enabled:
            import pygame
            self.pg = pygame
            try:
                pygame.display.init()
                pygame.font.init()
                self.screen = pygame.display.set_mode((1280, 720))
                pygame.display.set_caption('CL-Manip evaluation | policy RGB')
                self.font = pygame.font.SysFont('Arial', 18)
                self.clock = pygame.time.Clock()
            except BaseException:
                pygame.display.quit()
                pygame.font.quit()
                raise
        return self

    def __exit__(self, *exc):
        self.end_episode()
        if self.pg:
            self.pg.display.quit()
            self.pg.font.quit()

    def begin_episode(self, task, episode, phase, index, hz):
        self.end_episode()
        self.task, self.episode, self.phase = task, episode, phase
        self.index, self.hz, self.frames = index, hz, 0
        self.skipped = False
        self.camera = None
        self.cameras = ()
        self.image_shapes = {}
        self.layout = {}
        self.instruction_scroll = 0
        if self.video_dir is not None:
            import imageio.v2 as imageio
            # IDs are evaluator records, but never permit paths to escape output.
            if Path(episode).name != episode or episode in ('.', '..'):
                raise ValueError('invalid episode ID for video filename')
            self.writer = imageio.get_writer(str(self.video_dir / (episode + '.mp4')),
                                            fps=hz, codec='libx264', quality=8,
                                            macro_block_size=1, ffmpeg_log_level='error')

    def _events(self):
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT or (event.type == self.pg.KEYDOWN and event.key == self.pg.K_ESCAPE):
                self.aborted = True
                raise RuntimeError('evaluation aborted by viewer; no valid protocol score')
            if event.type == self.pg.KEYDOWN:
                if event.key == self.pg.K_s:
                    self.skipped = True
                    raise SkipEpisode('episode skipped by viewer (S)')
                if event.key == self.pg.K_SPACE:
                    self.paused = not self.paused
                elif event.key == self.pg.K_DOWN:
                    self.instruction_scroll += 1
                elif event.key == self.pg.K_UP:
                    self.instruction_scroll = max(0, self.instruction_scroll - 1)
                elif event.key == self.pg.K_TAB:
                    self.debug = not self.debug

    def _text(self, text, y, color=(230, 230, 230)):
        # Wrap by rendered width, including long instructions without spaces.
        line = ''
        for char in str(text):
            if char == '\n' or self.font.size(line + char)[0] > 300:
                self.screen.blit(self.font.render(line, True, color), (970, y))
                y += 21
                line = ''
                if y > 475:
                    return y
            if char != '\n':
                line += char
        self.screen.blit(self.font.render(line, True, color), (970, y))
        return y + 24

    def _instruction(self, text, y):
        lines, line = [], ''
        for word in str(text).split():
            candidate = (line + ' ' + word).strip()
            if line and self.font.size(candidate)[0] > 300:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        visible = max(1, (480 - y) // 21)
        self.instruction_scroll = min(self.instruction_scroll, max(0, len(lines) - visible))
        for line in lines[self.instruction_scroll:self.instruction_scroll + visible]:
            self.screen.blit(self.font.render(line, True, (230, 230, 230)), (970, y))
            y += 21

    def observe(self, env, observation, info):
        images = observation.get('images', {})
        frame, layout = compose_policy_images(images)
        ids = tuple(images)
        shapes = {k: list(v.shape) for k, v in images.items()}
        if self.cameras and (ids != self.cameras or shapes != self.image_shapes):
            raise ValueError('policy camera IDs/order/resolution changed during an episode')
        if not self.cameras:
            self.cameras, self.image_shapes, self.layout = ids, shapes, layout
            self.camera = ids[0]
            if self.writer is not None:
                import imageio.v2 as imageio
                for name in ids:
                    self.camera_writers[name] = imageio.get_writer(
                        str(self.video_dir / (self.episode + '.' + name + '.mp4')),
                        fps=self.hz, codec='libx264', quality=8,
                        macro_block_size=1, ffmpeg_log_level='error')
        if self.enabled:
            self._events()
            while True:
                self.screen.fill((24, 26, 30))
                surface = self.pg.surfarray.make_surface(frame.transpose(1, 0, 2))
                scale = min(960/frame.shape[1], 720/frame.shape[0])
                size = (round(frame.shape[1]*scale), round(frame.shape[0]*scale))
                origin = ((960-size[0])//2, (720-size[1])//2)
                self.screen.blit(self.pg.transform.scale(surface, size), origin)
                for name, (x, y, _, _) in layout.items():
                    label = self.font.render(name, True, (255,255,255), (24,26,30))
                    self.screen.blit(label, (origin[0]+round(x*scale)+4, origin[1]+round(y*scale)+4))
                y = 12
                values = [self.task, f'{self.phase} episode {self.index + 1}',
                          ', '.join(self.cameras), f'Simulation: {float(env.sim.data.time):.2f} s',
                          'PAUSED' if self.paused else 'RUNNING',
                          'Space: pause / resume', 'S: skip episode',
                          'Tab: external debug view', 'Esc: abort evaluation', 'Up/Down: scroll instruction']
                for key in ('total_score', 'score', 'success_state'):
                    if key in info:
                        values.append(f'{key}: {info[key]}')
                for value in values:
                    if y > 475:
                        break
                    y = self._text(value, y)
                self._instruction(info.get('language_prompt', ''), y)
                if self.debug:
                    debug_frame = env.render_camera('room_overview', width=320, height=240)
                    surface = self.pg.surfarray.make_surface(debug_frame.transpose(1, 0, 2))
                    self.screen.blit(surface, (960, 480))
                    self.screen.blit(self.font.render('DEBUG - not policy input', True, (255, 200, 80)), (970, 480))
                self.pg.display.flip()
                if not self.paused:
                    break
                self.clock.tick(30)
                self._events()
            self.clock.tick(self.hz)
        if self.writer is not None:
            self.writer.append_data(frame)
            for name, writer in self.camera_writers.items():
                writer.append_data(images[name])
        self.frames += 1

    def end_episode(self):
        if self.writer is not None:
            writer, self.writer = self.writer, None
            try:
                writer.close()
            finally:
                for camera_writer in self.camera_writers.values():
                    camera_writer.close()
                self.camera_writers.clear()
            (self.video_dir / (self.episode + '.json')).write_text(json.dumps({
                'task_id': self.task, 'episode_id': self.episode, 'phase': self.phase,
                'camera': self.camera, 'cameras': list(self.cameras),
                'image_shapes': self.image_shapes, 'layout_xywh': self.layout,
                'composite_video': self.episode + '.mp4',
                'camera_videos': {n: self.episode + '.' + n + '.mp4' for n in self.cameras},
                'fps': self.hz, 'frames': self.frames,
                'image_source': 'same-tick policy RGB; composite plus separate views; no debug overlay',
                'aborted': self.aborted,
                'skipped': self.skipped,
            }, indent=2) + '\n')
