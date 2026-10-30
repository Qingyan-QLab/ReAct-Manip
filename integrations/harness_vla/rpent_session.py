"""Optional RPent Planner/Toolkit binding; no simulator dependencies in the planner.

The model-specific VLA encoder/action decoder is supplied explicitly. Importing
this module requires RPent's base dependencies, not its LIBERO/RoboCasa extras.
"""

import asyncio
import importlib
import json
from pathlib import Path
import uuid

import numpy as np

from .bridge import HarnessVLAPolicy
from .primitives import Primitives
from .rgbd import back_project


def load_factory(reference):
    module, separator, name = reference.partition(':')
    if not separator or not module or not name:
        raise ValueError('Factory must be specified as module:function')
    return getattr(importlib.import_module(module), name)


def build(config, output_dir):
    """Build one instance-scoped policy; require an explicit frozen VLA adapter."""
    if not config.get('vla_factory') or not config.get('planner_model'):
        raise ValueError('Set vla_factory and planner_model before model evaluation; use --smoke-steps for infrastructure checks')
    backend = config.get('planner_backend', 'api')
    if backend not in ('api', 'codex'):
        raise ValueError('planner_backend must be api or codex')
    if backend == 'codex' and config.get('planner_base_url'):
        raise ValueError('Codex uses CODEX_BASE_URL, not planner_base_url')
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.planner.base import build_planner
    from rpent.session import EnvState
    from rpent.tools import tool, ToolResult
    from rpent.tools.toolkit import Toolkit
    from rpent.utils.logging import init_output_dir

    init_output_dir(output_dir)

    root = Path(output_dir) / ('instance_' + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    vla = load_factory(config['vla_factory'])(config.get('vla', {}))
    provenance = getattr(vla, 'provenance', {})
    primitive_class = Primitives
    if provenance.get('action_schema') == 'react-manip.rpent.v1':
        from integrations.rpent_embodiments.primitives import Primitives as NativePrimitives
        primitive_class = NativePrimitives
    if (provenance.get('weights_frozen') is not True
            or provenance.get('action_schema') not in ('react-manip.osc.v1', 'react-manip.rpent.v1')
            or not provenance.get('checkpoint_id')):
        close = getattr(vla, 'close', None)
        if close is not None:
            close()
        raise ValueError('VLA adapter must declare frozen weights, checkpoint_id, and a supported ReAct-Manip action schema')
    sink = NullDashboardEventSink()
    # One inbox per Task Instance, shared by practice and exam. No cross-instance
    # publication or downloaded task memories are performed by this adapter.
    manager = MemoryManager(root / 'memory', memory_access='inbox_write', inbox_cell_tag='instance')
    inbox = manager.root / '_internal' / 'inbox' / 'instance'

    class ReactManipToolkit(Toolkit):
        def __init__(self, env, directory, task_spec):
            self.env = env
            self.horizon_sec = getattr(task_spec, 'horizon_sec', None)
            self.primitives = primitive_class(env, vla)
            super().__init__(dashboard_events=sink, state=EnvState(directory), memory=manager)
            self.capture()

        def _register_common_tools(self):
            # RPent's general file tools permit files outside memory. Expose
            # scoped notes instead so the planner cannot read benchmark banks.
            for name in ('observe', 'locate', 'read_memory', 'write_memory', 'finish',
                         'move_to', 'move_pose', 'rotate_wrist', 'rotate_pitch',
                         'set_gripper', 'release', 'vla_act'):
                self.add_tool(getattr(self, name))
            if hasattr(self.primitives, 'move_base') and 'base' in self.primitives.contract['action_layout']:
                self.add_tool(self.move_base)

        def execute_tool(self, name, input_dict):
            if self.env.cancelled.is_set():
                raise asyncio.CancelledError()
            result = super().execute_tool(name, input_dict)
            if self.env.cancelled.is_set():
                raise asyncio.CancelledError()
            return result

        def capture(self, command=None, result=None, elapsed_s=None):
            current = self.env.current
            observation = current.observation
            state = dict(info=current.info,
                         proprioception={key: value for key, value in observation.items()
                                         if key.startswith('robot0_') or key in ('eef_pos', 'eef_mat', 'gripper_qpos')})
            with self.state.record_step(state=state, terminated=current.terminated,
                                        truncated=current.truncated, command=command,
                                        result=result, elapsed_s=elapsed_s):
                for camera, image in observation['images'].items():
                    self.state.save(camera + '.png', image)
                    # NPY is sequential; compressed NPZ requires seeks that
                    # are unsupported by the deployment's OSS filesystem.
                    if self.state.save(camera + '_depth.npy', observation['depth_m'][camera]) is None:
                        raise RuntimeError('Failed to save metric depth artifact')
                self.state.save('calibration.json', observation['camera_calibration'])
            return self.observe()

        def get_env_state(self, *, command, result, elapsed_s):
            return self.capture(command, result, elapsed_s)

        @tool(readonly=True)
        def observe(self) -> ToolResult:
            """Read current public images, robot poses and task feedback without advancing time."""
            import io
            from PIL import Image
            obs = self.env.current.observation
            images = []
            for rgb in obs['images'].values():
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format='PNG')
                images.append(buffer.getvalue())
            poses = {arm: dict(position=self.primitives.pose(arm)[0].tolist(),
                               rotation_matrix=self.primitives.pose(arm)[1].tolist())
                     for arm in self.primitives.contract['arms']}
            elapsed = self.env.current.info.get('elapsed_sec')
            remaining = (max(0., self.horizon_sec - elapsed)
                         if self.horizon_sec is not None and elapsed is not None else None)
            episode = dict(elapsed_sim_sec=elapsed, horizon_sim_sec=self.horizon_sec,
                           remaining_sim_sec=remaining,
                           terminated=self.env.current.terminated,
                           truncated=self.env.current.truncated)
            return ToolResult(data=dict(cameras=list(obs['images']), poses=poses, episode=episode,
                                        info=self.env.current.info), images=images)

        @tool(readonly=True)
        def locate(self, camera: str, row: int, column: int) -> ToolResult:
            """Back-project one visible pixel into world metres using RGB-D sensor calibration."""
            obs = self.env.current.observation
            point = back_project(obs['depth_m'][camera], obs['camera_calibration'][camera],
                                 np.array([[row, column]]))[0]
            return ToolResult(data={'xyz': point.tolist()})

        def note_path(self, layer):
            if layer not in ('task-specific', 'global'):
                raise ValueError('Memory layer must be task-specific or global')
            return inbox / (layer + '.md')

        @tool(readonly=True)
        def read_memory(self, layer: str) -> ToolResult:
            """Read task-specific strategy or global primitive lessons learned in this Task Instance."""
            path = manager.authorize_read(self.note_path(layer))
            return ToolResult(data={'content': path.read_text() if path.exists() else ''})

        @tool(readonly=True)
        def write_memory(self, layer: str, content: str) -> ToolResult:
            """Replace task-specific strategy or global lessons for subsequent episodes in this instance."""
            path = manager.authorize_write(self.note_path(layer))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return ToolResult(data={'saved': layer})

        @tool(readonly=True)
        def finish(self, status: str, summary: str) -> ToolResult:
            """Finish planning and save an outcome summary; only benchmark feedback determines the score."""
            return ToolResult(data={'_finish': True, 'status': status, 'summary': summary})

        @tool
        def move_to(self, xyz: list[float], arm: str = 'right', max_steps: int = 100) -> ToolResult:
            """Move to a world XYZ target in metres while holding orientation and gripper commands."""
            return ToolResult(data=self.primitives.move_to(xyz, arm=arm, max_steps=max_steps))

        @tool
        def move_pose(self, xyz: list[float], quaternion_xyzw: list[float], arm: str = 'right', max_steps: int = 100) -> ToolResult:
            """Move to world XYZ and absolute XYZW orientation using bounded robot-specific control steps."""
            return ToolResult(data=self.primitives.move_pose(xyz, quaternion_xyzw, arm=arm, max_steps=max_steps))

        @tool
        def rotate_wrist(self, radians: float, arm: str = 'right', max_steps: int = 100) -> ToolResult:
            """Rotate relative to the current local tool Z axis while holding position."""
            return ToolResult(data=self.primitives.rotate(radians, 2, arm=arm, max_steps=max_steps))

        @tool
        def rotate_pitch(self, radians: float, arm: str = 'right', max_steps: int = 100) -> ToolResult:
            """Rotate relative to the current local tool Y axis while holding position."""
            return ToolResult(data=self.primitives.rotate(radians, 1, arm=arm, max_steps=max_steps))

        @tool
        def set_gripper(self, closed: bool, arm: str = 'right', steps: int = 10) -> ToolResult:
            """Command open or closed gripper for a bounded number of control steps."""
            return ToolResult(data=self.primitives.set_gripper(closed, arm=arm, steps=steps))

        @tool
        def release(self, arm: str = 'right', steps: int = 10) -> ToolResult:
            """Open the gripper; inspect subsequent observations to verify physical release."""
            return ToolResult(data=self.primitives.set_gripper(False, arm=arm, steps=steps))

        @tool
        def move_base(self, velocity: list[float], torso: float = 0., steps: int = 10) -> ToolResult:
            """Move a mobile base with normalized local vx, vy, yaw commands in [-1,1].

            Torso is a normalized lift command. Executes steps then one stop tick.
            """
            return ToolResult(data=self.primitives.move_base(velocity, torso=torso, steps=steps))

        @tool
        def vla_act(self, prompt: str, max_chunks: int = 4) -> ToolResult:
            """Execute the frozen VLA on current public views for a bounded number of chunks."""
            return ToolResult(data=self.primitives.vla_act(prompt, max_chunks=max_chunks))

    def session(env, context, memory):
        if hasattr(vla, 'start_episode'):
            vla.start_episode()
        directory = root / f"{context['phase']}_{context['episode_index']:02d}"
        toolkit = ReactManipToolkit(env, directory, context.get('task_spec'))
        try:
            planner_options = dict(
                output_dir=directory, recipe_tag=directory.name, robot_name='react_manip',
                model=config['planner_model'], base_url=config.get('planner_base_url'),
                memory_dir=manager.root,
                reasoning_effort=config.get('planner_reasoning_effort', 'none'),
                planner_timeout_s=config.get('planner_timeout_s', 1200), dashboard_events=sink)
            if backend == 'codex':
                from .codex_planner import build_codex_planner
                planner = build_codex_planner(**planner_options)
            else:
                planner = build_planner(backend, **planner_options)
            result = planner.solve(
                system_prompt=(
                    'Solve the task with the fixed manipulation primitives and frozen VLA. '
                    'Read task-specific and global memory, then observe the current scene. '
                    'Ground spatial targets through visible RGB-D pixels with locate. '
                    'Use analytic motion for staging and transport; use vla_act for contact. '
                    'Verify outcomes from public observations and task feedback. '
                    'The episode budget is simulation time: only executed robot steps advance it; '
                    'model inference and network retries do not. Read episode.remaining_sim_sec '
                    'and episode.terminated/truncated before claiming timeout. '
                    'Give vla_act short concrete action instructions; spatial planning and '
                    'pixel localization belong to you and the locate/motion tools. '
                    'Save reusable strategies and failure lessons in memory during practice and exam. '
                    'No environment resets are available. Success and score belong to the benchmark. '
                    'finish stops planning; the adapter holds the current pose and gripper commands '
                    'until the benchmark ends the episode.'),
                user_message=json.dumps(dict(phase=context['phase'], episode_index=context['episode_index'],
                                             instruction=env.current.info.get('language_prompt', ''),
                                             previous_episode=memory.get('last_episode_feedback')), default=str),
                toolkit=toolkit, max_turns=config.get('max_turns', 100))
            (directory / 'planner_result.json').write_text(json.dumps(
                dict(finish=result.finish_result, stats=result.stats, error=result.error,
                     messages=result.messages), default=str, ensure_ascii=False))
            memory['last_planner_stats'] = result.stats
            if result.error and not env.cancelled.is_set():
                raise RuntimeError(result.error)
            while not env.cancelled.is_set():
                arm = next(iter(toolkit.primitives.grippers))
                env.step(toolkit.primitives.action(arm, np.zeros(6)))
        except asyncio.CancelledError:
            if not env.cancelled.is_set():
                raise
        finally:
            toolkit.close()

    policy = HarnessVLAPolicy(session, action_timeout_s=config.get('planner_timeout_s', 1200) + 30,
                             close_timeout_s=config.get('close_timeout_s', 30))
    def shutdown():
        try:
            policy.close()
        finally:
            close = getattr(vla, 'close', None)
            if close is not None:
                close()
    policy.shutdown = shutdown
    policy.provenance = dict(vla=provenance, planner=config['planner_model'],
                             planner_interface_version='react-manip.rpent-feedback.v2',
                             planner_backend=backend,
                             planner_reasoning_effort=config.get('planner_reasoning_effort', 'none'),
                             memory='fresh per instance; online practice/exam',
                             primitive_version='react-manip.harness-primitives.v1')
    return policy
