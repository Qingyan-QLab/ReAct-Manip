"""Lazy environment registration for the sixteen release tasks."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
import inspect
from .core.registry import get_task_registry
from .core.specs import TaskSpec, load_task_spec
from .robots.resolution import resolve_robosuite_robot, robot_id_from_model
from .instances.records import SUPPORTED_TASKS

def _lazy_constructor(module_name: str, class_name: str, default_robot: str, *, native_v2: bool = False) -> Callable[..., Any]:
    def factory(*, seed: int | None = None, robot_id: str | None = None, **kwargs: Any) -> Any:
        from importlib import import_module
        from .legacy import LegacyTaskAdapter, attach_robot_context

        scenario = kwargs.pop("scenario", None)
        cls = getattr(import_module(module_name), class_name)
        robot = robot_id or default_robot
        if native_v2:
            # Native v2 tasks are not robosuite environments. Construct them
            # from the parameters they actually advertise (CurlingSweep, for
            # example, accepts only ``seed``), while forwarding optional
            # arguments to richer native adapters.
            params = dict(kwargs)
            try:
                signature = inspect.signature(cls)
                accepts_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in signature.parameters.values()
                )
                accepted = set(signature.parameters)
            except (TypeError, ValueError):
                accepts_kwargs, accepted = True, set()
            if not accepts_kwargs:
                params = {key: value for key, value in params.items() if key in accepted}
            if accepts_kwargs or "seed" in accepted:
                params.setdefault("seed", seed)
            if "scenario" in accepted:
                params.setdefault("scenario", scenario)
            if "robot_id" in accepted:
                params.setdefault("robot_id", robot_id)
            if "robots" in accepted:
                mapped_robot = robot
                if isinstance(robot, str):
                    resolved = resolve_robosuite_robot(robot_id_from_model(robot))
                    mapped_robot = resolved.robot_name
                    if resolved.controller_configs is not None:
                        params.setdefault("controller_configs", resolved.controller_configs)
                    params.setdefault("gripper_types", resolved.gripper_types)
                    params.setdefault("initialization_noise", resolved.initialization_noise)
                params.setdefault("robots", mapped_robot)
            env = cls(**params)
            attach_robot_context(env, robot_id)
            env.scenario = scenario
            adapted = LegacyTaskAdapter(env)
            adapted.scenario = scenario
            return adapted
        if isinstance(robot, str):
            robot_key = robot_id_from_model(robot)
            resolved = resolve_robosuite_robot(robot_key)
            robot = resolved.robot_name
            # Preserve legacy task-tuned Panda/Sawyer defaults until their
            # controller profiles are migrated; new providers own defaults.
            if robot_key not in {"panda", "sawyer"}:
                kwargs.setdefault("controller_configs", resolved.controller_configs)
                kwargs.setdefault("gripper_types", resolved.gripper_types)
                kwargs.setdefault("initialization_noise", resolved.initialization_noise)
        env = LegacyTaskAdapter(cls(robots=robot, seed=seed, scenario=scenario, **kwargs))
        if isinstance(robot, str):
            env.env.robot_id = robot_key
        # Tasks may resolve a Profile themselves when no Scenario was supplied.
        # Preserve that concrete instance for replay IDs and benchmark logging.
        env.scenario = getattr(env.env, "scenario", None) or scenario
        return env

    return factory

def _laptop_factory(*, seed: int | None = None, robot_id: str | None = None, **kwargs: Any) -> Any:
    """Use the same physical env construction for v2 and task recordings."""

    from tasks.laptop_into_drawer.construction import make_env
    from .legacy import LegacyTaskAdapter

    scenario = kwargs.get("scenario")
    # Headless benchmark runs do not need an OpenGL context. Recorders opt in
    # explicitly so macOS CGL is never initialized on a non-GUI evaluation.
    kwargs.setdefault("has_offscreen_renderer", False)
    env = LegacyTaskAdapter(make_env(seed=0 if seed is None else seed, robot_id=robot_id or "dual_piperx", **kwargs))
    return env

def _maze_factory(*, seed=None, robot_id=None, **kwargs):
    from tasks.tilte_maze.env import TilteMazePhysical
    from .legacy import LegacyTaskAdapter
    env = TilteMazePhysical(robots=robot_id or 'dual_piperx', seed=seed, **kwargs)
    adapted = LegacyTaskAdapter(env)
    adapted.scenario = env.scenario
    return adapted

def _phone_lean_factory(**kwargs):
    from tasks.phone_lean.benchmark_adapter import make_benchmark_env
    return make_benchmark_env(**kwargs)

def _leveling_board1_factory(**kwargs):
    from tasks.leveling_board1.benchmark_adapter import make_benchmark_env
    return make_benchmark_env(**kwargs)

def _turntable_gamble1_factory(**kwargs):
    from tasks.turntable_gamble1.benchmark_adapter import make_benchmark_env
    return make_benchmark_env(**kwargs)

def _car_drive2_factory(**kwargs):
    from tasks.car_drive2.benchmark_adapter import make_benchmark_env
    return make_benchmark_env(**kwargs)

def _ring_toss1_factory(**kwargs):
    from tasks.ring_toss1.benchmark_adapter import make_benchmark_env
    return make_benchmark_env(**kwargs)

_SPECIAL_FACTORIES = {'laptop_into_drawer': _laptop_factory, 'tilte_maze': _maze_factory, 'phone_lean': _phone_lean_factory, 'leveling_board1': _leveling_board1_factory, 'turntable_gamble1': _turntable_gamble1_factory, 'car_drive2': _car_drive2_factory, 'ring_toss1': _ring_toss1_factory}

def discover_task_definitions(root: str | Path | None = None, *, task_ids=None):
    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[1] / 'tasks'
    definitions = []
    available = (task_id for task_id in SUPPORTED_TASKS
                 if (root_path / task_id / 'task_spec.json').is_file())
    for task_id in available if task_ids is None else task_ids:
        spec = load_task_spec(root_path / task_id / 'task_spec.json')
        class_name = ''.join(part.capitalize() for part in task_id.split('_'))
        factory = _SPECIAL_FACTORIES.get(task_id, _lazy_constructor(
            f'tasks.{task_id}.env', class_name,
            spec.supported_robots[0] if spec.supported_robots else 'dual_piperx',
            native_v2=bool(spec.metadata.get('native_v2', False))))
        definitions.append((spec, _camera_factory(factory, spec)))
    return tuple(definitions)

def _camera_factory(factory, spec=None):
    """Uniform opt-in RGB, independent of legacy use_camera_obs support."""
    from functools import wraps

    @wraps(factory)
    def create(*args, camera_observation_ids=(), camera_width=640, camera_height=480, **kwargs):
        if spec is not None:
            from .core.compatibility import resolve_task_robot
            kwargs['robot_id'] = resolve_task_robot(spec, kwargs.get('robot_id'))
        elif kwargs.get('robot_id') is not None:
            from .robots.catalog import normalize_robot_id
            kwargs['robot_id'] = normalize_robot_id(kwargs['robot_id'])
        env = factory(*args, **kwargs)
        try:
            from .cameras import configure_fixed_cameras
            configure_fixed_cameras(getattr(env, 'env', env))
            from .robots.agilex_robosuite import validate_compiled_grippers
            validate_compiled_grippers(env)
            env.set_camera_observations(camera_observation_ids, width=camera_width, height=camera_height)
            return env
        except Exception:
            env.close()
            raise
    return create

def register_builtin_tasks(*, replace: bool = False) -> None:
    """Register legacy tasks exactly once; safe to call repeatedly."""

    registry = get_task_registry()
    for spec, factory in discover_task_definitions():
        if spec.task_id in {definition.spec.task_id for definition in registry.list()} and not replace:
            continue
        registry.register(spec, factory, replace=replace)
