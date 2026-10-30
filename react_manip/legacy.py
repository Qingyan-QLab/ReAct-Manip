"""Compatibility helpers for migrating existing robosuite tasks.

The v1 environments remain untouched.  These helpers only provide metadata
and factory conventions so they can be evaluated by the v2 runner while task
implementations are migrated one at a time.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .core.protocol import normalize_reset, normalize_step
from .core.language import runtime_language_prompt


class TaskRobotProxy:
    """Embodiment context for task-level environments without arm physics.

    Native scene tasks (for example vehicle or board actuators) can accept a
    robot selection while keeping their explicit task-level action API.  This
    proxy makes that choice visible to generic frontends and benchmark logs;
    it intentionally exposes no EEF/controller attributes, so callers cannot
    mistake it for a simulated manipulator.
    """

    def __init__(self, robot_id: str):
        from .robots.catalog import normalize_robot_id
        self.robot_id = normalize_robot_id(robot_id)
        # Context is a label, not a loaded robot model. Do not load assets or
        # fabricate joints/arms for a native task-level scene.
        self.name = self.robot_id
        self.arms = ()

    @property
    def spec(self):
        from .robots.base import RobotSpec
        return RobotSpec(
            robot_id=self.robot_id, arm_names=(),
            arm_dof=(), gripper_type=None,
            control_modes=("task_level",), capabilities=frozenset({"task_level_proxy"}),
        )


def attach_robot_context(env: Any, robot_id: str | None) -> Any:
    """Attach a normalized robot context to native task-level environments."""
    if robot_id is None:
        return env
    from .robots.catalog import normalize_robot_id
    normalized = normalize_robot_id(robot_id)
    env.robot_id = normalized
    if getattr(env, 'robots', None):
        return env
    # Do not populate ``env.robots``: frontends reserve that attribute for
    # physically compiled robot models with controllers and EEF sites.
    env.robot_context = TaskRobotProxy(normalized)
    return env


class _NativeModelView:
    """Expose a raw ``mujoco.MjModel`` through robosuite's model surface."""

    def __init__(self, model: Any):
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class _NativeSimView:
    # Marker consumed by frontends that need to distinguish a raw MuJoCo
    # scene from robosuite's MjSim wrapper.  The model is intentionally still
    # exposed through the same ``sim.model / sim.data`` surface.
    native_mujoco = True

    def __init__(self, env: Any):
        self.model = _NativeModelView(env.model)
        self.data = env.data


class LegacyTaskAdapter:
    """Thin wrapper adding a v2-friendly signal surface to an existing env."""

    def __init__(self, env: Any):
        self.env = env
        self._counter_snapshot: dict[str, float] = {}
        self._episode_index = -1
        self.language_prompt: dict[str, Any] = {}
        self._camera_renderers = {}
        self._camera_observation_ids = ()
        self._camera_resolution = (640, 480)

    def get_camera_specs(self):
        from .cameras import camera_specs
        return camera_specs(self.env)

    def get_camera_calibration(self, camera_id, *, width=640, height=480):
        """Internal diagnostics only: optical +Z forward, +Y down."""
        import mujoco
        import numpy as np
        from .cameras import raw_sim
        _, model, data = raw_sim(self.env)
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        if cid < 0:
            raise ValueError(f'Unknown camera: {camera_id}')
        f = .5 * height / np.tan(np.deg2rad(model.cam_fovy[cid] / 2))
        rotation = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
        world_from_optical = np.eye(4)
        world_from_optical[:3, :3] = rotation
        world_from_optical[:3, 3] = data.cam_xpos[cid]
        return dict(K=np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]]),
                    world_from_optical=world_from_optical, width=width, height=height)

    def render_camera(self, camera_id='workspace_oblique', *, width=640, height=480):
        from react_manip.rendering import NativeRGBRenderer
        key = (int(width), int(height))
        if key not in self._camera_renderers:
            self._camera_renderers[key] = NativeRGBRenderer(*key)
        return self._camera_renderers[key].render(self.sim, camera_id).copy()

    def get_camera_observation(self, camera_ids=None, *, width=640, height=480):
        if camera_ids is None:
            camera_ids = [s['id'] for s in self.get_camera_specs() if s['policy_default']]
        elif isinstance(camera_ids, str):
            camera_ids = (camera_ids,)
        return {name: self.render_camera(name, width=width, height=height) for name in camera_ids}

    def set_camera_observations(self, camera_ids=(), *, width=640, height=480):
        """Opt-in RGB under observation['images']; no implicit room/debug input."""
        if isinstance(camera_ids, str):
            camera_ids = (camera_ids,)
        available = {s['id'] for s in self.get_camera_specs()}
        missing = set(camera_ids) - available
        if missing:
            raise ValueError(f'Unavailable cameras: {sorted(missing)}')
        if width <= 0 or height <= 0:
            raise ValueError('camera resolution must be positive')
        self._camera_observation_ids = tuple(camera_ids)
        self._camera_resolution = (int(width), int(height))

    def _with_camera_observations(self, observation):
        if not self._camera_observation_ids:
            return observation
        if not isinstance(observation, Mapping):
            observation = {'state': observation}
        width, height = self._camera_resolution
        return dict(observation, images=self.get_camera_observation(
            self._camera_observation_ids, width=width, height=height))

    @property
    def robot_id(self) -> str | None:
        explicit = getattr(self.env, "robot_id", None)
        if explicit:
            return str(explicit)
        robots = getattr(self.env, "robots", ()) or ()
        if robots:
            from .robots.resolution import robot_id_from_model
            try:
                return robot_id_from_model(getattr(robots[0], 'name', ''))
            except ValueError:
                pass
        return None

    @property
    def task_id(self) -> str:
        explicit = getattr(self.env, "task_id", None)
        if explicit:
            return str(explicit)
        name = self.env.__class__.__name__
        import re
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def get_language_prompt(self) -> dict[str, Any]:
        """Return the current episode's structured language instruction."""
        return runtime_language_prompt(self.env)

    _EVENT_COUNTERS = (
        ("in_box_count", "object_in_box"),
        ("hits", "hit"),
        ("counted", "juggle"),
        ("op", "round"),
    )

    @property
    def action_dim(self) -> int:
        return int(self.env.action_dim)

    @property
    def sim(self) -> Any:
        """Expose the simulator for read-only recorders and diagnostics."""
        sim = getattr(self.env, "sim", None)
        if sim is not None:
            return sim
        if hasattr(self.env, "model") and hasattr(self.env, "data"):
            return _NativeSimView(self.env)
        return self.env

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._counter_snapshot.clear()
        self._episode_index += 1
        value = self.env.reset(*args, **kwargs)
        sim = self.sim
        model = getattr(sim, 'model', None)
        if model is not None and hasattr(model, 'ncam'):
            from .cameras import configure_fixed_cameras, configure_policy_camera
            configure_policy_camera(self.env)
            configure_fixed_cameras(self.env)
        for attr, _ in self._EVENT_COUNTERS:
            current = getattr(self.env, attr, None)
            if isinstance(current, (int, float)):
                self._counter_snapshot[attr] = float(current)
        self._counter_snapshot["drop_count"] = float(getattr(self.env, "drop_count", 0))
        prompt = self.get_language_prompt()
        self.language_prompt = prompt
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
            info = dict(value[1])
            info["language_prompt"] = prompt["text"]
            info["language_prompt_structured"] = prompt
            return self._with_camera_observations(value[0]), info
        return self._with_camera_observations(value)

    def step(self, action: Any) -> Any:
        value = normalize_step(self.env.step(action))
        info = dict(value.info)
        self.language_prompt = self.get_language_prompt()
        info["language_prompt"] = self.language_prompt["text"]
        info["language_prompt_structured"] = self.language_prompt
        events = list(_as_events(info))
        for attr, event_name in self._EVENT_COUNTERS:
            value_now = getattr(self.env, attr, None)
            if isinstance(value_now, (int, float)):
                previous = self._counter_snapshot.get(attr, float(value_now))
                delta = int(max(0, float(value_now) - previous))
                events.extend([event_name] * delta)
                self._counter_snapshot[attr] = float(value_now)
        # Common v1 task fields use ``in`` or a positive score for one-shot
        # events but did not expose the new ``events`` key.
        if info.get("in") is True:
            events.append("in")
        if isinstance(info.get("score"), (int, float)) and info["score"] > 0:
            events.append("score")
        if events:
            info["events"] = events
        if getattr(self.env, "drop_count", 0) > self._counter_snapshot.get("drop_count", 0):
            info.setdefault("failure", {"recoverable": True})
        self._counter_snapshot["drop_count"] = float(getattr(self.env, "drop_count", 0))
        if "success_state" not in info:
            if hasattr(self.env, "_check_success"):
                info["success_state"] = bool(self.env._check_success())
            else:
                info["success_state"] = bool(value.reward > 0)
        if "success" not in info and "terminal_success" not in info:
            info["terminal_success"] = bool(info["success_state"] and value.done)
        sim = getattr(self.env, "sim", None)
        cfrc = getattr(getattr(sim, "data", None), "cfrc_ext", None)
        if cfrc is not None:
            try:
                info["peak_force_N"] = float(abs(cfrc).max())
            except (AttributeError, TypeError, ValueError):
                pass
        return self._with_camera_observations(value.observation), value.reward, value.terminated, value.truncated, info

    def close(self) -> None:
        for renderer in self._camera_renderers.values():
            renderer.close()
        self._camera_renderers.clear()
        self.env.close()

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Forward interactive rendering to the wrapped robosuite env."""

        viewer = getattr(self.env, "viewer", None)
        update_fn = getattr(viewer, "update", None)
        if callable(update_fn):
            # robosuite's native ``mjviewer`` has a no-op render() method;
            # update() is the operation that creates / presents the window.
            return update_fn()
        render_fn = getattr(self.env, "render", None)
        if not callable(render_fn):
            return None
        return render_fn(*args, **kwargs)


def legacy_factory(
    constructor: Callable[..., Any],
    *,
    robot_keyword: str = "robots",
    default_robot: str | None = None,
    fixed_kwargs: Mapping[str, Any] | None = None,
) -> Callable[..., LegacyTaskAdapter]:
    """Create a factory compatible with :class:`BenchmarkRunner`."""

    fixed_kwargs = dict(fixed_kwargs or {})

    def factory(*, seed: int | None = None, robot_id: str | None = None, **kwargs: Any) -> LegacyTaskAdapter:
        params = dict(fixed_kwargs)
        params.update(kwargs)
        if seed is not None:
            params.setdefault("seed", seed)
        if robot_id is not None or default_robot is not None:
            params.setdefault(robot_keyword, robot_id or default_robot)
        return LegacyTaskAdapter(constructor(**params))

    return factory


def _as_events(info: Mapping[str, Any]) -> list[Any]:
    events = info.get("events", info.get("event", ()))
    if events is None:
        return []
    if isinstance(events, (str, bytes)):
        return [events]
    try:
        return list(events)
    except TypeError:
        return [events]
