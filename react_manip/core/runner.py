"""Fixed-horizon policy evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math

import time
from typing import Any, Callable, Mapping

from .metrics import SustainedSuccessTracker
from .compatibility import validate_task_robot, resolve_task_robot
from .protocol import normalize_reset, normalize_step
from .observation import policy_info, policy_observation
from .registry import TaskDefinition
from .specs import TaskSpec
from .timeouts import DEFAULT_EPISODE_WALL_TIMEOUT_SEC, validate_timeout


def _call_with_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Call a factory without hiding errors raised inside its implementation."""

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fn(**kwargs)
    # Accommodate common adapter spellings without forcing every external
    # policy/factory to use our parameter names.  Build this mapping before
    # the direct-name fast path so ``act(obs, info)`` receives both values
    # (rather than only the directly named ``info`` argument).
    aliases = {
        "observation": ("obs", "observation", "state"),
        "info": ("info", "metadata"),
        "robot_id": ("robot_id", "robot", "embodiment"),
        "task_spec": ("task_spec", "spec", "task"),
        "seed": ("seed", "rng_seed"),
    }
    mapped: dict[str, Any] = {}
    for source, value in kwargs.items():
        for name in aliases.get(source, (source,)):
            if name in signature.parameters:
                mapped[name] = value
                break
    if mapped:
        return fn(**mapped)
    # Last resort for simple positional callables (e.g. lambda obs: action).
    positional = [
        param
        for param in signature.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    values = [kwargs[key] for key in ("observation", "info", "seed", "robot_id", "task_spec") if key in kwargs]
    if positional and values:
        return fn(*values[: len(positional)])
    return fn()


@dataclass
class EvaluationResult:
    task_id: str
    task_version: str
    robot_id: str | None
    policy_id: str
    seed: int | None
    metrics: dict[str, Any]
    wall_time_sec: float
    ended_early: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_api_version": "2.0",
            "metrics_schema_version": "2.0",
            "task_id": self.task_id,
            "task_version": self.task_version,
            "robot_id": self.robot_id,
            "policy_id": self.policy_id,
            "seed": self.seed,
            "wall_time_sec": round(self.wall_time_sec, 6),
            "ended_early": self.ended_early,
            **self.metrics,
        }


class BenchmarkRunner:
    """Run a policy for a fixed simulation horizon and wall-clock budget.

    The runner accepts either a :class:`TaskDefinition`, a ``TaskSpec`` plus a
    factory, or a raw factory.  This makes it possible to migrate existing
    robosuite environments gradually while new tasks use the registry.
    """

    def __init__(self, *, default_horizon_sec: float | None = None, default_control_hz: float | None = None,
                 episode_wall_timeout_sec: float | None = DEFAULT_EPISODE_WALL_TIMEOUT_SEC):
        self.default_horizon_sec = default_horizon_sec
        self.default_control_hz = default_control_hz
        self.episode_wall_timeout_sec = validate_timeout(episode_wall_timeout_sec, "episode_wall_timeout_sec")

    def evaluate(
        self,
        task: TaskDefinition | TaskSpec,
        factory: Callable[..., Any] | None = None,
        policy: Any | None = None,
        *,
        robot_id: str | None = None,
        seed: int | None = 0,
        horizon_sec: float | None = None,
        control_hz: float | None = None,
        policy_id: str | None = None,
        env_kwargs: Mapping[str, Any] | None = None,
        render: bool = False,
        render_realtime: bool = False,
        render_hold_sec: float = 3.0,
        step_callback: Callable[..., Any] | None = None,
        observation_callback: Callable[..., Any] | None = None,
        env: Any | None = None,
        stop_on_done: bool = False,
        score_key: str | None = None,
        camera_observations: bool = True,
        episode_wall_timeout_sec: float | None = None,
    ) -> EvaluationResult:
        # A supplied environment is borrowed: reset/step it, but leave closing
        # to its owner. stop_on_done prevents a multi-episode protocol from
        # consuming uncounted retries through a task's reset-on-done behavior.
        if isinstance(task, TaskDefinition):
            spec, task_factory = task.spec, task.factory
        elif isinstance(task, TaskSpec):
            if factory is None:
                raise TypeError("factory is required when task is a TaskSpec")
            spec, task_factory = task, factory
        else:  # pragma: no cover - defensive for callers using duck typing
            raise TypeError("task must be TaskDefinition or TaskSpec")

        hz = float(control_hz or self.default_control_hz or spec.control_hz)
        horizon = float(horizon_sec or self.default_horizon_sec or spec.horizon_sec)
        if hz <= 0 or horizon <= 0:
            raise ValueError("horizon_sec and control_hz must be positive")
        steps_limit = max(1, int(round(horizon * hz)))
        if render_hold_sec < 0:
            raise ValueError("render_hold_sec must be non-negative")
        episode_timeout = validate_timeout(
            self.episode_wall_timeout_sec if episode_wall_timeout_sec is None else episode_wall_timeout_sec,
            "episode_wall_timeout_sec",
        )

        robot_id = resolve_task_robot(spec, robot_id)
        robot_adapter = None
        if robot_id is not None:
            try:
                from .registry import get_robot_registry

                robot_adapter = get_robot_registry().get(robot_id.lower())
            except KeyError:
                # A custom application may provide a robot factory outside
                # the registry; task-level support checks still apply below.
                robot_adapter = None
        validate_task_robot(spec, robot_id, robot_adapter)
        if robot_adapter is not None:
            robot_spec = getattr(robot_adapter, "spec", None)
            capabilities = getattr(robot_spec, "capabilities", ())
            if "custom_mjcf" in capabilities and hasattr(robot_adapter, "validate_assets"):
                robot_adapter.validate_assets()

        factory_kwargs: dict[str, Any] = {"seed": seed, "robot_id": robot_id}
        if env_kwargs:
            factory_kwargs.update(dict(env_kwargs))
        if render:
            # robosuite environments use these constructor flags to create a
            # native viewer.  ``_call_with_supported_kwargs`` drops them for
            # custom factories that do not expose rendering options.
            factory_kwargs.setdefault("has_renderer", True)
            # Native mjviewer does not need an offscreen context. On macOS an
            # offscreen GLFW / CGL context can create an NSWindow from the
            # mjpython worker thread and abort the process.
            factory_kwargs.setdefault("has_offscreen_renderer", False)
            factory_kwargs.setdefault("use_camera_obs", False)
        owns_env = env is None
        if owns_env:
            env = _call_with_supported_kwargs(task_factory, factory_kwargs)
        tracker = SustainedSuccessTracker(control_hz=hz)
        started = time.perf_counter()
        ended_early = False
        terminal_success = False
        error_message: str | None = None
        reset_count = 0
        wall_clock_timeout = False
        try:
            from ..robots.sensors import policy_camera_ids, CAMERA_CONTRACT_VERSION
            from ..robots.catalog import ALL_ROBOT_IDS
            from ..optional_integrations import import_optional_integration
            rpent_catalog = import_optional_integration("integrations.rpent_embodiments.catalog")
            rpent_embodiments = () if rpent_catalog is None else rpent_catalog.EMBODIMENTS
            if robot_id in ALL_ROBOT_IDS or robot_id in rpent_embodiments:
                env.set_camera_observations(policy_camera_ids(robot_id) if camera_observations else ())
            reset_kwargs = {"seed": seed} if seed is not None else {}
            observation, reset_info = _reset_env(env, reset_kwargs)
            observation = policy_observation(observation, spec, env)
            # Every episode exposes the language instruction through reset
            # info, including legacy robosuite environments whose reset only
            # returns an observation.
            prompt_fn = getattr(env, "get_language_prompt", None)
            if callable(prompt_fn):
                prompt = prompt_fn()
                reset_info = dict(reset_info)
                if isinstance(prompt, Mapping):
                    reset_info.setdefault("language_prompt", prompt.get("text", ""))
                    reset_info.setdefault("language_prompt_structured", prompt)
            if render:
                _render_env(env)
            if policy is None:
                from react_manip.policies.base import RandomPolicy

                policy = RandomPolicy()
            if hasattr(policy, "set_action_dim"):
                policy.set_action_dim(_action_dim(env))
            if hasattr(policy, "reset"):
                _call_with_supported_kwargs(
                    policy.reset,
                    {"seed": seed, "task_spec": spec, "episode_index": reset_count},
                )

            last_info: Mapping[str, Any] = reset_info
            last_reward, terminated, truncated = 0.0, False, False
            try:
                for _ in range(steps_limit):
                    if episode_timeout is not None and time.perf_counter() - started >= episode_timeout:
                        wall_clock_timeout = True
                        ended_early = True
                        error_message = "wall_clock_timeout"
                        tracker.record_failure(at_sec=tracker.steps / hz)
                        break
                    # Evaluator display/recording is outside policy latency.
                    if observation_callback is not None:
                        observation_callback(env, observation, last_info)
                    tick_started = time.perf_counter()
                    policy_started = time.perf_counter()
                    action = _call_with_supported_kwargs(
                        policy.act,
                        {
                            "observation": observation,
                            "info": policy_info(last_info, policy, env),
                        },
                    )
                    inference_latency_ms = (time.perf_counter() - policy_started) * 1000.0
                    if episode_timeout is not None and time.perf_counter() - started >= episode_timeout:
                        wall_clock_timeout = True
                        ended_early = True
                        error_message = "wall_clock_timeout"
                        tracker.record_failure(at_sec=tracker.steps / hz)
                        break
                    result = normalize_step(env.step(action))
                    observation = policy_observation(result.observation, spec, env)
                    last_info = result.info
                    last_reward, terminated, truncated = result.reward, result.terminated, result.truncated
                    if step_callback is not None:
                        step_callback(env, _ + 1, result)
                    if render:
                        _render_env(env)
                        if render_realtime:
                            time.sleep(max(0.0, (1.0 / hz) - (time.perf_counter() - tick_started)))
                    # Preserve task-reported latency when present, otherwise
                    # make model inference cost measurable for every policy.
                    if "inference_latency_ms" not in last_info:
                        last_info = dict(last_info)
                        last_info["inference_latency_ms"] = inference_latency_ms
                    success_state = _success_state(last_info, result.reward)
                    events = _events(last_info)
                    failure = last_info.get("failure")
                    # A shaped reward may represent partial task progress
                    # (for example, a scored prerequisite).  When a task
                    # explicitly reports terminal success, that ground-truth
                    # value must take precedence over the legacy reward
                    # fallback.
                    if "terminal_success" in last_info:
                        terminal_success = bool(last_info["terminal_success"])
                    elif "success" in last_info:
                        terminal_success = bool(last_info["success"])
                    else:
                        terminal_success = result.reward > 0.0
                    behavior = spec.done_behavior
                    if result.done and not terminal_success and failure is None:
                        failure = {
                            "recoverable": behavior in {"reset", "fail_and_reset"},
                        }
                    tracker.update(
                        success_state,
                        events=events,
                        failure=failure,
                        dt=1.0 / hz,
                        info=last_info,
                    )
                    # A policy/environment call can itself consume the budget.
                    # Check after recording the completed tick so the result is
                    # still traceable without running another control step.
                    if (not result.done and episode_timeout is not None and
                            time.perf_counter() - started >= episode_timeout):
                        wall_clock_timeout = True
                        ended_early = True
                        error_message = "wall_clock_timeout"
                        tracker.record_failure(at_sec=tracker.steps / hz)
                        break
                    if result.done:
                        if not stop_on_done and behavior in {"reset", "fail_and_reset"} and _can_reset_for_horizon(steps_limit, _):
                            reset_count += 1
                            observation, last_info = _reset_env(
                                env,
                                {"seed": None if seed is None else seed + reset_count},
                            )
                            observation = policy_observation(observation, spec, env)
                            if hasattr(policy, "reset"):
                                _call_with_supported_kwargs(
                                    policy.reset,
                                    {
                                        "seed": None if seed is None else seed + reset_count,
                                        "task_spec": spec,
                                        "episode_index": reset_count,
                                    },
                                )
                            continue
                        ended_early = _ + 1 < steps_limit
                        break
            except Exception as exc:
                # The interactive evaluator uses a dedicated exception for
                # "skip current episode". Let the manifest runner handle it
                # between episodes instead of turning it into a normal failed
                # EvaluationResult.
                try:
                    from react_manip.evaluation.viewer import SkipEpisode
                except ImportError:  # pragma: no cover - optional UI module
                    SkipEpisode = ()
                if isinstance(exc, SkipEpisode):
                    raise
                error_message = f"{type(exc).__name__}: {str(exc)[:240]}"
                ended_early = True
                # The exception occurs between control ticks. Record it as a
                # failure without inventing an extra simulated tick.
                tracker.record_failure(at_sec=tracker.steps / hz)
        finally:
            if render and render_hold_sec > 0:
                time.sleep(render_hold_sec)
            if owns_env and hasattr(env, "close"):
                env.close()

        metrics = tracker.summarize(spec.success, terminal_success=terminal_success)
        if score_key is not None and error_message is None:
            value = last_info.get(score_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                error_message = f"missing or non-finite final task score: info.{score_key}"
            else:
                metrics["episode_score"] = float(value)
                metrics["score_source"] = f"info.{score_key}"
        sim_time = float(metrics.get("sim_time_sec", 0.0))
        metrics.update({
            "horizon_sec": horizon,
            "control_hz": hz,
            "horizon_steps": steps_limit,
            "real_time_factor": round(sim_time / max(time.perf_counter() - started, 1e-12), 6),
            "episode_wall_timeout_sec": episode_timeout,
            "wall_clock_timeout": wall_clock_timeout,
            "timeout_scope": "episode" if wall_clock_timeout else None,
        })
        if robot_id in ALL_ROBOT_IDS or robot_id in rpent_embodiments:
            metrics['camera_contract_version'] = CAMERA_CONTRACT_VERSION
            metrics['camera_ids'] = list(policy_camera_ids(robot_id)) if camera_observations else []
            metrics['observation_mode'] = 'rgb' if camera_observations else 'state_only_smoke'
        scenario = getattr(env, "scenario", None)
        if scenario is not None:
            metrics.update({"scenario_id": scenario.scenario_id, "scenario_digest": scenario.digest})
        if error_message is not None:
            metrics["error"] = error_message
            metrics["passed"] = False
        if hasattr(policy, "end_episode"):
            _call_with_supported_kwargs(policy.end_episode, {
                "observation": observation, "info": policy_info(last_info, policy, env),
                "reward": last_reward, "terminated": terminated,
                "truncated": truncated or wall_clock_timeout or (not terminated and tracker.steps >= steps_limit),
                "passed": metrics["passed"], "episode_score": metrics.get("episode_score"),
            })
        return EvaluationResult(
            task_id=spec.task_id,
            task_version=spec.task_version,
            robot_id=robot_id,
            policy_id=policy_id or getattr(policy, "policy_id", policy.__class__.__name__),
            seed=seed,
            metrics=metrics,
            wall_time_sec=time.perf_counter() - started,
            ended_early=ended_early,
        )


def _reset_env(env: Any, kwargs: dict[str, Any]) -> tuple[Any, Mapping[str, Any]]:
    """Reset old robosuite and Gymnasium environments without masking errors."""

    try:
        return normalize_reset(env.reset(**kwargs))
    except TypeError as exc:
        # Legacy robosuite reset() has no seed parameter. Only retry for a
        # signature mismatch; errors raised inside reset must propagate.
        try:
            signature = inspect.signature(env.reset)
        except (TypeError, ValueError):
            raise
        if kwargs and "seed" not in signature.parameters:
            return normalize_reset(env.reset())
        raise exc


def _can_reset_for_horizon(steps_limit: int, step_index: int) -> bool:
    # A reset at the final tick would not contribute any simulation time and
    # can make task traces look like an extra episode. Keep the horizon exact.
    return step_index + 1 < steps_limit

    def evaluate_many(
        self,
        task: TaskDefinition | TaskSpec,
        policy_factory: Callable[[], Any],
        *,
        factory: Callable[..., Any] | None = None,
        episodes: int = 3,
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        base_seed = kwargs.pop("seed", 0)
        return [
            self.evaluate(
                task,
                factory=factory,
                policy=policy_factory(),
                seed=None if base_seed is None else base_seed + index,
                **kwargs,
            )
            for index in range(episodes)
        ]


def _success_state(info: Mapping[str, Any], reward: float) -> bool | None:
    if "success_state" in info:
        return bool(info["success_state"])
    if "success" in info:
        return bool(info["success"])
    # Legacy robosuite tasks generally expose success through positive reward.
    return bool(reward > 0.0)


def _events(info: Mapping[str, Any]) -> list[Any]:
    events = info.get("events", info.get("event", ()))
    if events is None:
        return []
    if isinstance(events, (str, bytes)):
        return [events]
    if isinstance(events, int):
        return [None] * max(0, events)
    try:
        return list(events)
    except TypeError:
        return [events]


def _action_dim(env: Any) -> int:
    """Read action dimension from robosuite or Gymnasium-style environments."""

    value = getattr(env, "action_dim", None)
    if value is not None:
        return int(value)
    space = getattr(env, "action_space", None)
    shape = getattr(space, "shape", None)
    if shape:
        size = 1
        for dimension in shape:
            size *= int(dimension)
        return size
    if hasattr(space, "n"):
        return int(space.n)
    raise AttributeError("environment must expose action_dim or action_space.shape/n")


def _render_env(env: Any) -> None:
    """Refresh a native viewer when interactive rendering is requested."""

    # robosuite's ``mjviewer`` renderer implements ``render()`` as a no-op;
    # the actual window creation and frame presentation happen in
    # ``MjviewerRenderer.update()``.  Prefer that method when available.
    viewer = getattr(env, "viewer", None)
    update_fn = getattr(viewer, "update", None)
    if callable(update_fn):
        update_fn()
        return
    render_fn = getattr(env, "render", None)
    if not callable(render_fn):
        return
    try:
        render_fn()
    except TypeError:
        # Gymnasium environments may require an explicit mode when their
        # render method was not configured at construction time.
        render_fn(mode="human")
