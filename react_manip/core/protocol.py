"""Protocols shared by tasks, policies, and the fixed-horizon runner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class StepResult:
    """Normalized result of either Gymnasium (5-tuple) or robosuite (4-tuple)."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@runtime_checkable
class BenchmarkTask(Protocol):
    """Minimum environment interface expected by :class:`BenchmarkRunner`."""

    action_dim: int

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def step(self, action: Any) -> Any:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class Policy(Protocol):
    """Policy interface; model-specific dependencies stay outside the runner."""

    def reset(self, *, seed: int | None = None, task_spec: Any = None) -> None:
        ...

    def act(self, observation: Any, info: Mapping[str, Any] | None = None) -> Any:
        ...


def normalize_reset(value: Any) -> tuple[Any, Mapping[str, Any]]:
    """Normalize old robosuite and Gymnasium reset return conventions."""

    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        return value[0], value[1]
    return value, {}


def normalize_step(value: Sequence[Any]) -> StepResult:
    """Normalize a four- or five-element environment step result."""

    if len(value) == 5:
        obs, reward, terminated, truncated, info = value
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError(f"environment returned non-finite reward: {reward}")
        return StepResult(obs, reward, bool(terminated), bool(truncated), info or {})
    if len(value) == 4:
        obs, reward, done, info = value
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError(f"environment returned non-finite reward: {reward}")
        truncated = bool(done and (info or {}).get('TimeLimit.truncated', False))
        return StepResult(obs, reward, bool(done) and not truncated, truncated, info or {})
    raise ValueError("env.step(action) must return a 4- or 5-element tuple")


class AdaptivePolicy(Policy, Protocol):
    """Optional contract for task-selected practice/exam protocols.

    begin_episode clears transient episode state and sets trial RNG, preserving
    learned adaptation throughout practice AND exams. Snapshot hooks are only
    required by the historical frozen protocol, not online evaluation.
    end_episode receives filtered final feedback, including horizon timeouts.
    """

    def begin_episode(self, *, phase: str, episode_index: int, seed: int, task_spec: Any) -> None: ...
    def end_episode(self, **feedback: Any) -> None: ...
    def get_adaptation_state(self) -> Any: ...
    def set_adaptation_state(self, state: Any) -> None: ...
