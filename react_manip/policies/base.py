"""Policy adapters kept independent from model frameworks."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping

import numpy as np


class Policy:
    """Small base class for model adapters.

    External models may subclass this class or simply expose compatible
    ``reset`` and ``act`` methods (duck typing is supported by the runner).
    """

    policy_id = "policy"

    def reset(self, *, seed: int | None = None, task_spec: Any = None) -> None:
        del seed, task_spec

    def act(self, observation: Any, info: Mapping[str, Any] | None = None) -> Any:
        raise NotImplementedError


class RandomPolicy(Policy):
    """Reference policy used for smoke tests and legacy comparisons."""

    policy_id = "random"

    def __init__(self, action_dim: int | None = None, seed: int | None = None):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def set_action_dim(self, action_dim: int) -> None:
        self.action_dim = int(action_dim)

    def reset(self, *, seed: int | None = None, task_spec: Any = None) -> None:
        del task_spec
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def begin_episode(self, *, seed=None, **kwargs):
        self.reset(seed=seed)

    def end_episode(self, **feedback):
        pass

    def get_adaptation_state(self):
        # RandomPolicy has no learned state. Its RNG is reseeded per episode.
        return None

    def set_adaptation_state(self, state):
        if state is not None:
            raise ValueError('RandomPolicy has no adaptation state')

    def act(self, observation: Any, info: Mapping[str, Any] | None = None) -> np.ndarray:
        del observation, info
        if self.action_dim is None:
            raise RuntimeError("RandomPolicy action_dim was not set by the runner")
        return self.rng.uniform(-1.0, 1.0, size=self.action_dim)


class CallablePolicy(Policy):
    """Wrap a plain callable while supporting one- or two-argument callables."""

    def __init__(self, fn: Callable[..., Any], policy_id: str | None = None):
        self.fn = fn
        self.policy_id = policy_id or getattr(fn, "__name__", "callable")

    def act(self, observation: Any, info: Mapping[str, Any] | None = None) -> Any:
        try:
            parameters = inspect.signature(self.fn).parameters
            if len(parameters) >= 2:
                return self.fn(observation, info)
        except (TypeError, ValueError):
            pass
        return self.fn(observation)
