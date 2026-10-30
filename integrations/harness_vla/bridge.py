"""Rendezvous between blocking primitives and the benchmark-owned step loop.

Only the runner advances physics. Mid-episode rewards are unavailable through
Policy.act and are represented as None, never inferred from success feedback.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

import numpy as np


class EpisodeEnded(Exception):
    """The benchmark ended or cancelled the episode."""


@dataclass(frozen=True)
class Transition:
    observation: Any
    info: dict
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False


class PrimitiveEnv:
    """Public observation proxy; has no reference to a simulator or instance file.

    reset() only reads the initial observation, once. chunk_step() yields every
    action separately and returns a shorter chunk when the benchmark terminates.
    A session must use bounded external requests and check cancelled during work.
    """

    def __init__(self, initial: Transition, action_dim: int):
        self.current = initial
        self.action_dim = action_dim
        self.cancelled = threading.Event()
        self._condition = threading.Condition()
        self._request = None
        self._response = None
        self._inflight = False
        self._reset_read = False
        self._done = False
        self._error = None

    def reset(self):
        if self._reset_read or self._inflight:
            raise RuntimeError('Only the benchmark may reset an episode')
        self._reset_read = True
        return self.current.observation, self.current.info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Primitive action must be finite, normalized, and match action_dim')
        with self._condition:
            if self.cancelled.is_set() or self.current.terminated or self.current.truncated:
                raise EpisodeEnded('Episode has ended')
            if self._inflight:
                raise RuntimeError('Concurrent primitive steps are not supported')
            self._reset_read = True
            self._inflight = True
            self._request = action.copy()
            self._condition.notify_all()
            self._condition.wait_for(lambda: self._response is not None or self.cancelled.is_set())
            if self._response is None:
                raise EpisodeEnded('Episode cancelled')
            result, self._response = self._response, None
            self.current = result
            self._inflight = False
            self._condition.notify_all()
            return result.observation, result.reward, result.terminated, result.truncated, result.info

    def chunk_step(self, actions, *, return_all_frames=False):
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != self.action_dim:
            raise ValueError('Expected nonempty [steps, action_dim] chunk')
        observations, rewards, terms, truncs, infos = [], [], [], [], []
        for action in actions:
            obs, reward, term, trunc, info = self.step(action)
            observations.append(obs); rewards.append(reward)
            terms.append(term); truncs.append(trunc); infos.append(info)
            if term or trunc:
                break
        return (observations if return_all_frames else observations[-1], rewards,
                np.asarray(terms), np.asarray(truncs), infos)

    def cancel(self):
        self.cancelled.set()
        with self._condition:
            self._condition.notify_all()


class HarnessVLAPolicy:
    """Run one blocking session per episode, retaining instance-local memory.

    session(env, context, memory) must return only when finished and cooperate
    with env.cancelled. An early return is an integration error: it must not
    manufacture a benchmark success or silently replace the policy with zeros.
    """
    policy_id = 'harness-vla'
    requires_geometry_truth = False

    def __init__(self, session: Callable, *, action_timeout_s=1200.0, close_timeout_s=5.0):
        for value in (action_timeout_s, close_timeout_s):
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Timeouts must be finite and positive')
        self.session = session
        self.action_timeout_s = action_timeout_s
        self.close_timeout_s = close_timeout_s
        self.memory = {}
        self.feedback = []
        self._env = None
        self._worker = None
        self._pending = False
        self._context = None
        self.action_dim = None

    def set_action_dim(self, action_dim):
        if type(action_dim) is not int or action_dim <= 0:
            raise ValueError('action_dim must be a positive integer')
        self.action_dim = action_dim

    def begin_episode(self, *, phase, episode_index, seed=None, task_spec=None):
        self.close()
        if phase not in ('practice', 'exam'):
            raise ValueError('Unknown episode phase')
        self._context = dict(phase=phase, episode_index=episode_index, seed=seed, task_spec=task_spec)
        self._pending = False

    def reset(self, *, seed=None, task_spec=None, **kwargs):
        self.begin_episode(phase='practice', episode_index=0, seed=seed, task_spec=task_spec)

    def _run(self, env):
        try:
            self.session(env, deepcopy(self._context), self.memory)
        except EpisodeEnded as exc:
            if not env.cancelled.is_set() and not (env.current.terminated or env.current.truncated):
                env._error = exc
        except BaseException as exc:
            env._error = exc
        finally:
            with env._condition:
                env._done = True
                env._condition.notify_all()

    def act(self, observation, info):
        if self.action_dim is None or self._context is None:
            raise RuntimeError('Policy episode has not been initialized')
        transition = Transition(deepcopy(observation), deepcopy(info))
        if self._env is None:
            self._env = PrimitiveEnv(transition, self.action_dim)
            self._worker = threading.Thread(target=self._run, args=(self._env,), daemon=True,
                                            name='harness-vla-session')
            self._worker.start()
        env = self._env
        deadline = time.monotonic() + self.action_timeout_s
        try:
            with env._condition:
                if self._pending:
                    env._response = transition
                    self._pending = False
                    env._condition.notify_all()
                while env._request is None:
                    if env._error is not None:
                        error, env._error = env._error, None
                        raise RuntimeError('HarnessVLA session failed') from error
                    if env._done or env.cancelled.is_set():
                        raise RuntimeError('HarnessVLA session ended before benchmark termination')
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Timed out waiting for HarnessVLA action')
                    env._condition.wait(remaining)
                action, env._request = env._request, None
                self._pending = True
                return action
        except BaseException:
            env.cancel()
            raise

    def end_episode(self, **feedback):
        summary = {key: deepcopy(value) for key, value in feedback.items() if key != 'observation'}
        self.feedback.append(summary)
        self.memory['last_episode_feedback'] = summary
        env = self._env
        try:
            if env is not None:
                with env._condition:
                    if self._pending:
                        env._response = Transition(
                            deepcopy(feedback.get('observation')), deepcopy(feedback.get('info', {})),
                            feedback.get('reward'), bool(feedback.get('terminated')),
                            bool(feedback.get('truncated')))
                        self._pending = False
                    env.cancelled.set()
                    env._condition.notify_all()
            self.close()
        finally:
            self._context = None

    def close(self):
        if self._env is not None:
            self._env.cancel()
        if self._worker is not None:
            self._worker.join(self.close_timeout_s)
            if self._worker.is_alive():
                raise TimeoutError('HarnessVLA session did not cooperate with cancellation')
        error = self._env._error if self._env is not None else None
        self._worker = None
        self._env = None
        if error is not None:
            raise RuntimeError('HarnessVLA session failed during cleanup') from error
