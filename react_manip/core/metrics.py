"""Step-level tracking for fixed-horizon, sustained-success evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

from .specs import SuccessRule


def _failure_flags(failure: Any) -> tuple[bool, bool]:
    if failure is None or failure is False:
        return False, False
    if isinstance(failure, Mapping):
        return True, bool(failure.get("recoverable", False))
    return True, False


@dataclass
class SustainedSuccessTracker:
    """Collect continuous-state and discrete-event success signals."""

    control_hz: float = 20.0
    steps: int = 0
    state_steps: int = 0
    success_steps: int = 0
    longest_streak_steps: int = 0
    current_streak_steps: int = 0
    completed_events: int = 0
    failure_count: int = 0
    recoverable_failure_count: int = 0
    recovery_count: int = 0
    time_to_first_failure_sec: float | None = None
    recovery_latency_sec: list[float] = field(default_factory=list)
    max_scalars: dict[str, float] = field(default_factory=dict)
    sum_scalars: dict[str, float] = field(default_factory=dict)
    count_scalars: dict[str, int] = field(default_factory=dict)
    last_scalars: dict[str, float] = field(default_factory=dict)
    _failure_started_at: float | None = field(default=None, init=False, repr=False)
    _last_failure: bool = field(default=False, init=False, repr=False)
    _last_info: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False)

    def update(
        self,
        success_state: bool | None,
        *,
        events: Iterable[Any] = (),
        failure: Any = None,
        dt: float | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """Add one control tick to the trace.

        ``success_state=None`` means the task did not expose a continuous
        predicate for this tick. It is conservatively counted as a non-success
        tick, preventing missing signals from inflating occupancy.
        """

        dt = float(dt if dt is not None else 1.0 / self.control_hz)
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.steps += 1
        self.state_steps += 1
        if success_state is True:
            self.success_steps += 1
            self.current_streak_steps += 1
            self.longest_streak_steps = max(self.longest_streak_steps, self.current_streak_steps)
        elif success_state is False:
            self.current_streak_steps = 0
        else:
            # An unknown predicate cannot be part of a continuous success
            # streak and is conservatively treated as a failed tick.
            self.current_streak_steps = 0

        event_list = list(events)
        self.completed_events += len(event_list)
        failed, recoverable = _failure_flags(failure)
        now = (self.steps - 1) * dt
        if failed and not self._last_failure:
            self.failure_count += 1
            if recoverable:
                self.recoverable_failure_count += 1
            self._failure_started_at = now
            if self.time_to_first_failure_sec is None:
                self.time_to_first_failure_sec = now
        elif self._last_failure and self._failure_started_at is not None:
            self.recovery_count += 1
            self.recovery_latency_sec.append(max(0.0, now - self._failure_started_at))
            self._failure_started_at = None
        self._last_failure = failed
        self._last_info = info or {}
        # Optional, task-independent safety and real-time signals.  Tasks may
        # emit any of these fields without changing the tracker schema.
        for key in (
            "peak_force_N",
            "impulse_Ns",
            "joint_limit_violation",
            "inference_latency_ms",
            "action_age_ms",
            # Optional task scoring signals. Tasks may expose one or more of
            # these without creating a task-specific runner branch.
            "total_score",
            "circle_hold_score",
            "hole_drop_score",
            "time_score",
            "distance_score",
            "attempt_score",
            "distance_travelled_m",
            "distance_fraction",
            "lane_score",
            "max_lane_deviation_m",
            "speed_m_s",
            "insertion_depth_m",
            "max_insertion_depth_m",
            "current_circle_hold_sec",
            "max_circle_hold_sec",
        ):
            value = (info or {}).get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scalar = float(value)
                self.max_scalars[key] = max(self.max_scalars.get(key, float("-inf")), scalar)
                self.sum_scalars[key] = self.sum_scalars.get(key, 0.0) + scalar
                self.count_scalars[key] = self.count_scalars.get(key, 0) + 1
                self.last_scalars[key] = scalar

    def record_failure(self, *, recoverable: bool = False, at_sec: float | None = None) -> None:
        """Record an out-of-band failure without adding a control tick."""

        if not self._last_failure:
            self.failure_count += 1
            if recoverable:
                self.recoverable_failure_count += 1
            if self.time_to_first_failure_sec is None:
                self.time_to_first_failure_sec = (
                    float(at_sec) if at_sec is not None else self.steps / self.control_hz
                )
        self._last_failure = True

    def summarize(self, rule: SuccessRule, *, terminal_success: bool = False) -> dict[str, Any]:
        occupancy = self.success_steps / self.state_steps if self.state_steps else 0.0
        longest = self.longest_streak_steps / self.control_hz
        passed = (
            occupancy >= rule.min_occupancy
            and longest >= rule.min_longest_streak_sec
            and self.completed_events >= rule.min_events
            and (rule.max_failures is None or self.failure_count <= rule.max_failures)
            and (not rule.require_terminal_success or terminal_success)
        )
        metrics = {
            "steps": self.steps,
            "state_steps": self.state_steps,
            "sim_time_sec": self.steps / self.control_hz,
            "success_steps": self.success_steps,
            "success_occupancy": round(occupancy, 6),
            "longest_success_streak_sec": round(longest, 6),
            "completed_events": self.completed_events,
            "event_rate_hz": round(self.completed_events / (self.steps / self.control_hz), 6)
            if self.steps
            else 0.0,
            "failure_count": self.failure_count,
            "recoverable_failure_count": self.recoverable_failure_count,
            "recovery_count": self.recovery_count,
            "time_to_first_failure_sec": self.time_to_first_failure_sec,
            "mean_recovery_latency_sec": round(sum(self.recovery_latency_sec) / len(self.recovery_latency_sec), 6)
            if self.recovery_latency_sec
            else None,
            "terminal_success": bool(terminal_success),
            "passed": bool(passed),
        }
        for key, value in self.max_scalars.items():
            metrics[f"max_{key}"] = round(value, 6)
        if "max_peak_force_N" in metrics:
            metrics["peak_force_N"] = metrics["max_peak_force_N"]
        for key in (
            "circle_hold_score", "hole_drop_score", "attempt_score",
            "distance_score", "distance_travelled_m", "distance_fraction",
            "max_lane_deviation_m", "speed_m_s", "insertion_depth_m",
            "max_insertion_depth_m", "max_circle_hold_sec",
        ):
            max_key = f"max_{key}"
            if max_key in metrics:
                # Scores and a longest-duration measure are monotonic within
                # a task episode, so their maximum is the final audit value.
                metrics[key] = metrics[max_key]
        for key in ("time_score",):
            if key in self.last_scalars:
                metrics[key] = round(self.last_scalars[key], 6)
        # Lane and total scores may decrease after a vehicle leaves its lane.
        for key in ("lane_score", "total_score", "attempt_score"):
            if key in self.last_scalars:
                metrics[key] = round(self.last_scalars[key], 6)
        for key, value in self.sum_scalars.items():
            count = self.count_scalars[key]
            metrics[f"mean_{key}"] = round(value / count, 6) if count else None
        return metrics
