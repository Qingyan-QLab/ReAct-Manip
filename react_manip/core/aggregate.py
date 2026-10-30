"""Aggregate fixed-horizon episode results into benchmark-level statistics."""

from __future__ import annotations

from math import sqrt
from typing import Any, Iterable, Mapping


NUMERIC_FIELDS = (
    "success_occupancy",
    "longest_success_streak_sec",
    "completed_events",
    "event_rate_hz",
    "failure_count",
    "recoverable_failure_count",
    "recovery_count",
    "peak_force_N",
    "mean_inference_latency_ms",
    "mean_action_age_ms",
    "real_time_factor",
)


def aggregate_results(results: Iterable[Any]) -> dict[str, Any]:
    """Return deterministic summary statistics for a set of episodes."""

    rows = [_as_mapping(item) for item in results]
    if not rows:
        raise ValueError("cannot aggregate an empty result set")
    passed = [bool(row.get("passed", row.get("metrics", {}).get("passed", False))) for row in rows]
    summary: dict[str, Any] = {
        "episodes": len(rows),
        "pass_count": sum(passed),
        "pass_rate": round(sum(passed) / len(passed), 6),
    }
    for field in NUMERIC_FIELDS:
        values = []
        for row in rows:
            value = row.get(field)
            if value is None and isinstance(row.get("metrics"), Mapping):
                value = row["metrics"].get(field)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary[f"mean_{field}"] = round(mean, 6)
            summary[f"std_{field}"] = round(sqrt(variance), 6)
    return summary


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError(f"result must be a mapping or expose to_dict(), got {type(value).__name__}")
