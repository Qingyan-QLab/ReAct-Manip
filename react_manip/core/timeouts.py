"""Wall-clock limits shared by all evaluator entry points."""

from __future__ import annotations

import math


# These are infrastructure limits, distinct from a task's simulation horizon.
# They can be overridden by callers/CLI when a policy or machine needs a
# different budget.
DEFAULT_EPISODE_WALL_TIMEOUT_SEC = 30.0 * 60.0
DEFAULT_TASK_WALL_TIMEOUT_SEC = 60.0 * 60.0
TASK_WALL_TIMEOUT_OVERRIDES_SEC = {
    "conveyor_grasp": 2.0 * 60.0 * 60.0,
    "find_object_in_drawer2": 4.0 * 60.0 * 60.0,
}

# Distinguish an omitted task timeout from an explicit ``None``, which keeps
# the existing Python API behavior of disabling the task-level deadline.
USE_DEFAULT_TASK_WALL_TIMEOUT = object()


def validate_timeout(value: float | None, name: str) -> float | None:
    """Validate an optional positive wall-clock timeout."""

    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def default_task_wall_timeout_sec(task_id: str) -> float:
    """Return the release default for all episodes belonging to one task."""

    return TASK_WALL_TIMEOUT_OVERRIDES_SEC.get(task_id, DEFAULT_TASK_WALL_TIMEOUT_SEC)


def resolve_task_wall_timeout_sec(task_id: str, configured=USE_DEFAULT_TASK_WALL_TIMEOUT) -> float | None:
    """Resolve an omitted, explicitly disabled, or explicitly overridden limit."""

    value = default_task_wall_timeout_sec(task_id) if configured is USE_DEFAULT_TASK_WALL_TIMEOUT else configured
    return validate_timeout(value, "task_wall_timeout_sec")
