"""Task and robot registries used for additive extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .specs import TaskSpec

Factory = Callable[..., Any]


@dataclass(frozen=True)
class TaskDefinition:
    spec: TaskSpec
    factory: Factory


class TaskRegistry:
    def __init__(self) -> None:
        self._items: dict[str, TaskDefinition] = {}

    def register(self, spec: TaskSpec | dict[str, Any], factory: Factory, *, replace: bool = False) -> TaskDefinition:
        spec = spec if isinstance(spec, TaskSpec) else TaskSpec.from_mapping(spec)
        if spec.task_id in self._items and not replace:
            raise KeyError(f"task already registered: {spec.task_id}")
        definition = TaskDefinition(spec, factory)
        self._items[spec.task_id] = definition
        return definition

    def get(self, task_id: str) -> TaskDefinition:
        try:
            return self._items[task_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"unknown task {task_id!r}; registered tasks: {known}") from exc

    def list(self) -> tuple[TaskDefinition, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


class RobotRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def register(self, robot_id: str, adapter_or_factory: Any, *, replace: bool = False) -> Any:
        robot_id = str(robot_id).lower()
        if robot_id in self._items and not replace:
            raise KeyError(f"robot already registered: {robot_id}")
        self._items[robot_id] = adapter_or_factory
        return adapter_or_factory

    def get(self, robot_id: str) -> Any:
        robot_id = str(robot_id).lower()
        if robot_id in ("aloha", "aloha_mobile"):
            from ..robots.catalog import normalize_robot_id
            robot_id = normalize_robot_id(robot_id)
        try:
            return self._items[robot_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"unknown robot {robot_id!r}; registered robots: {known}") from exc

    def list(self) -> tuple[tuple[str, Any], ...]:
        return tuple((key, self._items[key]) for key in sorted(self._items))


_TASKS = TaskRegistry()
_ROBOTS = RobotRegistry()


def get_task_registry() -> TaskRegistry:
    return _TASKS


def get_robot_registry() -> RobotRegistry:
    return _ROBOTS


def register_task(spec: TaskSpec | dict[str, Any], factory: Factory | None = None, **kwargs: Any):
    """Register directly or use as ``@register_task(spec)`` decorator."""

    if factory is not None:
        return _TASKS.register(spec, factory, **kwargs)

    def decorator(fn: Factory) -> Factory:
        _TASKS.register(spec, fn, **kwargs)
        return fn

    return decorator


def register_robot(robot_id: str, adapter_or_factory: Any, **kwargs: Any) -> Any:
    return _ROBOTS.register(robot_id, adapter_or_factory, **kwargs)
