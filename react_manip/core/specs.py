"""Versioned task metadata and success criteria.

Task metadata is intentionally data-oriented.  A new task can provide a
``TaskSpec`` in Python or load the same fields from JSON/YAML without changing
the benchmark runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json
from .contracts import ObservationSpec, ActionSpec, FeedbackSpec, parse_contracts


@dataclass(frozen=True)
class SuccessRule:
    """Rule used to turn step-level signals into a benchmark pass/fail.

    ``min_occupancy`` is the fraction of control ticks with a valid success
    state.  ``min_longest_streak_sec`` prevents a policy from alternating
    between success and failure while still achieving a high average.  Event
    tasks use ``min_events`` for their throughput requirement.
    """

    min_occupancy: float = 0.95
    min_longest_streak_sec: float = 0.0
    min_events: int = 0
    max_failures: int | None = 0
    require_terminal_success: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_occupancy <= 1.0:
            raise ValueError("min_occupancy must be in [0, 1]")
        if self.min_longest_streak_sec < 0:
            raise ValueError("min_longest_streak_sec must be non-negative")
        if self.min_events < 0:
            raise ValueError("min_events must be non-negative")
        if self.max_failures is not None and self.max_failures < 0:
            raise ValueError("max_failures must be non-negative or None")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SuccessRule":
        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            min_occupancy=float(value.get("min_occupancy", 0.95)),
            min_longest_streak_sec=float(value.get("min_longest_streak_sec", 0.0)),
            min_events=int(value.get("min_events", 0)),
            max_failures=(
                None
                if value.get("max_failures", 0) is None
                else int(value.get("max_failures", 0))
            ),
            require_terminal_success=bool(value.get("require_terminal_success", False)),
        )


@dataclass(frozen=True)
class TaskSpec:
    """Stable metadata contract for a benchmark task."""

    task_id: str
    task_version: str = "1.0"
    api_version: str = "2.0"
    task_type: str = "continuous"  # continuous | cyclic | event | one_shot
    required_arms: int = 1
    required_gripper: str | None = None
    required_eef_dof: int | None = None
    supported_robots: tuple[str, ...] = ()
    horizon_sec: float = 60.0
    control_hz: float = 20.0
    done_behavior: str = "stop"  # stop | reset | fail_and_stop | fail_and_reset
    success: SuccessRule = field(default_factory=SuccessRule)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    action: ActionSpec = field(default_factory=ActionSpec)
    feedback: FeedbackSpec = field(default_factory=FeedbackSpec)
    evaluation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.success, SuccessRule):
            object.__setattr__(self, "success", SuccessRule.from_mapping(self.success))
        for key, cls in (("observation", ObservationSpec), ("action", ActionSpec), ("feedback", FeedbackSpec)):
            if not isinstance(getattr(self, key), cls): object.__setattr__(self, key, parse_contracts({key: getattr(self, key)})[("observation", "action", "feedback").index(key)])
        if not self.task_id or any(ch.isspace() for ch in self.task_id):
            raise ValueError("task_id must be a non-empty, whitespace-free string")
        if self.task_type not in {"continuous", "cyclic", "event", "one_shot"}:
            raise ValueError(f"unsupported task_type: {self.task_type}")
        # Native task-level environments may intentionally have no robot
        # embodiment (for example a scene-level curling actuator). Robot
        # operation tasks still declare one or more arms explicitly.
        if self.required_arms < 0:
            raise ValueError("required_arms must be >= 0")
        if self.required_eef_dof is not None and self.required_eef_dof < 1:
            raise ValueError("required_eef_dof must be >= 1")
        if self.horizon_sec <= 0 or self.control_hz <= 0:
            raise ValueError("horizon_sec and control_hz must be positive")
        if self.done_behavior not in {"stop", "reset", "fail_and_stop", "fail_and_reset"}:
            raise ValueError(f"unsupported done_behavior: {self.done_behavior}")

    @property
    def horizon_steps(self) -> int:
        return max(1, int(round(self.horizon_sec * self.control_hz)))

    @property
    def supports_all_robots(self) -> bool:
        """Whether the task accepts the canonical robot IDs without narrowing.

        Empty ``supported_robots`` is an intentional wildcard for task-level
        scenes.  It means the selected embodiment is recorded as context; it
        does not imply that an arm model is physically compiled into the scene.
        """
        return not self.supported_robots

    def supports_robot(self, robot_id: str) -> bool:
        return self.supports_all_robots or str(robot_id).lower() in {
            item.lower() for item in self.supported_robots
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskSpec":
        data = dict(value)
        supported = data.get("supported_robots", ())
        if isinstance(supported, str):
            supported = (supported,)
        return cls(
            task_id=str(data["task_id"]),
            task_version=str(data.get("task_version", "1.0")),
            api_version=str(data.get("api_version", "2.0")),
            task_type=str(data.get("task_type", "continuous")),
            required_arms=int(data.get("required_arms", 1)),
            required_gripper=data.get("required_gripper"),
            required_eef_dof=(
                None if data.get("required_eef_dof") is None else int(data["required_eef_dof"])
            ),
            supported_robots=tuple(str(item) for item in supported),
            horizon_sec=float(data.get("horizon_sec", 60.0)),
            control_hz=float(data.get("control_hz", 20.0)),
            done_behavior=str(data.get("done_behavior", "stop")),
            success=SuccessRule.from_mapping(data.get("success")),
            evaluation=dict(data.get("evaluation", {})),
            metadata=dict(data.get("metadata", {})),
            observation=parse_contracts(data)[0], action=parse_contracts(data)[1], feedback=parse_contracts(data)[2],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "task_type": self.task_type,
            "required_arms": self.required_arms,
            "required_gripper": self.required_gripper,
            "required_eef_dof": self.required_eef_dof,
            "supported_robots": list(self.supported_robots),
            "horizon_sec": self.horizon_sec,
            "control_hz": self.control_hz,
            "done_behavior": self.done_behavior,
            "success": {
                "min_occupancy": self.success.min_occupancy,
                "min_longest_streak_sec": self.success.min_longest_streak_sec,
                "min_events": self.success.min_events,
                "max_failures": self.success.max_failures,
                "require_terminal_success": self.success.require_terminal_success,
            },
            "evaluation": dict(self.evaluation),
            "metadata": dict(self.metadata),
            "observation": {"modalities": list(self.observation.modalities), "default_policy_view": list(self.observation.default_policy_view), "geometry_view": self.observation.geometry_view, "fields": dict(self.observation.fields), "visibility": dict(self.observation.visibility)},
            "action": {"type": self.action.type, "shape": list(self.action.shape), "bounds": list(self.action.bounds), "frame": self.action.frame, "fields": dict(self.action.fields)},
            "feedback": {"fields": dict(self.feedback.fields), "events": list(self.feedback.events), "success_signals": list(self.feedback.success_signals)},
        }


def load_task_spec(path: str | Path) -> TaskSpec:
    """Load a JSON or YAML task specification.

    YAML is optional so the core package remains dependency-light.  A clear
    error is raised if a YAML file is used without PyYAML installed.
    """

    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("Loading YAML specs requires PyYAML; use JSON instead") from exc
        data = yaml.safe_load(raw)
    if not isinstance(data, Mapping):
        raise ValueError(f"task spec must be a mapping: {path}")
    data = dict(data)
    evaluation = data.get("evaluation", {})
    if "config" in evaluation:
        if set(evaluation) != {"config"}:
            raise ValueError("evaluation.config cannot be combined with inline settings")
        config_path = path.parent / evaluation["config"]
        data["evaluation"] = json.loads(config_path.read_text(encoding="utf-8"))
    return TaskSpec.from_mapping(data)
