"""Machine-checkable observation, action and feedback declarations."""
from dataclasses import dataclass, field
from typing import Any, Mapping

@dataclass(frozen=True)
class ObservationSpec:
    modalities: tuple[str, ...] = ("proprio",)
    default_policy_view: tuple[str, ...] = ("proprio",)
    geometry_view: str = "internal"
    fields: Mapping[str, Any] = field(default_factory=dict)
    visibility: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class ActionSpec:
    type: str = "vector"
    shape: tuple[int, ...] = ()
    bounds: tuple[float, float] = (-1.0, 1.0)
    frame: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FeedbackSpec:
    fields: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[str, ...] = ()
    success_signals: tuple[str, ...] = ("success_state",)

def parse_contracts(data: Mapping[str, Any]):
    obs=data.get("observation", {}); act=data.get("action", {}); fb=data.get("feedback", {})
    fields=lambda x: x if isinstance(x, Mapping) else {str(v): {} for v in x}
    return (ObservationSpec(tuple(obs.get("modalities", ("proprio",))), tuple(obs.get("default_policy_view", obs.get("modalities", ("proprio",)))), str(obs.get("geometry_view", "internal")), fields(obs.get("fields", {})), dict(obs.get("visibility", {}))),
            ActionSpec(str(act.get("type", "vector")), tuple(act.get("shape", ())), tuple(act.get("bounds", (-1, 1))), act.get("frame"), fields(act.get("fields", {}))),
            FeedbackSpec(fields(fb.get("fields", {})), tuple(fb.get("events", ())), tuple(fb.get("success_signals", ("success_state",)))))

# Canonical field vocabulary for tasks whose legacy JSON has not yet been
# extended.  Task-specific specs may override these entries incrementally.
TASK_CONTRACT_FIELDS = {
    "conveyor_grasp": ("object.pose", "object.velocity", "belt.speed", "grasp_state"),
    "paddle_juggle": ("ball.pose", "ball.velocity", "paddle.pose", "bounce_count"),
    "pitch_pot": ("projectile.pose", "projectile.velocity", "pot.pose", "release_state"),
    "turntable_gamble": ("turntable.angle", "turntable.angular_velocity", "target_sector", "score"),
    "phone_power_on": ("phone.pose", "cup.pose", "button.pose", "press_state"),
}
