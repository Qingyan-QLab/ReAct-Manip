"""Default robot metadata registrations.

Calling ``register_default_robots`` is optional.  It is deliberately separate
from imports of robosuite so registry tooling also works in a lightweight
development environment.
"""

from __future__ import annotations

from .core.registry import get_robot_registry
from .robots.adapters import RobosuiteRobotAdapter
from .robots.agilex_adapter import AgileXAdapter


ROBOSUITE_NAMES = (
    "Panda", "UR5e", "XArm7", "Kinova3", "IIWA", "Sawyer", "Jaco", "Tiago",
    "SpotWithArm", "Baxter", "GR1ArmsOnly", "PandaOmron",
)


def register_default_robots(*, replace: bool = False) -> None:
    from .optional_integrations import import_optional_integration

    rpent = import_optional_integration("integrations.rpent_embodiments")
    if rpent is not None:
        rpent.register()
    registry = get_robot_registry()
    for name in ROBOSUITE_NAMES:
        key = name.lower()
        if key not in {item[0] for item in registry.list()} or replace:
            registry.register(key, RobosuiteRobotAdapter(name), replace=replace)
    for robot_id in ("piperx", "dual_piperx", "mobile_dual_piperx", "legacy_piperx", "aloha", "aloha_mobile"):
        if robot_id not in {item[0] for item in registry.list()} or replace:
            registry.register(robot_id, AgileXAdapter(robot_id), replace=replace)
