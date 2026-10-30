"""Task/robot capability checks shared by CLI and runners."""

from __future__ import annotations

from typing import Any

from .specs import TaskSpec


def resolve_task_robot(spec: TaskSpec, robot_id: str | None = None) -> str | None:
    """Select the explicit task default, or the sole supported embodiment."""
    if robot_id is None:
        if spec.metadata.get('default_robot') is not None:
            robot_id = spec.metadata['default_robot']
        elif len(spec.supported_robots) == 1:
            robot_id = spec.supported_robots[0]
        elif spec.supported_robots:
            raise ValueError(f"task {spec.task_id!r} has multiple supported robots; specify --robot: "
                             + ', '.join(spec.supported_robots))
        else:
            return None  # unrestricted external tasks may own their defaults
    from ..robots.catalog import normalize_robot_id
    robot_id = normalize_robot_id(robot_id)
    validate_task_robot(spec, robot_id)
    return robot_id


def validate_task_robot(spec: TaskSpec, robot_id: str | None, robot: Any | None = None) -> None:
    """Raise a descriptive error when a declared combination is unsupported."""

    if robot_id is None:
        return
    from ..robots.catalog import normalize_robot_id
    normalized = normalize_robot_id(robot_id)
    if not spec.supports_robot(normalized):
        raise ValueError(
            f"task {spec.task_id!r} does not support robot {robot_id!r}; "
            f"supported robots: {', '.join(spec.supported_robots)}"
        )
    if robot is None:
        return
    robot_spec = getattr(robot, "spec", robot)
    capabilities = set(getattr(robot_spec, 'capabilities', ()))
    required = set(spec.metadata.get('required_capabilities', ()))
    if required - capabilities:
        raise ValueError(f'task {spec.task_id!r} requires missing capabilities: {sorted(required - capabilities)}')
    if 'task_level_proxy' in capabilities:
        if spec.metadata.get('embodiment_mode') == 'physical':
            raise ValueError(f'task {spec.task_id!r} requires a physical robot, not a task-level proxy')
        return  # Proxy labels cannot attest to physical arm/gripper capabilities.
    arms = getattr(robot_spec, 'arm_names', getattr(robot_spec, 'arms', None))
    arm_count = len(arms) if arms is not None else None
    if arm_count is not None and arm_count < spec.required_arms:
        raise ValueError(
            f"task {spec.task_id!r} requires {spec.required_arms} arms, "
            f"robot {robot_id!r} provides {arm_count}"
        )
    gripper = spec.required_gripper
    actual_gripper = getattr(robot_spec, "gripper_type", None)
    if gripper and actual_gripper and actual_gripper not in {gripper, "robosuite"}:
        raise ValueError(
            f"task {spec.task_id!r} requires gripper {gripper!r}, "
            f"robot {robot_id!r} provides {actual_gripper!r}"
        )
