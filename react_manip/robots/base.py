"""Embodiment-neutral robot metadata and adapter contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    arm_names: tuple[str, ...] = ("right",)
    arm_dof: tuple[int, ...] = ()
    gripper_type: str | None = None
    control_modes: tuple[str, ...] = ("joint_position",)
    action_dim: int | None = None
    eef_sites: tuple[str, ...] = ()
    model_path: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.robot_id or any(ch.isspace() for ch in self.robot_id):
            raise ValueError("robot_id must be a non-empty, whitespace-free string")
        if not self.arm_names and 'task_level_proxy' not in self.capabilities:
            raise ValueError("arm_names must contain at least one arm")
        if self.arm_dof and len(self.arm_dof) != len(self.arm_names):
            raise ValueError("arm_dof must have one entry per arm")
        if self.action_dim is not None and self.action_dim < 1:
            raise ValueError("action_dim must be positive")

    @property
    def required_arms(self) -> int:
        return len(self.arm_names)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RobotSpec":
        arm_names = value.get("arm_names", ("right",))
        if isinstance(arm_names, str):
            arm_names = (arm_names,)
        control_modes = value.get("control_modes", ("joint_position",))
        if isinstance(control_modes, str):
            control_modes = (control_modes,)
        eef_sites = value.get("eef_sites", ())
        if isinstance(eef_sites, str):
            eef_sites = (eef_sites,)
        arm_dof = value.get("arm_dof", ())
        if isinstance(arm_dof, int):
            arm_dof = (arm_dof,)
        return cls(
            robot_id=str(value["robot_id"]),
            arm_names=tuple(arm_names),
            arm_dof=tuple(int(v) for v in arm_dof),
            gripper_type=value.get("gripper_type"),
            control_modes=tuple(control_modes),
            action_dim=None if value.get("action_dim") is None else int(value["action_dim"]),
            eef_sites=tuple(eef_sites),
            model_path=value.get("model_path"),
            capabilities=frozenset(value.get("capabilities", ())),
            metadata=dict(value.get("metadata", {})),
        )


class RobotAdapter(Protocol):
    """Protocol implemented by a robot embodiment integration."""

    spec: RobotSpec

    def validate_assets(self) -> None:
        ...

    def make(self, **kwargs: Any) -> Any:
        ...

    def load_mujoco_model(self) -> Any:
        ...


class ModelBackedAdapter:
    """Common asset validation for custom MJCF-backed robots."""

    spec: RobotSpec

    def __init__(self, spec: RobotSpec):
        self.spec = spec

    @property
    def model_path(self) -> Path | None:
        return Path(self.spec.model_path).expanduser() if self.spec.model_path else None

    def validate_assets(self) -> None:
        path = self.model_path
        if path is None:
            raise FileNotFoundError(
                f"{self.spec.robot_id} has no model_path; provide an MJCF asset before simulation"
            )
        if not path.is_file():
            raise FileNotFoundError(f"robot model does not exist: {path}")

    def load_mujoco_model(self) -> Any:
        """Load the configured MJCF for lightweight asset/control smoke tests."""

        self.validate_assets()
        import mujoco  # imported only when a simulator is actually requested

        return mujoco.MjModel.from_xml_path(str(self.model_path))

    def make(self, **kwargs: Any) -> Any:
        """Return integration-specific robot object.

        Subclasses should implement this once the exact robosuite/MuJoCo
        model is available.  Keeping the method explicit prevents a missing
        asset from silently falling back to Panda.
        """

        del kwargs
        self.validate_assets()
        raise NotImplementedError(f"{self.spec.robot_id} model loader is not configured")
