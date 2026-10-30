"""Built-in and custom robot embodiment adapters."""

from __future__ import annotations

from typing import Any, Mapping

from .base import ModelBackedAdapter, RobotSpec


class RobosuiteRobotAdapter:
    """Metadata adapter for a robot already supported by robosuite."""

    def __init__(self, robot_name: str):
        self.robot_name = robot_name
        bimanual = robot_name in {"Baxter", "GR1ArmsOnly"}
        self.spec = RobotSpec(
            robot_id=robot_name.lower(),
            arm_names=("left", "right") if bimanual else ("right",),
            gripper_type="robosuite",
            metadata={"robosuite_name": robot_name},
        )

    def validate_assets(self) -> None:
        # The robosuite installation performs the authoritative asset check.
        return None

    def make(self, **kwargs: Any) -> str:
        del kwargs
        return self.robot_name

    def load_mujoco_model(self) -> Any:
        raise NotImplementedError("robosuite robots are loaded through suite.make")


class PiperXAdapter(ModelBackedAdapter):
    """PiperX single-arm adapter.

    ``spec`` is configurable because Piper/PiperX hardware revisions and
    grippers may differ.  No default joint list is guessed here; the adapter
    must be constructed with the exact MJCF and calibrated metadata.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        arm_dof: int | None = None,
        action_dim: int | None = None,
        eef_site: str = "piperx_gripper_site",
        gripper_type: str = "parallel",
        metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            RobotSpec(
                robot_id="piperx",
                arm_names=("right",),
                arm_dof=() if arm_dof is None else (int(arm_dof),),
                gripper_type=gripper_type,
                action_dim=action_dim,
                eef_sites=(eef_site,),
                model_path=model_path,
                capabilities=frozenset({"single_arm", "parallel_gripper", "custom_mjcf"}),
                metadata=dict(metadata or {}),
            )
        )


class AlohaAdapter(ModelBackedAdapter):
    """Adapter for a split, fixed-base bimanual ALOHA embodiment.

    The two arm model paths may point to one combined MJCF or two independent
    arm MJCF files.  A later loader can choose the representation without
    changing task or policy APIs.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        left_model_path: str | None = None,
        right_model_path: str | None = None,
        arm_dof: int | None = None,
        gripper_type: str = "parallel",
        eef_sites: tuple[str, str] = ("aloha_left_gripper_site", "aloha_right_gripper_site"),
        metadata: Mapping[str, Any] | None = None,
    ):
        paths = {"left_model_path": left_model_path, "right_model_path": right_model_path}
        super().__init__(
            RobotSpec(
                robot_id="aloha",
                arm_names=("left", "right"),
                arm_dof=() if arm_dof is None else (int(arm_dof), int(arm_dof)),
                gripper_type=gripper_type,
                eef_sites=eef_sites,
                model_path=model_path,
                capabilities=frozenset({"bimanual", "parallel_gripper", "custom_mjcf"}),
                metadata={**paths, **dict(metadata or {})},
            )
        )

    def validate_assets(self) -> None:
        if self.model_path is not None:
            return super().validate_assets()
        missing = []
        for key in ("left_model_path", "right_model_path"):
            value = self.spec.metadata.get(key)
            if not value:
                missing.append(key)
            else:
                from pathlib import Path

                if not Path(str(value)).expanduser().is_file():
                    raise FileNotFoundError(f"robot model does not exist: {value}")
        if missing:
            raise FileNotFoundError(
                "aloha requires model_path or both left_model_path/right_model_path; missing "
                + ", ".join(missing)
            )

    def load_mujoco_model(self) -> Any:
        """Load one combined model or both split arm models.

        A split return value is a mapping so a higher-level bimanual scene
        builder can place and namespace the two arms explicitly.
        """

        if self.model_path is not None:
            return super().load_mujoco_model()
        self.validate_assets()
        import mujoco

        return {
            "left": mujoco.MjModel.from_xml_path(str(self.spec.metadata["left_model_path"])),
            "right": mujoco.MjModel.from_xml_path(str(self.spec.metadata["right_model_path"])),
        }
