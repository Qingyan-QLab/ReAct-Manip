"""Metadata facade for the generated official PiPER-X and custom dual assemblies."""
from .base import RobotSpec


class AgileXAdapter:
    def __init__(self, robot_id):
        if robot_id not in ("piperx", "dual_piperx", "mobile_dual_piperx", "legacy_piperx", "aloha", "aloha_mobile"):
            raise ValueError(robot_id)
        arms = ("right", "left") if robot_id not in ("piperx", "legacy_piperx") else ("right",)
        self.spec = RobotSpec(
            robot_id=robot_id, arm_names=arms, arm_dof=(6,)*len(arms),
            gripper_type="parallel", action_dim=7*len(arms), control_modes=("osc_pose",),
            eef_sites=tuple(f"gripper0_{arm}_grip_site" for arm in arms),
            capabilities=frozenset({"custom_mjcf", "parallel_gripper", "bimanual" if len(arms)==2 else "single_arm"}),
            metadata={"manufacturer": "AgileX", "arm_model": "legacy follower" if robot_id in ("legacy_piperx", "aloha", "aloha_mobile") else "PiPER-X", "asset_version": "legacy-agilex-family.v1" if robot_id in ("legacy_piperx", "aloha", "aloha_mobile") else "piperx-family.v1", "assembly": "custom" if len(arms) == 2 else "single", "mobile_base": robot_id in ("mobile_dual_piperx", "aloha_mobile"), "base_command_dim": 2 if robot_id in ("mobile_dual_piperx", "aloha_mobile") else 0, "calibrated": False,
                      "action_layout": "right_pose,left_pose,right_gripper,left_gripper" if len(arms)==2 else "right_pose,right_gripper"},
        )

    def validate_assets(self):
        from .agilex_robosuite import validate_bundle
        if self.spec.robot_id in ("legacy_piperx", "aloha", "aloha_mobile"):
            from .legacy_agilex_robosuite import validate_bundle
        validate_bundle()

    def make(self, **kwargs):
        from .agilex_robosuite import resolve_robot
        return resolve_robot(self.spec.robot_id)

    def load_mujoco_model(self):
        from robosuite.models.robots import create_robot
        from robosuite.models.grippers import gripper_factory
        robot = create_robot(self.make())
        for arm in robot.arms:
            robot.add_gripper(gripper_factory(robot.default_gripper[arm], idn=f"0_{arm}"), robot.eef_name[arm])
        return robot.get_model()
