"""Built-in ID snapshots plus live registry-based normalization.

Use registered_robot_ids() for extension-aware enumeration. Task compatibility
is separate from registration and may still narrow the available combinations.
"""

ALL_ROBOT_IDS = ("panda", "sawyer", "piperx", "dual_piperx", "mobile_dual_piperx", "legacy_piperx", "aloha", "aloha_mobile")
SINGLE_ARM_ROBOT_IDS = ("panda", "sawyer", "piperx", "legacy_piperx")


def normalize_robot_id(robot_id: str) -> str:
    from .resolution import registered_robot_ids
    value = str(robot_id).strip().lower()
    if value not in registered_robot_ids():
        raise ValueError(
            f"unknown robot {robot_id!r}; supported IDs: {', '.join(registered_robot_ids())}"
        )
    return value
