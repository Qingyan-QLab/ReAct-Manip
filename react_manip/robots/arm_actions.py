"""Build multi-arm actions using controller-advertised slices, not 7D chunks."""
import numpy as np


def arm_action(action_dim, layout, active_arm, pose, grippers):
    action = np.zeros(action_dim, dtype=np.float32)
    start, end = layout[active_arm]
    if end - start != 6:
        raise ValueError("keyboard pose control requires six Cartesian inputs per arm")
    action[start:end] = np.clip(np.asarray(pose), -1, 1)
    for arm, value in grippers.items():
        start, end = layout[arm + "_gripper"]
        if end - start != 1:
            raise ValueError("keyboard gripper control requires one input per gripper")
        action[start:end] = np.clip(value, -1, 1)
    return action

