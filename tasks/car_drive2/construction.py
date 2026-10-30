from react_manip.robots.agilex_robosuite import controller_config, resolve_robot
from tasks.car_drive2.env import CarDrive2


def make_env(seed=0, robot_id="dual_piperx", **kwargs):
    if robot_id not in ("dual_piperx", "aloha", "aloha_agilex_robotwin"):
        raise ValueError("car_drive2 uses the fixed dual PiPER-X; start it with --robot dual_piperx")
    from react_manip.robots.resolution import resolve_robosuite_robot
    resolved=resolve_robosuite_robot(robot_id)
    cfg = resolved.controller_configs
    # The arms are rigidly mounted on the vehicle.  Expressing OSC deltas in
    # the robot base frame keeps keyboard control valid after arbitrary yaw.
    for part in cfg["body_parts"].values():
        part["input_ref_frame"] = "base"
    env=CarDrive2(robots=resolved.robot_name,
                     controller_configs=cfg,
                     gripper_types="default", seed=seed, **kwargs)
    env.robot_id=robot_id
    return env
