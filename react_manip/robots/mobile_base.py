"""Shared wheel-drive helpers for the custom Mobile Dual PiPER-X asset."""
from __future__ import annotations

import numpy as np


def set_wheel_command(env, vx: float, wz: float) -> None:
    """Write normalized differential-drive wheel targets into ``data.ctrl``.

    This only sets actuator targets; MuJoCo integrates wheel/body dynamics on
    the subsequent physical step.  Tasks that expose ``set_mobile_base_command``
    may keep their own wrapper, while generic frontends can use this helper.
    """
    if not np.all(np.isfinite([vx, wz])):
        raise ValueError("mobile command must be finite")
    robot = getattr(env, "robots", ())[0]
    if getattr(robot, "name", "") not in ("MobileDualPiperX", "AgileXMobileAloha"):
        return
    model, data = env.sim.model, env.sim.data
    prefix = robot.robot_model.naming_prefix
    values = (float(np.clip(vx, -1, 1)), float(np.clip(wz, -1, 1)))
    try:
        radius_id = model.geom_name2id(prefix + "mobile_left_wheel_link_tire")
        radius = float(model.geom_size[radius_id, 0])
    except (ValueError, KeyError, IndexError):
        return
    for side, sign in (("left", -1.0), ("right", 1.0)):
        try:
            actuator = model.actuator_name2id(prefix + "mobile_" + side + "_wheel_motor")
        except (ValueError, KeyError):
            continue
        target = (0.25 * values[0] + sign * 0.5 * values[1] * 0.17) / radius
        data.ctrl[actuator] = float(np.clip(target, -6, 6))
