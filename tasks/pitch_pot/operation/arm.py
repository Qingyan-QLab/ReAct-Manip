"""Read-only, incremental pick/throw baseline. All motion goes through env.step.

This is a privileged geometric baseline, not a learned visual policy. Its
trajectory is deliberately explicit so timings can be calibrated physically.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ThrowMotion:
    swing_duration: float = 0.78
    open_at: float = 0.36
    swing_dx: float = 0.40
    swing_dz: float = 0.10
    lift_x: float = 0.38
    lift_z: float = 0.85
    swing_pitch: float = 0.0
    swing_dy: float = 0.0
    lift_y: float = 0.0


class PickThrowController:
    requires_geometry_truth = True

    def __init__(self, env, *, motion=None, release=True):
        self.env = getattr(env, "env", env)
        if motion is None:
            from .planning import motion_for_target
            motion = motion_for_target(self.env)
        self.settings = self.env.embodiment.get("controller", {})
        if "arm_kp" in self.settings:
            arm = self.env.robots[0].composite_controller.part_controllers[self.env.active_arm]
            arm.kp = np.full(6, float(self.settings["arm_kp"]))
            arm.kd = 2 * np.sqrt(arm.kp)
        for aid in range(self.env.sim.model.nu):
            if 'gripper' in self.env.sim.model.actuator_id2name(aid):
                self.env.sim.model.actuator_gainprm[aid, 0] = 2000.
                self.env.sim.model.actuator_biasprm[aid, 1] = -2000.
        for gid in self.env.gripper_geom_ids:
            self.env.sim.model.geom_friction[gid, 0] = 5.0
            self.env.sim.model.geom_condim[gid] = 6
        self.motion = motion
        self.release = release
        self.reset()

    def reset(self):
        self.stage = "close" if self.settings.get("start_in_gripper", False) else "approach"
        self.started = float(self.env.sim.data.time)
        self.target_ball = self.env.ball_pos()
        self.swing_start = None
        self.follow_target = None
        # Downward fingers; closing direction is world Y, clear of forward throw.
        self.base_orientation = np.array(self.settings.get("orientation", [[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]))
        self.orientation = self.base_orientation.copy()
        if self.settings.get("start_in_gripper", False):
            self.base_orientation = self.env.eef_mat()
            self.orientation = self.base_orientation.copy()
        self.layout = self.env.robots[0].composite_controller._action_split_indexes
        self.park = {arm: (self.env.eef_pos(arm), self.env.eef_mat(arm))
                     for arm in self.env.robots[0].arms if arm != self.env.active_arm}

    def _transition(self, stage):
        self.stage = stage
        self.started = float(self.env.sim.data.time)

    def _pose_action(self, position, gripper, fast=False):
        e = self.env
        limit = e.position_scale if fast else self.settings.get("position_limit", .025)
        error = (np.asarray(position) - e.eef_pos()) * self.settings.get("position_gain", 1.0)
        delta = np.clip(error, -limit, limit) / e.position_scale
        rot = Rotation.from_matrix(self.orientation @ e.eef_mat().T).as_rotvec()
        action = np.zeros(e.action_dim)
        rot_limit = self.settings.get("rotation_limit", e.rotation_scale)
        action[slice(*self.layout[e.active_arm])] = np.r_[
            delta, np.clip(rot, -rot_limit, rot_limit) / e.rotation_scale
        ]
        action[slice(*self.layout[e.active_arm + "_gripper"])] = gripper
        for arm, (p, r) in self.park.items():
            shift = (p-e.eef_pos(arm))/e.position_scale
            turn = Rotation.from_matrix(r@e.eef_mat(arm).T).as_rotvec()/e.rotation_scale
            action[slice(*self.layout[arm])] = np.clip(np.r_[shift, turn], -1, 1)
            action[slice(*self.layout[arm + "_gripper"])] = -1
        return action

    def act(self, observation=None, info=None):
        e = self.env
        elapsed = float(e.sim.data.time) - self.started
        b = self.target_ball
        closed = self.settings.get("closed_command", 1.0)
        if self.stage == "approach":
            target = b + np.array(self.settings.get("grasp_offset", [0, 0, -.0036])) + [0, 0, self.settings.get("approach_height", .1236)]
            if elapsed > 1.0 and np.linalg.norm(e.eef_pos() - target) < self.settings.get("approach_tolerance", .01):
                self._transition("descend")
            return self._pose_action(target, -1)
        if self.stage == "descend":
            target = b + np.array(self.settings.get("grasp_offset", [0, 0, -.0036]))
            if elapsed > 0.6 and np.linalg.norm(e.eef_pos() - target) < self.settings.get("grasp_tolerance", .006):
                self._transition("close")
            return self._pose_action(target, -1)
        if self.stage == "close":
            if elapsed > self.settings.get("close_seconds", 0.6):
                self._transition(("clearance" if self.settings.get("clearance_height") else "lift") if e._grasped else "failed")
            return self._pose_action(b + np.array(self.settings.get("grasp_offset", [0, 0, -.0036])), closed)
        if self.stage == "clearance":
            target = b + np.array(self.settings.get("grasp_offset", [0, 0, -.0036])) + [0, 0, self.settings["clearance_height"]]
            if elapsed > .6 and np.linalg.norm(e.eef_pos() - target) < .012:
                self._transition("lift")
            return self._pose_action(target, closed)
        if self.stage == "lift":
            target = [self.motion.lift_x, self.motion.lift_y, self.motion.lift_z]
            if elapsed > 1.0 and np.linalg.norm(e.eef_pos() - target) < .012:
                self.swing_start = e.eef_pos()
                self._transition("swing")
            return self._pose_action(target, closed)
        if self.stage == "swing":
            m = self.motion
            if e._released:
                self.follow_target = e.eef_pos() + [0.04, 0, 0.01]
                self._transition("follow_through")
                return self._pose_action(self.follow_target, -1 if self.release else closed)
            fraction = min(1.0, elapsed / m.swing_duration)
            target = self.swing_start + fraction * np.array([m.swing_dx, m.swing_dy, m.swing_dz])
            self.orientation = Rotation.from_rotvec([0, -fraction * m.swing_pitch, 0]).as_matrix() @ self.base_orientation
            grip = -1 if self.release and elapsed >= m.open_at else closed
            if elapsed >= m.swing_duration:
                self.follow_target = target
                self._transition("follow_through")
            return self._pose_action(target, grip, fast=True)
        if self.stage == "follow_through":
            return self._pose_action(self.follow_target,
                                     -1 if self.release else closed)
        return self._pose_action(e.eef_pos(), -1)
