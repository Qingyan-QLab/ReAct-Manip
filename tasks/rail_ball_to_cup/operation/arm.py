"""Privileged physical pick/place baseline; only returns arm actions."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..track import RELEASE_RANGE


class PickPlaceController:
    def __init__(self, env, *, release_x=.35, release=True):
        if not RELEASE_RANGE[0] <= release_x <= RELEASE_RANGE[1]:
            raise ValueError(f'release_x must be in {RELEASE_RANGE}')
        self.release_x=release_x
        self.env = getattr(env, 'env', env)
        self.release = release
        # The current PiperX asset has substantially softer default jaw
        # actuators than the legacy demo.  Calibrate them here so the ball is
        # not squeezed out while translating to the rail.
        for aid in range(self.env.sim.model.nu):
            if 'gripper' in self.env.sim.model.actuator_id2name(aid):
                self.env.sim.model.actuator_gainprm[aid, 0] = 2000.
                self.env.sim.model.actuator_biasprm[aid, 1] = -2000.
        self.reset()

    def reset(self):
        self.stage = 'approach'
        self.started = float(self.env.sim.data.time)
        self.target_ball = self.env.ball_pos()
        # Fingers close in X, leaving the gap between the two Y-separated rails free.
        self.settings = self.env.embodiment_config.get('controller', {})
        self.orientation=np.array(self.settings.get('grasp_orientation', np.diag([1.,-1.,-1.])), float)
        self.grasp_offset=np.array(self.settings.get('grasp_offset', [0,0,-.0036]), float)
        self.place_position=self.env.track.ball_position(self.release_x,self.settings.get('place_clearance', .004))
        self.ball_offset=None
        self.open_target=None

    requires_geometry_truth = True

    def _transition(self, stage):
        self.stage = stage
        self.started = float(self.env.sim.data.time)

    def _pose_action(self, position, gripper, fast=False):
        e = self.env
        limit = e.position_scale if fast else self.settings.get('position_limit', .025)
        delta = np.clip((np.asarray(position)-e.eef_pos())*self.settings.get('position_gain',1.), -limit, limit)/e.position_scale
        rot = Rotation.from_matrix(self.orientation@e.eef_mat().T).as_rotvec()
        return np.r_[delta, np.clip(rot,-self.settings.get('rotation_limit',e.rotation_scale),self.settings.get('rotation_limit',e.rotation_scale))/e.rotation_scale, gripper]

    def act(self, observation=None, info=None):
        e=self.env
        elapsed=float(e.sim.data.time)-self.started
        b = self.target_ball
        if self.stage == 'approach':
            target = b-self.orientation@self.grasp_offset + [0,0,.12]
            if elapsed > 1. and np.linalg.norm(e.eef_pos()-target) < .01:
                self._transition('descend')
            return self._pose_action(target,-1)
        if self.stage == 'descend':
            target = b-self.orientation@self.grasp_offset
            if elapsed > .6 and np.linalg.norm(e.eef_pos()-target) < self.settings.get('grasp_tolerance', .006):
                self._transition('close')
            return self._pose_action(target,-1)
        if self.stage == 'close':
            if elapsed > self.settings.get('close_seconds', .6):
                self._transition('lift' if e._grasped else 'failed')
            return self._pose_action(b-self.orientation@self.grasp_offset,1)
        if self.stage=='lift':
            target=self.settings.get('lift_target', [.48,.22,1.12])
            if elapsed>1. and np.linalg.norm(e.eef_pos()-target)<self.settings.get('lift_tolerance', .012):
                self.ball_offset=e.eef_mat().T@(e.ball_pos()-e.eef_pos())
                frame = np.array(self.settings.get('place_orientation', np.diag([1.,-1.,-1.])), float)
                self.orientation=Rotation.from_euler('y',-np.arctan(e.track.slope(self.release_x))).as_matrix()@frame
                self.ball_offset = self.orientation@self.ball_offset
                self._transition('above_track')
            return self._pose_action(target,1)
        if self.stage=='above_track':
            target=self.place_position-self.ball_offset+[0,0,self.settings.get('clearance', .15)]
            if elapsed>1. and np.linalg.norm(e.eef_pos()-target)<.01:
                self._transition('place')
            return self._pose_action(target,1)
        if self.stage=='place':
            target=(e.eef_pos()+self.place_position-e.ball_pos() if self.settings.get('track_ball')
                    else self.place_position-self.ball_offset)
            if elapsed>1. and np.linalg.norm(e.eef_pos()-target)<self.settings.get('place_tolerance', .006) and np.linalg.norm(e.ball_velocity())<.06:
                self.open_target = target.copy()
                self._transition('open')
            return self._pose_action(target,1)
        if self.stage=='open':
            target=self.open_target
            if self.release and not e._contacts()[1] and elapsed>.08:
                self._transition('retreat')
            return self._pose_action(target,-1 if self.release else 1)
        if self.stage=='retreat':
            return self._pose_action(self.open_target+np.asarray(self.settings.get('retreat_axis', -self.orientation[:,2]))*.18,-1)
        return np.r_[np.zeros(6),-1]
