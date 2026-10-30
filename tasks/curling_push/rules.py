"""Substep-level one-push state machine, independent of the simulator."""
import numpy as np


class PushState:
    def __init__(self, parameters):
        self.p = parameters
        self.phase = 'ready'
        self.first_contact = self.last_contact = None
        self.last_time = None
        self.rest_sec = 0.
        self.success = False
        self.failure = None
        self.finished = False

    def update(self, time, position, velocity, contact):
        if self.finished:
            return
        p, v = np.asarray(position), np.asarray(velocity)
        dt = 0. if self.last_time is None else max(0., time - self.last_time)
        self.last_time = time
        s, ice, r = (self.p[k] for k in ('stone', 'ice', 'rules'))
        if not np.isfinite(np.r_[p, v]).all():
            self.failure = 'non_finite'
        elif (p[0] - s['radius_m'] < .2 or p[0] + s['radius_m'] > ice['length_m']
              or abs(p[1]) + s['radius_m'] > ice['width_m']/2
              or p[2] < ice['height_m'] - .02):
            self.failure = 'out_of_bounds'
        elif p[2] > ice['height_m'] + s['height_m']/2 + r['max_lift_m']:
            self.failure = 'stone_lifted'
        if contact:
            if self.phase == 'sliding':
                self.failure = 'second_push'
            elif self.phase == 'ready':
                self.first_contact = time
                self.phase = 'pushing'
            self.last_contact = time
            if p[0] > r['push_end_x_m']:
                self.failure = 'push_beyond_launch_zone'
        if self.phase == 'pushing':
            if not contact and time - self.last_contact >= r['release_gap_sec']:
                self.phase = 'sliding'
            elif time - self.first_contact > r['max_push_sec']:
                self.failure = 'push_too_long'
        resting = (self.phase == 'sliding' and not contact
                   and np.linalg.norm(v[:3]) < r['stop_speed_mps']
                   and np.linalg.norm(v[3:]) < r['stop_angular_speed_rps'])
        self.rest_sec = self.rest_sec + dt if resting else 0.
        if self.rest_sec >= r['stop_hold_sec']:
            target = self.p['target']
            inside = np.linalg.norm(p[:2] - [target['distance_m'], 0]) + s['radius_m'] <= target['radius_m']
            self.success = bool(inside and not self.failure)
            if not self.success and not self.failure:
                self.failure = 'stopped_outside_target'
            self.finished = True
        if self.failure:
            self.success = False
            self.finished = True
