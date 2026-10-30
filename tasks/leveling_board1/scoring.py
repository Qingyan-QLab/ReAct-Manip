"""Stateful 20 Hz scoring, independent of rendering and policy observations."""
from collections import deque
import numpy as np

STABLE_SECONDS = 5.
MAX_BOARD_SPEED_RAD_S = .01
MAX_ANGLE_DRIFT_DEG = .1
MAX_SHIM_LINEAR_SPEED = .005
MAX_SHIM_ANGULAR_SPEED = .05
MAX_SHIM_DRIFT_M = .005
MIN_CONTACT_FORCE_N = 1e-4


def angle_score(angle):
    return float(np.clip(100 - 10 * max(0., abs(angle)-.5), 0, 100))


class LevelScore:
    def __init__(self, hz=20):
        self.samples = deque(maxlen=round(STABLE_SECONDS*hz)+1)
        self.hz = hz
        self.success = False

    def update(self, *, angle, released, supported, board_speed, positions,
               linear_speed, angular_speed):
        base = angle_score(angle)
        valid = (released and supported and abs(board_speed)<=MAX_BOARD_SPEED_RAD_S
                 and linear_speed<=MAX_SHIM_LINEAR_SPEED and angular_speed<=MAX_SHIM_ANGULAR_SPEED)
        positions = {k:np.asarray(v).copy() for k,v in positions.items()}
        if not valid:
            self.samples.clear()
        else:
            if self.samples and set(self.samples[-1][1]) != set(positions):
                self.samples.clear()
            self.samples.append((float(angle), positions))
            angles = [a for a,_ in self.samples]
            drift = max((np.linalg.norm(p[name]-positions[name])
                         for _,p in self.samples for name in positions), default=0.)
            if max(angles)-min(angles)>MAX_ANGLE_DRIFT_DEG+1e-12 or drift>MAX_SHIM_DRIFT_M+1e-12:
                self.samples.clear()
                self.samples.append((float(angle),positions))
        stable = len(self.samples)==self.samples.maxlen
        success = stable and all(abs(a)<=.5+1e-10 for a,_ in self.samples)
        self.success |= success
        multiplier = 0. if not released or not supported else (1. if stable else .6)
        return {'angle_score':base,'score':100. if self.success else base*multiplier,
                'released':bool(released),'independently_supported':bool(supported),
                'stable':stable,'stable_window_sec':max(0,len(self.samples)-1)/self.hz,
                'score_multiplier':multiplier,'terminal_success':self.success}
