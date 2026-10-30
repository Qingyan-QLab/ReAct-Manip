"""Reset-only flat spiral placement with exact segment lengths.

The elastic rest shape stays straight. Curvature is assigned to joint positions
only at reset, so the coil can physically unwind when lifted.
"""
from dataclasses import dataclass
import numpy as np

from .model import BAR, BAR_HALF_LENGTH, START, TABLE_CENTER, TABLE_HALF_SIZE, TABLE_Z


@dataclass(frozen=True)
class InitialLayout:
    layout: str = 'straight'
    center_xy: tuple = (.50, .27)
    inner_radius: float = .045
    pitch: float = .060
    phase_deg: float = 0.
    clockwise: bool = False
    left_links: int = 20
    bar_y: float = -.25

    def __post_init__(self):
        center = np.asarray(self.center_xy, dtype=float)
        if self.layout not in ('straight', 'coil', 'draped'):
            raise ValueError('initialization.layout must be straight, coil or draped')
        if (not isinstance(self.left_links, int) or isinstance(self.left_links, bool)
                or not np.isfinite(self.bar_y) or abs(self.bar_y-BAR[1]) >= BAR_HALF_LENGTH-.08):
            raise ValueError('Invalid draped initialization')
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError('initialization.center_xy must contain two finite coordinates')
        if not np.isfinite([self.inner_radius, self.pitch, self.phase_deg]).all():
            raise ValueError('Coil parameters must be finite')
        if not (.03 <= self.inner_radius <= .20 and .025 <= self.pitch <= .20):
            raise ValueError('Coil inner_radius or pitch is outside construction bounds')
        if not isinstance(self.clockwise, bool):
            raise ValueError('initialization.clockwise must be a boolean')

    def points(self, rope):
        if self.layout == 'draped':
            # Exact 25 mm chords around the bar; no initially stretched joints.
            if (rope.elastic or rope.segments != 40 or not np.isclose(rope.ds,.025)
                    or not np.isclose(rope.radius,.012) or not 8 <= self.left_links <= 26):
                raise ValueError('Draped reset currently requires 40 links, length 1 m, radius 12 mm')
            angles=np.deg2rad([90]*self.left_links+[75,45,15,-15,-45,-75]
                             +[-90]*(rope.segments-self.left_links-6))
            directions=rope.ds*np.c_[np.cos(angles),np.zeros(len(angles)),np.sin(angles)]
            points=np.vstack([np.zeros(3),np.cumsum(directions,axis=0)])
            width=points[self.left_links+6,0]-points[self.left_links,0]
            points += [BAR[0]-width/2,self.bar_y,BAR[2]+.006-points[self.left_links,2]]
            return points
        if self.layout == 'straight':
            points = START + np.c_[np.arange(rope.segments + 1)*rope.ds,
                                   np.zeros((rope.segments + 1, 2))]
            points[:, 2] = TABLE_Z + rope.radius + .001
        else:
            if self.pitch < 2*rope.radius + .012:
                raise ValueError('Coil pitch must leave at least 12 mm between rope surfaces')
            phase = np.deg2rad(self.phase_deg)
            direction = -1 if self.clockwise else 1

            def point(theta):
                radius = self.inner_radius + self.pitch*theta/(2*np.pi)
                angle = phase + direction*theta
                return np.asarray(self.center_xy) + radius*np.array([np.cos(angle), np.sin(angle)])

            theta = 0.
            xy = [point(theta)]
            # Solve each chord rather than resampling arc length: MuJoCo's
            # rigid segments must fit the initialization without stretching.
            for _ in range(rope.segments):
                # Bracket the first chord-length crossing. A single large
                # angular bracket is not monotone for long rigid rods.
                lo, hi = theta, theta+.1
                while np.linalg.norm(point(hi)-xy[-1]) < rope.ds:
                    lo, hi = hi, hi+.1
                    if hi-theta>8*np.pi:
                        raise ValueError('Rod is too long for this coil radius and pitch')
                for _ in range(45):
                    mid = (lo + hi)/2
                    if np.linalg.norm(point(mid)-xy[-1]) < rope.ds:
                        lo = mid
                    else:
                        hi = mid
                theta = (lo + hi)/2
                xy.append(point(theta))
            points = np.c_[xy, np.full(len(xy), TABLE_Z + rope.radius + .001)]
        clearance = TABLE_HALF_SIZE[:2] - rope.radius - .004
        if np.any(np.abs(points[:, :2]-TABLE_CENTER[:2]) > clearance):
            raise ValueError('Initial rope does not fit on the table; reduce length/radius/pitch or move center_xy')
        for y in (BAR[1]-BAR_HALF_LENGTH-.025, BAR[1]+BAR_HALF_LENGTH+.025):
            distance = np.maximum(np.abs(points[:, :2]-[BAR[0], y])-.025, 0)
            if np.any(np.linalg.norm(distance, axis=1) < rope.radius+.005):
                raise ValueError('Initial rope intersects a bar stand; move center_xy')
        return points

    def summary(self, rope):
        points = self.points(rope)
        if self.layout in ('straight','draped'):
            return {'layout': self.layout, 'turns': 0., 'outer_radius': 0.}
        offset = points[:, :2]-self.center_xy
        angles = np.unwrap(np.arctan2(offset[:, 1], offset[:, 0]))
        return {'layout': self.layout, 'turns': float(abs(angles[-1]-angles[0])/(2*np.pi)),
                'outer_radius': float(np.linalg.norm(offset, axis=1).max())}

    def apply(self, model, data, rope):
        """Initialize only rope DOFs; caller must invoke this only during reset."""
        import mujoco
        points = self.points(rope)
        tangents = np.diff(points, axis=0)
        if self.layout == 'draped':
            from scipy.spatial.transform import Rotation
            angles=np.arctan2(-tangents[:,2],tangents[:,0])
            rotations=Rotation.from_euler('y',angles)
        else:
            # Retain the original coil arithmetic, including its rounding, so
            # the saved long-horizon hanging trajectory remains reproducible.
            angles=np.unwrap(np.arctan2(tangents[:,1],tangents[:,0]))
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'rope_root')
        qa, va = model.jnt_qposadr[root_id], model.jnt_dofadr[root_id]
        quat=(rotations[0].as_quat()[[3,0,1,2]] if self.layout=='draped' else
              [np.cos(angles[0]/2),0,0,np.sin(angles[0]/2)])
        data.qpos[qa:qa+7] = np.r_[points[0],quat]
        data.qvel[va:va+6] = 0
        for i in range(1, rope.segments):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'rope_joint_{i}')
            qa, va = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
            angle=angles[i]-angles[i-1]
            data.qpos[qa:qa+4] = ((rotations[i-1].inv()*rotations[i]).as_quat()[[3,0,1,2]]
                                  if self.layout=='draped' else
                                  [np.cos(angle/2),0,0,np.sin(angle/2)])
            data.qvel[va:va+3] = 0
        mujoco.mj_forward(model, data)
