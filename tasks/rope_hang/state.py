"""Read-only geometric checks and continuous stability accumulation."""
import numpy as np
from .model import BAR, BAR_RADIUS, BAR_HALF_LENGTH


def hanging_geometry(points, radius):
    p=np.asarray(points)
    x=p[:,0]-BAR[0]
    sides=np.sign(x[np.abs(x)>BAR_RADIUS*.7])
    transitions=int(np.count_nonzero(np.diff(sides)))
    angles=np.unwrap(np.arctan2(x,p[:,2]-BAR[2]))
    return bool(p[0,2]<BAR[2]-.06 and p[-1,2]<BAR[2]-.06 and
                x[0]*x[-1]<0 and transitions==1 and
                np.ptp(angles)<2*np.pi+.15 and
                p[:,2].max()>BAR[2]+BAR_RADIUS*.5 and p[:,2].min()>radius+.02)


def support_coordinates(points, ds, radius, contact_links):
    """Return material distance and axial position at the unique supported top crossing.

    Unlike endpoint positions, these coordinates do not measure free-end swinging.
    Reject ambiguous crossings and crossings unrelated to a real bar contact.
    """
    p = np.asarray(points)
    if not np.isfinite(p).all():
        return None
    candidates = []
    for i, (a, b) in enumerate(zip(p[:-1], p[1:])):
        dx = b[0] - a[0]
        if abs(dx) < 1e-9:
            continue
        f = (BAR[0] - a[0]) / dx
        if not 0 <= f <= 1:
            continue
        q = a + f * (b - a)
        index = i + f
        if q[2] <= BAR[2]:
            continue
        if candidates and abs(index - candidates[-1][0]) < 1e-7:
            continue  # A crossing exactly at a node belongs to both segments.
        candidates.append((index, q))
    if len(candidates) != 1:
        return None
    index, q = candidates[0]
    if (abs(q[2] - (BAR[2] + BAR_RADIUS)) > radius + .015
            or abs(q[1] - BAR[1]) >= BAR_HALF_LENGTH - .06
            or not any(k - 1 <= index <= k + 2 for k in contact_links)):
        return None
    return np.array([index * ds, q[1]])


class Stability:
    """Five seconds of independent support with bounded material/axial slip.

    Range uses every physics sample. Least-squares trends reject sustained slow
    drift; resetting starts a new contiguous window, never accumulates fragments.
    """
    def __init__(self, duration=5., drift=.010, speed=.002):
        self.duration=duration
        self.drift=drift
        self.speed=speed
        self.reset()

    def reset(self):
        self.elapsed=0.
        self.reference=None
        self.range=np.zeros(2)
        self.trend=np.zeros(2)
        self.count=0
        self.sum_t=self.sum_tt=0.
        self.sum_q=np.zeros(2)
        self.sum_tq=np.zeros(2)

    def update(self, coordinates, eligible, dt):
        if (not eligible or coordinates is None or not np.isfinite(coordinates).all()
                or not np.isfinite(dt) or dt <= 0):
            self.reset()
            return False
        q=np.asarray(coordinates, dtype=float)
        if q.shape != (2,):
            raise ValueError('Expected material and axial support coordinates')
        if self.reference is not None:
            lo=np.minimum(self.minimum,q);hi=np.maximum(self.maximum,q)
            if np.any(hi-lo > self.drift):
                self.reset()
        if self.reference is None:
            self.reference=q.copy()
            self.minimum=q.copy();self.maximum=q.copy()
        self.minimum=np.minimum(self.minimum,q);self.maximum=np.maximum(self.maximum,q)
        self.range=self.maximum-self.minimum
        self.elapsed+=dt
        t=self.elapsed
        relative=q-self.reference
        self.count+=1
        self.sum_t+=t;self.sum_tt+=t*t
        self.sum_q+=relative;self.sum_tq+=t*relative
        denominator=self.count*self.sum_tt-self.sum_t**2
        if denominator > 1e-12:
            self.trend=(self.count*self.sum_tq-self.sum_t*self.sum_q)/denominator
        if self.elapsed + 1e-10 < self.duration:
            return False
        if np.any(np.abs(self.trend)>self.speed):
            self.reset()
            return False
        self.elapsed=max(self.elapsed,self.duration)
        return True
