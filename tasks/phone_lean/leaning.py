"""Orientation helper and compatibility access to environment lean metrics."""
import math
import numpy as np


def screen_up_orientation(direction, tilt_deg=30):
    """Local -X is the screen, +X the back, +Z the phone's top edge."""
    inward=np.array([direction[0],direction[1],0.],dtype=float)
    inward/=np.linalg.norm(inward)
    a=math.radians(tilt_deg);up=np.array([0.,0.,1.])
    back=inward*math.cos(a)-up*math.sin(a)
    top=inward*math.sin(a)+up*math.cos(a)
    return np.column_stack((back,np.cross(top,back),top))


def lean_sample(env,target):
    """Compatibility wrapper; the authoritative criteria live in ``env.py``."""
    return env.lean_sample(target)
