"""Versioned LIBERO delta integration on the unchanged benchmark control clock."""
import numpy as np
from scipy.spatial.transform import Rotation


def resample_libero(actions, source_hz, target_hz):
    """Integrate piecewise-constant twist / gripper increments over target ticks.

    No downsampling or fractional final tick is silently accepted. A 20 Hz
    checkpoint chunk of 50 actions becomes 125 actions on a 50 Hz task. This
    preserves duration and commanded displacement, not controller dynamics.
    """
    actions=np.asarray(actions,dtype=float)
    if actions.ndim!=2 or actions.shape[1]!=7 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError('Expected finite LIBERO [chunk,7]')
    if not np.isfinite([source_hz,target_hz]).all() or not 0<source_hz<=target_hz:
        raise ValueError('libero_delta_resample_v1 requires 0 < checkpoint_hz <= task_hz')
    ticks=len(actions)*target_hz/source_hz
    if not np.isclose(ticks,round(ticks),rtol=0,atol=1e-9):
        raise ValueError('Action chunk duration must contain an integer number of task ticks; adjust n_action_steps')
    out=[]
    for tick in range(round(ticks)):
        start=tick*source_hz/target_hz;end=(tick+1)*source_hz/target_hz
        value=np.zeros(7);rotation=Rotation.identity()
        for i in range(int(np.floor(start)),min(len(actions),int(np.ceil(end)))):
            fraction=max(0.,min(end,i+1)-max(start,i))
            value[:3]+=actions[i,:3]*fraction
            value[6]+=actions[i,6]*fraction
            rotation=Rotation.from_rotvec(actions[i,3:6]*.5*fraction)*rotation
        value[3:6]=rotation.as_rotvec()/.5
        out.append(value)
    return np.asarray(out)
