"""Explicit immutable drawer inventory and a five-object examination set."""
import math
from .objects import OBJECT_NAMES


def validate_inventory(inventory):
    if not isinstance(inventory, dict) or set(inventory) != {'names','drawers','poses','target_set'}:
        raise ValueError('explicit inventory required')
    if inventory['names'] != list(OBJECT_NAMES):
        raise ValueError('formal track uses all 36 catalog objects in catalog order')
    drawers, poses, targets = (inventory[k] for k in ('drawers','poses','target_set'))
    if not isinstance(drawers,list) or len(drawers)!=36 or any(type(d) is not int or not 0<=d<6 for d in drawers) or any(drawers.count(d)!=6 for d in range(6)):
        raise ValueError('each of six drawers must contain six distinct objects')
    if not isinstance(poses,list) or len(poses)!=36:
        raise ValueError('36 explicit poses required')
    for pose, drawer in zip(poses,drawers):
        if not isinstance(pose,list) or len(pose)!=7 or any(type(v) not in (int,float) or not math.isfinite(v) for v in pose):
            raise ValueError('invalid object pose')
        x,y,z,w,qx,qy,qz=pose
        y-=(-.20,.20)[drawer%2]
        floor=.658-.22*(drawer//2)
        if (min(abs(x-.030),abs(x-.140))>.0030001 or min(abs(y+.085),abs(y),abs(y-.085))>.0030001
                or not floor-.02<=z<=floor+.111 or abs(qx)>1e-12 or abs(qy)>1e-12
                or abs(w*w+qz*qz-1)>1e-10 or abs(2*math.atan2(qz,w))>.150001):
            raise ValueError('inventory pose outside fixed one-layer placement domain')
    if not isinstance(targets,list) or len(targets)!=5 or len(set(targets))!=5 or any(type(t)is not int or not 0<=t<36 for t in targets):
        raise ValueError('five distinct target indices required')


def capture_inventory(env):
    inventory=dict(names=list(env.object_names),drawers=env.object_drawers.tolist(),
                   poses=[env.sim.data.qpos[int(env.sim.model.jnt_qposadr[j]):int(env.sim.model.jnt_qposadr[j])+7].tolist() for j in env.object_joint_ids],
                   target_set=list(env._targets))
    validate_inventory(inventory)
    return inventory
