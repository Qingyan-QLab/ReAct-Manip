"""Only the initial laptop XY and yaw vary between episodes."""
import math
import numpy as np
from react_manip.instances.records import digest


def validate_episode_instance(task, scene):
    if not isinstance(scene, dict) or set(scene) != {'task_digest', 'episode_seed', 'xy_yaw'} or scene['task_digest'] != digest(task):
        raise ValueError('laptop scene must reference its Task Instance')
    if type(scene['episode_seed']) is not int or scene['episode_seed'] < 0:
        raise ValueError('invalid episode seed')
    pose = scene['xy_yaw']
    if (not isinstance(pose, list) or len(pose) != 3
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in pose)
            or not (.24 <= pose[0] <= .29 and -.13 <= pose[1] <= .13 and -math.pi <= pose[2] <= math.pi)):
        raise ValueError('laptop XY/yaw outside reset domain')


def sample_episode_instance(task, seed):
    rng = np.random.default_rng(seed)
    scene = dict(task_digest=digest(task), episode_seed=seed,
                 xy_yaw=[float(rng.uniform(.24,.29)),float(rng.uniform(-.13,.13)),float(rng.uniform(-math.pi,math.pi))])
    validate_episode_instance(task, scene)
    return scene
