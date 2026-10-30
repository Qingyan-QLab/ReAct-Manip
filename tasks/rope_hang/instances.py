"""Fixed draped resets; explicit legacy coil scenes retain translation support."""
from dataclasses import replace
import numpy as np
from react_manip.instances.records import digest
from .initialization import InitialLayout
from .model import RodChain


def validate_episode_instance(task, scene):
    if task['initialization']['layout'] == 'draped':
        if (set(scene) != {'task_digest', 'episode_seed', 'initialization'}
                or scene['task_digest'] != digest(task)
                or scene['initialization'] != task['initialization']):
            raise ValueError('Episode must preserve the fixed draped initialization')
        if type(scene['episode_seed']) is not int or scene['episode_seed'] < 0:
            raise ValueError('invalid Episode seed')
        InitialLayout(**scene['initialization']).points(RodChain(**task['chain']))
        return
    if set(scene) != {'task_digest', 'episode_seed', 'center_xy'} or scene['task_digest'] != digest(task):
        raise ValueError('Episode must reference its fixed Task Instance')
    if type(scene['episode_seed']) is not int or scene['episode_seed'] < 0:
        raise ValueError('invalid Episode seed')
    xy = scene['center_xy']
    if not isinstance(xy, list) or len(xy) != 2 or any(type(v) not in (int,float) for v in xy):
        raise ValueError('center_xy must contain two finite coordinates')
    if not (.47 <= xy[0] <= .53 and .24 <= xy[1] <= .30):
        raise ValueError('Episode translation outside +/-0.03 m domain')
    replace(InitialLayout(**task['initialization']), center_xy=tuple(xy)).points(RodChain(**task['chain']))


def sample_episode_instance(task, seed):
    if task['initialization']['layout'] == 'draped':
        from copy import deepcopy
        scene = dict(task_digest=digest(task), episode_seed=seed,
                     initialization=deepcopy(task['initialization']))
        validate_episode_instance(task, scene)
        return scene
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        scene = {'task_digest': digest(task), 'episode_seed': int(seed),
                 'center_xy': (np.array([.5,.27])+rng.uniform(-.03,.03,2)).tolist()}
        try:
            validate_episode_instance(task, scene)
            return scene
        except ValueError:
            continue
    raise ValueError('no valid coil position found in 1000 attempts')
