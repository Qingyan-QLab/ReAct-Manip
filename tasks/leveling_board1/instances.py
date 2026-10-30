"""Validate resolved shim dimensions and their matching explicit slot permutation."""
import hashlib
import json
import numpy as np

SHIM_NAMES = tuple(f'shim_{i:02d}' for i in range(1, 7))
SLOTS = ((.06,.18),(.24,.18),(.42,.18),(.06,.36),(.24,.36),(.42,.36))


def task_digest(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sample_task_instance(seed, profile=None):
    rng = np.random.default_rng(seed)
    heights = np.array([.03,.04,.05]); rng.shuffle(heights)
    dims, axes = {}, {}
    for i, name in enumerate(SHIM_NAMES):
        value = rng.uniform(.06,.14,3)
        if i < 3:
            axes[name] = int(rng.integers(0,3)); value[axes[name]] = float(heights[i])
        dims[name] = value.tolist()
    return {'shim_dimensions':dims, 'solution_axes':axes}


def validate_task_instance(task, profile=None):
    if set(task) != {'shim_dimensions','solution_axes'}:
        raise ValueError('only shim dimensions and solution axes belong to Task Instance')
    dims, axes = task['shim_dimensions'], task['solution_axes']
    if set(dims) != set(SHIM_NAMES) or len(axes) != 3 or not set(axes) <= set(SHIM_NAMES):
        raise ValueError('expected six shims and three solution axes')
    heights = []
    for name in SHIM_NAMES:
        values = dims[name]
        if not isinstance(values,list) or len(values)!=3:
            raise ValueError('expected three full side lengths')
        axis = axes.get(name)
        if name in axes and (type(axis) is not int or axis not in range(3)):
            raise ValueError('invalid thin axis')
        for j,value in enumerate(values):
            if type(value) not in (int,float) or not np.isfinite(value):
                raise ValueError('dimensions must be finite numbers')
            if j == axis:
                heights.append(value)
            elif not .06 <= value <= .14:
                raise ValueError('ordinary side length outside [0.06,0.14]')
    if sorted(heights) != [.03,.04,.05]:
        raise ValueError('thin dimensions must be a permutation of 3/4/5 cm')


def sample_episode_instance(task, seed):
    validate_task_instance(task)
    return {'task_digest':task_digest(task),'episode_seed':int(seed),
            'slot_indices':np.random.default_rng(seed).permutation(6).tolist()}


def validate_episode_instance(task, scene):
    validate_task_instance(task)
    if set(scene) != {'task_digest','episode_seed','slot_indices'} or scene['task_digest'] != task_digest(task):
        raise ValueError('Episode must reference its fixed Task Instance')
    slots = scene['slot_indices']
    if type(scene['episode_seed']) is not int or scene['episode_seed'] < 0:
        raise ValueError('invalid Episode seed')
    if not isinstance(slots,list) or any(type(s) is not int for s in slots) or sorted(slots)!=list(range(6)):
        raise ValueError('Episode must permute the six fixed slots exactly once')
