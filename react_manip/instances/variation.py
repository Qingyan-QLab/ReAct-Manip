"""Explicit, evaluator-private Episode variations for four feedback tasks.

Task parameters remain the baseline. Resolved variations are saved, validated,
and applied before compilation; no sampling takes place during replay or step.
"""
from copy import deepcopy
import numpy as np

TASKS = ('leveling_board1', 'conveyor_grasp', 'box_insertion', 'car_drive2')
DOMAINS = {
    'leveling_board1': {'ordinary_dimension_m': [.06, .14], 'thin_dimensions_m': [.03, .04, .05]},
    'conveyor_grasp': {'belt_speed_m_s': [.08, .20], 'target_size_scale': [.8, 1.15]},
    'box_insertion': {'friction_multiplier': [.6, 1.4], 'friction_limits': [.01, .6], 'mass_scale': [.7, 1.3]},
    'car_drive2': {'nominal_speed_m_s': [.9, 1.15]},
}


def _number(value, lo, hi):
    if type(value) not in (int, float) or not np.isfinite(value) or not lo-1e-12 <= value <= hi+1e-12:
        raise ValueError('Episode parameter outside its finite sampling domain')


def sample_variation(task_id, base, seed, index, attempt=0):
    """Each consecutive five Episodes cover five shuffled strata, in both phases.

    Rejection retries stay in the original stratum. Seeds and indices are private.
    Independent dimensions use independently shuffled strata.
    """
    if task_id not in TASKS or not 0 <= index < 15:
        raise ValueError('unsupported Episode variation request')
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, attempt, 261012]))
    axis = 0
    def draw(lo, hi):
        nonlocal axis
        order = np.random.default_rng(np.random.SeedSequence([seed, index//5, axis, 73])).permutation(5)
        axis += 1
        u = (int(order[index % 5]) + rng.uniform(.05, .95))/5
        return float(lo + (hi-lo)*u)
    if task_id == 'leveling_board1':
        from tasks.leveling_board1.instances import SHIM_NAMES
        useful = rng.choice(SHIM_NAMES, 3, replace=False).tolist()
        heights = rng.permutation([.03,.04,.05])
        axes = {n:int(rng.integers(3)) for n in useful}
        dims = {n:[draw(.06,.14) for _ in range(3)] for n in SHIM_NAMES}
        for n,h in zip(useful, heights):
            dims[n][axes[n]] = float(h)
        result = dict(shim_dimensions=dims, solution_axes=axes)
    elif task_id == 'conveyor_grasp':
        result = dict(belt_speed_m_s=draw(.08,.20),
                      target_size_scales=[draw(.8,1.15) for _ in range(base['task']['n_objects'])])
    elif task_id == 'box_insertion':
        result = dict(mass_scale=draw(.7,1.3), sliding_friction={
            side:draw(max(.01,base['block_friction'][side]*.6), min(.6,base['block_friction'][side]*1.4))
            for side in ('left','right')})
    else:
        result = dict(nominal_speed_m_s=draw(.9,1.15))
    effective_parameters(task_id, base, result)
    return result


def effective_parameters(task_id, base, variation):
    """Validate a strict whitelist and derive resolved parameters from the base."""
    p = deepcopy(base)
    expected = {
        'leveling_board1': {'shim_dimensions','solution_axes'},
        'conveyor_grasp': {'belt_speed_m_s','target_size_scales'},
        'box_insertion': {'mass_scale','sliding_friction'},
        'car_drive2': {'nominal_speed_m_s'},
    }
    if task_id not in expected or not isinstance(variation,dict):
        raise ValueError('invalid Episode variation fields')
    if set(variation) != expected[task_id]:
        raise ValueError('invalid Episode variation fields')
    v = deepcopy(variation)
    if task_id == 'leveling_board1':
        from tasks.leveling_board1.instances import validate_task_instance
        p.update(v)
        validate_task_instance(p)
    elif task_id == 'conveyor_grasp':
        _number(v['belt_speed_m_s'], .08,.20)
        scales = v['target_size_scales']
        if not isinstance(scales,list) or len(scales) != base['task']['n_objects']:
            raise ValueError('one size scale per target is required')
        for scale in scales:
            _number(scale,.8,1.15)
        p['belt']['speed'] = v['belt_speed_m_s']
    elif task_id == 'box_insertion':
        _number(v['mass_scale'],.7,1.3)
        if not isinstance(v['sliding_friction'],dict) or set(v['sliding_friction']) != {'left','right'}:
            raise ValueError('both block contact patches are required')
        for side, value in v['sliding_friction'].items():
            _number(value,max(.01,base['block_friction'][side]*.6),min(.6,base['block_friction'][side]*1.4))
            p['block_friction'][side] = value
    else:
        _number(v['nominal_speed_m_s'],.9,1.15)
        p['vehicle']['nominal_speed_m_s'] = v['nominal_speed_m_s']
    return p


def episode_scenario(scenario, variation):
    resolved = deepcopy(scenario)
    resolved.parameters = effective_parameters(scenario.task_id, scenario.parameters, variation)
    resolved.metadata.pop('initial_episode_scene', None)
    resolved.metadata['episode_variation'] = deepcopy(variation)
    return resolved


def construct_sampled_episode(scenario, robot, index, scene_seed, *, render=False):
    """Construct one sampled Episode for a task-default practice/exam run."""
    from .runtime import _construct
    variation = sample_variation(scenario.task_id, scenario.parameters, scenario.seed, index)
    resolved = episode_scenario(scenario, variation)
    scene = None
    if scenario.task_id == 'leveling_board1':
        from tasks.leveling_board1.instances import sample_episode_instance
        scene = sample_episode_instance(resolved.parameters, scene_seed)
    env = _construct(scenario.task_id, resolved, robot, episode=scene, render=render)
    return env, variation
