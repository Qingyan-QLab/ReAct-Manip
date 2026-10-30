"""Resolve all seven draws once; explicit instances never sample on replay."""
from copy import deepcopy
import math
import numpy as np

VALUES = (0, 8, 12, 20, 32, 48, 80, 100, 80, 48, 32, 20, 12, 8, 8, 4)


def derived(parameters):
    t, init = parameters['turntable'], parameters['initialization']
    scale = t['radius_m'] / .30
    inertia = t['inertia'] * scale ** 2
    return dict(scale=scale, inertia_diagonal=[.6*inertia, .6*inertia, inertia],
                center_y_m=.446*(scale-1),
                ring_yaw_rad=-math.pi/2-(init['sector']+init['fraction'])*2*math.pi/16)


def sample_task_instance(seed, profile):
    rng = np.random.default_rng(seed)
    p = deepcopy(profile['fixed'])
    # Preserve the original seed-to-scene mapping and conditional draw order.
    for group in ('physics', 'geometry'):
        for name, rule in profile[group].items():
            section, key = name.split('.')
            p.setdefault(section, {})[key] = float(rng.uniform(rule['low'], rule['high']))
    init = profile['initialization']
    score = int(rng.choice(init['allowed_scores']))
    sector = int(rng.choice(np.flatnonzero(np.asarray(VALUES) == score)))
    fraction = float(rng.uniform(*init['sector_fraction_range']))
    p['initialization'] = dict(score=score, sector=sector, fraction=fraction)
    p['derived'] = derived(p)
    return p


def validate_resolved(parameters, profile):
    """Return ordinary profile parameters after validating conditional fields."""
    p = deepcopy(parameters)
    init = p.get('initialization')
    if not isinstance(init, dict) or set(init) != {'score', 'sector', 'fraction'}:
        raise ValueError('explicit turntable initialization is required')
    if (type(init['score']) is not int or init['score'] not in profile['initialization']['allowed_scores']
            or type(init['sector']) is not int or not 0 <= init['sector'] < 16
            or VALUES[init['sector']] != init['score']
            or type(init['fraction']) not in (int, float)
            or not .2 <= init['fraction'] <= .8):
        raise ValueError('turntable conditional initialization is outside its domain')
    try:
        expected = derived(p)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('invalid turntable parameters') from exc
    if p.get('derived') != expected:
        raise ValueError('turntable derived geometry/inertia/ring yaw mismatch')
    p.pop('initialization'); p.pop('derived')
    return p
