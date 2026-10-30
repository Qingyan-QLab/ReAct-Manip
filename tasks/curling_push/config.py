"""One Profile owns both ranges and fixed settings; Scenario owns sampled values."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from react_manip.core.scenarios import sample_scenario
from react_manip.core.visual_randomization import resolve_scenario

PROFILE = Path(__file__).resolve().parents[2] / 'configs/curling_push/profile_default.json'


def load_scenario(scenario=None, *, seed=0, profile_path=None):
    result = resolve_scenario(scenario)
    if result is None:
        profile = json.loads(Path(profile_path or PROFILE).read_text())
        result = sample_scenario(profile, seed=seed or 0)
    if result.task_id != 'curling_push':
        raise ValueError('Scenario task_id must be curling_push')
    validate(result.parameters)
    return deepcopy(result)


def validate(p):
    """Reject impossible geometry and invalid dynamics before compiling MuJoCo."""
    keys = {
        'stone': ('mass_kg', 'radius_m', 'height_m', 'start_x_m'),
        'target': ('distance_m', 'radius_m'),
        'ice': ('length_m', 'width_m', 'height_m', 'friction'),
        'rules': ('push_end_x_m', 'max_push_sec', 'release_gap_sec', 'stop_speed_mps',
                  'stop_angular_speed_rps', 'stop_hold_sec', 'max_lift_m'),
        'control': ('hz', 'horizon_sec', 'position_scale', 'rotation_scale'),
    }
    for group, names in keys.items():
        for name in names:
            value = p[group][name]
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{group}.{name} must be finite and positive')
    s, t, ice, rules = (p[k] for k in ('stone', 'target', 'ice', 'rules'))
    if t['radius_m'] <= s['radius_m']:
        raise ValueError('Target must contain the entire stone')
    # The near edge of a 1 m target may overlap the launch zone. The rules
    # constrain the stone centre during contact, not the target circle edge.
    if not s['start_x_m'] + s['radius_m'] < rules['push_end_x_m'] < t['distance_m']:
        raise ValueError('Launch, push boundary and target centre must be ordered')
    if s['start_x_m'] + s['radius_m'] >= t['distance_m'] - t['radius_m']:
        raise ValueError('Initial stone must be outside the target circle')
    if t['distance_m'] + t['radius_m'] >= ice['length_m'] or 2*t['radius_m'] >= ice['width_m']:
        raise ValueError('Target must fit inside the ice lane')
    if s['start_x_m'] - s['radius_m'] <= .2:
        raise ValueError('Stone must start fully on the lane (x > 0.2 m)')
    if rules['release_gap_sec'] >= rules['max_push_sec']:
        raise ValueError('Release debounce must be shorter than the push limit')
    if int(p['control']['hz']) != p['control']['hz'] or 500 % p['control']['hz']:
        raise ValueError('Control hz must divide the 500 Hz physics rate')
