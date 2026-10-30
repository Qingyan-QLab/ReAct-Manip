"""Sample one reproducible physical ring and regular target grid per seed."""
import json
import math
from pathlib import Path


PROFILE_PATH = Path(__file__).resolve().parents[2] / "configs/ring_toss1/profile_default.json"


def sample_instance(seed=0, profile=None):
    seed = int(seed)
    if seed < 0:
        raise ValueError("ring_toss1 seed must be nonnegative")
    profile = json.loads(PROFILE_PATH.read_text()) if profile is None else profile
    from react_manip.core.scenarios import sample_scenario
    parameters = sample_scenario(profile, seed=seed).parameters
    return resolve_instance(parameters, seed=seed, profile=profile)


def resolve_instance(parameters, *, seed=0, profile=None):
    """Build the legacy scene view from concrete values, without any RNG."""
    profile = json.loads(PROFILE_PATH.read_text()) if profile is None else profile
    values = {f'{section}.{key}': value for section in ('ring', 'target')
              for key, value in parameters[section].items()}
    fixed = parameters
    ring_radius = values["ring.centerline_radius_m"]
    tube_radius = values["ring.tube_radius_m"]
    peg_radius = values["target.radius_m"]
    limits = fixed["placement_limits"]
    inner_radius = ring_radius * math.cos(math.pi / fixed["rings"]["segments"]) - tube_radius
    if peg_radius + limits["minimum_hole_clearance_m"] >= inner_radius:
        raise ValueError("Target peg leaves insufficient clearance inside the ring")

    near_y = float(fixed["targets"]["near_row_y_m"])
    row_spacing = values["target.row_spacing_m"]
    column_spacing = values["target.column_spacing_m"]
    targets = tuple((column * column_spacing, near_y + row * row_spacing,
                     peg_radius, float(fixed["targets"]["height_m"]))
                    for row in range(3) for column in (-1, 0, 1))
    # The table spans x=[-.72,.72], y=[-.75,1.35]. Keep the entire peg
    # footprint on its surface. A compiled ALOHA reach bound is tested separately.
    if column_spacing + peg_radius >= 0.72 or targets[-1][1] + peg_radius >= 1.35:
        raise ValueError("Target grid exceeds the table")
    if targets[-1][1] > limits["maximum_far_row_y_m"]:
        raise ValueError("Far row is too far from the pickup area")
    if near_y - peg_radius - (-0.68) <= limits["minimum_near_face_from_base_m"]:
        raise ValueError("Near row is too close to the robot base")
    return {
        "seed": seed, "profile_id": profile["profile_id"],
        "table_top_z_m": float(fixed["table_top_z_m"]),
        "pickup_xy_m": tuple(map(float, fixed["rings"]["pickup_xy_m"])),
        "task": fixed["task"],
        "ring_friction": fixed["rings"]["friction"],
        "points": tuple(score for score in fixed["targets"]["row_scores"] for _ in range(3)),
        "ring_centerline_radius_m": ring_radius,
        "ring_tube_radius_m": tube_radius,
        "ring_mass_kg": values["ring.mass_kg"],
        "target_radius_m": peg_radius,
        "target_column_spacing_m": column_spacing,
        "target_row_spacing_m": row_spacing,
        "inner_radius_m": inner_radius,
        "targets": targets,
    }


def derived(parameters, profile):
    resolved = resolve_instance(parameters, profile=profile)
    r, t, mass = (resolved[k] for k in ('ring_centerline_radius_m', 'ring_tube_radius_m', 'ring_mass_kg'))
    planar = mass*(r*r/2 + 5*t*t/8)
    polar = mass*(r*r + 3*t*t/4)
    return {'targets': [list(target) for target in resolved['targets']],
            'inner_radius_m': resolved['inner_radius_m'],
            'ring_inertia_diagonal': [planar, planar, polar]}


def sample_task_instance(seed, profile):
    from react_manip.core.scenarios import sample_scenario
    parameters = sample_scenario(profile, seed=seed).parameters
    parameters['derived'] = derived(parameters, profile)
    return parameters


def validate_resolved(parameters, profile):
    from copy import deepcopy
    p = deepcopy(parameters)
    actual = p.pop('derived', None)
    try:
        expected = derived(p, profile)
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise ValueError('invalid ring parameters') from exc
    if actual != expected:
        raise ValueError('ring derived targets/clearance/inertia mismatch')
    return p
