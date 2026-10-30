"""Two independent seeded layers; the JSON profile is the sampling source."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "configs/phone_lean/profile_default.json"
from .supports import (SUPPORT_POOL, LEGACY_SUPPORTS, RADII as SUPPORT_RADII,
                       BOTTOM_OFFSETS as SUPPORT_BOTTOMS)
# Legacy constants retained for historical standalone verification scripts.
OBJECTS = ("phone", *LEGACY_SUPPORTS)
RADII = {"phone": .085, **SUPPORT_RADII}
BOTTOM_OFFSETS = {"phone": .076, **SUPPORT_BOTTOMS}


def instance_objects(instance):
    return tuple(instance["position_xyz_m"]) if "position_xyz_m" in instance else ("phone", *instance.get("support_names", LEGACY_SUPPORTS))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()[:20]


def load_profile():
    return json.loads(PROFILE_PATH.read_text())


def _choose_roles(pool, rng, rules):
    """Select four supports, then assign two small roles from profile rules."""
    selected = list(rng.choice(pool, 4, replace=False))
    groups = rules['size_groups']
    small_count = int(groups['small_count'][0])
    large_only = set(groups.get('large_only_supports', ()))
    candidates = [name for name in selected if name not in large_only]
    if len(candidates) < small_count:
        raise ValueError('selected supports cannot satisfy configured small roles')
    small = set(rng.choice(candidates, small_count, replace=False))
    return selected, small


def sample_task_instance(task_seed=0, profile=None):
    profile = load_profile() if profile is None else profile
    task_seed = int(task_seed)
    if task_seed < 0:
        raise ValueError("task_seed must be nonnegative")
    rng = np.random.default_rng(task_seed)
    rules = profile["task_instance"]
    result = {
        "task_id": "phone_lean", "profile_id": profile["profile_id"],
        "profile_digest": digest(profile), "task_seed": task_seed,
        "gravity_m_s2": float(rng.uniform(*rules["gravity_m_s2"])),
        "mass_kg": {name: float(rng.uniform(*bounds))
                    for name, bounds in rules["mass_kg"].items()},
        "surface_friction": {
            name: [float(rng.uniform(*bounds)) for bounds in ranges]
            for name, ranges in rules["surface_friction"].items()
        },
        "gripper_contact_friction": list(profile["fixed"]["gripper_contact_friction"]),
        "gripper_contact_condim": profile["fixed"]["gripper_contact_condim"],
        "surface_contact_condim": profile["fixed"]["surface_contact_condim"],
        "simulation": json.loads(json.dumps(profile["fixed"]["simulation"])),
        "size_scale": {name: float(rng.uniform(*bounds))
                       for name, bounds in rules["size_scale"].items()},
    }
    pool = rules.get("support_pool", list(LEGACY_SUPPORTS))
    count = rules.get("support_count", 4)
    if (len(set(pool)) != len(pool) or not set(pool) <= set(SUPPORT_POOL)
            or type(count) is not int or count != 4 or len(pool) < count):
        raise ValueError("support_pool must contain unique catalog names and support_count must be 4")
    selected, small_names = _choose_roles(pool, rng, rules)
    result["support_names"] = [name for name in pool if name in selected]
    groups = rules['size_groups']
    result['small_support_names'] = [n for n in result['support_names'] if n in small_names]
    result['large_support_names'] = [n for n in result['support_names']
                                     if n not in small_names]
    for name in result['small_support_names']:
        extent = .152 * result['size_scale']['phone'] * rng.uniform(*groups['small_max_extent_to_phone_length'])
        scale = extent / groups['nominal_max_extent_m'][name]
        result['size_scale'][name] = float(scale)
        result['mass_kg'][name] *= float(scale**3)
    for key in ('mass_kg', 'size_scale'):
        result[key] = {n: result[key][n] for n in instance_objects(result)}
    result['surface_friction'] = {n:result['surface_friction'][n] for n in ('table', *instance_objects(result))}
    result["task_instance_id"] = "phone_lean_task_" + digest(result)
    return result


def layout_is_clear(positions, gap, scales=None, phone_approach_clearance=0.08):
    scales = scales or {name: 1. for name in positions}
    radii = {name: RADII[name]*scales[name] for name in positions}
    for first, second in itertools.combinations(positions, 2):
        if np.linalg.norm(np.array(positions[first])-positions[second]) < radii[first]+radii[second]+gap:
            return False
    phone = np.asarray(positions['phone'])
    # Preserve a straight approach corridor from the robot side (-Y) to the
    # Home-button end of the phone for either gripper.
    approach = phone + np.array([0., -phone_approach_clearance])
    for name in positions:
        if name != 'phone' and np.linalg.norm(np.asarray(positions[name])-approach) < radii[name]+.06*scales['phone']:
            return False
    return True


def sample_episode_instance(task_instance, episode_seed=0, profile=None):
    profile = load_profile() if profile is None else profile
    episode_seed = int(episode_seed)
    if episode_seed < 0:
        raise ValueError("episode_seed must be nonnegative")
    rng = np.random.default_rng(episode_seed)
    rules = profile["episode_instance"]
    selected = task_instance['support_names']
    names = instance_objects(task_instance)
    supports = list(selected)
    slot_names = ("left_outer", "left_inner", "right_inner", "right_outer")
    slots = [rules["support_position_xy_m"][n] for n in slot_names]
    scales = task_instance["size_scale"]
    for attempt in range(rules["max_layout_attempts"]):
        positions = {n: [float(rng.uniform(*axis)) for axis in ranges]
                     for n, ranges in zip(supports, slots)}
        positions["phone"] = [float(rng.uniform(*axis))
                              for axis in rules['phone_position_xy_m']]
        if layout_is_clear(positions, rules['minimum_gap_m'], scales,
                           rules['phone_approach_clearance_m']):
            break
    else:
        raise ValueError("No separated layout in configured sampling budget")
    result = {
        "task_instance_id": task_instance["task_instance_id"],
        "episode_seed": episode_seed, "layout_attempt": attempt,
        "position_xyz_m": {n: [*positions[n], profile['fixed']['table_top_z_m']
                               +BOTTOM_OFFSETS[n]*scales[n]+.0005] for n in names},
        "quaternion_wxyz": {n: [1., 0., 0., 0.] for n in names},
        "robot_initialization_noise": 0.0,
    }
    result['position_xyz_m']['phone'][2] = (profile['fixed']['table_top_z_m']
        +.0063*scales['phone']+.001)
    result['quaternion_wxyz']['phone'] = list(profile['fixed']['initial_quaternion_wxyz']['phone'])
    result['episode_instance_id'] = 'phone_lean_episode_'+digest(result)
    return result


def apply_physics_xml(arena, task_instance):
    """Scale all component masses and inertias before MuJoCo compilation."""
    import xml.etree.ElementTree as ET
    option = arena.root.find("option")
    if option is None:
        option = ET.SubElement(arena.root, "option")
    option.set("gravity", f"0 0 {-task_instance['gravity_m_s2']}")
    simulation = task_instance['simulation']
    option.set('integrator', simulation['integrator'])
    option.set('tolerance', str(simulation['solver_tolerance']))
    flag = option.find('flag')
    if flag is None:
        flag = ET.SubElement(option, 'flag')
    flag.set('multiccd', 'enable' if simulation['multiccd'] else 'disable')
    table = arena.worldbody.find("./geom[@name='table']")
    for name in ("table", *instance_objects(task_instance)):
        if name != "table" and arena.worldbody.find(f"./body[@name='{name}']") is None:
            continue
        if name == "table":
            geoms = [table]
        else:
            body = arena.worldbody.find(f"./body[@name='{name}']")
            scale = task_instance['size_scale'][name]
            # Scale visual and collision geometry together, including hollow
            # cup meshes, capsule endpoints, camera bumps and base pads.
            def scale_attribute(element, key, factor):
                if key in element.attrib:
                    element.set(key, ' '.join(str(float(v)*factor)
                                             for v in element.get(key).split()))
            meshes = set()
            for element in body.iter():
                if element is not body:
                    scale_attribute(element, 'pos', scale)
                if element.tag in ('geom', 'site'):
                    scale_attribute(element, 'size', scale)
                    scale_attribute(element, 'fromto', scale)
                    if element.get('mesh'):
                        meshes.add(element.get('mesh'))
                if element.tag == 'inertial':
                    scale_attribute(element, 'diaginertia', scale**2)
                    scale_attribute(element, 'fullinertia', scale**2)
            for mesh_name in meshes:
                mesh = arena.asset.find(f"./mesh[@name='{mesh_name}']")
                mesh.set('scale', ' '.join(str(float(v)*scale)
                                          for v in mesh.get('scale', '1 1 1').split()))
            geoms = list(body.iter("geom"))
            inertial = body.find("inertial")
            desired = task_instance["mass_kg"][name]
            if inertial is not None:
                ratio = desired / float(inertial.get("mass"))
                inertial.set("mass", str(desired))
                inertial.set("diaginertia", " ".join(str(float(v) * ratio)
                                                     for v in inertial.get("diaginertia").split()))
            else:
                # Visuals with density defaults must not introduce hidden mass.
                original = sum(float(g.get("mass", 0)) for g in geoms
                               if int(g.get("group", 0)) == 0)
                for geom in geoms:
                    mass = float(geom.get("mass", 0)) if int(geom.get("group", 0)) == 0 else 0.0
                    geom.set("mass", str(mass * desired / original))
        for geom in geoms:
            if name == 'phone':
                geom.set('solref', ' '.join(map(str, simulation['phone_contact_solref'])))
                geom.set('solimp', ' '.join(map(str, simulation['phone_contact_solimp'])))
            geom.set("friction", " ".join(map(str, task_instance["surface_friction"][name])))
            geom.set("condim", str(task_instance["surface_contact_condim"]))
            geom.set("priority", "0")
    # Explicit cup/base pairs previously overrode sampled surface friction.
    contact = arena.root.find("contact")
    for pair in contact.findall("pair") if contact is not None else ():
        if pair.get("geom1") == "table":
            friction = np.maximum(task_instance["surface_friction"]["table"],
                                  task_instance["surface_friction"]["cup"])
            pair.set("friction", " ".join(map(str, friction[[0, 0, 1, 2, 2]])))
            pair.set("condim", str(task_instance["surface_contact_condim"]))


def validate_task_instance(instance, profile=None):
    """Validate explicit values without using seeds to reconstruct the scene."""
    profile = load_profile() if profile is None else profile
    required = {'task_id', 'profile_id', 'profile_digest', 'task_seed', 'gravity_m_s2',
                'mass_kg', 'surface_friction', 'gripper_contact_friction',
                'gripper_contact_condim', 'surface_contact_condim', 'simulation',
                'size_scale', 'support_names', 'small_support_names',
                'large_support_names', 'task_instance_id'}
    if set(instance) != required or instance['task_id'] != 'phone_lean':
        raise ValueError('invalid phone Task Instance fields')
    if instance['profile_id'] != profile['profile_id']:
        raise ValueError('phone profile version mismatch')
    if instance['profile_digest'] != digest(profile):
        raise ValueError('phone profile version mismatch')
    if instance['task_instance_id'] != 'phone_lean_task_'+digest({k:v for k,v in instance.items() if k != 'task_instance_id'}):
        raise ValueError('phone Task Instance digest mismatch')
    rules = profile['task_instance']
    small_ratio_bounds = rules['size_groups']['small_max_extent_to_phone_length']
    selected = instance['support_names']
    if not isinstance(selected,list) or len(selected)!=4 or len(set(selected))!=4 or not set(selected)<=set(rules['support_pool']):
        raise ValueError('Task Instance must select four distinct supports')
    if selected != [n for n in rules['support_pool'] if n in selected]: raise ValueError('support order must follow catalog')
    small = instance['small_support_names']
    large = instance['large_support_names']
    groups = rules['size_groups']
    if (not isinstance(small, list) or len(small) != 2
            or len(set(small)) != len(small) or not set(small) <= set(selected)
            or small != [n for n in selected if n in small]):
        raise ValueError('invalid small support group')
    if (not isinstance(large, list) or len(large) != 2
            or large != [name for name in selected if name not in small]
            or set(large) != set(selected) - set(small)):
        raise ValueError('invalid large support roles')
    if set(small) & set(groups.get('large_only_supports', ())):
        raise ValueError('large-only support assigned to a small role')
    if type(instance['task_seed']) is not int or instance['task_seed'] < 0: raise ValueError('invalid task seed')
    for key in ('gripper_contact_friction', 'gripper_contact_condim', 'surface_contact_condim', 'simulation'):
        if instance[key] != profile['fixed'][key]: raise ValueError('frozen phone physics changed')
    def check(value, bounds):
        if type(value) not in (int,float) or not np.isfinite(value) or not bounds[0] <= value <= bounds[1]:
            raise ValueError('phone parameter outside sampling domain')
    check(instance['gravity_m_s2'], rules['gravity_m_s2'])
    for key in ('mass_kg', 'size_scale', 'surface_friction'):
        expected = {'phone', *selected} | ({'table'} if key == 'surface_friction' else set())
        if set(instance[key]) != expected: raise ValueError('phone selected-object parameters incomplete')
        for name, value in instance[key].items():
            bounds = rules[key][name]
            if key == 'surface_friction':
                if not isinstance(value, list) or len(value) != 3: raise ValueError('friction requires three components')
                for v,b in zip(value,bounds): check(v,b)
            elif name in small and key == 'size_scale':
                ratio = value * groups['nominal_max_extent_m'][name] / (.152*instance['size_scale']['phone'])
                check(ratio, [small_ratio_bounds[0]-1e-12,
                              small_ratio_bounds[1]+1e-12])
            elif name in small and key == 'mass_kg':
                scale = instance['size_scale'][name]
                check(value, [bounds[0]*scale**3, bounds[1]*scale**3])
            else: check(value,bounds)
    extents = {name: groups['nominal_max_extent_m'][name] * instance['size_scale'][name]
               for name in selected}
    if not min(extents[name] for name in large) > max(extents[name] for name in small):
        raise ValueError('large supports must be larger than small supports')
    if 'book' in selected:
        scale = instance['size_scale']['book']
        book = groups['book_dimensions_m']
        if book['thickness'] * scale > groups['book_max_thickness_m'] + 1e-12:
            raise ValueError('book exceeds configured maximum thickness')
        if book['height'] * scale < groups['book_min_realized_height_m'] - 1e-12:
            raise ValueError('book is below configured minimum height')


def validate_episode_instance(task, episode, profile=None):
    profile = load_profile() if profile is None else profile
    required = {'task_instance_id','episode_seed','layout_attempt','position_xyz_m',
                'quaternion_wxyz','robot_initialization_noise','episode_instance_id'}
    if set(episode) != required or episode['task_instance_id'] != task['task_instance_id']:
        raise ValueError('phone Episode Instance fields/reference mismatch')
    if episode['episode_instance_id'] != 'phone_lean_episode_'+digest({k:v for k,v in episode.items() if k != 'episode_instance_id'}):
        raise ValueError('phone Episode Instance digest mismatch')
    selected = task['support_names']
    names = set(instance_objects(task))
    if set(episode['position_xyz_m']) != names or set(episode['quaternion_wxyz']) != names:
        raise ValueError('episode pose keys must match selected objects')
    if episode['robot_initialization_noise'] != 0: raise ValueError('robot initial pose is fixed')
    for name in names:
        xyz, quat = np.asarray(episode['position_xyz_m'][name]), np.asarray(episode['quaternion_wxyz'][name])
        if xyz.shape != (3,) or quat.shape != (4,) or not np.isfinite(xyz).all() or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1)>1e-9:
            raise ValueError('invalid object pose')
    if not layout_is_clear({n:p[:2] for n,p in episode['position_xyz_m'].items()},
                           profile['episode_instance']['minimum_gap_m'], task['size_scale'],
                           profile['episode_instance']['phone_approach_clearance_m']):
        raise ValueError('overlapping episode layout')
    rules = profile['episode_instance']
    for key in ('episode_seed','layout_attempt'):
        if type(episode[key]) is not int or episode[key] < 0: raise ValueError('invalid episode provenance')
    if episode['layout_attempt'] >= rules['max_layout_attempts']: raise ValueError('layout sampling budget exceeded')
    if selected != [n for n in profile['task_instance']['support_pool'] if n in selected]: raise ValueError('support order must follow catalog')
    supports = list(selected)
    scales = task['size_scale']
    for name, slot in zip(supports, ('left_outer','left_inner','right_inner','right_outer')):
        xyz = episode['position_xyz_m'][name]
        for value,bounds in zip(xyz[:2],rules['support_position_xy_m'][slot]):
            if not bounds[0] <= value <= bounds[1]: raise ValueError('support position outside slot')
        if not np.isclose(xyz[2],profile['fixed']['table_top_z_m']+BOTTOM_OFFSETS[name]*scales[name]+.0005,rtol=0,atol=1e-12):
            raise ValueError('support height inconsistent with geometry')
        if episode['quaternion_wxyz'][name] != [1.,0.,0.,0.]: raise ValueError('support initial orientation is fixed')
    phone_xy = episode['position_xyz_m']['phone'][:2]
    for value, bounds in zip(phone_xy, rules['phone_position_xy_m']):
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError('phone position outside table-start domain')
    expected = [*phone_xy, profile['fixed']['table_top_z_m']+.0063*scales['phone']+.001]
    if not np.allclose(episode['position_xyz_m']['phone'],expected,rtol=0,atol=1e-12): raise ValueError('phone table-start position inconsistent with geometry')
    if episode['quaternion_wxyz']['phone'] != profile['fixed']['initial_quaternion_wxyz']['phone']: raise ValueError('phone initial orientation is fixed')
