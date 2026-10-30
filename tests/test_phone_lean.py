"""Config-only sampling and curved-support coverage for phone_lean."""
import json
from pathlib import Path

import mujoco
import pytest

from react_manip.builtin import discover_task_definitions
from react_manip.core.geometry import (
    GeometryFeatureAdapter,
    PhoneLeanGeometry,
)
from tasks.phone_lean import env as env_module
from tasks.phone_lean.env import PhoneLean, PhoneLeanArena, measure_lean
from tasks.phone_lean.instances import (
    digest,
    load_profile,
    sample_episode_instance,
    sample_task_instance,
    validate_episode_instance,
    validate_task_instance,
)
from tasks.phone_lean.leaning import screen_up_orientation
from tasks.phone_lean.supports import DIMENSIONS, SUPPORT_POOL


def reseal(task):
    task['task_instance_id'] = 'phone_lean_task_' + digest(
        {key: value for key, value in task.items() if key != 'task_instance_id'})
    return task


def test_geometry_adapter_uses_phone_lean_state_key():
    assert GeometryFeatureAdapter._registry['phone_lean'] is PhoneLeanGeometry


def test_sampling_has_two_large_two_small_without_target_roles():
    profile = load_profile()
    rules = profile['task_instance']
    groups = rules['size_groups']
    seen = set()
    for seed in range(1000):
        task = sample_task_instance(seed)
        validate_task_instance(task)
        seen.update(task['support_names'])
        assert len(task['support_names']) == 4
        assert len(task['small_support_names']) == 2
        assert len(task['large_support_names']) == 2
        assert set(task['small_support_names']).isdisjoint(task['large_support_names'])
        assert set(task['small_support_names']) | set(task['large_support_names']) == set(task['support_names'])
        assert 'witness_support' not in task and 'forbidden_support' not in task
        assert 'book' not in task['small_support_names']
        extents = {
            name: groups['nominal_max_extent_m'][name] * task['size_scale'][name]
            for name in task['support_names']
        }
        assert min(extents[name] for name in task['large_support_names']) > max(
            extents[name] for name in task['small_support_names'])
        for name in task['support_names']:
            lo, hi = rules['mass_kg'][name]
            if name in task['small_support_names']:
                scale = task['size_scale'][name]
                lo, hi = lo * scale**3, hi * scale**3
            assert lo <= task['mass_kg'][name] <= hi
    assert seen == set(SUPPORT_POOL)


def test_book_is_large_thin_tall_upright_and_front_faces_robot():
    profile = load_profile()
    groups = profile['task_instance']['size_groups']
    assert DIMENSIONS['book'] == pytest.approx((.110, .004, .30))
    for seed in range(1000):
        task = sample_task_instance(seed)
        if 'book' not in task['support_names']:
            continue
        scale = task['size_scale']['book']
        assert 'book' in task['large_support_names']
        assert 2 * DIMENSIONS['book'][1] * scale <= groups['book_max_thickness_m']
        assert DIMENSIONS['book'][2] * scale >= groups['book_min_realized_height_m']
        episode = sample_episode_instance(task, 3)
        assert episode['quaternion_wxyz']['book'] == [1., 0., 0., 0.]
        break
    else:
        pytest.fail('book was never sampled')

    arena = PhoneLeanArena(('book', 'bottle', 'can', 'vase'))
    model = mujoco.MjModel.from_xml_string(arena.get_xml())
    book = model.body('book').id
    collision = [
        index for index in range(model.ngeom)
        if model.geom_bodyid[index] == book
        and (model.geom_contype[index] != 0 or model.geom_conaffinity[index] != 0)
    ]
    assert collision
    assert all(model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX for index in collision)
    y_min = min(model.geom_pos[index, 1] - model.geom_size[index, 1] for index in collision)
    y_max = max(model.geom_pos[index, 1] + model.geom_size[index, 1] for index in collision)
    z_max = max(model.geom_pos[index, 2] + model.geom_size[index, 2] for index in collision)
    x_min = min(model.geom_pos[index, 0] - model.geom_size[index, 0] for index in collision)
    x_max = max(model.geom_pos[index, 0] + model.geom_size[index, 0] for index in collision)
    assert y_max - y_min == pytest.approx(.008)
    assert z_max == pytest.approx(.30)
    assert x_max - x_min == pytest.approx(.22)
    assert model.geom('book_front_cover').pos[1] < 0


def test_every_non_book_support_has_no_box_collision_geom():
    groups = [
        ('cup', 'bowl', 'pyramid', 'alarm_clock'),
        ('book', 'bottle', 'can', 'vase'),
        ('cone', 'dumbbell', 'cup', 'bowl'),
    ]
    checked = set()
    for selected in groups:
        model = mujoco.MjModel.from_xml_string(PhoneLeanArena(selected).get_xml())
        for name in selected:
            if name == 'book':
                continue
            body = model.body(name).id
            collision = [
                index for index in range(model.ngeom)
                if model.geom_bodyid[index] == body
                and (model.geom_contype[index] != 0 or model.geom_conaffinity[index] != 0)
            ]
            assert collision
            assert all(model.geom_type[index] != mujoco.mjtGeom.mjGEOM_BOX
                       for index in collision), name
            checked.add(name)
    assert checked == set(SUPPORT_POOL) - {'book'}


def test_profile_has_no_feasibility_or_critical_mass_rules():
    profile = load_profile()
    assert 'feasibility' not in profile
    groups = profile['task_instance']['size_groups']
    assert not any('critical' in key or 'witness' in key or 'forbidden' in key
                   for key in groups)
    assert groups['scoring_rule'] == 'all four selected supports are eligible to score'


def test_make_env_never_advances_seed_after_failure(monkeypatch):
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise ValueError('unsolvable by construction')
    monkeypatch.setattr(env_module, 'PhoneLean', fail)
    with pytest.raises(ValueError, match='unsolvable'):
        env_module.make_env(seed=17, robot_id='dual_piperx')
    assert len(calls) == 1
    assert calls[0]['task_seed'] == 17
    assert calls[0]['episode_seed'] == 0


@pytest.mark.parametrize('angle,accepted', [
    (45.0, True),
    (75.0, True),
    (44.999, False),
    (75.001, False),
])
def test_phone_angle_interval_is_45_to_75_inclusive(angle, accepted):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="table" type="box" pos="0 0 -.01" size="1 1 .01"/>
      <body name="support" pos="1 0 .08"><geom type="sphere" size=".08"/></body>
      <body name="phone" pos="0 0 .2"><freejoint/>
        <geom type="box" size=".0063 .0385 .08" mass=".18"/>
        <geom name="phone_bottom_round" type="cylinder" size=".0063 .0385"
              contype="0" conaffinity="0" mass="0"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    rotation = screen_up_orientation([0, 1], 90.0 - angle)
    mujoco.mju_mat2Quat(data.qpos[3:7], rotation.reshape(-1))
    mujoco.mj_forward(model, data)
    sample = measure_lean(model, data, 'support')
    assert sample['table_angle_deg'] == pytest.approx(angle)
    assert sample['upright_lean'] is accepted


def test_validation_rejects_config_out_of_range_mass():
    task = sample_task_instance(0)
    name = task['large_support_names'][0]
    task['mass_kg'][name] = load_profile()['task_instance']['mass_kg'][name][1] + 1
    with pytest.raises(ValueError):
        validate_task_instance(reseal(task))


def test_episode_replay_contract_and_registration():
    task = sample_task_instance(9)
    episode = sample_episode_instance(task, 4)
    validate_episode_instance(task, episode)
    assert episode == sample_episode_instance(task, 4)
    assert task == sample_task_instance(9)
    assert any(spec.task_id == 'phone_lean' for spec, _ in discover_task_definitions())
    contract = json.loads((Path(__file__).resolve().parents[1]
                           / 'tasks/phone_lean/instance_contract.json').read_text())
    assert contract['sampling_profile'] == load_profile()


def test_episode_record_schema_accepts_explicit_phone_lean_scene():
    from react_manip.instances.records import VERSION, seal

    task = sample_task_instance(0)
    scene = sample_episode_instance(task, 0)
    record = seal(
        'episode_instance',
        task_id='phone_lean',
        task_instance_id='phone_lean_000',
        task_instance_digest='0' * 64,
        episode_id='phone_lean_000_e000',
        initial_state={
            'format': 'mujoco.integration.v1',
            'state_spec': 1,
            'integration_state': [0.0],
        },
        round_plan=None,
        provenance={'sampler': VERSION, 'seed': 0},
        scene=scene,
    )
    assert record['scene'] == scene


def test_language_allows_any_support_and_disclaims_solvability():
    env = object.__new__(PhoneLean)
    env.horizon = 3600
    env.control_freq = 20
    text = env.get_episode_language_prompt()['text']
    assert 'any one of the four selected objects' in text
    assert 'All four objects are eligible to score' in text
    assert 'may make a stable lean impossible' in text
    assert '45-75 degrees' in text
    assert 'ten consecutive seconds' in text
