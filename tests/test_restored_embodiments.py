"""The restored models must remain distinct and expose matching contact surfaces."""
import numpy as np
import pytest
from react_manip.robots.resolution import resolve_robosuite_robot
from react_manip.robots.agilex_adapter import AgileXAdapter
from react_manip.core.compatibility import resolve_task_robot
from react_manip.builtin import discover_task_definitions


def test_restored_models_do_not_resolve_to_official_piperx():
    expected = {'legacy_piperx':'AgileXPiper', 'aloha':'AgileXAloha',
                'aloha_mobile':'AgileXMobileAloha', 'piperx':'PiperX',
                'dual_piperx':'DualPiperX', 'mobile_dual_piperx':'MobileDualPiperX',
                'panda':'Panda', 'sawyer':'CLSawyer'}
    for key, name in expected.items():
        assert resolve_robosuite_robot(key, include_controller=False).robot_name == name


@pytest.mark.parametrize('robot', ['legacy_piperx', 'aloha', 'aloha_mobile'])
def test_legacy_gripper_contacts_match_rendered_meshes(robot):
    model = AgileXAdapter(robot).load_mujoco_model()
    count = 0
    for geom in range(model.ngeom):
        name = model.geom(geom).name
        if not name.startswith('gripper') or '_collision' not in name:
            continue
        visual = model.geom(name.replace('_collision','_visual')).id
        assert model.geom_dataid[geom] == model.geom_dataid[visual]
        np.testing.assert_array_equal(model.geom_pos[geom], model.geom_pos[visual])
        np.testing.assert_array_equal(model.geom_quat[geom], model.geom_quat[visual])
        assert model.geom_contype[geom] and model.geom_conaffinity[geom]
        assert not model.geom_contype[visual] and not model.geom_conaffinity[visual]
        count += 1
    assert count > 2
    assert all(not model.geom(i).name.endswith(('_pad','_flat_pad')) for i in range(model.ngeom))


def test_optional_embodiments_preserve_task_defaults():
    for spec, _ in discover_task_definitions():
        default = resolve_task_robot(spec)
        assert default in ('piperx', 'dual_piperx', 'mobile_dual_piperx')
        for robot in spec.supported_robots:
            assert resolve_task_robot(spec, robot) == robot
