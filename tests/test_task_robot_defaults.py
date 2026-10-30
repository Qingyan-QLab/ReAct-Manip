"""Task configuration is authoritative for default evaluation embodiments."""
from dataclasses import replace
import pytest
from react_manip.builtin import discover_task_definitions
from react_manip.core.compatibility import resolve_task_robot
from react_manip.evaluation import EvaluationContext


def test_all_release_tasks_select_their_only_robot():
    for spec,factory in discover_task_definitions():
        assert len(spec.supported_robots)==1
        expected=spec.supported_robots[0]
        assert resolve_task_robot(spec)==expected
        context=EvaluationContext(spec,factory,lambda:None)
        assert context.robot_id==expected


def test_ambiguous_or_unsupported_selection_fails():
    spec=next(s for s,_ in discover_task_definitions() if s.task_id=='box_insertion')
    with pytest.raises(ValueError,match='does not support'):
        resolve_task_robot(spec,'dual_piperx')
    ambiguous=replace(spec,supported_robots=('piperx','dual_piperx'))
    with pytest.raises(ValueError,match='multiple supported robots'):
        resolve_task_robot(ambiguous)
    assert resolve_task_robot(ambiguous,'piperx')=='piperx'


def test_special_factory_rejects_override_before_creating_robot():
    spec,factory=next((s,f) for s,f in discover_task_definitions() if s.task_id=='laptop_into_drawer')
    assert resolve_task_robot(spec)=='mobile_dual_piperx'
    with pytest.raises(ValueError,match='does not support'):
        factory(robot_id='dual_piperx')
