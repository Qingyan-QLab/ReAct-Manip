"""Manual CLI resolves task embodiments before opening a window."""
import sys
from types import SimpleNamespace
import pytest
from react_manip.builtin import discover_task_definitions
from scripts import pygame_demo

class FactoryReached(Exception):
    pass

@pytest.mark.parametrize('task,override,expected', [
    ('curling_push', None, 'piperx'),
    ('find_object_in_drawer2', None, 'dual_piperx'),
    ('laptop_into_drawer', None, 'mobile_dual_piperx'),
    ('curling_push', 'piperx', 'piperx'),
    ('curling_push', 'dual_piperx', None),
])
def test_manual_cli_robot_selection(task, override, expected, monkeypatch, capsys):
    from react_manip import builtin, robot_registry
    from react_manip.core import registry
    spec = next(s for s, _ in discover_task_definitions() if s.task_id == task)
    calls = []
    def factory(**kwargs):
        calls.append(kwargs['robot_id'])
        raise FactoryReached
    definition = SimpleNamespace(spec=spec, factory=factory)
    monkeypatch.setitem(sys.modules, 'pygame', SimpleNamespace(quit=lambda: None))
    monkeypatch.setattr(builtin, 'register_builtin_tasks', lambda: None)
    monkeypatch.setattr(robot_registry, 'register_default_robots', lambda: None)
    monkeypatch.setattr(registry, 'get_task_registry', lambda: SimpleNamespace(get=lambda name: definition))
    argv = ['--task', task, '--seed', '0']
    if override is not None:
        argv += ['--robot', override]
    if expected is None:
        with pytest.raises(SystemExit) as exc:
            pygame_demo.main(argv)
        assert exc.value.code == 2
        assert 'does not support' in capsys.readouterr().err
        assert calls == []
    else:
        with pytest.raises(FactoryReached):
            pygame_demo.main(argv)
        assert calls == [expected]
