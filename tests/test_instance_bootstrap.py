"""Explicit episode scenes must be honored before the first replay reset."""
from react_manip.core.scenarios import Scenario
from react_manip.instances.runtime import _construct


def test_saved_bootstrap_scene_is_forwarded_without_mutating_record(monkeypatch):
    import react_manip.builtin
    from types import SimpleNamespace
    received = {}
    def factory(**kwargs):
        received.update(kwargs)
        kwargs['episode_instance']['episode_seed'] = 99
        return 'environment'
    monkeypatch.setattr(react_manip.builtin, 'discover_task_definitions',
                        lambda **kwargs: [(SimpleNamespace(task_id='phone_lean'), factory)])
    scene = {'episode_seed': 3}
    scenario = Scenario('1.0', 'phone_lean', 'example', 0, {},
                        metadata={'initial_episode_scene': scene})
    assert _construct('phone_lean', scenario, 'piperx') == 'environment'
    assert received['robot_id'] == 'piperx'
    assert scene == {'episode_seed': 3}


def test_explicit_episode_takes_precedence_over_bootstrap(monkeypatch):
    import react_manip.builtin
    from types import SimpleNamespace
    monkeypatch.setattr(react_manip.builtin, 'discover_task_definitions',
                        lambda **kwargs: [(SimpleNamespace(task_id='phone_lean'), lambda **kw: kw)])
    scenario = Scenario('1.0', 'phone_lean', 'example', 0, {},
                        metadata={'initial_episode_scene': {'episode_seed': 3}})
    result = _construct('phone_lean', scenario, 'piperx', episode={'episode_seed': 7})
    assert result['episode_instance'] == {'episode_seed': 7}


def test_turntable_accepts_explicit_headless_renderer_flag():
    import json
    from pathlib import Path
    from react_manip.core.scenarios import Scenario
    from react_manip.instances.runtime import spec_for
    from tasks.turntable_gamble1.instances import sample_task_instance
    root = Path(__file__).resolve().parents[1]
    profile = json.loads(
        (root/'configs/turntable_gamble1/profile_default.json').read_text())
    scenario = Scenario('1.0', 'turntable_gamble1',
                        'turntable_gamble1_0', 0,
                        sample_task_instance(0, profile))
    robot = spec_for('turntable_gamble1').supported_robots[0]
    env = _construct('turntable_gamble1', scenario, robot, render=False)
    try:
        env.reset()
        assert env.env.has_renderer is False
    finally:
        env.close()
