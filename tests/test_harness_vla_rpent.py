"""Offline tests against real RPent schemas/storage; no LLM or VLA service."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

# Run with RPent base dependencies installed and its checkout on PYTHONPATH.
pytest.importorskip('rpent.tools', reason='RPent base dependencies are required')

from integrations.harness_vla import rpent_session


def observation():
    return dict(
        images={'main': np.zeros((4, 4, 3), dtype=np.uint8)},
        depth_m={'main': np.ones((4, 4), dtype=np.float32)},
        camera_calibration={'main': {'intrinsics': np.eye(3), 'world_from_camera': np.eye(4)}},
        robot0_eef_pos=np.zeros(3), robot0_eef_quat_site=np.array([0, 0, 0, 1]),
        control_contract=dict(arms={'right': dict(input_min=[-1]*6, input_max=[1]*6,
                                                  output_min=[-.015]*3+[-.08]*3,
                                                  output_max=[.015]*3+[.08]*3)},
                              action_layout={'right': [0, 6], 'right_gripper': [6, 7]}))


@pytest.mark.parametrize('backend', ['api', 'codex'])
def test_real_tool_schema_artifacts_memory_and_planner_binding(tmp_path, monkeypatch, backend):
    import rpent.planner.base
    counts = []
    class Planner:
        def solve(self, *, toolkit, **kwargs):
            names = {tool.name for tool in toolkit.list_tools()}
            assert 'vla_act' in names and 'move_to' in names
            assert not names.intersection({'reset', 'execute_action', 'read_text_file', 'list_dir'})
            prior = toolkit.execute_tool('read_memory', {'layer': 'task-specific'})
            count = int(prior.data['content'] or 0) + 1
            counts.append(count)
            assert not toolkit.execute_tool('write_memory', {'layer': 'task-specific', 'content': str(count)}).is_error
            view = toolkit.execute_tool('observe', {})
            assert view.images
            assert view.data['episode'] == dict(elapsed_sim_sec=10., horizon_sim_sec=60.,
                                               remaining_sim_sec=50., terminated=False,
                                               truncated=False)
            target = toolkit.execute_tool('locate', {'camera': 'main', 'row': 1, 'column': 1})
            assert target.data['xyz'] == [1, 1, 1]
            moved = toolkit.execute_tool('move_to', {'xyz': [.1, 0., 0.], 'max_steps': 1})
            assert not moved.is_error
            return SimpleNamespace(finish_result={'status': 'stuck'}, stats={}, messages=[], error=None)
    planner_calls = []
    def factory(*args, **kwargs):
        planner_calls.append((args, kwargs))
        return Planner()
    monkeypatch.setattr(rpent.planner.base, 'build_planner', factory)
    from integrations.harness_vla import codex_planner
    monkeypatch.setattr(codex_planner, 'build_codex_planner', factory)
    fake_vla = SimpleNamespace(provenance=dict(weights_frozen=True, checkpoint_id='test-only',
                                              action_schema='react-manip.osc.v1'))
    monkeypatch.setattr(rpent_session, 'load_factory', lambda _: lambda _: fake_vla)
    policy = rpent_session.build(dict(vla_factory='test:build', planner_model='test',
        planner_backend=backend, planner_reasoning_effort='high'), tmp_path)
    policy.set_action_dim(7)
    try:
        for i, phase in enumerate(['practice', 'exam']):
            policy.begin_episode(phase=phase, episode_index=i,
                                 task_spec=SimpleNamespace(horizon_sec=60.))
            assert policy.act(observation(), {'language_prompt': 'test', 'elapsed_sec': 10.})[0] == 1
            hold = policy.act(observation(), {})
            assert np.allclose(hold[:6], 0)
            policy.end_episode(observation=observation(), info={}, truncated=True)
        assert counts == [1, 2]
        assert policy.provenance['planner_backend'] == backend
        assert len(planner_calls) == 2
        for args, kwargs in planner_calls:
            assert args == (('api',) if backend == 'api' else ())
            assert kwargs['model'] == 'test'
            assert kwargs['reasoning_effort'] == 'high'
            assert Path(kwargs['memory_dir']).is_relative_to(tmp_path)
        assert len(list(tmp_path.rglob('states.json'))) == 2
        assert len(list(tmp_path.rglob('planner_result.json'))) == 2
        depth_files = list(tmp_path.glob('**/main_depth.npy/*.npy'))
        assert depth_files
        for depth_file in depth_files:
            np.testing.assert_array_equal(np.load(depth_file), observation()['depth_m']['main'])
    finally:
        policy.close()


def test_planner_backend_validation_precedes_service_start(tmp_path):
    for settings in ({'planner_backend':'unknown'},
                     {'planner_backend':'codex','planner_base_url':'https://example.invalid'}):
        with pytest.raises(ValueError):
            rpent_session.build(dict(vla_factory='missing:factory',planner_model='test',**settings),tmp_path)


def test_model_contract_rejects_unidentified_or_unfrozen_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(rpent_session, 'load_factory', lambda _: lambda _: SimpleNamespace(provenance={}))
    with pytest.raises(ValueError, match='declare frozen weights'):
        rpent_session.build(dict(vla_factory='test:build', planner_model='test'), tmp_path)


@pytest.mark.parametrize('terminal_during_primitive', [False, True])
def test_actual_api_planner_loop_and_terminal_cancellation(tmp_path, monkeypatch, terminal_during_primitive):
    # Exclude cold filesystem imports from the five-second model-loop budget.
    import rpent.planner.api_loop
    import json
    from pydantic_ai.models.function import FunctionModel, DeltaToolCall
    import rpent.planner.base
    calls = []
    async def stream(messages, info):
        calls.append(True)
        if len(calls) == 1:
            yield {0: DeltaToolCall(name='move_to', json_args=json.dumps(dict(xyz=[.1, 0., 0.], max_steps=1)))}
        else:
            yield {0: DeltaToolCall(name='finish', json_args=json.dumps(dict(status='stuck', summary='test')))}
    monkeypatch.setattr(rpent.planner.base, 'build_api_model', lambda *a: FunctionModel(stream_function=stream))
    fake_vla = SimpleNamespace(provenance=dict(weights_frozen=True, checkpoint_id='test-only',
                                              action_schema='react-manip.osc.v1'))
    monkeypatch.setattr(rpent_session, 'load_factory', lambda _: lambda _: fake_vla)
    policy = rpent_session.build(dict(vla_factory='test:build', planner_model='test',
                                      planner_timeout_s=5, close_timeout_s=10), tmp_path)
    policy.set_action_dim(7)
    policy.begin_episode(phase='practice', episode_index=0)
    try:
        assert policy.act(observation(), {'language_prompt': 'test'})[0] == 1
        if not terminal_during_primitive:
            assert np.allclose(policy.act(observation(), {})[:6], 0)
        policy.end_episode(observation=observation(), info={}, truncated=True)
        assert policy._worker is None
        assert len(calls) == (1 if terminal_during_primitive else 2)
    finally:
        policy.shutdown()
