"""Runtime language reaches policy info without JSON fallbacks or stale rounds."""
from types import SimpleNamespace

import numpy as np
import pytest

from react_manip.builtin import discover_task_definitions
from react_manip.core.language import runtime_language_prompt
from react_manip.legacy import LegacyTaskAdapter


@pytest.mark.parametrize('spec,factory', discover_task_definitions(), ids=lambda x: getattr(x, 'task_id', 'factory'))
def test_each_release_task_reset_and_step_publish_runtime_language(spec, factory):
    assert 'language' not in spec.metadata
    env = factory(seed=0, has_offscreen_renderer=False, use_camera_obs=False)
    try:
        env.reset()
        prompt = env.get_language_prompt()
        assert prompt['task_instruction'].strip() and prompt['text'].strip()
        import json
        assert json.dumps(prompt, ensure_ascii=False).isascii()
        assert 'round_instruction' in prompt and 'round_index' in prompt
        result = env.step(np.zeros(env.action_dim))
        info = result[-1]
        assert info['language_prompt_structured'] == env.get_language_prompt()
        assert info['language_prompt'] == env.get_language_prompt()['text']
    finally:
        env.close()


def test_missing_or_invalid_runtime_interface_does_not_fall_back_to_json():
    env = SimpleNamespace(task_spec=SimpleNamespace(metadata={'language': {'instruction': 'stale'}}))
    with pytest.raises(RuntimeError, match='get_episode_language_prompt'):
        runtime_language_prompt(env)
    env.get_episode_language_prompt = lambda: None
    with pytest.raises(TypeError, match='mapping'):
        runtime_language_prompt(env)
    env.get_episode_language_prompt = lambda: {'task_instruction': ''}
    with pytest.raises(ValueError, match='non-empty'):
        runtime_language_prompt(env)


def test_adapter_replaces_stale_step_info_when_round_advances():
    class Task:
        round_index = 0

        def reset(self):
            self.round_index = 0
            return {}, {}

        def get_episode_language_prompt(self):
            return {'task_instruction': 'Complete two rounds.',
                    'round_instruction': f'Target {self.round_index}',
                    'round_index': self.round_index}

        def step(self, action):
            self.round_index += 1
            return {}, 0, False, False, {'language_prompt': 'stale', 'language_prompt_structured': {}}

    env = LegacyTaskAdapter(Task())
    _, info = env.reset()
    assert info['language_prompt_structured']['round_index'] == 0
    info = env.step([])[-1]
    assert info['language_prompt_structured']['round_index'] == 1
    assert 'Target 1' in info['language_prompt'] and 'Target 0' not in info['language_prompt']


def test_button_finished_prompt_clears_target_without_revealing_future_plan():
    _, factory = next((s, f) for s, f in discover_task_definitions() if s.task_id == 'button_indicator_matching')
    env = factory(seed=0, has_offscreen_renderer=False, use_camera_obs=False)
    try:
        env.reset()
        assert env.get_language_prompt()['round_index'] == 0
        raw = env.env
        for _ in range(5):
            raw._process_round(np.zeros(raw.n_buttons, dtype=bool), np.zeros(raw.n_buttons, dtype=bool),
                               raw._round_started_sec, raw._round_started_sec + 24)
        prompt = env.get_language_prompt()
        assert prompt['round_instruction'] == '' and prompt['round_index'] is None
        assert prompt['public_context']['target'] is None
        assert prompt['public_context']['completed_rounds'] == 5
        assert 'assignment' not in prompt['public_context']
        assert 'targets' not in prompt['public_context']
        raw._round_results = []
        raw._finished = True
        assert env.get_language_prompt()['round_index'] is None
    finally:
        env.close()


@pytest.mark.parametrize('layout', ['draped', 'folded'])
def test_rope_both_layouts_provide_complete_runtime_rules(layout):
    from tasks.rope_hang.env import RopeHang
    env = object.__new__(RopeHang)
    env.initial_layout = SimpleNamespace(layout=layout)
    env.horizon = 9000
    env.control_freq = 20
    env.robot_id = 'dual_piperx'
    prompt = runtime_language_prompt(env)
    assert '450 seconds' in prompt['text']
    assert 'five continuous seconds' in prompt['task_instruction']
    assert ('already draped' in prompt['task_instruction']) == (layout == 'draped')
