"""Five-round language transitions use public targets only."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.find_object_in_drawer2.env import FindObjectInDrawer2, ManipulationEnv
from tasks.find_object_in_drawer2.objects import OBJECT_NAMES, LABELS


def fixture():
    e=object.__new__(FindObjectInDrawer2)
    e.object_names=list(OBJECT_NAMES)
    e.target_count=len(e.object_names)
    e.object_drawers=np.arange(e.target_count)%6
    e.opened_drawers=np.ones(6,dtype=bool)
    e.control_freq=20;e.horizon=12000
    e.robot_id='dual_piperx';e._episode_index=0
    e.scenario=SimpleNamespace(scenario_id='language_test')
    e.done=False
    e.set_episode_targets([0,1,2,3,4])
    return e


def test_five_round_transitions_use_runtime_only():
    e=fixture();fixed=e._task_instruction()
    path=Path(__file__).resolve().parents[1]/'tasks/find_object_in_drawer2/task_spec.json'
    assert 'language' not in json.loads(path.read_text())['metadata']
    for round_index in range(5):
        p=e.get_episode_language_prompt();en=LABELS[e.object_names[e.target_index]][0]
        assert p['task_instruction']==fixed
        assert p['round_index']==p['public_context']['round_index']==round_index
        assert f'Instruction {round_index+1} of 5: find the {en}' in p['round_instruction']
        assert json.dumps(p,ensure_ascii=False).isascii()
        assert 'either arm or both arms' in fixed
        assert p['round_instruction'] in p['text'] and fixed in p['text']
        assert p['round_instruction'] in p['instruction']
        assert 'drawer_index' not in p['public_context'] and 'target_set' not in p['public_context']
        for _ in range(9): assert not e._settle_instruction(True)
        assert e.get_episode_language_prompt()==p
        assert e._settle_instruction(True)
        assert e.total_score==20*(round_index+1)
    p=e.get_episode_language_prompt()
    assert p['round_instruction']=='' and p['round_index'] is None
    assert 'target_name' not in p['public_context']
    assert p['public_context']['completed_rounds']==5
    assert 'All five instructions completed' in p['text']
    assert json.dumps(p,ensure_ascii=False).isascii()
    assert 'Instruction 6' not in p['text']
    e.set_episode_targets([4,3,2,1,0])
    p=e.get_episode_language_prompt()
    assert p['round_index']==0 and p['task_instruction']==fixed
    assert e.total_score==0 and 'toy car' in p['round_instruction']


def test_timeout_clears_pending_instruction_without_awarding_score():
    e=fixture();e.done=True
    p=e.get_episode_language_prompt()
    assert p['round_instruction']=='' and p['round_index'] is None
    assert 'Time limit reached' in p['text']
    assert json.dumps(p,ensure_ascii=False).isascii()
    assert 'target_name' not in p['public_context'] and e.total_score==0


@pytest.mark.parametrize('last', [False,True])
def test_step_publishes_new_round_in_same_transition(monkeypatch,last):
    e=fixture()
    e._round_index=4 if last else 0
    e.target_index=e._targets[e._round_index]
    e.total_score=80 if last else 0
    e._stable_ticks=9
    e._action_dim=1
    e.robots=[SimpleNamespace(composite_controller=SimpleNamespace(_action_split_indexes={}))]
    e.sim=SimpleNamespace(data=SimpleNamespace(qpos=np.full(6,-.1)))
    e.drawer_qpos_addrs=np.arange(6)
    e._placement_state=lambda:dict(box_contact=True,released=True,contained=True,stable=True)
    monkeypatch.setattr(ManipulationEnv,'step',lambda self,action:({},0.,False,{}))
    _,reward,terminated,truncated,info=e.step(np.zeros(1))
    p=info['language_prompt_structured']
    assert info['language_prompt']==p['text']
    assert p==e.get_episode_language_prompt()
    assert terminated==last and not truncated
    assert p['round_index']==(None if last else 1)
    assert info['score_delta']==20 and reward==1.


def test_real_adapter_reset_and_step_keep_language_consistent():
    from react_manip.builtin import discover_task_definitions
    _,factory=next((s,f) for s,f in discover_task_definitions()
                   if s.task_id=='find_object_in_drawer2')
    env=factory(seed=0,has_offscreen_renderer=False,use_camera_obs=False)
    try:
        env.reset()
        first=env.get_language_prompt()
        assert first['round_index']==0 and first['round_instruction']
        assert first['task_instruction']==env.env._task_instruction()
        _,_,_,_,info=env.step(np.zeros(env.action_dim))
        assert info['language_prompt_structured']==env.get_language_prompt()
        assert info['language_prompt']==env.get_language_prompt()['text']
        env.env.done=True
        assert env.get_language_prompt()['round_index'] is None
        env.reset()
        reset=env.get_language_prompt()
        assert reset['round_index']==0 and reset['round_instruction']==first['round_instruction']
        assert reset['task_instruction']==first['task_instruction']
    finally:
        env.close()


def test_both_arm_and_gripper_commands_reach_physics(monkeypatch):
    e = fixture()
    e._action_dim = 14
    e.robots = [SimpleNamespace(composite_controller=SimpleNamespace(
        _action_split_indexes={'right': (0, 6), 'left': (6, 12),
                              'right_gripper': (12, 13), 'left_gripper': (13, 14)}))]
    e.sim = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(6)))
    e.drawer_qpos_addrs = np.arange(6)
    e._placement_state = lambda: dict(box_contact=False, released=False, contained=False, stable=False)
    received = []

    def physics_step(self, action):
        received.append(action.copy())
        return {}, 0., False, {}

    monkeypatch.setattr(ManipulationEnv, 'step', physics_step)
    action = np.linspace(-.8, .8, 14)
    action[12:] = [.7, .9]
    e.step(action)
    np.testing.assert_array_equal(received[0], action)
    spec = json.loads((Path(__file__).resolve().parents[1] / 'tasks/find_object_in_drawer2/task_spec.json').read_text())
    assert spec['required_arms'] == 2
    assert spec['metadata']['active_arms'] == ['right', 'left']
    assert all('ignored' not in field for field in spec['action']['fields'])
