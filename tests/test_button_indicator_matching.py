"""Release environment terminal scores keep signed rewards and reset cleanly."""
import numpy as np
import pytest
from react_manip.builtin import discover_task_definitions


@pytest.mark.parametrize('truncate,raw', [(False,-80.), (False,80.), (True,-80.)])
def test_final_score_outputs(truncate,raw):
    spec,factory=next((s,f) for s,f in discover_task_definitions() if s.task_id=='button_indicator_matching')
    env=factory(seed=22,has_offscreen_renderer=False,use_camera_obs=False)
    try:
        env.reset(); inner=env.env
        inner._score=raw
        inner._round_results=[{'outcome':'timeout','score':-20.} for _ in range(4)]
        if truncate:inner.horizon=1
        else:inner._round_started_sec=float(inner.sim.data.time)-24.
        obs,reward,terminated,truncated,info=env.step(np.zeros(env.action_dim))
        expected_raw=raw if truncate else raw-20
        assert terminated == (not truncate) and truncated == truncate
        assert reward == info['score_delta'] == (0 if truncate else -20)
        assert info['raw_score'] == expected_raw
        assert info['total_score'] == inner.total_score == float(obs['score'][0]) == max(0,expected_raw)
        assert not info['terminal_success']
        obs,info=env.reset()
        assert info['raw_score']==0 and inner.total_score==0
    finally:env.close()
