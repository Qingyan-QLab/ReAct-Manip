"""Deterministic measured-state rule fixtures, not a robot demonstration."""
import numpy as np
import pytest
from itertools import product
from tasks.button_indicator_matching.env import ButtonIndicatorMatching


def fixture():
    e=object.__new__(ButtonIndicatorMatching);e.n_buttons=3;e.assignment=np.arange(3)
    e._prev_pressed=np.zeros(3,dtype=bool);e._press_started_sec=np.full(3,np.nan)
    e._correct_press_awarded=np.zeros(3,dtype=bool);e._target_indicator=0
    e._episode_targets=(0,1,2,0,1);e._target_cursor=1;e._round_index=0
    e._round_started_sec=0.;e._round_results=[];e._score=0.;e._events=[]
    return e


def tick(e,buttons,now,start=None):
    p=np.isin(np.arange(3),buttons)
    return e._process_round(p,p,now-.05 if start is None else start,now)


def test_five_timeouts_clamp_final_score_only():
    e=fixture()
    for n in range(1,6):
        assert tick(e,[],24*n)[0]==-20
        assert len(e._round_results)==n
        assert e.total_score == (-20*n if n < 5 else 0)
    assert e._score==-100 and not e._score_is_success() and e._target_cursor==5


@pytest.mark.parametrize('outcomes', list(product((False, True), repeat=5)))
def test_final_score_independent_of_order(outcomes):
    e = fixture()
    for correct in outcomes:
        start = e._round_started_sec
        tick(e, [], start)
        button = e._target_indicator if correct else (e._target_indicator + 1) % 3
        delta = tick(e, [button], start + .05, start=start)[0]
        if correct:
            delta = tick(e, [button], start + .5)[0]
        assert delta == (20 if correct else -20)
    raw = 40 * sum(outcomes) - 100
    assert e._score == raw
    assert e.total_score == max(0, raw)
    assert e._score_is_success() == all(outcomes)


def test_correct_deadline_and_mixed_score():
    e=fixture()
    for n in range(5):
        start=e._round_started_sec
        if n<3:
            tick(e,[e._target_indicator],start+23.5,start=start+23.5)
            assert tick(e,[e._target_indicator],start+24)[0]==20
        else:
            assert tick(e,[],start+24)[0]==-20
        tick(e,[],start+24) if n<4 else None
    assert e._score==20 and len(e._round_results)==5


def test_wrong_priority_and_held_button_not_reused():
    e=fixture();tick(e,[0],.05,start=0)
    result=tick(e,[0,1,2],.5)
    assert result[0]==-20 and len(e._round_results)==1 and e._target_indicator==1
    assert tick(e,[0,1,2],1)[0]==0
    tick(e,[],1.05)
    tick(e,[1],1.10,start=1.05)
    assert tick(e,[1],1.55)[0]==20


def test_all_five_correct():
    e=fixture()
    for _ in range(5):
        start=e._round_started_sec
        tick(e,[],start)
        tick(e,[e._target_indicator],start+.05,start=start)
        assert tick(e,[e._target_indicator],start+.5)[0]==20
    assert e._score_is_success() and e._score==100


def test_short_hold_release_resets():
    e=fixture();tick(e,[0],.05,start=0);tick(e,[],.45)
    tick(e,[0],.5,start=.45)
    assert tick(e,[0],.9)[0]==0
    assert tick(e,[0],.95)[0]==20
