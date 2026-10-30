import threading
import numpy as np
import pytest

from integrations.harness_vla.bridge import HarnessVLAPolicy, EpisodeEnded


def start(session, **kwargs):
    policy = HarnessVLAPolicy(session, action_timeout_s=.5, **kwargs)
    policy.set_action_dim(2)
    policy.begin_episode(phase='practice', episode_index=0)
    return policy


def test_chunk_rendezvous_and_terminal_cuts_remaining_actions():
    received = []
    def session(env, context, memory):
        received.append(env.reset())
        received.append(env.chunk_step([[.1, .2], [.3, .4], [.5, .6]], return_all_frames=True))
    policy = start(session)
    assert np.allclose(policy.act({'tick': 0}, {}), [.1, .2])
    assert len(received) == 1
    assert np.allclose(policy.act({'tick': 1}, {'round': 2}), [.3, .4])
    policy.end_episode(observation={'tick': 2}, info={'score': 30}, reward=3, terminated=True)
    frames, rewards, term, trunc, info = received[1]
    assert frames == [{'tick': 1}, {'tick': 2}]
    assert rewards == [None, 3]
    assert list(term) == [False, True]
    assert policy._worker is None


def test_memory_survives_episodes_but_not_new_policy():
    def session(env, context, memory):
        memory['count'] = memory.get('count', 0) + 1
        env.step([0, 0])
    policy = start(session)
    for i in range(15):
        policy.begin_episode(phase='practice' if i < 10 else 'exam', episode_index=i)
        policy.act({}, {})
        policy.end_episode(observation={}, info={}, truncated=True)
    assert policy.memory['count'] == 15
    assert policy.memory['last_episode_feedback']['truncated'] is True
    assert start(session).memory == {}


@pytest.mark.parametrize('action', [[0], [0, np.nan], [0, 1.01]])
def test_invalid_actions_are_not_clipped(action):
    policy = start(lambda env, *_: env.step(action))
    try:
        with pytest.raises(RuntimeError, match='session failed') as exc:
            policy.act({}, {})
        assert isinstance(exc.value.__cause__, ValueError)
    finally:
        policy.close()


def test_reset_cannot_restart_scene():
    def session(env, *_):
        env.reset()
        env.reset()
    policy = start(session)
    try:
        with pytest.raises(RuntimeError, match='session failed') as exc:
            policy.act({}, {})
        assert 'benchmark may reset' in str(exc.value.__cause__)
    finally:
        policy.close()


def test_exception_and_early_finish_do_not_manufacture_actions():
    policy = start(lambda *_: None)
    try:
        with pytest.raises(RuntimeError, match='before benchmark termination'):
            policy.act({}, {})
    finally:
        policy.close()


def test_timeout_cancels_cooperative_worker():
    cancelled = threading.Event()
    def session(env, *_):
        env.cancelled.wait()
        cancelled.set()
    policy = HarnessVLAPolicy(session, action_timeout_s=.02)
    policy.set_action_dim(2)
    policy.begin_episode(phase='exam', episode_index=0)
    with pytest.raises(TimeoutError):
        policy.act({}, {})
    policy.close()
    assert cancelled.is_set()


def test_step_after_terminal_is_rejected():
    stopped = []
    def session(env, *_):
        env.step([0, 0])
        with pytest.raises(EpisodeEnded):
            env.step([0, 0])
        stopped.append(True)
    policy = start(session)
    policy.act({}, {})
    policy.end_episode(observation={}, info={}, truncated=True)
    assert stopped == [True]


def test_failure_during_terminal_cleanup_is_reported():
    def session(env, *_):
        env.step([0, 0])
        raise ValueError('artifact write failed')
    policy = start(session)
    policy.act({}, {})
    with pytest.raises(RuntimeError, match='session failed'):
        policy.end_episode(observation={}, info={}, truncated=True)
