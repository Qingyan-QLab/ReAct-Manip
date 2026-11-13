from types import SimpleNamespace

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

from tasks.ring_toss1.env import RingToss1


def test_scored_ring_terminates_before_horizon(monkeypatch):
    monkeypatch.setattr(
        ManipulationEnv,
        "_post_action",
        lambda self, action: (0.0, False, {}),
    )
    env = object.__new__(RingToss1)
    env.sim = SimpleNamespace(
        model=SimpleNamespace(geom_bodyid=[], body_names=[]),
        data=SimpleNamespace(ncon=0, contact=[]),
    )
    env.ring_ids = [0]
    env.awarded = {0: 0}
    env.points = [60]
    env.total_score = 60
    env.last_target = 0
    env.flight = [0.1]
    env.lifted = [True]
    env.settled = [0.5]
    env.num_rings = 1
    env.instance = {"seed": 0, "task": {"flight_sec": 0.1}}

    reward, done, info = env._post_action(None)

    assert reward == 0.0
    assert done is True
    assert info["terminal_success"] is True
    assert info["termination_reason"] == "success"
