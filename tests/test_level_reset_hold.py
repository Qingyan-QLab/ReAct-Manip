"""Scene settling must hold both arms and grippers without position clipping."""
import numpy as np
import pytest
from react_manip.builtin import discover_task_definitions


def test_every_settling_step_holds_robot_and_grippers(monkeypatch):
    from robosuite.utils.binding_utils import MjSim
    original = MjSim.step
    samples = []
    def step(sim, *args, **kwargs):
        result = original(sim, *args, **kwargs)
        m, d = sim.model, sim.data
        names = [n for n in m.joint_names if n and n.startswith(('robot', 'gripper'))]
        values = {n: d.qpos[m.get_joint_qpos_addr(n)].copy() for n in names}
        samples.append(values)
        return result
    monkeypatch.setattr(MjSim, 'step', step)
    _, factory = next((s, f) for s, f in discover_task_definitions() if s.task_id == 'leveling_board1')
    env = factory(seed=0, has_offscreen_renderer=False, use_camera_obs=False)
    try:
        for _ in range(2):
            samples.clear()
            env.reset()
            assert len(samples) == 300
            for sample in samples:
                for arm in ('right', 'left'):
                    np.testing.assert_allclose([sample[f'gripper0_{arm}_finger_joint7'],
                                                sample[f'gripper0_{arm}_finger_joint8']], [.04, -.04], atol=1e-6)
                    np.testing.assert_allclose([sample[f'robot0_{arm}_joint{i}'] for i in range(1, 7)],
                                               [0., 1.2, -1.1, 1.2, 0., 0.], atol=1e-5)
            env.env._validate_robot_initial_state()
            assert env.sim.data.time == 0.
            assert abs(env.env.board_angle_deg) > .5
        # The guard rejects a bad joint instead of silently clipping it.
        q = env.sim.model.get_joint_qpos_addr('gripper0_right_finger_joint7')
        env.sim.data.qpos[q] = -.001
        with pytest.raises(ValueError, match='outside limits'):
            env.env._validate_robot_initial_state()
        assert env.sim.data.qpos[q] == -.001
    finally:
        env.close()
