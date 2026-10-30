"""Pan reset pose must support the pie before any toss command."""
import numpy as np
import pytest
from react_manip.builtin import discover_task_definitions
from scripts.pygame_demo import KeyboardPoseControl


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_pan_starts_level_forward_and_supports_pie(seed):
    _, factory = next((s,f) for s,f in discover_task_definitions() if s.task_id == 'pan_flip')
    env = factory(seed=seed, has_offscreen_renderer=False)
    try:
        env.reset()
        raw=env.env;robot=raw.robots[0]
        np.testing.assert_allclose(raw.pan_mat(), np.eye(3), atol=1e-6)
        distance=raw.pan_radius-.02+raw.handle_length*raw.grip_fraction
        np.testing.assert_allclose(raw.pan_pos()-raw.eef_pos(), [distance,0,0], atol=1e-6)
        cam=raw.sim.model.camera_name2id('robot0_wrist_right')
        R=raw.sim.data.cam_xmat[cam].reshape(3,3)
        # Validate the embodiment-local contract rather than assuming a
        # particular world orientation for this task's eef frame.  The
        # sensor is mounted above/rearward and looks toward the tool target.
        body=raw.sim.model.cam_bodyid[cam]
        body_R=raw.sim.data.xmat[body].reshape(3,3)
        local_R=body_R.T@R
        position=np.array([-.1,0.,-.08]);target=np.array([0.,0.,.06])-position
        view=target/np.linalg.norm(target)
        up=np.array([-1.,0.,0.]);up-=np.dot(up,view)*view;up/=np.linalg.norm(up)
        np.testing.assert_allclose(local_R[:,1],up,atol=1e-6)
        np.testing.assert_allclose(-local_R[:,2],view,atol=1e-6)
        manual=KeyboardPoseControl(robot,raw.control_timestep)
        for _ in range(100):
            env.step(manual.action(env.action_dim,robot.composite_controller._action_split_indexes,
                                   'right',np.zeros(6),{'right':-1}))
            assert raw._failure is None
            assert raw.pan_mat()[2,2] > np.cos(np.deg2rad(1))
        local=raw.pan_mat().T@(raw.pie_pos()-raw.pan_pos())
        assert np.linalg.norm(local[:2])+raw.pie_radius < raw.pan_radius
        assert abs(local[2]-raw.pie_half_thickness)<.001
        assert raw.phase=='ready' and not raw._finished
    finally:
        env.close()
