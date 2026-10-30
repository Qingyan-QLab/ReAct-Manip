from types import SimpleNamespace
import numpy as np
import pytest

from integrations.harness_vla.rgbd import RGBDSensor, back_project


def test_metric_depth_projection_and_read_only_render():
    import mujoco
    model = mujoco.MjModel.from_xml_string('''<mujoco><visual><global offwidth="160" offheight="120"/></visual>
    <worldbody><geom type="plane" size="5 5 .01" rgba=".5 .5 .5 1"/>
    <camera name="sensor" pos="0 0 2" fovy="60"/>
    <body pos="0 0 .2"><freejoint/><geom type="box" size=".1 .1 .2" mass="1"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = SimpleNamespace(model=model, data=data)
    state_type = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.zeros(mujoco.mj_stateSize(model, state_type))
    after = before.copy()
    mujoco.mj_getState(model, data, before, state_type)
    sensor = RGBDSensor(160, 120)
    try:
        rgb, depth, valid, calibration = sensor.capture(sim, 'sensor')
        assert rgb.shape == (120, 160, 3) and rgb.dtype == np.uint8
        assert depth.shape == valid.shape == (120, 160)
        assert depth.dtype == np.float32 and valid.dtype == bool
        assert depth[60, 80] == pytest.approx(1.6, abs=.001)
        assert depth[20, 20] == pytest.approx(2, abs=.001)
        points = back_project(depth, calibration, np.array([[20, 20], [90, 130]]))
        assert np.allclose(points[:, 2], 0, atol=.001)
        assert points[0, 0] < 0 and points[0, 1] > 0
        assert points[1, 0] > 0 and points[1, 1] < 0
        rgb2, depth2, _, _ = sensor.capture(sim, 'sensor')
        assert np.array_equal(rgb, rgb2) and np.array_equal(depth, depth2)
        mujoco.mj_getState(model, data, after, state_type)
        assert np.array_equal(before, after)
    finally:
        sensor.close()


@pytest.mark.parametrize('pixels', [[[-1, 0]], [[2, 0]], [[.2, .1]]])
def test_projection_rejects_invalid_pixels(pixels):
    with pytest.raises(ValueError):
        back_project(np.ones((2, 2)), {}, np.array(pixels))


def test_projection_rejects_background():
    with pytest.raises(ValueError, match='valid metric depth'):
        back_project(np.zeros((2, 2)), {}, np.array([[0, 0]]))
