"""Evaluator-owned RGB-D sensor extension; the policy never receives sim handles."""
from __future__ import annotations

import numpy as np

from . import OBSERVATION_VERSION


def back_project(depth_m, calibration, pixels):
    """Project (row, column) pixels to world metres using sensor calibration."""
    depth = np.asarray(depth_m)
    pixels = np.asarray(pixels)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError('pixels must be an integer [N, 2] array in row/column order')
    rows, cols = pixels.T
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= depth.shape[0]) or np.any(cols >= depth.shape[1]):
        raise ValueError('Pixel is outside the depth image')
    z = depth[rows, cols]
    if not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError('Pixel has no valid metric depth')
    k = np.asarray(calibration['intrinsics'])
    points = np.column_stack(((cols - k[0, 2]) * z / k[0, 0],
                              (rows - k[1, 2]) * z / k[1, 1], z))
    transform = np.asarray(calibration['world_from_camera'])
    return points @ transform[:3, :3].T + transform[:3, 3]


class RGBDSensor:
    """Same visual geometry and top-down orientation as the benchmark RGB view."""

    def __init__(self, width=640, height=480):
        from react_manip.rendering import NativeRGBRenderer
        self._rgb = NativeRGBRenderer(width, height)
        self.width, self.height = width, height

    def capture(self, sim, camera):
        import mujoco
        rgb = self._rgb.render(sim, camera).copy()
        renderer = self._rgb._renderer
        model = self._rgb._model
        data = getattr(sim.data, '_data', sim.data)
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if index < 0:
            raise ValueError('RGB-D requires a named sensor camera')
        if model.cam_orthographic[index] or np.any(model.cam_sensorsize[index]):
            raise ValueError('rgbd.v1 supports perspective fovy cameras only')
        renderer.enable_depth_rendering()
        try:
            # MuJoCo Renderer returns optical-axis distance in metres, not a
            # normalized depth buffer; converting it a second time is wrong.
            depth = renderer.render().copy().astype(np.float32)
        finally:
            renderer.disable_depth_rendering()
        near = float(model.vis.map.znear * model.stat.extent)
        far = float(model.vis.map.zfar * model.stat.extent)
        valid = np.isfinite(depth) & (depth > near) & (depth < far * (1 - 1e-5))
        depth[~valid] = 0
        focal = 0.5 * self.height / np.tan(np.deg2rad(model.cam_fovy[index]) / 2)
        k = np.array([[focal, 0, (self.width - 1) / 2],
                      [0, focal, (self.height - 1) / 2], [0, 0, 1]])
        world_from_camera = np.eye(4)
        world_from_camera[:3, :3] = data.cam_xmat[index].reshape(3, 3) @ np.diag([1, -1, -1])
        world_from_camera[:3, 3] = data.cam_xpos[index]
        return rgb, depth, valid, dict(
            intrinsics=k, world_from_camera=world_from_camera,
            depth_units='metres', depth_convention='optical_axis_z',
            camera_axes='x_right_y_down_z_forward', pixel_order='row_column',
            image_origin='top_left', near_m=near, far_m=far,
            simulation_time_s=float(data.time))

    def close(self):
        self._rgb.close()


def controller_contract(raw):
    """Publish robot control calibration only, without task/object parameters."""
    robot = raw.robots[0]
    controller = robot.composite_controller
    layout = {k: list(v) for k, v in controller._action_split_indexes.items()}
    arms = {}
    for name, part in controller.part_controllers.items():
        if name.endswith('_gripper'):
            continue
        if name not in ('right', 'left') or part.control_dim != 6:
            raise ValueError('Unsupported PiPER-X controller layout')
        if part.input_type != 'delta' or part.input_ref_frame not in ('world', 'base'):
            raise ValueError('Integration requires world/base-frame delta OSC')
        arms[name] = {key: np.asarray(getattr(part, key)).tolist() for key in
                      ('input_min', 'input_max', 'output_min', 'output_max')}
        rotation = np.eye(3) if part.input_ref_frame == 'world' else np.asarray(part.origin_ori)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError('Controller reference-frame calibration is unavailable')
        arms[name].update(input_type=part.input_type, input_ref_frame=part.input_ref_frame,
                          world_from_control_rotation=rotation.tolist())
    return dict(schema_version='react-manip.osc.v1', action_layout=layout, arms=arms,
                gripper_open=-1.0, gripper_close=1.0)


class RGBDReplayEnv:
    """Decorate validated ReplayEnv without changing bank, physics, or score rules."""

    def __init__(self, instance, *, cameras=True):
        if not cameras:
            raise ValueError('RGB-D evaluation requires camera observations')
        from react_manip.instances.runtime import ReplayEnv
        self._replay = ReplayEnv(instance, cameras=True)
        self._sensor = RGBDSensor()
        self._camera_ids = tuple(instance['runtime']['cameras'])

    @property
    def env(self):
        # Used exclusively by benchmark-side observation visibility filtering.
        return self._replay.env

    def __getattr__(self, name):
        return getattr(self._replay, name)

    def bind(self, episode):
        self._sensor.close()
        self._replay.bind(episode)

    def set_camera_observations(self, camera_ids=(), *, width=640, height=480):
        if tuple(camera_ids) != self._camera_ids or (width, height) != (640, 480):
            raise ValueError('RGB-D v1 uses the recorded policy cameras at 640x480')
        self._replay.set_camera_observations(camera_ids, width=width, height=height)

    def _enrich(self, observation):
        images, depths, masks, calibrations = {}, {}, {}, {}
        for camera in self._camera_ids:
            images[camera], depths[camera], masks[camera], calibrations[camera] = self._sensor.capture(self.env.sim, camera)
        return dict(observation, images=images, depth_m=depths, depth_valid=masks,
                    camera_calibration=calibrations, control_contract=controller_contract(self.env),
                    observation_schema=OBSERVATION_VERSION)

    def reset(self, **kwargs):
        observation, info = self._replay.reset(**kwargs)
        return self._enrich(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._replay.step(action)
        return self._enrich(observation), reward, terminated, truncated, info

    def close(self):
        try:
            self._sensor.close()
        finally:
            self._replay.close()
