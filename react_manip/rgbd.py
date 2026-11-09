"""Shared RGB-D observation contract for policy-facing cameras."""

from __future__ import annotations

import numpy as np


OBSERVATION_VERSION = "react-manip.rgbd.v1"


def back_project(depth_m, calibration, pixels):
    """Project (row, column) pixels to world metres using sensor calibration."""
    depth = np.asarray(depth_m)
    pixels = np.asarray(pixels)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError("pixels must be an integer [N, 2] array in row/column order")
    rows, cols = pixels.T
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= depth.shape[0]) or np.any(cols >= depth.shape[1]):
        raise ValueError("Pixel is outside the depth image")
    z = depth[rows, cols]
    if not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError("Pixel has no valid metric depth")
    intrinsics = np.asarray(calibration["intrinsics"])
    points = np.column_stack(
        (
            (cols - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    world_from_camera = np.asarray(calibration["world_from_camera"])
    return points @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]


class RGBDSensor:
    """Capture aligned RGB, metric depth, validity and camera calibration."""

    def __init__(self, width=640, height=480):
        from .rendering import NativeRGBRenderer

        self.width, self.height = int(width), int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("RGB-D resolution must be positive")
        self._rgb = NativeRGBRenderer(self.width, self.height)

    def capture(self, sim, camera):
        import mujoco

        rgb = self._rgb.render(sim, camera).copy()
        renderer = self._rgb._renderer
        model = self._rgb._model
        data = getattr(sim.data, "_data", sim.data)
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if index < 0:
            raise ValueError("RGB-D requires a named sensor camera")
        if model.cam_orthographic[index] or np.any(model.cam_sensorsize[index]):
            raise ValueError("rgbd.v1 supports perspective fovy cameras only")
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
        intrinsics = np.array(
            [
                [focal, 0, (self.width - 1) / 2],
                [0, focal, (self.height - 1) / 2],
                [0, 0, 1],
            ]
        )
        world_from_camera = np.eye(4)
        world_from_camera[:3, :3] = data.cam_xmat[index].reshape(3, 3) @ np.diag([1, -1, -1])
        world_from_camera[:3, 3] = data.cam_xpos[index]
        calibration = {
            "intrinsics": intrinsics,
            "world_from_camera": world_from_camera,
            "depth_units": "metres",
            "depth_convention": "optical_axis_z",
            "camera_axes": "x_right_y_down_z_forward",
            "pixel_order": "row_column",
            "image_origin": "top_left",
            "near_m": near,
            "far_m": far,
            "simulation_time_s": float(data.time),
        }
        return rgb, depth, valid, calibration

    def close(self):
        self._rgb.close()
