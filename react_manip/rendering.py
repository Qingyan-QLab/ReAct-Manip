"""Native RGB rendering with explicit model lifetime and visual-only groups."""
import mujoco


class NativeRGBRenderer:
    def __init__(self, width=640, height=480, include_collision=False):
        self.width, self.height = width, height
        self.include_collision = include_collision
        self._model = None
        self._renderer = None
        self._option = mujoco.MjvOption()
        self._option.geomgroup[0] = 0
        self._option.geomgroup[1] = 1
        self._option.geomgroup[2] = 0
        if include_collision:
            self._option.geomgroup[0] = 1

    def render(self, sim, camera):
        """Render robosuite ``MjSim`` and native MuJoCo task objects."""
        model_wrapper = getattr(sim, "model", sim)
        data_wrapper = getattr(sim, "data", sim)
        model = getattr(model_wrapper, "_model", model_wrapper)
        data = getattr(data_wrapper, "_data", data_wrapper)
        # ``LegacyTaskAdapter`` exposes native tasks through _NativeSimView,
        # whose model/data wrappers are deliberately compatible with
        # robosuite.  Check the explicit marker as well as the direct raw
        # model case; otherwise the visual-only group filter hides native
        # scenes that keep their only geometry in collision group 0.
        native_mujoco = bool(getattr(sim, "native_mujoco", False)) or model_wrapper is model
        # Robosuite tasks use group 1 for their visual geometry and keep
        # collision-only geoms in group 0. Imported native scenes generally
        # put their only geometry in group 0, so hiding it yields a blank UI.
        self._option.geomgroup[0] = int(native_mujoco or self.include_collision)
        if model is not self._model:
            self.close()
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
            self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
            self._model = model
        camera_arg = camera
        if camera == '__free__':
            free = mujoco.MjvCamera()
            free.type = mujoco.mjtCamera.mjCAMERA_FREE
            free.lookat[:] = model.stat.center
            free.distance = max(0.5, 2.2 * float(model.stat.extent))
            free.azimuth, free.elevation = 38, -25
            camera_arg = free
        elif isinstance(camera, str) and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError(f'Unknown camera: {camera}')
        self._renderer.update_scene(data, camera=camera_arg, scene_option=self._option)
        # Native Renderer already returns top-down RGB (no [::-1]).
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None
        self._model = None
