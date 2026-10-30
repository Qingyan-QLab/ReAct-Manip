"""Preserve arena visual settings when robosuite merges a task model."""
from copy import deepcopy
from robosuite.models.tasks import ManipulationTask as RobosuiteTask
from robosuite.utils import transform_utils as _transform_utils


from .robots.sensors import ensure_robot_overhead_camera

# robosuite 1.5.2's Numba mat2quat can fail when a long-lived process changes
# tasks and the dispatcher sees a second array memory layout. Its Python
# implementation is deterministic and the overhead is negligible relative to
# a MuJoCo control step. Keep the workaround local to this one conversion.
if hasattr(_transform_utils.mat2quat, "py_func"):
    _transform_utils.mat2quat = _transform_utils.mat2quat.py_func


class ManipulationTask(RobosuiteTask):
    def __init__(self, *args, **kwargs):
        # Add embodiment cameras before robosuite merges the robot and arena
        # XML.  This covers Panda, Sawyer, PiperX, ALOHA and Mobile ALOHA
        # without requiring task-specific camera code.
        robots = kwargs.get("mujoco_robots", ())
        for robot in robots or ():
            ensure_robot_overhead_camera(robot)
            from react_manip.cameras import ensure_robot_cameras
            ensure_robot_cameras(robot)
        from react_manip.cameras import ensure_scene_cameras
        if kwargs.get('mujoco_arena') is not None:
            arena = kwargs['mujoco_arena']
            ensure_scene_cameras(arena)
            # Some imported arenas (e.g. pen_strike) have only group-0
            # geometry, with no separate visual copies. Keep them visible
            # in the same group-1 render path without changing collisions.
            geoms = list(arena.worldbody.iter('geom'))
            if not any(g.get('group', '0') == '1' for g in geoms):
                for g in geoms:
                    if g.get('group', '0') == '0':
                        g.set('group', '1')
        super().__init__(*args, **kwargs)
        visual = self.mujoco_arena.root.find("visual")
        if visual is not None:
            target = self.root.find("visual")
            if target is None:
                self.root.append(deepcopy(visual))
            else:
                for child in visual:
                    existing = target.find(child.tag)
                    if existing is not None:
                        target.remove(existing)
                    target.append(deepcopy(child))
