"""Opt-in RPent embodiments. Importing does not change existing task defaults."""

from .catalog import EMBODIMENTS


def register():
    """Register lazy model providers without importing simulator dependencies."""
    from react_manip.robots.resolution import register_robosuite_robot, registered_robot_ids
    from react_manip.core.registry import get_robot_registry
    from .models import resolve, EmbodimentAdapter

    for robot_id in EMBODIMENTS:
        if robot_id not in registered_robot_ids():
            register_robosuite_robot(robot_id, lambda *, include_controller=True, key=robot_id:
                                    resolve(key, include_controller=include_controller))
        if robot_id not in {name for name,_ in get_robot_registry().list()}:
            get_robot_registry().register(robot_id, EmbodimentAdapter(robot_id))
