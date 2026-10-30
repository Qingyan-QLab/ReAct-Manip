"""Tilting maze board task.

The MuJoCo environment is imported lazily from ``tasks.tilte_maze.env`` so
task discovery and schema tools work in installations without MuJoCo.
"""


def __getattr__(name):
    if name in {"SPEC", "TilteMaze", "TilteMazePhysical"}:
        from .env import SPEC, TilteMaze, TilteMazePhysical
        return {"SPEC": SPEC, "TilteMaze": TilteMaze, "TilteMazePhysical": TilteMazePhysical}[name]
    raise AttributeError(name)

__all__ = ["SPEC", "TilteMaze", "TilteMazePhysical"]
