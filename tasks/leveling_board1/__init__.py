"""Physical leveling task; instance validation does not import the simulator."""

__all__ = ["LevelingBoard1", "make_env"]


def __getattr__(name):
    if name in __all__:
        from .env import LevelingBoard1, make_env
        return {'LevelingBoard1':LevelingBoard1,'make_env':make_env}[name]
    raise AttributeError(name)
