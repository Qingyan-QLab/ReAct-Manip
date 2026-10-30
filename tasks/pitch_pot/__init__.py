"""Physical multi-embodiment single-ball pitch-pot environment."""
from .env import PitchPot
from . import geometry  # register the task-owned geometry adapter

__all__ = ["PitchPot"]
