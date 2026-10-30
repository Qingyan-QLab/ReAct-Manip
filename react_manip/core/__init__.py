"""Public API for the extensible, fixed-horizon react_manip.

The modules in :mod:`react_manip.core` deliberately do not import MuJoCo or
robosuite.  This keeps the protocol, metrics, and registry usable by tooling
and CI on machines where the simulator is not installed.
"""

from .metrics import SustainedSuccessTracker
from .aggregate import aggregate_results
from .compatibility import validate_task_robot
from .protocol import BenchmarkTask, Policy, StepResult, normalize_reset, normalize_step
from .registry import (
    RobotRegistry,
    TaskDefinition,
    TaskRegistry,
    get_robot_registry,
    get_task_registry,
    register_robot,
    register_task,
)
from .runner import BenchmarkRunner, EvaluationResult
from .specs import SuccessRule, TaskSpec
from .scenarios import Scenario, sample_scenario
from .contracts import ObservationSpec, ActionSpec, FeedbackSpec
from .geometry import GeometryFeatureAdapter
from .observation import policy_observation
from .language import EpisodeLanguagePrompt, build_episode_prompt
from .visual_randomization import ensure_workspace_camera

__all__ = [
    "BenchmarkRunner",
    "BenchmarkTask",
    "EvaluationResult",
    "Policy",
    "RobotRegistry",
    "StepResult",
    "SuccessRule",
    "SustainedSuccessTracker",
    "TaskDefinition",
    "TaskRegistry",
    "TaskSpec",
    "get_robot_registry",
    "get_task_registry",
    "register_robot",
    "register_task",
    "normalize_reset",
    "normalize_step",
    "validate_task_robot",
    "Scenario",
    "sample_scenario",
    "ObservationSpec",
    "ActionSpec",
    "FeedbackSpec",
    "GeometryFeatureAdapter",
    "policy_observation",
    "aggregate_results",
    "EpisodeLanguagePrompt",
    "build_episode_prompt",
    "ensure_workspace_camera",
]
