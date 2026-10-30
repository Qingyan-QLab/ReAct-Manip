"""Canonical environment construction, independent of reference policies."""
from __future__ import annotations
import os
from typing import Any
from .env import LaptopIntoDrawer
from react_manip.robots.resolution import resolve_robosuite_robot

def make_env(
    seed: int = 0,
    robot_id: str = "panda",
    deterministic_initial_laptop: bool = False,
    render_mode: str | None = None,
    has_renderer: bool = False,
    has_offscreen_renderer: bool = True,
    **kwargs: Any,
) -> LaptopIntoDrawer:
    """Construct the canonical physical environment used by all frontends."""

    from robosuite.controllers import load_composite_controller_config

    resolved = resolve_robosuite_robot(robot_id)
    robot_name = resolved.robot_name
    if robot_id == "panda":
        config = load_composite_controller_config(
            controller=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "assets",
                "default_panda_ik.json",
            )
        )
    elif robot_id == "sawyer":
        # Sawyer uses the repository's floor-contacting pedestal and its
        # native right-arm controller.  Keep the same physical task scene;
        # only the embodiment and controller asset change.
        config = load_composite_controller_config(controller="BASIC", robot="Sawyer")
        kwargs.setdefault("gripper_types", "default")
        kwargs.setdefault("initialization_noise", None)
    else:
        config = resolved.controller_configs
        kwargs.setdefault("gripper_types", resolved.gripper_types)
        kwargs.setdefault("initialization_noise", resolved.initialization_noise)
    if render_mode not in (None, "human", "rgb_array"):
        raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
    if render_mode == "human":
        has_renderer, has_offscreen_renderer = True, False
    elif render_mode == "rgb_array":
        has_renderer, has_offscreen_renderer = False, True
    scenario = kwargs.pop("scenario", None)
    if scenario is None:
        import json
        from pathlib import Path
        from react_manip.core.scenarios import sample_scenario
        profile = Path(__file__).resolve().parents[2] / 'configs/laptop_into_drawer/profile_baseline.json'
        scenario = sample_scenario(json.loads(profile.read_text()), seed=seed)
    if isinstance(scenario, (str, os.PathLike)):
        from react_manip.core.scenarios import Scenario
        scenario = Scenario.load(scenario)
    params = dict(
        robots=robot_name,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        use_camera_obs=False,
        use_object_obs=False,
        render_camera="frontview",
        horizon=2400,
        controller_configs=config,
        seed=seed,
        lite_physics=False,
        # Use robosuite's standard Panda reset distribution. Passing None
        # bypasses its nominal arm posture on some robosuite versions and can
        # leave IK_POSE with a stale / extreme target after reset.
        initialization_noise="default",
        deterministic_initial_laptop=deterministic_initial_laptop,
        scenario=scenario,
    )
    params.update(kwargs)
    env = LaptopIntoDrawer(**params)
    env.scenario = scenario
    env.robot_id = robot_id
    env.action_layout = dict(env.robots[0].composite_controller._action_split_indexes)
    if scenario is not None:
        env.scenario_digest = scenario.digest
    # Keep the Panda joint posture identical across reset() calls. Object
    # placement remains seed-controlled, while keyboard reset reproduces the
    # same EEF pose seen at demo startup.
    return env
