"""Canonical robot-id to robosuite construction resolution.

Keeping this mapping in one place prevents task factories from silently
falling back to Panda and ensures every physical task uses the same arm order,
controller and gripper configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any



@dataclass(frozen=True)
class RobosuiteResolution:
    robot_id: str
    robot_name: str
    controller_configs: Any = None
    gripper_types: str = "default"
    initialization_noise: Any = None
    arms: tuple[str, ...] = ("right",)
    mobile_base: bool = False


def _builtin_resolution(key: str, *, include_controller: bool = True) -> RobosuiteResolution:
    if key in ('panda_libero','aloha_agilex_robotwin','dual_arx_x5_robodojo','panda_omron_robocasa'):
        from integrations.rpent_embodiments.models import resolve
        return resolve(key, include_controller=include_controller)
    if key == "panda":
        return RobosuiteResolution(key, "Panda", gripper_types="PandaGripper", initialization_noise="default", arms=("right",))
    if key == "sawyer":
        from .sawyer_robosuite import register_sawyer_with_base
        register_sawyer_with_base()
        return RobosuiteResolution(key, "CLSawyer", arms=("right",))
    from .agilex_robosuite import controller_config, resolve_robot
    arms = ("right",) if key in ("piperx", "legacy_piperx") else ("right", "left")
    return RobosuiteResolution(
        key, resolve_robot(key),
        controller_configs=controller_config(arms) if include_controller else None,
        initialization_noise=None, arms=arms, mobile_base=key in ("mobile_dual_piperx", "aloha_mobile"),
    )


_PROVIDERS = {key: (lambda *, include_controller=True, key=key:
                   _builtin_resolution(key, include_controller=include_controller))
              for key in ('panda', 'sawyer', 'piperx', 'dual_piperx', 'mobile_dual_piperx', 'legacy_piperx', 'aloha', 'aloha_mobile', 'panda_libero', 'aloha_agilex_robotwin', 'dual_arx_x5_robodojo', 'panda_omron_robocasa')}
_MODEL_IDS = {'liberopanda':'panda_libero', 'robotwinaloha':'aloha_agilex_robotwin', 'robodojox5':'dual_arx_x5_robodojo', 'robocasapandaomron':'panda_omron_robocasa', 'panda': 'panda', 'sawyer': 'sawyer', 'clsawyer': 'sawyer',
              'agilexpiper': 'legacy_piperx', 'agilexaloha': 'aloha', 'agilexmobilealoha': 'aloha_mobile',
              'piperx': 'piperx', 'dualpiperx': 'dual_piperx', 'mobiledualpiperx': 'mobile_dual_piperx'}


def robot_id_from_model(name):
    """Resolve a canonical ID or a registered simulator-model alias."""
    key = str(name).strip().lower()
    if key in _PROVIDERS:
        return key
    if key in _MODEL_IDS:
        return _MODEL_IDS[key]
    raise ValueError(f'unknown robot/model {name!r}; register its provider first')


def default_controller_config(robot):
    """Resolve embodiment controller defaults for direct environment creation."""
    resolved = resolve_robosuite_robot(robot_id_from_model(robot))
    if resolved.controller_configs is not None:
        return resolved.controller_configs
    from robosuite.controllers import load_composite_controller_config
    return load_composite_controller_config(robot='Sawyer' if resolved.robot_id == 'sawyer' else resolved.robot_name)


def registered_robot_ids():
    """Live catalog; registration does not import simulation dependencies."""
    return tuple(_PROVIDERS)


def register_robosuite_robot(robot_id, provider):
    """Register a lazy provider accepting ``include_controller=bool``.

    The provider registers its model with robosuite and returns a
    RobosuiteResolution. This does not assert physical task compatibility.
    """
    if not isinstance(robot_id, str) or not robot_id or robot_id != robot_id.strip().lower() or any(c.isspace() for c in robot_id):
        raise ValueError('robot_id must be a nonempty lowercase identifier')
    if robot_id in _PROVIDERS:
        raise ValueError(f'robot already registered: {robot_id}')
    if robot_id in _MODEL_IDS and _MODEL_IDS[robot_id] != robot_id:
        raise ValueError(f'robot id conflicts with model alias: {robot_id}')
    if not callable(provider):
        raise TypeError('provider must be callable')
    _PROVIDERS[robot_id] = provider


def resolve_robosuite_robot(robot_id: str, *, include_controller: bool = True) -> RobosuiteResolution:
    from .catalog import normalize_robot_id
    key = normalize_robot_id(robot_id)
    result = _PROVIDERS[key](include_controller=include_controller)
    if not isinstance(result, RobosuiteResolution) or result.robot_id != key:
        raise ValueError(f'invalid robot provider result for {key}')
    alias = result.robot_name.lower()
    if (alias in _MODEL_IDS and _MODEL_IDS[alias] != key) or (alias in _PROVIDERS and alias != key):
        raise ValueError(f'robot model alias already owned: {result.robot_name}')
    _MODEL_IDS[alias] = key
    return result
