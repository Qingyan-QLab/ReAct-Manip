"""Narrow helpers for integrations that are not part of the public wheel."""

from __future__ import annotations

import importlib
from types import ModuleType


def import_optional_integration(module: str) -> ModuleType | None:
    """Import an integration module, allowing only a missing root package.

    The public wheel intentionally omits ``integrations``.  If that package is
    present, import failures from the integration or one of its dependencies
    remain visible instead of being mistaken for an optional-feature absence.
    """

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "integrations":
            return None
        raise


def preserve_native_action_units(config: dict, robot_id: str) -> None:
    """Apply the RPent Panda override only when that embodiment needs it.

    Public PiPER-X tasks use their task-local action scaling and do not need
    the separately distributed RPent integration.  The Panda-LIBERO path,
    however, must still fail clearly when that integration is unavailable.
    """

    if robot_id != "panda_libero":
        return

    models = import_optional_integration("integrations.rpent_embodiments.models")
    if models is None:
        error = ModuleNotFoundError(
            "The panda_libero embodiment requires integrations.rpent_embodiments"
        )
        error.name = "integrations"
        raise error
    models.preserve_native_action_units(config, robot_id)
