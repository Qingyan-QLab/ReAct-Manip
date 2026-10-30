import importlib

import pytest

from react_manip.optional_integrations import (
    import_optional_integration,
    preserve_native_action_units,
)


def _missing(name):
    error = ModuleNotFoundError(f"No module named {name!r}")
    error.name = name
    return error


def test_missing_integrations_root_is_optional(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda _: (_ for _ in ()).throw(_missing("integrations")))
    assert import_optional_integration("integrations.rpent_embodiments.catalog") is None


def test_integration_internal_import_errors_remain_visible(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda _: (_ for _ in ()).throw(_missing("integration_dependency")))
    with pytest.raises(ModuleNotFoundError, match="integration_dependency"):
        import_optional_integration("integrations.rpent_embodiments.catalog")


def test_piperx_action_units_do_not_import_integrations(monkeypatch):
    def unexpected_import(_):
        raise AssertionError("PiPER-X must not import the RPent integration")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    config = {"body_parts": {}}
    preserve_native_action_units(config, "single_piperx")
    assert config == {"body_parts": {}}


def test_panda_libero_action_units_require_integration(monkeypatch):
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _: (_ for _ in ()).throw(_missing("integrations")),
    )
    with pytest.raises(ModuleNotFoundError, match="panda_libero embodiment requires"):
        preserve_native_action_units({"body_parts": {}}, "panda_libero")


def test_panda_libero_internal_import_errors_remain_visible(monkeypatch):
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _: (_ for _ in ()).throw(_missing("integration_dependency")),
    )
    with pytest.raises(ModuleNotFoundError, match="integration_dependency"):
        preserve_native_action_units({"body_parts": {}}, "panda_libero")
