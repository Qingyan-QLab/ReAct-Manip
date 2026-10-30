"""Policy-observation filtering for privileged task state.

Geometry truth is exposed through :class:`GeometryFeatureAdapter`, not through
the observation passed to a normal policy.  Legacy robosuite tasks can still
produce named object sensors, so the benchmark runner enforces the task's
visibility contract at its final policy boundary.
"""

from __future__ import annotations

from typing import Any, Mapping


def policy_observation(observation: Any, task_spec: Any, env: Any = None) -> Any:
    """Return the non-privileged view that may be passed to a policy.

    Tasks list any legacy raw keys in ``privileged_observation_keys``.  The
    robosuite aggregate ``object-state`` is always privileged when geometry is
    declared internal.  Non-mapping observations are returned unchanged so
    native state-vector tasks keep their existing API.
    """

    if not isinstance(observation, Mapping):
        return observation
    contract = getattr(task_spec, "observation", None)
    if contract is None or getattr(contract, "geometry_view", "internal") != "internal":
        return observation
    if "geometry" in tuple(getattr(contract, "default_policy_view", ())):
        return observation

    task_env = getattr(env, "env", env)
    hidden = set(getattr(task_env, "privileged_observation_keys", ()))
    hidden.add("object-state")
    if not hidden.intersection(observation):
        return observation

    filtered = observation.copy()
    for key in hidden:
        filtered.pop(key, None)
    return filtered


def policy_info(info: Mapping[str, Any], policy: Any, env: Any) -> Mapping[str, Any]:
    """Attach geometry truth only for an explicitly privileged oracle.

    A regular policy sees the original public ``info`` mapping.  Oracle code
    must opt in with ``requires_geometry_truth = True``; this makes privileged
    use visible in code review and prevents accidental model leakage.
    """

    if not bool(getattr(policy, "requires_geometry_truth", False)):
        raw = getattr(env, 'env', env)
        hidden = set(getattr(raw, 'privileged_info_keys', ()))
        return {k:v for k,v in info.items() if k not in hidden} if hidden else info
    from .geometry import GeometryFeatureAdapter

    enriched = dict(info)
    enriched["geometry_ground_truth"] = GeometryFeatureAdapter.for_task(env).extract()
    return enriched
