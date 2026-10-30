"""Episode-level natural-language task prompt generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .specs import TaskSpec

def normalize_language_prompt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Add explicit task / current-round scopes without breaking legacy consumers.

    A round is a subgoal within an episode, not another environment reset.
    Tasks without subgoals leave round_instruction empty and round_index None.
    Legacy instruction/text retain their existing combined representation.
    """
    result = dict(value)
    result.setdefault("task_instruction", str(result.get("instruction", "")))
    result.setdefault("round_instruction", "")
    result.setdefault("round_index", None)
    combined = " ".join(str(result[key]) for key in ("task_instruction", "round_instruction") if result[key])
    result.setdefault("instruction", combined)
    result.setdefault("text", combined)
    return result


@dataclass(frozen=True)
class EpisodeLanguagePrompt:
    task_id: str
    instruction: str
    robot_id: str | None
    episode_index: int
    scenario_id: str | None
    success_condition: str
    constraints: tuple[str, ...] = ()
    public_context: Mapping[str, Any] | None = None
    task_instruction: str = ""
    round_instruction: str = ""
    round_index: int | None = None

    @property
    def text(self) -> str:
        robot = self.robot_id or "unspecified embodiment"
        context = ""
        if self.public_context:
            context = " Public scene context: " + ", ".join(
                f"{k}={v}" for k, v in self.public_context.items()
            ) + "."
        limits = "" if not self.constraints else " Constraints: " + "; ".join(self.constraints) + "."
        current = f" Current round: {self.round_instruction}" if self.round_instruction else ""
        return f"Task: {self.task_instruction or self.instruction}{current} Embodiment: {robot}. Success: {self.success_condition}.{context}{limits}"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["constraints"] = list(self.constraints)
        value["public_context"] = dict(self.public_context or {})
        value["text"] = self.text
        value["task_instruction"] = self.task_instruction or self.instruction
        return normalize_language_prompt(value)


def _robot_id_from_env(env: Any) -> str | None:
    if getattr(env, "robot_id", None):
        return str(env.robot_id)
    robots = getattr(env, "robots", ()) or ()
    if not robots:
        return None
    name = str(getattr(robots[0], "name", "")).lower()
    from ..robots.resolution import robot_id_from_model
    try:return robot_id_from_model(name)
    except ValueError:pass
    for key, marker in (("aloha_mobile", "agilexmobilealoha"), ("aloha", "agilexaloha"), ("legacy_piperx", "agilexpiper"), ("mobile_dual_piperx", "mobiledualpiperx"), ("dual_piperx", "dualpiperx"),
                        ("piperx", "piperx"), ("sawyer", "sawyer"), ("panda", "panda")):
        if marker in name:
            return key
    return None


def task_language_prompt(env: Any, *, task_id: str, task_instruction: str,
                         success_condition: str, constraints: tuple[str, ...] = ()) -> dict[str, Any]:
    """Format rules owned by a task's runtime environment, never by JSON."""
    return EpisodeLanguagePrompt(
        task_id=task_id, instruction=task_instruction, task_instruction=task_instruction,
        robot_id=_robot_id_from_env(env),
        episode_index=max(0, int(getattr(env, "_episode_index", 0))),
        scenario_id=getattr(getattr(env, "scenario", None), "scenario_id", None),
        success_condition=success_condition, constraints=constraints,
    ).to_dict()


def runtime_language_prompt(env: Any) -> dict[str, Any]:
    """Require the task-owned runtime interface; missing rules are an error."""
    method = getattr(env, "get_episode_language_prompt", None)
    if not callable(method):
        raise RuntimeError(f"{type(env).__name__} must implement get_episode_language_prompt()")
    value = method()
    if not isinstance(value, Mapping):
        raise TypeError("get_episode_language_prompt() must return a mapping")
    prompt = normalize_language_prompt(value)
    if not str(prompt["task_instruction"]).strip() or not str(prompt["text"]).strip():
        raise ValueError("Runtime language prompt must contain non-empty task_instruction and text")
    return prompt


def build_episode_prompt(spec: TaskSpec, env: Any = None, *, robot_id: str | None = None,
                         scenario: Any = None, episode_index: int = 0) -> EpisodeLanguagePrompt:
    """Compatibility formatter using runtime rules only; JSON is never a fallback."""
    prompt = runtime_language_prompt(env)
    return EpisodeLanguagePrompt(
        task_id=prompt.get("task_id", spec.task_id), instruction=prompt["instruction"],
        task_instruction=prompt["task_instruction"], round_instruction=prompt["round_instruction"],
        round_index=prompt["round_index"], robot_id=robot_id or prompt.get("robot_id"),
        episode_index=episode_index,
        scenario_id=getattr(scenario, "scenario_id", None) or prompt.get("scenario_id"),
        success_condition=prompt.get("success_condition", ""),
        constraints=tuple(prompt.get("constraints", ())), public_context=prompt.get("public_context", {}),
    )
