#!/usr/bin/env python3
"""Evaluate an external policy on a task or an explicit external suite.

    python scripts/evaluate_policy.py --task conveyor_grasp --state-only-smoke --horizon-sec 1
    python scripts/evaluate_policy.py --manifest /path/to/manifest.json --policy my_policy:build --out outputs/result.json

Candidate banks and manifests are not bundled with the source distribution.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any

# Running ``python scripts/evaluate_policy.py`` places ``scripts/`` (rather
# than the repository root) on sys.path.  Make the CLI work both that way and
# via ``python -m`` without requiring PYTHONPATH setup.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from react_manip.builtin import register_builtin_tasks
from react_manip.core import BenchmarkRunner, get_task_registry
from react_manip.policies import CallablePolicy, RandomPolicy
from react_manip.robot_registry import register_default_robots


def _load_object(path: str) -> Any:
    if ":" not in path:
        raise ValueError(f"expected module:object, got {path!r}")
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--candidate-bank", type=Path,
                        help="explicit external candidate bank")
    parser.add_argument("--draw-seed", type=int, help="required candidate-bank selection seed, independent of policy seed")
    parser.add_argument("--selection-out", type=Path, help="new draw directory; defaults beside result")
    parser.add_argument("--manifest", type=Path, help="explicit Task/Episode Instance suite manifest")
    parser.add_argument("--state-only-smoke", action="store_true", help="disable policy images for smoke checks; not an RGB evaluation")
    parser.add_argument("--trace-dir", type=Path, help="manifest only: new public trajectory directory")
    parser.add_argument("--visualize", action="store_true", help="display policy main/wrist RGB; Space pauses, Tab toggles debug view, Esc aborts")
    parser.add_argument("--video-dir", type=Path, help="new directory for per-episode composite and per-camera MP4 videos")
    parser.add_argument("--task", help="registered task id")
    parser.add_argument("--scenario", type=Path, help="resolved Scenario JSON to load")
    parser.add_argument("--policy", help="policy factory/object module:object")
    parser.add_argument("--robot", default=None, help="robot embodiment override; defaults to the task's sole supported robot")
    parser.add_argument("--episodes", type=int, help="independent protocol only (default: task config or 3)")
    parser.add_argument("--groups", type=int, default=1, help="number of independent hidden-parameter groups")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon-sec", type=float)
    parser.add_argument("--control-hz", type=float)
    parser.add_argument("--episode-wall-timeout-sec", type=float,
                        help="real-time limit per episode (default: 1800 seconds)")
    parser.add_argument("--task-wall-timeout-sec", type=float,
                        help="real-time limit for all episodes of one task (defaults: 2h conveyor_grasp, 4h find_object_in_drawer2, 1h otherwise)")
    parser.add_argument(
        "--render",
        action="store_true",
        help="open and refresh the native simulator viewer (interactive mode)",
    )
    parser.add_argument(
        "--render-realtime",
        action="store_true",
        help="pace rendered steps at control_hz instead of running as fast as possible",
    )
    parser.add_argument(
        "--render-hold-sec",
        type=float,
        default=3.0,
        help="seconds to keep the viewer open after each rendered episode (default: 3)",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark_v2_results.json"))
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    for name in ('horizon_sec', 'control_hz', 'render_hold_sec',
                 'episode_wall_timeout_sec', 'task_wall_timeout_sec'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or
                                  (value < 0 if name == 'render_hold_sec' else value <= 0)):
            parser.error('--' + name.replace('_', '-') + ' must be finite and ' +
                         ('nonnegative' if name == 'render_hold_sec' else 'positive'))
    if args.candidate_bank is None and args.manifest is None and args.task is None:
        parser.error('provide --task, --manifest or --candidate-bank')
    if args.robot is not None:
        from react_manip.robots.catalog import normalize_robot_id
        args.robot = normalize_robot_id(args.robot)
    supplied = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
    selection = None
    if args.visualize or args.video_dir is not None:
        if not (args.candidate_bank or args.manifest or args.task) or args.state_only_smoke:
            parser.error('--visualize/--video-dir require an RGB task, candidate-bank or manifest evaluation')
        if args.video_dir is not None and args.video_dir.exists():
            parser.error('video directory already exists; choose a new --video-dir path')
        from react_manip.evaluation.viewer import check_dependencies
        try:
            check_dependencies(args.visualize, args.video_dir is not None)
        except ImportError as exc:
            parser.error(str(exc))
    if args.candidate_bank:
        if args.manifest or args.draw_seed is None or args.draw_seed < 0:
            parser.error("--candidate-bank requires nonnegative --draw-seed and excludes --manifest")
        # Apply manifest-mode conflicts before creating any selection files.
        forbidden_bank = {"--task", "--scenario", "--robot",
                          "--episodes", "--groups", "--seed", "--horizon-sec", "--control-hz",
                          "--render", "--render-realtime", "--render-hold-sec"}
        if supplied & forbidden_bank:
            parser.error("candidate bank owns environment/protocol settings: " + ", ".join(sorted(supplied & forbidden_bank)))
        if args.out.exists() or (args.trace_dir is not None and args.trace_dir.exists()):
            parser.error("result/trace output already exists")
        from react_manip.instances.candidate_bank import draw
        try:
            args.manifest, selection = draw(args.candidate_bank, seed=args.draw_seed,
                out=args.selection_out or args.out.with_name(args.out.stem + "_selection"))
        except (ValueError, TypeError, OSError) as exc:
            parser.error(str(exc))
    elif args.draw_seed is not None or args.selection_out is not None:
        parser.error("--draw-seed and --selection-out require --candidate-bank")
    if args.manifest:
        # Candidate storage must not be evaluated directly as a selected suite.
        bank_path = args.manifest.parent / "bank.json"
        if bank_path.exists():
            from react_manip.instances.candidate_bank import load_bank
            _, candidate_path, _, _ = load_bank(bank_path)
            if args.manifest.resolve() == candidate_path:
                parser.error("candidate storage is not an evaluation manifest; use --candidate-bank or draw first")
        forbidden = {"--task", "--scenario", "--robot",
                     "--episodes", "--groups", "--seed", "--horizon-sec", "--control-hz",
                     "--render", "--render-realtime", "--render-hold-sec"}
        conflicts = sorted(supplied & forbidden)
        if conflicts:
            parser.error("--manifest owns environment and protocol settings; incompatible: " + ", ".join(conflicts))
        if args.out.exists():
            parser.error("manifest result file already exists; choose a new --out path")
        if args.trace_dir is not None and args.trace_dir.exists():
            parser.error("trace directory already exists; choose a new --trace-dir path")
    elif args.trace_dir is not None:
        parser.error("--trace-dir requires --manifest")
    if args.groups < 1 or (args.episodes is not None and args.episodes < 1) or args.seed < 0:
        parser.error("groups/episodes must be positive and seed must be nonnegative")

    def policy_factory() -> Any:
        if not args.policy:
            return RandomPolicy(seed=args.seed)
        obj = _load_object(args.policy)
        if isinstance(obj, type):
            value = obj()
        elif callable(obj) and not hasattr(obj, "act"):
            try:
                inspect.signature(obj).bind()
            except TypeError:
                value = obj
            else:
                value = obj()
        else:
            value = deepcopy(obj)
        if callable(value) and not hasattr(value, "act"):
            value = CallablePolicy(value)
        return value

    def report(row):
        if "error" in row:
            print(f"[benchmark error] {row['error']}", file=sys.stderr)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

    if args.manifest:
        from react_manip.instances.suite import evaluate as evaluate_manifest
        from react_manip.instances.candidate_bank import selection_for
        try:
            selection = selection_for(args.manifest)
        except (ValueError, TypeError, OSError) as exc:
            parser.error(str(exc))
        try:
            timeout_options = ({'task_wall_timeout_sec': args.task_wall_timeout_sec}
                               if args.task_wall_timeout_sec is not None else {})
            evaluation = evaluate_manifest(args.manifest, policy_factory,
                                           cameras=not args.state_only_smoke,
                                           trace_dir=args.trace_dir, on_result=report,
                                           visualize=args.visualize, video_dir=args.video_dir,
                                           episode_wall_timeout_sec=args.episode_wall_timeout_sec,
                                           **timeout_options)
        except (ValueError, TypeError, OSError) as exc:
            parser.error(str(exc))
        rows = evaluation.get("episodes", [])
        for group in evaluation["groups"]:
            if group.get("error"):
                print(f"[benchmark error] {group['task_instance_id']}: {group['error']}", file=sys.stderr)
        payload = {
            "benchmark_api_version": "2.0", "metrics_schema_version": "2.0",
            "manifest_path": str(args.manifest.resolve()),
            "policy_factory": args.policy,
            "policy_id": rows[0]["policy_id"] if rows else None,
            "summary": {"groups": len(evaluation["groups"]), "completed_episodes": len(rows),
                        "invalid_groups": sum(not g["valid"] for g in evaluation["groups"]),
                        "score": evaluation["protocol_score"]},
            **evaluation,
            **({"candidate_selection": selection} if selection is not None else {}),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"results -> {args.out}")
        return 0 if evaluation["valid"] else 1

    register_default_robots()
    register_builtin_tasks()

    if not args.task:
        parser.error('provide --candidate-bank, --manifest or --task')
    try:
        definition = get_task_registry().get(args.task)
    except KeyError:
        parser.error(f'unknown release task: {args.task}')
    spec, factory = definition.spec, definition.factory
    if args.control_hz is not None and args.control_hz != spec.control_hz:
        parser.error(f'--control-hz must match the task physics rate ({spec.control_hz:g} Hz); '
                     'changing evaluator tick counts does not change robot control timing')
    from react_manip.core.compatibility import resolve_task_robot
    try:
        args.robot = resolve_task_robot(spec, args.robot)
    except ValueError as exc:
        parser.error(str(exc))

    runner = BenchmarkRunner()
    scenario = None
    if args.scenario:
        from react_manip.core.scenarios import Scenario
        scenario = Scenario.load(args.scenario)
        if scenario.task_id != spec.task_id:
            parser.error(f"scenario task {scenario.task_id!r} does not match {spec.task_id!r}")
    from react_manip.evaluation import EvaluationContext, load_protocol

    try:
        protocol = load_protocol(spec)
        from react_manip.evaluation.viewer import EvaluationViewer
        task_timeout_options = ({'task_wall_timeout_sec': args.task_wall_timeout_sec}
                                if args.task_wall_timeout_sec is not None else {})
        with EvaluationViewer(enabled=args.visualize, video_dir=args.video_dir) as viewer:
            evaluation = protocol.run(EvaluationContext(
                spec=spec, factory=factory, policy_factory=policy_factory,
                viewer=viewer if viewer.active else None,
                robot_id=args.robot, seed=args.seed, scenario=scenario,
                groups=args.groups, episodes=args.episodes, runner=runner,
                runner_options=dict(camera_observations=not args.state_only_smoke,
                                    horizon_sec=args.horizon_sec, control_hz=args.control_hz,
                                    episode_wall_timeout_sec=args.episode_wall_timeout_sec,
                                    render=args.render, render_realtime=args.render_realtime,
                                    render_hold_sec=args.render_hold_sec,
                                    **task_timeout_options), on_result=report))
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    results = evaluation.get("episodes", [])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_api_version": "2.0",
        "metrics_schema_version": "2.0",
        "task_id": spec.task_id,
        "task_version": spec.task_version,
        "robot_id": args.robot,
        "observation_mode": "state_only_smoke" if args.state_only_smoke else "rgb",
        "scenario_id": scenario.scenario_id if scenario else None,
        "scenario_digest": scenario.digest if scenario else None,
        "policy_id": results[0]["policy_id"] if results else None,
        **evaluation,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results -> {args.out}")
    return 0 if evaluation.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
