#!/usr/bin/env python3
"""Evaluate the separate RGB-D HarnessVLA track or run a labelled bridge smoke test."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NUMBA_DISABLE_JIT', '0')


def diagnostic_session(env, context, memory):
    """Zero-action infrastructure probe; never used as a HarnessVLA model policy."""
    import numpy as np
    from integrations.harness_vla import OBSERVATION_VERSION
    obs, _ = env.reset()
    if obs.get('observation_schema') != OBSERVATION_VERSION:
        raise ValueError('Missing RGB-D schema')
    for camera, image in obs['images'].items():
        depth, valid = obs['depth_m'][camera], obs['depth_valid'][camera]
        if image.shape != (480, 640, 3) or depth.shape != (480, 640) or valid.shape != depth.shape:
            raise ValueError('Invalid RGB-D sensor shape')
        if not np.isfinite(depth).all() or not valid.any():
            raise ValueError('Camera has no valid metric depth')
    while not env.cancelled.is_set():
        _, _, term, trunc, _ = env.step(np.zeros(env.action_dim))
        if term or trunc:
            break


def annotate_traces(directory, *, provenance_digest):
    """Identify the additional sensor track in each native trace header."""
    from integrations.harness_vla import OBSERVATION_VERSION
    for path in Path(directory).glob('*/trajectory.jsonl'):
        temporary = path.with_suffix('.jsonl.tmp')
        try:
            with path.open() as source, temporary.open('x') as target:
                header = json.loads(source.readline())
                if header.get('type') != 'header':
                    raise ValueError('Invalid trajectory header')
                header.update(policy_track='public_rgbd_extension',
                              observation_schema=OBSERVATION_VERSION,
                              extension_source_digest=provenance_digest)
                target.write(json.dumps(header) + '\n')
                for line in source:
                    target.write(line)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--candidate-bank', type=Path,
                        help='explicit external candidate bank')
    parser.add_argument('--draw-seed', type=int)
    parser.add_argument('--task', help='Select only one task from the bank for diagnostics or partial evaluation')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke-steps', type=int, help='Diagnostic episode cap; disables all aggregate scores')
    parser.add_argument('--trace-dir', type=Path)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('--out must be a new path')
    if args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error('--smoke-steps must be positive')
    if args.manifest and (args.candidate_bank or args.task
                          or args.draw_seed is not None):
        parser.error('--manifest cannot be combined with bank selection options')
    if not args.manifest and args.candidate_bank is None:
        parser.error('provide --manifest or --candidate-bank')
    if not args.manifest and (args.draw_seed is None or args.draw_seed < 0):
        parser.error('--candidate-bank requires nonnegative --draw-seed')
    config = json.loads(args.config.read_text()) if args.config else {}
    if args.smoke_steps is None:
        if not config.get('planner_model') or not config.get('vla_factory'):
            parser.error('Model evaluation requires configured planner_model and vla_factory; no checkpoint is selected by default')
        rpent_root = config.get('rpent_root')
        if rpent_root:
            sys.path.insert(0, str(Path(rpent_root).expanduser().resolve()))

    from integrations.harness_vla import OBSERVATION_VERSION
    from integrations.harness_vla.bridge import HarnessVLAPolicy
    from integrations.harness_vla.rgbd import RGBDReplayEnv
    from react_manip.core.runner import BenchmarkRunner
    from react_manip.instances import suite
    from react_manip.instances.candidate_bank import draw, select_task
    from react_manip.instances.records import digest

    policies = []
    session_output = args.out.parent / (args.out.stem + '_sessions')
    if session_output.exists():
        parser.error('Session output directory already exists')

    def factory():
        if args.smoke_steps is not None:
            policy = HarnessVLAPolicy(diagnostic_session, action_timeout_s=30)
            policy.policy_id = 'rgbd-bridge-diagnostic-zero'
        else:
            from integrations.harness_vla.rpent_session import build
            policy = build(config, session_output)
        policies.append(policy)
        return policy

    # Fail on missing dependencies/model-adapter configuration before simulation
    # startup or selection output. Reuse this first instance's fresh policy.
    first = factory()
    def checked_factory():
        nonlocal first
        if first is not None:
            policy, first = first, None
            return policy
        return factory()

    class Runner(BenchmarkRunner):
        def evaluate(self, *positional, **kwargs):
            if args.smoke_steps is not None:
                kwargs['horizon_sec'] = args.smoke_steps / kwargs['control_hz']
            result = super().evaluate(*positional, **kwargs)
            result.metrics['observation_mode'] = 'rgbd_extension'
            result.metrics['observation_schema'] = OBSERVATION_VERSION
            return result

    cleanup_errors = []
    try:
        manifest = args.manifest
        if manifest is None:
            selection = args.out.parent / (args.out.stem + '_selection')
            if args.task:
                manifest = select_task(args.candidate_bank, task=args.task, seed=args.draw_seed, out=selection)
            else:
                manifest, _ = draw(args.candidate_bank, seed=args.draw_seed, out=selection)
        result = suite.evaluate(manifest, checked_factory, cameras=True, env_factory=RGBDReplayEnv,
                                runner=Runner(), trace_dir=args.trace_dir, visualize=args.visualize)
    finally:
        for policy in policies:
            try:
                getattr(policy, 'shutdown', policy.close)()
            except Exception as exc:
                cleanup_errors.append(f'{type(exc).__name__}: {exc}')
    if cleanup_errors:
        result['valid'] = False
        result['cleanup_errors'] = cleanup_errors
        result['protocol_score'] = None
    extension_files = sorted((ROOT / 'integrations/harness_vla').glob('*.py')) + [Path(__file__).resolve()]
    provenance = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in extension_files}
    result['extension'] = dict(schema_version=OBSERVATION_VERSION, source_digest=digest(provenance),
                               source_files=provenance, config=config,
                               calibration='metric optical-axis depth; x-right y-down z-forward',
                               policy_provenance=[getattr(policy, 'provenance', {}) for policy in policies])
    if args.trace_dir is not None:
        annotate_traces(args.trace_dir, provenance_digest=digest(provenance))
    result['protocol'] = dict(result['protocol'], policy_track='public_rgbd_extension')
    result['mode'] = 'rgbd_bridge_smoke' if args.smoke_steps is not None else 'public_rgbd_extension'
    # Keep this score distinct from the original RGB protocol, even for a full run.
    result['rgbd_protocol_score'] = result.pop('protocol_score') if len(result['groups']) == 16 and args.smoke_steps is None else None
    result['protocol_score'] = None
    result['official_score'] = None
    if args.smoke_steps is not None:
        result['diagnostic_step_cap'] = args.smoke_steps
        for group in result['groups']:
            group['score'] = None
            group['diagnostic_exam_scores'] = group.pop('exam_scores')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(valid=result['valid'], mode=result['mode'], groups=len(result['groups']),
                          episodes=len(result['episodes']), output=str(args.out))))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
