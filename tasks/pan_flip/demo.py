#!/usr/bin/env python3
"""Run the calibrated PiperX pan-flip controller."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', default='piperx',
                        choices=('panda', 'sawyer', 'piperx', 'aloha', 'aloha_mobile'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scenario', type=Path)
    parser.add_argument('--profile', type=Path,
                        default=ROOT / 'configs/pan_flip/profile_reference.json')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'outputs/pan_flip/piperx/demo.json')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args(argv)
    from tasks.pan_flip.runtime import make_env
    from tasks.pan_flip.operation import PanFlipController
    spec, env = make_env(seed=args.seed, scenario=args.scenario, profile=args.profile,
                         robot_id=args.robot)
    info = {}
    try:
        env.reset()
        controller = PanFlipController(env)
        for _ in range(spec.horizon_steps):
            _, _, terminated, truncated, info = env.step(controller.act())
            if terminated or truncated:
                break
        payload = dict(robot_id=args.robot, success=bool(env._success), phase=env.phase,
                       failure=info.get('failure'), elapsed_sec=float(env.sim.data.time),
                       scenario=env.scenario.to_dict())
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2) + '\n')
        except OSError as exc:
            print(f'[pan_flip] warning: could not save result: {exc}', flush=True)
        print(json.dumps(payload), flush=True)
        return 0 if payload['success'] else 1
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
