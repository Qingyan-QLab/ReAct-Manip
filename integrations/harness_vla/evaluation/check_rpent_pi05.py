#!/usr/bin/env python3
"""Bounded real Pi0.5 RPC/action check; produces no benchmark score."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--candidate-bank', type=Path, required=True)
    parser.add_argument('--task', default='box_insertion')
    parser.add_argument('--draw-seed', type=int, default=42)
    parser.add_argument('--chunks', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.chunks <= 20:
        parser.error('--chunks must be between 1 and 20')
    if args.out.exists():
        parser.error('--out must be new')
    config = json.loads(args.config.read_text())
    policy_config = config.get('vla', config)
    if policy_config.get('robot_id') != 'panda_libero':
        parser.error('This check requires panda_libero')
    if config.get('rpent_root'):
        sys.path.insert(0, str(Path(config['rpent_root']).resolve()))

    from integrations.rpent_embodiments import register
    from integrations.rpent_embodiments.control import decode_native
    from integrations.rpent_embodiments.environment import NativeReplayEnv
    from integrations.rpent_embodiments.policy import NativePolicy
    from react_manip.instances.candidate_bank import choose_index, load_bank

    register()
    bank, _, _, groups = load_bank(args.candidate_bank)
    if args.task not in groups:
        parser.error('Task is absent from the bank')
    choices = groups[args.task]
    instance, episodes = choices[choose_index(args.draw_seed, args.task, len(choices))]
    if instance['robot_id'] != 'panda_libero':
        parser.error('Selected task does not use panda_libero')
    result = dict(mode='real_pi05_bounded_check', valid=False, task=args.task,
                  bank_digest=bank['content_digest'], draw_seed=args.draw_seed,
                  episode_id=episodes[0]['episode_id'], chunks=[], steps=0,
                  planner_used=False, protocol_score=None, official_score=None)
    env = policy = None
    try:
        env = NativeReplayEnv(instance, cameras=True)
        env.bind(episodes[0])
        obs, _ = env.reset()
        prompt = env.get_language_prompt()['text']
        result['prompt'] = prompt
        result['target_control_hz'] = obs['control_contract']['control_hz']
        before = np.array(obs['proprio']['right']['qpos'])
        policy = NativePolicy(policy_config)
        result['policy'] = policy.provenance
        policy.start_episode()
        client = policy.client
        raw = []

        class RecordingClient:
            def predict(self, payload):
                reply = client.predict(payload)
                raw.append(np.asarray(reply).tolist())
                return reply

        policy.client = RecordingClient()
        for index in range(args.chunks):
            start = time.monotonic()
            actions = policy.predict(obs, prompt)
            record = dict(index=index, inference_seconds=time.monotonic()-start,
                          raw_actions=raw[-1], adapted_actions=actions.tolist(),
                          executed_steps=0)
            result['chunks'].append(record)
            terminated = truncated = False
            for action in actions:
                normalized = decode_native(action, obs, 'libero_delta7')
                obs, _, terminated, truncated, _ = env.step(normalized)
                record['executed_steps'] += 1
                result['steps'] += 1
                if terminated or truncated:
                    break
            print(json.dumps(dict(chunk=index, steps=result['steps'],
                                  inference_seconds=record['inference_seconds'])), flush=True)
            if terminated or truncated:
                break
        after = np.asarray(obs['proprio']['right']['qpos'])
        if not np.isfinite(after).all():
            raise ValueError('Nonfinite robot state after action execution')
        result.update(valid=True, joint_displacement_l2=float(np.linalg.norm(after-before)),
                      terminated=bool(terminated), truncated=bool(truncated))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            if policy is not None:
                policy.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(valid=result['valid'], steps=result['steps'], out=str(args.out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
