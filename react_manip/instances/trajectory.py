"""Evaluator-owned, optional trajectories with public observations and array files."""
import json
from pathlib import Path

import numpy as np


class TrajectoryWriter:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / 'trajectory.jsonl').open('x', encoding='utf-8')
        self.index = 0
        self.write({'type': 'header', 'schema_version': 'cl-manip.trajectory.v1', **metadata})

    def write(self, record):
        arrays = {}
        def encode(value):
            if isinstance(value, np.ndarray):
                name = f'a{len(arrays)}'
                if value.dtype.hasobject:
                    raise ValueError('trajectory refuses object arrays')
                arrays[name] = value
                return {'array_file': f'{self.index:06d}.npz', 'array_key': name,
                        'dtype': str(value.dtype), 'shape': list(value.shape)}
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {str(k): encode(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(v) for v in value]
            return value
        encoded = encode(record)
        line = json.dumps(encoded, ensure_ascii=False, allow_nan=False)
        if arrays:
            np.savez_compressed(self.directory / f'{self.index:06d}.npz', **arrays)
        self.stream.write(line + '\n')
        self.stream.flush()
        self.index += 1

    def close(self):
        self.stream.close()


class RecordingPolicy:
    """Log policy inputs before act; successful ticks are logged by the evaluator."""
    def __init__(self, policy, writer):
        self.policy, self.writer, self.tick = policy, writer, 0

    def __getattr__(self, key):
        return getattr(self.policy, key)

    def act(self, observation, info):
        self.writer.write({'type': 'policy_input', 'tick': self.tick,
                           'observation': observation, 'info': info})
        action = self.policy.act(observation, info)
        self.writer.write({'type': 'requested_action', 'tick': self.tick, 'action': action})
        self.tick += 1
        return action
