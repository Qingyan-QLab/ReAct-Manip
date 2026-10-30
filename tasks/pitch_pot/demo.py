#!/usr/bin/env python3
"""Run the calibrated pitch-pot demo on the selected embodiment."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', default='piperx',
                        choices=('panda', 'sawyer', 'piperx', 'aloha', 'aloha_mobile'))
    args, forwarded = parser.parse_known_args(argv)
    from tasks.pitch_pot.embodiment_demo import main as run_demo
    return run_demo(args.robot, forwarded)


if __name__ == '__main__':
    raise SystemExit(main())
