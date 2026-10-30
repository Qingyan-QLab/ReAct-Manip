"""Run a command with a Codex API key loaded from an operator-controlled file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--codex-home", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("Usage: with_private_codex_key.py ... -- COMMAND [ARGS...]")
    if not args.key_file.is_file():
        raise SystemExit(f"Missing key file: {args.key_file}")
    key = args.key_file.read_text(encoding="utf-8").strip()
    if not key or any(char.isspace() for char in key):
        raise SystemExit("Key file must contain exactly one nonempty token")
    if not os.environ.get("CODEX_BASE_URL"):
        raise SystemExit("Set CODEX_BASE_URL in the calling environment")

    args.codex_home.mkdir(parents=True, exist_ok=True)
    os.chmod(args.codex_home, 0o700)
    if args.working_directory is not None:
        os.chdir(args.working_directory)
    os.environ["CODEX_API_KEY"] = key
    os.environ["CODEX_HOME"] = str(args.codex_home)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
