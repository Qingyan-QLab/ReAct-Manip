"""Start an RPent Pi0.5 server with deterministic eager inference for validation."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        help="Disable compile for RPent's LIBERO Pi0.5 implementation.",
    )
    args, rpent_args = parser.parse_known_args()

    import numpy as np
    import torch
    from rpent.robots.components import pi05_vla_server as server

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.disable_torch_compile:
        server.PI05_EMBODIMENTS["libero"]["openpi"]["torch_compile"] = False
    sys.argv = [sys.argv[0], *rpent_args]
    server.main()


if __name__ == "__main__":
    main()
