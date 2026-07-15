"""Offline analysis CLI scaffold.

The full video/frame processing chain is planned in P0 item 9. The dry-run
mode is intentionally model-free so environment setup can be checked on CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("shoplift/configs/pipeline.example.yml"))
    parser.add_argument("--input", type=Path, default=None, help="Video path or frame directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs/shoplift"))
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without running models.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        payload = {
            "config": str(args.config),
            "input": str(args.input) if args.input is not None else None,
            "output": str(args.output),
            "status": "dry_run_ok",
        }
        print(json.dumps(payload, indent=2))
        return 0
    parser.error("offline analysis execution is scheduled for P0 item 9; use --dry-run for setup checks")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
