#!/usr/bin/env python3
"""Command-line entry point for reproducing Figures 1--10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hassager2021.figures import reproduce_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the computational figures from Hassager, Wang and "
            "Huang, Physics of Fluids 33, 123108 (2021)."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Output directory (default: outputs).",
    )
    parser.add_argument(
        "--quality",
        choices=("fast", "standard"),
        default="standard",
        help="Spatial resolution preset.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution in dots per inch.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")
    metrics = reproduce_all(args.output, quality=args.quality, dpi=args.dpi)
    print(json.dumps(metrics, indent=2))
    print(f"\nFigures and data written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
