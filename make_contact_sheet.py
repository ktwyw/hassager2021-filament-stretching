#!/usr/bin/env python3
"""Assemble the generated panels into a single contact sheet and a figure index."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()

    figure_dir = args.output / "figures"
    panels = sorted(figure_dir.glob("figure_*.png"))
    if not panels:
        raise SystemExit(f"No panels found in {figure_dir}")

    columns = max(1, args.columns)
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6.0 * columns, 4.9 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, panel in zip(axes.ravel(), panels, strict=False):
        ax.imshow(mpimg.imread(panel))
        ax.set_title(panel.name, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(args.output / "contact_sheet.png", dpi=90)

    index = ["# Generated figure panels", ""]
    index += [f"- `figures/{panel.name}`" for panel in panels]
    (args.output / "FIGURE_INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Contact sheet with {len(panels)} panels written to {args.output / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
