"""
Figure 1: Fisher and multi-scale Fisher for a one-dimensional stimulus (toy model).

Input: the output of analysis/tuning_toy_1d.py.

Usage:
    python figures/fig1.py --input runs/analysis/tuning_toy_1d.npz --out-dir figures/out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures._style import plt, save_figure  # noqa: E402

SCRIPT = "fig1"
BLUE, ORANGE = "#1F77B4", "#FF7F0E"


def labels(ax):
    ax.text(0.68, 0.95, "bell-shaped", color=BLUE, transform=ax.transAxes, fontsize=8, va="top")
    ax.text(0.68, 0.85, "monotonic", color=ORANGE, transform=ax.transAxes, fontsize=8, va="top")


def main(argv: list[str] | None = None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", default="runs/analysis/tuning_toy_1d.npz")
    p.add_argument("--out-dir", default="figures/out")
    args = p.parse_args(argv)

    if not Path(args.input).exists():
        print(f"[{SCRIPT}] {args.input} not found (run analysis/tuning_toy_1d.py); nothing to draw.")
        return []
    d = np.load(args.input)
    x, show = d["x_tuning"], d["x_show"]
    keep = np.abs(x) <= 3
    t = d["t"][: d["bell_contribution"].shape[0]]

    fig = plt.figure(figsize=(10, 5))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1.08, 0.95], hspace=0.48, wspace=0.42)
    panels = {k: fig.add_subplot(grid[i, j]) for k, (i, j) in
              zip("ABCDEF", ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)))}
    for lab, name, col, title in (("A", "bell", BLUE, "bell-shaped tuning curve"),
                                  ("B", "monotonic", ORANGE, "monotonic tuning curve")):
        ax = panels[lab]
        ax.plot(x[keep], d[f"{name}_f"][keep], color=col, lw=1.5)
        ax.set(xlabel="$x$", ylabel="$f(x)$", title=title, xlim=(-3, 3))
    ax = panels["C"]
    ax.plot(show, d["bell_J"], color=BLUE, lw=1.5, label="bell-shaped")
    ax.plot(show, d["monotonic_J"], color=ORANGE, lw=1.5, ls="--", label="monotonic")
    ax.set(xlabel="$x$", ylabel="$J(x)$", xlim=(-3, 3))
    labels(ax)
    vmax = max(d["bell_contribution"].max(), d["monotonic_contribution"].max())
    for lab, name in (("D", "bell"), ("E", "monotonic")):
        ax = panels[lab]
        im = ax.pcolormesh(t, show, d[f"{name}_contribution"].T, cmap="inferno", vmin=0, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.set(xscale="log", xlabel="$t$", ylabel="$x$")
    bar = fig.colorbar(im, ax=panels["E"], fraction=0.05, pad=0.03)
    bar.ax.set_title("contribution\nat scale $t$", fontsize=8)
    ax = panels["F"]
    ax.plot(show, d["bell_G"], color=BLUE, lw=1.5, label="bell-shaped")
    ax.plot(show, d["monotonic_G"], color=ORANGE, lw=1.5, label="monotonic")
    ax.set(xlabel="$x$", ylabel="$G(x)$", xlim=(-3, 3), ylim=(0, None))
    labels(ax)
    for lab, ax in panels.items():
        ax.text(-0.17, 1.06, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
        if lab not in "DE":
            ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, Path(args.out_dir), "fig1")


if __name__ == "__main__":
    main()
