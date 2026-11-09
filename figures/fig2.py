"""
Figure 2: multi-scale geometry for a two-dimensional stimulus (toy model).

Input: the output of analysis/tuning_toy_2d.py.

Usage:
    python figures/fig2.py --input runs/analysis/tuning_toy_2d.npz --out-dir figures/out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.tuning_toy_2d import A, K  # noqa: E402
from figures._style import plt, save_figure  # noqa: E402

SCRIPT = "fig2"
BLUE, ORANGE = "#1F77B4", "#FF7F0E"
BACKGROUND = "#4a4a4a"


def draw_map(ax, metric, show, density=None):
    """Ellipses whose semi-axes are sqrt(eigenvalue) along the eigenvectors, scaled per panel."""
    ax.set_facecolor(BACKGROUND)
    if density is not None:
        rgba = colormaps["inferno"](density.T / density.max())
        rgba[..., 3] = np.clip(density.T / (0.15 * density.max()), 0, 1)
        ax.imshow(rgba, extent=(-5, 5, -5, 5), origin="lower", interpolation="bilinear")
    largest = np.sqrt(np.linalg.eigvalsh(metric).max())
    for i, x1 in enumerate(show):
        for j, x2 in enumerate(show):
            e, v = np.linalg.eigh(metric[i, j])
            e = 0.55 * np.sqrt(np.clip(e, 0, None)) / largest
            ax.add_patch(Ellipse((x1, x2), 2 * e[1], 2 * e[0], fill=False, ec="w", lw=0.9,
                                 angle=np.degrees(np.arctan2(v[1, 1], v[0, 1]))))
    ax.set(xlim=(-5, 5), ylim=(-5, 5), aspect="equal", xlabel="stim dim 1", ylabel="stim dim 2")


def main(argv: list[str] | None = None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", default="runs/analysis/tuning_toy_2d.npz")
    p.add_argument("--out-dir", default="figures/out")
    args = p.parse_args(argv)

    if not Path(args.input).exists():
        print(f"[{SCRIPT}] {args.input} not found (run analysis/tuning_toy_2d.py); nothing to draw.")
        return []
    d = np.load(args.input)
    show = d["show"]

    fig = plt.figure(figsize=(14, 6))
    grid = fig.add_gridspec(2, 6, width_ratios=[1.9, 1, 1, 1, 1, 0.05], hspace=0.38, wspace=0.42)
    ax = fig.add_subplot(grid[0:2, 0], projection="3d")
    s = np.linspace(-5, 5, 200)
    ax.plot(s, np.zeros_like(s), A / (1 + np.exp(-K * s)), color=BLUE, lw=1.6)
    ax.plot(np.zeros_like(s), s, A / (1 + np.exp(K * s)), color=ORANGE, lw=1.6)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect(None, zoom=0.95)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("stim dim 1", fontsize=8, labelpad=2)
    ax.set_ylabel("stim dim 2", fontsize=8, labelpad=2)
    ax.set_zlabel("$f(x)$", fontsize=8, labelpad=2)
    ax.set_title("tuning curves", fontsize=10, y=1.02)
    ax.text2D(0.0, 1.05, "A", transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.text2D(0.02, 0.97, "neuron 1", color=BLUE, transform=ax.transAxes, fontsize=8)
    ax.text2D(0.02, 0.92, "neuron 2", color=ORANGE, transform=ax.transAxes, fontsize=8)
    titles = ("$t=0$", "$t=1.5$", "$t=4$", "$G(x)$")
    letters = {("uniform", 0): "B", ("uniform", 3): "C", ("ridge", 0): "D", ("ridge", 3): "E"}
    for row, prior in enumerate(("uniform", "ridge")):
        density = None if prior == "uniform" else d["ridge_density"]
        for col, key in enumerate(("J0", "J1p5", "J4", "G")):
            ax = fig.add_subplot(grid[row, col + 1])
            draw_map(ax, d[f"{prior}_{key}"], show, density)
            ax.set_title(titles[col], fontsize=10)
            if (prior, col) in letters:
                ax.text(-0.22, 1.05, letters[prior, col], transform=ax.transAxes, fontsize=13,
                        fontweight="bold", va="bottom")
    cax = fig.add_subplot(grid[1, 5])
    fig.colorbar(ScalarMappable(Normalize(0, d["ridge_density"].max()), "inferno"), cax=cax, label="$p(x)$")
    return save_figure(fig, Path(args.out_dir), "fig2")


if __name__ == "__main__":
    main()
