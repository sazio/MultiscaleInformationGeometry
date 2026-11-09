"""
Figure 7: eigenvector statistics by rank (Appendix F).

Mean over the test images (lines) and s.e.m. (shading) of the spatial extent,
spatial frequency and number of connected components of the k-th eigenvector of
G(x) (top) and J(x) (bottom), for the Inception (filled circles) and ResNet-50
(open squares) models fitted to V1 (blue) and V4 (yellow).

Input: the .npz written by analysis/eigvec_statistics.py.

Usage:
    python figures/fig_eigstats.py --input runs/analysis/eigvec_statistics.npz \\
        --out-dir figures/out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures._style import plt, save_figure  # noqa: E402

SCRIPT = "fig_eigstats"
V1_COLOR, V4_COLOR = "#3B7DC4", "#E3A33B"
NAMES = ["spatial extent (px)", "spatial frequency", "connected components"]


def main(argv: list[str] | None = None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", default="runs/analysis/eigvec_statistics.npz")
    p.add_argument("--out-dir", default="figures/out")
    args = p.parse_args(argv)

    if not Path(args.input).exists():
        print(f"[{SCRIPT}] {args.input} not found (run analysis/eigvec_statistics.py); "
              "nothing to draw.")
        return []
    with np.load(args.input) as d:
        G, J = d["G"], d["J"]                      # image, encoder, area, rank, measure

    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
    k = np.arange(1, G.shape[3] + 1)
    for r, (M, lab) in enumerate(((G, "multi-scale $G(x)$"), (J, "Fisher $J(x)$"))):
        for q in range(3):
            ax = axs[r, q]
            for a, c, area in ((0, V1_COLOR, "V1"), (1, V4_COLOR, "V4")):
                for e, (enc, marker, face) in enumerate((("Inception", "o", c),
                                                         ("ResNet-50", "s", "white"))):
                    per = M[:, e, a, :, q]         # image x rank, one encoder
                    mu = per.mean(0)
                    sem = per.std(0, ddof=1) / np.sqrt(per.shape[0])
                    ax.fill_between(k, mu - sem, mu + sem, color=c, alpha=0.25, lw=0)
                    ax.plot(k, mu, "-", marker=marker, color=c, mfc=face, ms=4, lw=1.3,
                            label=f"{area}, {enc}")
            ax.set_title(f"{lab}: {NAMES[q]}", fontsize=9)
            if r == 1:
                ax.set_xlabel("Eigenvector rank")
            ax.tick_params(labelsize=8)
        axs[r, 0].set_ylabel("pixels")
        axs[r, 1].set_ylabel("cycles/image")
        axs[r, 2].set_ylabel("count")
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return save_figure(fig, Path(args.out_dir), "fig7_eigstats")


if __name__ == "__main__":
    main()
