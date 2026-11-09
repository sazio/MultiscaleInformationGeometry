"""
Figure 8: Monte Carlo convergence of the estimator (Appendix G).

  A  coefficient of variation of the leading eigenvalue of G(x) across runs against
     N_MC, with an N_MC^(-1/2) reference through N_MC = 25.
  B  mean pairwise alignment of the top-5 eigenvectors across runs against N_MC.

Mean and s.e.m. over images. Input: the output of
analysis/mc_convergence_summary.py.

Usage:
    python figures/fig_mc.py --input runs/analysis/mc_convergence_summary.json \\
        --out-dir figures/out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures._style import plt, save_figure  # noqa: E402

SCRIPT = "fig_mc"
V1_COLOR, V4_COLOR = "#3B7DC4", "#E3A33B"


def main(argv: list[str] | None = None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", default="runs/analysis/mc_convergence_summary.json")
    p.add_argument("--out-dir", default="figures/out")
    args = p.parse_args(argv)

    if not Path(args.input).exists():
        print(f"[{SCRIPT}] {args.input} not found (run analysis/mc_convergence_summary.py); "
              "nothing to draw.")
        return []
    with open(args.input) as f:
        d = json.load(f)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.45))
    for area, c in (("V1", V1_COLOR), ("V4", V4_COLOR)):
        n = np.array(sorted(int(k) for k in d[area] if k.isdigit()))
        r = [d[area][str(k)] for k in n]
        a1.errorbar(n, [x["lam1_cv_mean"] for x in r], yerr=[x["lam1_cv_sem"] for x in r], color=c,
                    marker="o", ms=4, lw=1.5, capsize=2, label=area)
        a2.errorbar(n, [x["top5_align_mean"] for x in r], yerr=[x["top5_align_sem"] for x in r],
                    color=c, marker="o", ms=4, lw=1.5, capsize=2, label=area)
    ref = d["V1"]["25"]["lam1_cv_mean"] * np.sqrt(25 / n)
    a1.plot(n, ref, color="0.6", ls=":", lw=1.2, label=r"$\propto N_{\rm MC}^{-1/2}$")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a2.set_xscale("log")
    for ax in (a1, a2):
        ax.set_xticks(n)
        ax.set_xticklabels([str(k) for k in n])
        ax.minorticks_off()
        ax.axvline(200, color="0.85", lw=1, zorder=0)
        ax.set_xlabel(r"Monte Carlo samples per scale, $N_{\rm MC}$")
    a1.set_yticks([0.02, 0.05, 0.1])
    a1.set_yticklabels(["0.02", "0.05", "0.1"])
    a1.set_ylabel("CV of leading eigenvalue")
    a2.set_ylabel("Top-5 alignment across runs")
    a2.set_ylim(0.6, 1.0)
    a1.legend(frameon=False, fontsize=8)
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    for ax, lab in ((a1, "A"), (a2, "B")):
        ax.text(-0.27, 1.04, lab, transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="bottom")
    return save_figure(fig, Path(args.out_dir), "fig8_mc")


if __name__ == "__main__":
    main()
