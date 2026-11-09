"""
Figure 6: the multi-scale geometry under correlated noise (Appendix D).

  A  mutual information (closed form) and 1/2 <Tr G> (estimator) for the noise
     structures of Table 2.
  B  change in information relative to the trial-shuffled surrogate for each
     correlated noise structure, with its mean pairwise noise correlation.

Input: the output of analysis/correlated_noise.py, which also prints Table 2.

Usage:
    python figures/fig_corrnoise.py --input runs/analysis/correlated_noise.json \\
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

SCRIPT = "fig_corrnoise"
NAMES = ["Independent", "After encoding, along signal ($c=1$)",
         "After encoding, along signal ($c=4$)", "After encoding, reduced along signal",
         "Before encoding, $f=0.1$", "Before encoding, $f=0.3$", "Before encoding, $f=1.0$"]
COLORS = ["0.45", "#C0392B", "#C0392B", "#2E86C1", "#7D3C98", "#7D3C98", "#7D3C98"]


def main(argv: list[str] | None = None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", default="runs/analysis/correlated_noise.json")
    p.add_argument("--out-dir", default="figures/out")
    args = p.parse_args(argv)

    if not Path(args.input).exists():
        print(f"[{SCRIPT}] {args.input} not found (run analysis/correlated_noise.py); "
              "nothing to draw.")
        return []
    with open(args.input) as f:
        rows = json.load(f)["rows"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9),
                                 gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.75))
    x = np.arange(len(rows))
    a1.bar(x - 0.2, [r["I_exact"] for r in rows], 0.4, color="0.3", label="$I(R;X)$, closed form")
    a1.bar(x + 0.2, [r["half_trace_estimate"] for r in rows], 0.4, yerr=[r["sem"] for r in rows],
           color="#7FA7D9", label=r"$\frac{1}{2}\langle\mathrm{Tr}\,G\rangle$, estimator")
    a1.set_xticks(x)
    a1.set_xticklabels(NAMES, rotation=35, ha="right", fontsize=8)
    a1.set_ylabel("nats")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("The estimator recovers the information", fontsize=10)
    yy = np.arange(len(rows) - 1)[::-1]
    for y, r, c in zip(yy, rows[1:], COLORS[1:]):
        change = r["delta_I_vs_shuffled_pct"]
        a2.barh(y, change, color=c, height=0.65)
        a2.text(change + (1.5 if change > 0 else -1.5), y, f"corr. {r['mean_corr']:.2f}", va="center",
                ha="left" if change > 0 else "right", fontsize=8)
    a2.set_yticks(yy)
    a2.set_yticklabels(NAMES[1:], fontsize=8)
    a2.axvline(0, color="0.3", lw=0.8)
    a2.set_xlim(-75, 38)
    a2.set_xlabel("change in information vs shuffled (%)")
    a2.set_title("Information change depends on the noise direction", fontsize=10)
    for ax, lab in ((a1, "A"), (a2, "B")):
        ax.text(-0.12 if ax is a1 else -0.72, 1.08, lab, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom")
    return save_figure(fig, Path(args.out_dir), "fig6_corrnoise")


if __name__ == "__main__":
    main()
