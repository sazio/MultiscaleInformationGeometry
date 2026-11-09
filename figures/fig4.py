"""
Figure 4: robustness of the geometry across encoding-model architectures.

  A    predictive performance of the Inception, ResNet-50 and ON-OFF encoders in
       V1 and V4: box plot of the per-neuron test Pearson correlations (box =
       quartiles, line = median), with the mean ± s.e.m. across neurons
       overlaid as a black point. Source: Stage-1 checkpoints.
  B    eigenvectors of G(x) for the example image, one row per model, V1 block
       above the V4 block. Source: Stage-4 per-image files.
  C    same as B for J(x).
  D-E  subspace alignment 𝒜_k = (1/k)‖V_Aᵀ V_B‖_F² of the top-k eigenvectors
       of G (D) and J (E), mean ± s.e.m. across images. Solid: models fitted
       to the same area (Inception vs ResNet-50); dashed: the same model fitted
       to V1 and V4. Source: runs/metrics/cross_encoder_metrics.json (Stage 5).

The tiles of B-C are selected, not in rank order; figures/_eigvec_selection.py
states the rule, and the selected ranks are written to
fig4_selection_img{idx}.json. Eigenvector colour scales as in fig3.py.

Every curve of D-E comes from the 200-epoch models and the 200 test images. In
the printed panel D, one curve differs: "V1 vs V4 (Res)" of G comes from the
epoch-50 checkpoints of the ResNet-50 diffusion models, on the first 40 test
images. --as-printed-res-v1v4 redraws that curve from an .npz file holding the
arrays k, ep050_ResV1vsV4_mean and ep050_ResV1vsV4_sem. Neither that file nor
the epoch-50 checkpoints are distributed (see README.md).

Usage:
    python figures/fig4.py --runs-dir runs --out-dir figures/out --img-idx 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures._eigvec_selection import N_TILES_FIG4, fig4_tiles
from figures._style import (
    AREAS, ENCODER_COLORS, ENCODER_LABELS, ENCODERS, METRIC_TITLES, as_numpy, blank,
    cross_encoder_metrics, encoder_checkpoint, panel_label, per_image, plot_comparisons, plt,
    save_figure, show_eigenvector,
)

SCRIPT = "fig4"


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="figures/out")
    p.add_argument("--img-idx", type=int, default=0, help="example image (Stage-4 index)")
    p.add_argument("--n-eigvecs", type=int, default=N_TILES_FIG4)
    p.add_argument("--min-k", type=int, default=1,
                   help="first k drawn in D-E (2 omits k = 1, a comparison of single vectors)")
    p.add_argument("--as-printed-res-v1v4", default=None, metavar="NPZ",
                   help="redraw the G curve 'V1 vs V4 (Res)' of panel D as printed in the paper, "
                        "from the epoch-50 ResNet-50 models on 40 test images (arrays k, "
                        "ep050_ResV1vsV4_mean, ep050_ResV1vsV4_sem). Default: the 200-epoch "
                        "curve from Stage 5, like every other curve.")
    return p.parse_args(argv)


def panel_performance(ax, ckpts: dict) -> None:
    width, gap = 0.24, 0.04
    handles = []
    for ei, enc in enumerate(ENCODERS):
        for gi, area in enumerate(AREAS):
            ckpt = ckpts.get((enc, area))
            if ckpt is None:
                continue
            r = as_numpy(ckpt["per_neuron_test_corr"]).astype(float)
            pos = gi + (ei - 1) * (width + gap)
            bp = ax.boxplot([r], positions=[pos], widths=width, patch_artist=True,
                            showfliers=True, flierprops=dict(marker="o", markersize=2.5,
                                                             markerfacecolor="0.6",
                                                             markeredgecolor="none"))
            bp["boxes"][0].set_facecolor(ENCODER_COLORS[enc])
            bp["boxes"][0].set_edgecolor("black")
            bp["boxes"][0].set_linewidth(0.7)
            bp["medians"][0].set_color("black")
            sem = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
            ax.errorbar(pos, r.mean(), yerr=sem, fmt="o", color="black", markersize=3,
                        capsize=0, lw=1.0, zorder=5)
        if any((enc, a) in ckpts for a in AREAS):
            handles.append(enc)
    ax.axhline(0, color="0.6", ls=":", lw=0.7)
    ax.set_xticks(range(len(AREAS)))
    ax.set_xticklabels(AREAS)
    ax.set_xlim(-0.6, len(AREAS) - 0.4)
    ax.set_ylim(-0.15, 1.32)                      # headroom above the whiskers for the encoder labels
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Test Pearson $r$")
    for i, enc in enumerate(handles):
        ax.text(0.98, 0.99 - 0.07 * i, ENCODER_LABELS[enc], transform=ax.transAxes,
                ha="right", va="top", color=ENCODER_COLORS[enc], fontweight="bold",
                fontsize=10)


def select_tiles(records: dict, metric: str, n_show: int):
    """Selected tiles {(encoder, area): [n, P]} and their ranks, for one metric."""
    tiles, ranks = {}, {}
    for area in AREAS:
        inc, res, onoff = (records.get((e, area)) for e in ("inception", "resnet50", "onoff"))
        if inc is None or res is None:
            continue
        by_encoder, ranks[f"{metric}_{area}"] = fig4_tiles(
            inc[f"{metric}_eigenvectors"], res[f"{metric}_eigenvectors"],
            None if onoff is None else onoff[f"{metric}_eigenvectors"], metric, n_show)
        for enc, vecs in by_encoder.items():
            tiles[(enc, area)] = vecs
    return tiles, ranks


def panel_eigvecs(fig, spec, tiles: dict, metric: str, n_show: int):
    """V1 block above V4 block; one row per encoder; returns the title axes."""
    n_enc = len(ENCODERS)
    heights = [0.25] + ([0.22] + [1.0] * n_enc) * len(AREAS)
    inner = spec.subgridspec(len(heights), n_show + 2, height_ratios=heights,
                             width_ratios=[0.35] + [1.0] * n_show + [0.3],
                             hspace=0.08, wspace=0.04)
    title_ax = fig.add_subplot(inner[0, 1:-1])
    title_ax.axis("off")
    title_ax.text(0.5, 0.0, METRIC_TITLES[metric], ha="center", va="bottom", fontsize=13,
                  transform=title_ax.transAxes)
    row = 1
    for area in AREAS:
        head = fig.add_subplot(inner[row, 1:-1])
        head.axis("off")
        head.text(0.5, 0.0, area, ha="center", va="bottom", fontsize=11,
                  transform=head.transAxes)
        row += 1
        for enc in ENCODERS:
            vecs = tiles.get((enc, area))
            lab = fig.add_subplot(inner[row, 0])
            lab.axis("off")
            lab.text(1.0, 0.5, ENCODER_LABELS[enc], rotation=90, ha="right", va="center",
                     fontsize=10, transform=lab.transAxes)
            for k in range(n_show):
                ax = fig.add_subplot(inner[row, 1 + k])
                if vecs is None or k >= len(vecs):
                    blank(ax)
                else:
                    show_eigenvector(ax, vecs[k])
            dots = fig.add_subplot(inner[row, -1])
            dots.axis("off")
            dots.text(0.5, 0.5, "...", ha="center", va="center", fontsize=12,
                      transform=dots.transAxes)
            row += 1
    return title_ax


def as_printed_curve(path: str) -> dict:
    """The epoch-50, 40-image ResNet-50 V1 vs V4 curve of the printed panel D."""
    with np.load(path) as d:
        curve = (d["k"], d["ep050_ResV1vsV4_mean"], d["ep050_ResV1vsV4_sem"])
    print(f"[{SCRIPT}] panel D: 'V1 vs V4 (Res)' of G from {path} (epoch 50, 40 images), "
          "as printed; all other curves from the 200-epoch models")
    return {"cross_area_resnet50": curve}


def panel_alignment(ax, metrics, metric: str, min_k: int = 1, legend: bool = True,
                    replace: dict | None = None) -> None:
    n = plot_comparisons(ax, metrics, metric, "alignment", SCRIPT, min_k, replace)
    if n == 0:
        blank(ax)
        return
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=min_k if min_k > 1 else 0)
    ax.set_xlabel("Number of eigenvectors")
    ax.set_ylabel("Subspace alignment")
    ax.set_title(METRIC_TITLES[metric], fontsize=12)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="lower right")


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    ckpts, records = {}, {}
    for enc in ENCODERS:
        for area in AREAS:
            ckpt = encoder_checkpoint(runs_dir, enc, area, SCRIPT)
            if ckpt is not None:
                ckpts[(enc, area)] = ckpt
            rec = per_image(runs_dir, enc, area, args.img_idx, SCRIPT)
            if rec is not None:
                records[(enc, area)] = rec
    metrics = cross_encoder_metrics(runs_dir, SCRIPT)

    fig = plt.figure(figsize=(17, 8.5))
    outer = fig.add_gridspec(1, 4, width_ratios=[0.8, 1.6, 1.6, 1.25], wspace=0.22)

    a_spec = outer[0].subgridspec(3, 1, height_ratios=[0.6, 1.0, 0.6])
    ax_a = fig.add_subplot(a_spec[1])
    panel_performance(ax_a, ckpts)

    tiles_g, ranks_g = select_tiles(records, "G", args.n_eigvecs)
    tiles_j, ranks_j = select_tiles(records, "J", args.n_eigvecs)
    ax_b = panel_eigvecs(fig, outer[1], tiles_g, "G", args.n_eigvecs)
    ax_c = panel_eigvecs(fig, outer[2], tiles_j, "J", args.n_eigvecs)

    de_spec = outer[3].subgridspec(5, 1, height_ratios=[0.25, 1.0, 0.35, 1.0, 0.25])
    ax_d = fig.add_subplot(de_spec[1])
    ax_e = fig.add_subplot(de_spec[3])
    replace = as_printed_curve(args.as_printed_res_v1v4) if args.as_printed_res_v1v4 else None
    panel_alignment(ax_d, metrics, "G", args.min_k, replace=replace)
    panel_alignment(ax_e, metrics, "J", args.min_k, legend=False)

    for letter, ax in zip("ABCDE", (ax_a, ax_b, ax_c, ax_d, ax_e)):
        panel_label(fig, ax, letter)
    out_dir = Path(args.out_dir)
    paths = save_figure(fig, out_dir, f"fig4_img{args.img_idx:04d}")
    selection = out_dir / f"fig4_selection_img{args.img_idx:04d}.json"
    selection.write_text(json.dumps({**ranks_g, **ranks_j}, indent=1))
    print(f"saved {selection}")
    return paths + [selection]


if __name__ == "__main__":
    main()
