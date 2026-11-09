"""
Figure 3: multi-scale geometry of the V1 and V4 population models (Inception
encoder).

  A    model schematic. Drawn separately and NOT produced here; the left of the
       top row is left empty for it.
  B-C  measured versus predicted test responses, all 50 neurons and all test
       images pooled, for V1 (B) and V4 (C). The title gives the mean
       per-neuron Pearson correlation. Source: Stage-1 checkpoints
       (test_responses, test_predictions, per_neuron_test_corr).
  D    the example image with the V1 (blue) and V4 (yellow) receptive-field
       regions outlined. Each neuron's region is where its Gaussian readout
       mask is at least half its peak; the outline is the boundary of the
       union over the 50 neurons (see `readout_rf_union` for the mapping from
       readout coordinates to pixels).
  E    four eigenvectors of G(x) for the example image, V1 (top) and V4
       (bottom). Source: Stage-4 per-image files of Inception and ResNet-50.
  F    same as E for J(x).
  G    top-10 eigenvectors of G(x) across the 200 test images, V1 against V4:
       spatial extent and spatial frequency, one point per image, averaged over
       ranks and the two encoders. Source: analysis/eigvec_statistics.py.
  H    same as G for J(x). G and H share their axes for each measure.

The tiles of E-F are selected, not in rank order; figures/_eigvec_selection.py
states the rule, and the selected ranks are written to
fig3_selection_img{idx}.json. They are drawn with a symmetric diverging colour
map (red positive, blue negative), each scaled to its own 99th percentile of
|v|, with the largest-magnitude pixel positive.

Usage:
    python figures/fig3.py --runs-dir runs --out-dir figures/out --img-idx 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures._eigvec_selection import N_TILES_FIG3, fig3_tiles
from figures._style import (
    AREA_COLORS, AREAS, IMAGE_SIZE, METRIC_TITLES, RF_COLORS, as_numpy, blank,
    encoder_checkpoint, panel_label, per_image, plt, save_figure, show_eigenvector, skip,
)

SCRIPT = "fig3"
ENCODER = "inception"
PARTNER = "resnet50"                   # second encoder of the tile selection rule
STATISTIC_TITLES = ("Spatial extent (px)", "Spatial frequency (cycles/image)")
# Feature-map side length per backbone, used only if a checkpoint lacks the
# readout grid buffer (src/encoders.py: out_spatial).
FEATURE_SIDE = {"inception": 12, "resnet50": 16, "onoff": 64}


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="figures/out")
    p.add_argument("--img-idx", type=int, default=0, help="example image (Stage-4 index)")
    p.add_argument("--n-eigvecs", type=int, default=N_TILES_FIG3)
    p.add_argument("--statistics", default=None,
                   help="output of analysis/eigvec_statistics.py for panels G-H; "
                        "default: <runs-dir>/analysis/eigvec_statistics.npz")
    return p.parse_args(argv)


def readout_rf_union(state: dict, encoder: str = ENCODER, resolution: int = 256):
    """
    Union of the neurons' half-maximum readout regions, sampled on a
    `resolution`² grid over the 64 × 64 image.

    The readout (src/encoders.py, GaussianReadout) weights feature-map cell
    (h, w) by exp(−‖(u_w, u_h) − μ_n‖² / (2σ_n²)), with the S cells placed at
    u = linspace(−1, 1, S); μ_n[0] is the horizontal (column) and μ_n[1] the
    vertical (row) coordinate, and σ_n is clamped at 0.05 as in the forward
    pass. We take feature cell s to cover image pixels [s·64/S, (s+1)·64/S),
    so a point at pixel coordinate p (pixel k spans [k, k+1)) has
        u(p) = 2 (p·S/64 − 1/2) / (S − 1) − 1.
    This maps cell centres to pixel positions only; the backbone's own
    receptive field, which widens each region further, is not included.

    Returns (field [resolution, resolution] = max_n of the normalised
    Gaussians, pixel coordinates [resolution] of the samples).
    """
    mu = torch.as_tensor(state["readout.mu"]).double()                       # [N, 2]
    sigma = torch.as_tensor(state["readout.sigma"]).double().clamp(min=0.05)  # [N]
    grid = state.get("readout.grid_x")
    S = int(grid.numel()) if grid is not None else FEATURE_SIDE[encoder]
    p = (torch.arange(resolution, dtype=torch.float64) + 0.5) * IMAGE_SIZE / resolution
    u = 2.0 * (p * S / IMAGE_SIZE - 0.5) / (S - 1) - 1.0
    dx2 = (u.unsqueeze(0) - mu[:, 0:1]) ** 2                                  # [N, R] columns
    dy2 = (u.unsqueeze(0) - mu[:, 1:2]) ** 2                                  # [N, R] rows
    field = torch.exp(-(dy2.unsqueeze(2) + dx2.unsqueeze(1))
                      / (2.0 * sigma.view(-1, 1, 1) ** 2))                    # [N, R, R]
    return field.max(dim=0).values.numpy(), p.numpy()


def panel_scatter(ax, ckpt: dict | None, area: str) -> None:
    if ckpt is None or "test_predictions" not in ckpt:
        if ckpt is not None:
            skip(SCRIPT, f"{area} scatter: checkpoint has no test_predictions")
        blank(ax)
        return
    pred = as_numpy(ckpt["test_predictions"]).ravel()
    meas = as_numpy(ckpt["test_responses"]).ravel()
    r_mean = float(np.mean(as_numpy(ckpt["per_neuron_test_corr"])))
    ax.scatter(meas, pred, s=4, alpha=0.4, c=AREA_COLORS[area], edgecolors="none",
               rasterized=True)
    lo = min(meas.min(), pred.min())
    hi = max(meas.max(), pred.max())
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, "--", color="black", lw=1.0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("measured response")
    ax.set_ylabel("predicted response")
    ax.set_title(rf"{area}   $\bar r$ = {r_mean:.2f}", fontsize=12)


def panel_rf(ax, image, ckpts: dict) -> None:
    if image is None:
        blank(ax)
        return
    ax.imshow(as_numpy(image).reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray", vmin=-1, vmax=1,
              extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0))
    x_text = 0.0
    for area in AREAS:
        ckpt = ckpts.get(area)
        if ckpt is None:
            continue
        field, coords = readout_rf_union(ckpt["model_state_dict"])
        ax.contour(coords, coords, field, levels=[0.5], colors=[RF_COLORS[area]],
                   linewidths=2.0)
        ax.text(x_text, -0.03, f"{area} RFs", transform=ax.transAxes, color=RF_COLORS[area],
                fontsize=11, fontweight="bold", va="top")
        x_text += 0.5
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def panel_eigvecs(fig, spec, tiles: dict, metric: str, n_show: int):
    """Two rows (V1, V4) of the selected `n_show` eigenvectors of one metric."""
    inner = spec.subgridspec(3, n_show, height_ratios=[0.12, 1, 1], hspace=0.25, wspace=0.04)
    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.text(0.5, 0.0, METRIC_TITLES[metric], ha="center", va="bottom", fontsize=13,
                  transform=title_ax.transAxes)
    first = None
    for r, area in enumerate(AREAS):
        vecs = tiles.get(area)
        for k in range(n_show):
            ax = fig.add_subplot(inner[r + 1, k])
            first = first or ax
            if vecs is None or k >= len(vecs):
                blank(ax)
            else:
                show_eigenvector(ax, vecs[k])
            if k == 0:
                ax.set_ylabel(area, fontsize=11)
    return first


def select_tiles(records: dict, partners: dict, metric: str, n_show: int):
    """Selected tiles and their ranks, per area, for one metric."""
    tiles, ranks = {}, {}
    for area in AREAS:
        rec, partner = records.get(area), partners.get(area)
        if rec is None or partner is None:
            continue
        tiles[area], ranks[f"{metric}_{area}"] = fig3_tiles(
            rec[f"{metric}_eigenvectors"], partner[f"{metric}_eigenvectors"], metric, n_show)
    return tiles, ranks


def area_means(statistics: dict, metric: str, measure: int) -> tuple:
    """Per image, V1 and V4: one measure averaged over ranks and encoders."""
    per_image_values = statistics[metric][:, :, :, :, measure].mean(3)   # image, encoder, area
    return per_image_values[:, :, 0].mean(1), per_image_values[:, :, 1].mean(1)


def panel_statistics(fig, spec, statistics: dict | None) -> tuple:
    """Panels G and H: V1 against V4, one point per image, extent and frequency."""
    inner = spec.subgridspec(1, 4, wspace=0.45)
    axes = [fig.add_subplot(inner[0, j]) for j in range(4)]
    for j, (metric, measure) in enumerate((("G", 0), ("G", 1), ("J", 0), ("J", 1))):
        ax = axes[j]
        if statistics is None:
            blank(ax)
            continue
        v1, v4 = area_means(statistics, metric, measure)
        pool = np.concatenate([x for m in ("G", "J") for x in area_means(statistics, m, measure)])
        lo, hi = float(pool.min()), float(pool.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", lw=1.2, zorder=0)
        ax.scatter(v1, v4, s=10, color="0.25", alpha=0.6, lw=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("V1", color=AREA_COLORS["V1"])
        ax.set_ylabel("V4", color=AREA_COLORS["V4"])
        ax.set_title(STATISTIC_TITLES[measure], fontsize=10)
    return axes[0], axes[2]


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    ckpts = {area: encoder_checkpoint(runs_dir, ENCODER, area, SCRIPT) for area in AREAS}
    records = {area: per_image(runs_dir, ENCODER, area, args.img_idx, SCRIPT) for area in AREAS}
    partners = {area: per_image(runs_dir, PARTNER, area, args.img_idx, SCRIPT) for area in AREAS}
    image = next((rec["image"] for rec in records.values() if rec is not None), None)
    stats_path = Path(args.statistics) if args.statistics else \
        runs_dir / "analysis" / "eigvec_statistics.npz"
    statistics = None
    if stats_path.exists():
        statistics = dict(np.load(stats_path, allow_pickle=False))
    else:
        skip(SCRIPT, f"panels G-H: {stats_path} not found (run analysis/eigvec_statistics.py)")

    fig = plt.figure(figsize=(15, 10.8))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 0.85], hspace=0.32)
    top = outer[0].subgridspec(1, 4, width_ratios=[2.2, 1, 1, 0.1], wspace=0.35)
    ax_b = fig.add_subplot(top[1])
    ax_c = fig.add_subplot(top[2])
    panel_scatter(ax_b, ckpts["V1"], "V1")
    panel_scatter(ax_c, ckpts["V4"], "V4")

    middle = outer[1].subgridspec(1, 3, width_ratios=[1.0, 2.2, 2.2], wspace=0.15)
    ax_d = fig.add_subplot(middle[0])
    panel_rf(ax_d, image, ckpts)
    tiles_g, ranks_g = select_tiles(records, partners, "G", args.n_eigvecs)
    tiles_j, ranks_j = select_tiles(records, partners, "J", args.n_eigvecs)
    ax_e = panel_eigvecs(fig, middle[1], tiles_g, "G", args.n_eigvecs)
    ax_f = panel_eigvecs(fig, middle[2], tiles_j, "J", args.n_eigvecs)
    ax_g, ax_h = panel_statistics(fig, outer[2], statistics)

    for letter, ax in zip("BCDEFGH", (ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h)):
        panel_label(fig, ax, letter)
    paths = save_figure(fig, out_dir, f"fig3_img{args.img_idx:04d}")
    selection = out_dir / f"fig3_selection_img{args.img_idx:04d}.json"
    selection.write_text(json.dumps({**ranks_g, **ranks_j}, indent=1))
    print(f"saved {selection}")
    return paths + [selection]


if __name__ == "__main__":
    main()
