"""
Eigenvector statistics across test images: Fig. 3G-H, Fig. 7 and Appendix F.

For the top-10 eigenvectors v of G(x) and J(x) at each of the 200 test images
(Stage-4 files <runs-dir>/per_image/<encoder>/<area>/imgNNNN.pt), for the
Inception and ResNet-50 encoders fitted to V1 and V4, measure:

  extent      energy-weighted RMS radius of |v|² around its centroid (pixels)
  frequency   RMS spatial frequency of the 2-D power spectrum (cycles/image)
  components  number of connected components of |v| > 0.2 max|v|

Statistics:
  per_image    average each measure over ranks 1-10 and both encoders, then
               compare V1 with V4 on the same image (two-sided Wilcoxon
               signed-rank test; counts of images with V1 < V4 and V1 > V4).
               This is what the paper reports. Also the mean over images of
               |V1 - V4| (Section 6).
  per_encoder  the same for each encoder, with its medians.
  per_rank     the same for each rank.
  pooled       all eigenvectors pooled per area, two-sided Mann-Whitney U. This
               treats the eigenvectors of one image as independent, so the
               p-values are optimistic; the paper does not use it.

Output: eigvec_statistics.npz, the per-image values [image, encoder, area, rank,
measure] for G and J, read by figures/fig3.py and figures/fig_eigstats.py, and
eigvec_statistics.json with the statistics.

Usage:
    python analysis/eigvec_statistics.py --runs-dir runs --out-dir runs/analysis
    python analysis/eigvec_statistics.py --from-npz runs/analysis/eigvec_statistics.npz \\
        --out-dir runs/analysis      # statistics only, from saved per-image values
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as cc_label
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import provenance  # noqa: E402

ENCS, AREAS, K, H = ("inception", "resnet50"), ("V1", "V4"), 10, 64
NAMES = ("extent_px", "sf_cyc_per_img", "components")
ys, xs = np.mgrid[0:H, 0:H]
fy = np.fft.fftfreq(H) * H
FY, FX = np.meshgrid(fy, fy, indexing="ij")
F2 = FY ** 2 + FX ** 2


def extent(V: np.ndarray) -> float:
    p = V ** 2
    p = p / p.sum()
    cy, cx = (p * ys).sum(), (p * xs).sum()
    return float(np.sqrt((p * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum()))


def sf(V: np.ndarray) -> float:
    P = np.abs(np.fft.fft2(V)) ** 2
    P = P / P.sum()
    return float(np.sqrt((P * F2).sum()))


def comps(V: np.ndarray) -> int:
    A = np.abs(V)
    return int(cc_label(A > 0.2 * A.max())[1])


def measure_all(runs_dir: Path) -> tuple[dict, list[str]]:
    """Per-image values {"G", "J": [image, encoder, area, rank, measure]} and image names."""
    N = runs_dir / "per_image"
    ids = sorted(os.path.basename(p) for p in glob.glob(f"{N}/inception/V1/img*.pt"))
    if not ids:
        raise FileNotFoundError(f"no Stage-4 files in {N}/inception/V1")
    X = {m: np.zeros((len(ids), 2, 2, K, 3)) for m in ("G", "J")}
    for i, f in enumerate(ids):
        for e, enc in enumerate(ENCS):
            for a, area in enumerate(AREAS):
                d = torch.load(N / enc / area / f, weights_only=False)
                for m in ("G", "J"):
                    V = d[f"{m}_eigenvectors"].numpy().astype(np.float64)[:K]
                    for r in range(K):
                        v = V[r].reshape(H, H)
                        X[m][i, e, a, r] = (extent(v), sf(v), comps(v))
    return X, ids


def summarise(X: dict) -> dict:
    n_images = X["G"].shape[0]
    out = {"n_images": n_images, "top_k": K, "encoders": ENCS}
    for m in ("G", "J"):
        R = {}
        for q, nm in enumerate(NAMES):
            v1, v4 = X[m][:, :, 0, :, q], X[m][:, :, 1, :, q]
            pooled = dict(V1_median=float(np.median(v1)), V4_median=float(np.median(v4)),
                          V1_mean=float(v1.mean()), V4_mean=float(v4.mean()),
                          n_per_area=int(v1.size),
                          mannwhitney_p=float(mannwhitneyu(v1.ravel(), v4.ravel()).pvalue))
            a1, a4 = v1.mean((1, 2)), v4.mean((1, 2))        # per image: mean over encoders, ranks
            per_img = dict(V1_median=float(np.median(a1)), V4_median=float(np.median(a4)),
                           wilcoxon_p=float(wilcoxon(a1, a4).pvalue),
                           n_V1_lt_V4=int((a1 < a4).sum()), n_V1_gt_V4=int((a1 > a4).sum()),
                           mean_abs_V1_minus_V4=float(np.abs(a1 - a4).mean()))
            per_enc = {enc: dict(V1_median=float(np.median(v1[:, e].mean(1))),
                                 V4_median=float(np.median(v4[:, e].mean(1))),
                                 wilcoxon_p=float(wilcoxon(v1[:, e].mean(1), v4[:, e].mean(1)).pvalue),
                                 n_V1_lt_V4=int((v1[:, e].mean(1) < v4[:, e].mean(1)).sum()))
                       for e, enc in enumerate(ENCS)}
            per_rank = [dict(rank=r + 1, V1_median=float(np.median(v1[:, :, r].mean(1))),
                             V4_median=float(np.median(v4[:, :, r].mean(1))),
                             wilcoxon_p=float(wilcoxon(v1[:, :, r].mean(1), v4[:, :, r].mean(1)).pvalue),
                             n_V1_lt_V4=int((v1[:, :, r].mean(1) < v4[:, :, r].mean(1)).sum()))
                        for r in range(K)]
            R[nm] = dict(pooled=pooled, per_image=per_img, per_encoder=per_enc, per_rank=per_rank)
        out[m] = R
    return out


def report(out: dict) -> None:
    n = out["n_images"]
    for m in ("G", "J"):
        print(f"== {m}")
        for nm in NAMES:
            r = out[m][nm]
            q = r["per_image"]
            print(f"  {nm:15s} per image median V1 {q['V1_median']:.2f} V4 {q['V4_median']:.2f}, "
                  f"V1<V4 in {q['n_V1_lt_V4']}/{n}, V1>V4 in {q['n_V1_gt_V4']}/{n}, "
                  f"Wilcoxon p={q['wilcoxon_p']:.1e}, mean |V1-V4| {q['mean_abs_V1_minus_V4']:.2f} | "
                  f"per encoder median V1, V4: "
                  + ", ".join(f"{e} {v['V1_median']:.2f} {v['V4_median']:.2f}"
                              for e, v in r["per_encoder"].items())
                  + f" | ranks V1<V4: {[x['n_V1_lt_V4'] for x in r['per_rank']]}")


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--runs-dir", default="runs", help="root with per_image/<encoder>/<area>/")
    p.add_argument("--out-dir", default="runs/analysis")
    p.add_argument("--from-npz", default=None,
                   help="recompute the statistics from a saved eigvec_statistics.npz")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_npz:
        with np.load(args.from_npz) as d:
            X = {"G": d["G"], "J": d["J"]}
    else:
        X, ids = measure_all(Path(args.runs_dir))
        np.savez(out_dir / "eigvec_statistics.npz", G=X["G"], J=X["J"], ids=np.array(ids),
                 measures=np.array(NAMES))
    out = summarise(X)
    out["provenance"] = provenance()
    with open(out_dir / "eigvec_statistics.json", "w") as f:
        json.dump(out, f, indent=1)
    report(out)
    return out


if __name__ == "__main__":
    main()
