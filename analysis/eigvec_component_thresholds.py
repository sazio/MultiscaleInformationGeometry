"""
Robustness of the connected-components statistic of Appendix F to its threshold
and connectivity.

Same data and per-image statistic as analysis/eigvec_statistics.py: top-10
eigenvectors of G(x) and J(x), Inception and ResNet-50, V1 and V4, 200 test
images. Components of |v| > thr * max|v| for thr in {0.1, 0.2, 0.3, 0.5} with
4-connectivity (the scipy default, as in the paper), plus 8-connectivity at
thr = 0.2. Per image: mean over ranks and the two encoders; V1 vs V4 by two-sided
Wilcoxon signed-rank test; also per encoder.

Usage:
    python analysis/eigvec_component_thresholds.py --runs-dir runs \\
        --out runs/analysis/eigvec_component_thresholds.json
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
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import provenance  # noqa: E402

ENCS, AREAS, K, H = ("inception", "resnet50"), ("V1", "V4"), 10, 64
CFGS = [("4conn_0.1", 0.1, None), ("4conn_0.2", 0.2, None), ("4conn_0.3", 0.3, None),
        ("4conn_0.5", 0.5, None), ("8conn_0.2", 0.2, np.ones((3, 3), int))]


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--runs-dir", default="runs", help="root with per_image/<encoder>/<area>/")
    p.add_argument("--out", default="runs/analysis/eigvec_component_thresholds.json")
    args = p.parse_args(argv)

    N = Path(args.runs_dir) / "per_image"
    ids = sorted(os.path.basename(f) for f in glob.glob(f"{N}/inception/V1/img*.pt"))
    if not ids:
        raise FileNotFoundError(f"no Stage-4 files in {N}/inception/V1")
    X = {m: np.zeros((len(ids), 2, 2, K, len(CFGS))) for m in ("G", "J")}
    for i, f in enumerate(ids):
        for e, enc in enumerate(ENCS):
            for a, area in enumerate(AREAS):
                d = torch.load(N / enc / area / f, weights_only=False)
                for m in ("G", "J"):
                    V = d[f"{m}_eigenvectors"].numpy().astype(np.float64)[:K]
                    for r in range(K):
                        A = np.abs(V[r].reshape(H, H))
                        for c, (_, thr, st) in enumerate(CFGS):
                            X[m][i, e, a, r, c] = cc_label(A > thr * A.max(), structure=st)[1]
    out = {"n_images": len(ids)}
    for m in ("G", "J"):
        out[m] = {}
        for c, (nm, _, _) in enumerate(CFGS):
            v1, v4 = X[m][:, :, 0, :, c], X[m][:, :, 1, :, c]
            a1, a4 = v1.mean((1, 2)), v4.mean((1, 2))
            res = dict(V1_median=float(np.median(a1)), V4_median=float(np.median(a4)),
                       n_V1_lt_V4=int((a1 < a4).sum()), n_V1_gt_V4=int((a1 > a4).sum()),
                       wilcoxon_p=float(wilcoxon(a1, a4).pvalue))
            for e, enc in enumerate(ENCS):
                b1, b4 = v1[:, e].mean(1), v4[:, e].mean(1)
                res[enc] = dict(n_V1_lt_V4=int((b1 < b4).sum()), wilcoxon_p=float(wilcoxon(b1, b4).pvalue))
            res["ranks_with_V1_lt_V4_majority"] = [
                r + 1 for r in range(K) if (v1[:, :, r].mean(1) < v4[:, :, r].mean(1)).sum() > len(ids) / 2]
            out[m][nm] = res
            print(m, nm, json.dumps(res), flush=True)
    out["provenance"] = provenance()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()
