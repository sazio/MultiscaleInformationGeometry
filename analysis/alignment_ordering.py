"""
Per-image test of the ordering of the alignment curves in Fig. 4D-E (Section 6).

At each k, per image: same-area alignment = mean of A_k(Inc V1, Res V1) and
A_k(Inc V4, Res V4); cross-area alignment = mean of A_k(Inc V1, Inc V4) and
A_k(Res V1, Res V4); A_k = ||V_A^T V_B||_F^2 / k (Eq. 42). Reports, for G and J:
images with same-area > cross-area, the two-sided Wilcoxon signed-rank p on the
differences across images, and the same count for each of the four (same-area
pair, cross-area pair) combinations. Section 6 quotes k = 10.

Usage:
    python analysis/alignment_ordering.py --runs-dir runs \\
        --out runs/analysis/alignment_ordering.json
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
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import provenance  # noqa: E402

PAIRS = {"IncV1vsV4": (("inception", "V1"), ("inception", "V4")),
         "ResV1vsV4": (("resnet50", "V1"), ("resnet50", "V4")),
         "IncResV1": (("inception", "V1"), ("resnet50", "V1")),
         "IncResV4": (("inception", "V4"), ("resnet50", "V4"))}


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--runs-dir", default="runs", help="root with per_image/<encoder>/<area>/")
    p.add_argument("--top-k", type=int, default=50, help="largest k tested")
    p.add_argument("--out", default="runs/analysis/alignment_ordering.json")
    args = p.parse_args(argv)

    N = Path(args.runs_dir) / "per_image"
    ids = sorted(os.path.basename(f) for f in glob.glob(f"{N}/inception/V1/img*.pt"))
    if not ids:
        raise FileNotFoundError(f"no Stage-4 files in {N}/inception/V1")
    ks = np.arange(1, args.top_k + 1)
    out = {"n_images": len(ids)}
    for m in ("G", "J"):
        R = {c: np.zeros((len(ids), len(ks))) for c in PAIRS}
        for i, f in enumerate(ids):
            V = {(e, a): torch.load(N / e / a / f, weights_only=False)[f"{m}_eigenvectors"]
                 .numpy().astype(np.float64)
                 for e in ("inception", "resnet50") for a in ("V1", "V4")}
            for c, (q, s) in PAIRS.items():
                C2 = np.cumsum(np.cumsum((V[q][:args.top_k] @ V[s][:args.top_k].T) ** 2, 0), 1)
                R[c][i] = np.diag(C2) / ks
        same = (R["IncResV1"] + R["IncResV4"]) / 2
        cross = (R["IncV1vsV4"] + R["ResV1vsV4"]) / 2
        d = same - cross
        res = {}
        for j, k in enumerate(ks):
            combos = {f"{s}>{x}": int((R[s][:, j] > R[x][:, j]).sum())
                      for s in ("IncResV1", "IncResV4") for x in ("IncV1vsV4", "ResV1vsV4")}
            res[int(k)] = dict(n_same_gt_cross=int((d[:, j] > 0).sum()),
                               n_same_lt_cross=int((d[:, j] < 0).sum()),
                               wilcoxon_p=float(wilcoxon(d[:, j]).pvalue),
                               median_diff=float(np.median(d[:, j])), combos=combos)
        out[m] = res
        for k in (1, 2, 3, 5, 10, 20, 50):
            if k in res:
                print(m, "k=%d" % k, json.dumps(res[k]), flush=True)
    out["provenance"] = provenance()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()
