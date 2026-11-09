"""
Summary of the Monte Carlo convergence runs (Appendix G, Fig. 8).

Reads the files of analysis/mc_convergence.py. Per area and N, over images:
coefficient of variation of the leading eigenvalue across the runs; mean pairwise
top-5 subspace alignment (mean squared cosine, Eq. 42) across runs; relative
spread of the top-50 trace. Also the slope of log CV against log N, over all N
(cv_slope) and between 25 and 200 samples, the range quoted in Appendix G
(cv_slope_25_200). figures/fig_mc.py draws Fig. 8 from the output.

Usage:
    python analysis/mc_convergence_summary.py --mc-dir runs/analysis/mc \\
        --out runs/analysis/mc_convergence_summary.json
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--mc-dir", default="runs/analysis/mc", help="folder with <area>/img*.npz")
    p.add_argument("--out", default="runs/analysis/mc_convergence_summary.json")
    args = p.parse_args(argv)

    out = {}
    for area in ("V1", "V4"):
        fs = sorted(glob.glob(f"{args.mc_dir}/{area}/img*.npz"))
        if not fs:
            raise FileNotFoundError(f"no files in {args.mc_dir}/{area}")
        D = [np.load(f) for f in fs]
        Ns = list(D[0]["Ns"])
        res = {}
        for ni, n in enumerate(Ns):
            cv, al, tr = [], [], []
            for d in D:
                L, V = d["eigenvalues"][ni], d["eigenvectors"][ni].astype(np.float64)
                cv.append(L[:, 0].std(ddof=1) / L[:, 0].mean())
                tr.append(L.sum(1).std(ddof=1) / L.sum(1).mean())
                al.append(np.mean([((V[a, :5] @ V[b, :5].T) ** 2).sum() / 5
                                   for a, b in itertools.combinations(range(V.shape[0]), 2)]))
            res[int(n)] = dict(lam1_cv_mean=float(np.mean(cv)),
                               lam1_cv_sem=float(np.std(cv, ddof=1) / np.sqrt(len(cv))),
                               top5_align_mean=float(np.mean(al)),
                               top5_align_sem=float(np.std(al, ddof=1) / np.sqrt(len(al))),
                               trace50_cv_mean=float(np.mean(tr)), n_images=len(D))
        out[area] = res
        print(area, "N: lam1 CV | top-5 alignment across runs | trace CV")
        for n, r in res.items():
            print(f"  {n:4d}: {r['lam1_cv_mean']:.3f} +- {r['lam1_cv_sem']:.3f} | "
                  f"{r['top5_align_mean']:.3f} +- {r['top5_align_sem']:.3f} | {r['trace50_cv_mean']:.3f}")
        Nn = np.array(list(res))
        cvs = np.array([res[n]["lam1_cv_mean"] for n in res])
        s = np.polyfit(np.log(Nn), np.log(cvs), 1)[0]
        keep = (Nn >= 25) & (Nn <= 200)
        s200 = np.polyfit(np.log(Nn[keep]), np.log(cvs[keep]), 1)[0]
        print(f"  slope of log CV vs log N: {s:.2f} over all N, {s200:.2f} from 25 to 200 "
              "(1/sqrt(N) gives -0.5)")
        out[area]["cv_slope"] = float(s)
        out[area]["cv_slope_25_200"] = float(s200)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()
