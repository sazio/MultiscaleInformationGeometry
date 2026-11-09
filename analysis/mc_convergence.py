"""
Monte Carlo convergence of the G(x) estimator (Appendix G, Fig. 8): raw runs.

Algorithm 1 (src/estimator.py, configs/estimator.yaml) with the 200-epoch
diffusion models and the Inception encoders, V1 and V4. For each test image in
[--start, --end) and each number of Monte Carlo samples per scale N in
{25, 50, 100, 200, 400} (the paper uses 200), run R = 5 independent estimates
with different random seeds and store the top-50 eigenvalues and the top-10
eigenvectors. analysis/mc_convergence_summary.py aggregates the files. One run
of all ten images in one area costs about 4 A100-GPU-hours; images already
written are skipped, so the job can be split or resumed.

Usage:
    python analysis/mc_convergence.py --area V1 --start 0 --end 10 \\
        --runs-dir runs --out-dir runs/analysis/mc/V1 --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loaders import CANONICAL_TRANSFORM, TinyImageNetDataset  # noqa: E402
from src.diffusion_model import build_diffusion_model, build_scheduler  # noqa: E402
from src.encoders import build_encoder  # noqa: E402
from src.estimator import conditional_pair_estimate  # noqa: E402
from src.utils import load_config, make_generator  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--area", required=True, choices=["V1", "V4"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--Ns", default="25,50,100,200,400")
    p.add_argument("--runs-dir", default="runs",
                   help="root with encoders/inception/<area>/ and diffusion/inception/<area>/")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-dir", default=None, help="override NEURIPS_DATA_DIR")
    a = p.parse_args(argv)

    if a.data_dir:
        os.environ["NEURIPS_DATA_DIR"] = a.data_dir
    dev = torch.device(a.device)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "estimator.yaml")
    diff_cfg = load_config(ROOT / "configs" / "diffusion.yaml")
    enc_cfg = load_config(ROOT / "configs" / "inception.yaml")
    runs = Path(a.runs_dir)
    encoder = build_encoder(enc_cfg).to(dev)
    encoder.load_state_dict(torch.load(runs / f"encoders/inception/{a.area}/best_model.pt",
                                       map_location=dev, weights_only=False)["model_state_dict"])
    encoder.eval()
    unet = build_diffusion_model(diff_cfg).to(dev)
    unet.load_state_dict(torch.load(runs / f"diffusion/inception/{a.area}/final_model.pt",
                                    map_location=dev, weights_only=False)["model_state_dict"])
    unet.eval()
    ac = build_scheduler(diff_cfg).alphas_cumprod.to(dev)
    data = TinyImageNetDataset(split="val", transform=CANONICAL_TRANSFORM)
    Ns = [int(n) for n in a.Ns.split(",")]
    for i in range(a.start, a.end):
        f = out / f"img{i:04d}.npz"
        if f.exists():
            continue
        x = data[i].unsqueeze(0).to(dev)
        L = np.zeros((len(Ns), a.reps, 50), np.float32)
        V = np.zeros((len(Ns), a.reps, 10, 4096), np.float32)
        t0 = time.time()
        for ni, n in enumerate(Ns):
            c = copy.deepcopy(cfg)
            c.monte_carlo.n_samples = n
            for r in range(a.reps):
                g = conditional_pair_estimate(
                    image=x, encoder=encoder, unet=unet, alphas_cumprod=ac, cfg=c, device=dev,
                    generator=make_generator(10_000_000 + 100_000 * i + 1000 * n + r, dev))
                L[ni, r] = g.eigenvalues.cpu().numpy()[:50]
                V[ni, r] = g.eigenvectors.cpu().numpy()[:10]
        np.savez(f, Ns=np.array(Ns), eigenvalues=L, eigenvectors=V)
        print(f"{a.area} img {i} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
