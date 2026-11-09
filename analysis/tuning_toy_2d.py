"""
Toy model of Fig. 2: two neurons, a two-dimensional stimulus, uniform or ridge-shaped prior.

  x on the square [-5, 5]^2; r_i = f_i(x_i) + sigma * noise, independent standard Gaussian noise.
  f_1(x_1) = A / (1 + exp(-k x_1)), f_2(x_2) = A / (1 + exp(k x_2)).
  Priors: uniform, or p(x) proportional to exp(-(x_2 - a x_1^2 - b)^2 / (2 w^2)).

  J_t(y) is the Fisher information of p_t(r | y) = int p(r | x) p(x | y) dx with respect to y,
  y = x + sqrt(t) * noise. All integrals are on grids. G(x) = int_0^inf [phi_t * J_t](x) dt,
  trapezoid in log t; the part below the smallest t is t_min [phi * J](x) at that t.
  Maps are written at t = 0, 1.5 and 4 and for G, on an 8 x 8 grid of stimuli.

Check: I(R;X) = 1/2 E_x[Tr G(x)], with I(R;X) = H(R) - H(R|X) computed directly from p(r).

Under a minute in total on one GPU.

Usage:
    python analysis/tuning_toy_2d.py --out runs/analysis/tuning_toy_2d.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

A, K, SIGMA = 10.0, 1.5, 1.0
RIDGE_A, RIDGE_B, RIDGE_W = 0.5, -4.0, 0.45
HALF = 5.0
SHOW = (np.arange(8) - 3.5) * 1.2


def log_prior(x1, x2, prior):
    if prior == "uniform":
        return torch.zeros_like(x1)
    return -(x2 - RIDGE_A * x1 ** 2 - RIDGE_B) ** 2 / (2 * RIDGE_W ** 2)


def fisher_diffused(t, x, lp, l1, l2, r, ny, device, batch=256):
    """J_t [ny^2, 2, 2] on a grid of y, and the density of y on that grid."""
    x1, x2 = torch.meshgrid(x, x, indexing="ij")
    hw = HALF + 5 * np.sqrt(t)
    y = torch.linspace(-hw, hw, ny, device=device, dtype=x.dtype)
    y1, y2 = torch.meshgrid(y, y, indexing="ij")
    pts = torch.stack([y1.ravel(), y2.ravel()], 1)
    prior = torch.exp(lp - lp.max())
    prior = prior / prior.sum()
    dr2 = (r[1] - r[0]) ** 2
    j = torch.zeros((len(pts), 2, 2), device=device, dtype=x.dtype)
    density = torch.zeros(len(pts), device=device, dtype=x.dtype)
    for i in range(0, len(pts), batch):
        q = pts[i:i + batch]
        d2 = (x1[None] - q[:, 0, None, None]) ** 2 + (x2[None] - q[:, 1, None, None]) ** 2
        density[i:i + batch] = (prior[None] * torch.exp(-d2 / (2 * t))).sum((1, 2)) / (2 * np.pi * t)
        log_w = lp[None] - d2 / (2 * t)
        post = torch.exp(log_w - log_w.amax((1, 2), keepdim=True))
        post = post / post.sum((1, 2), keepdim=True)
        m1 = (post * x1[None]).sum((1, 2), keepdim=True)
        m2 = (post * x2[None]).sum((1, 2), keepdim=True)
        p = l1.T @ post @ l2
        d1 = l1.T @ (post * (x1[None] - m1) / t) @ l2
        d2_ = l1.T @ (post * (x2[None] - m2) / t) @ l2
        ip = 1 / (p + 1e-300)
        j[i:i + batch, 0, 0] = (d1 ** 2 * ip).sum((1, 2)) * dr2
        j[i:i + batch, 1, 1] = (d2_ ** 2 * ip).sum((1, 2)) * dr2
        j[i:i + batch, 0, 1] = j[i:i + batch, 1, 0] = (d1 * d2_ * ip).sum((1, 2)) * dr2
    return pts, j, density, (y[1] - y[0]) ** 2


def smooth(pts, j, t, device, dtype):
    grid = np.array(np.meshgrid(SHOW, SHOW, indexing="ij")).reshape(2, -1).T
    g = torch.tensor(grid, device=device, dtype=dtype)
    k = torch.exp(-((g[:, None] - pts[None]) ** 2).sum(2) / (2 * t))
    k = k / k.sum(1, keepdim=True)
    return (k[..., None, None] * j[None]).sum(1).cpu().numpy().reshape(8, 8, 2, 2)


def run_prior(prior, sigma=SIGMA, nx=201, ny=81, nr=121, per_decade=8, device="cpu") -> dict:
    dtype = torch.float64
    x = torch.linspace(-HALF, HALF, nx, device=device, dtype=dtype)
    r = torch.linspace(-4 * sigma, A + 4 * sigma, nr, device=device, dtype=dtype)
    norm = np.sqrt(2 * np.pi * sigma ** 2)
    f1, f2 = A / (1 + torch.exp(-K * x)), A / (1 + torch.exp(K * x))
    l1 = torch.exp(-(r[None] - f1[:, None]) ** 2 / (2 * sigma ** 2)) / norm
    l2 = torch.exp(-(r[None] - f2[:, None]) ** 2 / (2 * sigma ** 2)) / norm
    x1, x2 = torch.meshgrid(x, x, indexing="ij")
    lp = log_prior(x1, x2, prior)
    ts = 10 ** (-2.5 + np.arange(int(6.5 * per_decade) + 1) / per_decade)
    maps, trace = [], []
    for t in ts:
        pts, j, density, dy2 = fisher_diffused(t, x, lp, l1, l2, r, ny, device)
        maps.append(smooth(pts, j, t, device, dtype))
        w = density / density.sum()
        trace.append(float((w * (j[:, 0, 0] + j[:, 1, 1])).sum()))
    trace = np.array(trace)
    maps = np.array(maps)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    g = ts[0] * maps[0] + trapezoid(ts[:, None, None, None, None] * maps, np.log(ts), axis=0)
    half_mean = 0.5 * (ts[0] * trace[0] + trapezoid(ts * trace, np.log(ts)))
    at = {}
    for t in (1.5, 4.0):
        pts, j, _, _ = fisher_diffused(t, x, lp, l1, l2, r, ny, device)
        at[t] = smooth(pts, j, t, device, dtype)
    prob = torch.exp(lp - lp.max())
    prob = prob / prob.sum()
    pr = l1.T @ prob @ l2
    dr = float(r[1] - r[0])
    h_r = -float((pr * torch.log(pr + 1e-300)).sum()) * dr ** 2
    mi = h_r - np.log(2 * np.pi * np.e * sigma ** 2)
    slope = (A * K * np.exp(-K * SHOW) / (1 + np.exp(-K * SHOW)) ** 2) ** 2 / sigma ** 2
    j0 = np.zeros((8, 8, 2, 2))
    j0[..., 0, 0], j0[..., 1, 1] = slope[:, None], slope[None, :]
    density = prob.cpu().numpy() / float(x[1] - x[0]) ** 2
    return {f"{prior}_J0": j0, f"{prior}_J1p5": at[1.5], f"{prior}_J4": at[4.0], f"{prior}_G": g,
            f"{prior}_mi_direct": mi, f"{prior}_half_mean_trG": half_mean, f"{prior}_density": density}


def run(sigma=SIGMA, device="cpu", **grid) -> dict:
    out = {"show": SHOW}
    for prior in ("uniform", "ridge"):
        out.update(run_prior(prior, sigma, device=device, **grid))
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/analysis/tuning_toy_2d.npz")
    args = p.parse_args(argv)
    out = run(device=args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    for prior in ("uniform", "ridge"):
        print(f"{prior}: I(R;X) = {out[f'{prior}_mi_direct']:.4f} nats, "
              f"1/2 <Tr G> = {out[f'{prior}_half_mean_trG']:.4f} nats")
    return out


if __name__ == "__main__":
    main()
