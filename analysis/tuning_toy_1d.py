"""
Toy model of Fig. 1: one neuron, a scalar stimulus, two tuning curves with the same |f'(x)|.

  x uniform on [-L, L]; r = f(x) + sigma * noise, noise standard Gaussian.
  bell       f(x) = exp(-x^2 / (2 s^2))
  monotonic  f(x) = bell for x < 0, 2 - bell for x >= 0

  J(x) = f'(x)^2 / sigma^2. J_t(y) is the Fisher information of
  p_t(r | y) = int p(r | x) p(x | y) dx with respect to y, where y = x + sqrt(t) * noise.
  The score uses d/dy p(x | y) = p(x | y) (x - E[x | y]) / t. All integrals are on grids.
  G(x) = int_0^inf [phi_t * J_t](x) dt, trapezoid in log t; the t < t_min part is t_min J(x).

Panels D-E show the contribution of one scale step, [phi_t * J_t](x) dt, for t up to 10^2.5.
G and the check below integrate up to t = 1e5.

Check: I(R;X) = 1/2 E_x[G(x)], with I(R;X) = H(R) - H(R|X) computed directly from p(r).

With the default grids this takes under a minute on a CPU.

Usage:
    python analysis/tuning_toy_1d.py --out runs/analysis/tuning_toy_1d.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

S, SIGMA, L = 0.55, 0.175, 8.0
HEAT_T_MAX = 10 ** 2.5
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def tuning(x: np.ndarray, s: float = S) -> dict[str, np.ndarray]:
    bell = np.exp(-x ** 2 / (2 * s ** 2))
    return {"bell": bell, "monotonic": np.where(x < 0, bell, 2 - bell)}


def fisher_diffused(f, x, y, t, sigma, r):
    """J_t on the points y, for the uniform prior on the grid x."""
    log_w = -(y[:, None] - x[None]) ** 2 / (2 * t)
    post = np.exp(log_w - log_w.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    mean = post @ x
    lik = np.exp(-(r[:, None] - f[None]) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
    p = post @ lik.T
    dp = (post * (x[None] - mean[:, None]) / t) @ lik.T
    return np.sum(dp ** 2 / (p + 1e-300), 1) * (r[1] - r[0])


def mutual_information(f, sigma, r):
    lik = np.exp(-(r[:, None] - f[None]) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
    p = lik.mean(1)
    h_r = -np.sum(p * np.log(p + 1e-300)) * (r[1] - r[0])
    return h_r - 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)


def run(s=S, sigma=SIGMA, half_range=L, nx=1601, ny=801, nr=400, per_decade=15, n_show=241) -> dict:
    x = np.linspace(-half_range, half_range, nx)
    show = np.linspace(-3, 3, n_show)
    ts = 10 ** (-3 + np.arange(8 * per_decade + 1) / per_decade)
    d_log = np.log(ts[1] / ts[0])
    r = np.linspace(-6 * sigma, 2 + 6 * sigma, nr)
    bell = np.exp(-x ** 2 / (2 * s ** 2))
    j0 = (x / s ** 2 * bell) ** 2 / sigma ** 2
    out = {"t": ts, "x_show": show, "x_tuning": x, "params": np.array([s, sigma, half_range])}
    for name, f in tuning(x, s).items():
        on_show, on_prior = [], []
        for t in ts:
            hw = half_range + 7 * np.sqrt(t)
            y = np.linspace(-hw, hw, ny)
            jt = fisher_diffused(f, x, y, t, sigma, r)
            for grid, store in ((show, on_show), (x, on_prior)):
                k = np.exp(-(grid[:, None] - y[None]) ** 2 / (2 * t))
                store.append((k / k.sum(1, keepdims=True)) @ jt)
        on_show, on_prior = np.array(on_show), np.array(on_prior)
        tail = ts[0] * np.interp(show, x, j0)
        g = tail + trapezoid(ts[:, None] * on_show, np.log(ts), axis=0)
        g_prior = ts[0] * j0 + trapezoid(ts[:, None] * on_prior, np.log(ts), axis=0)
        out[f"{name}_f"] = f
        out[f"{name}_J"] = np.interp(show, x, j0)
        out[f"{name}_G"] = g
        out[f"{name}_contribution"] = (ts[:, None] * on_show * d_log)[ts <= HEAT_T_MAX * 1.001]
        out[f"{name}_mi_direct"] = mutual_information(f, sigma, r)
        out[f"{name}_half_mean_G"] = 0.5 * g_prior.mean()
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", default="runs/analysis/tuning_toy_1d.npz")
    args = p.parse_args(argv)
    out = run()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    for name in ("bell", "monotonic"):
        print(f"{name}: I(R;X) = {out[f'{name}_mi_direct']:.4f} nats, "
              f"1/2 <G> = {out[f'{name}_half_mean_G']:.4f} nats")
    return out


if __name__ == "__main__":
    main()
