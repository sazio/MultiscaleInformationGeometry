"""
Periodic stimuli (Appendix E, Table 3): I(R;X) = 1/2 E[Tr G(x)] for Poisson
populations encoding a circular stimulus.

  x uniform on [-pi, pi); f_i(x) = a_i exp[(cos(x - theta_i) - 1)/w_i] + 0.01;
  r_i ~ Poisson(f_i(x)), independent. Coarse-graining: Brownian motion on the circle
  (wrapped Gaussian heat kernel phi_t); G(x) = int_0^inf [J_t * phi_t](x) dt.
  I(R;X) = H(X) - E_r[H(X|r)], 2048-point grid, Monte Carlo over r.
  1/2 E[Tr G] = 1/2 int_0^inf E_x[J_t(x)] dt (circular convolution leaves the
  average over a uniform x unchanged), with
  E_x[J_t] = E_{r ~ p(r)} [ E_{x ~ p_t(x|r)} (d/dx log p_t(r|x))^2 ],
  p_t(r|x) = [L_r * phi_t](x), L_r(x') = p(r|x').
  The convolution and its derivative are exact Fourier sums (coefficient
  exp(-n^2 t/2), derivative i n), so the score uses no Tweedie identity. Scales: 39
  log-spaced t in [1e-5, 40], trapezoid in log t (integrand t E[J_t]); the t < 1e-5
  part is added as 1e-5 E[J_0], J_0 = sum_i f_i'^2 / f_i. Error: s.e.m. over response
  samples (the same samples at every scale, integrated per sample). The two columns
  use independent response samples.

Amplitudes 3.8-34.2 and widths 0.145-0.58 are evenly spaced (linear) over the 10
neurons, assigned to neurons in order of preferred angle.

With the default 1e5 sampled responses this takes tens of minutes on a CPU.

Usage:
    python analysis/periodic_toy.py --out runs/analysis/periodic_toy.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

NG = 2048
xg = -np.pi + 2 * np.pi * np.arange(NG) / NG
n = np.fft.fftfreq(NG, 1 / NG)
BATCH = 1000
TS = np.logspace(-5, np.log10(40), 39)
HX = np.log(2 * np.pi)
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def tuning(a, w, th):
    a, w, th = map(np.asarray, (a, w, th))
    F = a[:, None] * np.exp((np.cos(xg[None] - th[:, None]) - 1) / w[:, None]) + 0.01
    dF = -a[:, None] * np.sin(xg[None] - th[:, None]) / w[:, None] \
        * np.exp((np.cos(xg[None] - th[:, None]) - 1) / w[:, None])
    return F, dF


def configs() -> dict:
    def ev(N):
        return -np.pi + 2 * np.pi * np.arange(N) / N
    C = {}
    for N in (1, 4, 10, 20):
        C[f"{N} neuron{'s' if N > 1 else ''}"] = (np.full(N, 19.), np.full(N, 0.29), ev(N))
    th = ev(10)
    C["10 neurons, uneven spacing"] = (np.full(10, 19.), np.full(10, 0.29), th + 0.5 * np.sin(2 * th))
    C["10 neurons, uneven amplitude"] = (np.linspace(3.8, 34.2, 10), np.full(10, 0.29), th)
    C["10 neurons, uneven width"] = (np.full(10, 19.), np.linspace(0.145, 0.58, 10), th)
    C["4 neurons, broad tuning, low rate"] = (np.full(4, 5.), np.full(4, 1.5), ev(4))
    return C


def run(n_samples: int = 100000, names: list[str] | None = None) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for name, (a, w, th) in configs().items():
        if names is not None and name not in names:
            continue
        t0 = time.time()
        F, dF = tuning(a, w, th)
        logF = np.log(F)
        sumF = F.sum(0)
        J0 = float(((dF ** 2) / F).sum(0).mean())
        Hpost, integ = [], []
        for _ in range(0, n_samples, BATCH):
            x0 = rng.integers(0, NG, BATCH)                 # x uniform on the grid
            r = rng.poisson(F[:, x0].T)                     # responses for I(R;X)
            LL = r @ logF - sumF[None]
            LL -= LL.max(1, keepdims=True)
            L = np.exp(LL)                                  # likelihood on the grid (scaled)
            post = L / L.sum(1, keepdims=True)
            Hpost.append(-(post * np.log(post + 1e-300)).sum(1) + np.log(2 * np.pi / NG))
            # independent responses for 1/2<Tr G>: with shared responses the two columns
            # agree sample by sample (de Bruijn's identity for the heat flow of each
            # posterior), which would test only the quadrature
            x1 = rng.integers(0, NG, BATCH)
            r = rng.poisson(F[:, x1].T)
            LL = r @ logF - sumF[None]
            LL -= LL.max(1, keepdims=True)
            L = np.exp(LL)
            Lh = np.fft.fft(L, axis=1)
            vals = np.zeros((BATCH, len(TS)))
            for k, t in enumerate(TS):
                K = np.exp(-n ** 2 * t / 2)
                P = np.fft.ifft(Lh * K, axis=1).real
                dP = np.fft.ifft(Lh * K * 1j * n, axis=1).real
                P = np.maximum(P, 1e-300)
                pt = P / P.sum(1, keepdims=True)
                ok = pt > 1e-13                             # skip negligible posterior mass
                vals[:, k] = np.where(ok, pt * (dP / P) ** 2, 0.0).sum(1)   # E_{p_t(x|r)} score^2
            lt = np.log(TS)
            y = vals * TS[None]                             # integrand in log t
            integ.append(trapezoid(y, lt, axis=1) + 1e-5 * J0)
        Hpost = np.concatenate(Hpost)
        integ = np.concatenate(integ)
        I = HX - Hpost.mean()
        I_se = Hpost.std(ddof=1) / np.sqrt(len(Hpost))
        G = 0.5 * integ.mean()
        G_se = 0.5 * integ.std(ddof=1) / np.sqrt(len(integ))
        out[name] = dict(I=float(I), I_sem=float(I_se), halfTrG=float(G), halfTrG_sem=float(G_se),
                         rel_diff_pct=float(100 * (G - I) / I))
        print(f"{name:36s} I = {I:.4f} +- {I_se:.4f}   1/2<TrG> = {G:.4f} +- {G_se:.4f}   "
              f"rel {100 * (G - I) / I:+.2f}%   ({time.time() - t0:.0f} s)", flush=True)
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--n-samples", type=int, default=100000, help="sampled responses per row")
    p.add_argument("--out", default="runs/analysis/periodic_toy.json")
    args = p.parse_args(argv)

    out = run(args.n_samples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()
