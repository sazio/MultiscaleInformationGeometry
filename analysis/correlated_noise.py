"""
Correlated neural noise (Appendix D, Table 2 and Fig. 6).

One linear-Gaussian population, two ways of introducing noise correlations:

  After encoding:  r = A x + eta, eta ~ N(0, Sigma) with Sigma prescribed
                   non-diagonal: extra noise variance along the dominant signal
                   direction u (information-limiting), or variance removed along
                   it (information-enhancing).
  Before encoding: r = A(x + eta_in) + eta_out, eta_in ~ N(0, f I). The injected
                   noise is independent; the encoder propagates it into structured
                   output correlations, Sigma = f A A^T + sigma^2 I.

For each configuration:
  - mean pairwise noise correlation;
  - I(R;X) in closed form for the correlated Gaussian channel;
  - 1/2 <Tr G(x)> from the conditional-pair form of Eq. 17 with a correlated
    response sampler r = A x + L xi, Sigma = L L^T, using exact Gaussian posterior
    means in place of a trained diffusion model;
  - the relative difference of the two;
  - the effect on I(R;X) relative to the trial-shuffled control diag(Sigma).

Pure NumPy, about a minute on a CPU. figures/fig_corrnoise.py draws Fig. 6 from
the output.

Usage:
    python analysis/correlated_noise.py --out runs/analysis/correlated_noise.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.linalg import cholesky, eigh, inv, slogdet

d, n = 2, 6
Sigma_x = np.array([[1.0, 0.30], [0.30, 1.0]])
_A = np.random.default_rng(1).normal(size=(n, d))
A = _A + 0.6 * _A.mean(0, keepdims=True)      # shared component -> similar tuning
Id, In = np.eye(d), np.eye(n)
sig2 = 0.30                                   # baseline per-neuron noise variance

# dominant signal direction in response space
S_sig = A @ Sigma_x @ A.T
u = eigh(S_sig)[1][:, -1]
u = u / np.linalg.norm(u)


def mean_corr(S):
    """Mean pairwise noise correlation coefficient (off-diagonal)."""
    dg = np.sqrt(np.diag(S))
    C = S / np.outer(dg, dg)
    iu = np.triu_indices(n, 1)
    return C[iu].mean()


def MI(S):
    """Exact I(R;X) for the linear-Gaussian channel, nats."""
    return 0.5 * slogdet(Id + Sigma_x @ (A.T @ inv(S) @ A))[1]


def diffused_fisher(S, t):
    Pt = Sigma_x + t * Id
    M = A @ Sigma_x @ inv(Pt)
    return M.T @ inv(t * (A @ Sigma_x @ inv(Pt) @ A.T) + S) @ M


def cond_pair_halftrace(S, rng, nmc=120000, tmin=1e-5, tmax=1e3, nt=110):
    """
    1/2 <Tr G> from the conditional-pair estimator, correlated sampler.

    Returns (value, sem). Integration in t by trapezoid on a log grid, plus the
    analytic t->0 tail (< 0.001 nats).
    """
    tg = np.logspace(np.log10(tmin), np.log10(tmax), nt)
    P = Sigma_x
    Ln = cholesky(S + 1e-12 * In)
    half, var = np.zeros(nt), np.zeros(nt)
    x = rng.multivariate_normal(np.zeros(d), P, size=nmc)
    for k, t in enumerate(tg):
        Pt = P + t * Id
        Post = P @ inv(Pt)                        # E[x|x_t]
        Sig_x_xt = P - P @ inv(Pt) @ P
        Cov_c = np.block([[Pt, P @ A.T], [A @ P, A @ P @ A.T + S]])
        L = np.hstack([P, P @ A.T]) @ inv(Cov_c)  # E[x|x_t,r] = L @ [x_t; r]
        Lx = cholesky(Sig_x_xt + 1e-12 * Id)

        x_t = x + np.sqrt(t) * rng.standard_normal((nmc, d))
        xhat = x_t @ Post.T

        def draw_r():                             # r ~ p(r|x_t): resample x, then r
            xp = xhat + rng.standard_normal((nmc, d)) @ Lx.T
            return xp @ A.T + rng.standard_normal((nmc, n)) @ Ln.T   # correlated sampler

        dd = np.hstack([x_t, draw_r()]) @ L.T - np.hstack([x_t, draw_r()]) @ L.T
        # per-sample 1/2 Tr of the rank-1 outer product, scaled by 1/(2 t^2)
        per = (dd ** 2).sum(1) / (4.0 * t * t)
        half[k] = per.mean()
        var[k] = per.var(ddof=1) / nmc
    # integrate on the log-t axis: int f(t) dt = int f(t) t d(log t)
    lg = np.log(tg)
    w = np.zeros(nt)
    w[0] = 0.5 * (lg[1] - lg[0])
    w[-1] = 0.5 * (lg[-1] - lg[-2])
    w[1:-1] = 0.5 * (lg[2:] - lg[:-2])
    w = w * tg
    val = float(w @ half)
    sem = float(np.sqrt((w ** 2) @ var))
    tail = 0.5 * np.trace(diffused_fisher(S, 0.0)) * tmin      # analytic t->0
    return val + tail, sem


def analytic_halftrace(S, tmin=1e-8, tmax=1e4, nt=4000):
    tg = np.logspace(np.log10(tmin), np.log10(tmax), nt)
    ht = np.array([0.5 * np.trace(diffused_fisher(S, t)) for t in tg])
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trapezoid(ht, tg) + 0.5 * np.trace(diffused_fisher(S, 0.0)) * tmin


def post(c):
    """Extra (c>0) or reduced (c<0) noise variance along the signal direction u."""
    return sig2 * (In + c * np.outer(u, u))


def pre(f):
    """Independent upstream noise propagated through the encoder."""
    return sig2 * In + f * (A @ A.T)


CONFIGS = [
    ("Independent (control)", sig2 * In),
    ("After encoding, along signal (c=1)", post(1.0)),
    ("After encoding, along signal (c=4)", post(4.0)),
    ("After encoding, reduced along signal", post(-0.75)),
    ("Before encoding, f=0.1", pre(0.10)),
    ("Before encoding, f=0.3", pre(0.30)),
    ("Before encoding, f=1.0", pre(1.00)),
]


def run(nmc: int = 120000) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for name, S in CONFIGS:
        I_ex = MI(S)
        I_sh = MI(np.diag(np.diag(S)))
        an = analytic_halftrace(S)
        est, sem = cond_pair_halftrace(S, rng, nmc=nmc)
        rows.append(dict(name=name, mean_corr=float(mean_corr(S)), I_exact=float(I_ex),
                         half_trace_analytic=float(an), half_trace_estimate=float(est),
                         sem=float(sem), rel_diff_pct=float(100.0 * (est - I_ex) / I_ex),
                         delta_I_vs_shuffled_pct=float(100.0 * (I_ex / I_sh - 1.0))))
    return rows


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--nmc", type=int, default=120000, help="Monte Carlo samples per scale")
    p.add_argument("--out", default="runs/analysis/correlated_noise.json")
    args = p.parse_args(argv)

    rows = run(args.nmc)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(nmc=args.nmc, seed=0, rows=rows), f, indent=1)
    print(f"{'configuration':38s} {'corr':>7s} {'I exact':>9s} {'1/2TrG an':>10s} "
          f"{'1/2TrG est':>11s} {'sem':>7s} {'rel%':>7s} {'dI vs shuf%':>12s}")
    for r in rows:
        print(f"{r['name']:38s} {r['mean_corr']:7.3f} {r['I_exact']:9.4f} "
              f"{r['half_trace_analytic']:10.4f} {r['half_trace_estimate']:11.4f} {r['sem']:7.4f} "
              f"{r['rel_diff_pct']:+7.2f} {r['delta_I_vs_shuffled_pct']:+12.1f}")
    return rows


if __name__ == "__main__":
    main()
