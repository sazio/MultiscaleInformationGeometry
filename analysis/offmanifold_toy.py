"""
Off-manifold toy (Appendix C, Table 1).

Stimulus x in R^2 with a correlated Gaussian prior: variance s_on^2 = 1 along the
"manifold" direction u_on and s_off^2 = 0.01 along the orthogonal "off-manifold"
direction u_off (u_on at 30 degrees). n = 10 linear-Gaussian neurons, r = A x +
noise, noise N(0, sigma^2 I), sigma^2 = 0.5, A = 2 a_on u_on^T + 2 beta a_off u_off^T
with fixed random unit-norm weight vectors a_on, a_off (orthogonalised). beta sets
how strongly the neurons encode the off-manifold direction. For each beta we
compute, exactly:
  - KL divergence between the response distributions p(r) at beta and at beta = 0;
  - the Fisher information J = A^T A / sigma^2 and the multi-scale metric
    G = int_0^inf J_t dt, with J_t the Fisher information of
    p_t(r|x) = N(A M_t x, A C_t A^T + sigma^2 I), M_t = S (S + t I)^-1,
    C_t = S - M_t S (numerical quadrature over log t); both are constant in x;
  - for J and G: angle of the leading eigenvector from the manifold direction, and
    the relative change of the matrix from its beta = 0 value (Frobenius norm,
    after normalising each matrix by its trace);
  - I(R;X) = 1/2 log|I + S J| (checked against 1/2 Tr G).

At beta = 1 the two eigenvalues of J are equal, so its leading angle is arbitrary
(Table 1 leaves that cell empty). Runs in seconds on a CPU.

Usage:
    python analysis/offmanifold_toy.py --out runs/analysis/offmanifold_toy.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec

BETAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


def run() -> list[dict]:
    rng = np.random.default_rng(1)
    th = np.deg2rad(30)
    u_on = np.array([np.cos(th), np.sin(th)])
    u_off = np.array([-np.sin(th), np.cos(th)])
    S = np.outer(u_on, u_on) * 1.0 + np.outer(u_off, u_off) * 0.01
    s2, nN = 0.5, 10
    a_on = rng.standard_normal(nN)
    a_on /= np.linalg.norm(a_on)
    a_off = rng.standard_normal(nN)
    a_off -= (a_off @ a_on) * a_on
    a_off /= np.linalg.norm(a_off)

    def G_of(A):
        def f(u):
            t = np.exp(u)
            M = S @ np.linalg.inv(S + t * np.eye(2))
            C = S - M @ S
            Cr = A @ C @ A.T + s2 * np.eye(nN)
            return t * (M.T @ A.T @ np.linalg.solve(Cr, A @ M))
        return quad_vec(f, -30, 30, epsabs=1e-12, epsrel=1e-10)[0]

    def kl(C1, C0):   # KL(N(0,C1) || N(0,C0))
        return 0.5 * (np.trace(np.linalg.solve(C0, C1)) - nN
                      + np.log(np.linalg.det(C0) / np.linalg.det(C1)))

    def lead_angle(M):
        w, V = np.linalg.eigh(M)
        v = V[:, -1]
        return float(np.rad2deg(np.arccos(min(1.0, abs(v @ u_on)))))

    def A_of(b):
        return np.outer(a_on, u_on) * 2.0 + b * np.outer(a_off, u_off) * 2.0

    def nrm(M):
        return M / np.trace(M)

    A0 = A_of(0.0)
    C0 = A0 @ S @ A0.T + s2 * np.eye(nN)
    J0 = A0.T @ A0 / s2
    G0 = G_of(A0)
    rows = []
    for b in BETAS:
        A = A_of(b)
        C = A @ S @ A.T + s2 * np.eye(nN)
        J = A.T @ A / s2
        G = G_of(A)
        I = 0.5 * np.log(np.linalg.det(np.eye(2) + S @ J))
        rows.append(dict(
            beta=b, KL_responses_vs_beta0=float(kl(C, C0)),
            offmanifold_response_variance_frac=float(b ** 2 * 4 * 0.01 / np.trace(A @ S @ A.T)),
            J_lead_angle_deg=lead_angle(J), G_lead_angle_deg=lead_angle(G),
            J_rel_change=float(np.linalg.norm(nrm(J) - nrm(J0)) / np.linalg.norm(nrm(J0))),
            G_rel_change=float(np.linalg.norm(nrm(G) - nrm(G0)) / np.linalg.norm(nrm(G0))),
            J_off_over_on=float(u_off @ J @ u_off / (u_on @ J @ u_on)),
            G_off_over_on=float(u_off @ G @ u_off / (u_on @ G @ u_on)),
            I_nats=float(I), half_trG=float(0.5 * np.trace(G))))
    return rows


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", default="runs/analysis/offmanifold_toy.json")
    args = p.parse_args(argv)

    rows = run()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(setup="see the module docstring", rows=rows), f, indent=1)
    print(f"{'beta':>5} {'KL(p(r))':>9} {'offvar%':>8} {'J angle':>8} {'G angle':>8} {'J chg':>7} "
          f"{'G chg':>7} {'J off/on':>9} {'G off/on':>9} {'I':>7} {'TrG/2':>7}")
    for r in rows:
        print(f"{r['beta']:5.1f} {r['KL_responses_vs_beta0']:9.4f} "
              f"{100 * r['offmanifold_response_variance_frac']:8.2f} {r['J_lead_angle_deg']:8.1f} "
              f"{r['G_lead_angle_deg']:8.1f} {r['J_rel_change']:7.3f} {r['G_rel_change']:7.3f} "
              f"{r['J_off_over_on']:9.3f} {r['G_off_over_on']:9.3f} {r['I_nats']:7.4f} "
              f"{r['half_trG']:7.4f}")
    return rows


if __name__ == "__main__":
    main()
