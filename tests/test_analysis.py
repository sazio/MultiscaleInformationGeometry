"""
Tests for the supporting analyses in analysis/.

calibrate_sigma: per-trial residuals of ±a_c (alternating over an even
number of trials) have standard deviation exactly a_c, so the expected median
is known without re-implementing the computation.

estimator_variance (UC form): with the affine mock score network of
test_estimator.py and a constant encoder μ_0, the UC row is
√(w_j/N_MC)·(σ A ξ − b) where b = ε(x_t,t,0) − (A μ_0 + h) is the mismatch of
the null-token prediction. Hence E[D_UCᵀ D_UC] = W (σ² A Aᵀ + b bᵀ),
W = Σ_j w_j. With b = 0 this equals the CC expectation σ² W A Aᵀ (the scaling
derived in the module docstring); with b ⟂ range(A) an extra eigenvalue W‖b‖²
appears along b.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import estimator_cfg, linear_beta_alphas_cumprod, write_fake_tvsd
from src.data_loaders import AREA_CHANNELS, load_tvsd

pytestmark = pytest.mark.filterwarnings("ignore:`product` is deprecated:DeprecationWarning")


# ── calibrate_sigma ────────────────────────────────────────────────────────
def test_calibrate_sigma_recovers_planted_trial_std(tmp_path):
    from analysis.calibrate_sigma import main

    n_trials, n_test = 6, 4
    write_fake_tvsd(tmp_path, n_test=n_test, n_trials=n_trials)
    rng = np.random.default_rng(3)
    a = rng.uniform(0.1, 0.5, 1024)                       # per-channel residual std
    mean = rng.standard_normal((n_test, 1024))
    sign = np.where(np.arange(n_trials) % 2 == 0, 1.0, -1.0)[:, None, None]
    reps = mean[None] + sign * a[None, None, :]
    with h5py.File(tmp_path / "monkeyN_THINGS_normMUA.mat", "r+") as f:
        del f["test_MUA_reps"], f["test_MUA"]
        f["test_MUA_reps"] = reps
        f["test_MUA"] = mean

    out = main(["--tvsd-dir", str(tmp_path), "--n-neurons", "7",
                "--out", str(tmp_path / "sigma.json")])
    saved = json.loads((tmp_path / "sigma.json").read_text())
    assert saved["areas"].keys() == out["areas"].keys() == {"V1", "V4"}
    assert out["max_abs_trial_mean_minus_test_MUA"] < 1e-12
    for area, (lo, hi) in AREA_CHANNELS.items():
        top = load_tvsd(area, n_neurons=7, tvsd_dir=tmp_path)["neuron_indices"] + lo
        r = out["areas"][area]
        assert r["top_channel_indices_global"] == top.tolist()
        assert r["sigma_median_top_channels"] == pytest.approx(np.median(a[top]), rel=1e-12)
        assert r["sigma_median_all_area_channels"] == pytest.approx(
            np.median(a[lo:hi + 1]), rel=1e-12)
    assert "git_commit" in saved["provenance"]


# ── estimator_variance ─────────────────────────────────────────────────────
H = W = 8
P = H * W
N_NEURONS = 3
SIGMA = 0.3
MU0 = torch.tensor([0.5, -1.0, 2.0])


class MismatchUNet(nn.Module):
    """ε(r) = A r + h(x_t) for r ≠ 0; null branch A μ_0 + h(x_t) + b."""

    def __init__(self, A, b):
        super().__init__()
        self.register_buffer("A", A.float())
        self.register_buffer("b", b.float().view(1, 1, H, W))
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x_t, t, r):
        h = 5.0 * torch.sin(3.0 * x_t)
        cond = (r @ self.A.T).view_as(x_t) + h
        null = (MU0 @ self.A.T).view(1, 1, H, W) + h + self.b
        return torch.where((r == 0).all(dim=1).view(-1, 1, 1, 1), null, cond)


class ConstantEncoder(nn.Module):
    n_neurons = N_NEURONS

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return MU0.expand(x.shape[0], -1).clone()


def _problem(beta: float):
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((P, N_NEURONS + 1)))
    U, bdir = Q[:, :N_NEURONS], Q[:, N_NEURONS]
    R, _ = np.linalg.qr(rng.standard_normal((N_NEURONS, N_NEURONS)))
    s = np.array([3.0, 2.0, 1.0])
    A = torch.from_numpy(U @ np.diag(s) @ R.T)
    return A, torch.from_numpy(beta * bdir), torch.from_numpy(U), torch.from_numpy(bdir), s


def _grid_W(start, stop, step):
    ab = linear_beta_alphas_cumprod()[list(range(start, stop + 1, step))]
    lt = torch.log((1 - ab) / ab)
    return float(lt[-1] - lt[0])


def test_uc_without_mismatch_matches_cc_expectation():
    pytest.importorskip("diffusers")
    from analysis.estimator_variance import unconditional_conditional_estimate

    A, b, U, _, s = _problem(beta=0.0)
    cfg = estimator_cfg(400, 100, 100, 900, 100, SIGMA, top_k=4)
    out = unconditional_conditional_estimate(
        torch.zeros(1, 1, H, W), ConstantEncoder(), MismatchUNet(A, b),
        linear_beta_alphas_cumprod().float(), cfg, torch.device("cpu"),
        generator=torch.Generator().manual_seed(0))
    expected = SIGMA ** 2 * _grid_W(100, 900, 100) * s ** 2
    np.testing.assert_allclose(out.eigenvalues[:3].double().numpy(), expected, rtol=0.10)
    assert float(out.eigenvalues[3]) < 1e-5 * float(out.eigenvalues[0])
    assert float(((out.eigenvectors[:3].double() @ U) ** 2).sum() / 3) == pytest.approx(1, abs=1e-5)


def test_uc_mismatch_adds_W_b2_along_b():
    pytest.importorskip("diffusers")
    from analysis.estimator_variance import unconditional_conditional_estimate

    beta = 4.0
    A, b, _, bdir, s = _problem(beta=beta)
    cfg = estimator_cfg(400, 100, 100, 900, 100, SIGMA, top_k=4)
    out = unconditional_conditional_estimate(
        torch.zeros(1, 1, H, W), ConstantEncoder(), MismatchUNet(A, b),
        linear_beta_alphas_cumprod().float(), cfg, torch.device("cpu"),
        generator=torch.Generator().manual_seed(1))
    Wsum = _grid_W(100, 900, 100)
    assert float(out.eigenvalues[0]) == pytest.approx(Wsum * beta ** 2, rel=0.05)
    assert abs(float(out.eigenvectors[0].double() @ bdir)) > 0.99
    np.testing.assert_allclose(out.eigenvalues[1:4].double().numpy(),
                               SIGMA ** 2 * Wsum * s ** 2, rtol=0.10)


def test_variance_study_ratio_reflects_mismatch():
    pytest.importorskip("diffusers")
    from analysis.estimator_variance import run_variance_study

    beta = 4.0
    A, b, *_ = _problem(beta=beta)
    cfg = estimator_cfg(100, 50, 200, 800, 200, SIGMA, top_k=5)
    res = run_variance_study(torch.zeros(2, 1, H, W), ConstantEncoder(), MismatchUNet(A, b),
                             linear_beta_alphas_cumprod().float(), cfg, torch.device("cpu"),
                             n_runs=3, top_k=3, n_mc_uc=150, log=lambda *a: None)
    assert res["n_mc_cc"] == 100 and res["n_mc_uc"] == 150
    assert res["unet_passes_per_timestep"] == {"CC": 300, "UC": 300}
    assert len(res["per_image"]) == 2
    expected_ratio = beta ** 2 / (SIGMA ** 2 * 9.0)          # W‖b‖² / (σ² W s_1²)
    for p in res["per_image"]:
        assert np.array(p["eigenvalues_CC"]).shape == (3, 3)
        assert p["ratio_top_eigenvalue_UC_over_CC"] == pytest.approx(expected_ratio, rel=0.2)
    json.dumps(res)                                           # serialisable


def test_matched_pass_count():
    from analysis.estimator_variance import uc_samples_for_matched_passes

    assert uc_samples_for_matched_passes(200, 25) == 300      # 3·200 = 2·300
    assert uc_samples_for_matched_passes(100, 50) == 150
