"""
Tests for the eigenvector statistics, the alignment ordering test, the convergence
summary and the simulations of Appendices C-E, and smoke tests of the scripts.

Expected values: extent, frequency and component counts of images with a known
answer; the toy results are checked against their closed forms (for the off-manifold
toy, I(R;X) = 1/2 Tr G and J off/on = beta^2; for the correlated-noise toy, the
analytic half trace).
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

H = 64
P = H * H


# ── per-image tree ─────────────────────────────────────────────────────────
def _orthonormal_rows(k: int, g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(P, k, generator=g))
    return q.T.contiguous()


def _write_per_image(root, n_images: int = 6, k: int = 12, same_area_shared: bool = True):
    """Same-area models share their eigenvectors for G; for J every model has its own."""
    g = torch.Generator().manual_seed(0)
    for i in range(n_images):
        shared = {a: _orthonormal_rows(k, g) for a in ("V1", "V4")}
        for enc in ("inception", "resnet50"):
            for area in ("V1", "V4"):
                d = root / "per_image" / enc / area
                d.mkdir(parents=True, exist_ok=True)
                torch.save({"G_eigenvectors": shared[area] if same_area_shared else _orthonormal_rows(k, g),
                            "G_eigenvalues": torch.ones(k),
                            "J_eigenvectors": _orthonormal_rows(k, g),
                            "J_eigenvalues": torch.ones(k)}, d / f"img{i:04d}.pt")


# ── eigvec_statistics ──────────────────────────────────────────────────────
def test_extent_of_a_single_pixel_and_of_two_pixels():
    from analysis.eigvec_statistics import extent

    v = np.zeros((H, H))
    v[10, 20] = 1.0
    assert extent(v) == pytest.approx(0.0)
    v[10, 30] = -1.0                                 # equal energy, 10 px apart
    assert extent(v) == pytest.approx(5.0)


def test_frequency_of_a_plane_wave():
    from analysis.eigvec_statistics import sf

    v = np.cos(2 * np.pi * 7 * np.arange(H) / H)[None, :] * np.ones((H, 1))
    assert sf(v) == pytest.approx(7.0)


def test_components_counts_separate_blobs_and_ignores_sign():
    from analysis.eigvec_statistics import comps

    v = np.zeros((H, H))
    v[5:8, 5:8] = 1.0
    v[30:33, 30:33] = -1.0
    assert comps(v) == 2


def test_statistics_script_writes_values_and_recomputes_from_npz(tmp_path, capsys):
    from analysis.eigvec_statistics import main

    _write_per_image(tmp_path)
    out = main(["--runs-dir", str(tmp_path), "--out-dir", str(tmp_path / "a")])
    saved = np.load(tmp_path / "a" / "eigvec_statistics.npz")
    assert saved["G"].shape == (6, 2, 2, 10, 3)
    again = main(["--from-npz", str(tmp_path / "a" / "eigvec_statistics.npz"),
                  "--out-dir", str(tmp_path / "b")])
    assert again["G"]["extent_px"]["per_image"] == out["G"]["extent_px"]["per_image"]
    assert "git_commit" in json.loads((tmp_path / "b" / "eigvec_statistics.json").read_text())["provenance"]


# ── alignment_ordering ─────────────────────────────────────────────────────
def test_alignment_ordering_detects_shared_same_area_eigenvectors(tmp_path):
    from analysis.alignment_ordering import main

    _write_per_image(tmp_path)
    out = main(["--runs-dir", str(tmp_path), "--top-k", "10",
                "--out", str(tmp_path / "ordering.json")])
    assert out["n_images"] == 6
    assert out["G"][10]["n_same_gt_cross"] == 6      # same-area alignment is exactly 1 here
    assert out["G"][10]["median_diff"] == pytest.approx(1.0, abs=0.05)
    assert out["J"][10]["n_same_gt_cross"] + out["J"][10]["n_same_lt_cross"] == 6


def test_component_thresholds_script(tmp_path):
    from analysis.eigvec_component_thresholds import CFGS, main

    _write_per_image(tmp_path)
    out = main(["--runs-dir", str(tmp_path), "--out", str(tmp_path / "c.json")])
    assert set(out["G"]) == {c[0] for c in CFGS}
    assert out["G"]["8conn_0.2"]["V1_median"] <= out["G"]["4conn_0.2"]["V1_median"]


# ── mc_convergence_summary ─────────────────────────────────────────────────
def test_mc_summary_zero_spread_and_full_alignment(tmp_path):
    from analysis.mc_convergence_summary import main

    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((P, 10)))
    Ns = np.array([25, 50, 100, 200, 400])
    rng = np.random.default_rng(1)
    for area in ("V1", "V4"):
        d = tmp_path / "mc" / area
        d.mkdir(parents=True)
        for i in range(3):
            # eigenvalue spread falls as 1/sqrt(N) by construction; eigenvectors are identical
            L = np.ones((5, 5, 50), np.float32)
            noise = rng.standard_normal(5)
            noise = (noise - noise.mean()) / noise.std(ddof=1)
            for j, n in enumerate(Ns):
                L[j, :, 0] = 1.0 + noise * 0.1 / np.sqrt(n)
            V = np.broadcast_to(q.T[None, None], (5, 5, 10, P)).astype(np.float32)
            np.savez(d / f"img{i:04d}.npz", Ns=Ns, eigenvalues=L, eigenvectors=V)
    out = main(["--mc-dir", str(tmp_path / "mc"), "--out", str(tmp_path / "s.json")])
    assert out["V1"][200]["top5_align_mean"] == pytest.approx(1.0, abs=1e-5)
    assert out["V1"][200]["lam1_cv_mean"] == pytest.approx(0.1 / np.sqrt(200), rel=0.01)
    assert out["V4"]["cv_slope_25_200"] == pytest.approx(-0.5, abs=0.01)


# ── simulations ────────────────────────────────────────────────────────────
def test_offmanifold_toy_matches_closed_forms():
    from analysis.offmanifold_toy import run

    rows = run()
    assert [r["beta"] for r in rows] == [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    for r in rows:
        assert r["half_trG"] == pytest.approx(r["I_nats"], abs=1e-6)
        assert r["J_off_over_on"] == pytest.approx(r["beta"] ** 2, abs=1e-9)
        assert r["G_lead_angle_deg"] == pytest.approx(0.0, abs=1e-3)
    assert rows[0]["I_nats"] == pytest.approx(0.5 * np.log(1 + 8.0), abs=1e-6)  # 1 + s_on^2 4/s^2


def test_correlated_noise_closed_forms_and_analytic_half_trace():
    from analysis.correlated_noise import CONFIGS, MI, analytic_halftrace, mean_corr, post, sig2

    assert len(CONFIGS) == 7
    assert mean_corr(sig2 * np.eye(6)) == pytest.approx(0.0)
    for _, S in CONFIGS:
        assert analytic_halftrace(S) == pytest.approx(MI(S), abs=1e-3)
    assert MI(post(1.0)) < MI(sig2 * np.eye(6)) < MI(post(-0.75))


@pytest.mark.slow
def test_correlated_noise_estimator_recovers_information():
    from analysis.correlated_noise import MI, cond_pair_halftrace, sig2

    S = sig2 * np.eye(6)
    est, sem = cond_pair_halftrace(S, np.random.default_rng(0), nmc=20000)
    assert abs(est - MI(S)) < 5 * sem


@pytest.mark.slow
def test_periodic_toy_identity_on_one_row():
    from analysis.periodic_toy import run

    out = run(n_samples=2000, names=["1 neuron"])
    row = out["1 neuron"]
    assert row["halfTrG"] == pytest.approx(row["I"], abs=5 * (row["I_sem"] + row["halfTrG_sem"]))


def test_tuning_toy_1d_identity():
    from analysis.tuning_toy_1d import run

    out = run(half_range=4.0, nx=401, ny=201, nr=200, per_decade=5, n_show=41)
    assert out["bell_J"] == pytest.approx(out["monotonic_J"])
    assert (out["monotonic_G"] > out["bell_G"]).all()
    for name in ("bell", "monotonic"):
        assert out[f"{name}_half_mean_G"] == pytest.approx(out[f"{name}_mi_direct"], abs=0.01)


def test_tuning_toy_2d_identity():
    from analysis.tuning_toy_2d import run_prior

    for prior in ("uniform", "ridge"):
        out = run_prior(prior, nx=61, ny=25, nr=61, per_decade=3)
        assert out[f"{prior}_half_mean_trG"] == pytest.approx(out[f"{prior}_mi_direct"], abs=0.05)


# ── scripts ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("module", [
    "analysis.alignment_ordering", "analysis.correlated_noise",
    "analysis.eigvec_component_thresholds", "analysis.eigvec_statistics",
    "analysis.mc_convergence_summary", "analysis.offmanifold_toy", "analysis.periodic_toy",
    "analysis.tuning_toy_1d", "analysis.tuning_toy_2d", "figures.fig1", "figures.fig2",
    "figures.fig_corrnoise", "figures.fig_eigstats", "figures.fig_mc",
])
def test_script_shows_help(module, capsys):
    mod = importlib.import_module(module)
    with pytest.raises(SystemExit) as e:
        mod.main(["--help"])
    assert e.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_mc_convergence_shows_help(capsys):
    pytest.importorskip("diffusers")
    from analysis.mc_convergence import main

    with pytest.raises(SystemExit):
        main(["--help"])
    assert "--area" in capsys.readouterr().out


@pytest.mark.slow
def test_appendix_figures_write_outputs(tmp_path):
    from figures import fig_corrnoise, fig_eigstats, fig_mc

    rng = np.random.default_rng(0)
    np.savez(tmp_path / "stats.npz", G=rng.uniform(1, 9, (5, 2, 2, 10, 3)),
             J=rng.uniform(1, 9, (5, 2, 2, 10, 3)))
    fig_eigstats.main(["--input", str(tmp_path / "stats.npz"), "--out-dir", str(tmp_path)])
    summary = {a: {str(n): dict(lam1_cv_mean=0.5 / np.sqrt(n), lam1_cv_sem=0.001,
                                top5_align_mean=0.9, top5_align_sem=0.01)
                   for n in (25, 50, 100, 200, 400)} for a in ("V1", "V4")}
    for a in summary:
        summary[a]["cv_slope"] = -0.5
    (tmp_path / "mc.json").write_text(json.dumps(summary))
    fig_mc.main(["--input", str(tmp_path / "mc.json"), "--out-dir", str(tmp_path)])
    rows = [dict(I_exact=2.0, half_trace_estimate=2.0, sem=0.01, mean_corr=0.1,
                 delta_I_vs_shuffled_pct=d) for d in (0, -8, -14, 22, -15, -27, -44)]
    (tmp_path / "cn.json").write_text(json.dumps({"rows": rows}))
    fig_corrnoise.main(["--input", str(tmp_path / "cn.json"), "--out-dir", str(tmp_path)])
    for name in ("fig6_corrnoise", "fig7_eigstats", "fig8_mc"):
        assert (tmp_path / f"{name}.pdf").stat().st_size > 0


def test_appendix_figures_skip_missing_input(tmp_path, capsys):
    from figures import fig_corrnoise, fig_eigstats, fig_mc

    for mod in (fig_corrnoise, fig_eigstats, fig_mc):
        assert mod.main(["--input", str(tmp_path / "absent"), "--out-dir", str(tmp_path)]) == []
