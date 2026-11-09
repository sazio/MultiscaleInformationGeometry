# Supporting analyses

Scripts for numbers and figures quoted in the appendices of the paper. They are
not part of the main pipeline (`stages/01` to `stages/06`). Unless stated, each
reads Stage-4 files from `<runs-dir>/per_image/<encoder>/<area>/` and writes to
`runs/analysis/`. [`../README.md`](../README.md) lists which paper item each one
produces.

| Script | Paper item | Inputs | Cost |
|---|---|---|---|
| `calibrate_sigma.py` | Per-electrode trial-to-trial variability behind σ (Appendix H.1) | TVSD `monkeyN_THINGS_normMUA.mat`, `things_imgs.mat` | seconds, CPU |
| `estimator_variance.py` | CC versus UC estimator (Appendix H.2) | Stage-1 encoder and Stage-3 diffusion checkpoints, test images | under an hour, one GPU |
| `eigvec_statistics.py` | Fig. 3G-H, Fig. 7, Appendix F, and the differences between areas in Section 6 | Stage 4 | minutes, CPU |
| `eigvec_component_thresholds.py` | Thresholds of the component count (Appendix F) | Stage 4 | minutes, CPU |
| `alignment_ordering.py` | Per-image ordering at k = 10 (Section 6) | Stage 4 | minutes, CPU |
| `mc_convergence.py` | Raw runs for Appendix G and Fig. 8 | Inception encoder and diffusion checkpoints, test images | about 8 A100-GPU-hours in total |
| `mc_convergence_summary.py` | Appendix G numbers, input of Fig. 8 | output of `mc_convergence.py` | seconds, CPU |
| `offmanifold_toy.py` | Table 1 (Appendix C) | none | seconds, CPU |
| `correlated_noise.py` | Table 2, input of Fig. 6 (Appendix D) | none | about a minute, CPU |
| `periodic_toy.py` | Table 3 (Appendix E) | none | tens of minutes, CPU |
| `tuning_toy_1d.py` | Toy model, input of Fig. 1 | none | under a minute, CPU |
| `tuning_toy_2d.py` | Toy model, input of Fig. 2 | none | under a minute, GPU |

## calibrate_sigma.py

All runs use σ = 0.22. It was calibrated to the residual trial-to-trial
variability of the recordings, and this script computes the per-electrode
medians.

`test_MUA_reps` holds the normalised MUA of the 30 repetitions of each of the
100 test images (h5py order: `[30, 100, 1024]`). For each channel the script
takes the root-mean-square, over trials and images, of the per-trial response
minus its trial average, and reports the median over channels. The primary
number uses the top-50 channels per area by oracle reliability (the encoder
targets, selected with `src.data_loaders.load_tvsd`). The median over all
channels of the area is reported alongside it.

    python analysis/calibrate_sigma.py --out runs/analysis/sigma_calibration.json

## estimator_variance.py

Runs the conditional-pair (CC) estimator of `src.estimator` and the
unconditional-conditional (UC) form on a few test images, several times each
with independent seeds, and compares the across-run variance of the top-5
eigenvalues. The paper reports a median 67% lower variance for CC, and a leading
UC eigenvalue 2.8 to 5.5 times the CC value, from the command below.

The UC row is `sqrt(w_j / N_MC) * (eps(x_t, t, r) - eps(x_t, t, 0))`, with one
response draw per sample. The factor 1/2 of the CC row is absent because the
UC difference is centred on the null-token prediction, which equals the mean
of the conditional predictions when the models are correct. Both forms then
have the same expectation; any mismatch between the null-token prediction and
that mean adds a bias to UC only. The module docstring gives the derivation.

The null-token pass that gives the denoised image is reused as the UC
subtraction term, so a UC sample costs 2 UNet passes and a CC sample costs 3.
By default (`--compute-match passes`) UC gets 1.5 times as many samples, so
that both spend the same number of passes. `--compute-match samples` uses the
same sample count for both.

    python analysis/estimator_variance.py \
        --encoder-cfg configs/inception.yaml \
        --encoder-ckpt runs/encoders/inception/V1/best_model.pt \
        --diffusion-ckpt runs/diffusion/inception/V1/final_model.pt \
        --n-images 6 --n-runs 6 --n-mc 100 \
        --out runs/analysis/estimator_variance_inception_V1.json

The output reports both the raw-variance reduction and the reduction in
squared coefficient of variation (variance divided by squared mean). The raw
variance also reflects the larger overall scale of the UC estimate, while the
coefficient of variation does not.

## Eigenvector statistics

    python analysis/eigvec_statistics.py --runs-dir runs --out-dir runs/analysis
    python analysis/eigvec_component_thresholds.py --runs-dir runs \
        --out runs/analysis/eigvec_component_thresholds.json

`eigvec_statistics.npz` holds the per-image values and is read by
`figures/fig3.py` (panels G-H) and `figures/fig_eigstats.py` (Fig. 7).
`--from-npz` recomputes the statistics from a saved file without the Stage-4
outputs.

## Alignment ordering

    python analysis/alignment_ordering.py --runs-dir runs \
        --out runs/analysis/alignment_ordering.json

The curves of Fig. 4D-E and Fig. 9 themselves come from Stage 5.

## Monte Carlo convergence

    for area in V1 V4; do
      python analysis/mc_convergence.py --area $area --start 0 --end 10 \
          --runs-dir runs --out-dir runs/analysis/mc/$area --device cuda:0
    done
    python analysis/mc_convergence_summary.py --mc-dir runs/analysis/mc \
        --out runs/analysis/mc_convergence_summary.json

## Simulations

    python analysis/offmanifold_toy.py --out runs/analysis/offmanifold_toy.json
    python analysis/correlated_noise.py --out runs/analysis/correlated_noise.json
    python analysis/periodic_toy.py --out runs/analysis/periodic_toy.json
    python analysis/tuning_toy_1d.py --out runs/analysis/tuning_toy_1d.npz
    python analysis/tuning_toy_2d.py --out runs/analysis/tuning_toy_2d.npz

The first three need only NumPy and SciPy, use fixed seeds, and print the table rows. The
tuning toys use grids rather than sampling, and print the mutual information and 1/2 <Tr G>;
`tuning_toy_2d.py` also needs PyTorch.
