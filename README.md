# A multi-scale information geometry reveals the structure of mutual information in neural populations

Code for the NeurIPS 2026 paper by Simone Azeglio, Steeve Laquitaine, Ulisse
Ferrari and Matthew Chalk.

- Project page: <https://multigeom.github.io/>
- Checkpoints: <https://huggingface.co/sazio/multigeom>

The paper defines the *multi-scale Fisher information matrix*

```math
G(x) = \int_0^\infty [\phi_t * J_t](x)\,dt ,
```

the Fisher information of the stimulus-diffused likelihood integrated over all
diffusion scales, and shows that it is directly related to the mutual
information between stimulus and response. This repository estimates the
leading eigenvectors of $G(x)$ for natural images from a conditional diffusion
model, compares them with those of the classical Fisher information $J(x)$, and
holds the code of the macaque V1/V4 analyses and of the simulations in the
appendices. [Where each result comes from](#where-each-result-comes-from) lists
every figure and table, including what is not produced here.

## What the pipeline does

1. **Encoders** (`stages/01_train_encoder.py`). A frozen backbone followed by a
   trainable readout, fitted to the trial-averaged responses of the 50 most
   reliable V1 or V4 channels of the THINGS Ventral-stream Spiking Dataset
   (Papale et al., 2025). The Inception and ResNet-50 encoders use InceptionV3
   truncated at `Conv2d_4a_3x3` and ResNet-50 truncated at `layer1.0`, fixed
   choices selected as described in Appendix I.2, with a per-neuron Gaussian
   spatial-attention readout. The ON-OFF encoder uses the front-end of Berardino
   et al. with a dense linear readout (409,650 parameters,
   `configs/onoff_dense.yaml`).
2. **Synthetic responses** (`stages/02_precompute_responses.py`). The encoder
   $f_\phi$ is evaluated on every TinyImageNet training image; these 50-d
   vectors condition the diffusion model.
3. **Conditional diffusion model** (`stages/03_train_diffusion.py`). A
   `UNet2DConditionModel` (64×64 grayscale, about 68M parameters) conditioned on
   the responses through a 3-layer MLP and cross-attention. Training follows the
   Gaussian observation model $p(r\mid x)=\mathcal N(f_\phi(x),\sigma^2 I)$
   with $\sigma=0.22$, and uses classifier-free-guidance dropout (rate 0.1) so
   that the same network also gives the unconditional denoiser. 200 epochs.
4. **G(x) and J(x)** (`stages/04_estimate_gx.py`). The top-50 eigenpairs of
   $G(x)$ from the conditional-pair estimator (Algorithm 1 of the paper,
   `src/estimator.py`), and of $J(x)=\sigma^{-2}J_{f_\phi}^\top J_{f_\phi}$ by
   autograd (`src/fisher.py`), on 200 held-out TinyImageNet validation images.
5. **Cross-model comparison** (`stages/05_compute_metrics.py`). Subspace
   alignment $\mathcal A_k=\tfrac1k\lVert V_A^\top V_B\rVert_F^2$ and Bures-Wasserstein
   distance between every pair of (encoder, area) configurations, for
   $k=1,\dots,50$, averaged over images.
6. **Local metric field** (`stages/06_metric_field.py`). The 2×2 projection of
   $G$ and $J$ onto the leading two eigenvectors at a reference image, on a grid
   of images perturbed along those eigenvectors in steps of 1% of the norm of the
   reference image (Fig. 5).

Figures and tables are drawn from these outputs by the scripts in `figures/`.
`analysis/` holds the supporting analyses and simulations of the appendices.

## Installation

```bash
git clone https://github.com/sazio/MultiscaleInformationGeometry.git
cd MultiscaleInformationGeometry
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions we ran (Python 3.11, CUDA 12.6)
pip install -e .                       # or: pip install -r requirements.txt
```

`requirements-lock.txt` pins the versions the pipeline and tests were run
with; `requirements.txt` / `pyproject.toml` declare the supported ranges.

## Data

See [`data/README.md`](data/README.md) for the three datasets (TVSD, THINGS
images, TinyImageNet) and the expected layout. TVSD is at
<https://doi.gin.g-node.org/10.12751/g-node.hc7zlv/>. Pretrained backbone weights
are downloaded automatically.

## Checkpoints

The trained checkpoints are at <https://huggingface.co/sazio/multigeom>. To place
them where the scripts look for them:

```bash
hf download sazio/multigeom --local-dir runs
```

The `hf` command comes with `huggingface_hub`, which the requirements install. This
gives, for `<encoder>` in `inception`, `resnet50`, `onoff` and `<area>` in `V1`, `V4`:

| Path under `runs/` | Contents |
|---|---|
| `encoders/<encoder>/<area>/best_model.pt` | The six encoders; the ON-OFF encoders have the dense readout |
| `diffusion/<encoder>/<area>/final_model.pt` | The six conditional diffusion models after 200 epochs |
| `diffusion/<encoder>/<area>/train_log.json` | Loss per epoch |
| `diffusion/diffusion.yaml`, `diffusion/SHA256SUMS`, `diffusion/README.md` | Training configuration, checksums, description of the models |

Only checkpoints are released. The per-image eigenvectors are produced by running
Stage 4, and everything that reads them needs Stage 4 first.

### Checkpoint and output names

| File | Written by | Meaning |
|---|---|---|
| `runs/encoders/<encoder>/<area>/best_model.pt` | Stage 1 | Encoder at the epoch with the best validation correlation, with test-set predictions and per-neuron correlations |
| `runs/responses/<encoder>/<area>/{train,val}_responses.pt` | Stage 2 | Encoder responses to TinyImageNet |
| `runs/diffusion/<encoder>/<area>/final_model.pt` | Stage 3 | Diffusion model after the last epoch; used for every figure unless stated |
| `runs/diffusion/<encoder>/<area>/model_epoch_NNN.pt` | Stage 3 | Snapshot every `--save-every` epochs |
| `runs/per_image/<encoder>/<area>/imgNNNN.pt` | Stage 4 | Top-50 eigenpairs of G(x) and J(x) for test image NNNN |
| `runs/metrics/cross_encoder_metrics.json` | Stage 5 | Alignment and Bures-Wasserstein curves |
| `runs/metric_field/imgNNNN/{G,J}_{V1,V4}.pt` | Stage 6 | 2×2 projected metrics on the perturbation grid |

`<encoder>` is `inception`, `resnet50` or `onoff`; `<area>` is `V1` or `V4`.

### `tools/` and `--evaluate-only`

The encoders of the paper were trained by the original research code, before this
repository existed. `tools/convert_legacy_encoder.py` converts such a checkpoint
to the format used here and checks that it loads strictly. You need it only if you
hold a checkpoint from that code; the encoders on Hugging Face are already
converted.

`stages/01_train_encoder.py --evaluate-only --encoder-ckpt <file>` does not train.
It runs an existing encoder on the TVSD test images and writes it to `--out-dir`
with its test-set predictions and per-neuron correlations added. Fig. 3B-C, Fig. 4A
and Table 4 read these fields, so use it for an encoder checkpoint that does not
have them. It needs the TVSD files and the THINGS test images. For ON-OFF, use
`--config configs/onoff_dense.yaml`:

```bash
python stages/01_train_encoder.py --config configs/inception.yaml --area V1 --evaluate-only \
    --encoder-ckpt runs/encoders/inception/V1/best_model.pt \
    --out-dir runs/encoders/inception/V1
```

## Running

One (encoder, area) configuration end to end, stages 1-4:

```bash
ENCODER=inception AREA=V1 bash reproduce.sh       # also: resnet50, onoff × V1, V4
```

With the checkpoints in place, skip stages 1-3 and run Stage 4 directly:

```bash
python stages/04_estimate_gx.py --config configs/estimator.yaml \
    --encoder-cfg configs/inception.yaml \
    --encoder-ckpt runs/encoders/inception/V1/best_model.pt \
    --diffusion-ckpt runs/diffusion/inception/V1/final_model.pt \
    --area V1 --device cuda:0 --out-dir runs/per_image/inception/V1
```

Stage 4 accepts `--start/--end` to split the 200 test images across GPUs. Each
image is seeded independently (`seed + image index`), so the split does not change
the results. Every checkpoint and output file stores a provenance record (git
commit, command line, seed, config files, package versions).

Then the comparisons and the appendix analyses, all on the 200 test images:

```bash
python stages/05_compute_metrics.py --runs-dir runs --out-dir runs/metrics
python analysis/eigvec_statistics.py --runs-dir runs --out-dir runs/analysis
python analysis/eigvec_component_thresholds.py --runs-dir runs
python analysis/alignment_ordering.py --runs-dir runs
```

### The example image of Figs. 3D-F, 4B-C and 5

The example image is a chosen image, not one of the 200 test images and not picked
by any rule in this code: it is the frog `n01644900_0.JPEG` from the TinyImageNet
*training* set, so it is also in the training set of the diffusion models. Stage 4
reads only the validation split, so we made a one-image copy of that layout and a
second runs tree that links to the first. No code changes are needed:

```bash
mkdir -p frog_data/tiny-imagenet-200/val/frog runs_frog
cp data/tiny-imagenet-200/train/n01644900/images/n01644900_0.JPEG \
   frog_data/tiny-imagenet-200/val/frog/
for d in encoders diffusion metrics analysis; do ln -s "../runs/$d" "runs_frog/$d"; done
sed 's/^n_test_images: .*/n_test_images: 1/' configs/estimator.yaml > runs_frog/estimator_frog.yaml

# Stage 4 for each (encoder, area); use configs/onoff_dense.yaml for onoff
NEURIPS_DATA_DIR=frog_data python stages/04_estimate_gx.py \
    --config runs_frog/estimator_frog.yaml --encoder-cfg configs/inception.yaml \
    --encoder-ckpt runs/encoders/inception/V1/best_model.pt \
    --diffusion-ckpt runs/diffusion/inception/V1/final_model.pt \
    --area V1 --device cuda:0 --out-dir runs_frog/per_image/inception/V1

python stages/06_metric_field.py --img-idx 0 --runs-dir runs_frog --data-dir frog_data
python figures/fig3.py --runs-dir runs_frog --out-dir figures/out --img-idx 0
python figures/fig4.py --runs-dir runs_frog --out-dir figures/out --img-idx 0
python figures/fig5.py --runs-dir runs_frog --out-dir figures/out --img-idx 0
```

Without the frog, `--runs-dir runs --img-idx 0` draws the same figures for the first
test image.

## Where each result comes from

"Printed" means the version in the paper. Layout and fonts of the scripts' output
differ from the printed figures, which were composed from panels.

| Paper item | Command or script | Inputs | Notes |
|---|---|---|---|
| Fig. 1 | `analysis/tuning_toy_1d.py`, then `figures/fig1.py` | none | Toy model |
| Fig. 2 | `analysis/tuning_toy_2d.py`, then `figures/fig2.py` | none | Toy model |
| Fig. 3A | none | | No code in this repository (schematic) |
| Fig. 3B-C | `figures/fig3.py` | Stage 1 test-set fields (`--evaluate-only`) | |
| Fig. 3D | `figures/fig3.py` | Stage 4 on the frog, Stage 1 readouts | Chosen image, see above. `fig3.py` draws the outlines from the readout parameters; the printed panel was kept from an earlier version of the figure |
| Fig. 3E-F | `figures/fig3.py` | Stage 4 on the frog | Selected tiles, not rank order: the rule is in `figures/_eigvec_selection.py`, the ranks go to `fig3_selection_img0000.json`. |
| Fig. 3G-H | `analysis/eigvec_statistics.py`, then `figures/fig3.py` | Stage 4 on the 200 test images | `fig3.py --statistics <npz>` if the file is elsewhere |
| Fig. 4A | `figures/fig4.py` | Stage 1 test-set fields | The printed panel was kept from an earlier version of the figure |
| Fig. 4B-C | `figures/fig4.py` | Stage 4 on the frog | Same rule as Fig. 3, with ON-OFF matching; ranks in `fig4_selection_img0000.json` |
| Fig. 4D-E | `stages/05_compute_metrics.py`, then `figures/fig4.py` | Stage 4 on the 200 test images | **By default every curve is from the 200-epoch models and 200 images. The printed panel D differs in one curve:** "V1 vs V4 (Res)" of G is from the epoch-50 ResNet-50 models on the first 40 test images. `fig4.py --as-printed-res-v1v4 <npz>` draws it from a file with arrays `k`, `ep050_ResV1vsV4_mean`, `ep050_ResV1vsV4_sem`. That file, the epoch-50 checkpoints and the script that wrote the file are not in this repository or in the release |
| Fig. 5 | `stages/06_metric_field.py`, then `figures/fig5.py` | Encoders, diffusion models, frog | Defaults are those of the printed figure: 5×5 grid, steps of 1% of the norm of the image, Inception and ResNet-50 |
| Fig. 6 | `analysis/correlated_noise.py`, then `figures/fig_corrnoise.py` | none | Simulation |
| Fig. 7 | `analysis/eigvec_statistics.py`, then `figures/fig_eigstats.py` | Stage 4 on the 200 test images | |
| Fig. 8 | `analysis/mc_convergence.py`, `analysis/mc_convergence_summary.py`, then `figures/fig_mc.py` | Inception encoders and diffusion models | About 8 A100-GPU-hours |
| Fig. 9 | `stages/05_compute_metrics.py`, then `figures/fig_app_bw.py` | Stage 4 on the 200 test images | Colours differ from print; the values are the same |
| Table 1 | `analysis/offmanifold_toy.py` | none | Simulation |
| Table 2 | `analysis/correlated_noise.py` | none | Simulation |
| Table 3 | `analysis/periodic_toy.py` | none | Simulation |
| Table 4 | `figures/table4.py` | Stage 1 test-set fields | |
| Section 6, ordering at k = 10 | `analysis/alignment_ordering.py` | Stage 4 on the 200 test images | 200-epoch models |
| Section 6, differences between areas | `analysis/eigvec_statistics.py` | Stage 4 on the 200 test images | `mean_abs_V1_minus_V4` in the JSON |
| Appendix F numbers | `analysis/eigvec_statistics.py`, `analysis/eigvec_component_thresholds.py` | Stage 4 on the 200 test images | Inception and ResNet-50 |
| Appendix G numbers | `analysis/mc_convergence_summary.py` | `analysis/mc_convergence.py` | `cv_slope_25_200` for the quoted slopes |
| CC versus UC (Appendix H.2): median 67% lower variance, UC 2.8-5.5 times CC | `analysis/estimator_variance.py` | Inception V1 encoder and diffusion model | 6 images, 6 runs, 300 network passes per scale for each estimator |
| σ = 0.22 (Appendix H.1) | `analysis/calibrate_sigma.py` | TVSD | All runs use σ = 0.22, calibrated to the residual trial-to-trial variability; the script computes the per-electrode medians |

The truncation layers are fixed in `configs/`; the code that selected them is not in
this repository. Appendices A and B are derivations and have no code here.

Figure commands:

```bash
python figures/table4.py      --runs-dir runs --out-dir figures/out
python figures/fig1.py          --input runs/analysis/tuning_toy_1d.npz       --out-dir figures/out
python figures/fig2.py          --input runs/analysis/tuning_toy_2d.npz       --out-dir figures/out
python figures/fig_corrnoise.py --input runs/analysis/correlated_noise.json   --out-dir figures/out
python figures/fig_eigstats.py  --input runs/analysis/eigvec_statistics.npz   --out-dir figures/out
python figures/fig_mc.py        --input runs/analysis/mc_convergence_summary.json --out-dir figures/out
python figures/fig_app_bw.py  --runs-dir runs --out-dir figures/out
```

See [`analysis/README.md`](analysis/README.md) for the commands of the analyses.

## Compute

All numbers are from the paper, on NVIDIA A100-SXM4-40GB GPUs:

| Step | Cost |
|---|---|
| Encoder readouts | under 1 A100-GPU-hour in total |
| Stage 2, TinyImageNet responses | minutes |
| Stage 3, one diffusion model, 200 epochs | about 13.5 h on one A100 (about 4.1 min per epoch); Stage 3 uses a single GPU |
| Stage 3, all six models | about 81 A100-GPU-hours |
| Stage 4, per image | 33-85 s, median about 40 s |
| Stage 4, 200 images × 6 configurations | about 15 A100-GPU-hours |
| **Total for the reported experiments** | **about 100 A100-GPU-hours** |
| Monte Carlo convergence (Appendix G) | a further 8 A100-GPU-hours or so |
| Simulations of Appendices C-E | CPU |
| Toy models of Figs. 1 and 2 | under a minute each (Fig. 2 on a GPU) |

## Tests

```bash
pytest -m "not slow and not pretrained and not gpu"   # CPU, no downloads
pytest                                                # also slow and pretrained tests
```

## Repository layout

```
configs/      encoder, diffusion and estimator settings (one YAML per component)
src/          library: data loading, encoders, diffusion model, estimator, Fisher, metrics
stages/       numbered pipeline scripts 01-06
figures/      figure and table scripts; _eigvec_selection.py holds the tile rule
analysis/     supporting analyses and simulations of the appendices
tools/        converter for checkpoints of the original research code
tests/        pytest suite
data/         dataset location (see data/README.md)
reproduce.sh  stages 1-4 for one (encoder, area) configuration
```

## Citation

See [`CITATION.cff`](CITATION.cff).

```bibtex
@inproceedings{azeglio2026multiscale,
  title     = {A multi-scale information geometry reveals the structure of mutual information in neural populations},
  author    = {Azeglio, Simone and Laquitaine, Steeve and Ferrari, Ulisse and Chalk, Matthew},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`). The datasets keep their own licenses: TVSD
(Papale et al.), THINGS images (Hebart et al.), TinyImageNet. The ON-OFF
front-end and its pretrained parameters come from `plenoptic`; the
conditional UNet from Hugging Face `diffusers`.
