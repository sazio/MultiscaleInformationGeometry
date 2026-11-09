"""
Convert an encoder checkpoint saved by the original research code (used for
the NeurIPS 2026 figures) to the format of this repository.

The original Inception and ResNet-50 encoders have the same architecture as
`src.encoders.InceptionEncoder` / `ResNet50Encoder` (frozen backbone + per-neuron
Gaussian spatial readout); only parameter names differ:

    original                          this repository
    mu, sigma, feature_weights, ...   readout.mu, readout.sigma, ...
    backbone.features.conv1.0.*       backbone.features.conv1.*   (ResNet)
    backbone.features.conv1.1.*       backbone.features.bn1.*     (ResNet)
    backbone.features.layer1_0.*      backbone.features.layer1.0.* (ResNet)

The original OnOff encoders (frozen plenoptic front-end + dense linear readout,
the encoders of Table 4) are converted to `OnOffEncoder` with
`readout: dense` (configs/onoff_dense.yaml):

    readout_state_dict.{weight,bias}  ->  readout.linear.{weight,bias}
    frontend_state_dict.*             ->  backbone.frontend.*

Their checkpoints do not store the filter buffers plenoptic caches on the
first forward pass; these are taken from a freshly built front-end after
checking that the stored front-end parameters equal its parameters.

The converted state dict must load with strict=True; the script checks this.

Usage:
    python tools/convert_legacy_encoder.py --encoder inception \\
        --legacy-ckpt <original checkpoint> \\
        --area V1 --out runs/encoders/inception/V1/best_model.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoders import build_encoder
from src.utils import ENCODER_CONFIGS, load_config, project_root, provenance

READOUT_KEYS = ("mu", "sigma", "feature_weights", "bias", "grid_x", "grid_y")
RESNET_RENAMES = (
    ("backbone.features.conv1.0.", "backbone.features.conv1."),
    ("backbone.features.conv1.1.", "backbone.features.bn1."),
    ("backbone.features.layer1_0.", "backbone.features.layer1.0."),
)


def remap(state: dict, encoder: str) -> dict:
    out = {}
    for k, v in state.items():
        if k in READOUT_KEYS:
            k = "readout." + k
        elif encoder == "resnet50":
            for old, new in RESNET_RENAMES:
                if k.startswith(old):
                    k = new + k[len(old):]
                    break
        out[k] = v
    return out


def remap_onoff(legacy: dict, fresh: dict) -> dict:
    """Original OnOff checkpoint -> state dict of OnOffEncoder(readout="dense")."""
    state = dict(fresh)                                 # keeps the cached filter buffers
    for k, v in legacy["readout_state_dict"].items():
        state["readout.linear." + k] = v
    for k, v in legacy["frontend_state_dict"].items():
        key = "backbone.frontend." + k
        if key not in fresh:
            raise KeyError(f"front-end parameter {k!r} has no counterpart in the repo encoder")
        if not torch.equal(v, fresh[key]):
            raise ValueError(f"front-end parameter {k!r} differs from the pretrained front-end; "
                             "the cached filters would not match it")
        state[key] = v
    return state


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, choices=["inception", "resnet50", "onoff"])
    p.add_argument("--legacy-ckpt", required=True)
    p.add_argument("--area", required=True, choices=["V1", "V4"])
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None,
                   help="default: configs/<encoder>.yaml (configs/onoff_dense.yaml for onoff)")
    args = p.parse_args(argv)

    cfg_path = args.config or str(project_root() / "configs" / ENCODER_CONFIGS[args.encoder])
    legacy = torch.load(args.legacy_ckpt, map_location="cpu", weights_only=False)
    encoder = build_encoder(load_config(cfg_path))
    if args.encoder == "onoff":
        state = remap_onoff(legacy, encoder.state_dict())
    else:
        state = remap(legacy["model_state_dict"], args.encoder)
    encoder.load_state_dict(state, strict=True)        # fails loudly on any mismatch

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": encoder.state_dict(),
        "config": {"area": args.area, "encoder": args.encoder},
        "val_corr": legacy.get("val_corr"),
        "legacy_source": str(args.legacy_ckpt),
        "legacy_best_epoch": legacy.get("epoch"),
        "legacy_test_corr": legacy.get("test_corr"),       # as stored by the original code
        "provenance": provenance({"encoder": cfg_path}),
    }, out)
    print(f"converted {args.legacy_ckpt} -> {out} (strict load OK, "
          f"legacy best epoch {legacy.get('epoch')})")


if __name__ == "__main__":
    main()
