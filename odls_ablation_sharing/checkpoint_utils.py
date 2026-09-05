"""
Shared checkpoint-loading helper for the weight-sharing-only ablation,
mirroring ../odls/checkpoint_utils.py.
"""

from __future__ import annotations

import torch

from model import ODLSAblationSharing


def load_model_weights(model: ODLSAblationSharing, checkpoint_path: str, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded weights from epoch {checkpoint.get('epoch', '?')} of {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"loaded weights from {checkpoint_path}")

    model.clamp_thresholds_()
