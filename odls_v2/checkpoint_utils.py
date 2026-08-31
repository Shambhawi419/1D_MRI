"""
Shared checkpoint-loading helper for ODLS-v2, mirroring ../odls/checkpoint_utils.py:
accepts either a plain state_dict ("odls_best.pt" / "odls_final.pt") or a
full training-state dict ("latest.pt", with a "model_state_dict" key).
"""

from __future__ import annotations

import torch

from model import ODLSv2


def load_model_weights(model: ODLSv2, checkpoint_path: str, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded weights from epoch {checkpoint.get('epoch', '?')} of {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"loaded weights from {checkpoint_path}")

    model.clamp_thresholds_()
