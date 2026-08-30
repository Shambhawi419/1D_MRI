"""
Shared checkpoint-loading helper used by both train.py (--init-checkpoint,
resume via "latest.pt") and evaluate.py (--checkpoint), so a checkpoint in
either of the two formats train.py can produce is accepted everywhere:

  - a plain state_dict, as saved by torch.save(model.state_dict(), ...)
    for "odls_best.pt" / "odls_final.pt", or
  - a full training-state dict (train.py's per-epoch "latest.pt"), with
    a "model_state_dict" key alongside optimizer/scheduler/epoch state.
"""

from __future__ import annotations

import torch

from model import ODLS


def load_model_weights(model: ODLS, checkpoint_path: str, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded weights from epoch {checkpoint.get('epoch', '?')} of {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"loaded weights from {checkpoint_path}")

    # Keeps a loaded checkpoint's stored threshold values consistent with
    # the model's current max_threshold cap -- see ODLS.clamp_thresholds_.
    model.clamp_thresholds_()
