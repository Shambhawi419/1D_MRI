"""
Shared checkpoint-loading helper used by both train.py (--init-checkpoint,
resume via "latest.pt") and evaluate.py (--checkpoint), so a checkpoint in
either of the two formats train.py can produce is accepted everywhere:

  - a plain state_dict, as saved by torch.save(model.state_dict(), ...)
    for "odls_best.pt" / "odls_final.pt", or
  - a full training-state dict (train.py's per-epoch "latest.pt"), with
    a "model_state_dict" key alongside optimizer/scheduler/epoch state.

IMPORTANT -- data-scale breaking change: fastmri_data.py now normalizes
every slice's k-space amplitude (see compute_normalization_scale), which
did not happen before. A checkpoint trained before that change has every
weight tuned to the old (~1e-5 magnitude) input scale; loading it now and
continuing training (via --init-checkpoint or a resume) would feed those
weights a completely different input distribution (~O(1) magnitude) than
they were ever trained on. This function will load such a checkpoint
without error (the state_dict shapes/keys are unaffected), but the
resulting model's behavior on real data should not be trusted -- start
training from a fresh checkpoint directory instead.
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
