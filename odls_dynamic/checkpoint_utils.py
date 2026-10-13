"""
Checkpointing built for free Colab + Google Drive, where a session can die at
any moment -- including in the middle of writing a file.

Files in a run's checkpoint directory (all on Drive):
  latest.pt        full training state: model, optimizer, scheduler, AMP
                   scaler, epoch + batch position inside the epoch, partial
                   epoch sums, best-so-far bookkeeping, every RNG state, the
                   model config and the full training args. Written every
                   epoch and every --save-every-steps optimizer steps.
  latest_prev.pt   the previous latest.pt, kept as a fallback in case the
                   newest file is truncated or corrupted.
  best.pt          weights + config of the best smoothed validation loss.
  final.pt         weights + config when training finishes.
  history.csv      one row per epoch (losses, lr, per-phase eta/lambda).
  run_log.txt      one line per session start / resume / stop.

Every file is written to "<name>.tmp" first and then renamed over the target,
so an interruption mid-write can never leave a half-written checkpoint under
the real name.

Every checkpoint carries `model_config`, so any later workflow (fine-tuning,
another AF, evaluation, visualisation, your own script) rebuilds the exact
model from the file alone:

    from checkpoint_utils import build_model_from_checkpoint
    model, ckpt = build_model_from_checkpoint(".../best.pt", device="cuda")
"""

from __future__ import annotations

import os
import random
import shutil
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from model import ODLSDynamic

FORMAT_VERSION = 1
LATEST, LATEST_PREV, BEST, FINAL = "latest.pt", "latest_prev.pt", "best.pt", "final.pt"


def _load(path: str, device) -> Dict:
    # Our own files contain numpy RNG state and plain Python objects, so they
    # need weights_only=False (the default flipped to True in torch 2.6).
    return torch.load(path, map_location=device, weights_only=False)


def _replace(src: str, dst: str) -> None:
    try:
        os.replace(src, dst)
    except OSError:  # some network filesystems refuse replace-over-existing
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)


def atomic_save(obj: Dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    _replace(tmp, path)


def save_latest(state: Dict, checkpoint_dir: str) -> None:
    """Writes latest.pt, first rotating the old one to latest_prev.pt. The
    new state is fully written (as latest.pt.tmp) before anything moves."""
    latest = os.path.join(checkpoint_dir, LATEST)
    tmp = latest + ".tmp"
    torch.save(state, tmp)
    if os.path.exists(latest):
        _replace(latest, os.path.join(checkpoint_dir, LATEST_PREV))
    _replace(tmp, latest)


def load_latest(checkpoint_dir: str, device) -> Optional[Dict]:
    """Newest readable full training state, or None if there is none yet.
    Falls back to latest_prev.pt if latest.pt is missing or corrupted. If a
    session died between save_latest's two renames, latest.pt is missing but
    latest.pt.tmp is complete (it was fully written first), so try it too."""
    candidates = [LATEST, LATEST_PREV]
    if not os.path.exists(os.path.join(checkpoint_dir, LATEST)):
        candidates.insert(1, LATEST + ".tmp")
    for name in candidates:
        path = os.path.join(checkpoint_dir, name)
        if not os.path.exists(path):
            continue
        try:
            state = _load(path, device)
            if name != LATEST:
                print(f"WARNING: {LATEST} unreadable or missing, resuming from {name}")
            return state
        except Exception as exc:  # truncated / corrupted file
            print(f"WARNING: could not read {path} ({exc!r}); trying the fallback")
    return None


def weights_checkpoint(model: ODLSDynamic, **meta) -> Dict:
    return {"format_version": FORMAT_VERSION, "model_config": dict(model.config),
            "model_state_dict": model.state_dict(), **meta}


def rng_state() -> Dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        except RuntimeError:  # different GPU count than when saved
            pass


def load_model_weights(model: ODLSDynamic, checkpoint_path: str, device) -> Dict:
    """Loads weights from any of latest.pt / latest_prev.pt / best.pt /
    final.pt (or a bare state_dict) into an existing model."""
    ckpt = _load(checkpoint_path, device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.clamp_thresholds_()
    meta = ckpt if isinstance(ckpt, dict) and "model_state_dict" in ckpt else {}
    where = f"epoch {meta['epoch']} of " if "epoch" in meta else ""
    print(f"loaded weights from {where}{checkpoint_path}")
    return meta


def build_model_from_checkpoint(checkpoint_path: str, device="cpu",
                                **overrides) -> Tuple[ODLSDynamic, Dict]:
    """Rebuilds the model described by the checkpoint and loads its weights.
    `overrides` change construction args that don't affect weight shapes
    (e.g. max_threshold)."""
    ckpt = _load(checkpoint_path, device)
    if not (isinstance(ckpt, dict) and "model_config" in ckpt):
        raise ValueError(f"{checkpoint_path} has no model_config; build ODLSDynamic yourself "
                         f"and use load_model_weights()")
    model = ODLSDynamic(**{**ckpt["model_config"], **overrides}).to(device)
    load_model_weights(model, checkpoint_path, device)
    return model, ckpt
