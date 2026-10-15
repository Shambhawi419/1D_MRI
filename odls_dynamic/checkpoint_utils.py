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
  snapshots/       full state at the end of each of the last 3 epochs --
                   further fallbacks (already fully uploaded to Drive).
  best.pt          weights + config of the best smoothed validation loss.
  final.pt         weights + config when training finishes.
  history.csv      one row per epoch (losses, lr, per-phase eta/lambda).
  run_log.txt      one line per session start / resume / stop.

Every file is written to "<name>.tmp" first, fsync'ed, then renamed over the
target, so an interruption mid-write can never leave a half-written checkpoint
under the real name. If every state file is unreadable anyway, loading raises
CheckpointUnreadable rather than letting train.py start over on top of the
run (a fresh start always moves an earlier run's files to previous_<time>/).

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


SNAPSHOT_DIR, KEEP_SNAPSHOTS = "snapshots", 3


class CheckpointUnreadable(RuntimeError):
    """Checkpoint files exist but none can be read. Raised instead of
    returning None so a caller never mistakes it for "no run yet" and starts
    over on top of a run's files (that would overwrite its best.pt)."""


def _write_durable(obj: Dict, path: str) -> None:
    """torch.save + flush + fsync. On Colab's Google Drive mount a file can
    look complete to the VM while its upload is still in flight; if the
    session dies then, Drive keeps a truncated file. fsync pushes the data
    out before we rename it into place."""
    with open(path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())


def atomic_save(obj: Dict, path: str) -> None:
    tmp = path + ".tmp"
    _write_durable(obj, tmp)
    _replace(tmp, path)


def save_latest(state: Dict, checkpoint_dir: str, snapshot: bool = False) -> None:
    """Writes latest.pt, first rotating the old one to latest_prev.pt. The
    new state is fully written (as latest.pt.tmp) before anything moves.
    With snapshot=True (every epoch end) also keeps a copy in snapshots/,
    pruned to the newest KEEP_SNAPSHOTS: older snapshots have long since
    finished uploading to Drive, so they survive what kills latest.pt."""
    latest = os.path.join(checkpoint_dir, LATEST)
    tmp = latest + ".tmp"
    _write_durable(state, tmp)
    if os.path.exists(latest):
        _replace(latest, os.path.join(checkpoint_dir, LATEST_PREV))
    _replace(tmp, latest)
    if snapshot:
        snap_dir = os.path.join(checkpoint_dir, SNAPSHOT_DIR)
        os.makedirs(snap_dir, exist_ok=True)
        atomic_save(state, os.path.join(snap_dir, f"epoch_{state['epoch'] - 1:04d}.pt"))
        snaps = sorted(n for n in os.listdir(snap_dir) if n.startswith("epoch_") and n.endswith(".pt"))
        for old in snaps[:-KEEP_SNAPSHOTS]:
            os.remove(os.path.join(snap_dir, old))


def state_candidates(checkpoint_dir: str):
    """Existing full-state files, newest first."""
    names = [LATEST, LATEST + ".tmp", LATEST_PREV]
    paths = [os.path.join(checkpoint_dir, n) for n in names]
    snap_dir = os.path.join(checkpoint_dir, SNAPSHOT_DIR)
    if os.path.isdir(snap_dir):
        paths += [os.path.join(snap_dir, n) for n in sorted(os.listdir(snap_dir), reverse=True)
                  if n.startswith("epoch_") and n.endswith(".pt")]
    return [p for p in paths if os.path.exists(p)]


def load_latest(checkpoint_dir: str, device) -> Optional[Dict]:
    """Newest readable full training state; None only if the folder holds no
    training state at all. Tries latest.pt, latest.pt.tmp (complete if a
    session died between save_latest's renames), latest_prev.pt, then the
    epoch snapshots newest first. Raises CheckpointUnreadable if files exist
    but every one is corrupted."""
    candidates = state_candidates(checkpoint_dir)
    for i, path in enumerate(candidates):
        try:
            state = _load(path, device)
            if i:
                print(f"WARNING: resuming from fallback {path} (epoch {state.get('epoch')}, "
                      f"batch {state.get('batch_in_epoch')})")
            return state
        except Exception as exc:  # truncated / corrupted file
            print(f"WARNING: could not read {path} ({exc!r}); trying the next fallback")
    if candidates:
        raise CheckpointUnreadable(f"none of {len(candidates)} checkpoint file(s) in {checkpoint_dir} "
                                   f"could be read")
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
