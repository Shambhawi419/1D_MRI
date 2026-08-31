"""
Training script for ODLS-v2 (the weight-shared / denoiser-prior /
cross-phase-attention research variant -- see model.py's module
docstring for what's different from the ../odls baseline).

Everything about checkpointing, resume, --init-checkpoint pretraining,
the smoothed best-checkpoint criterion, and the Colab-friendly worker/
persistent-cache setup is identical in spirit to ../odls/train.py; the
only functional difference here is the model class (ODLSv2) and its
extra --embed-dim hyperparameter for the phase-conditioning embedding.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import ODLSHybridDataset
from losses import ODLSLoss
from model import ODLSv2
from fastmri_data import FastMRICorpdDataset, find_corpd_files
from checkpoint_utils import load_model_weights


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ODLS-v2 (research variant)")

    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--train-volumes", type=str, nargs="+",
                             help="Paths to .npy files, each holding a complex "
                                  "(n_slices, M, N, J) fully-sampled k-space volume.")
    data_group.add_argument("--fastmri-train-root", type=str,
                             help="Directory of fastMRI knee multicoil .h5 files. "
                                  "Only CORPD_FBK volumes are used unless "
                                  "--fastmri-fat-suppressed is set.")

    p.add_argument("--val-volumes", type=str, nargs="+", default=None)
    p.add_argument("--fastmri-val-root", type=str, default=None)
    p.add_argument("--fastmri-fat-suppressed", action="store_true",
                    help="Use CORPDFS_FBK (fat-suppressed) instead of CORPD_FBK.")
    p.add_argument("--n-virtual-coils", type=int, default=8,
                    help="SVD coil-compression target for fastMRI inputs "
                         "(ignored for --train-volumes). Must equal --n-coils.")
    p.add_argument("--crop-fe-to", type=int, default=224)
    p.add_argument("--crop-pe-to", type=int, default=224)

    p.add_argument("--mask-type", type=str, default="cartesian",
                    choices=["cartesian", "uniform", "partial_fourier"])
    p.add_argument("--af", type=float, default=4.0)
    p.add_argument("--n-coils", type=int, required=True)
    p.add_argument("--n-phases", type=int, default=10)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--embed-dim", type=int, default=16,
                    help="Dimension of the learnable per-phase embedding used "
                         "to FiLM-condition the shared conv stacks and offset "
                         "the shared eta1/eta2/threshold scalars (weight "
                         "sharing, direction 1).")
    p.add_argument("--max-threshold", type=float, default=0.05,
                    help="Upper bound on the shared soft-threshold's base value "
                         "(init 0.001) -- same runaway-growth safeguard as the "
                         "baseline; weight sharing doesn't remove the underlying "
                         "incentive that caused it to explode there.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay", type=float, default=0.99)
    p.add_argument("--sym-weight", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--init-checkpoint", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--best-checkpoint-window", type=int, default=5)
    return p


def load_volumes(paths: List[str]) -> List[np.ndarray]:
    volumes = []
    for path in paths:
        vol = np.load(path)
        if not np.iscomplexobj(vol):
            raise ValueError(f"{path}: expected complex-valued k-space, got dtype {vol.dtype}")
        if vol.ndim != 4:
            raise ValueError(f"{path}: expected shape (n_slices, M, N, J), got {vol.shape}")
        volumes.append(vol)
    return volumes


def run_epoch(model: ODLSv2, loader: DataLoader, criterion: ODLSLoss,
              optimizer, device: str, train: bool, epoch: int, total_epochs: int) -> dict:
    model.train(mode=train)
    total_loss, total_err, total_sym, n_batches = 0.0, 0.0, 0.0, 0

    phase_name = "train" if train else "val"
    progress = tqdm(loader, desc=f"epoch {epoch}/{total_epochs} [{phase_name}]", leave=False)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in progress:
            z = batch["z"].to(device)
            e_ref = batch["e_ref"].to(device)
            mask = batch["mask"].to(device)

            e_preds, sym_pairs = model(z, mask)
            loss, parts = criterion(e_ref, e_preds, sym_pairs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_err += parts["err"].item()
            total_sym += parts["sym"].item()
            n_batches += 1
            progress.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "err": total_err / max(n_batches, 1),
        "sym": total_sym / max(n_batches, 1),
    }


def build_train_and_val_sets(args):
    if args.fastmri_train_root:
        if args.n_virtual_coils != args.n_coils:
            raise ValueError(
                f"--n-virtual-coils ({args.n_virtual_coils}) must equal --n-coils "
                f"({args.n_coils})."
            )
        train_files = find_corpd_files(args.fastmri_train_root, fat_suppressed=args.fastmri_fat_suppressed)
        if not train_files:
            raise ValueError(f"No matching CORPD{'FS' if args.fastmri_fat_suppressed else ''}_FBK "
                              f"files found under {args.fastmri_train_root}")
        train_set = FastMRICorpdDataset(
            train_files, mask_type=args.mask_type, af=args.af,
            n_virtual_coils=args.n_virtual_coils, crop_fe_to=args.crop_fe_to,
            crop_pe_to=args.crop_pe_to,
        )

        val_set = None
        if args.fastmri_val_root:
            val_files = find_corpd_files(args.fastmri_val_root, fat_suppressed=args.fastmri_fat_suppressed)
            if not val_files:
                raise ValueError(f"No matching CORPD{'FS' if args.fastmri_fat_suppressed else ''}_FBK "
                                  f"files found under {args.fastmri_val_root}")
            val_set = FastMRICorpdDataset(
                val_files, mask_type=args.mask_type, af=args.af,
                n_virtual_coils=args.n_virtual_coils, crop_fe_to=args.crop_fe_to,
                crop_pe_to=args.crop_pe_to,
                fixed_mask=True, seed=0,
            )
        return train_set, val_set

    train_volumes = load_volumes(args.train_volumes)
    train_set = ODLSHybridDataset(train_volumes, mask_type=args.mask_type, af=args.af)

    val_set = None
    if args.val_volumes:
        val_volumes = load_volumes(args.val_volumes)
        val_set = ODLSHybridDataset(val_volumes, mask_type=args.mask_type, af=args.af,
                                     fixed_mask=True, seed=0)
    return train_set, val_set


def _load_resume_state(checkpoint_dir: str, device: str) -> Optional[dict]:
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if not os.path.exists(latest_path):
        return None
    return torch.load(latest_path, map_location=device)


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_set, val_set = build_train_and_val_sets(args)
    use_workers = args.num_workers > 0
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True,
                               persistent_workers=use_workers, pin_memory=torch.cuda.is_available())

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 persistent_workers=use_workers, pin_memory=torch.cuda.is_available())

    model = ODLSv2(n_coils=args.n_coils, n_phases=args.n_phases, width=args.width,
                    max_threshold=args.max_threshold, embed_dim=args.embed_dim).to(args.device)
    criterion = ODLSLoss(n_coils=args.n_coils, sym_weight=args.sym_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    start_epoch = 1
    best_val_loss = float("inf")
    best_smoothed_val_loss = float("inf")
    val_loss_history: List[float] = []

    resume_state = None if args.no_resume else _load_resume_state(args.checkpoint_dir, args.device)
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        model.clamp_thresholds_()
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        best_val_loss = resume_state["best_val_loss"]
        best_smoothed_val_loss = resume_state.get("best_smoothed_val_loss", float("inf"))
        val_loss_history = resume_state.get("val_loss_history", [])
        start_epoch = resume_state["epoch"] + 1
        print(f"resumed from {os.path.join(args.checkpoint_dir, 'latest.pt')}: "
              f"continuing at epoch {start_epoch}/{args.epochs} "
              f"(best_val_loss so far = {best_val_loss:.6f}, "
              f"best {args.best_checkpoint_window}-epoch trailing avg so far = "
              f"{best_smoothed_val_loss:.6f})")
    elif args.init_checkpoint:
        load_model_weights(model, args.init_checkpoint, args.device)
        print("optimizer/scheduler/epoch counter starting fresh from epoch 1 "
              "(pretrained weights only).")

    if start_epoch > args.epochs:
        print(f"checkpoint already reached epoch {start_epoch - 1} >= --epochs "
              f"{args.epochs}; nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, optimizer, args.device,
                                 train=True, epoch=epoch, total_epochs=args.epochs)
        scheduler.step()

        msg = (f"epoch {epoch:3d}/{args.epochs} | "
               f"train_loss={train_stats['loss']:.6f} "
               f"(err={train_stats['err']:.6f}, sym={train_stats['sym']:.6f}) | "
               f"lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loader is not None:
            val_stats = run_epoch(model, val_loader, criterion, optimizer, args.device,
                                   train=False, epoch=epoch, total_epochs=args.epochs)
            msg += f" | val_loss={val_stats['loss']:.6f}"
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]

            val_loss_history.append(val_stats["loss"])
            if len(val_loss_history) > args.best_checkpoint_window:
                val_loss_history.pop(0)
            smoothed_val_loss = sum(val_loss_history) / len(val_loss_history)
            msg += f" | val_loss_avg{len(val_loss_history)}={smoothed_val_loss:.6f}"

            if smoothed_val_loss < best_smoothed_val_loss:
                best_smoothed_val_loss = smoothed_val_loss
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "odls_best.pt"))

        print(msg)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_smoothed_val_loss": best_smoothed_val_loss,
            "val_loss_history": val_loss_history,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }, os.path.join(args.checkpoint_dir, "latest.pt"))

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "odls_final.pt"))


if __name__ == "__main__":
    main()
