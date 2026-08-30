"""
Training script for ODLS, matching Sec. III-D's stated configuration:

  - Optimizer: Adam
  - Initial learning rate: 0.001, exponential decay of 0.99 (per epoch)
  - Batch size: 128
  - Epochs: 300
  - Weight init: Xavier (applied inside model.ODLS._xavier_init)
  - K = 10 unrolled phases
  - Each conv layer: 48 1D filters, kernel size 3

This script only wires the pieces together (model, loss, data, optimizer,
schedule); it expects the caller to supply fully-sampled k-space volumes
(the paper's in-vivo knee/brain datasets are not public). Not executed
here -- provided as a ready-to-run reference implementation.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ODLSHybridDataset
from losses import ODLSLoss
from model import ODLS
from fastmri_data import FastMRICorpdDataset, find_corpd_files


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ODLS (arXiv:2112.04721)")

    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--train-volumes", type=str, nargs="+",
                             help="Paths to .npy files, each holding a complex "
                                  "(n_slices, M, N, J) fully-sampled k-space volume.")
    data_group.add_argument("--fastmri-train-root", type=str,
                             help="Directory of fastMRI knee multicoil .h5 files "
                                  "(https://www.kaggle.com/datasets/arafatshovon/"
                                  "fastmri-knee-multicoil). Only CORPD_FBK volumes "
                                  "are used unless --fastmri-fat-suppressed is set.")

    p.add_argument("--val-volumes", type=str, nargs="+", default=None)
    p.add_argument("--fastmri-val-root", type=str, default=None)
    p.add_argument("--fastmri-fat-suppressed", action="store_true",
                    help="Use CORPDFS_FBK (fat-suppressed) instead of CORPD_FBK.")
    p.add_argument("--n-virtual-coils", type=int, default=8,
                    help="SVD coil-compression target for fastMRI inputs "
                         "(ignored for --train-volumes). Must equal --n-coils.")
    p.add_argument("--crop-fe-to", type=int, default=None,
                    help="Readout size after removing fastMRI's 2x oversampling "
                         "(default: half the raw readout size).")

    p.add_argument("--mask-type", type=str, default="cartesian",
                    choices=["cartesian", "uniform", "partial_fourier"])
    p.add_argument("--af", type=float, default=4.0)
    p.add_argument("--n-coils", type=int, required=True)
    p.add_argument("--n-phases", type=int, default=10)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay", type=float, default=0.99)
    p.add_argument("--sym-weight", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--num-workers", type=int, default=4)
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


def run_epoch(model: ODLS, loader: DataLoader, criterion: ODLSLoss,
              optimizer, device: str, train: bool) -> dict:
    model.train(mode=train)
    total_loss, total_err, total_sym, n_batches = 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
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

    return {
        "loss": total_loss / max(n_batches, 1),
        "err": total_err / max(n_batches, 1),
        "sym": total_sym / max(n_batches, 1),
    }


def build_train_and_val_sets(args):
    """Returns (train_set, val_set_or_None), sourcing from either plain
    .npy volumes or a fastMRI CORPD_FBK/.h5 directory depending on which
    mutually-exclusive flag was given."""
    if args.fastmri_train_root:
        if args.n_virtual_coils != args.n_coils:
            raise ValueError(
                f"--n-virtual-coils ({args.n_virtual_coils}) must equal --n-coils "
                f"({args.n_coils}): the model's channel count is fixed at build time "
                f"and must match the coil-compressed data it receives."
            )
        train_files = find_corpd_files(args.fastmri_train_root, fat_suppressed=args.fastmri_fat_suppressed)
        if not train_files:
            raise ValueError(f"No matching CORPD{'FS' if args.fastmri_fat_suppressed else ''}_FBK "
                              f"files found under {args.fastmri_train_root}")
        train_set = FastMRICorpdDataset(
            train_files, mask_type=args.mask_type, af=args.af,
            n_virtual_coils=args.n_virtual_coils, crop_fe_to=args.crop_fe_to,
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


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_set, val_set = build_train_and_val_sets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    model = ODLS(n_coils=args.n_coils, n_phases=args.n_phases, width=args.width).to(args.device)
    criterion = ODLSLoss(n_coils=args.n_coils, sym_weight=args.sym_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, optimizer, args.device, train=True)
        scheduler.step()

        msg = (f"epoch {epoch:3d}/{args.epochs} | "
               f"train_loss={train_stats['loss']:.6f} "
               f"(err={train_stats['err']:.6f}, sym={train_stats['sym']:.6f}) | "
               f"lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loader is not None:
            val_stats = run_epoch(model, val_loader, criterion, optimizer, args.device, train=False)
            msg += f" | val_loss={val_stats['loss']:.6f}"
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "odls_best.pt"))

        print(msg)

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "odls_final.pt"))


if __name__ == "__main__":
    main()
