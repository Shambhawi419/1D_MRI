"""
Trains dynamic ODLS on preprocessed OCMR data, built to survive free-Colab
disconnects without losing more than a few minutes of work.

Paper settings kept (Sec. III-D): Adam, lr 1e-3 with exponential decay 0.99
per epoch, Xavier init, K = 10, 48 filters, sym_weight 0.01. Changed for
dynamic data on a T4: batch size 32 rows (each row is now N x T, ~20x the
paper's 1D sample) and mixed precision (FFTs stay fp32).

Resume: re-run the exact same command. train.py finds <checkpoint-dir>/
latest.pt (or the fallbacks, see checkpoint_utils.py) and restores model,
optimizer, scheduler, AMP scaler, every RNG, the epoch AND the batch
position inside it -- batch order is a pure function of (seed, epoch), so
the run continues exactly where it stopped. A checkpoint is written every
--save-every-steps steps and at every epoch end; --time-limit-min stops
cleanly (with a save) before Colab's session limit.

Other workflows starting from these checkpoints:
  --init-checkpoint best.pt   new run initialised from trained weights
                              (fine-tune on another AF, more data, ...)
  evaluate.py / checkpoint_utils.build_model_from_checkpoint
                              rebuild the model from the file alone
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from checkpoint_utils import (BEST, FINAL, CheckpointUnreadable, atomic_save, load_latest, load_model_weights,
                              restore_rng_state, rng_state, save_latest, weights_checkpoint)
from data import OCMRRowDataset, SliceBatchSampler, memory_balanced_batch_sizes
from losses import ODLSLoss
from model import ODLSDynamic

HISTORY_FIELDS = ["epoch", "train_loss", "train_err", "train_sym", "val_loss", "val_loss_smoothed", "lr",
                  "epoch_minutes", "data_wait_pct", "nonfinite_skipped", "lambda", "eta1", "eta2"]
EMPTY_PARTIAL = {"loss": 0.0, "err": 0.0, "sym": 0.0, "n": 0, "skipped": 0, "seconds": 0.0,
                 "data_s": 0.0, "compute_s": 0.0}


def optional_float(s: str) -> Optional[float]:
    return None if s.lower() in ("none", "null", "") else float(s)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train dynamic ODLS on preprocessed OCMR")
    p.add_argument("--data-dir", required=True, help="preprocess_ocmr.py output (ideally a local-disk copy).")
    p.add_argument("--checkpoint-dir", required=True, help="Persistent run folder (on Google Drive).")
    p.add_argument("--af", type=float, default=4.0)
    # model (stored in every checkpoint; a resume refuses to change these)
    p.add_argument("--n-phases", type=int, default=10)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--kernel-t", type=int, default=3,
                   help="Temporal kernel size. 1 = frame-by-frame static baseline, same pipeline.")
    p.add_argument("--sparse-domain", choices=["yt", "yf"], default="yt")
    p.add_argument("--ds-residual", action=argparse.BooleanOptionalAction, default=True,
                   help="Sparse module as x + Psi3(soft(Psi2 x)) - Psi3(Psi2 x) (default): identical to "
                        "Eq. 14 once Psi3 o Psi2 = I, but starts as the identity. --no-ds-residual = "
                        "Eq. 14 literally.")
    p.add_argument("--max-threshold", type=optional_float, default=None,
                   help="Optional cap on each phase's soft-threshold (default: none, as in the paper).")
    # optimisation
    p.add_argument("--batch-size", type=int, default=32,
                   help="Rows per batch for a median-size (144 ky x 22 frames) slice; scaled per slice "
                        "so GPU memory stays ~constant (see data.memory_balanced_batch_sizes).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay", type=float, default=0.99)
    p.add_argument("--sym-weight", type=float, default=0.01)
    p.add_argument("--grad-clip", type=optional_float, default=None,
                   help="Optional max gradient norm (default: off, as in the paper).")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Mixed precision on CUDA (default on; --no-amp to disable).")
    p.add_argument("--seed", type=int, default=0)
    # checkpointing / session management
    p.add_argument("--save-every-steps", type=int, default=200)
    p.add_argument("--time-limit-min", type=optional_float, default=None,
                   help="Stop cleanly (with a checkpoint) after this many minutes of training.")
    p.add_argument("--best-window", type=int, default=5,
                   help="best.pt tracks the trailing mean of val loss over this many epochs.")
    p.add_argument("--init-checkpoint", default=None,
                   help="Start a NEW run from these weights (fresh optimizer, epoch 1). Ignored when "
                        "--checkpoint-dir already holds a run to resume.")
    p.add_argument("--no-resume", action="store_true",
                   help="Start over even if --checkpoint-dir holds a run (its files are moved to "
                        "previous_<time>/ inside the folder, not deleted).")
    # data loading
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-slices", type=int, default=None, help="Debug: use only the first N slices.")
    p.add_argument("--max-val-slices", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--stop-after-steps", type=int, default=None, help=argparse.SUPPRESS)  # simulate a crash
    return p


def model_config_from_args(args, n_coils: int) -> Dict:
    return dict(n_coils=n_coils, n_phases=args.n_phases, width=args.width, kernel_t=args.kernel_t,
                sparse_domain=args.sparse_domain, max_threshold=args.max_threshold,
                ds_residual=args.ds_residual)


def log_line(checkpoint_dir: str, msg: str) -> None:
    print(msg)
    with open(os.path.join(checkpoint_dir, "run_log.txt"), "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}\n")


def write_history(checkpoint_dir: str, history: List[Dict]) -> None:
    path = os.path.join(checkpoint_dir, "history.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        w.writerows(history)
    os.replace(path + ".tmp", path)


ARCHIVE_PREFIXES = ("previous_", "incompatible_")


def archive_run_files(checkpoint_dir: str, prefix: str) -> Optional[str]:
    """Moves everything a previous run left in checkpoint_dir (except earlier
    archives) into <checkpoint_dir>/<prefix><timestamp>/ so a fresh run can
    never overwrite another run's best.pt / history. Returns the archive path,
    or None if there was nothing to move."""
    names = [n for n in os.listdir(checkpoint_dir) if not n.startswith(ARCHIVE_PREFIXES)]
    if not names:
        return None
    archive = os.path.join(checkpoint_dir, f"{prefix}{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(archive)
    for name in names:
        shutil.move(os.path.join(checkpoint_dir, name), os.path.join(archive, name))
    return archive


def make_loader(dataset, sampler, args):
    workers = args.num_workers
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=args.device.startswith("cuda"))


def evaluate_loss(model, loader, criterion, device, amp: bool) -> float:
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            z, e_ref, mask = (batch[k].to(device, non_blocking=True) for k in ("z", "e_ref", "mask"))
            with torch.autocast(dev_type, enabled=amp):
                e_preds, sym_pairs = model(z, mask, need_sym=True)
            loss, _ = criterion(e_ref, e_preds, sym_pairs)
            total += loss.item()
            n += 1
    return total / max(n, 1)


def main(argv=None):
    args = build_argparser().parse_args(argv)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.manual_seed(args.seed)  # before the model is built: deterministic Xavier init
    device = args.device
    # cuDNN autotunes the fastest conv algorithm per input shape; OCMR has a
    # limited set of (ky, frames) sizes, so after the first epoch all are tuned.
    torch.backends.cudnn.benchmark = device.startswith("cuda")
    amp = bool(args.amp and device.startswith("cuda"))
    dev_type = "cuda" if device.startswith("cuda") else "cpu"

    train_set = OCMRRowDataset(args.data_dir, "train", args.af, fixed_mask=False, seed=args.seed,
                               max_slices=args.max_train_slices)
    val_set = OCMRRowDataset(args.data_dir, "val", args.af, fixed_mask=True, seed=args.seed,
                             max_slices=args.max_val_slices)
    n_coils = train_set.n_coils
    train_sampler = SliceBatchSampler(train_set.rows_of_slice,
                                      memory_balanced_batch_sizes(train_set.rows, args.batch_size),
                                      shuffle=True, seed=args.seed)
    val_sampler = SliceBatchSampler(val_set.rows_of_slice,
                                    memory_balanced_batch_sizes(val_set.rows, args.batch_size), shuffle=False)
    train_loader = make_loader(train_set, train_sampler, args)
    val_loader = make_loader(val_set, val_sampler, args)

    model_config = model_config_from_args(args, n_coils)
    model = ODLSDynamic(**model_config).to(device)
    criterion = ODLSLoss(sym_weight=args.sym_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ------------------------------------------------------------ state ----
    epoch, batch_in_epoch, global_step = 1, 0, 0
    partial = dict(EMPTY_PARTIAL)
    history: List[Dict] = []
    val_hist: List[float] = []
    best_smoothed = math.inf

    try:
        state = None if args.no_resume else load_latest(args.checkpoint_dir, device)
    except CheckpointUnreadable as exc:
        best = os.path.join(args.checkpoint_dir, BEST)
        sys.exit(f"ERROR: {exc} (most likely a Colab session ended while Google Drive was still uploading "
                 f"them). Nothing was changed{' -- best.pt is untouched' if os.path.exists(best) else ''}.\n"
                 f"Options:\n"
                 f"  * continue from the best model as a new run: use a new --checkpoint-dir / RUN_NAME and add\n"
                 f"      --init-checkpoint {best}\n"
                 f"  * start this run over: add --no-resume (the old files are moved to previous_<time>/, "
                 f"not deleted)")
    if state is not None and set(state["model_state_dict"]) != set(model.state_dict()):
        # Run folder from an older model.py: move it aside and start fresh instead of stopping.
        archive = archive_run_files(args.checkpoint_dir, "incompatible_")
        print(f"NOTE: {args.checkpoint_dir} held a run from an older version of model.py; moved it to "
              f"{archive} and starting a fresh run.")
        state = None
    if state is not None:
        if state["model_config"] != model_config:
            sys.exit(f"ERROR: {args.checkpoint_dir} holds a run with model config\n  {state['model_config']}\n"
                     f"but this command asks for\n  {model_config}\nUse a new --checkpoint-dir (optionally "
                     f"with --init-checkpoint) or the original flags.")
        if float(state["args"]["af"]) != args.af:
            sys.exit(f"ERROR: run was trained at AF={state['args']['af']}, this command says --af {args.af}. "
                     f"Use a new --checkpoint-dir with --init-checkpoint to fine-tune at another AF.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        restore_rng_state(state["rng"])
        epoch, batch_in_epoch, global_step = state["epoch"], state["batch_in_epoch"], state["global_step"]
        partial, history = {**EMPTY_PARTIAL, **state["partial"]}, state["history"]
        val_hist, best_smoothed = state["val_hist"], state["best_smoothed"]
        if state["args"]["batch_size"] != args.batch_size and batch_in_epoch:
            print("batch size changed: restarting the interrupted epoch from its first batch")
            batch_in_epoch = 0
            partial = dict(EMPTY_PARTIAL)
        log_line(args.checkpoint_dir, f"RESUME epoch {epoch} batch {batch_in_epoch} (step {global_step})")
    else:
        if args.init_checkpoint:  # load before archiving: it may live in this same folder
            load_model_weights(model, args.init_checkpoint, device)
        archive = archive_run_files(args.checkpoint_dir, "previous_")
        if archive:
            print(f"NOTE: moved files of an earlier run in {args.checkpoint_dir} to {archive} "
                  f"so this new run cannot overwrite them.")
        log_line(args.checkpoint_dir, f"START new run (init={args.init_checkpoint}) | model {model_config} | "
                                      f"{sum(p.numel() for p in model.parameters())} params | "
                                      f"{len(train_set)} train rows / {len(train_set.rows)} slices, "
                                      f"{len(val_set)} val rows / {len(val_set.rows)} slices")
        with open(os.path.join(args.checkpoint_dir, "run_config.json"), "w") as f:
            json.dump({"args": vars(args), "model_config": model_config}, f, indent=2)

    def checkpoint_now(reason: str, snapshot: bool = False):
        save_latest({
            "format_version": 1, "model_config": model_config, "args": vars(args),
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "rng": rng_state(), "epoch": epoch, "batch_in_epoch": batch_in_epoch,
            "global_step": global_step, "partial": partial, "history": history,
            "val_hist": val_hist, "best_smoothed": best_smoothed, "saved_reason": reason,
        }, args.checkpoint_dir, snapshot=snapshot)

    if epoch > args.epochs:
        final = os.path.join(args.checkpoint_dir, FINAL)
        if not os.path.exists(final):  # the session ended between the last epoch and writing final.pt
            atomic_save(weights_checkpoint(model, epoch=epoch - 1, args=vars(args)), final)
            log_line(args.checkpoint_dir, f"FINISHED {epoch - 1} epochs (final.pt written on re-run)")
        print(f"run already finished {epoch - 1} epochs; raise --epochs to continue training.")
        return

    session_start = time.time()
    steps_this_session = 0
    while epoch <= args.epochs:
        train_sampler.set_epoch(epoch, batch_in_epoch)
        model.train()
        epoch_t0 = time.time()
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", initial=batch_in_epoch,
                   total=batch_in_epoch + len(train_sampler))
        t_ready = time.time()
        for batch in bar:
            t_got = time.time()
            partial["data_s"] += t_got - t_ready      # time the GPU waited for the DataLoader
            z, e_ref, mask = (batch[k].to(device, non_blocking=True) for k in ("z", "e_ref", "mask"))
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(dev_type, enabled=amp):
                    e_preds, sym_pairs = model(z, mask, need_sym=True)
                loss, parts = criterion(e_ref, e_preds, sym_pairs)
                finite = bool(torch.isfinite(loss))
                if finite:
                    scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                del batch
                optimizer.zero_grad(set_to_none=True)
                checkpoint_now("out of memory")
                log_line(args.checkpoint_dir, f"STOP out of GPU memory at epoch {epoch} batch {batch_in_epoch} "
                                              f"(N={z.shape[-2]}, T={z.shape[-1]}, rows={z.shape[0]}). Re-run with "
                                              f"a smaller --batch-size (e.g. {max(4, args.batch_size // 2)}); "
                                              f"training resumes from here.")
                sys.exit(1)
            if finite:
                if args.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                partial["loss"] += loss.item()
                partial["err"] += parts["err"].item()
                partial["sym"] += parts["sym"].item()
                partial["n"] += 1
            else:
                partial["skipped"] += 1
                if partial["skipped"] > 50:
                    checkpoint_now("too many non-finite losses")
                    sys.exit("ERROR: >50 non-finite losses this epoch -- try --no-amp or --grad-clip 1.0")
            batch_in_epoch += 1
            global_step += 1
            steps_this_session += 1
            t_ready = time.time()                     # loss.item() above already synced the GPU
            partial["compute_s"] += t_ready - t_got
            wait = 100 * partial["data_s"] / max(partial["data_s"] + partial["compute_s"], 1e-9)
            bar.set_postfix(loss=f"{partial['loss'] / max(partial['n'], 1):.5f}", data_wait=f"{wait:.0f}%")

            if args.stop_after_steps and steps_this_session >= args.stop_after_steps:
                print("simulated crash (no save)")
                return
            if global_step % args.save_every_steps == 0:
                partial["seconds"] += time.time() - epoch_t0
                epoch_t0 = time.time()
                checkpoint_now("periodic")
            if args.time_limit_min and time.time() - session_start > 60 * args.time_limit_min:
                partial["seconds"] += time.time() - epoch_t0
                checkpoint_now("time limit")
                log_line(args.checkpoint_dir, f"STOP time limit at epoch {epoch} batch {batch_in_epoch}; "
                                              f"re-run the same command to continue")
                return
        bar.close()

        # ------------------------------------------------- end of epoch ----
        partial["seconds"] += time.time() - epoch_t0
        val_loss = evaluate_loss(model, val_loader, criterion, device, amp)
        val_hist = (val_hist + [val_loss])[-args.best_window:]
        smoothed = sum(val_hist) / len(val_hist)
        n = max(partial["n"], 1)
        scalars = model.scalar_report()
        history.append({
            "epoch": epoch, "train_loss": partial["loss"] / n, "train_err": partial["err"] / n,
            "train_sym": partial["sym"] / n, "val_loss": val_loss, "val_loss_smoothed": smoothed,
            "lr": scheduler.get_last_lr()[0], "epoch_minutes": round(partial["seconds"] / 60, 2),
            "data_wait_pct": round(100 * partial["data_s"] / max(partial["data_s"] + partial["compute_s"], 1e-9), 1),
            "nonfinite_skipped": partial["skipped"],
            **{k: json.dumps([round(v, 6) for v in vals]) for k, vals in scalars.items()},
        })
        print(f"epoch {epoch}: train {history[-1]['train_loss']:.6f} | val {val_loss:.6f} "
              f"(avg{len(val_hist)} {smoothed:.6f}) | lambda {min(scalars['lambda']):.4g}..{max(scalars['lambda']):.4g}")
        if smoothed < best_smoothed:
            best_smoothed = smoothed
            atomic_save(weights_checkpoint(model, epoch=epoch, val_loss=val_loss, val_loss_smoothed=smoothed,
                                           args=vars(args)), os.path.join(args.checkpoint_dir, BEST))
        scheduler.step()
        epoch += 1
        batch_in_epoch = 0
        partial = dict(EMPTY_PARTIAL)
        checkpoint_now("epoch end", snapshot=True)
        write_history(args.checkpoint_dir, history)

    atomic_save(weights_checkpoint(model, epoch=args.epochs, args=vars(args)),
                os.path.join(args.checkpoint_dir, FINAL))
    log_line(args.checkpoint_dir, f"FINISHED {args.epochs} epochs")


if __name__ == "__main__":
    main()
