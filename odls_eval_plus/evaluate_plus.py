"""
Scores ONE method on one split at one AF, with the extended metrics:

    RLNE, PSNR, SSIM (the paper's global definition -- same as odls_dynamic),
    SSIM (standard windowed definition -- see ssim_windowed.py),
    reconstruction time per slice, and for networks: parameters + peak GPU memory.

Methods:
    --method model        a trained ODLS checkpoint (dynamic or frame-by-frame)
    --method zero_filled  inverse FFT of the undersampled data (no reconstruction)
    --method lps          L+S, the classical low-rank + sparse method (lps.py)

Every method sees EXACTLY the same input as odls_dynamic/evaluate.py: the same
slices, and the same fixed k-t mask per slice (data.fixed_slice_mask, keyed on
the slice name, AF and seed). So all numbers are directly comparable.

Output: <out-root>/<split>_af<AF>/<name>/results_plus.json, with metrics per
frame AND per slice (per-slice values are what compare_plus.py uses for the
statistics), plus a few example reconstructions as .npz for figures.
Nothing is written into the training run folders.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm

import paths  # noqa: F401  (makes ../odls_dynamic importable)
from checkpoint_utils import build_model_from_checkpoint
from data import fixed_slice_mask, load_slice, read_manifest, select_split
from evaluate import reconstruct_slice, sos_images
from lps import lps_magnitude, lps_reconstruct
from masks import effective_af
from metrics import psnr, rlne, ssim
from ssim_windowed import ssim_windowed

METRICS = ["RLNE", "PSNR", "SSIM_global", "SSIM_windowed"]


def frame_metrics(ref: np.ndarray, est: np.ndarray) -> Dict[str, np.ndarray]:
    """ref, est: (T, M, N) magnitude images -> one value per frame for each metric."""
    r, e = torch.from_numpy(ref), torch.from_numpy(est)
    return {"RLNE": rlne(r, e).numpy(), "PSNR": psnr(r, e).numpy(),
            "SSIM_global": ssim(r, e).numpy(), "SSIM_windowed": ssim_windowed(ref, est)}


def mean_std(values: List[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std())}


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extended evaluation of one method")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-root", required=True, help="e.g. <Drive>/odls_dynamic/eval_plus")
    p.add_argument("--method", required=True, choices=["model", "zero_filled", "lps"])
    p.add_argument("--checkpoint", default=None, help="best.pt of a training run (--method model)")
    p.add_argument("--af", type=float, default=None, help="required unless the checkpoint records it")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--name", default=None, help="output sub-folder; default: run name / method")
    p.add_argument("--seed", type=int, default=0, help="mask seed (same default as odls_dynamic)")
    p.add_argument("--max-slices", type=int, default=None)
    p.add_argument("--rows-per-batch", type=int, default=32)
    p.add_argument("--lps-params", default=None, help="JSON written by tune_lps.py (lam_l, lam_s, n_iter)")
    p.add_argument("--lps-lam-l", type=float, default=0.01)
    p.add_argument("--lps-lam-s", type=float, default=0.01)
    p.add_argument("--lps-iters", type=int, default=50)
    p.add_argument("--save-examples", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None) -> Dict:
    args = build_argparser().parse_args(argv)
    device = args.device

    # ---- what are we evaluating? ---------------------------------------------
    model, ckpt, label, lps_cfg = None, {}, None, None
    if args.method == "model":
        if not args.checkpoint:
            raise SystemExit("--method model needs --checkpoint")
        model, ckpt = build_model_from_checkpoint(args.checkpoint, device)
        model.eval()
        kernel_t = ckpt["model_config"]["kernel_t"]
        label = "dynamic ODLS" if kernel_t > 1 else "frame-by-frame ODLS"
        name = args.name or os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    elif args.method == "zero_filled":
        label, name = "zero-filled", args.name or "zero_filled"
    else:
        lps_cfg = {"lam_l": args.lps_lam_l, "lam_s": args.lps_lam_s, "n_iter": args.lps_iters}
        if args.lps_params:
            with open(args.lps_params) as f:
                lps_cfg.update({k: v for k, v in json.load(f)["best"].items() if k in lps_cfg})
        label, name = "L+S", args.name or "lps"

    af = args.af if args.af is not None else float(ckpt.get("args", {}).get("af", 0) or 0)
    if not af:
        raise SystemExit("--af is required for this method")
    out_dir = os.path.join(args.out_root, f"{args.split}_af{af:g}", name)
    os.makedirs(out_dir, exist_ok=True)

    rows = select_split(read_manifest(args.data_dir), args.split, args.max_slices)
    if model is not None and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    per_frame = {k: [] for k in METRICS}
    per_slice, times, afs, lps_info = [], [], [], []
    for i, row in enumerate(tqdm(rows, desc=f"{label} | {args.split} AF {af:g}")):
        ref = load_slice(args.data_dir, row)                      # (M, N, J, T) fully sampled
        mask = fixed_slice_mask(row, af, args.seed)                # same mask as odls_dynamic/evaluate.py
        afs.append(effective_af(mask, (row["ky_valid_start"], row["ky_valid_stop"])))
        z = (ref * mask[None, :, None, :]).astype(np.complex64)   # what the scanner would measure
        ref_mag = sos_images(ref)                                  # (T, M, N)

        sync(device)
        t0 = time.perf_counter()
        if args.method == "model":
            est_mag = sos_images(reconstruct_slice(model, z, mask, device, args.rows_per_batch))
        elif args.method == "zero_filled":
            est_mag = sos_images(z)
        else:
            zt, mt = torch.from_numpy(z).to(device), torch.from_numpy(mask).to(device)
            m, maps, info = lps_reconstruct(zt, mt, lps_cfg["lam_l"], lps_cfg["lam_s"], lps_cfg["n_iter"])
            est_mag = lps_magnitude(m, maps)
            lps_info.append(info)
        sync(device)
        times.append(time.perf_counter() - t0)

        fm = frame_metrics(ref_mag, est_mag)
        entry = {"name": row["name"], "scanner": row["scanner"], "view": row["view"], "T": row["T"],
                 "time_s": times[-1]}
        for k in METRICS:
            per_frame[k].extend(fm[k].tolist())
            entry[k] = float(np.mean(fm[k]))
        per_slice.append(entry)

        if i < args.save_examples:
            np.savez_compressed(os.path.join(out_dir, f"{row['name']}.npz"), ref=ref_mag.astype(np.float16),
                                est=est_mag.astype(np.float16), mask=mask.astype(np.uint8))

    results = {
        "method": args.method, "label": label, "name": name, "split": args.split, "af": af,
        "af_measured_mean": float(np.mean(afs)), "mask_seed": args.seed, "n_slices": len(rows),
        "n_frames": len(per_frame["RLNE"]), "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"), "model_config": ckpt.get("model_config"),
        "n_params": sum(p.numel() for p in model.parameters()) if model is not None else None,
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                            if model is not None and device.startswith("cuda") else None),
        "device": device,
        # the first slice includes one-off start-up costs (cuDNN tuning), so the median is reported
        "time_per_slice_s": {"median": float(np.median(times)), "mean": float(np.mean(times))},
        "lps_params": lps_cfg,
        "lps_iterations_mean": float(np.mean([d["iterations"] for d in lps_info])) if lps_info else None,
        "summary_per_slice": {k: mean_std([s[k] for s in per_slice]) for k in METRICS},
        "summary_per_frame": {k: mean_std(per_frame[k]) for k in METRICS},
        "per_slice": per_slice,
    }
    with open(os.path.join(out_dir, "results_plus.json"), "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary_per_slice"]
    print(f"{label} | {args.split} AF {af:g} | {len(rows)} slices | "
          f"RLNE {s['RLNE']['mean'] * 100:.2f}% | PSNR {s['PSNR']['mean']:.2f} dB | "
          f"SSIM global {s['SSIM_global']['mean'] * 100:.2f}% | SSIM windowed {s['SSIM_windowed']['mean'] * 100:.2f}% | "
          f"{results['time_per_slice_s']['median']:.2f} s/slice")
    print(f"saved {os.path.join(out_dir, 'results_plus.json')}")
    return results


if __name__ == "__main__":
    main()
