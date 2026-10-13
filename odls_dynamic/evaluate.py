"""
Evaluates a dynamic ODLS checkpoint (or only the zero-filled baseline) on a
split of the preprocessed OCMR data.

Per slice (paper Fig. 6, per frame): one fixed k-t mask shared by all M rows
(as in a real acquisition, different for every slice) -> reconstruct all rows
-> PE inverse FFT -> root-sum-of-squares coil combination -> RLNE / PSNR /
SSIM (Supplement S1) for every frame. Reported as mean +/- std over all
frames of all slices, plus per scanner and per view, always next to the
zero-filled baseline.

The model is rebuilt from the checkpoint itself (no architecture flags), and
--af defaults to the AF the checkpoint was trained at. Results go to a JSON
file next to the checkpoint (on Drive), plus a few example reconstructions.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from checkpoint_utils import build_model_from_checkpoint
from data import fixed_slice_mask, load_slice, read_manifest, select_split
from masks import effective_af
from metrics import psnr, rlne, ssim
from transforms import np_ifftc


def sos_images(hybrid: np.ndarray) -> np.ndarray:
    """(M, N, J, T) hybrid -> (T, M, N) root-sum-of-squares magnitude."""
    img = np_ifftc(hybrid, axis=1)
    return np.transpose(np.sqrt((np.abs(img) ** 2).sum(axis=2)), (2, 0, 1)).astype(np.float32)


def frame_metrics(ref: np.ndarray, est: np.ndarray) -> Dict[str, np.ndarray]:
    r, e = torch.from_numpy(ref), torch.from_numpy(est)
    return {"RLNE": rlne(r, e).numpy(), "PSNR": psnr(r, e).numpy(), "SSIM": ssim(r, e).numpy()}


@torch.no_grad()
def reconstruct_slice(model, z: np.ndarray, mask: np.ndarray, device: str, rows_per_batch: int) -> np.ndarray:
    """z: (M, N, J, T) zero-filled hybrid, mask: (N, T) -> (M, N, J, T) reconstruction."""
    M, N, J, T = z.shape
    zc = np.transpose(z, (0, 2, 1, 3))                                     # (M, J, N, T)
    zc = np.concatenate([zc.real, zc.imag], axis=1).astype(np.float32)     # (M, 2J, N, T)
    mask_t = torch.from_numpy(mask[None, None].astype(np.float32)).to(device)
    out = np.empty_like(zc)
    for i in range(0, M, rows_per_batch):
        zb = torch.from_numpy(zc[i:i + rows_per_batch]).to(device)
        with torch.autocast("cuda" if device.startswith("cuda") else "cpu", enabled=device.startswith("cuda")):
            e_preds, _ = model(zb, mask_t.expand(zb.shape[0], -1, -1, -1), need_sym=False)
        out[i:i + rows_per_batch] = e_preds[-1].float().cpu().numpy()
    est = out[:, :J] + 1j * out[:, J:]
    return np.transpose(est, (0, 2, 1, 3)).astype(np.complex64)


def summarise(values: Dict[str, List[float]]) -> Dict[str, float]:
    out = {}
    for k, v in values.items():
        v = np.asarray(v, dtype=np.float64)
        out[f"{k}_mean"], out[f"{k}_std"] = float(v.mean()), float(v.std())
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate dynamic ODLS on OCMR")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="best.pt / final.pt / latest.pt. Omit to score only the zero-filled baseline.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--af", type=float, default=None, help="Default: the AF the checkpoint was trained at.")
    p.add_argument("--seed", type=int, default=0, help="Mask seed (same default as validation).")
    p.add_argument("--rows-per-batch", type=int, default=32)
    p.add_argument("--max-slices", type=int, default=None)
    p.add_argument("--save-examples", type=int, default=3,
                   help="Save magnitude ref / zero-filled / recon of this many slices (float16 .npz).")
    p.add_argument("--out-dir", default=None, help="Default: next to the checkpoint.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    model, ckpt = None, {}
    if args.checkpoint:
        model, ckpt = build_model_from_checkpoint(args.checkpoint, args.device)
        model.eval()
    af = args.af if args.af is not None else float(ckpt.get("args", {}).get("af", 0) or 0)
    if not af:
        raise SystemExit("--af is required when the checkpoint doesn't record one (or with no checkpoint)")

    tag = f"eval_{args.split}_af{af:g}"
    out_dir = args.out_dir or (os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), tag)
                               if args.checkpoint else os.path.join(args.data_dir, tag + "_zerofilled"))
    os.makedirs(out_dir, exist_ok=True)

    rows = select_split(read_manifest(args.data_dir), args.split, args.max_slices)
    methods = ["zero_filled"] + (["odls_dynamic"] if model is not None else [])
    allf = {m: defaultdict(list) for m in methods}
    groups = {m: defaultdict(lambda: defaultdict(list)) for m in methods}
    per_slice = []
    afs = []

    for i, row in enumerate(tqdm(rows, desc=f"{args.split} AF={af:g}")):
        ref = load_slice(args.data_dir, row)
        mask = fixed_slice_mask(row, af, args.seed)
        afs.append(effective_af(mask, (row["ky_valid_start"], row["ky_valid_stop"])))
        z = (ref * mask[None, :, None, :]).astype(np.complex64)

        ref_mag = sos_images(ref)
        mags = {"zero_filled": sos_images(z)}
        if model is not None:
            mags["odls_dynamic"] = sos_images(reconstruct_slice(model, z, mask, args.device, args.rows_per_batch))

        entry = {"name": row["name"], "scanner": row["scanner"], "view": row["view"], "T": row["T"]}
        for m, mag in mags.items():
            fm = frame_metrics(ref_mag, mag)
            for k, v in fm.items():
                allf[m][k].extend(v.tolist())
                groups[m][f"scanner={row['scanner']}"][k].extend(v.tolist())
                groups[m][f"view={row['view']}"][k].extend(v.tolist())
                entry[f"{m}_{k}"] = float(np.mean(v))
        per_slice.append(entry)

        if i < args.save_examples:
            np.savez_compressed(os.path.join(out_dir, f"{row['name']}.npz"), ref=ref_mag.astype(np.float16),
                                mask=mask.astype(np.uint8), **{m: v.astype(np.float16) for m, v in mags.items()})

    results = {
        "checkpoint": args.checkpoint, "checkpoint_epoch": ckpt.get("epoch"), "model_config": ckpt.get("model_config"),
        "split": args.split, "af_requested": af, "af_measured_mean": float(np.mean(afs)), "mask_seed": args.seed,
        "n_slices": len(rows), "n_frames": len(allf["zero_filled"]["RLNE"]),
        "overall": {m: summarise(allf[m]) for m in methods},
        "groups": {m: {g: summarise(v) for g, v in sorted(groups[m].items())} for m in methods},
        "per_slice": per_slice,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{args.split}: {len(rows)} slices, {results['n_frames']} frames, AF {af:g} "
          f"(measured {results['af_measured_mean']:.2f})")
    for m in methods:
        o = results["overall"][m]
        print(f"  {m:13s} RLNE {o['RLNE_mean'] * 100:6.2f} +/- {o['RLNE_std'] * 100:5.2f} (x1e-2) | "
              f"PSNR {o['PSNR_mean']:6.2f} +/- {o['PSNR_std']:5.2f} dB | "
              f"SSIM {o['SSIM_mean'] * 100:6.2f} +/- {o['SSIM_std'] * 100:5.2f} (x1e-2)")
    print(f"saved {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
