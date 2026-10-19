"""
Chooses L+S's two settings (lam_L, lam_S) on the VALIDATION split.

L+S has no training, but it has two regularisation weights. Picking them on
the test set would flatter it, so -- like the networks' best.pt, which is
chosen by validation loss -- they are chosen on the validation slices:
every combination in the grid is run on every validation slice, and the
pair with the lowest mean per-slice RLNE wins (RLNE is the ODLS paper's
main error measure).

If the winner sits on the EDGE of the grid, the true optimum may lie outside
it, so the grid is extended in that direction (halving / doubling the extreme
value) and the new combinations are tried -- up to --max-widen times -- until
the best value is inside the grid. Combinations already tried are not re-run.

Output: <out-root>/lps_params_af<AF>.json  ->  {"best": {...}, "grid": [...]}
which evaluate_plus.py --method lps --lps-params <that file> then uses on the
test split.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
from tqdm.auto import tqdm

import paths  # noqa: F401
from data import fixed_slice_mask, load_slice, read_manifest, select_split
from evaluate import sos_images
from lps import lps_magnitude, lps_reconstruct
from metrics import psnr, rlne


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune L+S on the validation split")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--af", type=float, required=True)
    p.add_argument("--lam-l", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.04])
    p.add_argument("--lam-s", type=float, nargs="+", default=[0.000625, 0.00125, 0.0025, 0.005, 0.01])
    p.add_argument("--max-widen", type=int, default=3,
                   help="how many times to extend the grid when the best value is on its edge")
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-slices", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    rows = select_split(read_manifest(args.data_dir), "val", args.max_slices)

    # Load each validation slice once (ref image + undersampled input), then try every setting.
    slices = []
    for row in rows:
        ref = load_slice(args.data_dir, row)
        mask = fixed_slice_mask(row, args.af, args.seed)
        z = torch.from_numpy((ref * mask[None, :, None, :]).astype(np.complex64)).to(args.device)
        slices.append((torch.from_numpy(sos_images(ref)), z, torch.from_numpy(mask).to(args.device)))

    def score(lam_l, lam_s):
        rl, ps, its = [], [], []
        for ref_mag, z, mask in slices:
            m, maps, info = lps_reconstruct(z, mask, lam_l, lam_s, args.n_iter)
            est = torch.from_numpy(lps_magnitude(m, maps))
            rl.append(rlne(ref_mag, est).mean().item())
            ps.append(psnr(ref_mag, est).mean().item())
            its.append(info["iterations"])
        return {"lam_l": lam_l, "lam_s": lam_s, "n_iter": args.n_iter, "val_RLNE": float(np.mean(rl)),
                "val_PSNR": float(np.mean(ps)), "mean_iterations": float(np.mean(its))}

    lam_l_vals, lam_s_vals = sorted(args.lam_l), sorted(args.lam_s)
    tried = {}                                     # (lam_l, lam_s) -> result
    t0 = time.time()
    for round_ in range(args.max_widen + 1):
        todo = [c for c in itertools.product(lam_l_vals, lam_s_vals) if c not in tried]
        for lam_l, lam_s in tqdm(todo, desc=f"L+S grid AF {args.af:g}, round {round_ + 1}"):
            tried[(lam_l, lam_s)] = score(lam_l, lam_s)
        best = min(tried.values(), key=lambda g: g["val_RLNE"])
        on_edge = {"l_low": best["lam_l"] == lam_l_vals[0], "l_high": best["lam_l"] == lam_l_vals[-1],
                   "s_low": best["lam_s"] == lam_s_vals[0], "s_high": best["lam_s"] == lam_s_vals[-1]}
        edge = any(on_edge.values())
        if not edge or round_ == args.max_widen:
            break
        # extend the grid towards the edge the best value sits on
        if on_edge["l_low"]:
            lam_l_vals.insert(0, lam_l_vals[0] / 2)
        if on_edge["l_high"]:
            lam_l_vals.append(lam_l_vals[-1] * 2)
        if on_edge["s_low"]:
            lam_s_vals.insert(0, lam_s_vals[0] / 2)
        if on_edge["s_high"]:
            lam_s_vals.append(lam_s_vals[-1] * 2)
        print(f"best (lam_L={best['lam_l']}, lam_S={best['lam_s']}) is on the grid edge -- widening")
    grid = sorted(tried.values(), key=lambda g: (g["lam_l"], g["lam_s"]))

    os.makedirs(args.out_root, exist_ok=True)
    path = os.path.join(args.out_root, f"lps_params_af{args.af:g}.json")
    with open(path, "w") as f:
        json.dump({"version": 2, "af": args.af, "split": "val", "n_slices": len(rows), "best": best,
                   "grid": grid, "lam_l_values": lam_l_vals, "lam_s_values": lam_s_vals,
                   "best_on_grid_edge": edge, "rounds": round_ + 1, "seconds": time.time() - t0}, f, indent=2)

    print(f"best on validation: lam_L={best['lam_l']} lam_S={best['lam_s']} | "
          f"RLNE {best['val_RLNE'] * 100:.2f}% | PSNR {best['val_PSNR']:.2f} dB")
    if edge:
        print(f"NOTE: still on the grid edge after {args.max_widen} widenings -- consider a wider --lam-l / --lam-s")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
