"""
Builds the final comparison table + significance tests from all
results_plus.json files under --out-root.

For each (split, AF):
  * one row per method: mean +/- std OVER SLICES of RLNE, PSNR, global SSIM,
    windowed SSIM, plus time per slice, parameters and peak GPU memory.
    (Per slice, not per frame: the frames of one cine are strongly correlated,
    so the slice is the honest unit for a spread or a test.)
  * Wilcoxon signed-rank tests, dynamic ODLS vs every other method, on the
    per-slice values (paired: the same slices, the same masks). Two-sided.
    With 13 test slices the smallest possible p-value is 2/2^13 = 0.00024,
    reached when one method wins on every slice.

Writes <out-root>/summary_<split>_af<AF>.md and prints the same tables.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon

ORDER = ["zero-filled", "L+S", "frame-by-frame ODLS", "dynamic ODLS"]
REFERENCE = "dynamic ODLS"
METRICS = [("RLNE", "RLNE (%)", 100, "lower"), ("PSNR", "PSNR (dB)", 1, "higher"),
           ("SSIM_global", "SSIM global (%)", 100, "higher"), ("SSIM_windowed", "SSIM windowed (%)", 100, "higher")]


def load_results(out_root: str) -> Dict:
    groups = defaultdict(dict)
    for path in glob.glob(os.path.join(out_root, "*", "*", "results_plus.json")):
        with open(path) as f:
            r = json.load(f)
        groups[(r["split"], r["af"])][r["label"]] = r
    return groups


def table(results: Dict[str, Dict]) -> List[str]:
    lines = ["| Method | " + " | ".join(h for _, h, _, _ in METRICS) + " | Time / slice | Params | Peak GPU mem |",
             "|---" * (len(METRICS) + 4) + "|"]
    for label in [m for m in ORDER if m in results] + [m for m in results if m not in ORDER]:
        r = results[label]
        cells = [f"{r['summary_per_slice'][k]['mean'] * scale:.2f} ± {r['summary_per_slice'][k]['std'] * scale:.2f}"
                 for k, _, scale, _ in METRICS]
        params = f"{r['n_params'] / 1e6:.2f} M" if r.get("n_params") else "–"
        mem = f"{r['peak_gpu_mem_mb']:.0f} MB" if r.get("peak_gpu_mem_mb") else "–"
        lines.append(f"| {label} | " + " | ".join(cells) +
                     f" | {r['time_per_slice_s']['median']:.2f} s | {params} | {mem} |")
    return lines


def significance(results: Dict[str, Dict]) -> List[str]:
    if REFERENCE not in results:
        return []
    ref = {s["name"]: s for s in results[REFERENCE]["per_slice"]}
    lines = [f"Wilcoxon signed-rank, {REFERENCE} vs each method (per-slice, two-sided):", "",
             "| vs | " + " | ".join(f"{k}: wins, p" for k, _, _, _ in METRICS) + " |",
             "|---" * (len(METRICS) + 1) + "|"]
    for label in [m for m in ORDER if m in results and m != REFERENCE]:
        other = {s["name"]: s for s in results[label]["per_slice"]}
        names = sorted(set(ref) & set(other))
        cells = []
        for k, _, _, better in METRICS:
            a = np.array([ref[n][k] for n in names])
            b = np.array([other[n][k] for n in names])
            wins = int(np.sum(a < b) if better == "lower" else np.sum(a > b))
            p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
            cells.append(f"{wins}/{len(names)}, p={p:.2g}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description="Comparison table + significance tests")
    p.add_argument("--out-root", required=True)
    args = p.parse_args(argv)

    groups = load_results(args.out_root)
    if not groups:
        raise SystemExit(f"no results_plus.json under {args.out_root} -- run evaluate_plus.py first")
    for (split, af), results in sorted(groups.items()):
        n = {r["n_slices"] for r in results.values()}
        lines = [f"## {split} set, AF {af:g} ({', '.join(map(str, sorted(n)))} slices)", ""]
        lines += table(results) + [""] + significance(results) + [""]
        text = "\n".join(lines)
        print(text)
        with open(os.path.join(args.out_root, f"summary_{split}_af{af:g}.md"), "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
