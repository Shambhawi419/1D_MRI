"""
Streams OCMR fully-sampled cine files from the public S3 bucket, one file at
a time, and writes compact training-ready slices -- never more than one raw
file on local disk, and a hard cap on how much lands in --out-dir (Drive).

Per file:
  1. download the raw .h5 to --tmp-dir (Colab's local disk), verify its size
  2. read k-space with ocmr_io.read_ocmr (verified identical to OCMR's reader)
  3. per slice (averages averaged; kz / set / repetition must be singleton):
       a. inverse FFT along kx, crop the centre half -> removes OCMR's 2x
          readout oversampling (verified 2.0 on all 165 fs files) and gives
          the FE-decoupled hybrid data Z = Psi_FE^* Y of paper Eq. 1
       b. SVD coil compression to --n-virtual-coils (paper Sec. IV-A, ref [49])
       c. intensity normalisation
     Both (b) and (c) use only the centre ky lines that every mask samples in
     every frame (masks.center_lines), so the preprocessing uses nothing a
     real undersampled scan wouldn't have. This replaces ../odls's
     normalisation by the fully-sampled image, which leaked ground truth.
  4. save <out-dir>/slices/<name>.npy (complex64, (M, N, J, T)), update
     manifest.csv, delete the raw file
Safe to re-run at any time: finished slices are skipped, so a disconnect
just means re-running the cell. The train/val/test split (by file, so no
slice of one acquisition lands in two splits) is fixed in split.csv on the
first run and reused afterwards; --resplit-only rewrites it without
downloading anything.

Default selection (the plan agreed for this project): fully-sampled,
symmetric-echo, single-slice SAX + LAX series -- 128 files, ~36 GB raw,
~4.8 GB written. Asymmetric-echo files are always skipped (their readout is
partially sampled, which breaks the FE-decoupling assumption).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from masks import center_lines
from ocmr_io import read_ocmr
from transforms import np_ifftc

BASE_URL = "https://ocmr.s3.amazonaws.com"
ATTRIBUTES_URL = f"{BASE_URL}/ocmr_data_attributes.csv"
MANIFEST_FIELDS = ["name", "file", "slice", "split", "M", "N", "J", "T", "ky_valid_start",
                   "ky_valid_stop", "scale", "scanner", "view", "field_T", "coils_raw", "bytes"]


# ------------------------------------------------------------ selection ----

def load_attributes(out_dir: str) -> List[Dict]:
    path = os.path.join(out_dir, "ocmr_data_attributes.csv")
    if not os.path.exists(path):
        import requests
        resp = requests.get(ATTRIBUTES_URL, timeout=60)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("file name", "").strip()]


def select_files(attrs: List[Dict], views: List[str], include_stacks: bool) -> List[Dict]:
    out = [a for a in attrs
           if a["smp"] == "fs" and a["ech"] == "sym" and a["viw"].lower() in views
           and (include_stacks or a["sli"] == "ind")]
    return sorted(out, key=lambda a: a["file name"])


def make_split(files: List[Dict], args) -> Dict[str, str]:
    """By file, so no two slices of one acquisition land in different splits."""
    names = [a["file name"] for a in files]
    rng = np.random.default_rng(args.seed)
    if args.split_mode == "holdout-scanner":
        test = [a["file name"] for a in files if a["scn"] == args.holdout_scanner]
        rest = [n for n in names if n not in set(test)]
        n_val = int(round(args.val_frac * len(rest)))
    else:
        perm = [str(n) for n in rng.permutation(names)]
        n_test = int(round(args.test_frac * len(names)))
        test, rest = perm[:n_test], perm[n_test:]
        n_val = int(round(args.val_frac * len(names)))
    rest = [str(n) for n in rng.permutation(rest)]
    if len(names) >= 3:
        n_val = max(1, n_val)
    split = {n: "test" for n in test}
    split.update({n: "val" for n in rest[:n_val]})
    split.update({n: "train" for n in rest[n_val:]})
    return split


def read_split(out_dir: str) -> Optional[Dict[str, str]]:
    path = os.path.join(out_dir, "split.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return {r["file"]: r["split"] for r in csv.DictReader(f)}


def write_csv_atomic(path: str, fields: List[str], rows: List[Dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


# ----------------------------------------------------------- processing ----

def process_file(kdata: np.ndarray, hdr: Dict, n_virtual_coils: int) -> List[Dict]:
    """kdata: [kx, ky, kz, coil, phase, set, slice, rep, avg] -> list of
    {"slice", "hybrid" (M, N, J, T) complex64, "valid", "scale"}."""
    kx, ky, kz, n_coils, n_t, n_set, n_sl, n_rep, n_avg = kdata.shape
    if kz != 1 or n_set != 1 or n_rep != 1:
        raise ValueError(f"unsupported layout kz={kz} set={n_set} rep={n_rep} (expected 1/1/1)")
    if n_t < 3:
        raise ValueError(f"only {n_t} frames -- not a cine series")
    ratio = hdr["fov_enc_x"] / hdr["fov_rec_x"] if hdr["fov_rec_x"] else 2.0
    if abs(ratio - 2.0) > 0.05:
        raise ValueError(f"readout oversampling {ratio:.3f}, expected 2.0")

    out = []
    for s in range(n_sl):
        ks = kdata[:, :, 0, :, :, 0, s, 0, :].mean(axis=-1)  # (kx, ky, J, T), averages averaged

        # (a) FE inverse FFT + crop centre half: removes 2x readout oversampling.
        hybrid = np_ifftc(ks, axis=0)
        m = kx // 2
        hybrid = hybrid[kx // 2 - m // 2: kx // 2 - m // 2 + m]

        # Put the ky centre line at index N // 2 (true for every OCMR file;
        # zero-pad defensively otherwise so the centered FFTs stay correct).
        c = hdr["ky_center"] if hdr["ky_center"] >= 0 else ky // 2
        n_new = 2 * max(c, ky - c)
        pad_before = n_new // 2 - c
        if n_new != ky or pad_before:
            hybrid = np.pad(hybrid, ((0, 0), (pad_before, n_new - ky - pad_before), (0, 0), (0, 0)))
        valid = (pad_before, pad_before + ky)
        n = hybrid.shape[1]
        cal = center_lines(n, valid)

        # (b) SVD coil compression from calibration lines only.
        if n_coils > n_virtual_coils:
            calib = np.moveaxis(hybrid[:, cal], 2, -1).reshape(-1, n_coils)
            _, _, vh = np.linalg.svd(calib, full_matrices=False)
            hybrid = np.einsum("mnjt,jv->mnvt", hybrid, vh[:n_virtual_coils].conj().T)
        elif n_coils < n_virtual_coils:
            raise ValueError(f"only {n_coils} coils < --n-virtual-coils {n_virtual_coils}")

        # (c) normalise by the peak of the time-averaged low-resolution SoS
        # image built from the calibration lines.
        lowres = np.zeros_like(hybrid)
        lowres[:, cal] = hybrid[:, cal]
        img = np_ifftc(lowres, axis=1).mean(axis=-1)              # (M, N, J)
        scale = float(np.sqrt((np.abs(img) ** 2).sum(axis=-1)).max())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("degenerate normalisation scale")
        hybrid = (hybrid / scale).astype(np.complex64)
        out.append({"slice": s, "hybrid": np.ascontiguousarray(hybrid), "valid": valid, "scale": scale})
    return out


# ------------------------------------------------------------- download ----

def download(url: str, dst: str, retries: int = 4) -> None:
    import requests
    from tqdm.auto import tqdm
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                                desc=os.path.basename(dst), leave=False) as bar:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        bar.update(len(chunk))
            if total and done != total:
                raise IOError(f"incomplete download: {done} of {total} bytes")
            return
        except Exception as exc:
            print(f"  download attempt {attempt}/{retries} failed: {exc!r}")
            if os.path.exists(dst):
                os.remove(dst)
            if attempt == retries:
                raise
            time.sleep(10 * attempt)


def dir_bytes(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))


# ----------------------------------------------------------------- main ----

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stream + preprocess OCMR fully-sampled cine data")
    p.add_argument("--out-dir", required=True, help="Persistent output (a Google Drive folder on Colab).")
    p.add_argument("--tmp-dir", default="/content/ocmr_tmp",
                   help="Scratch space for the one raw file being processed (Colab local disk).")
    p.add_argument("--views", nargs="+", default=["sax", "lax"])
    p.add_argument("--include-stacks", action="store_true",
                   help="Also use multi-slice stacks (11 files, +33.5 GB to download, ~+4.5 GB written).")
    p.add_argument("--n-virtual-coils", type=int, default=8)
    p.add_argument("--max-gb", type=float, default=8.0,
                   help="Hard cap on <out-dir>/slices. Processing stops before exceeding it.")
    p.add_argument("--max-files", type=int, default=None,
                   help="Process at most this many new files in this run (for a quick trial).")
    p.add_argument("--split-mode", choices=["random", "holdout-scanner"], default="random")
    p.add_argument("--holdout-scanner", default="0_55freemax",
                   help="With --split-mode holdout-scanner: every file from this scanner becomes "
                        "the test set (the paper's 'mismatched reconstruction' experiment).")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resplit-only", action="store_true",
                   help="Recompute split.csv / the manifest's split column and exit (no downloads).")
    p.add_argument("--local-h5", nargs="+", default=None,
                   help="Process these local .h5 files instead of downloading (testing).")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    slices_dir = os.path.join(args.out_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    split_path = os.path.join(args.out_dir, "split.csv")

    if args.local_h5:
        files = [{"file name": os.path.basename(p), "scn": "local", "viw": "local", "_path": p}
                 for p in args.local_h5]
    else:
        files = select_files(load_attributes(args.out_dir), [v.lower() for v in args.views],
                             args.include_stacks)
    print(f"{len(files)} files selected")

    # The split is fixed on first use. Files added later (e.g. --include-stacks)
    # are split among themselves and appended; existing assignments never move.
    split = None if args.resplit_only else read_split(args.out_dir)
    if split is None:
        split = make_split(files, args)
    else:
        new = [a for a in files if a["file name"] not in split]
        if new:
            split.update(make_split(new, args))
    write_csv_atomic(split_path, ["file", "split"],
                     [{"file": k, "split": v} for k, v in sorted(split.items())])
    print(f"split ({split_path}): " + ", ".join(
        f"{s}={sum(v == s for v in split.values())}" for s in ("train", "val", "test")))

    manifest: List[Dict] = []
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as f:
            manifest = list(csv.DictReader(f))
    for row in manifest:
        row["split"] = split.get(row["file"], row["split"])
    write_csv_atomic(manifest_path, MANIFEST_FIELDS, manifest)
    if args.resplit_only:
        print("split updated; nothing downloaded")
        return

    done_files = {r["file"] for r in manifest}
    todo = [a for a in files if a["file name"] not in done_files]
    if args.max_files is not None:
        todo = todo[:args.max_files]
    budget = int(args.max_gb * 1e9)
    print(f"{len(done_files)} already done, {len(todo)} to go; "
          f"{dir_bytes(slices_dir) / 1e9:.2f} GB used of {args.max_gb} GB cap")

    failed_path = os.path.join(args.out_dir, "failed.txt")
    for i, a in enumerate(todo, 1):
        name = a["file name"]
        raw_path = a.get("_path") or os.path.join(args.tmp_dir, name)
        t0 = time.time()
        try:
            if "_path" not in a:
                download(f"{BASE_URL}/data/{name}", raw_path)
            kdata, hdr = read_ocmr(raw_path)
            results = process_file(kdata, hdr, args.n_virtual_coils)
            del kdata

            new_bytes = sum(r["hybrid"].nbytes for r in results)
            if dir_bytes(slices_dir) + new_bytes > budget:
                print(f"STOP: {name} would push {slices_dir} past --max-gb {args.max_gb}. "
                      f"Raise --max-gb to continue.")
                break

            for r in results:
                slice_name = f"{os.path.splitext(name)[0]}_s{r['slice']:02d}"
                dst = os.path.join(slices_dir, slice_name + ".npy")
                with open(dst + ".tmp", "wb") as f:
                    np.save(f, r["hybrid"])
                os.replace(dst + ".tmp", dst)
                M, N, J, T = r["hybrid"].shape
                manifest.append({
                    "name": slice_name, "file": name, "slice": r["slice"], "split": split[name],
                    "M": M, "N": N, "J": J, "T": T, "ky_valid_start": r["valid"][0],
                    "ky_valid_stop": r["valid"][1], "scale": f"{r['scale']:.8g}",
                    "scanner": a.get("scn", ""), "view": a.get("viw", ""), "field_T": hdr["field_T"],
                    "coils_raw": hdr["n_coils"], "bytes": r["hybrid"].nbytes,
                })
            write_csv_atomic(manifest_path, MANIFEST_FIELDS, manifest)
            print(f"[{i}/{len(todo)}] {name}: {len(results)} slice(s) {results[0]['hybrid'].shape} "
                  f"in {time.time() - t0:.0f}s | {dir_bytes(slices_dir) / 1e9:.2f} GB used")
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {name}: FAILED {exc!r} (logged, continuing)")
            with open(failed_path, "a") as f:
                f.write(f"{name}\t{exc!r}\n")
        finally:
            if "_path" not in a and os.path.exists(raw_path):
                os.remove(raw_path)

    counts = {s: sum(r["split"] == s for r in manifest) for s in ("train", "val", "test")}
    print(f"done: {len(manifest)} slices in manifest ({counts}), "
          f"{dir_bytes(slices_dir) / 1e9:.2f} GB in {slices_dir}")


if __name__ == "__main__":
    sys.exit(main())
