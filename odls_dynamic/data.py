"""
Row-wise dynamic training samples from preprocessed OCMR slices.

ODLS's 1D learning scheme (paper Sec. III-A) extended to cine: kx is fully
sampled, so the FE inverse FFT still decouples a slice into M independent
rows. Each row is now a ky-t signal of shape (N, J, T) -- the same row
across all cardiac frames -- and every row of every slice is one sample.

Preprocessed layout (written by preprocess_ocmr.py):
    <data_dir>/manifest.csv
    <data_dir>/slices/<name>.npy    complex64 (M, N, J, T) hybrid data
                                    (FE-IFT'd, coil-compressed, normalised)

Batching: N and T vary between OCMR files (ky 104-180, 10-31 frames), so
SliceBatchSampler builds every batch from rows of a single slice. Nothing is
resized or interpolated, so no k-space or temporal content is altered.

Fixes relative to ../odls/data.py and fastmri_data.py:
  - Training masks are a pure function of (seed, epoch, sample index), not
    of a stateful RNG. The old dataset's np.random.Generator was copied
    verbatim into every DataLoader worker, so workers drew identical mask
    streams; here every sample gets its own mask regardless of which worker
    loads it, and a resumed run sees exactly the masks the uninterrupted run
    would have seen.
  - Validation/test masks are fixed per *slice* (all rows of a slice share
    one k-t mask, exactly like a real acquisition) and differ between
    slices (paper Sec. IV-A: each image owns a different mask).
"""

from __future__ import annotations

import csv
import os
import zlib
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from masks import kt_random_mask
from transforms import np_to_channels

MANIFEST = "manifest.csv"


def read_manifest(data_dir: str) -> List[Dict]:
    path = os.path.join(data_dir, MANIFEST)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run preprocess_ocmr.py first "
                                f"(or check that --data-dir points at its --out-dir).")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("M", "N", "J", "T", "slice", "ky_valid_start", "ky_valid_stop"):
            r[k] = int(r[k])
        r["scale"] = float(r["scale"])
    return rows


def select_split(rows: List[Dict], split: str, max_slices: Optional[int] = None) -> List[Dict]:
    out = [r for r in rows if r["split"] == split]
    return out[:max_slices] if max_slices else out


def slice_path(data_dir: str, row: Dict) -> str:
    return os.path.join(data_dir, "slices", row["name"] + ".npy")


def load_slice(data_dir: str, row: Dict, mmap: bool = False) -> np.ndarray:
    """Full (M, N, J, T) complex64 hybrid array of one slice."""
    return np.load(slice_path(data_dir, row), mmap_mode="r" if mmap else None)


def fixed_slice_mask(row: Dict, af: float, seed: int = 0) -> np.ndarray:
    """Deterministic k-t mask for one slice (validation / test), keyed on the
    slice name so it never changes when the manifest grows or is re-split."""
    rng = np.random.default_rng([seed, zlib.crc32(row["name"].encode()), int(round(af * 1000))])
    return kt_random_mask(row["N"], row["T"], af, rng,
                          valid=(row["ky_valid_start"], row["ky_valid_stop"]))


class OCMRRowDataset(Dataset):
    """Samples: {"z": (2J, N, T) zero-filled input, "e_ref": (2J, N, T) label,
    "mask": (1, N, T)} -- the dynamic analogue of ../odls's row samples."""

    def __init__(self, data_dir: str, split: str, af: float, fixed_mask: bool,
                 seed: int = 0, max_slices: Optional[int] = None):
        self.data_dir = data_dir
        self.rows = select_split(read_manifest(data_dir), split, max_slices)
        if not self.rows:
            raise ValueError(f"No '{split}' slices in {data_dir}/{MANIFEST}")
        coils = {r["J"] for r in self.rows}
        if len(coils) != 1:
            raise ValueError(f"Mixed coil counts {coils} in split '{split}'")
        self.n_coils = coils.pop()
        self.af = af
        self.fixed_mask = fixed_mask
        self.seed = seed

        self.index = [(s, m) for s, r in enumerate(self.rows) for m in range(r["M"])]
        self.rows_of_slice = []
        start = 0
        for r in self.rows:
            self.rows_of_slice.append(np.arange(start, start + r["M"]))
            start += r["M"]

        self._arrays: Dict[int, np.ndarray] = {}
        self._fixed_masks: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _array(self, s: int) -> np.ndarray:
        # Memory-mapped per process: only the requested rows are read from disk.
        if s not in self._arrays:
            self._arrays[s] = load_slice(self.data_dir, self.rows[s], mmap=True)
        return self._arrays[s]

    def _mask(self, s: int, epoch: int, idx: int) -> np.ndarray:
        r = self.rows[s]
        if self.fixed_mask:
            if s not in self._fixed_masks:
                self._fixed_masks[s] = fixed_slice_mask(r, self.af, self.seed)
            return self._fixed_masks[s]
        rng = np.random.default_rng([self.seed, epoch, idx])
        return kt_random_mask(r["N"], r["T"], self.af, rng, valid=(r["ky_valid_start"], r["ky_valid_stop"]))

    def __getitem__(self, key):
        # SliceBatchSampler yields (epoch, idx) pairs so masks can depend on the epoch.
        epoch, idx = key if isinstance(key, tuple) else (0, key)
        s, m = self.index[idx]
        e_ref = np.asarray(self._array(s)[m])         # (N, J, T) complex
        mask = self._mask(s, epoch, idx)              # (N, T)
        z = e_ref * mask[:, None, :]
        return {
            "z": torch.from_numpy(np_to_channels(z)),
            "e_ref": torch.from_numpy(np_to_channels(e_ref)),
            "mask": torch.from_numpy(mask[None]),
        }


REFERENCE_AREA = 144 * 22  # median OCMR ky x frames


def memory_balanced_batch_sizes(rows: List[Dict], batch_size: int, min_rows: int = 8,
                                max_rows: int = 64) -> List[int]:
    """Per-slice batch sizes holding N*T*rows ~ constant: `batch_size` rows of a
    median 144x22 slice, fewer rows for bigger slices, more for smaller ones,
    so GPU memory per step stays flat across OCMR's 104-180 ky / 10-31 frames
    (fixed 32 rows would need ~2x the memory on the largest slices)."""
    return [int(np.clip(round(batch_size * REFERENCE_AREA / (r["N"] * r["T"])), min_rows, max_rows))
            for r in rows]


class SliceBatchSampler(Sampler[List[int]]):
    """Batches of rows that all come from one slice (shared N and T).

    The order is a pure function of (seed, epoch), which is what makes exact
    mid-epoch resume possible: train.py stores how many batches of the
    current epoch are done and sets `start_batch` on resume.
    """

    def __init__(self, rows_of_slice: List[np.ndarray], batch_sizes: List[int], shuffle: bool,
                 seed: int = 0, drop_small: bool = True):
        self.rows_of_slice = rows_of_slice
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.seed = seed
        self.drop_small = drop_small
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch = epoch
        self.start_batch = start_batch

    def batches(self) -> List[List[int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        out = []
        for idxs, bs in zip(self.rows_of_slice, self.batch_sizes):
            idxs = rng.permutation(idxs) if self.shuffle else idxs
            for i in range(0, len(idxs), bs):
                b = idxs[i:i + bs]
                # A tiny leftover batch gives noisy BatchNorm statistics.
                if self.drop_small and self.shuffle and len(b) < max(2, min(bs, len(idxs)) // 2):
                    continue
                out.append([int(x) for x in b])
        if self.shuffle:
            out = [out[i] for i in rng.permutation(len(out))]
        return out

    def __iter__(self) -> Iterator[List[tuple]]:
        for b in self.batches()[self.start_batch:]:
            yield [(self.epoch, i) for i in b]

    def __len__(self) -> int:
        return len(self.batches()) - self.start_batch
