"""
k-t undersampling masks for dynamic (cine) MRI.

Undersampling is along ky (phase encoding) and differs from frame to frame,
which is what lets a dynamic reconstruction exploit temporal redundancy: a
ky line missed in one frame is usually acquired in a neighbouring one. kx
(readout) is always fully sampled, so ODLS's FE-IFT row splitting
(paper Sec. II, Eq. 1) still applies.

Fixes relative to ../odls/masks.py:
  - AF is exact, centre lines included: AF = acquired ky lines /
    sampled ky lines per frame (paper Sec. II). The old uniform_mask added
    the centre block on top of the lattice, giving AF ~3.25 at "AF=4".
  - The centre block sits at the true k-space centre (index n // 2, which
    every OCMR file uses -- verified on all 165 fully-sampled headers). The
    old partial_fourier_mask kept indices [0, n/af), i.e. the high-frequency
    edge, never the centre.

The centre block (CENTER_FRACTION of the acquired lines) is sampled in
every frame. preprocess_ocmr.py uses exactly these lines for coil
compression and intensity normalisation, so neither step needs anything
that wouldn't be available from a real undersampled scan.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

CENTER_FRACTION = 0.08


def center_lines(n: int, valid: Optional[Tuple[int, int]] = None,
                 center_fraction: float = CENTER_FRACTION, af: Optional[float] = None) -> np.ndarray:
    """Indices of the always-sampled calibration block around ky = n // 2.

    `valid` = (start, stop): range of ky lines that exist in the data
    (all of [0, n) for OCMR). If `af` is given the block is capped so it
    never exceeds the per-frame line budget.
    """
    start_v, stop_v = valid if valid is not None else (0, n)
    n_valid = stop_v - start_v
    n_center = max(1, int(round(n_valid * center_fraction)))
    if af is not None:
        n_center = min(n_center, lines_per_frame(n_valid, af))
    start = max(start_v, min(n // 2 - n_center // 2, stop_v - n_center))
    return np.arange(start, start + n_center)


def lines_per_frame(n_valid: int, af: float) -> int:
    return max(1, int(round(n_valid / af)))


def kt_random_mask(n: int, n_frames: int, af: float, rng: np.random.Generator,
                   valid: Optional[Tuple[int, int]] = None,
                   center_fraction: float = CENTER_FRACTION,
                   sigma_fraction: float = 0.25) -> np.ndarray:
    """Variable-density random Cartesian k-t mask, shape (n, n_frames), 0/1.

    Each frame keeps the centre block plus an independent Gaussian-weighted
    random draw of the remaining lines, so every frame has exactly
    lines_per_frame(n_valid, af) sampled lines (exact AF).
    """
    start_v, stop_v = valid if valid is not None else (0, n)
    n_valid = stop_v - start_v
    budget = lines_per_frame(n_valid, af)
    center = center_lines(n, valid, center_fraction, af)
    n_extra = budget - center.size

    candidates = np.setdiff1d(np.arange(start_v, stop_v), center)
    weights = np.exp(-0.5 * ((candidates - n / 2.0) / (sigma_fraction * n)) ** 2)
    weights /= weights.sum()

    mask = np.zeros((n, n_frames), dtype=np.float32)
    mask[center, :] = 1.0
    if n_extra > 0:
        for t in range(n_frames):
            chosen = rng.choice(candidates, size=n_extra, replace=False, p=weights)
            mask[chosen, t] = 1.0
    return mask


def effective_af(mask: np.ndarray, valid: Optional[Tuple[int, int]] = None) -> float:
    """Measured AF of an (n, n_frames) mask over the valid ky range."""
    n = mask.shape[0]
    start_v, stop_v = valid if valid is not None else (0, n)
    return (stop_v - start_v) * mask.shape[1] / float(mask[start_v:stop_v].sum())
