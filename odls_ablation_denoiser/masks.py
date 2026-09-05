"""
Undersampling patterns along the phase-encoding (PE) axis, matching the
three scenarios evaluated in Fig. 11 / Table IV:

  - Cartesian random undersampling (variable density, more samples near
    the k-space center) at a given acceleration factor (AF).
  - Uniform (equispaced) undersampling at a given AF.
  - Partial Fourier undersampling (e.g. 3/4 coverage) at a given AF.

AF is defined as (fully sampled points) / (undersampled points) (Sec. II).
A handful of low-frequency lines are always kept (autocalibration-style
center), consistent with standard Cartesian CS-MRI sampling practice.
"""

from __future__ import annotations

import numpy as np


def cartesian_random_mask(n: int, af: float, center_fraction: float = 0.08,
                           rng: np.random.Generator = None) -> np.ndarray:
    """Variable-density random Cartesian mask along a length-n PE axis."""
    if rng is None:
        rng = np.random.default_rng()

    mask = np.zeros(n, dtype=np.float32)

    n_center = max(1, int(round(n * center_fraction)))
    center_start = n // 2 - n_center // 2
    mask[center_start:center_start + n_center] = 1.0

    n_target = int(round(n / af))
    n_remaining = max(0, n_target - n_center)

    remaining_idx = np.setdiff1d(np.arange(n), np.arange(center_start, center_start + n_center))
    # Gaussian-weighted probability, denser near the center, for
    # variable-density random Cartesian sampling.
    center = n / 2.0
    sigma = n / 4.0
    weights = np.exp(-0.5 * ((remaining_idx - center) / sigma) ** 2)
    weights = weights / weights.sum()

    n_remaining = min(n_remaining, remaining_idx.size)
    chosen = rng.choice(remaining_idx, size=n_remaining, replace=False, p=weights)
    mask[chosen] = 1.0
    return mask


def uniform_mask(n: int, af: float, center_fraction: float = 0.08) -> np.ndarray:
    """Equispaced undersampling with a fully-sampled low-frequency center."""
    mask = np.zeros(n, dtype=np.float32)

    n_center = max(1, int(round(n * center_fraction)))
    center_start = n // 2 - n_center // 2
    mask[center_start:center_start + n_center] = 1.0

    step = int(round(af))
    offset = 0
    mask[offset::step] = 1.0
    return mask


def partial_fourier_mask(n: int, af: float) -> np.ndarray:
    """Partial Fourier undersampling: keep a single contiguous low-frequency
    block of n/af points (one-sided truncation), so the retained fraction
    is always consistent with AF = fully_sampled / undersampled (Sec. II).

    E.g. Fig. 11(b)'s "3/4 partial Fourier, AF=3" case keeps a 1/3-sized
    contiguous block starting at the low-frequency edge; the "3/4" in the
    paper's label names the acquisition scheme (asymmetric echo / partial
    Fourier), not the retained fraction, which is set by AF as usual.
    """
    mask = np.zeros(n, dtype=np.float32)
    n_keep = max(1, int(round(n / af)))
    mask[:n_keep] = 1.0
    return mask


MASK_FACTORIES = {
    "cartesian": cartesian_random_mask,
    "uniform": uniform_mask,
    "partial_fourier": partial_fourier_mask,
}
