"""
1D training-sample construction, matching the learning scheme of
Fig. 2(d) / Table I: "1D learning with 1D hybrid data as training samples".

Given a fully-sampled multi-coil 2D k-space volume of shape
(n_slices, M, N, J) -- M: frequency-encoding (FE), N: phase-encoding (PE),
J: coils -- the pipeline is:

  1. Retrospectively undersample along PE with a mask (Sec. II, Eq. 1).
  2. Take the 1D FE inverse FFT to decouple the 2D problem into M
     independent 1D hybrid rows per slice (Eq. 1): Z = Psi*_FE(Y).
  3. Each row z_m in C^{N x J} becomes one training sample. This is why
     Table I reports N_TS * N_slice * M available samples of size N*J
     each, versus N_TS * N_slice samples of size M*N*J for 2D learning.

This module only implements the sample-construction logic; it expects
fully-sampled k-space volumes as input (e.g. loaded from .npy/.mat files
by the caller) since the paper's in-vivo knee/brain datasets are not
public.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from masks import MASK_FACTORIES


def fe_ifft(k_space: np.ndarray) -> np.ndarray:
    """1D inverse FFT along the frequency-encoding axis (axis 0 of an
    (M, N, J) k-space array), i.e. Psi*_FE in Eq. 1."""
    return np.fft.ifft(np.fft.ifftshift(k_space, axes=0), axis=0, norm="ortho")


def pe_ifft(hybrid: np.ndarray, axis: int = 1) -> np.ndarray:
    """1D inverse FFT along the phase-encoding axis, i.e. Psi*_PE, used to
    go from the reconstructed hybrid data to the final image.

    `axis` defaults to 1, matching the (M, N, J) volume convention used
    throughout this module (M=frequency-encoding, N=phase-encoding,
    J=coils) -- NOT the last axis, which here is coils, not PE.
    """
    return np.fft.ifft(np.fft.ifftshift(hybrid, axes=axis), axis=axis, norm="ortho")


def _to_real_imag_channels(x: np.ndarray) -> np.ndarray:
    """(N, J) complex -> (2*J, N) real array: J real channels then J
    imaginary channels, matching the convention used in model.py."""
    x = np.transpose(x, (1, 0))  # (J, N)
    return np.concatenate([x.real, x.imag], axis=0).astype(np.float32)


class ODLSHybridDataset(Dataset):
    """Builds the 1D hybrid-domain (row-wise) training samples described
    above from a set of fully-sampled multi-coil k-space volumes.

    Parameters
    ----------
    volumes : list of np.ndarray
        Each array has shape (n_slices, M, N, J), complex-valued,
        fully-sampled k-space.
    mask_type : str
        One of "cartesian", "uniform", "partial_fourier" (masks.py).
    af : float
        Acceleration factor for the undersampling mask.
    fixed_mask : bool
        If False (default, matching Sec. IV-A: "each image of training
        and test datasets owns different undersampling masks with the
        same AF"), a new random mask is drawn per __getitem__ call for
        mask types that are stochastic (cartesian).
    seed : Optional[int]
        RNG seed for reproducibility.
    """

    def __init__(self, volumes, mask_type: str = "cartesian", af: float = 4.0,
                 fixed_mask: bool = False, seed: Optional[int] = None):
        if mask_type not in MASK_FACTORIES:
            raise ValueError(f"Unknown mask_type '{mask_type}', expected one of {list(MASK_FACTORIES)}")

        self.mask_factory: Callable = MASK_FACTORIES[mask_type]
        self.af = af
        self.fixed_mask = fixed_mask
        self.rng = np.random.default_rng(seed)

        self._index = []  # (volume_idx, slice_idx, row_idx)
        self._volumes = volumes
        for v_idx, vol in enumerate(volumes):
            n_slices, M, N, J = vol.shape
            for s_idx in range(n_slices):
                for row_idx in range(M):
                    self._index.append((v_idx, s_idx, row_idx))

        self._n_pe = volumes[0].shape[2] if volumes else None
        self._fixed_masks = {}

    def __len__(self):
        return len(self._index)

    def _get_mask(self, n: int) -> np.ndarray:
        if self.fixed_mask:
            if n not in self._fixed_masks:
                self._fixed_masks[n] = self.mask_factory(n, self.af)
            return self._fixed_masks[n]
        try:
            return self.mask_factory(n, self.af, rng=self.rng)
        except TypeError:
            # uniform_mask / partial_fourier_mask don't take an rng kwarg
            return self.mask_factory(n, self.af)

    def __getitem__(self, idx: int):
        v_idx, s_idx, row_idx = self._index[idx]
        vol = self._volumes[v_idx]  # (n_slices, M, N, J) complex k-space
        k_space_slice = vol[s_idx]  # (M, N, J)

        hybrid_full = fe_ifft(k_space_slice)  # Psi*_FE(Y) -> (M, N, J), Eq. 1
        e_ref_row = hybrid_full[row_idx]  # (N, J) fully-sampled label row

        n = e_ref_row.shape[0]
        mask = self._get_mask(n)  # (N,) 0/1

        z_row = e_ref_row * mask[:, None]  # zero-filled undersampled row

        e_ref_ch = _to_real_imag_channels(e_ref_row)  # (2J, N)
        z_ch = _to_real_imag_channels(z_row)  # (2J, N)
        mask_ch = mask.astype(np.float32)[None, :]  # (1, N)

        return {
            "z": torch.from_numpy(z_ch),
            "e_ref": torch.from_numpy(e_ref_ch),
            "mask": torch.from_numpy(mask_ch),
        }
