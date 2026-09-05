"""
Adapter for training ODLS on the fastMRI knee multicoil dataset
(https://www.kaggle.com/datasets/arafatshovon/fastmri-knee-multicoil),
restricted to the CORPD_FBK acquisition (proton-density weighted,
*without* fat suppression -- fastMRI knee files are split between
"CORPD_FBK" and "CORPDFS_FBK" (fat-suppressed); this module keeps only
the former, matching the "CORPD" the user asked for).

Why this adapter is needed (differences from data.py's assumptions)
--------------------------------------------------------------------
1. Storage format: fastMRI ships per-volume ``.h5`` files, not ``.npy``
   arrays, with kspace stored as (num_slices, num_coils, H, W) --
   coil axis *first*, not last as in this project's (M, N, J) convention.
2. Coil count varies by volume (commonly 15-16 for knee) and is far
   larger than the paper's post-compression J=8, and DataLoader batching
   requires a fixed channel count -- so coil compression to a fixed
   number of virtual coils is applied (Sec. IV-A of the paper compresses
   to 8 coils for the same reason, citing Zhang et al.'s SVD-based coil
   compression, ref. [49]).
3. The readout (frequency-encoding, axis H) direction is acquired at 2x
   oversampling in fastMRI raw data, which is standard practice removed
   before use (fastmri.io / fastmri.data.transforms.center_crop convention)
   so the reconstructed FOV is square-ish and the row count isn't doubled
   for no signal benefit.

Everything downstream (row-wise 1D sample construction, masks, model,
loss) is unchanged; this module only produces (n_slices, M, N, J) complex
arrays / row dicts compatible with data.ODLSHybridDataset's contract.
"""

from __future__ import annotations

import os
import glob
from typing import List, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

import h5py

from data import fe_ifft, _to_real_imag_channels
from masks import MASK_FACTORIES


def find_corpd_files(root: str, fat_suppressed: bool = False,
                      pattern: str = "*.h5") -> List[str]:
    """Scan `root` for fastMRI knee .h5 files whose 'acquisition' attribute
    matches CORPD_FBK (fat_suppressed=False) or CORPDFS_FBK (True)."""
    target = "CORPDFS_FBK" if fat_suppressed else "CORPD_FBK"
    matches = []
    for path in sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True)):
        try:
            with h5py.File(path, "r") as hf:
                acquisition = hf.attrs.get("acquisition", None)
        except OSError:
            continue
        if acquisition == target:
            matches.append(path)
    return matches


def coil_compress(kspace_mnj: np.ndarray, n_virtual_coils: int) -> np.ndarray:
    """SVD-based coil compression (Sec. IV-A, ref. [49]): projects the J
    physical coils of a single (M, N, J) k-space slice onto the top
    `n_virtual_coils` virtual coils.
    """
    M, N, J = kspace_mnj.shape
    if J <= n_virtual_coils:
        return kspace_mnj

    flat = kspace_mnj.reshape(M * N, J)  # (samples, J)
    # Economy SVD of the (samples, J) data matrix: flat = U @ diag(S) @ Vh
    _, _, vh = np.linalg.svd(flat, full_matrices=False)  # vh: (J, J)
    compress_mat = vh[:n_virtual_coils, :].conj().T  # (J, n_virtual_coils)
    compressed = flat @ compress_mat  # (samples, n_virtual_coils)
    return compressed.reshape(M, N, n_virtual_coils).astype(np.complex64)


def _resize_via_kspace(kspace_mnj: np.ndarray, axis: int, target_size: Optional[int]) -> np.ndarray:
    """Forces `kspace_mnj`'s size along `axis` to exactly `target_size` by
    inverse-FFTing to image space, center-cropping (if the axis is larger
    than target) or zero-padding (if smaller) around the center, then
    FFTing back to k-space.

    This is the physically correct way to change the reconstructed
    image's matrix size along one axis while leaving the encoded
    resolution/content alone -- unlike truncating k-space directly
    (dropping outer/high-frequency samples), which instead reduces
    resolution (a low-pass effect) and does NOT correspond to "center-
    cropping the image" the way the paper's Sec. IV-A preprocessing does.
    Zero-padding k-space is the mirror-image, resolution-preserving way
    to *enlarge* the FOV when a source file is smaller than the target
    (standard "zero-fill interpolation").

    Forcing every slice to the exact same target size on both spatial
    axes (not just when it happens to need cropping) is what guarantees
    every sample in a batch has an identical shape -- fastMRI knee
    volumes vary in both readout size and phase-encoding width from file
    to file, and PyTorch's default collate crashes the instant two
    samples in the same batch disagree on either axis.
    """
    if target_size is None:
        return kspace_mnj

    current = kspace_mnj.shape[axis]
    if current == target_size:
        return kspace_mnj

    img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kspace_mnj, axes=axis), axis=axis, norm="ortho"), axes=axis)

    if current > target_size:
        start = (current - target_size) // 2
        slicer = [slice(None)] * kspace_mnj.ndim
        slicer[axis] = slice(start, start + target_size)
        img = img[tuple(slicer)]
    else:
        pad_total = target_size - current
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        pad_width = [(0, 0)] * kspace_mnj.ndim
        pad_width[axis] = (pad_before, pad_after)
        img = np.pad(img, pad_width, mode="constant")

    ks = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=axis), axis=axis, norm="ortho"), axes=axis)
    return ks.astype(np.complex64)


def remove_readout_oversampling(kspace_mnj: np.ndarray, target_m: Optional[int] = None) -> np.ndarray:
    """Removes the 2x readout oversampling fastMRI raw k-space carries
    along the frequency-encoding axis (axis 0 here, M): resizes (via
    _resize_via_kspace) to `target_m` rows, default M // 2 (the standard
    2x factor) when `target_m` isn't given.
    """
    if target_m is None:
        target_m = kspace_mnj.shape[0] // 2
    return _resize_via_kspace(kspace_mnj, axis=0, target_size=target_m)


def resize_pe(kspace_mnj: np.ndarray, target_n: Optional[int]) -> np.ndarray:
    """Forces the phase-encoding axis (axis 1, N) to exactly `target_n`,
    matching the paper's "center-cropped to 224x224" preprocessing
    (Sec. IV-A) and, just as importantly, guaranteeing every fastMRI
    sample has identical N so DataLoader batching can never crash on a
    shape mismatch, however the raw files vary."""
    return _resize_via_kspace(kspace_mnj, axis=1, target_size=target_n)


def compute_normalization_scale(kspace_mnj: np.ndarray, eps: float = 1e-12) -> float:
    """Per-slice amplitude scale: the max magnitude of the coil-combined
    (SoS) image-domain reconstruction of `kspace_mnj`.

    Raw MRI k-space -- fastMRI included -- is naturally tiny in magnitude
    (mean SoS image magnitude on the order of 1e-5, confirmed by directly
    inspecting the NYU knee dataset ODLS's own paper sources its data
    from). Training on that scale unnormalized lets a network cheaply
    minimize raw MSE by collapsing toward a near-zero output rather than
    learning real reconstruction -- observed in practice on this project
    (byte-identical, near-total reconstruction failure across every
    checkpoint tested, all epochs, both before and after independently
    fixing an unrelated runaway-threshold issue). Dividing k-space by
    this scale before it reaches the network is the same technique used
    by the published training code for that NYU dataset (the exact data
    the paper's own knee experiments used): VLOGroup/mri-variationalnetwork's
    mridata.py computes `norm = max(abs(zero-filled reconstruction))` and
    divides k-space, the initial image, and the reference/target by that
    one value. This adapts that same idea to a fully-sampled reference
    (rather than a specific undersampled realization), so the resulting
    scale doesn't depend on which random mask a given training sample
    happens to draw.
    """
    img = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace_mnj, axes=(0, 1)), axes=(0, 1), norm="ortho"),
        axes=(0, 1),
    )
    sos = np.sqrt((np.abs(img) ** 2).sum(axis=-1))
    scale = float(sos.max())
    return scale if scale > eps else 1.0


def load_fastmri_volume(path: str, n_virtual_coils: Optional[int] = 8,
                         crop_fe_to: Optional[int] = 224,
                         crop_pe_to: Optional[int] = 224) -> np.ndarray:
    """Loads one fastMRI .h5 volume and returns a (n_slices, M, N, J)
    complex64 k-space array in this project's convention (coils last).

    Defaults to 224 on both spatial axes, matching the paper's Sec. IV-A
    "center-cropped to 224x224" preprocessing. This isn't just cosmetic:
    fastMRI knee files vary in both readout size and phase-encoding width
    from file to file, so forcing every slice -- from every file -- to
    the exact same (crop_fe_to, crop_pe_to) is what prevents a shape
    mismatch crash when the DataLoader batches slices from different
    files together. crop_fe_to=None restores the old "just remove 2x
    oversampling, keep native size" behavior (not batch-safe across a
    mixed set of files); crop_pe_to=None disables PE resizing entirely.
    """
    with h5py.File(path, "r") as hf:
        kspace = hf["kspace"][()]  # (n_slices, n_coils, H, W), complex64

    kspace = np.transpose(kspace, (0, 2, 3, 1))  # -> (n_slices, H, W, n_coils) = (n_slices, M, N, J)

    processed_slices = []
    for s in range(kspace.shape[0]):
        sl = kspace[s]
        sl = remove_readout_oversampling(sl, target_m=crop_fe_to)
        sl = resize_pe(sl, target_n=crop_pe_to)
        if n_virtual_coils is not None:
            sl = coil_compress(sl, n_virtual_coils)
        scale = compute_normalization_scale(sl)
        sl = (sl / scale).astype(np.complex64)
        processed_slices.append(sl)

    return np.stack(processed_slices, axis=0)


class FastMRICorpdDataset(Dataset):
    """Row-wise (1D hybrid-domain) training samples built lazily from a
    list of fastMRI CORPD_FBK .h5 files, matching the sample contract of
    data.ODLSHybridDataset (dict with "z", "e_ref", "mask" tensors) so it
    is a drop-in replacement in train.py / evaluate.py.

    Volumes are loaded and cached slice-by-slice (bounded LRU cache) to
    avoid materializing the whole dataset in memory, since fastMRI knee
    is far larger than the paper's original in-vivo datasets.
    """

    def __init__(self, file_paths: List[str], mask_type: str = "cartesian", af: float = 4.0,
                 n_virtual_coils: Optional[int] = 8, crop_fe_to: Optional[int] = 224,
                 crop_pe_to: Optional[int] = 224,
                 fixed_mask: bool = False, seed: Optional[int] = None,
                 slice_cache_size: Optional[int] = None):
        """crop_fe_to / crop_pe_to default to 224 (the paper's Sec. IV-A
        preprocessing target) and, critically, MUST be set to the same
        fixed value across every file for a given dataset instance:
        fastMRI knee files vary in both readout size and phase-encoding
        width, and PyTorch's default collate_fn crashes the moment two
        samples in the same batch disagree on shape. Passing None for
        either disables that axis's resizing -- only safe if you already
        know every file in `file_paths` shares the same native size on
        that axis (e.g. a single-source dataset), otherwise batching will
        fail as soon as two differently-sized files land in one batch.

        slice_cache_size: how many (file, slice) preprocessed hybrid
        arrays to keep in memory (each holding coil compression's SVD and
        two FFT round-trips' worth of computed work). Left as None, this
        defaults to caching every slice in `file_paths` -- with
        `crop_fe_to`/`crop_pe_to` fixed at 224 and `n_virtual_coils=8`
        each cached slice is only a few MB, so caching everything for a
        dataset of a few dozen files costs a few GB of RAM, comfortably
        affordable, and avoids redoing that SVD/FFT work on nearly every
        sample -- which is what happens with a small cache under
        `shuffle=True`, since row-level shuffling almost never revisits
        the same slice twice in a row. Pass an explicit smaller value
        only if you know your dataset is too large for this to fit in
        memory. NOTE: with DataLoader `num_workers > 0`, each worker
        process holds its own independent copy of this cache (it is not
        shared across processes) -- pass `persistent_workers=True` to the
        DataLoader (train.py already does) so each worker's cache survives
        across epochs instead of being rebuilt from scratch every epoch.
        """
        if mask_type not in MASK_FACTORIES:
            raise ValueError(f"Unknown mask_type '{mask_type}', expected one of {list(MASK_FACTORIES)}")
        if not file_paths:
            raise ValueError("file_paths is empty -- did find_corpd_files find any matches?")

        self.file_paths = file_paths
        self.mask_factory = MASK_FACTORIES[mask_type]
        self.af = af
        self.n_virtual_coils = n_virtual_coils
        self.crop_fe_to = crop_fe_to
        self.crop_pe_to = crop_pe_to
        self.fixed_mask = fixed_mask
        self.rng = np.random.default_rng(seed)
        self._fixed_masks: Dict[int, np.ndarray] = {}

        # Build a global (file_idx, slice_idx, row_idx) index by peeking at
        # each file's kspace shape without loading the full array.
        self._index = []
        self._slice_counts = []
        self._m_per_file = []
        for f_idx, path in enumerate(file_paths):
            with h5py.File(path, "r") as hf:
                n_slices, n_coils, h, w = hf["kspace"].shape
            m = (self.crop_fe_to if self.crop_fe_to is not None else h // 2)
            self._slice_counts.append(n_slices)
            self._m_per_file.append(m)
            for s_idx in range(n_slices):
                for row_idx in range(m):
                    self._index.append((f_idx, s_idx, row_idx))

        total_slices = sum(self._slice_counts)
        self._slice_cache_size = slice_cache_size if slice_cache_size is not None else total_slices
        self._slice_cache: Dict[tuple, np.ndarray] = {}
        self._slice_cache_order: List[tuple] = []

    def __len__(self):
        return len(self._index)

    def _get_slice(self, f_idx: int, s_idx: int) -> np.ndarray:
        """Returns the FE-inverse-FFT'd hybrid data for one slice, i.e.
        Z = Psi*_FE(Y) of shape (M, N, J), from cache or disk."""
        key = (f_idx, s_idx)
        if key in self._slice_cache:
            return self._slice_cache[key]

        path = self.file_paths[f_idx]
        with h5py.File(path, "r") as hf:
            raw = hf["kspace"][s_idx]  # (n_coils, H, W)
        raw = np.transpose(raw, (1, 2, 0))  # (H, W, n_coils) = (M_raw, N, J_raw)

        raw = remove_readout_oversampling(raw, target_m=self.crop_fe_to)
        raw = resize_pe(raw, target_n=self.crop_pe_to)
        if self.n_virtual_coils is not None:
            raw = coil_compress(raw, self.n_virtual_coils)

        scale = compute_normalization_scale(raw)
        raw = (raw / scale).astype(np.complex64)

        hybrid = fe_ifft(raw)  # (M, N, J)

        self._slice_cache[key] = hybrid
        self._slice_cache_order.append(key)
        if len(self._slice_cache_order) > self._slice_cache_size:
            oldest = self._slice_cache_order.pop(0)
            self._slice_cache.pop(oldest, None)

        return hybrid

    def _get_mask(self, n: int) -> np.ndarray:
        if self.fixed_mask:
            if n not in self._fixed_masks:
                try:
                    self._fixed_masks[n] = self.mask_factory(n, self.af, rng=self.rng)
                except TypeError:
                    self._fixed_masks[n] = self.mask_factory(n, self.af)
            return self._fixed_masks[n]
        try:
            return self.mask_factory(n, self.af, rng=self.rng)
        except TypeError:
            return self.mask_factory(n, self.af)

    def __getitem__(self, idx: int):
        f_idx, s_idx, row_idx = self._index[idx]
        hybrid = self._get_slice(f_idx, s_idx)  # (M, N, J)
        e_ref_row = hybrid[row_idx]  # (N, J)

        n = e_ref_row.shape[0]
        mask = self._get_mask(n)

        z_row = e_ref_row * mask[:, None]

        e_ref_ch = _to_real_imag_channels(e_ref_row)
        z_ch = _to_real_imag_channels(z_row)
        mask_ch = mask.astype(np.float32)[None, :]

        return {
            "z": torch.from_numpy(z_ch),
            "e_ref": torch.from_numpy(e_ref_ch),
            "mask": torch.from_numpy(mask_ch),
        }
