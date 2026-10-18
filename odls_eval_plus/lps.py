"""
L+S: low-rank plus sparse reconstruction for dynamic MRI -- the classical
(no deep learning, no training) comparison method.

Reference: R. Otazo, E. Candes, D. K. Sodickson, "Low-rank plus sparse matrix
decomposition for accelerated dynamic MRI with separation of background and
dynamic components", Magn. Reson. Med. 73(3), 2015 (ref. [20] of the ODLS paper).

The idea
--------
Write the cine (all pixels x all frames) as a matrix M of size
(pixels, frames) -- the "Casorati" matrix -- and split it as

    M = L + S
    L : LOW-RANK  -> what is shared by all frames (static background); a few
                     singular vectors describe it
    S : SPARSE    -> what changes (the beating heart); sparse after a Fourier
                     transform along time (periodic motion = few frequencies)

and solve
    min  1/2 || E(L + S) - d ||^2  +  lam_L ||L||_*  +  lam_S || F_t S ||_1
where d = the measured (undersampled) k-space and E = the acquisition
(coil sensitivities -> FFT along ky -> k-t mask).

The algorithm (Otazo 2015, the iteration in their Table 1):
    M_0 = E^H d,  S_0 = 0,  L_0 = M_0
    repeat
        L_k = SVT( M_{k-1} - S_{k-1} ; tau_L )           singular value soft-thresholding
        S_k = F_t^H soft( F_t (M_{k-1} - L_{k-1}) ; tau_S )   soft-threshold temporal spectrum
        M_k = L_k + S_k - E^H( E(L_k + S_k) - d )          data-consistency step
    until || M_k - M_{k-1} || < tol * || M_{k-1} ||
tau_L and tau_S are given RELATIVE to the data (lam_L * largest singular value
of M_0, lam_S * largest temporal-spectrum magnitude of M_0), so the same
lam_L / lam_S work for every slice. tune_lps.py picks them on the validation set.

Data layout (same as odls_dynamic)
----------------------------------
    hybrid k-space z : (M, N, J, T) complex -- x (already image space), ky, coil, frame
    mask             : (N, T) 0/1 -- which ky lines were measured in each frame
    image m          : (M, N, T) complex -- x, y, frame (one coil-combined image)
Because kx is fully sampled and already inverse-FFT'd (the hybrid domain), the
acquisition only needs an FFT along ky.

Coil sensitivity maps
---------------------
L+S needs coil maps to combine the 8 coils into one image. As in Otazo 2015
they are estimated from the TIME-AVERAGED undersampled data: different ky
lines are measured in different frames, so averaging over frames fills most
of the low-frequency k-space. A Hann window keeps only the central ky lines
(coil maps are smooth), and each coil image is divided by the root-sum-of-
squares over coils. Nothing from the fully-sampled reference is used.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------- FFT helpers --

def fftc(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Centered, orthonormal FFT along one dimension (same convention as odls_dynamic)."""
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"), dim=dim)


def ifftc(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"), dim=dim)


def soft_complex(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Complex soft-thresholding: shrink the magnitude by tau, keep the phase."""
    mag = x.abs()
    return x * (torch.clamp(mag - tau, min=0) / torch.clamp(mag, min=1e-12))


# ---------------------------------------------------------------- coil maps --

def estimate_coil_maps(z: torch.Tensor, mask: torch.Tensor, n_lines: int = 32) -> torch.Tensor:
    """z: (M, N, J, T) undersampled hybrid k-space, mask: (N, T). Returns maps (M, N, J)."""
    M, N, J, T = z.shape
    counts = mask.sum(dim=1).clamp(min=1)                       # how often each ky line was measured
    k_avg = z.sum(dim=-1) / counts[None, :, None]               # time-averaged k-space (M, N, J)

    width = min(n_lines, N)                                     # Hann window over the central ky lines
    taper = torch.zeros(N, dtype=torch.float32, device=z.device)
    start = N // 2 - width // 2
    taper[start:start + width] = torch.hann_window(width, periodic=False, device=z.device)
    low_res = ifftc(k_avg * taper[None, :, None], dim=1)        # smooth coil images (M, N, J)

    rss = torch.sqrt((low_res.abs() ** 2).sum(dim=-1, keepdim=True))
    return low_res / (rss + 1e-6 * rss.max())


# ------------------------------------------------------- acquisition model E --

class Acquisition:
    """E: image (M, N, T) -> measured k-space (M, N, J, T); E^H is its adjoint."""

    def __init__(self, maps: torch.Tensor, mask: torch.Tensor):
        self.maps = maps                                       # (M, N, J)
        self.mask = mask[None, :, None, :]                     # (1, N, 1, T)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        coil_images = self.maps[..., None] * m[:, :, None, :]  # (M, N, J, T)
        return self.mask * fftc(coil_images, dim=1)

    def adjoint(self, d: torch.Tensor) -> torch.Tensor:
        coil_images = ifftc(self.mask * d, dim=1)
        return (self.maps.conj()[..., None] * coil_images).sum(dim=2)


# ---------------------------------------------------------------- L+S itself --

@torch.no_grad()
def lps_reconstruct(z: torch.Tensor, mask: torch.Tensor, lam_l: float, lam_s: float,
                    n_iter: int = 50, tol: float = 2.5e-3, maps: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """z: (M, N, J, T) complex zero-filled hybrid k-space, mask: (N, T).
    Returns (image (M, N, T) complex, coil maps (M, N, J), info)."""
    if maps is None:
        maps = estimate_coil_maps(z, mask)
    E = Acquisition(maps, mask)
    M_, N_, T_ = z.shape[0], z.shape[1], z.shape[-1]

    m0 = E.adjoint(z)                                            # zero-filled, coil-combined
    tau_l = lam_l * torch.linalg.svdvals(m0.reshape(-1, T_))[0].item()
    tau_s = lam_s * torch.fft.fft(m0, dim=-1, norm="ortho").abs().max().item()

    m_prev, low, sparse = m0, m0, torch.zeros_like(m0)
    iterations = 0
    for iterations in range(1, n_iter + 1):
        # L: singular value soft-thresholding of the Casorati matrix (pixels x frames)
        u, s, vh = torch.linalg.svd((m_prev - sparse).reshape(-1, T_), full_matrices=False)
        low_new = ((u * torch.clamp(s - tau_l, min=0)) @ vh).reshape(M_, N_, T_)
        # S: soft-threshold the temporal Fourier spectrum (uses the previous L, as in Otazo 2015)
        spectrum = torch.fft.fft(m_prev - low, dim=-1, norm="ortho")
        sparse_new = torch.fft.ifft(soft_complex(spectrum, tau_s), dim=-1, norm="ortho")
        low, sparse = low_new, sparse_new
        # data consistency: one gradient step on 1/2 ||E(L+S) - d||^2 (step size 1, ||E|| <= 1)
        x = low + sparse
        m = x - E.adjoint(E.forward(x) - z)
        change = (torch.linalg.norm(m - m_prev) / torch.linalg.norm(m_prev)).item()
        m_prev = m
        if change < tol:
            break
    info = {"iterations": iterations, "final_change": change, "tau_l": tau_l, "tau_s": tau_s,
            "rank_L": int((torch.clamp(s - tau_l, min=0) > 0).sum().item())}
    return m_prev, maps, info


def lps_magnitude(m: torch.Tensor, maps: torch.Tensor) -> np.ndarray:
    """Root-sum-of-squares magnitude of the coil images maps*m, as (T, M, N) -- the
    same kind of image every other method is scored on (odls_dynamic.evaluate.sos_images)."""
    rss_maps = torch.sqrt((maps.abs() ** 2).sum(dim=-1))       # ~1 inside the body
    mag = m.abs() * rss_maps[..., None]                         # (M, N, T)
    return mag.permute(2, 0, 1).float().cpu().numpy()
