"""
Dynamic ODLS: ODLS (Wang et al., arXiv:2112.04721) extended to cine MRI.

The unrolled structure is the paper's Fig. 4 / Eqs. 12-14, unchanged:

  DLR  r^(k) = eta1^(k) * D1(e^(k-1))                                   (12)
  DC   d^(k) = e^(k-1) - eta2^(k) * [ Lambda*(Lambda e^(k-1) - z) + 2 r^(k) ] (13)
  DS   e^(k) = Psi_PE * Psi3[ soft( Psi2( Psi_PE^-1 d^(k) ); lambda^(k) ) ]   (14)

with e^(0) = Lambda* z (zero-filled), K = 10 independent phases, 6/3/3 conv
layers for D1/Psi2/Psi3, 48 filters of size 3, BN + ReLU, Xavier init,
eta1 = 0.001, eta2 = 1, lambda = 0.001.

What's new for dynamic data: each sample is one FE row across all frames,
shape (B, 2J, N, T), and the convolutions are 2D over (ky, t), so every
module sees the temporal neighbourhood. Time is padded circularly (a cine
loop is one periodic cardiac cycle); ky is zero-padded.

Fix relative to ../odls/model.py (the important one): the soft-threshold
now sits *between* Psi2 and Psi3, in a `width`-channel transform domain,
exactly as Eq. 14 (and the PISTA iteration of Eq. 11 it unrolls) says. The
baseline thresholded after Psi3, i.e. directly on image values, so the
learned "sparsifying transform" never actually sparsified anything -- which
is also the likely cause of the threshold blow-up the baseline needed a
max_threshold cap for. The symmetry loss ||Psi3(Psi2(x)) - x||^2 now does
its intended job (keeping Psi3 an approximate inverse of Psi2). Sanity
check: with 1D kernels this layout gives 658,910 parameters vs the paper's
reported 664,350 (0.8% off); the baseline's layout gives 566,430 (15% off).

Trainability (ds_residual, default on): Eq. 14 taken literally makes the DS
output Psi3(soft(Psi2(x))) with no path from x to the output, so at
initialisation every phase emits noise and the network must first learn an
identity map through two deep BN/ReLU stacks. Measured on a real OCMR slice
(AF 4, small model, 25 epochs, same settings): literal form RLNE 44.7% /
PSNR 22.0 dB -- worse than zero-filled (31.6% / 25.0 dB) -- vs 16.4% /
30.7 dB with the residual form. The default
computes  x + Psi3(soft(Psi2 x)) - Psi3(Psi2 x)  instead. Once the symmetry
loss holds (Psi3 o Psi2 = identity) this equals Eq. 14 exactly, but it
starts as the identity (lambda ~ 0 at init), and costs nothing in training
since Psi3(Psi2 x) is computed for L_sym anyway. --no-ds-residual gives the
literal form.

Options (all stored in the checkpoint's model config):
  kernel_t      temporal kernel size; 1 = frame-by-frame (static) baseline
                with the identical pipeline, for the "does time help?"
                comparison.
  sparse_domain "yt" (paper: image space along ky, frames as-is) or "yf"
                (temporal Fourier transform first -- the classic x-f
                sparsity domain of k-t methods).
  max_threshold optional cap on lambda (None = no cap, as in the paper);
                train.py logs every phase's lambda so a blow-up is visible.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transforms import PE_DIM, TIME_DIM, fftc, ifftc


class KTConv(nn.Module):
    """2D conv over (ky, t): zero padding along ky, circular along t."""

    def __init__(self, c_in: int, c_out: int, kernel_n: int = 3, kernel_t: int = 3):
        super().__init__()
        self.pad_t = kernel_t // 2
        self.conv = nn.Conv2d(c_in, c_out, (kernel_n, kernel_t), padding=(kernel_n // 2, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_t:
            x = F.pad(x, (self.pad_t, self.pad_t, 0, 0), mode="circular")
        return self.conv(x)


def conv_stack(c_in: int, c_out: int, n_layers: int, width: int, kernel_t: int) -> nn.Sequential:
    """n_layers convs of `width` filters; BN + ReLU after every layer but the
    last (whose output is the signed signal / transform coefficients)."""
    layers: List[nn.Module] = []
    c = c_in
    for i in range(n_layers):
        last = i == n_layers - 1
        out = c_out if last else width
        layers.append(KTConv(c, out, 3, kernel_t))
        if not last:
            layers += [nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
        c = out
    return nn.Sequential(*layers)


class DeepLowRankModule(nn.Module):
    """Eq. 12. D1 replaces the null-space filterbank Q (Sec. III-C.1); it
    operates on hybrid k-space (ky, t), so its kernels span neighbouring
    ky lines (the Hankel/annihilation-filter structure) and neighbouring
    frames."""

    def __init__(self, channels: int, width: int, kernel_t: int):
        super().__init__()
        self.D1 = conv_stack(channels, channels, 6, width, kernel_t)
        self.eta1 = nn.Parameter(torch.tensor(0.001))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.eta1 * self.D1(e)


class DataConsistencyModule(nn.Module):
    """Eq. 13. Lambda is the 0/1 k-t mask (B, 1, N, T); self-adjoint."""

    def __init__(self):
        super().__init__()
        self.eta2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, e: torch.Tensor, r: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return e - self.eta2 * (mask * (mask * e - z) + 2.0 * r)


class DeepSparseModule(nn.Module):
    """Eq. 14 with the threshold in the learned transform domain."""

    def __init__(self, channels: int, width: int, kernel_t: int, sparse_domain: str,
                 max_threshold: Optional[float], ds_residual: bool = True):
        super().__init__()
        if sparse_domain not in ("yt", "yf"):
            raise ValueError(f"sparse_domain must be 'yt' or 'yf', got {sparse_domain!r}")
        self.ds_residual = ds_residual
        self.psi2 = conv_stack(channels, width, 3, width, kernel_t)  # forward transform -> width-ch domain
        self.psi3 = conv_stack(width, channels, 3, width, kernel_t)  # inverse transform
        # lambda = exp(log_threshold), init 0.001 (paper). Stored in log space so
        # it is always > 0 and its gradient never vanishes: a plain parameter
        # clamped at 0 was pushed negative within the first steps of a real
        # OCMR run, got stuck at exactly 0 (clamp has zero gradient), and that
        # turns the residual DS into the identity for the rest of training.
        self.log_threshold = nn.Parameter(torch.tensor(math.log(0.001)))
        self.sparse_domain = sparse_domain
        self.max_threshold = max_threshold

    def effective_threshold(self) -> torch.Tensor:
        lam = self.log_threshold.exp()
        return lam if self.max_threshold is None else lam.clamp(max=self.max_threshold)

    def _to_sparse_domain(self, d: torch.Tensor) -> torch.Tensor:
        x = ifftc(d, PE_DIM)                 # hybrid k-space -> image (y, t)
        if self.sparse_domain == "yf":
            x = fftc(x, TIME_DIM)            # (y, t) -> (y, f)
        return x

    def _from_sparse_domain(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparse_domain == "yf":
            x = ifftc(x, TIME_DIM)
        return fftc(x, PE_DIM)

    def forward(self, d: torch.Tensor, need_sym: bool = True):
        x = self._to_sparse_domain(d)
        coeffs = self.psi2(x)
        lam = self.effective_threshold().to(coeffs.dtype)
        sparse = torch.sign(coeffs) * F.relu(coeffs.abs() - lam)
        out = self.psi3(sparse)
        full = self.psi3(coeffs) if (need_sym or self.ds_residual) else None
        if self.ds_residual:
            # x + Psi3(soft(Psi2 x)) - Psi3(Psi2 x): equals Eq. 14's Psi3(soft(Psi2 x))
            # exactly once Psi3 o Psi2 = identity (what L_sym enforces), but starts
            # as the identity map instead of a random one (see module docstring).
            out = x + out - full
        e_next = self._from_sparse_domain(out)
        # Symmetry pair of Eq. 15: Psi3(Psi2(x)) vs x, on the *unthresholded* coefficients.
        sym = (full, x) if need_sym else None
        return e_next, sym


class ODLSPhase(nn.Module):
    def __init__(self, channels: int, width: int, kernel_t: int, sparse_domain: str,
                 max_threshold: Optional[float], ds_residual: bool):
        super().__init__()
        self.dlr = DeepLowRankModule(channels, width, kernel_t)
        self.dc = DataConsistencyModule()
        self.ds = DeepSparseModule(channels, width, kernel_t, sparse_domain, max_threshold, ds_residual)

    def forward(self, e, z, mask, need_sym: bool = True):
        r = self.dlr(e)
        d = self.dc(e, r, z, mask)
        return self.ds(d, need_sym)


class ODLSDynamic(nn.Module):
    def __init__(self, n_coils: int = 8, n_phases: int = 10, width: int = 48, kernel_t: int = 3,
                 sparse_domain: str = "yt", max_threshold: Optional[float] = None, ds_residual: bool = True):
        super().__init__()
        self.config: Dict = dict(n_coils=n_coils, n_phases=n_phases, width=width, kernel_t=kernel_t,
                                 sparse_domain=sparse_domain, max_threshold=max_threshold,
                                 ds_residual=ds_residual)
        channels = 2 * n_coils
        self.phases = nn.ModuleList([
            ODLSPhase(channels, width, kernel_t, sparse_domain, max_threshold, ds_residual)
            for _ in range(n_phases)
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, mask: torch.Tensor, need_sym: Optional[bool] = None
                ) -> Tuple[List[torch.Tensor], List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """z: (B, 2J, N, T) zero-filled hybrid k-space; mask: (B, 1, N, T).
        Returns e_preds (K outputs, for L_err) and sym_pairs (K pairs, for
        L_sym; None entries when need_sym is False, e.g. at inference)."""
        if need_sym is None:
            need_sym = self.training
        e = z
        e_preds, sym_pairs = [], []
        for phase in self.phases:
            e, sym = phase(e, z, mask, need_sym)
            e_preds.append(e)
            sym_pairs.append(sym)
        return e_preds, sym_pairs

    @torch.no_grad()
    def scalar_report(self) -> Dict[str, List[float]]:
        """Per-phase eta1 / eta2 / lambda, logged every epoch by train.py."""
        return {
            "eta1": [p.dlr.eta1.item() for p in self.phases],
            "eta2": [p.dc.eta2.item() for p in self.phases],
            "lambda": [p.ds.effective_threshold().item() for p in self.phases],
        }

    @torch.no_grad()
    def clamp_thresholds_(self) -> None:
        for p in self.phases:
            if p.ds.max_threshold is not None:
                p.ds.log_threshold.clamp_(max=math.log(p.ds.max_threshold))
