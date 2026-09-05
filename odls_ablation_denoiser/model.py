"""
Ablation variant: DENOISER PRIOR ONLY, isolated from the other two
changes in odls_v2/model.py.

Architecturally, this is ../odls/model.py (10 independently-trained
copies per role, no weight sharing, no phase embeddings, no cross-phase
attention) with exactly one change: the low-rank-motivated N1 module is
replaced by a general-purpose residual denoiser (direction 2 in
odls_v2/README.md): `output = input - predicted_residual`, framed as
"clean up artifacts" rather than "detect low-rank violations" -- still
just a 1D CNN, so nothing about parameter count or the paper's
1D-convolution efficiency argument changes.

Purpose: odls_v2's combined model (sharing + denoiser + attention)
underperformed the ../odls baseline in testing, across two width
settings. With three entangled changes and an unexplained shortfall,
this is the second of three isolated variants (after
../odls_ablation_sharing) needed to find out which change is actually
responsible.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_conv_stack(in_ch: int, out_ch: int, n_layers: int, width: int, kernel_size: int = 3):
    layers = []
    c_in = in_ch
    for i in range(n_layers):
        c_out = width if i < n_layers - 1 else out_ch
        layers.append(nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2))
        if i < n_layers - 1:
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ReLU(inplace=True))
        c_in = c_out
    return nn.Sequential(*layers)


class DeepDenoiserModule(nn.Module):
    """Replaces ../odls/model.py's DeepLowRankModule (r = eta1 * D1(e_prev))
    with residual learning: r = eta1 * (e_prev - D1(e_prev)), i.e. D1 is
    trained to predict the artifact/residual to subtract, not the
    correction to add directly. Independent per phase (no sharing) --
    same architecture (6 conv layers, 48 filters) and same eta1
    init/behavior as the baseline, only the forward pass's framing
    changes.
    """

    def __init__(self, channels: int, width: int = 48, n_layers: int = 6):
        super().__init__()
        self.D1 = _make_conv_stack(channels, channels, n_layers, width)
        self.eta1 = nn.Parameter(torch.tensor(0.001))

    def forward(self, e_prev: torch.Tensor) -> torch.Tensor:
        denoised_delta = e_prev - self.D1(e_prev)  # DnCNN-style residual learning
        return self.eta1 * denoised_delta


class DataConsistencyModule(nn.Module):
    """Unchanged from ../odls/model.py -- Eq. 13's gradient-descent
    data-fidelity step, not part of this ablation."""

    def __init__(self):
        super().__init__()
        self.eta2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, e_prev: torch.Tensor, r: torch.Tensor,
                z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        data_term = mask * (mask * e_prev - z)
        grad = data_term + 2.0 * r
        return e_prev - self.eta2 * grad


class DeepSparseModule(nn.Module):
    """Unchanged from ../odls/model.py -- the learned sparsifying
    transform pair + soft-threshold, including the runaway-threshold cap
    and the FFT-centering fix. Not part of this ablation."""

    def __init__(self, channels: int, width: int = 48, n_layers: int = 3,
                 max_threshold: float = 0.05):
        super().__init__()
        self.psi2 = _make_conv_stack(channels, channels, n_layers, width)
        self.psi3 = _make_conv_stack(channels, channels, n_layers, width)
        self.threshold = nn.Parameter(torch.tensor(0.001))
        self.max_threshold = max_threshold

    def _soft_threshold(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        lam = torch.clamp(lam, min=0.0, max=self.max_threshold)
        return torch.sign(x) * torch.relu(torch.abs(x) - lam)

    @staticmethod
    def _ifft_pe(k_space: torch.Tensor, n_coils: int) -> torch.Tensor:
        real, imag = k_space[:, :n_coils], k_space[:, n_coils:]
        complex_ks = torch.complex(real, imag)
        img = torch.fft.fftshift(
            torch.fft.ifft(torch.fft.ifftshift(complex_ks, dim=-1), dim=-1, norm="ortho"), dim=-1
        )
        return torch.cat([img.real, img.imag], dim=1)

    @staticmethod
    def _fft_pe(image: torch.Tensor, n_coils: int) -> torch.Tensor:
        real, imag = image[:, :n_coils], image[:, n_coils:]
        complex_img = torch.complex(real, imag)
        ks = torch.fft.fftshift(
            torch.fft.fft(torch.fft.ifftshift(complex_img, dim=-1), dim=-1, norm="ortho"), dim=-1
        )
        return torch.cat([ks.real, ks.imag], dim=1)

    def transform_and_reconstruct(self, d: torch.Tensor, n_coils: int):
        img = self._ifft_pe(d, n_coils)
        transformed = self.psi3(self.psi2(img))
        sparsified = self._soft_threshold(transformed, self.threshold)
        e_next = self._fft_pe(sparsified, n_coils)
        return e_next, transformed, img

    def forward(self, d: torch.Tensor, n_coils: int) -> torch.Tensor:
        e_next, _, _ = self.transform_and_reconstruct(d, n_coils)
        return e_next


class ODLSPhase(nn.Module):
    """One unrolled iteration: DeepDenoiserModule -> DC -> DS."""

    def __init__(self, n_coils: int, width: int = 48, max_threshold: float = 0.05):
        super().__init__()
        channels = 2 * n_coils
        self.n_coils = n_coils
        self.denoiser = DeepDenoiserModule(channels, width=width, n_layers=6)
        self.dc = DataConsistencyModule()
        self.ds = DeepSparseModule(channels, width=width, n_layers=3, max_threshold=max_threshold)

    def forward(self, e_prev: torch.Tensor, z: torch.Tensor, mask: torch.Tensor):
        r = self.denoiser(e_prev)
        d = self.dc(e_prev, r, z, mask)
        e_next, transformed, img = self.ds.transform_and_reconstruct(d, self.n_coils)
        return e_next, transformed, img


class ODLSAblationDenoiser(nn.Module):
    """Denoiser-only ablation: same 10-independent-phases architecture as
    the baseline (no weight sharing, no cross-phase attention), only the
    low-rank module's framing changes to residual denoising."""

    def __init__(self, n_coils: int, n_phases: int = 10, width: int = 48, max_threshold: float = 0.05):
        super().__init__()
        self.n_coils = n_coils
        self.phases = nn.ModuleList([
            ODLSPhase(n_coils, width=width, max_threshold=max_threshold) for _ in range(n_phases)
        ])
        self._xavier_init()

    def _xavier_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def clamp_thresholds_(self) -> None:
        for phase in self.phases:
            ds = phase.ds
            ds.threshold.data.clamp_(min=0.0, max=ds.max_threshold)

    def forward(self, z: torch.Tensor, mask: torch.Tensor):
        e = z
        e_preds = []
        sym_pairs = []
        for phase in self.phases:
            e, transformed, img = phase(e, z, mask)
            e_preds.append(e)
            sym_pairs.append((transformed, img))
        return e_preds, sym_pairs

    def reconstruct_image(self, e_final: torch.Tensor) -> torch.Tensor:
        real, imag = e_final[:, : self.n_coils], e_final[:, self.n_coils :]
        complex_ks = torch.complex(real, imag)
        img = torch.fft.fftshift(
            torch.fft.ifft(torch.fft.ifftshift(complex_ks, dim=-1), dim=-1, norm="ortho"), dim=-1
        )
        return torch.cat([img.real, img.imag], dim=1)
