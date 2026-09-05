"""
Ablation variant: WEIGHT SHARING ONLY, isolated from the other two
changes in odls_v2/model.py.

This keeps the same phase-embedding + FiLM sharing machinery as
odls_v2 (one shared network each for the low-rank module and the
sparse-transform pair, conditioned on a learnable per-phase embedding),
but:
  - the low-rank module keeps the ORIGINAL low-rank-motivated framing
    from ../odls/model.py's DeepLowRankModule (r = eta1 * D1(e_prev)),
    NOT odls_v2's residual-subtraction denoiser reframing;
  - cross-phase attention is removed entirely.

Purpose: odls_v2's combined model (sharing + denoiser + attention) has
underperformed the ../odls baseline in testing so far, across two
different width settings. This variant exists to answer one specific
question -- is weight sharing itself responsible for that gap, isolated
from the other two changes? -- rather than continuing to guess at
hyperparameters for the fully-combined model.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseEmbedding(nn.Module):
    def __init__(self, n_phases: int, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_phases, embed_dim)

    def forward(self, phase_idx: int, device: torch.device) -> torch.Tensor:
        idx = torch.tensor([phase_idx], device=device)
        return self.embedding(idx)  # (1, embed_dim)


class FiLM(nn.Module):
    def __init__(self, embed_dim: int, n_channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(embed_dim, 2 * n_channels)
        nn.init.zeros_(self.to_gamma_beta.weight)
        with torch.no_grad():
            self.to_gamma_beta.bias[:n_channels] = 1.0
            self.to_gamma_beta.bias[n_channels:] = 0.0

    def forward(self, x: torch.Tensor, phase_embed: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_gamma_beta(phase_embed)
        n_channels = x.shape[1]
        gamma, beta = gamma_beta[:, :n_channels], gamma_beta[:, n_channels:]
        gamma = gamma.view(1, n_channels, 1)
        beta = beta.view(1, n_channels, 1)
        return gamma * x + beta


class PhaseScalarHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.to_offset = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.to_offset.weight)
        nn.init.zeros_(self.to_offset.bias)

    def forward(self, base_value: torch.Tensor, phase_embed: torch.Tensor) -> torch.Tensor:
        offset = self.to_offset(phase_embed).view(())
        return base_value + offset


class _FiLMConvStack(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_layers: int, width: int,
                 embed_dim: int, kernel_size: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.films = nn.ModuleList()
        c_in = in_ch
        self.n_layers = n_layers
        for i in range(n_layers):
            c_out = width if i < n_layers - 1 else out_ch
            self.convs.append(nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(c_out))
                self.films.append(FiLM(embed_dim, c_out))
            c_in = c_out

    def forward(self, x: torch.Tensor, phase_embed: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < self.n_layers - 1:
                x = self.bns[i](x)
                x = self.films[i](x, phase_embed)
                x = F.relu(x, inplace=True)
        return x


class SharedLowRankModule(nn.Module):
    """Same role and framing as ../odls/model.py's DeepLowRankModule
    (r = eta1 * D1(e_prev), the original low-rank-motivated design --
    NOT odls_v2's residual-subtraction denoiser reframing), but D1 is a
    single shared, FiLM-conditioned network used at every phase instead
    of 10 independently-trained copies."""

    def __init__(self, channels: int, embed_dim: int, width: int = 48, n_layers: int = 6):
        super().__init__()
        self.net = _FiLMConvStack(channels, channels, n_layers, width, embed_dim)
        self.eta1_base = nn.Parameter(torch.tensor(0.001))
        self.eta1_head = PhaseScalarHead(embed_dim)

    def forward(self, e_prev: torch.Tensor, phase_embed: torch.Tensor) -> torch.Tensor:
        eta1 = self.eta1_head(self.eta1_base, phase_embed)
        return eta1 * self.net(e_prev, phase_embed)


class SharedDataConsistencyModule(nn.Module):
    """Same physics as ../odls/model.py's DataConsistencyModule (Eq. 13's
    gradient-descent data-fidelity step); phase-aware only through eta2."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.eta2_base = nn.Parameter(torch.tensor(1.0))
        self.eta2_head = PhaseScalarHead(embed_dim)

    def forward(self, e_prev: torch.Tensor, r: torch.Tensor, z: torch.Tensor,
                mask: torch.Tensor, phase_embed: torch.Tensor) -> torch.Tensor:
        eta2 = self.eta2_head(self.eta2_base, phase_embed)
        data_term = mask * (mask * e_prev - z)
        grad = data_term + 2.0 * r
        return e_prev - eta2 * grad


class SharedDeepSparseModule(nn.Module):
    """Same role as ../odls/model.py's DeepSparseModule (learned
    sparsifying transform pair + soft-threshold); Psi2/Psi3 are a single
    shared, FiLM-conditioned pair instead of 10 independent copies, but
    -- unlike odls_v2 -- there is no cross-phase attention here."""

    def __init__(self, channels: int, embed_dim: int, width: int = 48,
                 n_layers: int = 3, max_threshold: float = 0.05):
        super().__init__()
        self.psi2 = _FiLMConvStack(channels, channels, n_layers, width, embed_dim)
        self.psi3 = _FiLMConvStack(channels, channels, n_layers, width, embed_dim)
        self.threshold_base = nn.Parameter(torch.tensor(0.001))
        self.threshold_head = PhaseScalarHead(embed_dim)
        self.max_threshold = max_threshold

    @staticmethod
    def _ifft_pe(k_space: torch.Tensor, n_coils: int) -> torch.Tensor:
        """Properly centered inverse FFT: ifftshift-transform-fftshift."""
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

    def _soft_threshold(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        lam = torch.clamp(lam, min=0.0, max=self.max_threshold)
        return torch.sign(x) * torch.relu(torch.abs(x) - lam)

    def transform_and_reconstruct(self, d: torch.Tensor, n_coils: int,
                                   phase_embed: torch.Tensor):
        img = self._ifft_pe(d, n_coils)
        transformed = self.psi3(self.psi2(img, phase_embed), phase_embed)
        threshold = self.threshold_head(self.threshold_base, phase_embed)
        sparsified = self._soft_threshold(transformed, threshold)
        e_next = self._fft_pe(sparsified, n_coils)
        return e_next, transformed, img

    def clamp_thresholds_(self):
        with torch.no_grad():
            self.threshold_base.data.clamp_(min=0.0, max=self.max_threshold)


class ODLSAblationSharing(nn.Module):
    """Weight-sharing-only ablation: same phase-embedding/FiLM sharing
    machinery as ODLSv2, but the original low-rank module framing (not
    the denoiser reframing) and no cross-phase attention."""

    def __init__(self, n_coils: int, n_phases: int = 10, width: int = 48,
                 max_threshold: float = 0.05, embed_dim: int = 16):
        super().__init__()
        self.n_coils = n_coils
        self.n_phases = n_phases
        channels = 2 * n_coils

        self.phase_embedding = PhaseEmbedding(n_phases, embed_dim)
        self.low_rank = SharedLowRankModule(channels, embed_dim, width=width, n_layers=6)
        self.dc = SharedDataConsistencyModule(embed_dim)
        self.sparse = SharedDeepSparseModule(channels, embed_dim, width=width, n_layers=3,
                                              max_threshold=max_threshold)
        self._xavier_init()

    def _xavier_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def clamp_thresholds_(self) -> None:
        self.sparse.clamp_thresholds_()

    def forward(self, z: torch.Tensor, mask: torch.Tensor):
        device = z.device
        e = z
        e_preds = []
        sym_pairs = []

        for k in range(self.n_phases):
            phase_embed = self.phase_embedding(k, device)

            r = self.low_rank(e, phase_embed)
            d = self.dc(e, r, z, mask, phase_embed)
            e, transformed, img = self.sparse.transform_and_reconstruct(d, self.n_coils, phase_embed)

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
