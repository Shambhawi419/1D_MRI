"""
ODLS-v2: a research extension of the faithful replication in ../odls/model.py.

This is NOT a reproduction of arXiv:2112.04721 -- it's a deliberate departure
from it, exploring three architectural changes on top of the same 1D
unrolled DLR/DC/DS skeleton:

  1. Weight sharing across phases: one shared network each for the
     low-rank-prior module, and the sparse-transform pair, instead of 10
     independently-trained copies of each -- conditioned on a learnable
     per-phase embedding (via FiLM) so phase-specific behavior survives
     sharing. ~7-8x fewer parameters; each shared parameter now receives
     gradient signal from every phase every step.
  2. The low-rank module (N1) is replaced by a general-purpose residual
     ("DnCNN-style") denoiser: output = input - predicted_residual,
     framed as "clean up artifacts" rather than "detect low-rank
     violations" -- still just a 1D CNN under the hood, so the paper's
     efficiency argument for 1D convolution still applies.
  3. Cross-phase attention: each of the three shared modules keeps a
     memory of its own past phases' outputs; the current phase attends
     back over that memory (pooled queries/keys, full feature-map
     values) before proceeding, so later phases have direct access to
     earlier intermediate states instead of only the immediately
     preceding phase's output.

Everything else (data consistency module's physics, the overall
DLR -> DC -> DS unrolling, the symmetry-loss contract, the checkpoint
format's high-level shape) is kept as close to ../odls/model.py as
sharing allows, so losses.py / train.py / evaluate.py stay structurally
similar to the baseline implementation.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase conditioning: one learnable embedding per phase, used both to FiLM-
# modulate the shared conv stacks and to nudge the per-phase scalar
# parameters (eta1, eta2, threshold) away from a single shared value.
# ---------------------------------------------------------------------------

class PhaseEmbedding(nn.Module):
    def __init__(self, n_phases: int, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_phases, embed_dim)

    def forward(self, phase_idx: int, device: torch.device) -> torch.Tensor:
        idx = torch.tensor([phase_idx], device=device)
        return self.embedding(idx)  # (1, embed_dim)


class FiLM(nn.Module):
    """Feature-wise linear modulation: gamma/beta computed from the phase
    embedding, applied per-channel after a conv+BN layer. This is how a
    single shared conv stack gets phase-specific behavior instead of
    reverting to 10 independent copies of the network."""

    def __init__(self, embed_dim: int, n_channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(embed_dim, 2 * n_channels)
        # init near-identity (gamma~1, beta~0) so early training behaves
        # like the un-modulated shared network before FiLM has learned
        # anything useful yet.
        nn.init.zeros_(self.to_gamma_beta.weight)
        with torch.no_grad():
            self.to_gamma_beta.bias[:n_channels] = 1.0
            self.to_gamma_beta.bias[n_channels:] = 0.0

    def forward(self, x: torch.Tensor, phase_embed: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_gamma_beta(phase_embed)  # (1, 2*C)
        n_channels = x.shape[1]
        gamma, beta = gamma_beta[:, :n_channels], gamma_beta[:, n_channels:]
        gamma = gamma.view(1, n_channels, 1)
        beta = beta.view(1, n_channels, 1)
        return gamma * x + beta


class PhaseScalarHead(nn.Module):
    """Maps a phase embedding to a small additive offset on a shared base
    scalar (eta1 / eta2 / threshold), so each of the 10 phases can still
    land on a different effective value despite the base parameter (and
    the conv stack producing it) being shared."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.to_offset = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.to_offset.weight)
        nn.init.zeros_(self.to_offset.bias)

    def forward(self, base_value: torch.Tensor, phase_embed: torch.Tensor) -> torch.Tensor:
        offset = self.to_offset(phase_embed).view(())
        return base_value + offset


# ---------------------------------------------------------------------------
# Cross-phase attention: a small memory bank of each shared module's past
# outputs, attended over via pooled query/key vectors with full feature
# maps as values.
# ---------------------------------------------------------------------------

class CrossPhaseAttention(nn.Module):
    """At phase k, attends the current feature map back over every stored
    feature map from phases 0..k-1 (same module type), returning a
    residual context to add before proceeding. Query/key are computed
    from global-average-pooled channel descriptors (cheap, avoids an
    attention matrix sized by the full spatial extent); values are the
    full stored feature maps, so spatial detail from earlier phases is
    still recoverable, not just a pooled summary."""

    def __init__(self, n_channels: int, key_dim: int = 32):
        super().__init__()
        self.query_proj = nn.Linear(n_channels, key_dim)
        self.key_proj = nn.Linear(n_channels, key_dim)
        self.out_scale = nn.Parameter(torch.tensor(0.0))  # starts as a no-op residual

    def forward(self, current: torch.Tensor, memory: List[torch.Tensor]) -> torch.Tensor:
        """current: (batch, C, N). memory: list of earlier (batch, C, N)
        feature maps from this same module (may be empty, e.g. phase 0)."""
        if not memory:
            return current

        pooled_current = current.mean(dim=-1)  # (batch, C)
        query = self.query_proj(pooled_current)  # (batch, key_dim)

        stacked = torch.stack(memory, dim=1)  # (batch, T, C, N)
        pooled_memory = stacked.mean(dim=-1)  # (batch, T, C)
        keys = self.key_proj(pooled_memory)  # (batch, T, key_dim)

        scores = torch.einsum("bd,btd->bt", query, keys) / (keys.shape[-1] ** 0.5)
        weights = F.softmax(scores, dim=-1)  # (batch, T)

        context = torch.einsum("bt,btcn->bcn", weights, stacked)  # (batch, C, N)
        return current + self.out_scale * context


# ---------------------------------------------------------------------------
# Shared conv-stack builder, reused for both the denoiser and the sparse
# transform pair, each with its own FiLM layers per conv block.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared modules (one instance each, used at every phase)
# ---------------------------------------------------------------------------

class SharedDenoiserModule(nn.Module):
    """Replaces the paper's low-rank-motivated N1 with a general-purpose
    residual denoiser (direction 2): output = input - predicted_residual,
    framed as artifact removal rather than low-rank-violation detection.
    Still a plain 1D CNN, so the paper's efficiency argument for 1D
    convolution over 2D is untouched -- only the *framing* and the
    residual-learning formulation change.
    """

    def __init__(self, channels: int, embed_dim: int, width: int = 48, n_layers: int = 6):
        super().__init__()
        self.net = _FiLMConvStack(channels, channels, n_layers, width, embed_dim)
        self.eta1_base = nn.Parameter(torch.tensor(0.001))
        self.eta1_head = PhaseScalarHead(embed_dim)
        self.attention = CrossPhaseAttention(channels)

    def forward(self, e_prev: torch.Tensor, phase_embed: torch.Tensor,
                memory: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.net(e_prev, phase_embed)
        denoised_delta = e_prev - residual  # DnCNN-style residual learning
        denoised_delta = self.attention(denoised_delta, memory)
        eta1 = self.eta1_head(self.eta1_base, phase_embed)
        r = eta1 * denoised_delta
        return r, denoised_delta  # second value cached into this phase's memory


class SharedDataConsistencyModule(nn.Module):
    """Same physics as ../odls/model.py's DataConsistencyModule (Eq. 13's
    gradient-descent data-fidelity step); made phase-aware only through
    eta2 so the step size can still vary per phase under sharing."""

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
    sparsifying transform pair + soft-threshold), but Psi2/Psi3 are
    shared across phases (FiLM-conditioned) and the threshold gets both
    a per-phase offset and the same runaway-growth cap used in the
    baseline (max_threshold), since nothing about weight sharing removes
    the underlying incentive that caused the threshold explosion there.
    """

    def __init__(self, channels: int, embed_dim: int, width: int = 48,
                 n_layers: int = 3, max_threshold: float = 0.05):
        super().__init__()
        self.psi2 = _FiLMConvStack(channels, channels, n_layers, width, embed_dim)
        self.psi3 = _FiLMConvStack(channels, channels, n_layers, width, embed_dim)
        self.threshold_base = nn.Parameter(torch.tensor(0.001))
        self.threshold_head = PhaseScalarHead(embed_dim)
        self.max_threshold = max_threshold
        self.attention = CrossPhaseAttention(channels)

    @staticmethod
    def _ifft_pe(k_space: torch.Tensor, n_coils: int) -> torch.Tensor:
        """Properly centered inverse FFT: ifftshift-transform-fftshift.
        See ../odls/model.py's DeepSparseModule._ifft_pe docstring for why
        the closing fftshift matters (was missing in an earlier version of
        this same function, inherited from the baseline before it was
        independently fixed there)."""
        real, imag = k_space[:, :n_coils], k_space[:, n_coils:]
        complex_ks = torch.complex(real, imag)
        img = torch.fft.fftshift(
            torch.fft.ifft(torch.fft.ifftshift(complex_ks, dim=-1), dim=-1, norm="ortho"), dim=-1
        )
        return torch.cat([img.real, img.imag], dim=1)

    @staticmethod
    def _fft_pe(image: torch.Tensor, n_coils: int) -> torch.Tensor:
        """Forward FFT inverting _ifft_pe; needs the matching opening
        ifftshift to correctly undo _ifft_pe's now-complete centering."""
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
                                   phase_embed: torch.Tensor, memory: List[torch.Tensor]):
        img = self._ifft_pe(d, n_coils)
        transformed = self.psi3(self.psi2(img, phase_embed), phase_embed)
        transformed = self.attention(transformed, memory)
        threshold = self.threshold_head(self.threshold_base, phase_embed)
        sparsified = self._soft_threshold(transformed, threshold)
        e_next = self._fft_pe(sparsified, n_coils)
        return e_next, transformed, img

    def clamp_thresholds_(self):
        with torch.no_grad():
            self.threshold_base.data.clamp_(min=0.0, max=self.max_threshold)


# ---------------------------------------------------------------------------
# Full network: K phases, each using the SAME three shared modules,
# conditioned on a per-phase embedding.
# ---------------------------------------------------------------------------

class ODLSv2(nn.Module):
    """Weight-shared, denoiser-prior, cross-phase-attention variant of the
    baseline ODLS in ../odls/model.py. See module docstring for the three
    architectural changes this makes relative to arXiv:2112.04721.
    """

    def __init__(self, n_coils: int, n_phases: int = 10, width: int = 48,
                 max_threshold: float = 0.05, embed_dim: int = 16):
        super().__init__()
        self.n_coils = n_coils
        self.n_phases = n_phases
        channels = 2 * n_coils

        self.phase_embedding = PhaseEmbedding(n_phases, embed_dim)
        self.denoiser = SharedDenoiserModule(channels, embed_dim, width=width, n_layers=6)
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
        """Kept for interface parity with the baseline's ODLS.clamp_thresholds_
        (called by checkpoint_utils.load_model_weights) -- only one shared
        threshold parameter exists now, so this simply clamps that one."""
        self.sparse.clamp_thresholds_()

    def forward(self, z: torch.Tensor, mask: torch.Tensor):
        """
        z:    (batch, 2*n_coils, N) zero-filled undersampled hybrid k-space
        mask: (batch, 1, N) binary undersampling mask along the PE axis

        Returns the same shape of outputs as the baseline ODLS.forward:
        e_preds (list of length K) and sym_pairs (list of length K of
        (transformed, img) tuples), so losses.py needs no changes.
        """
        device = z.device
        e = z
        e_preds = []
        sym_pairs = []
        denoiser_memory: List[torch.Tensor] = []
        sparse_memory: List[torch.Tensor] = []

        for k in range(self.n_phases):
            phase_embed = self.phase_embedding(k, device)

            r, denoised_delta = self.denoiser(e, phase_embed, denoiser_memory)
            denoiser_memory.append(denoised_delta.detach())

            d = self.dc(e, r, z, mask, phase_embed)

            e, transformed, img = self.sparse.transform_and_reconstruct(
                d, self.n_coils, phase_embed, sparse_memory
            )
            sparse_memory.append(transformed.detach())

            e_preds.append(e)
            sym_pairs.append((transformed, img))

        return e_preds, sym_pairs

    def reconstruct_image(self, e_final: torch.Tensor) -> torch.Tensor:
        """Includes the closing fftshift a properly centered inverse FFT
        requires -- see SharedDeepSparseModule._ifft_pe's docstring."""
        real, imag = e_final[:, : self.n_coils], e_final[:, self.n_coils :]
        complex_ks = torch.complex(real, imag)
        img = torch.fft.fftshift(
            torch.fft.ifft(torch.fft.ifftshift(complex_ks, dim=-1), dim=-1, norm="ortho"), dim=-1
        )
        return torch.cat([img.real, img.imag], dim=1)
