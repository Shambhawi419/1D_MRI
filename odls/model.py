"""
ODLS: One-dimensional Deep Low-rank and Sparse Network for Accelerated MRI
Zi Wang, Chen Qian, Di Guo, Hongwei Sun, Rushuai Li, Bo Zhao, Xiaobo Qu
arXiv:2112.04721

Implements the unrolled K-phase network of Fig. 4:
  DLR  (deep low-rank module,  Eq. 12)
  DC   (data consistency module, Eq. 13)
  DS   (deep sparse module,    Eq. 14)

Data convention
----------------
A 1D hybrid-domain training sample e_m is a complex vector of length N
(the phase-encoding dimension) with J coils. It is represented as a real
tensor of shape (batch, 2*J, N): J real channels followed by J imaginary
channels, so all convolutions are ordinary real-valued 1D convolutions
(as used in the paper's TensorFlow implementation).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_conv_stack(in_ch: int, out_ch: int, n_layers: int, width: int, kernel_size: int = 3):
    """`n_layers` 1D conv layers of `width` filters, BN + ReLU, as described
    in Sec. III-D ("Each convolutional layer contains 48 1D convolution
    filters of size 3, followed by batch normalization and ReLU")."""
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


class DeepLowRankModule(nn.Module):
    """Eq. 12: r_m^(k) = eta1^(k) * D1(e_m^(k-1))

    D1 is a multi-layer encoder-decoder 1D CNN that replaces the
    null-space filterbank Q of the structured low-rank model (Sec. III-C.1).
    Six conv layers, 48 filters, kernel size 3 (Sec. III-D).
    """

    def __init__(self, channels: int, width: int = 48, n_layers: int = 6):
        super().__init__()
        self.D1 = _make_conv_stack(channels, channels, n_layers, width)
        # learnable scalar eta1, initialized to 0.001 (Sec. III-C.1)
        self.eta1 = nn.Parameter(torch.tensor(0.001))

    def forward(self, e_prev: torch.Tensor) -> torch.Tensor:
        return self.eta1 * self.D1(e_prev)


class DataConsistencyModule(nn.Module):
    """Eq. 13: d_m^(k) = e_m^(k-1) - eta2^(k) * [ Lambda*(Lambda e_m^(k-1) - z_m) + 2 r_m^(k) ]

    Lambda is the (real-valued, per-channel) undersampling mask applied in
    the hybrid k-space domain -- the same domain the acquired data z_m
    lives in, since PE undersampling directly subsamples this axis
    (Sec. II, Eq. 1). Lambda is self-adjoint (a diagonal 0/1 mask), so
    Lambda* == Lambda.

    eta2 is a learnable scalar step size, initialized to 1 (Sec. III-C.2).
    """

    def __init__(self):
        super().__init__()
        self.eta2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, e_prev: torch.Tensor, r: torch.Tensor,
                z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch, 1, N) or (batch, C, N) broadcastable 0/1 tensor
        data_term = mask * (mask * e_prev - z)
        grad = data_term + 2.0 * r
        return e_prev - self.eta2 * grad


class DeepSparseModule(nn.Module):
    """Eq. 14: e_m^(k) = Psi_PE^{-1}[ soft( Psi3(Psi2(Psi_PE^{-1} d_m^(k))); lambda^(k) ) ]

    Psi_PE^{-1} is the fixed 1D PE inverse Fourier transform taking the
    hybrid k-space signal to image space (Sec. II). Psi2 / Psi3 are the
    learned forward/inverse sparsifying transforms (two 3-layer 1D CNNs,
    Sec. III-C.3), and lambda^(k) is a learnable soft threshold
    initialized to 0.001, allowed to vary per phase.

    `max_threshold` caps how large lambda^(k) can grow. The paper doesn't
    specify a ceiling, but nothing else in the architecture stops gradient
    descent from inflating the threshold without bound -- observed in
    practice (fastMRI CORPD_FBK training) growing 250-800x from its 0.001
    init within ~17 epochs, at which point it was suppressing nearly all
    signal through the sparsifying step (a soft-threshold that large zeros
    out most values it's applied to), producing a near-all-zero output and
    a worsening validation loss over subsequent epochs. Capping it keeps
    the shrinkage useful without letting it collapse the output.
    """

    def __init__(self, channels: int, width: int = 48, n_layers: int = 3,
                 max_threshold: float = 0.05):
        super().__init__()
        self.psi2 = _make_conv_stack(channels, channels, n_layers, width)  # forward transform
        self.psi3 = _make_conv_stack(channels, channels, n_layers, width)  # inverse transform
        self.threshold = nn.Parameter(torch.tensor(0.001))
        self.max_threshold = max_threshold

    def _soft_threshold(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        # clamp 0 <= lam <= max_threshold: a negative threshold would invert
        # soft-thresholding into amplification (never intended for a
        # learnable shrinkage threshold), and an unbounded-above threshold
        # can grow large enough to suppress nearly all signal (observed in
        # practice -- see class docstring).
        lam = torch.clamp(lam, min=0.0, max=self.max_threshold)
        return torch.sign(x) * torch.relu(torch.abs(x) - lam)

    @staticmethod
    def _ifft_pe(k_space: torch.Tensor, n_coils: int) -> torch.Tensor:
        """1D inverse FFT along the PE axis, applied per coil, on the
        (real, imag) channel-stacked representation.

        A properly centered inverse FFT needs ifftshift-transform-fftshift;
        this was previously missing the closing fftshift, leaving the true
        image center wrapped out to the array's corners instead of sitting
        in the middle -- a needlessly awkward, artificially-split input
        shape for psi2/psi3's local convolutions to process. See
        data.py's fe_ifft docstring for the full explanation.
        """
        real, imag = k_space[:, :n_coils], k_space[:, n_coils:]
        complex_ks = torch.complex(real, imag)
        img = torch.fft.fftshift(
            torch.fft.ifft(torch.fft.ifftshift(complex_ks, dim=-1), dim=-1, norm="ortho"), dim=-1
        )
        return torch.cat([img.real, img.imag], dim=1)

    @staticmethod
    def _fft_pe(image: torch.Tensor, n_coils: int) -> torch.Tensor:
        """Forward FFT inverting _ifft_pe -- needs the matching opening
        ifftshift (previously missing) so it correctly undoes _ifft_pe's
        now-complete centering rather than only half of it."""
        real, imag = image[:, :n_coils], image[:, n_coils:]
        complex_img = torch.complex(real, imag)
        ks = torch.fft.fftshift(
            torch.fft.fft(torch.fft.ifftshift(complex_img, dim=-1), dim=-1, norm="ortho"), dim=-1
        )
        return torch.cat([ks.real, ks.imag], dim=1)

    def transform_and_reconstruct(self, d: torch.Tensor, n_coils: int):
        """Runs Psi_PE^{-1} then Psi3(Psi2(.)) once and returns:
          - e_next:     the soft-thresholded k-space output e^(k) (Eq. 14)
          - transformed: Psi3(Psi2(x))
          - img (x):     Psi_PE^{-1}(d), the *pre-threshold* transform input

        The symmetry loss (Eq. 15) is ||Psi3(Psi2(x)) - x||^2, i.e. it
        compares `transformed` against `img`, not against the final
        (post-threshold) reconstruction -- both are returned so the loss
        can be computed correctly without a redundant CNN forward pass.
        """
        img = self._ifft_pe(d, n_coils)
        transformed = self.psi3(self.psi2(img))
        sparsified = self._soft_threshold(transformed, self.threshold)
        e_next = self._fft_pe(sparsified, n_coils)
        return e_next, transformed, img

    def forward(self, d: torch.Tensor, n_coils: int) -> torch.Tensor:
        e_next, _, _ = self.transform_and_reconstruct(d, n_coils)
        return e_next


class ODLSPhase(nn.Module):
    """One unrolled iteration: DLR -> DC -> DS (Fig. 4(a))."""

    def __init__(self, n_coils: int, width: int = 48, max_threshold: float = 0.05):
        super().__init__()
        channels = 2 * n_coils  # real + imaginary stacked
        self.n_coils = n_coils
        self.dlr = DeepLowRankModule(channels, width=width, n_layers=6)
        self.dc = DataConsistencyModule()
        self.ds = DeepSparseModule(channels, width=width, n_layers=3, max_threshold=max_threshold)

    def forward(self, e_prev: torch.Tensor, z: torch.Tensor, mask: torch.Tensor):
        r = self.dlr(e_prev)
        d = self.dc(e_prev, r, z, mask)
        e_next, transformed, img = self.ds.transform_and_reconstruct(d, self.n_coils)
        return e_next, transformed, img


class ODLS(nn.Module):
    """The full recursive ODLS network (Fig. 4(a)).

    K = 10 phases is reported as "an optimal trade-off between the
    reconstruction performance and time consumption" (Sec. III-D).
    """

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
        """In-place clamp of every phase's stored threshold parameter into
        [0, max_threshold]. The forward pass already clamps the threshold
        at use-time regardless (see DeepSparseModule._soft_threshold), so
        this doesn't change model behavior -- it exists to keep a loaded
        checkpoint's *stored* values consistent with the cap, rather than
        leaving a stale out-of-range number sitting in the parameter that
        could (a) still receive a few steps of Adam-momentum-driven drift
        even though it can no longer affect output, or (b) spring back to
        life with no real meaning if max_threshold is ever raised again in
        a later run. Call this right after loading a checkpoint's weights,
        before resuming or evaluating.
        """
        for phase in self.phases:
            ds = phase.ds
            ds.threshold.data.clamp_(min=0.0, max=ds.max_threshold)

    def forward(self, z: torch.Tensor, mask: torch.Tensor):
        """
        z:    (batch, 2*n_coils, N) zero-filled undersampled hybrid k-space
        mask: (batch, 1, N) binary undersampling mask along the PE axis

        Returns:
          e_preds:  list of length K, the reconstructed hybrid k-space
                    e^(k) after each phase (same shape as z), used for the
                    L_err term of Eq. 15.
          sym_pairs: list of length K of (transformed, img) tuples, where
                    transformed = Psi3(Psi2(img)) and img = Psi_PE^{-1}(d^(k)),
                    used for the L_sym term of Eq. 15.
        """
        e = z  # e^(0) = Lambda* z (zero-filled initialization, Sec. III-C.1)
        e_preds = []
        sym_pairs = []
        for phase in self.phases:
            e, transformed, img = phase(e, z, mask)
            e_preds.append(e)
            sym_pairs.append((transformed, img))
        return e_preds, sym_pairs

    def reconstruct_image(self, e_final: torch.Tensor) -> torch.Tensor:
        """Apply the final 1D PE inverse FFT to get the image-domain
        reconstruction (S_hat = Psi*_PE(E_hat), Sec. II). Includes the
        closing fftshift a properly centered inverse FFT requires -- see
        DeepSparseModule._ifft_pe's docstring."""
        real, imag = e_final[:, : self.n_coils], e_final[:, self.n_coils :]
        complex_ks = torch.complex(real, imag)
        img = torch.fft.fftshift(
            torch.fft.ifft(torch.fft.ifftshift(complex_ks, dim=-1), dim=-1, norm="ortho"), dim=-1
        )
        return torch.cat([img.real, img.imag], dim=1)
