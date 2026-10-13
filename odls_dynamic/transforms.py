"""
Centered FFT helpers and the real/imag channel convention, shared by the
data pipeline (NumPy), the model, the loss and evaluation (PyTorch).

Tensor convention for one dynamic sample (one FE row across all frames):
    complex (N, J, T)  <->  real (2*J, N, T): J real channels, then J imaginary
N = ky / phase-encoding, J = coils, T = cardiac frames. Batched: (B, 2J, N, T).

Every transform is the properly centered ifftshift -> (i)fft -> fftshift,
orthonormal, i.e. the version ../odls ended up with after its centering fix.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------- NumPy ----

def np_ifftc(x: np.ndarray, axis: int) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis, norm="ortho"), axes=axis)


def np_fftc(x: np.ndarray, axis: int) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis, norm="ortho"), axes=axis)


def np_to_channels(x: np.ndarray) -> np.ndarray:
    """(N, J, T) complex -> (2J, N, T) float32."""
    x = np.transpose(x, (1, 0, 2))
    return np.concatenate([x.real, x.imag], axis=0).astype(np.float32)


# -------------------------------------------------------------- PyTorch ----

def to_complex(x: torch.Tensor) -> torch.Tensor:
    """(B, 2J, ...) real -> (B, J, ...) complex64. Always float32 underneath:
    cuFFT in half precision only supports power-of-two sizes, and OCMR's ky
    sizes (104-180) are not, so FFTs stay in fp32 even under autocast."""
    j = x.shape[1] // 2
    x = x.float()
    return torch.complex(x[:, :j], x[:, j:])


def to_channels(z: torch.Tensor) -> torch.Tensor:
    return torch.cat([z.real, z.imag], dim=1)


def ifftc(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Centered orthonormal inverse FFT of a channel-stacked real tensor."""
    z = to_complex(x)
    z = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(z, dim=dim), dim=dim, norm="ortho"), dim=dim)
    return to_channels(z)


def fftc(x: torch.Tensor, dim: int) -> torch.Tensor:
    z = to_complex(x)
    z = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(z, dim=dim), dim=dim, norm="ortho"), dim=dim)
    return to_channels(z)


# Axis names for (B, 2J, N, T) tensors.
PE_DIM = -2
TIME_DIM = -1
