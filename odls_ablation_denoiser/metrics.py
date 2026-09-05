"""
Evaluation criteria, Supplementary Material S1:

  RLNE = ||x - x_hat||_2 / ||x||_2                                (S1-1)
  PSNR = 10 * log10( M*N*max(|x|)^2 / ||x - x_hat||_2^2 )         (S1-2)
  SSIM = standard structural similarity index (Zhou et al., 2004) (S1-3)

x, x_hat are column-stacked fully-sampled / reconstructed magnitude
images after coil combination by square-root-of-sum-of-squares (SoS).
"""

from __future__ import annotations

import torch


def coil_combine_sos(image_channels: torch.Tensor, n_coils: int) -> torch.Tensor:
    """Square-root of sum of squares coil combination.

    image_channels: (batch, 2*n_coils, N) real/imag-stacked multi-coil image
    returns:        (batch, N) magnitude image
    """
    real, imag = image_channels[:, :n_coils], image_channels[:, n_coils:]
    mag_sq = real ** 2 + imag ** 2
    return torch.sqrt(mag_sq.sum(dim=1) + 1e-12)


def rlne(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Eq. S1-1. x, x_hat: (batch, ...) magnitude images."""
    x_flat = x.flatten(1)
    xh_flat = x_hat.flatten(1)
    num = torch.linalg.norm(x_flat - xh_flat, dim=1)
    den = torch.linalg.norm(x_flat, dim=1) + eps
    return num / den


def psnr(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Eq. S1-2. x, x_hat: (batch, M, N) or (batch, N) magnitude images."""
    x_flat = x.flatten(1)
    xh_flat = x_hat.flatten(1)
    n_pixels = x_flat.shape[1]
    peak = x_flat.max(dim=1).values
    mse_sum = torch.sum((x_flat - xh_flat) ** 2, dim=1) + eps
    return 10.0 * torch.log10(n_pixels * peak ** 2 / mse_sum)


def ssim(x: torch.Tensor, x_hat: torch.Tensor, data_range: float = None,
         c1_scale: float = 0.01, c2_scale: float = 0.03) -> torch.Tensor:
    """Eq. S1-3, global (whole-image) SSIM as used for RLNE/PSNR reporting
    alongside it in the paper's tables (single scalar per image, not a
    sliding-window map).

    x, x_hat: (batch, M, N) magnitude images.
    """
    x_flat = x.flatten(1)
    xh_flat = x_hat.flatten(1)

    if data_range is None:
        data_range = x_flat.max(dim=1, keepdim=True).values

    c1 = (c1_scale * data_range) ** 2
    c2 = (c2_scale * data_range) ** 2

    mu_x = x_flat.mean(dim=1, keepdim=True)
    mu_xh = xh_flat.mean(dim=1, keepdim=True)
    var_x = x_flat.var(dim=1, unbiased=False, keepdim=True)
    var_xh = xh_flat.var(dim=1, unbiased=False, keepdim=True)
    cov = ((x_flat - mu_x) * (xh_flat - mu_xh)).mean(dim=1, keepdim=True)

    numerator = (2 * mu_x * mu_xh + c1) * (2 * cov + c2)
    denominator = (mu_x ** 2 + mu_xh ** 2 + c1) * (var_x + var_xh + c2)
    return (numerator / denominator).squeeze(1)
