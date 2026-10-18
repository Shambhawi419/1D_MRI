"""
Standard (windowed) SSIM, as used by most MRI reconstruction papers.

Why: odls_dynamic reports SSIM with the ODLS paper's definition (Supplement
S1-3), which computes ONE mean / variance / covariance over the whole image.
A large dark background makes that "global" SSIM very high (~99%).

Windowed SSIM (Wang et al., IEEE TIP 2004) computes SSIM inside a small
Gaussian window around every pixel and averages the resulting map, so it
reacts to local structure. Settings are the usual ones:
    Gaussian window, sigma = 1.5, 11 x 11 (truncated at 3.5 sigma)
    K1 = 0.01, K2 = 0.03
    data range = max of the reference frame
    only pixels where the window fits completely inside the image ("valid")
This matches skimage.metrics.structural_similarity(..., gaussian_weights=True,
sigma=1.5, use_sample_covariance=False, data_range=...) -- selftest_plus.py
checks that when scikit-image is installed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_window(sigma: float = 1.5, truncate: float = 3.5) -> torch.Tensor:
    """2D Gaussian window, normalised to sum 1. Size 2*int(truncate*sigma+0.5)+1 = 11."""
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim_windowed(ref: np.ndarray, est: np.ndarray, sigma: float = 1.5,
                  k1: float = 0.01, k2: float = 0.03) -> np.ndarray:
    """ref, est: (T, H, W) magnitude frames. Returns one SSIM value per frame, shape (T,)."""
    r = torch.from_numpy(np.asarray(ref, dtype=np.float64))[:, None]   # (T, 1, H, W)
    e = torch.from_numpy(np.asarray(est, dtype=np.float64))[:, None]
    w = gaussian_window(sigma)[None, None]                              # (1, 1, k, k)

    def blur(x):
        return F.conv2d(x, w)                                           # "valid": no padding

    data_range = r.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)       # per frame
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2

    mu_r, mu_e = blur(r), blur(e)
    var_r = blur(r * r) - mu_r ** 2
    var_e = blur(e * e) - mu_e ** 2
    cov = blur(r * e) - mu_r * mu_e

    ssim_map = ((2 * mu_r * mu_e + c1) * (2 * cov + c2)) / ((mu_r ** 2 + mu_e ** 2 + c1) * (var_r + var_e + c2))
    return ssim_map.flatten(1).mean(dim=1).numpy()
