"""
Loss function for ODLS, Eq. 15:

  L_total(Theta_ODLS) = L_err + L_sym

  L_err = (1/KT) * sum_k sum_t || Psi*_PE(e_ref,t - e_m^(k,t)) ||_2^2
  L_sym = (1/KT) * sum_k sum_t || Psi3(Psi2(Psi*_PE e_m^(k,t))) - Psi*_PE e_m^(k,t) ||_2^2

K is the number of unrolled phases, T the number of training samples in
the batch. The symmetry term encourages Psi2 / Psi3 to behave like an
invertible transform pair (Sec. III-C.3), and is weighted by 0.01
relative to the error term in the paper's ablation description.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import torch
import torch.nn as nn


class ODLSLoss(nn.Module):
    def __init__(self, n_coils: int, sym_weight: float = 0.01):
        super().__init__()
        self.n_coils = n_coils
        self.sym_weight = sym_weight

    def _to_image(self, k_space: torch.Tensor) -> torch.Tensor:
        real, imag = k_space[:, : self.n_coils], k_space[:, self.n_coils :]
        complex_ks = torch.complex(real, imag)
        img = torch.fft.ifft(torch.fft.ifftshift(complex_ks, dim=-1), dim=-1, norm="ortho")
        return torch.cat([img.real, img.imag], dim=1)

    def forward(self, e_ref: torch.Tensor, e_preds: List[torch.Tensor],
                sym_pairs: List[Tuple[torch.Tensor, torch.Tensor]]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        e_ref:    (batch, 2*n_coils, N) fully-sampled label hybrid data
        e_preds:  list of length K, one reconstruction e^(k) per unrolled
                  phase (model.ODLS.forward's first return value)
        sym_pairs: list of length K of (transformed, img) tuples where
                  transformed = Psi3(Psi2(img)) and img = Psi_PE^{-1}(d^(k))
                  (model.ODLS.forward's second return value). The symmetry
                  loss compares these two directly, per Eq. 15 -- it does
                  NOT involve e_ref or the post-threshold reconstruction.
        """
        ref_img = self._to_image(e_ref)
        K = len(e_preds)

        err_loss = ref_img.new_zeros(())
        sym_loss = ref_img.new_zeros(())
        for e_k, (transformed, img) in zip(e_preds, sym_pairs):
            pred_img = self._to_image(e_k)
            err_loss = err_loss + torch.mean((ref_img - pred_img) ** 2)
            sym_loss = sym_loss + torch.mean((transformed - img) ** 2)
        err_loss = err_loss / K
        sym_loss = sym_loss / K

        total = err_loss + self.sym_weight * sym_loss
        return total, {"err": err_loss.detach(), "sym": sym_loss.detach()}
