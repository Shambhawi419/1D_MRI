"""
Eq. 15 of the paper, for dynamic samples:

  L = (1/K) sum_k mean| Psi_PE^-1(e_ref - e^(k)) |^2
      + sym_weight * (1/K) sum_k mean| Psi3(Psi2(x^(k))) - x^(k) |^2

The error term is measured in image space (PE inverse FFT, frames as-is);
sym_weight = 0.01 as in the paper. Computed in fp32 even under autocast.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from transforms import PE_DIM, ifftc


class ODLSLoss(nn.Module):
    def __init__(self, sym_weight: float = 0.01):
        super().__init__()
        self.sym_weight = sym_weight

    def forward(self, e_ref: torch.Tensor, e_preds: List[torch.Tensor],
                sym_pairs: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ref_img = ifftc(e_ref, PE_DIM)
        err = ref_img.new_zeros(())
        sym = ref_img.new_zeros(())
        n_sym = 0
        for e_k, pair in zip(e_preds, sym_pairs):
            err = err + torch.mean((ref_img - ifftc(e_k, PE_DIM)) ** 2)
            if pair is not None:
                transformed, x = pair
                sym = sym + torch.mean((transformed.float() - x.float()) ** 2)
                n_sym += 1
        err = err / len(e_preds)
        sym = sym / max(n_sym, 1)
        total = err + self.sym_weight * sym
        return total, {"err": err.detach(), "sym": sym.detach()}
