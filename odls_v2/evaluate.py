"""
Evaluation script for ODLS-v2, mirroring ../odls/evaluate.py: reconstructs
held-out multi-coil k-space volumes and reports RLNE / PSNR / SSIM.

Reconstruction flow per slice:
  1. Take the 1D FE inverse FFT of the undersampled 2D k-space -> Z.
  2. Reconstruct every row of Z in parallel through the trained ODLSv2.
  3. Stitch rows back together -> E_hat.
  4. Take the 1D PE inverse FFT of E_hat -> final image S_hat.
  5. Coil-combine by square-root of sum of squares before scoring.
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm

from data import fe_ifft, pe_ifft, _to_real_imag_channels
from masks import MASK_FACTORIES
from metrics import coil_combine_sos, rlne, psnr, ssim
from model import ODLSv2
from fastmri_data import find_corpd_files, load_fastmri_volume
from checkpoint_utils import load_model_weights


def reconstruct_slice(model: ODLSv2, k_space_slice: np.ndarray, mask: np.ndarray,
                       device: str) -> np.ndarray:
    M, N, J = k_space_slice.shape
    hybrid_full = fe_ifft(k_space_slice)  # Psi*_FE(Y), (M, N, J)

    z_rows = hybrid_full * mask[None, :, None]
    z_ch = np.stack([_to_real_imag_channels(z_rows[m]) for m in range(M)], axis=0)  # (M, 2J, N)

    z_tensor = torch.from_numpy(z_ch).to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)
    mask_tensor = mask_tensor.view(1, 1, N).expand(M, 1, N)

    model.eval()
    with torch.no_grad():
        e_preds, _ = model(z_tensor, mask_tensor)
        e_final = e_preds[-1]

    e_final_np = e_final.cpu().numpy()
    real, imag = e_final_np[:, :J], e_final_np[:, J:]
    e_hat = real + 1j * imag
    e_hat = np.transpose(e_hat, (0, 2, 1))

    s_hat = pe_ifft(e_hat)
    return s_hat


def evaluate_dataset(model: ODLSv2, volumes: List[np.ndarray], mask_type: str,
                      af: float, device: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    mask_factory = MASK_FACTORIES[mask_type]

    total_slices = sum(vol.shape[0] for vol in volumes)
    progress = tqdm(total=total_slices, desc=f"testing ({len(volumes)} volumes)")

    rlnes, psnrs, ssims = [], [], []
    for vol in volumes:
        n_slices, M, N, J = vol.shape
        try:
            mask = mask_factory(N, af, rng=rng)
        except TypeError:
            mask = mask_factory(N, af)

        for s_idx in range(n_slices):
            k_space_slice = vol[s_idx]
            s_hat = reconstruct_slice(model, k_space_slice, mask, device)
            s_ref = fe_ifft(k_space_slice)
            s_ref = pe_ifft(s_ref)

            def to_channels(img):
                img_t = torch.from_numpy(np.transpose(img, (2, 0, 1)))
                ch = torch.cat([img_t.real, img_t.imag], dim=0).unsqueeze(0)
                return ch

            ref_ch = to_channels(s_ref)
            hat_ch = to_channels(s_hat)
            ref_mag = coil_combine_sos(ref_ch.flatten(2), J).view(1, M, N)
            hat_mag = coil_combine_sos(hat_ch.flatten(2), J).view(1, M, N)

            slice_rlne = rlne(ref_mag, hat_mag).item()
            rlnes.append(slice_rlne)
            psnrs.append(psnr(ref_mag, hat_mag).item())
            ssims.append(ssim(ref_mag, hat_mag).item())

            progress.set_postfix(rlne=f"{slice_rlne:.4f}")
            progress.update(1)

    progress.close()

    return {
        "RLNE_mean": float(np.mean(rlnes)), "RLNE_std": float(np.std(rlnes)),
        "PSNR_mean": float(np.mean(psnrs)), "PSNR_std": float(np.std(psnrs)),
        "SSIM_mean": float(np.mean(ssims)), "SSIM_std": float(np.std(ssims)),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained ODLS-v2 checkpoint")

    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--test-volumes", type=str, nargs="+")
    data_group.add_argument("--fastmri-test-root", type=str)

    p.add_argument("--fastmri-fat-suppressed", action="store_true")
    p.add_argument("--n-virtual-coils", type=int, default=8)
    p.add_argument("--crop-fe-to", type=int, default=224)
    p.add_argument("--crop-pe-to", type=int, default=224)

    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n-coils", type=int, required=True)
    p.add_argument("--n-phases", type=int, default=10)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--embed-dim", type=int, default=16,
                    help="Must match whatever --embed-dim the checkpoint was "
                         "trained with -- a construction-time hyperparameter, "
                         "not part of the saved state_dict.")
    p.add_argument("--max-threshold", type=float, default=0.05)
    p.add_argument("--mask-type", type=str, default="cartesian",
                    choices=["cartesian", "uniform", "partial_fourier"])
    p.add_argument("--af", type=float, default=4.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def load_test_volumes(args) -> List[np.ndarray]:
    if args.fastmri_test_root:
        if args.n_virtual_coils != args.n_coils:
            raise ValueError(
                f"--n-virtual-coils ({args.n_virtual_coils}) must equal --n-coils "
                f"({args.n_coils})."
            )
        paths = find_corpd_files(args.fastmri_test_root, fat_suppressed=args.fastmri_fat_suppressed)
        if not paths:
            raise ValueError(f"No matching CORPD{'FS' if args.fastmri_fat_suppressed else ''}_FBK "
                              f"files found under {args.fastmri_test_root}")
        return [load_fastmri_volume(p, n_virtual_coils=args.n_virtual_coils,
                                     crop_fe_to=args.crop_fe_to,
                                     crop_pe_to=args.crop_pe_to) for p in paths]
    return [np.load(p) for p in args.test_volumes]


def main():
    args = build_argparser().parse_args()

    model = ODLSv2(n_coils=args.n_coils, n_phases=args.n_phases, width=args.width,
                    max_threshold=args.max_threshold, embed_dim=args.embed_dim).to(args.device)
    load_model_weights(model, args.checkpoint, args.device)

    volumes = load_test_volumes(args)
    results = evaluate_dataset(model, volumes, args.mask_type, args.af, args.device)

    print(f"RLNE: {results['RLNE_mean']*100:.2f} +/- {results['RLNE_std']*100:.2f} (x1e-2)")
    print(f"PSNR: {results['PSNR_mean']:.2f} +/- {results['PSNR_std']:.2f} dB")
    print(f"SSIM: {results['SSIM_mean']*100:.2f} +/- {results['SSIM_std']*100:.2f} (x1e-2)")


if __name__ == "__main__":
    main()
