"""
Self-test for odls_eval_plus on synthetic data (CPU, about a minute; no OCMR,
no GPU). Run it after any change:   python selftest_plus.py

Checks:
  1. windowed SSIM: 1 for identical images, drops with noise, equals
     scikit-image's implementation (if scikit-image is installed)
  2. L+S: E^H is the exact adjoint of E; L+S beats zero-filled
  3. evaluate_plus.py zero-filled numbers == odls_dynamic/evaluate.py's
     (proves both see the same slices and the same masks)
  4. tune_lps.py -> evaluate_plus.py --method lps / --method model ->
     compare_plus.py runs end to end and writes the table + Wilcoxon tests
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

import paths  # noqa: F401
from checkpoint_utils import weights_checkpoint
from data import load_slice, read_manifest
from lps import Acquisition, estimate_coil_maps, lps_magnitude, lps_reconstruct
from masks import kt_random_mask
from metrics import rlne
from model import ODLSDynamic
from ssim_windowed import ssim_windowed
from transforms import np_ifftc

HERE = os.path.dirname(os.path.abspath(__file__))


def ok(msg):
    print(f"  ok  {msg}")


def load_dynamic_selftest():
    """odls_dynamic's selftest module, for its synthetic-data generator (not run, just imported)."""
    spec = importlib.util.spec_from_file_location("dyn_selftest", os.path.join(paths.DYN_DIR, "selftest.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(script_dir, script, args):
    r = subprocess.run([sys.executable, os.path.join(script_dir, script)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{r.stdout[-2500:]}\n{r.stderr[-2500:]}")
    return r.stdout


def test_ssim():
    rng = np.random.default_rng(0)
    ref = rng.random((2, 40, 36)).astype(np.float32)
    assert np.allclose(ssim_windowed(ref, ref), 1.0)
    noisy = [ssim_windowed(ref, ref + s * rng.standard_normal(ref.shape).astype(np.float32)).mean()
             for s in (0.05, 0.2)]
    assert noisy[0] > noisy[1]
    msg = "windowed SSIM: 1 for identical frames, drops with noise"
    if importlib.util.find_spec("skimage"):
        from skimage.metrics import structural_similarity
        est = ref + 0.1 * rng.standard_normal(ref.shape).astype(np.float32)
        sk = [structural_similarity(ref[t], est[t], data_range=ref[t].max(), gaussian_weights=True,
                                    sigma=1.5, use_sample_covariance=False) for t in range(2)]
        assert np.allclose(ssim_windowed(ref, est), sk, atol=1e-6)
        msg += ", matches scikit-image"
    ok(msg)


def test_lps(data_dir):
    row = [r for r in read_manifest(data_dir) if r["N"] == 30][0]
    h = load_slice(data_dir, row)
    mask = kt_random_mask(row["N"], row["T"], 4, np.random.default_rng(1))
    z, mk = torch.from_numpy((h * mask[None, :, None, :]).astype(np.complex64)), torch.from_numpy(mask)
    E = Acquisition(estimate_coil_maps(z, mk), mk)
    x = torch.randn(z.shape[0], z.shape[1], z.shape[3], dtype=torch.complex64)
    y = torch.randn(z.shape, dtype=torch.complex64)
    lhs, rhs = torch.vdot(E.forward(x).flatten(), y.flatten()), torch.vdot(x.flatten(), E.adjoint(y).flatten())
    assert abs(lhs - rhs) / abs(lhs) < 1e-4, "E^H is not the adjoint of E"
    ref_mag = torch.from_numpy(np.sqrt((np.abs(np_ifftc(h, 1)) ** 2).sum(2)).transpose(2, 0, 1).copy())
    zf_mag = torch.from_numpy(np.sqrt((np.abs(np_ifftc(z.numpy(), 1)) ** 2).sum(2)).transpose(2, 0, 1).copy())
    m, maps, _ = lps_reconstruct(z, mk, 0.01, 0.01)
    e_lps = rlne(ref_mag, torch.from_numpy(lps_magnitude(m, maps))).mean().item()
    e_zf = rlne(ref_mag, zf_mag).mean().item()
    assert e_lps < 0.7 * e_zf, (e_lps, e_zf)
    ok(f"L+S: E^H is the adjoint of E; RLNE {e_zf:.3f} (zero-filled) -> {e_lps:.3f} (L+S)")


def test_pipeline(data_dir, tmp):
    out = os.path.join(tmp, "eval_plus")
    common = ["--data-dir", data_dir, "--out-root", out, "--device", "cpu"]

    # 3. same masks / slices as odls_dynamic's own evaluation
    run(HERE, "evaluate_plus.py", common + ["--method", "zero_filled", "--af", "4", "--split", "train"])
    run(paths.DYN_DIR, "evaluate.py", ["--data-dir", data_dir, "--af", "4", "--split", "train", "--device", "cpu",
                                       "--out-dir", os.path.join(tmp, "dyn_eval")])
    mine = json.load(open(os.path.join(out, "train_af4", "zero_filled", "results_plus.json")))
    theirs = json.load(open(os.path.join(tmp, "dyn_eval", "results.json")))
    for a, b in zip(mine["per_slice"], theirs["per_slice"]):
        assert a["name"] == b["name"] and abs(a["RLNE"] - b["zero_filled_RLNE"]) < 1e-6
    ok("evaluate_plus zero-filled == odls_dynamic/evaluate.py zero-filled (same slices, same masks)")

    # 4. L+S tuning + evaluation, two (untrained) networks, then the comparison
    run(HERE, "tune_lps.py", ["--data-dir", data_dir, "--out-root", out, "--af", "4", "--device", "cpu",
                              "--lam-l", "0.01", "0.02", "--lam-s", "0.01", "0.02", "--n-iter", "20", "--max-widen", "1"])
    run(HERE, "evaluate_plus.py", common + ["--method", "lps", "--af", "4", "--split", "train",
                                            "--lps-params", os.path.join(out, "lps_params_af4.json")])
    for kt in (3, 1):
        model = ODLSDynamic(n_coils=8, n_phases=2, width=8, kernel_t=kt)
        ck = os.path.join(tmp, f"kt{kt}", "best.pt")
        os.makedirs(os.path.dirname(ck))
        torch.save(weights_checkpoint(model, epoch=1, args={"af": 4.0}), ck)
        run(HERE, "evaluate_plus.py", common + ["--method", "model", "--checkpoint", ck, "--split", "train"])
    txt = run(HERE, "compare_plus.py", ["--out-root", out])
    for needle in ("zero-filled", "L+S", "frame-by-frame ODLS", "dynamic ODLS", "Wilcoxon", "p="):
        assert needle in txt, needle
    assert os.path.exists(os.path.join(out, "summary_train_af4.md"))
    ok("tune_lps -> evaluate_plus (zero-filled, L+S, both networks) -> compare_plus table + Wilcoxon")


def main():
    tmp = tempfile.mkdtemp(prefix="odls_eval_plus_selftest_")
    try:
        data_dir = os.path.join(tmp, "data")
        load_dynamic_selftest().make_synthetic(data_dir)
        test_ssim()
        test_lps(data_dir)
        test_pipeline(data_dir, tmp)
        print("ALL EVAL-PLUS SELF-TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
