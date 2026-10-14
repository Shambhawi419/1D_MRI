"""
Self-test on synthetic cine data -- no OCMR download, runs on a CPU laptop
in a minute or two. Run before spending Colab time after any code change:

    python selftest.py

Checks: exact-AF k-t masks, FFT centring, dataset/sampler contracts, every
parameter receives a gradient, bit-exact resume after a simulated crash,
corrupted-checkpoint fallback, refusal to resume with a different
architecture, and evaluate.py end to end.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from checkpoint_utils import LATEST, LATEST_PREV, build_model_from_checkpoint, load_latest  # noqa: E402
from data import OCMRRowDataset, SliceBatchSampler  # noqa: E402
from losses import ODLSLoss  # noqa: E402
from masks import center_lines, effective_af, kt_random_mask  # noqa: E402
from model import ODLSDynamic  # noqa: E402
from preprocess_ocmr import MANIFEST_FIELDS  # noqa: E402
from transforms import fftc, ifftc, np_fftc, np_ifftc  # noqa: E402


def ok(msg):
    print(f"  ok  {msg}")


def make_synthetic(data_dir: str, n_coils: int = 8) -> None:
    """A beating-ellipse 'heart' with smooth coil maps; hybrid = FFT along ky."""
    os.makedirs(os.path.join(data_dir, "slices"), exist_ok=True)
    rng = np.random.default_rng(0)
    specs = [("train", 16, 24, 8), ("train", 16, 30, 10), ("train", 12, 24, 8),
             ("val", 16, 24, 8), ("test", 16, 28, 9)]
    rows = []
    for i, (split, M, N, T) in enumerate(specs):
        y, x = np.mgrid[-1:1:M * 1j, -1:1:N * 1j]
        frames = []
        for t in range(T):
            r = 0.35 + 0.1 * np.sin(2 * np.pi * t / T)
            frames.append((x ** 2 + y ** 2 < 0.7).astype(float) * 0.5 + (x ** 2 + y ** 2 < r ** 2) * 1.0)
        img = np.stack(frames, -1)                                            # (M, N, T)
        maps = np.stack([np.exp(-((x - np.cos(c)) ** 2 + (y - np.sin(c)) ** 2)) * np.exp(1j * c)
                         for c in np.linspace(0, 2 * np.pi, n_coils, endpoint=False)], -1)  # (M, N, J)
        coil_img = img[:, :, None, :] * maps[..., None] + 0.01 * rng.standard_normal((M, N, n_coils, T))
        hybrid = np_fftc(coil_img, axis=1).astype(np.complex64)              # (M, N, J, T)
        name = f"synth_{i:02d}_s00"
        np.save(os.path.join(data_dir, "slices", name + ".npy"), hybrid)
        rows.append({"name": name, "file": f"synth_{i:02d}.h5", "slice": 0, "split": split, "M": M, "N": N,
                     "J": n_coils, "T": T, "ky_valid_start": 0, "ky_valid_stop": N, "scale": 1.0,
                     "scanner": "synthA" if i % 2 else "synthB", "view": "sax", "field_T": 1.5,
                     "coils_raw": n_coils, "bytes": hybrid.nbytes})
    with open(os.path.join(data_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_masks():
    rng = np.random.default_rng(1)
    for n in (104, 144, 180):
        for af in (4, 8):
            m = kt_random_mask(n, 20, af, rng)
            per_frame = m.sum(0)
            assert np.all(per_frame == round(n / af)), per_frame
            assert np.all(m[center_lines(n)] == 1)
            assert not np.all(m[:, 0] == m[:, 1]), "frames should differ"
            assert abs(effective_af(m) - n / round(n / af)) < 1e-9
    ok("k-t masks: exact AF per frame, centre always sampled, frames differ")


def test_fft_centering():
    x = np.zeros((1, 32, 2, 4), np.complex64)
    x[0, 16, :, :] = 1  # point at the true centre
    assert np.allclose(np_ifftc(np_fftc(x, 1), 1), x, atol=1e-6)
    t = torch.from_numpy(np.concatenate([x.real, x.imag], 2).transpose(0, 2, 1, 3).copy())  # (1, 2J, N, T)
    back = ifftc(fftc(t, -2), -2)
    assert torch.allclose(back, t, atol=1e-6) and back[0, 0, 16, 0] > 0.99
    ok("centered FFT round trips (NumPy + torch), centre stays at the centre")


def test_dataset_and_sampler(data_dir):
    ds = OCMRRowDataset(data_dir, "train", af=4, fixed_mask=False)
    a, b = ds[(0, 5)], ds[(0, 5)]
    assert torch.equal(a["mask"], b["mask"]) and not torch.equal(a["mask"], ds[(1, 5)]["mask"])
    assert a["z"].shape == a["e_ref"].shape == (16, 24, 8) and a["mask"].shape == (1, 24, 8)
    sampler = SliceBatchSampler(ds.rows_of_slice, [6] * len(ds.rows), shuffle=True, seed=0)
    sampler.set_epoch(3)
    full = list(sampler)
    sampler.set_epoch(3, start_batch=2)
    assert list(sampler) == full[2:]
    for batch in full:
        slices = {ds.index[i][0] for _, i in batch}
        assert len(slices) == 1, "a batch must come from one slice"
    ok("dataset: masks keyed on (epoch, index); sampler: single-slice batches, exact skip-ahead")


def test_model_gradients():
    torch.manual_seed(0)
    for kt, dom, res in ((3, "yt", True), (3, "yf", True), (1, "yt", True), (3, "yt", False)):
        model = ODLSDynamic(n_coils=2, n_phases=2, width=8, kernel_t=kt, sparse_domain=dom, ds_residual=res)
        z, ref = torch.randn(3, 4, 12, 6), torch.randn(3, 4, 12, 6)
        mask = (torch.rand(3, 1, 12, 6) > 0.5).float()
        e_preds, sym = model(z, mask)
        loss, _ = ODLSLoss()(ref, e_preds, sym)
        loss.backward()
        dead = [n for n, p in model.named_parameters() if p.grad is None or not torch.any(p.grad != 0)]
        assert not dead, f"no gradient for {dead}"
        model.eval()
        e_preds, sym = model(z, mask)
        assert all(s is None for s in sym) and e_preds[-1].shape == z.shape
    ds = ODLSDynamic(n_coils=2, n_phases=1, width=8).phases[0].ds.eval()
    d = torch.randn(2, 4, 12, 6)
    with torch.no_grad():
        rel = (ds(d, need_sym=False)[0] - d).norm() / d.norm()
    assert rel < 0.05, f"residual DS should start near identity, got rel. change {rel:.3f}"
    ok("model: every parameter gets a gradient (yt, yf, kernel_t=1, literal Eq. 14); "
       "eval skips the symmetry pass; residual DS starts as identity")


def run(args, check=True):
    r = subprocess.run([sys.executable, os.path.join(HERE, "train.py")] + args, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-3000:])
    return r


def test_resume_exact(data_dir, tmp):
    common = ["--data-dir", data_dir, "--epochs", "2", "--batch-size", "6", "--n-phases", "2", "--width", "8",
              "--num-workers", "0", "--device", "cpu", "--save-every-steps", "2"]
    a, b = os.path.join(tmp, "runA"), os.path.join(tmp, "runB")
    run(common + ["--checkpoint-dir", a])
    run(common + ["--checkpoint-dir", b, "--stop-after-steps", "5"])     # "disconnect" after step 5
    state = load_latest(b, "cpu")
    assert state["global_step"] == 4, state["global_step"]               # last periodic save
    run(common + ["--checkpoint-dir", b])                                 # re-run the same command
    wa = torch.load(os.path.join(a, "final.pt"), weights_only=False)["model_state_dict"]
    wb = torch.load(os.path.join(b, "final.pt"), weights_only=False)["model_state_dict"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa), "resumed run diverged"
    hist = list(csv.DictReader(open(os.path.join(b, "history.csv"))))
    assert [h["epoch"] for h in hist] == ["1", "2"]
    lams = [v for h in hist for v in json.loads(h["lambda"])]
    assert all(v > 0 for v in lams), f"a soft-threshold died at 0: {lams}"
    ok("crash after step 5 + re-run == uninterrupted run, bit for bit; history has no duplicates; "
       "every soft-threshold stays > 0")
    return a, b


def test_fallback_and_guard(data_dir, run_dir):
    with open(os.path.join(run_dir, LATEST), "wb") as f:
        f.write(b"truncated garbage")
    state = load_latest(run_dir, "cpu")
    assert state is not None and os.path.exists(os.path.join(run_dir, LATEST_PREV))
    ok("corrupted latest.pt -> falls back to latest_prev.pt")
    r = run(["--data-dir", data_dir, "--checkpoint-dir", run_dir, "--epochs", "3", "--width", "16",
             "--n-phases", "2", "--num-workers", "0", "--device", "cpu"], check=False)
    assert r.returncode != 0 and "model config" in (r.stdout + r.stderr)
    ok("resume with a different architecture is refused with a clear message")


def test_evaluate(data_dir, run_dir):
    model, ckpt = build_model_from_checkpoint(os.path.join(run_dir, "best.pt"))
    assert model.config["width"] == 8 and ckpt["args"]["af"] == 4.0
    r = subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), "--data-dir", data_dir,
                        "--checkpoint", os.path.join(run_dir, "best.pt"), "--device", "cpu"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert os.path.exists(os.path.join(run_dir, "eval_test_af4", "results.json"))
    ok("model rebuilt from best.pt alone; evaluate.py writes results.json + examples")


def main():
    tmp = tempfile.mkdtemp(prefix="odls_dyn_selftest_")
    try:
        data_dir = os.path.join(tmp, "data")
        make_synthetic(data_dir)
        test_masks()
        test_fft_centering()
        test_dataset_and_sampler(data_dir)
        test_model_gradients()
        run_a, run_b = test_resume_exact(data_dir, tmp)
        test_evaluate(data_dir, run_a)
        test_fallback_and_guard(data_dir, run_b)
        print("ALL SELF-TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
