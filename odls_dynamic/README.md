# Dynamic ODLS on OCMR cardiac cine

ODLS (Wang et al., "One-dimensional Deep Low-rank and Sparse Network for
Accelerated MRI", arXiv:2112.04721) extended to dynamic MRI, trained on the
OCMR dataset. `../odls/` (the static fastMRI replication) is untouched.

Run it with `../colab/ODLS_Dynamic_OCMR_Colab.ipynb` (free Colab T4 is enough).

## Design

- **Row splitting still works.** OCMR cine is undersampled along ky and time;
  kx (readout) is always fully sampled. An inverse FFT along kx decouples each
  slice into M independent rows (paper Eq. 1), exactly as in the static paper.
- **One sample = one row across all frames**: shape `(2J, N, T)` = coils
  (real, imag) x ky x frames. Convolutions are 2D over (ky, t) with circular
  padding in time (a cine loop is one periodic cardiac cycle).
- **Network**: the paper's unrolled DLR -> DC -> DS (Eqs. 12-14), K = 10
  independent phases, 6/3/3 conv layers, 48 filters, BN + ReLU, Xavier init,
  eta1 = 0.001, eta2 = 1, lambda = 0.001, loss = Eq. 15 with sym weight 0.01.
- **Sparse module form** (`--ds-residual`, default on): the sparse module
  computes `x + Ψ3(soft(Ψ2 x)) − Ψ3(Ψ2 x)`.
  - Once the symmetry loss holds (Ψ3∘Ψ2 = identity), this is exactly Eq. 14.
    The difference is that it starts as the identity map instead of noise,
    so training starts from the zero-filled input rather than from random
    output.
  - `--no-ds-residual` is the literal Eq. 14. See `model.py` for the
    measured difference.
- **k-t masks**: variable-density random Cartesian lines, different in every
  frame, centre lines always sampled, exact AF per frame.

## Problems from `../odls` and how they're handled here

| Problem in `../odls` | Here |
|---|---|
| Soft-threshold applied *after* Psi3 (on image values), not in the transform domain as Eq. 14 says; parameter count 15% below the paper's | Threshold between Psi2 and Psi3 in a 48-channel domain. With 1D kernels: 658,910 params vs the paper's 664,350 (0.8%). The threshold cap is now optional (off by default) and every phase's lambda is logged per epoch |
| `partial_fourier_mask` skipped the k-space centre; `uniform_mask` AF 4 was really 3.25 | New k-t mask with exact AF, centre at the true centre (verified for all 165 OCMR files) |
| DataLoader workers drew identical random masks | Masks are a pure function of (seed, epoch, sample), so workers never collide and resume is exact |
| Normalisation used the fully-sampled image (ground-truth leak) | Coil compression and normalisation use only the centre lines every mask samples |
| One test mask per volume | One fixed mask per slice, keyed on the slice name, shared by its rows (like a real scan) |
| `--crop-*-to None` advertised but impossible; FIFO cache called LRU | No resizing at all (batches are built per slice, so sizes may differ); memory-mapped data instead of a cache |

## Data: OCMR

Checked from the public metadata only (attributes CSV and HDF5 headers): 165
fully-sampled files (74.5 GB), 0.55 / 1.5 / 3 T. Default selection: 128
symmetric-echo, single-slice SAX + LAX series, giving 102/13/13
train/val/test files. Asymmetric-echo files are excluded (their readout is
partially sampled). Multi-slice stacks are excluded by default
(`--include-stacks` adds 11 files).

`preprocess_ocmr.py` streams one raw file at a time from S3 to Colab's
temporary disk, processes it, and deletes it. It writes **~4.8 GB** to Drive,
and a hard cap (`--max-gb`, default 8) stops it before exceeding that. Each
slice is processed as follows:

1. kx inverse FFT, then crop to remove the 2x readout oversampling
2. SVD coil compression to 8 virtual coils
3. Normalisation
4. Save as `slices/<name>.npy`, complex64 `(M, N, J, T)`

The run is resumable and the split is fixed in `split.csv`.
`--split-mode holdout-scanner --resplit-only` gives the paper-style
mismatch test (unseen 0.55 T scanner) without re-downloading.

## Checkpoints (everything on Drive)

A run folder holds the following files:

| File | Contents |
|---|---|
| `latest.pt` | Full state: model, optimizer, scheduler, AMP scaler, all RNGs, epoch and batch position, history |
| `latest_prev.pt` | The previous `latest.pt`, as a fallback |
| `best.pt` | Best 5-epoch-average validation loss |
| `final.pt` | Weights at the end of training |
| `history.csv` | Per-epoch log |
| `run_log.txt` | One line per start, resume, or stop |
| `run_config.json` | Training arguments and model config |
| `eval_*/results.json` | Evaluation results |

- **Saved every 200 steps and at the end of every epoch.** Writes are atomic
  (temp file, then rename), so a disconnect can't corrupt a checkpoint.
- **Resume**: re-run the same command. It continues from the exact batch,
  bit-for-bit identical to an uninterrupted run (tested in `selftest.py`).
  It refuses to resume with a different architecture or AF.
- **`--time-limit-min`**: stops cleanly before the Colab session limit.
- **Out of GPU memory**: saves a checkpoint and stops, telling you which
  smaller `--batch-size` to use.
- **Other workflows**: every checkpoint stores its model config.
  `build_model_from_checkpoint(path)` rebuilds the model with no flags.
  `--init-checkpoint best.pt` starts a new run from trained weights
  (fine-tune at another AF, more data, ...).

## Files

| File | What |
|---|---|
| `ocmr_io.py` | ISMRMRD `.h5` reader via h5py (identical output to OCMR's official reader on a real file, ~0.3 s) |
| `preprocess_ocmr.py` | Streaming download + preprocessing + split |
| `masks.py` | k-t masks |
| `transforms.py` | Centered FFTs, channel convention |
| `data.py` | Row dataset, per-slice memory-balanced batch sampler |
| `model.py` | `ODLSDynamic` |
| `losses.py` | Eq. 15 |
| `metrics.py` | RLNE / PSNR / SSIM (Supplement S1, same as `../odls`) |
| `checkpoint_utils.py` | Atomic, self-describing checkpoints |
| `train.py` | Training with exact resume |
| `evaluate.py` | Per-frame metrics vs the zero-filled baseline, per scanner and per view |
| `selftest.py` | Synthetic end-to-end tests (CPU, ~1 min) |

## Suggested experiments

1. `AF=4, KERNEL_T=3` (dynamic) vs `KERNEL_T=1` (frame-by-frame, same
   pipeline) vs zero-filled. The first gap shows what using time buys.
2. The same at AF 8, fine-tuned from the AF 4 model via `--init-checkpoint`.
3. `--sparse-domain yf` (temporal-Fourier sparsity) vs `yt` (the paper's).
4. The mismatch test: train on 1.5/3 T, test on 0.55 T.

## Notes

- The batch size is defined for a median 144 ky x 22 frame slice. It is
  scaled per slice (8-64 rows) so GPU memory stays at about 4-5 GB with mixed
  precision. FFTs always run in fp32, because cuFFT half precision needs
  power-of-two sizes.
- The exact AF per frame is limited by integer line counts. For example,
  N = 110 at AF 4 gives 28 lines, an effective AF of 3.93. `evaluate.py`
  reports the measured AF.
- Calibration (centre) lines are 8% of ky. They are used for normalisation
  and coil compression, which assumes AF <= about 12.
- The OCMR CSV doesn't identify subjects, so the split is by file. An LAX
  and an SAX file could come from the same volunteer.
