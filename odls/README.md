# ODLS -- One-dimensional Deep Low-rank and Sparse Network

PyTorch implementation of the architecture described in:

> Z. Wang, C. Qian, D. Guo, H. Sun, R. Li, B. Zhao, X. Qu,
> "One-dimensional Deep Low-rank and Sparse Network for Accelerated MRI,"
> arXiv:2112.04721.

## Files

| File          | Paper section(s) it implements                                   |
|---------------|--------------------------------------------------------------------|
| `masks.py`    | Sec. II (AF definition) + Fig. 11 undersampling scenarios (Cartesian, uniform, partial Fourier) |
| `data.py`     | Sec. II (1D FE/PE transforms, Eq. 1) + Sec. III-A (1D training-sample construction, Fig. 2(d) / Table I) |
| `model.py`    | Sec. III-C (DLR/DC/DS modules, Eqs. 12-14) + Sec. III-D (network architecture, Fig. 4) |
| `losses.py`   | Eq. 15 (reconstruction + symmetry loss) |
| `metrics.py`  | Supplement S1 (RLNE, PSNR, SSIM) |
| `train.py`    | Sec. III-D training configuration (Adam, lr schedule, batch size, epochs); tqdm progress bars; per-epoch Drive-safe checkpointing with auto-resume and `--init-checkpoint` pretraining |
| `evaluate.py` | Sec. III-D inference flow (Fig. 6): reconstruct rows in parallel, stitch, final PE IFT; tqdm progress bar over test slices |
| `fastmri_data.py` | Adapter for training/evaluating on the fastMRI knee multicoil dataset (see below) |
| `checkpoint_utils.py` | Shared checkpoint loader used by both `train.py` and `evaluate.py` |

## Checkpointing, resuming, and pretraining

`train.py` writes to `--checkpoint-dir` every single epoch:

- **`latest.pt`** -- the full training state (model + optimizer + scheduler
  + epoch + best_val_loss). If training is interrupted (a Colab
  disconnect, a crash, anything), re-running the exact same command with
  the same `--checkpoint-dir` picks up automatically from the next epoch
  instead of restarting at epoch 1. Point `--checkpoint-dir` at a
  persistent location (e.g. Google Drive in Colab) -- it's local disk
  otherwise, which is wiped on a Colab runtime reset.
- **`odls_best.pt`** -- plain model weights, written whenever validation
  loss improves.
- **`odls_final.pt`** -- plain model weights, written once training
  finishes.

Use `--init-checkpoint <path>` to load weights from any of the three as a
*pretrained* starting point for a new run (new data, new mask/AF,
fine-tuning) with a fresh optimizer/scheduler and epoch counter reset to
1 -- this is different from the automatic resume above, which continues
the literal same run. `evaluate.py --checkpoint <path>` accepts any of
the three formats too (`checkpoint_utils.load_model_weights` detects
which one it's looking at).

`odls_best.pt` is chosen by a trailing average of val_loss over the last
`--best-checkpoint-window` epochs (default 5), not a single epoch's raw
value -- a small validation set combined with BatchNorm's running
statistics (computed only from randomly-masked training batches, not
matching a fixed validation mask as well in every epoch) can make raw
per-epoch val_loss noisy enough that a lucky low outlier would otherwise
get locked in as "best" while genuinely better later epochs never
overwrite it for not beating that fluke.

## A real failure mode found during training: runaway soft-threshold

Training on fastMRI CORPD_FBK surfaced a genuine architectural gap, not
just an engineering bug: each phase's learnable soft-threshold (Sec.
III-C.3, init 0.001) has no upper bound in the architecture as described,
and nothing else in the model stops gradient descent from inflating it
without limit. In practice it grew 250-800x within ~17 epochs (measured:
0.24-0.87 across phases, from a 0.001 start), at which point it was
suppressing nearly all signal passing through the sparsifying step --
producing a near-all-zero reconstructed output (RLNE ~1.0, PSNR ~10dB,
SSIM ~0 on the test set) despite training loss looking fine throughout
(raw MSE on small-magnitude values doesn't punish "shrink everything
toward zero" nearly as harshly as a scale-normalized metric like RLNE
does -- which is exactly why this went unnoticed until a real test-set
evaluation was run).

The fix: `DeepSparseModule` (and `ODLSPhase` / `ODLS` above it) now take a
`max_threshold` (default 0.05, exposed as `--max-threshold` on both
`train.py` and `evaluate.py` -- keep it identical between training and
evaluating the same checkpoint), and `_soft_threshold` clamps to
`[0, max_threshold]` before applying the shrinkage. `ODLS.clamp_thresholds_()`
additionally resets any already-out-of-range *stored* threshold value the
moment a checkpoint is loaded (called from both train.py's resume path
and `checkpoint_utils.load_model_weights`) -- the forward-pass clamp
alone already makes model *behavior* correct regardless of the stored
value, but leaving a stale out-of-range number sitting in the parameter
is still worth cleaning up: Adam's leftover per-parameter momentum could
otherwise keep nudging a value that can no longer affect output, and a
stale large value would spring back to life with no real meaning if
`max_threshold` is ever raised again in a later run.

## Design notes / where the source PDF was ambiguous

The paper PDF's math symbols (Ψ, Φ, Γ, Ω, subscripts) were badly mangled by
text extraction, so a few operators had to be reconstructed from the
surrounding prose rather than copied character-for-character:

- **Undersampling operator Λ** is implemented as a plain elementwise 0/1
  mask applied directly in the hybrid k-space domain, since PE
  undersampling subsamples exactly that axis (Sec. II) and a diagonal
  0/1 mask is self-adjoint (Λ* = Λ), which keeps the data-consistency
  module (Eq. 13) simple and correct.
- **Symmetry loss target**: the paper states the extra loss term is
  `||Ψ3(Ψ2(x)) - x||^2` where `x` is the input to the learned sparsifying
  transform (i.e. *before* soft-thresholding). `model.py`'s
  `DeepSparseModule.transform_and_reconstruct` returns both `x` (called
  `img`) and `Ψ3(Ψ2(x))` (called `transformed`) explicitly so `losses.py`
  compares the correct pair, rather than comparing against the final
  post-threshold reconstruction.
- **Undersampling mask shapes** (`masks.py`): the paper's Fig. 11 shows
  the three sampling patterns visually rather than giving closed-form
  definitions, so `cartesian_random_mask`, `uniform_mask`, and
  `partial_fourier_mask` are standard, reasonable implementations of
  "variable-density random Cartesian," "equispaced," and "one-sided
  low-frequency truncation" respectively, each parameterized so the
  retained fraction matches the paper's AF definition
  (AF = fully-sampled points / undersampled points, Sec. II).

Everything with a concrete, unambiguous number in the paper is matched
exactly: K=10 unrolled phases; 6/3/3 conv layers for D1/Ψ2/Ψ3; 48 filters
of kernel size 3 per layer with BatchNorm+ReLU; learnable η1 init 0.001,
η2 init 1, soft-threshold λ init 0.001; Xavier weight init; Adam with
lr=0.001 and exponential decay 0.99; batch size 128; 300 epochs; symmetry
loss weighted 0.01 relative to the reconstruction error term.

## Data you need to supply

The paper's original in-vivo knee/brain k-space datasets are not public.
Two ways to supply data:

1. **Plain `.npy` volumes** -- complex-valued, shape `(n_slices, M, N, J)`
   where `M` is frequency-encoding, `N` is phase-encoding, `J` is coils.
2. **fastMRI knee multicoil** (what we're actually using) -- see below.

### Training on fastMRI knee multicoil (CORPD)

The Kaggle mirror
(https://www.kaggle.com/datasets/arafatshovon/fastmri-knee-multicoil) ships
fastMRI's original per-volume `.h5` files. This required a dedicated
adapter (`fastmri_data.py`) because fastMRI's format differs from what the
rest of this codebase assumes:

- **Storage**: `.h5` files, `kspace` dataset shaped
  `(n_slices, n_coils, H, W)` -- coil axis *first*, not last.
  `fastmri_data.py` transposes this into the project's `(M, N, J)`
  convention (coils last) before anything else touches it.
- **Acquisition filtering**: fastMRI knee files are split between
  `CORPD_FBK` (proton density) and `CORPDFS_FBK` (fat-suppressed proton
  density). `find_corpd_files(root)` scans a directory and keeps only
  `CORPD_FBK` by default (pass `fastmri_fat_suppressed=True` / the
  `--fastmri-fat-suppressed` CLI flag for the FS variant instead).
- **Coil count**: fastMRI knee volumes commonly carry 15-16 physical
  coils, which varies file to file and is far more than the paper's
  post-compression J=8 -- and DataLoader batching needs a fixed channel
  count anyway. `coil_compress()` applies the SVD-based virtual-coil
  compression the paper itself cites (ref. [49], Zhang et al.) down to a
  fixed `n_virtual_coils` (default 8, matching the paper).
- **Readout oversampling, and non-uniform matrix sizes across files**:
  fastMRI's raw readout (frequency-encoding) axis is acquired at 2x
  oversampling, and both the readout size and the phase-encoding width
  vary from file to file. Left alone, this crashes DataLoader batching
  the instant two differently-shaped samples land in the same batch (this
  actually happened during Colab testing). `_resize_via_kspace()` forces
  both spatial axes to an exact fixed target size via an ifft /
  center-crop-or-zero-pad / fft round trip -- `remove_readout_oversampling()`
  applies it to the FE axis (default target: 224, matching the paper's
  Sec. IV-A "224x224" preprocessing) and `resize_pe()` applies it to the
  PE axis (same default). This is deliberately a *round trip through
  image space*, not a raw truncation of k-space samples: cropping k-space
  directly would reduce resolution (a low-pass effect) rather than
  reducing the reconstructed FOV, which is what "center-cropped to
  224x224" actually means. Zero-padding (when a file is smaller than the
  target) is the resolution-preserving mirror operation for enlarging the
  FOV. `crop_fe_to` / `crop_pe_to` are exposed as `--crop-fe-to` /
  `--crop-pe-to` on both `train.py` and `evaluate.py` -- keep them
  identical between training and evaluation of the same checkpoint.
- **Amplitude normalization**: raw MRI k-space -- fastMRI included -- is
  naturally tiny in magnitude (confirmed by directly inspecting real data
  from the NYU knee dataset ODLS's own knee experiments were sourced
  from: mean coil-combined image magnitude ~1e-5). Training on that scale
  unnormalized lets the network cheaply minimize raw MSE by collapsing
  toward a near-zero output rather than learning real reconstruction --
  which is exactly what happened here: RLNE ~1.0 (near-total
  reconstruction failure) on held-out test data, identically across every
  checkpoint tested from epoch 1 through epoch 35, despite training loss
  looking fine throughout. `compute_normalization_scale()` divides each
  slice's k-space by the max magnitude of its own coil-combined image
  reconstruction before anything else touches it, bringing every sample
  to a consistent ~O(1) scale -- the same technique (max of the
  zero-filled reconstruction) used by the published training code for
  that exact NYU dataset (VLOGroup/mri-variationalnetwork's `mridata.py`),
  adapted here to the fully-sampled reference so the scale doesn't depend
  on which random undersampling mask a given sample draws. This is a
  breaking change to the numeric scale of every value the network sees --
  a checkpoint trained before this fix cannot be resumed afterward (see
  `checkpoint_utils.py`'s docstring / the Colab notebook's `CHECKPOINT_DIR`
  comment); training needs to restart from epoch 1 into a fresh
  checkpoint directory.

## A second breaking fix: FFT centering (fe_ifft / pe_ifft / _ifft_pe / _fft_pe / reconstruct_image / _to_image)

A properly centered inverse FFT needs three steps: `ifftshift` the
input, transform, then `fftshift` the output back. `data.py`'s
`fe_ifft`/`pe_ifft`, `model.py`'s `DeepSparseModule._ifft_pe`/`_fft_pe`/
`ODLS.reconstruct_image`, and `losses.py`'s `_to_image` were all missing
the closing shift -- confirmed by direct visual inspection during Colab
testing (both the ground-truth AND the reconstructed image showed the
same "cut into quarters and rearranged" look, ruling out a reconstruction
-quality issue -- both go through this same shared, buggy centering).

This did **not** invalidate training: the same incomplete convention was
applied consistently to the reference and the prediction everywhere it
mattered (including inside the loss), so `L_err` still compared
corresponding pixels correctly, and the RLNE/PSNR/SSIM numbers obtained
before this fix are genuine, not fabricated by the bug. What it did cost:
every convolution in the network was processing anatomy with its true
center wrapped out to the array's corners instead of sitting centered
and spatially coherent -- a needlessly awkward input shape for a local
convolution kernel, likely making the learning task harder than
necessary. Verified fixed with actual point-source round-trip tests (not
just by inspection) in both NumPy (`fe_ifft`+`pe_ifft`) and PyTorch
(`DeepSparseModule._ifft_pe`+`_fft_pe`): a bright pixel placed at the
true center of a synthetic image now correctly returns to the center
after the round trip.

Like the normalization fix, this changes the actual spatial
representation the network is trained on -- a checkpoint trained before
this fix should not be resumed afterward; start fresh into a new
checkpoint directory. `../odls_v2/` received the identical fix (its
`SharedDeepSparseModule`/`ODLSv2` were built directly on top of this same
code before the bug was found), so a baseline-vs-v2 comparison still
isolates only the intended architectural differences, not this fix.

- **Memory, and why the GPU can otherwise sit idle**: fastMRI knee is far
  larger than the paper's original datasets, so `FastMRICorpdDataset`
  loads and processes slices lazily from disk rather than materializing
  whole volumes up front, unlike `ODLSHybridDataset`. Each slice's
  preprocessing (coil-compression SVD plus the FE/PE FFT round-trips
  above) is real CPU work, so it's cached per (file, slice) -- but with
  `shuffle=True` scrambling access order at the row level, a *small*
  cache almost never gets revisited before eviction, so this work would
  otherwise be redone on nearly every sample, starving the GPU while the
  CPU chases FFTs and SVDs (observed directly during Colab testing: GPU
  memory near-idle while the first epoch crawled). `slice_cache_size`
  therefore defaults to caching *every* slice in the dataset rather than
  a small fixed number -- at `crop_fe_to`/`crop_pe_to`=224 and
  `n_virtual_coils`=8 this is only a few MB per slice, so caching a
  whole multi-file dataset costs a few GB of RAM, which is affordable.
  This only pays off if the cache survives between epochs, though: with
  `DataLoader(num_workers>0)`, each worker holds its own independent copy
  of the dataset (and thus its own independent cache) since PyTorch
  respawns worker processes fresh every epoch by default -- so
  `train.py`'s DataLoaders are built with `persistent_workers=True`
  whenever `--num-workers > 0`, keeping each worker (and its cache) alive
  across all 300 epochs instead of rebuilding it from scratch every time.
  `--num-workers` also now defaults to 2 rather than 4, since Colab's
  typical 2-CPU allocation means more workers just causes contention
  rather than faster loading (visible as a PyTorch `UserWarning` about
  worker count if you push past your actual CPU count).

Everything downstream -- row-wise sample construction, undersampling
masks, the ODLS model, the loss -- is untouched; `FastMRICorpdDataset`
just produces the same `{"z", "e_ref", "mask"}` sample dict.

### Running on Google Colab

`../colab/ODLS_Colab.ipynb` is a ready-to-run Colab notebook for the case
where the 20 downloaded fastMRI files live in Google Drive: it mounts
Drive, stages/extracts the files onto local Colab disk, filters to
CORPD_FBK via `find_corpd_files`, splits into train/val folders, clones
this repo, and runs `train.py` / `evaluate.py`. Edit the **Config** cell
(Drive folder path, coil counts, mask/AF) before running top to bottom.

## Usage (reference only -- not run as part of this delivery)

```
# .npy volumes
python train.py --train-volumes train1.npy train2.npy \
                 --val-volumes val.npy \
                 --n-coils 8 --mask-type cartesian --af 4 \
                 --checkpoint-dir checkpoints

# fastMRI knee multicoil, CORPD_FBK only
python train.py --fastmri-train-root /path/to/fastmri_knee/train \
                 --fastmri-val-root /path/to/fastmri_knee/val \
                 --n-coils 8 --n-virtual-coils 8 \
                 --mask-type cartesian --af 4 \
                 --checkpoint-dir checkpoints

python evaluate.py --fastmri-test-root /path/to/fastmri_knee/test \
                    --checkpoint checkpoints/odls_best.pt \
                    --n-coils 8 --n-virtual-coils 8 \
                    --mask-type cartesian --af 4
```

Note: `--n-coils` and `--n-virtual-coils` must match when using the
fastMRI path -- the model's channel count is fixed at construction time
and has to agree with however many virtual coils the data was compressed
to; `build_train_and_val_sets` / `load_test_volumes` raise a clear error
if they don't.
