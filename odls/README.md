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
| `train.py`    | Sec. III-D training configuration (Adam, lr schedule, batch size, epochs) |
| `evaluate.py` | Sec. III-D inference flow (Fig. 6): reconstruct rows in parallel, stitch, final PE IFT |
| `fastmri_data.py` | Adapter for training/evaluating on the fastMRI knee multicoil dataset (see below) |

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
- **Readout oversampling**: fastMRI's raw readout (frequency-encoding)
  axis is acquired at 2x oversampling. `remove_readout_oversampling()`
  inverse-FFTs to image space, center-crops to half the readout length
  (or a custom `crop_fe_to`), and FFTs back to a properly-sized k-space
  before it ever reaches the row-wise 1D sample construction in
  `data.py`.
- **Memory**: fastMRI knee is far larger than the paper's original
  datasets, so `FastMRICorpdDataset` loads and processes slices lazily
  from disk (a small bounded LRU cache of recent slices) instead of
  materializing whole volumes up front, unlike `ODLSHybridDataset`.

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
