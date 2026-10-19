# odls_eval_plus: stronger evaluation and a classical baseline

This folder adds two things on top of `../odls_dynamic/`, **without changing
anything there**:

1. **A more complete evaluation** of the trained models:
   - standard windowed SSIM, next to the paper's global SSIM;
   - statistics over slices instead of frames;
   - Wilcoxon significance tests;
   - reconstruction time, parameter count and GPU memory.
2. **L+S (low-rank + sparse)**, a classical, non-deep-learning dynamic-MRI method
   (Otazo et al., MRM 2015). It's scored on exactly the same data and masks.

`odls_dynamic/` is only **imported** (data loading, masks, model loading, the
original metrics). Results go to a separate folder (`<Drive>/odls_dynamic/eval_plus/`),
so the training run folders and their `results.json` are never touched.

Run everything with `../colab/ODLS_EvalPlus_Colab.ipynb` (about 30–60 min on a T4).

## Files, in reading order

| File | What it does |
|---|---|
| `paths.py` | Makes `../odls_dynamic` importable. Every script starts with `import paths`. |
| `ssim_windowed.py` | Standard SSIM: an 11×11 Gaussian window, σ = 1.5. Verified equal to scikit-image's. |
| `lps.py` | The L+S algorithm, the acquisition model `E`, and coil-map estimation. |
| `evaluate_plus.py` | Scores **one** method (`model`, `zero_filled` or `lps`) on one split at one AF. |
| `tune_lps.py` | Picks L+S's two settings on the **validation** split. |
| `compare_plus.py` | Builds the final table and the Wilcoxon tests from all results. |
| `selftest_plus.py` | Synthetic end-to-end tests (CPU, under 1 min). Run it after any change. |

Run the scripts from inside this folder (`cd odls_eval_plus && python ...`). They
need `scipy` on top of `../odls_dynamic/requirements.txt`; Colab has it.

## 1. What the extra metrics mean

- **Global SSIM (the paper's)** computes one mean, one variance and one
  covariance for the whole image. A large dark background makes it very high
  (around 99%). It's kept for comparison with the ODLS paper.
- **Windowed SSIM (standard)** computes SSIM inside a small window around every
  pixel, then averages. It's more sensitive to local detail and blur, and it is
  what most MRI papers report. Expect it to be noticeably lower than the global
  value.
- **Per-slice vs per-frame.** The 22 or so frames of one cine are very similar,
  so treating 251 frames as 251 independent samples makes the spread look too
  small. Each slice's metrics are first averaged over its frames, and the table
  shows mean ± std over the **13 test slices**.
- **Wilcoxon signed-rank test.** For every test slice, compare dynamic ODLS
  with method X; the test asks whether dynamic is better consistently, or
  only by chance. It's paired (same slices, same masks) and two-sided.
  - With 13 slices, winning on all 13 gives p = 0.00024, the smallest
    possible value.
  - "12/13, p=0.0005" means it won on 12 of 13 slices, and a difference that
    consistent is very unlikely to be chance.
- **Time per slice** is the *median* over slices, because the first slice
  includes one-off GPU start-up. **Peak GPU memory** and **parameters** are
  reported for the networks.

## 2. How L+S works (`lps.py`)

Arrange the cine as a matrix with one row per pixel and one column per frame,
and split it into two parts, M = L + S:

- **L, low-rank:** what all frames share (static anatomy). A handful of
  singular vectors describe it.
- **S, sparse:** what moves (the heart). Periodic motion has few temporal
  frequencies, so S is sparse after an FFT along time.

The algorithm alternates three steps, about 50 iterations per slice:

1. **L-step:** take the SVD of the current image minus S, shrink every
   singular value by τ_L, and rebuild the matrix. This removes the weak
   components, the noise and the aliasing.
2. **S-step:** FFT the current image minus L along time, shrink every
   coefficient's magnitude by τ_S, and transform back.
3. **Data consistency:** add L + S, then correct towards the measured k-space
   lines with one gradient step on ‖E(L+S) − d‖².

`E` is the acquisition:
- multiply the image by each coil's sensitivity;
- FFT along ky (kx is already in image space in the hybrid data);
- keep only the measured lines of each frame.

`Eᴴ` reverses it. The self-test checks `Eᴴ` is the exact adjoint of `E`.

**Coil sensitivities** are estimated from the undersampled data itself, as in
Otazo 2015:
- average the measured lines over all frames (different frames measure
  different lines, so the average fills most of low-frequency k-space);
- keep the central lines with a Hann window (sensitivities are smooth);
- divide each coil image by the root-sum-of-squares.

Nothing from the fully-sampled reference is used.

**Settings.** τ_L = `lam_l` × (largest singular value), and τ_S = `lam_s` ×
(largest temporal-spectrum magnitude), so the same `lam_l` and `lam_s` work for
every slice. `tune_lps.py` tries a grid of values on the validation slices and
keeps the pair with the lowest RLNE. If that pair sits on the edge of the grid,
the optimum may lie outside it, so the grid is extended in that direction
(halving or doubling the extreme value) and the new pairs are tried, up to 3
times, until the best pair is inside. Tuning on validation, not on test, is the
same rule the networks follow when `best.pt` is chosen.

## 3. Outputs

```
<Drive>/odls_dynamic/eval_plus/
├── lps_params_af4.json, lps_params_af8.json      L+S tuning: best settings + the whole grid
├── test_af4/
│   ├── zero_filled/results_plus.json  (+ 3 example .npz)
│   ├── lps/results_plus.json
│   ├── dyn_af4_kt1_yt_v2/results_plus.json      frame-by-frame ODLS
│   └── dyn_af4_kt3_yt_v2/results_plus.json      dynamic ODLS
├── test_af8/   (same layout, for AF 8)
├── summary_test_af4.md                           final table + Wilcoxon tests
└── summary_test_af8.md
```

`results_plus.json` holds:
- per-slice and per-frame summaries of all four metrics;
- every slice's own values;
- timing;
- parameters and peak memory (networks);
- the settings used (L+S).

The example `.npz` files hold `ref`, `est` and `mask` for the first 3 test
slices. They're the same slices for every method, so the notebook can put
them side by side.

## 4. Consistency with `odls_dynamic`

- **Same slices and masks:** `fixed_slice_mask` is keyed on the slice name,
  AF and seed. The self-test checks that the zero-filled numbers here equal
  `odls_dynamic/evaluate.py`'s exactly.
- **RLNE, PSNR and global SSIM** come from `odls_dynamic/metrics.py`, so they
  match the numbers you already have. They are averaged per slice here rather
  than per frame, so means can differ slightly.
