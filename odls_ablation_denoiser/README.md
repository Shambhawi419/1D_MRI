# Ablation: denoiser prior only

Isolates **direction 2** from `odls_v2` (replacing the low-rank-motivated
N1 module with a general-purpose residual denoiser) from the other two
changes (weight sharing, cross-phase attention), so it can be tested
against the `../odls` baseline on its own.

## Why this exists

See `../odls_ablation_sharing/README.md` for the full context: `odls_v2`'s
combined model underperformed the baseline across two width settings,
with three entangled changes making it impossible to tell which one is
responsible. This is the second of three isolated variants.

## What's different from `../odls/model.py`

- `DeepDenoiserModule` replaces `DeepLowRankModule`. Same architecture
  (6 conv layers, 48 filters, same `eta1` init/behavior) -- the only
  change is the forward pass's framing: `r = eta1 * (e_prev - D1(e_prev))`
  (residual/DnCNN-style learning) instead of `r = eta1 * D1(e_prev)`
  (the paper's original framing).
- Everything else is **architecturally identical to the baseline**: 10
  independently-trained phases (`nn.ModuleList`, no weight sharing), no
  phase embeddings, no FiLM conditioning, no cross-phase attention. This
  is deliberate -- isolating the denoiser reframing means changing
  nothing else, so this variant's CLI doesn't even have an `--embed-dim`
  flag (there's nothing to condition).

## What's unchanged from `../odls/`

`masks.py`, `metrics.py`, `data.py`, `fastmri_data.py`, `losses.py` are
copied verbatim (carrying the normalization and FFT-centering fixes).
`checkpoint_utils.py`, `train.py`, `evaluate.py` are adapted only to
construct `ODLSAblationDenoiser` instead of `ODLS`.

## Status: implemented and integration-tested, not yet trained

Verified before writing this README: forward pass produces correct
shapes, backward pass confirms zero parameters without a gradient, a
real `ODLSLoss` + `optimizer.step()` runs cleanly, `clamp_thresholds_()`
works, and a point-source round-trip test confirms FFT centering.

## Usage

```
python train.py --fastmri-train-root <dir> --fastmri-val-root <dir> \
    --n-coils 8 --n-virtual-coils 8 \
    --mask-type cartesian --af 4 --checkpoint-dir <dir>

python evaluate.py --fastmri-test-root <dir> --checkpoint <dir>/odls_best.pt \
    --n-coils 8 --n-virtual-coils 8 --mask-type cartesian --af 4
```
