# Ablation: weight sharing only

Isolates **direction 1** from `odls_v2` (weight sharing across phases,
via a learnable phase embedding + FiLM conditioning) from the other two
changes (the denoiser reframing and cross-phase attention), so it can be
tested against the `../odls` baseline on its own.

## Why this exists

`odls_v2`'s combined model (sharing + denoiser + attention) underperformed
the baseline in testing, across two different width settings (48 and 64
-- see `../odls_v2/README.md` and the project history for details).
Diagnostics showed the phase embeddings and cross-phase attention were
both genuinely active, not inert, which weakened the "not enough
capacity to differentiate phases" theory. With three entangled changes
and an unexplained shortfall, the only way to know *which* change is
responsible is to test each one in isolation. This is the first of three
such isolated variants (weight sharing, denoiser, attention).

## What's different from `../odls_v2/model.py`

- `SharedLowRankModule` keeps `../odls/model.py`'s original low-rank
  framing (`r = eta1 * D1(e_prev)`), NOT `odls_v2`'s residual-subtraction
  denoiser reframing (`output = input - predicted_residual`).
- No `CrossPhaseAttention` anywhere -- removed entirely, not just
  disabled, so there's no ambiguity about whether it's contributing.
- Everything else -- the phase-embedding/FiLM sharing mechanism, the
  data-consistency module's physics, the FFT-centering fix, k-space
  normalization, the runaway-threshold cap -- is identical to `odls_v2`
  and `../odls`, so any difference in results traces back to weight
  sharing alone, not an unrelated infrastructure difference.

## What's unchanged from `../odls_v2/`

`masks.py`, `metrics.py`, `data.py`, `fastmri_data.py`, `losses.py` are
copied verbatim (already carrying the normalization and FFT-centering
fixes). `checkpoint_utils.py`, `train.py`, `evaluate.py` are adapted only
to construct `ODLSAblationSharing` instead of `ODLSv2`.

## Status: implemented and integration-tested, not yet trained

Verified before writing this README, not just compiled: a forward pass
produces correct shapes, a backward pass confirms zero parameters
without a gradient (sharing/FiLM genuinely wired into the graph), a real
`ODLSLoss` + `optimizer.step()` runs cleanly, `clamp_thresholds_()` works,
and a point-source round-trip test confirms the FFT centering fix
carried over correctly.

## Usage

```
python train.py --fastmri-train-root <dir> --fastmri-val-root <dir> \
    --n-coils 8 --n-virtual-coils 8 --embed-dim 16 \
    --mask-type cartesian --af 4 --checkpoint-dir <dir>

python evaluate.py --fastmri-test-root <dir> --checkpoint <dir>/odls_best.pt \
    --n-coils 8 --n-virtual-coils 8 --embed-dim 16 --mask-type cartesian --af 4
```
