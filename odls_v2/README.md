# ODLS-v2 -- research extension (not the paper's method)

This folder is a deliberate departure from the faithful replication in
`../odls/`, exploring three architectural changes suggested as directions
for improving on arXiv:2112.04721, rather than reproducing it. `../odls/`
is left untouched as the working, validated baseline (RLNE ~14%, PSNR
~27dB, SSIM ~95% on fastMRI CORPD_FBK after the normalization fix) --
this folder is where new ideas get tried without risking that baseline.

## The three changes (see `model.py`'s module docstring for full detail)

1. **Weight sharing across phases.** The baseline trains 10 independent
   copies each of the low-rank module, and the sparse-transform pair (30
   sub-networks total). Here, each of those three roles has exactly ONE
   shared network, used at every phase. Phase-specific behavior is
   preserved via a learnable **phase embedding** (one per phase index)
   that FiLM-modulates the shared conv stacks (`FiLM`, `PhaseEmbedding`)
   and nudges the shared scalar parameters -- eta1, eta2, the soft
   threshold -- via a small per-phase offset (`PhaseScalarHead`). Roughly
   7-8x fewer parameters than the baseline; every shared parameter gets a
   gradient update from every phase, every step.
2. **Denoiser prior instead of the low-rank framing.** The baseline's N1
   module is explicitly motivated by structured-low-rank Hankel-matrix
   theory. Here, `SharedDenoiserModule` reframes it as a general-purpose
   residual denoiser: `output = input - predicted_residual` (DnCNN-style
   residual learning), dropping the low-rank framing entirely. Still a
   plain 1D CNN under the hood, so the paper's efficiency argument for 1D
   over 2D convolution is untouched.
3. **Cross-phase attention.** In the baseline, phase *k* only ever sees
   phase *k-1*'s output -- everything from phases 1..k-2 is gone.
   `CrossPhaseAttention` keeps a memory of each shared module's past
   outputs and lets the current phase attend back over all of them
   (pooled query/key vectors for the attention weights, full feature maps
   as values, so spatial detail is recoverable, not just a pooled
   summary) before proceeding. One instance each for the denoiser and the
   sparse-transform pair.

These are complementary, not conflicting: sharing means the *weights*
computing each phase's transformation are the same, but each phase still
receives a different input (its own predecessor's output) and a different
phase embedding, so its *computed* output still differs -- attention
looks at those different computed outputs, not at the shared weights.

## What's unchanged from `../odls/`

`masks.py`, `metrics.py`, `data.py`, `fastmri_data.py`, `losses.py` are
copied verbatim -- none of them depend on model internals, only on the
`(e_preds, sym_pairs)` contract that `ODLSv2.forward` still produces in
the same shape as the baseline's `ODLS.forward`. `checkpoint_utils.py`,
`train.py`, `evaluate.py` are adapted only to construct `ODLSv2` (with
its extra `--embed-dim` hyperparameter) instead of `ODLS` -- the
checkpointing, resume, `--init-checkpoint` pretraining, and smoothed
best-checkpoint logic are otherwise identical in behavior to the
baseline.

## Status: implemented and integration-tested, NOT yet trained or validated

Before writing this README, the actual code was verified, not just
compiled:
- A forward pass through a small `ODLSv2` produces the right output
  shapes and dtypes.
- A backward pass confirms **every** parameter receives a gradient (zero
  dead parameters) -- weight sharing, FiLM, and attention are all
  actually wired into the computation graph, not silently bypassed.
- `clamp_thresholds_()` runs without error, mirroring the baseline's
  runaway-threshold safeguard (same `max_threshold` mechanism, since
  nothing about sharing removes the underlying incentive that caused the
  threshold to explode in the baseline before that fix existed).
- A full integration test using the *actual* `ODLSLoss` class (not a toy
  loss) and a real `optimizer.step()` runs cleanly end to end.

None of that proves it trains *well* or improves on the baseline -- only
that it's not broken at the level tests can catch without real data and
real training time. Whether weight sharing, the denoiser reframing, and
cross-phase attention actually help (per the suggested direction's own
claims) is an empirical question this code hasn't yet been used to
answer.

## Usage (same CLI shape as `../odls/`)

```
python train.py --fastmri-train-root <dir> --fastmri-val-root <dir> \
    --n-coils 8 --n-virtual-coils 8 --embed-dim 16 \
    --mask-type cartesian --af 4 --checkpoint-dir <dir>

python evaluate.py --fastmri-test-root <dir> --checkpoint <dir>/odls_best.pt \
    --n-coils 8 --n-virtual-coils 8 --embed-dim 16 --mask-type cartesian --af 4
```

`--embed-dim` must match between training and evaluation of the same
checkpoint (it's a construction-time hyperparameter, not part of the
saved `state_dict`) -- same rule as `--max-threshold` / `--n-phases` /
`--width` in the baseline.
