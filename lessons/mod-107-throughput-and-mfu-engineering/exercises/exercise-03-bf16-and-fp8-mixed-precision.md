# exercise-03: BF16 and FP8 Mixed-Precision Rollout

**Estimated effort:** 4 hours

## Objective

Turn chapters 3 and 4 into a working, defensible mixed-precision
rollout. First: build the canonical BF16 policy on FSDP2 and prove
it does not degrade convergence relative to an FP32 reference. Then
(on Hopper only): layer FP8 via Transformer Engine on top and repeat
the convergence + lift measurement. Deliverable: a trainer that can
switch between the three precisions on a CLI flag, and a report
that publishes the MFU delta and a loss-curve equivalence check for
each transition.

## Prerequisites

- Chapters 3 and 4 of this module.
- Exercises 1 and 2 completed. You are measuring on top of a
  known-good baseline with FlashAttention already integrated.
- FSDP2-capable PyTorch (2.4+ recommended). Confirm
  `torch.distributed.fsdp.fully_shard` is available.
- For the FP8 half of the exercise: Hopper hardware (H100 or
  H200) and NVIDIA Transformer Engine installed (`pip install
  transformer_engine[pytorch]` or the source build per the TE
  user guide). Non-Hopper hardware: complete only the BF16 half
  of the exercise and note the reason in the report.
- The chapter-2 verification discipline: you will do profile-based
  checks that the intended kernel actually dispatched.

## Problem statement

Your training run is currently on FP32 (or on a shaky autocast
setup you inherited). Management wants BF16 for the wall-clock
saving; the ML team wants proof it does not degrade convergence.
On Hopper, FP8 is the next available lift, but only if you can show
it converges. Both transitions are numerically delicate; both are
also standard, and the discipline for validating them is well-
established.

You will:

1. Build the FP32 reference (baseline for convergence).
2. Migrate to BF16 with FSDP2 `MixedPrecisionPolicy`. Publish the
   MFU delta and the convergence equivalence.
3. On Hopper only: migrate the model to Transformer Engine modules
   and enable FP8 under `te.fp8_autocast`. Publish the MFU delta
   and the convergence equivalence.

## Requirements

Ship one directory:

- `mp_bench/` — the harness (Python).
- `mp_rollout_report.md` — the report.

### 1. The harness (`mp_bench/`)

Extend the exercise-1 trainer:

- A `--precision {fp32, bf16, fp8}` CLI flag.
  - `fp32`: no mixed precision anywhere.
  - `bf16`: FSDP2 `MixedPrecisionPolicy(param_dtype=bfloat16,
    reduce_dtype=float32)`; no `autocast`.
  - `fp8`: same BF16 policy plus TE modules and
    `te.fp8_autocast(DelayedScaling(Format.HYBRID))` around the
    forward pass.
- For the FP8 branch, factor the model to use TE modules
  (`te.Linear`, `te.LayerNormMLP`, `te.RMSNorm`,
  `te.DotProductAttention`). Do this cleanly — no mixing
  `nn.Linear` with `te.Linear` in the same block.
- Same seed for all three precisions across a given equivalence
  run; separate seeds fine for the throughput measurement.

### 2. Convergence equivalence

For each transition, run for the same number of steps (a few
thousand is enough to see divergence trends) with the same seed
and data ordering:

- **FP32 → BF16.** Publish the max relative divergence in loss
  and a plot of both curves. Small noise (~1e-3 rel) is expected;
  visible drift is a bug in the policy.
- **BF16 → FP8.** Same procedure. The FP8 curve is expected to
  be slightly noisier for the first ~1000 steps as the
  `amax_history` fills. If the FP8 curve settles at a
  materially higher loss than BF16, either the recipe is wrong
  or a non-TE module is still running in BF16 inside the
  `fp8_autocast` scope (chapter 4).

### 3. Throughput / MFU

For each precision, run the exercise-1 measurement harness (500
warm + 500 measured steps). Publish:

- Median step time and P95.
- MFU (with the correct `P` for the precision — remember to switch
  to the H100 FP8 peak of 1979 TFLOP/s for the FP8 configuration
  when computing MFU denominator? Actually — chapter 1 says use
  the peak *at the precision you train the matmuls in*, so yes.
  Explain your choice in one paragraph in the report).
- Peak HBM per rank.

### 4. Profile verification (FP8 only)

Chapter 4's verification requirement: profile one step of the
FP8 configuration and confirm you see FP8 kernel names (`fp8`,
`e4m3`, `e5m2`, `hopper_fp8_gemm`, etc.). If the kernels are still
BF16, the `fp8_autocast` scope is wrong or a non-TE `nn.Linear`
slipped in. Screenshot or extract the top-N kernel table.

## The `mp_rollout_report.md`

Structure:

1. Hardware/software matrix (GPU, CUDA, PyTorch, TE version).
2. The MixedPrecisionPolicy used (spelled out; not "we used the
   defaults").
3. FP32 → BF16 transition:
   - Convergence equivalence: max-rel loss divergence, curve
     figure, one-paragraph interpretation.
   - Throughput: median/P95 step time, MFU (each precision using
     its own peak; explain the denominator choice explicitly).
   - Any surprises (e.g., "cross-entropy was silently downcast
     until we explicitly upcast logits to FP32").
4. BF16 → FP8 transition (Hopper only):
   - The exact TE recipe (`DelayedScaling(Format.HYBRID,
     amax_history_len=...)`).
   - Which modules are TE (list them) and which are not.
   - Convergence equivalence: same as above.
   - Throughput: same as above.
   - Profile verification: kernel names dispatched.
5. Shipping recommendation: BF16 always? BF16 + FP8 only for the
   pretraining phase? BF16 + FP8 with a specific rollback trigger?
   Defend in a paragraph.

### 5. Failure modes hit and diagnosed

At least three real issues from the rollout. Starting points:

- A hand-written softmax that reduced in BF16 and produced a
  visible loss plateau (chapter 3).
- The `autocast` context still present inside FSDP2's
  MixedPrecisionPolicy, causing double-cast surprises.
- A `nn.Linear` still living in a TE block, muting the FP8 lift.
- A DelayedScaling `amax_history` too short, causing scale
  factors to jitter and the loss curve to jitter with them.
- Optimizer state accidentally in BF16, causing slow drift after
  ~10k steps.
- An FP8 attention run where the surrounding model was still
  BF16 (chapter 4's warning), producing scale mismatches.

For each: symptom, diagnosis, fix.

## Starter guidance

- **Do the transitions one at a time.** FP32 → BF16 first, land
  it, then BF16 → FP8. Combining the two rollouts is the single
  fastest way to end up debugging both at once.
- **Run the equivalence check with the same seed and data
  ordering.** Any variation drowns out the actual precision-
  induced divergence.
- **Do not leave `autocast` around when using MixedPrecisionPolicy.**
  Chapter 3 warns about this; the harness should make it a
  hard error rather than a warning.
- **Standardize on TE modules in the FP8 branch — no half-
  measures.** A single `nn.Linear` inside an otherwise-TE block
  will look like a mysterious 15% throughput hole in the
  profile.
- **Do not attempt FP8 attention through SDPA.** Chapter 2 and
  chapter 4 both say this: SDPA does not know about the
  `fp8_autocast` recipe. If you want FP8 attention, use
  `te.DotProductAttention`.
- **Log the scale factors.** TE exposes the current amax and
  scale for each FP8 tensor; log them per step so any scale
  storm is visible immediately.

## Acceptance criteria

- Three precision configurations run to completion (or two if
  FP8 is not supported on the hardware).
- Convergence equivalence is quantified for each transition,
  with the max-rel-divergence number and a plot.
- MFU is published for each precision with the correct
  denominator (dense peak at that precision), and the choice is
  explained.
- Profile verification confirms FP8 kernels dispatched (Hopper
  only).
- The report includes at least three real, diagnosed failure
  modes.
- The shipping recommendation is opinionated and defended.
- No invented numbers; failed configurations are documented as
  failed.

## Stretch goals

- **`reduce_dtype` ablation.** At your world size, does
  `reduce_dtype=torch.bfloat16` on the gradient reduce-scatter
  degrade convergence measurably vs. `reduce_dtype=torch.float32`?
  Chapter 3 says "yes at multi-thousand-GPU scale"; if you can
  reproduce it at 8 or 16 GPUs, note whether the effect is
  detectable at your scale. If not, publish the null result — it
  is genuinely useful data.
- **`amax_history_len` ablation.** Try FP8 with
  `amax_history_len ∈ {16, 128, 1024}`. Plot loss noise vs.
  history length. Recommend a value.
- **Compare `Format.HYBRID` vs. `Format.E4M3`.** Show that E4M3-
  all-round diverges (chapter 4 predicts it will) and how quickly.
  Bring receipts.
- **Optimizer-in-FP8 experiment (research-adjacent).** TE has
  optimizer-state FP8 work in flight; if your TE version supports
  it, try it in a small sandbox and report what breaks. This is
  strictly a sandbox exercise; do not ship.
