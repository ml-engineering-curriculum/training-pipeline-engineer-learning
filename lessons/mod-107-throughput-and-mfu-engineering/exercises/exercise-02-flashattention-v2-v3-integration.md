# exercise-02: FlashAttention v2 / v3 Integration and Lift Measurement

**Estimated effort:** 4 hours

## Objective

Turn chapter 2 into a working integration and, more importantly, a
defensible measurement of what FlashAttention actually bought you.
Deliverable: a small trainer with three attention configurations
(SDPA math backend, FA v2, FA v3 where hardware permits), the four-
step measurement protocol from chapter 2 executed end-to-end, and a
written report that publishes the kernel-isolation lift, the full-
step lift, and the profile verification for each.

## Prerequisites

- Chapter 2 of this module.
- Exercise 1 completed. You have a working baseline trainer and a
  measured MFU number that this exercise will move.
- One of:
  - A100 (SXM4 or PCIe): you can run baseline + FA v2. FA v3 is
    Hopper-only; skip that configuration.
  - H100 or H200 SXM5: you can run all three.
- FlashAttention installed. For v2, `pip install flash-attn` is
  the standard path; for v3, follow the build matrix at
  https://github.com/Dao-AILab/flash-attention and pin the version
  in the report.
- `nvidia-smi` output, `torch.version.cuda`, and the
  `flash_attn.__version__` all recorded in the report — this is
  the single most common source of "why did my numbers not
  reproduce" questions.

## Problem statement

Your team has a trainer that has been running on the SDPA default
attention. A colleague is convinced FlashAttention is "a big win" but
cannot say how big or on which specific piece of the step. You are
going to answer that with numbers, verify the right kernel actually
dispatched, and check that the numerics are within tolerance.

## Requirements

Ship one directory:

- `fa_bench/` — the harness (Python).
- `attention_lift_report.md` — the report.

### 1. The harness (`fa_bench/`)

Reuse the exercise-1 trainer as the base. Add:

- A `--attn-backend {math, efficient, flash2, flash3}` CLI flag.
  Inside the training step, use `sdpa_kernel` to force the
  backend for `math` / `efficient` / `flash2`; for `flash3`,
  either rely on the SDPA dispatcher picking it (on recent
  PyTorch on Hopper) or call the `flash_attn` package directly for
  the v3 kernel.
- A `microbench_attn.py` script that runs the kernel-isolation
  microbench from chapter 2 for each backend, for the exact
  `(B, H, S, D, dtype, causal)` your model uses.
- Structured JSONL output for all measurements. `report.md` should
  be reproducible from the JSONL.

### 2. The four-step measurement protocol

For each backend (math baseline, FA v2, FA v3 if Hopper), execute
in this order:

- **Step 1: Kernel-isolation microbench.** Publish attention-op
  wall time per call, and the throughput implied. Include:
  - The `(B, H, S, D, dtype, is_causal)` used.
  - The number of iterations and warm-up count.
  - Median and P95 per-call time.
- **Step 2: Full-step lift.** Run the exercise-1 trainer for 500
  warm + 500 measured steps in each configuration. Publish median
  step time, P95 step time, MFU, and HFU (following exercise 1's
  MFU protocol).
- **Step 3: Profile verification.** For each configuration, capture
  a `torch.profiler` trace covering one step. Confirm that the
  attention kernel name matches what you expect (`flash_fwd_kernel`
  / `flash_bwd_kernel` for v2; a Hopper-specific FA3 kernel name
  for v3; something SDPA-math-shaped for the baseline). Screenshot
  or extract the top-N kernel table into the report.
- **Step 4: Numerical equivalence.** Run 200 identical training
  steps from the same seed with each backend, and log per-step
  loss. Publish the max relative divergence between the loss
  curves. Small floating-point noise (< 1e-3 rel) is expected;
  visible drift is a bug.

### 3. The lift accounting

For each backend transition (math → v2, v2 → v3), publish:

- The **kernel-only speedup**: ratio of attention op time.
- The **full-step speedup**: ratio of step time.
- The **MFU delta** in percentage points.

Explain the difference between kernel-only and full-step: if
kernel-only says 2× and full-step says 1.15×, that tells you
attention was ~15% of the step and there is not more to get from
attention alone — it points you at the next chapter of the module.

## The `attention_lift_report.md`

Structure:

1. Hardware and software matrix (GPU, CUDA, PyTorch, flash-attn
   version).
2. The `(B, H, S, D, dtype)` tuple used across all measurements.
3. Kernel-isolation microbench table: rows are backends, columns
   are median-per-call, throughput, and relative speedup.
4. Full-step measurement table: rows are backends, columns are
   median step time, P95 step time, MFU, HFU, delta from baseline.
5. Profile verification: one figure or table per backend showing
   the actual attention kernel names that fired. Explicitly call
   out if a backend fell back silently.
6. Numerical equivalence: the loss-curve divergence numbers and a
   one-paragraph interpretation ("agrees within FP noise" or
   "diverges — likely because …").
7. Recommendation for shipping.

### 6. Optional: FP8 attention preview (Hopper only)

If your hardware supports it and you have Transformer Engine
installed, add a fourth configuration: FA v3 in FP8 mode via
`te.DotProductAttention` under `te.fp8_autocast`. This is a preview
of exercise 3; do the numerical equivalence check especially
carefully, and note in the report that this configuration is
correct only when the *surrounding* model is also FP8 (which it
will not be here) — the point is to see the kernel dispatch, not
to ship this configuration.

## Starter guidance

- **Verify the FA install before benchmarking.** A version mismatch
  between `flash-attn`, `torch`, and CUDA is the most common
  reason FA silently falls back. Print `flash_attn.__version__`,
  `torch.__version__`, `torch.version.cuda` at the top of the
  script.
- **Force the backend for the baseline.** Recent PyTorch's SDPA
  will *auto-dispatch* to FLASH_ATTENTION on Hopper. Your "math
  backend baseline" must actively use `sdpa_kernel(SDPBackend.MATH)`
  or `EFFICIENT_ATTENTION`, and the profile should confirm the FA
  kernel did *not* fire.
- **Micro-bench shapes must match training.** Do not benchmark
  attention at some canonical `(4, 32, 2048, 128)` when your
  training uses `(2, 40, 4096, 128)`. Speedups vary by shape.
- **Do the numerical check with the exact same seed and data.**
  Any variation in the data or seed will drown out the actual
  backend divergence.
- **Do not conflate the FA v3 dispatch path with the FA v3 FP8
  path.** V3 in BF16 vs. v3 in FP8 are two different kernels and
  two different correctness cases.

## Acceptance criteria

- All measurements from the four-step protocol are published for
  each backend the hardware supports. If FA v3 is not supported on
  your hardware, that row is marked N/A with the reason.
- Every kernel-name row is verified by a profile — no
  "FlashAttention should have fired" without a screenshot or
  extract that proves it did.
- MFU deltas are computed with the exercise-1 formula and the
  same `P` value from the same datasheet.
- The numerical-equivalence check produces a max-relative-
  divergence number, not a hand-wavy "looked fine".
- The shipping recommendation is defended against the delta.
  "Ship FA v2" and "ship FA v3" are both acceptable recommendations
  if the evidence supports them; "ship whichever is newer" is not.
- No invented numbers; if a run failed, the report explains what
  went wrong.

## Stretch goals

- **Sequence-length sweep.** Repeat the full-step measurement at
  `S ∈ {1024, 2048, 4096, 8192, 16384}` for the best backend. Plot
  MFU vs. `S`. Explain the shape (attention becomes a larger
  fraction of step time as `S` grows, so the FA lift grows too).
- **Head-dimension sweep.** Some model configs use non-standard
  `D` (96, 192, 256). Show which values FA supports on your
  version and which fall back to the SDPA `efficient` backend.
  This is a report-a-shape-gap-to-the-performance-team stretch.
- **Sequence packing preview.** Use `flash_attn_varlen_func` with
  a synthetic packed batch (real-length distribution, `cu_seqlens`
  computed) and compare against fixed-length padded. Report the
  effective-token-throughput lift. This is a preview of exercise
  4.
- **Nightly CI integration.** Fold the microbench into the
  exercise-1 nightly job so that any FA version bump or PyTorch
  upgrade that changes the dispatched kernel is caught the day it
  happens.
