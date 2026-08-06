# exercise-01: MFU Calculation and Gap-to-Peak Analysis

**Estimated effort:** 3 hours

## Objective

Turn chapter 1 into a working measurement: instrument a small training
run, compute MFU and HFU with the correct numerator and denominator,
and fill in the five-bucket gap-to-peak budget for the run you have.
Deliverable: an `mfu_report.md` in a small repo, plus the harness
that produced it, in enough detail that a teammate could re-run it
on their allocation and get comparable numbers.

## Prerequisites

- Chapter 1 of this module. You should be able to recite
  `MFU = F_step / (G · P · t_step)` from memory before starting.
- mod-101 background on FSDP2 well enough to launch a small
  sharded training loop.
- Access to at least 2 GPUs of a single vendor generation (any of
  A100, H100, H200 works). Note the exact SXM / PCIe variant in
  the report; peak FLOPs differ.
- A dense decoder-only Transformer of a few hundred million to a
  few billion parameters. Any of nanoGPT-style, torchtitan's
  Llama-3, or an internal model works — the exact model does not
  matter as long as the parameter count is known.
- PyTorch with `torch.profiler`. FlashAttention need *not* be
  installed for this exercise — one point is to measure the
  non-FA baseline first.

## Problem statement

Every subsequent exercise in this module targets one or more
buckets of the MFU gap-to-peak budget. Before optimizing anything,
you need to know where you actually stand and where the room to
grow is largest. Your task is to publish an MFU number for your
baseline run — computed correctly, defensible against arithmetic
review — and to decompose the missing `(1 − MFU)` into the five
buckets so the rest of the module's chapters are ordered by expected
return.

The scope is deliberately narrow: no optimization changes, just
measurement.

## Requirements

Ship one directory:

- `mfu_bench/` — the measurement harness (Python).
- `mfu_report.md` — the written report.

### 1. The measurement harness (`mfu_bench/`)

At minimum:

- A `train_step.py` that runs one training step (forward + loss +
  backward + optimizer.step), instrumented to measure:
  - Wall-clock step time via `torch.cuda.Event`s (not
    `time.perf_counter`, which can miss CUDA-async gaps).
  - Peak HBM allocated per rank
    (`torch.cuda.max_memory_allocated`).
  - A `torch.profiler` window of a few steady-state steps.
- A `flops_count.py` that computes `F_step` from the model
  definition — either symbolically from the config (preferred) or
  via `torch.utils.flop_counter.FlopCounterMode` (available in
  recent PyTorch). If you use the counter, cross-check against the
  `6·N·B·S` + attention analytic formula from chapter 1 and
  reconcile any discrepancy in the report.
- A `run.sh` that launches the harness under `torchrun` for the
  world size you are testing at, and saves outputs to a versioned
  results directory.

### 2. Baseline configuration

Pick and freeze one configuration and use it for every measurement
in this exercise. Suggested defaults (adjust for what your hardware
can hold):

- Model: dense decoder-only Transformer, 1–3 B parameters.
- Global batch: whatever fits at a per-rank micro-batch of 1–4
  sequences.
- Sequence length: 2048 or 4096.
- Precision: BF16 with `MixedPrecisionPolicy(param_dtype=bfloat16,
  reduce_dtype=float32)`. (BF16 is chapter 3's default; using it
  for the baseline keeps this exercise from double-counting with
  exercise 3.)
- Attention: the *default* SDPA backend (i.e., whatever PyTorch
  picks without `sdpa_kernel(FLASH_ATTENTION)`). Do not enable FA
  explicitly for this baseline — the point is to measure the
  starting point.
- No `torch.compile`. No activation checkpointing. No sequence
  packing.
- `torch.optim.AdamW`.

Document the exact config and commit it to the harness.

### 3. The measurements

Run 500 warm steps, then 500 measured steps in each of these
configurations:

- **Baseline** as above.
- **Baseline + `sdpa_kernel(FLASH_ATTENTION)`** as a *comparison
  data point only* (this is an early preview of exercise 2; report
  the MFU delta but do not treat it as your "primary" number).

For each configuration, publish:

- Median step time and P95 step time in ms.
- Peak HBM per rank in GB.
- The `torch.profiler` trace file (attach or link).

## The `mfu_report.md`

Answer these sections in order, with numbers:

### 1. Hardware and denominator

- Accelerator model, form factor, and count (e.g., "16× H100 SXM5,
  in 2× 8-GPU DGX nodes").
- Peak FLOPs `P` at the precision you ran the matmuls in. Cite the
  datasheet URL you took the number from. Note that `P` is the
  **dense** number, not the sparsity-doubled marketing number
  (chapter 1).
- Fabric summary in one paragraph, cross-referenced to mod-105.

### 2. Numerator: model FLOPs per step

- The `6·N·B·S` term with `N`, `B`, `S` written out and computed.
  Show the arithmetic.
- The attention term. Show the layer count, head count, per-head
  dimension, and the constant you used (be explicit about
  forward-only vs. forward+backward and the `12` vs. `24` factor;
  see chapter 1).
- The counter-based estimate from `FlopCounterMode`, and the ratio
  between it and the analytic estimate. Reconcile any > ±5%
  discrepancy in one paragraph.

### 3. Baseline MFU and HFU

- Publish the formula, plug in the numbers, publish the ratio.
- Also compute HFU (executed FLOPs / denominator). For the baseline
  with no activation checkpointing, HFU and MFU should be nearly
  equal; if they differ, explain what recompute happened.

### 4. FA comparison delta (informational)

- MFU with the default SDPA backend and MFU with
  `FLASH_ATTENTION` forced. Report the delta. Do not spend a
  paragraph interpreting it — that is exercise 2's job.

### 5. Gap-to-peak budget

Fill in the five buckets from chapter 1 with numeric attributions
that sum to `(1 − MFU) · 100%`. For each bucket, cite your
evidence:

| Bucket                    | Attribution (percentage points) | Evidence / source                              |
| ------------------------- | ------------------------------- | ---------------------------------------------- |
| Kernel / dtype gap        | ?                               | e.g., "attention is 40% of step time in the profiler, and default SDPA is measurably slower than FA in the FA-comparison data point above" |
| Memory-bandwidth gap      | ?                               |                                                |
| Recomputation gap         | ?                               | (0 for a baseline with no AC)                  |
| Communication gap         | ?                               | e.g., "reduce-scatter shows N ms exposed per step"  |
| Everything-else           | ?                               |                                                |
| **Total**                 | `100 · (1 − MFU)` %             |                                                |

Attribution does not have to be surgically precise; a rough (± few
percentage points) allocation is the goal. Use the profiler trace to
justify each row.

### 6. What order should the next exercises run in for this run

Given the budget, which chapter of the module has the largest
expected return? Order exercises 2–6 by expected MFU lift and
defend the order in one paragraph each. It is fine — expected,
even — if your ordering differs from the module's chapter order for
your specific run.

## Starter guidance

- **Warm up before measuring.** The first few dozen steps are
  polluted by lazy initialization, cudnn benchmarking, and cache
  warm-up. Discard them.
- **Compute FLOPs symbolically first.** The counter can miss ops
  or double-count in some configurations; the analytic estimate
  is your sanity check.
- **Do not enable FA implicitly.** By default, recent PyTorch's
  SDPA dispatches to `FLASH_ATTENTION` on eligible hardware
  automatically. To measure a true "no FA" baseline, wrap the
  training step in `sdpa_kernel(SDPBackend.MATH)` or
  `SDPBackend.EFFICIENT_ATTENTION`, and confirm in the profile
  that the flash kernel did not fire. Note the choice in the
  report.
- **Do not fabricate the gap-to-peak numbers.** If you cannot
  attribute a bucket with any evidence, put it in "everything-
  else" and say so. Better a low-confidence but honest budget
  than a precise-looking fiction.
- **Log everything to disk in JSONL.** `report.md` should be
  reproducible from the raw data; someone reading it should be
  able to re-derive every number.

## Acceptance criteria

- The report includes an MFU number expressed to a single decimal
  point, with the numerator and denominator arithmetic fully shown.
- The peak-FLOPs `P` value is cited to a specific NVIDIA (or other
  vendor) document; the dense number is used, not the sparsity-
  doubled one.
- The numerator uses the correct dense-Transformer accounting or
  is verified against `FlopCounterMode`; discrepancies > ±5% are
  reconciled in writing.
- HFU is published alongside MFU. For a no-AC baseline they should
  be nearly equal.
- The gap-to-peak budget rows sum (within rounding) to
  `100 · (1 − MFU)` percentage points. Each row cites its evidence.
- The exercise-ordering paragraph is opinionated and defended.
- No invented profiler numbers. If the profile does not exist or
  the run did not complete, say so; do not fabricate.

## Stretch goals

- Repeat the measurement at a second world size (e.g., 8 GPUs and
  16 GPUs) and show the MFU-vs-world-size curve. Interpret any
  scaling loss as an early preview of the chapter-7 overlap
  discussion.
- Add a second precision configuration (FP32 or FP16) at the same
  batch size. Show the MFU delta and explain why the wall-clock
  delta is smaller than the FLOPs-peak ratio would suggest.
- Fill in the gap-to-peak budget for a very different model
  (e.g., a much smaller model where attention is not the
  bottleneck) and compare the shapes of the two budgets. Which
  chapter would you prioritize differently?
- Wire the harness into a nightly CI job that runs the baseline
  and posts the MFU number to a metrics store. Regression alerts
  when MFU drops > 3% become an early-warning system for
  framework upgrades that broke the fast path (chapter 2's kernel
  fallback failure mode).
