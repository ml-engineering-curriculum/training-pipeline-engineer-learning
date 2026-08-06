# exercise-05: `torch.compile` with FSDP2 — Integration, Recompiles, and Cache

**Estimated effort:** 3 hours

## Objective

Turn chapter 6 into a working `torch.compile` rollout on top of the
BF16 + FSDP2 trainer you built in exercises 1–3. Deliverable: the
compile turned on cleanly (no graph breaks you cannot explain), a
report that measures the incremental MFU lift, and an operational
runbook covering the two failure modes chapter 6 flags: recompile
storms and shared-FS cache thrash on the first step at scale.

## Prerequisites

- Chapter 6 of this module.
- Exercises 1–3 completed. You have a BF16 + FSDP2 trainer (with
  or without FA) and an MFU harness.
- PyTorch 2.4 or newer. Confirm `torch.compile` is available and
  that FSDP2's `fully_shard` API is present. Note both versions
  in the report.
- Access to at least 4 GPUs to see the cache-thrash issue
  meaningfully (single-node runs mask it because the shared FS
  is the local FS).
- Familiarity with the `TORCH_LOGS` environment variable and the
  `torch._dynamo` debug knobs.

## Problem statement

Your BF16 trainer is running at a stable MFU (from exercise 1's
baseline, further improved by exercise 2's FA integration). You
have been told `torch.compile` "typically gives 10–20% more MFU"
and you want to see if that lands on your workload — but you also
want to avoid the well-known operational pitfalls (first-step
minutes-long compile per rank, recompile storms from variable
shapes, silent graph breaks that leave you at eager speed
thinking compile fired).

You will:

1. Turn compile on at the top level and measure the MFU lift.
2. Verify no unintended graph breaks are silently muting the
   win.
3. Verify no per-step recompiles are happening under your
   loader's shape distribution.
4. Set up the Inductor cache correctly for a multi-rank cluster.
5. Publish an operational runbook covering the three failure
   modes.

## Requirements

Ship one directory:

- `compile_bench/` — the harness (Python).
- `compile_rollout_report.md` — the report and runbook.

### 1. The harness (`compile_bench/`)

- CLI flags:
  - `--compile {off, top-level, block-level, max-autotune}`.
    - `top-level`: `model = torch.compile(model, mode="default")`
      *after* FSDP2 wrapping.
    - `block-level`: compile each `Block._forward_impl`
      independently.
    - `max-autotune`: `mode="max-autotune"` at the top level.
  - `--dynamic-shapes {off, on}`. Controls `dynamic=None` vs.
    `dynamic=True` in the `torch.compile` call.
  - Optionally `--fullgraph` for the debug bring-up.
- Emit compile-timing logs at:
  - The first step (compile happens here).
  - Every recompile event
    (via `torch._dynamo.utils.compile_times()`).
- The exercise-1 MFU harness for step-time measurement, with an
  extra "step 1" bucket separated from "steady-state" so the
  first-step compile time does not pollute the average.
- Cache-directory management:
  - Support setting `TORCHINDUCTOR_CACHE_DIR` per rank.
  - Report the size and location of the cache after a run.

### 2. The four measurements

For each of `off / top-level / block-level / max-autotune`:

- **First-step time.** Wall-clock of step 1 (includes compile).
  Report per rank and worst-case across ranks.
- **Steady-state step time and MFU** (from step 500 onward),
  using the exercise-1 harness.
- **Recompile count** across the 500-step window
  (`torch._dynamo.utils.compile_times()` or the `TORCH_LOGS=recompiles`
  count).
- **Graph-break count** in a single-step debug trace with
  `TORCH_LOGS=graph_breaks`. List the top 3 break reasons in
  the report; explain each in one sentence.

### 3. The cache-thrash drill

On at least 4 GPUs, run the same top-level compile configuration
in two setups:

- **Bad setup:** `TORCHINDUCTOR_CACHE_DIR=/shared/fs/inductor` (a
  parallel FS mount, or `/tmp` if you have no parallel FS —
  simulate with a single directory shared across ranks). All
  ranks write to the same directory concurrently.
- **Good setup:** `TORCHINDUCTOR_CACHE_DIR=/local/nvme/inductor-
  rank-${LOCAL_RANK}` (or similar per-rank local path).

Publish:

- First-step wall-clock in each setup.
- Any errors observed in the bad setup (locking failures,
  partial writes, mysterious slow compilations).
- The size of the cache after each run.

### 4. The recompile-storm drill

Run with `--dynamic-shapes off` (the default) on a loader that
produces variable batch shapes each step. Either:

- Turn *off* the sequence-packing bin size and let the loader
  emit whatever comes out.
- Or use a synthetic loader that varies micro-batch length by
  step.

Measure recompiles per 100 steps and the impact on steady-state
step time. Then repeat with `--dynamic-shapes on` and show the
recompile count drops (or with fixed bucketing that fixes the
shape variation). Publish both configurations.

## The `compile_rollout_report.md`

Structure:

1. Software matrix (PyTorch version, FSDP2 status, whether FA is
   installed).
2. The four-measurement table for
   `off / top-level / block-level / max-autotune`:
   first-step time, steady-state step time, MFU, MFU delta from
   baseline, recompile count, graph-break count.
3. Graph-breaks list: top three break reasons, one-sentence
   explanation each, and (if you eliminated any) what change you
   made to remove them.
4. Cache-thrash drill: the two setups, the first-step numbers,
   and the recommended production setup.
5. Recompile-storm drill: the two loader configurations, the
   recompile counts, and the recommended production setup.
6. Interaction with FA and BF16: any observation about compile
   affecting the profile of the attention or the mixed-precision
   dispatched kernels.
7. Recommendation: which compile mode ships, with which cache
   configuration, with which shape strategy.

### 8. Failure modes hit and diagnosed

At least three real issues. Starting points:

- A `.item()` inside the training loop that Dynamo could not
  trace, producing a graph break in the middle of each block.
- A custom module using `hasattr` on tensor properties that
  Dynamo evaluated at compile time, leading to a stale
  specialization.
- A shared cache directory that produced compile timeouts and
  filesystem errors at 32+ ranks.
- A `mode="max-autotune"` that made step 1 take an unusably long
  time on a first cold-cache run.
- A `torch.compile` call *before* FSDP2 wrap that produced silent
  correctness issues on the first forward.
- Interaction with the DCP checkpoint path (mod-106) where the
  cache warm-up on resume was slower than a cold cache because
  Dynamo's specializations were invalidated by loaded parameter
  strides.

For each: symptom, diagnosis, fix.

## Starter guidance

- **Wrap first, compile second.** Chapter 6 is explicit; the
  reverse ordering silently corrupts strides.
- **Start with `fullgraph=False` and inspect graph breaks; then
  try `fullgraph=True` after you have cleaned them up.** Skipping
  the inspection step is how you ship a "compiled" run that runs
  at eager speed.
- **Use `TORCH_LOGS=graph_breaks,recompiles,inductor` for the
  debug run.** The output is noisy; grep for the actual events.
- **Set `TORCHINDUCTOR_CACHE_DIR` explicitly.** The default is
  under the home directory, which on many clusters is a shared
  NFS mount — see the cache-thrash drill.
- **Do not benchmark first-step time as if it were normal step
  time.** The first step includes compilation; report it
  separately.
- **Do not fold `torch.compile` into the FA integration reports.
  Keep the measurements clean.**

## Acceptance criteria

- The four-measurement table is completed for at least three of
  the four modes. Skipping `max-autotune` with the reason "cold
  compile too long for the exercise window" is acceptable if
  documented.
- Graph-break count is verified from the `TORCH_LOGS=graph_breaks`
  output; do not report "zero" without the log evidence.
- The cache-thrash drill shows a first-step time delta between
  the bad and good setups. If your cluster's shared FS is
  actually fast enough to make the difference invisible, report
  the null result and explain why.
- The recompile-storm drill shows non-trivial recompile counts
  when dynamic shapes are off, and a reduction when they are on.
- The recommendation names a specific mode, cache setup, and
  shape strategy — not a menu.
- At least three real failure modes, diagnosed.
- No invented numbers.

## Stretch goals

- **Compile time budgets.** Wrap `torch.compile` in a wall-clock
  budget so that a runaway first-step (>10 minutes) times out and
  falls back to eager with an alert. Discuss whether you would
  ship this to production.
- **Cache pre-warming.** Add a one-time "compile-warmup" job to
  your CI that runs the compile against a canonical shape set
  and publishes the resulting cache tarball to your artifact
  registry. Training jobs download and unpack the cache on
  startup, skipping the first-step compile. Measure the
  end-to-end saving on a 32-rank cold start.
- **`unsloth` / library kernels.** Integrate one published Triton
  kernel from a library (e.g., a fused SwiGLU) that registers as
  a `torch.library` op. Confirm it composes with `torch.compile`
  and does not introduce a graph break. This is exercise 8's
  boundary (chapter 8) made concrete: you *integrate* the
  kernel; you do not author it.
- **Compare against `torch._inductor.config` tweaks.** Modes like
  `coordinate_descent_tuning=True`, `max_autotune_gemm=True`,
  `epilogue_fusion=True`. Report which tuned deltas were worth
  the compile-time cost.
