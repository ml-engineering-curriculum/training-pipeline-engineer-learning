# exercise-06: Communication–Compute Overlap Measurement

**Estimated effort:** 3 hours

## Objective

Turn chapter 7 into a working measurement of the FSDP2 forward-all-
gather and backward-reduce-scatter overlap ratios on your cluster,
identify the overlap bugs the chapter warns about, and publish an
overlap-tuning report with numeric before/after evidence. Deliverable:
a `torch.profiler`- or Nsight-based measurement of the overlap ratio,
a per-lever tuning table, and a shipping recommendation.

## Prerequisites

- Chapter 7 of this module.
- Exercises 1–5 completed. The trainer is BF16, FA-integrated,
  FSDP2-wrapped, and (from exercise 5) known to compile cleanly.
- Access to at least 2 nodes' worth of GPUs (ideally 4+). Overlap
  characteristics change materially between single-node and multi-
  node because the fabric changes (mod-105).
- mod-105 chapter 5 completed for at least a rough `nccl-tests`
  characterization of your fabric. You need bandwidth numbers to
  compare "exposed comm" against theoretical comm.
- `torch.profiler` available. Optional: `nsys` (NVIDIA Nsight
  Systems) installed for the cross-node view.

## Problem statement

Your MFU has improved through exercises 1–5. The remaining gap in
your budget (chapter 1) attributed some percentage points to
"communication". Chapter 7 says the tool for that bucket is to
raise the overlap ratio — the fraction of collective time hidden
inside compute. You need to (a) measure the current overlap ratio,
(b) find and fix the loop-internal synchronization bugs that
silently reduce it, (c) tune the FSDP2 wrap policy and the
`limit_all_gathers` knob, and (d) publish a defensible overlap
number with the context to be re-run later.

## Requirements

Ship one directory:

- `overlap_bench/` — the harness (Python).
- `overlap_report.md` — the report.

### 1. The harness (`overlap_bench/`)

- The exercise-1/exercise-5 trainer with instrumentation added:
  - `torch.profiler` with a bounded schedule (a few warm steps,
    a few active steps), configured to record CPU + CUDA
    activities.
  - NVTX ranges around the training loop's major phases:
    `data`, `forward`, `loss`, `backward`, `optim_step` —
    makes the profile easier to read and lets you correlate
    NCCL kernels with the phase they belong to.
- CLI flags for the tuning variables:
  - `--wrap-granularity {block, layer, sublayer}` — controls
    how many `fully_shard` calls you make and thus how many
    collectives.
  - `--limit-all-gathers <N>` for the FSDP2 in-flight cap.
  - `--reduce-dtype {fp32, bf16}` (already from exercise 3).
- A `parse_overlap.py` script that consumes the chrome-trace
  JSON from `torch.profiler` and computes the overlap ratio by
  summing intersections between NCCL kernels and compute
  kernels on the same stream.

### 2. The measurements

**Baseline overlap.** Run the trainer with default FSDP2 settings
(wrap at block granularity, `limit_all_gathers` at its default).
Capture a profile. Compute the overlap ratio for:

- Forward all-gather across all layers.
- Backward reduce-scatter across all layers.
- Per-layer overlap ratio (or averaged if layers are uniform).

Publish these as your baseline.

**Predicted overlap.** Independently, from your mod-105 `nccl-
tests` numbers and your chapter-1 compute-per-layer estimate,
compute the *theoretical* overlap ratio: `min(t_compute, t_comm) /
t_comm` per layer. Compare against the measured value. If the
measured value is materially lower, that gap is what the rest of
this exercise is chasing.

**Loop-internal-sync bug hunt.** Chapter 7 lists the four common
overlap-killing bugs:

- `.item()` or `.cpu()` calls inside the training step.
- `clip_grad_norm_` using the non-FSDP-native version.
- Profiler unintentionally slowing kernels.
- Blocking dataloader `next()` calls.

Inspect the harness and any inherited code for these. Publish a
"before" and "after" overlap ratio for each bug you found and
fixed. If you found none, do a code review and defend the
result — a large codebase with genuinely zero synchronization
bugs is rare.

**Lever sweep.** For each combination that changes the overlap:

- `limit_all_gathers ∈ {1, 2, 4}` at the default wrap
  granularity.
- Wrap granularity: block-level vs. layer-level (finer).
- `reduce_dtype ∈ {fp32, bf16}`, only at world sizes where you
  are willing to characterize the impact.

Publish a table: rows are configurations, columns are overlap
ratio (forward + backward), median step time, MFU, peak HBM per
rank.

### 3. Cross-check with the Nsight timeline (stretch)

If `nsys` is available, capture a 3-step profile with `nsys
profile --stats=true -o overlap`. Screenshot or export the
timeline showing NCCL and compute streams. Confirm the visual
overlap matches the number from `parse_overlap.py`. This is the
"trust but verify" pass on the automated computation.

## The `overlap_report.md`

Structure:

1. Hardware / fabric summary (cross-reference to mod-105
   chapter 5's `nccl-tests` numbers).
2. Baseline overlap: forward and backward, measured and
   predicted. Interpret the gap in one paragraph.
3. Bug-hunt findings: each bug found, its before-and-after
   effect on the overlap ratio, the fix.
4. Lever sweep table.
5. Chosen configuration, and the paragraph defending it.
6. Chapter-7-style publish block: framework version, wrap
   policy, mixed-precision policy, world size, per-rank micro-
   batch, sequence length, `t_comm` and `t_compute` per layer,
   overlap ratio, exposed comm time per step, fabric backing.
7. Recommendation for the next optimization cycle — if the
   overlap ratio is already high (say > 0.85), the remaining
   gap-to-peak comes from other buckets and this chapter is
   done for now. If it is low even after tuning, you have a
   comm-bound run and need to grow per-layer compute (chapter 5
   packing + larger micro-batch) or reduce comm bytes.

### 8. Failure modes hit and diagnosed

At least three real issues. Examples in addition to chapter 7's
common bugs:

- A `torch.distributed.barrier()` left over from an older
  debugging session, serializing every step.
- A dataloader that returns tensors on CPU and a `.to("cuda")`
  in the loop that blocks the next kernel launch.
- An NCCL kernel taking longer than predicted because the
  fabric was serving another job (a topology / scheduler
  concern; mod-104 chapter 4). Discuss whether to reschedule
  or accept the noise.
- A framework upgrade that changed the FSDP2 defaults on
  `limit_all_gathers` and silently changed your overlap ratio.

For each: symptom, diagnosis, fix.

## Starter guidance

- **Do not trust the framework knobs. Profile.** Even with the
  correct settings, a Python-level sync in the loop can undo
  everything.
- **Use NVTX ranges liberally.** They make the profile readable
  and let you sum overlap per phase, not just per kernel.
- **Compute the overlap ratio programmatically.** Eyeballing a
  timeline is fine for a sanity check but not for a published
  number.
- **When measuring, disable other one-time work.** Skip DCP
  checkpoints during the overlap-measurement window. They
  compete for the same NCCL / storage resources and pollute
  the measurement.
- **Sanity-check against the theoretical ratio.** If your
  measured ratio is 0.4 and the theoretical is 0.95, you have a
  sync bug. If both are ~0.4, your run is genuinely comm-bound
  and no amount of prefetch tuning will fix it — the fix is in
  chapter 5's compute-per-layer axis.

## Acceptance criteria

- Baseline overlap ratios (forward + backward) are published
  with the profile as evidence.
- Predicted (theoretical) overlap ratios are computed and
  compared.
- At least one loop-internal sync bug is either found and fixed,
  or a written argument explains why none exists in the
  codebase.
- Lever sweep table has at least three rows and each row
  publishes the overlap ratio, step time, MFU, and peak HBM.
- The chosen configuration is defended against the sweep, not
  just picked.
- The chapter-7 publish block is complete.
- At least three real failure modes are documented.
- No invented numbers.

## Stretch goals

- **Cross-node vs. intra-node comparison.** Run the same
  measurement at world size N on a single node and on N/2 GPUs
  per node × 2 nodes. Publish the delta and interpret in terms
  of NVLink vs. IB bandwidth (mod-105 chapters 2 and 3).
- **SHARP on / off.** If your fabric supports SHARP (mod-105
  chapter 3), toggle it via NCCL environment variables and
  publish the overlap delta.
- **Backward-only overlap.** Some FSDP2 versions expose more
  granular knobs for backward prefetch. Explore them and
  document what the current release supports.
- **Nightly overlap regression.** Add the overlap ratio to the
  exercise-1 CI job. A framework upgrade or a well-meaning
  optimization elsewhere in the codebase that reintroduces a
  loop-internal sync is caught the day it happens.
- **Correlate with fabric telemetry.** If mod-108 (observability)
  has NIC or NVLink counters plumbed, cross-reference a low
  overlap window with those counters. A low overlap that also
  shows fabric contention is a scheduler problem; a low overlap
  with idle fabric is a training-loop bug.
