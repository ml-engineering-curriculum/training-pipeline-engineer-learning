# exercise-04: Checkpointing and Sequence Packing to a Memory Budget

**Estimated effort:** 3 hours

## Objective

Turn chapter 5 into a working "hit a fixed per-rank HBM budget while
paying as little MFU as possible" exercise. Deliverable: a trainer
that supports the four levers from the chapter (no-AC, selective
AC, full AC, sequence packing) plus micro-batch reduction with
gradient accumulation; a report that walks through the decision
list on your hardware; and a set of `torch.cuda` memory snapshots
that visually justify each choice.

## Prerequisites

- Chapter 5 of this module.
- Exercises 1–3 completed. You have a trainer with FA and BF16
  (and possibly FP8) working, and an exercise-1 measurement
  harness that computes MFU.
- Access to at least 2 GPUs; the exercise works with any modern
  Ampere or Hopper generation.
- FlashAttention with variable-length support (`flash_attn_varlen_func`)
  installed for the packing branch.
- A dataset (or synthetic dataset generator) that produces
  variable-length sequences. The packing win only shows up on
  distributions with real length variation.

## Problem statement

Your model currently OOMs at the micro-batch and sequence length
you want to run at. You could reduce the micro-batch, but that
raises comm-per-token and can hurt convergence dynamics. You could
add full activation checkpointing, but that costs ~33% of your
forward FLOPs. You could pack sequences, but that requires
loader changes. What is the cheapest combination that fits your
budget?

You will:

1. Measure the un-tuned per-rank HBM at a target
   `(micro_batch, seq_len)`.
2. Set a memory target `M` (e.g., 60 GB on an 80-GB H100 to
   leave headroom for the loader, DCP staging, and NCCL
   buffers).
3. Walk chapter 5's decision list: pack → selective AC → full AC
   → gradient-accumulation reduction, measuring at each step.
4. Publish the final configuration, its HBM, and its MFU.

## Requirements

Ship one directory:

- `mem_bench/` — the harness (Python).
- `mem_target_report.md` — the report.

### 1. The harness (`mem_bench/`)

Extend the exercise-1/exercise-3 trainer:

- CLI flags:
  - `--activation-checkpointing {none, selective, full}`.
    Use `torch.utils.checkpoint` (`use_reentrant=False`) or the
    FSDP2 `apply_activation_checkpointing` wrapper.
  - `--pack-sequences {off, on}`. When on, the dataloader
    concatenates short sequences up to `max_seq_len` and returns
    `(input_ids, cu_seqlens)` for `flash_attn_varlen_func`.
  - `--micro-batch <N>` and `--grad-accum-steps <K>`. Effective
    batch is `N × K × world_size`.
- Memory instrumentation:
  - `torch.cuda.max_memory_allocated()` published per step.
  - A `torch.cuda.memory._record_memory_history()` snapshot for
    one full step in each configuration, exported for the memory-
    viz viewer (https://pytorch.org/memory_viz).
- Same MFU measurement path as exercise 1.

### 2. The walkthrough

For the target `(micro_batch, seq_len)`:

- **Row 0: baseline.** No AC, no packing. If this fits your target
  `M`, publish MFU and stop. Otherwise, record the OOM (or
  peak-HBM) and continue.
- **Row 1: packing only.** Enable `--pack-sequences on`. Publish
  peak HBM, throughput in *useful tokens/s* (not padded
  tokens/s), and MFU (using useful tokens in the numerator).
- **Row 2: packing + selective AC.** Same measurements.
- **Row 3: packing + full AC.** Same measurements.
- **Row 4: packing + selective AC + reduced micro-batch with
  matching gradient accumulation.** Same measurements. Effective
  batch size must match Row 2's.
- **Row 5: packing + full AC + further-reduced micro-batch with
  matching gradient accumulation.** Same measurements.

Stop at the first row that fits `M` with the highest MFU. Publish
that row as the recommended configuration and defend the choice.

### 3. Memory-viz snapshots

For at least three configurations (baseline, best-of-AC-only,
best-of-packed-plus-AC), attach or link the memory-viz snapshot
that shows the activation footprint over one forward+backward
step. Annotate what the eye should see: the flat activation
plateau in baseline, the sawtooth from AC's re-materialization,
the compressed profile with packing.

## The `mem_target_report.md`

Structure:

1. Target `M`, and the rationale for its value (leave HBM
   headroom for DCP async staging, NCCL buffers, occasional
   allocator fragmentation).
2. Dataset length distribution: histogram of sequence lengths
   before packing, and the packing efficiency (mean useful
   tokens per packed batch / `max_seq_len`).
3. Walkthrough table (rows 0–5 above): peak HBM (GB),
   useful-tokens/s, MFU, comment.
4. Chosen configuration, and the paragraph defending it against
   chapter 5's decision list.
5. Memory-viz snapshots for the three configurations, with the
   what-you're-looking-at annotation.

### 6. Failure modes hit and diagnosed

At least three real issues. Starting points:

- Using `use_reentrant=True` and getting silent gradient errors
  under FSDP2.
- Wrapping every sub-module with `checkpoint()` and getting a
  tiny memory saving with a large compute cost (over-checkpointing).
- Packing sequences without resetting position IDs and getting
  quality regression (chapter 5).
- Forgetting to swap to `flash_attn_varlen_func` for the packed
  path and materializing a huge `(seq_len × seq_len)` mask that
  OOMs.
- Reducing micro-batch without matching gradient accumulation
  and changing the effective batch size.
- CPU-offloading FSDP2 params to work around HBM and getting a
  massive PCIe-copy slowdown (chapter 7's overlap discussion).

For each: symptom, diagnosis, fix.

## Starter guidance

- **Warm up before measuring HBM.** The first few steps allocate
  and free various one-time buffers; the steady-state peak is
  what matters.
- **`empty_cache()` before each measurement.** Otherwise
  fragmentation from a prior configuration inflates the measured
  peak.
- **Compute useful-tokens/s carefully.** With packing, the batch
  contains `sum(cu_seqlens)` real tokens; do not use
  `batch_size × max_seq_len` as the token count.
- **Do not pack sequences that require cross-sequence attention.**
  E.g., long-context QA where the answer references a
  concatenated document. The exercise assumes standard causal-
  LM training.
- **Do not add "just a little" AC to a block that FA already
  handles.** FA already recomputes the attention softmax on
  backward; adding AC around the whole block re-runs FA's
  forward for the attention, which is a compute cost with no
  additional memory saving.
- **Interpret the memory-viz timeline before adjusting.** The
  timeline shows which specific allocations survive the forward
  and get retained for backward. Any tensor that survives and
  is on the AC list should shrink in the AC'd version.

## Acceptance criteria

- The walkthrough table has at least rows 0 through 4 completed
  (row 5 optional if row 4 already met the target).
- Peak HBM is published per row and matches the memory-viz
  snapshot for at least three rows.
- Useful-tokens/s is used for the packed rows, not padded
  tokens/s.
- MFU is published for each row, computed with useful tokens in
  the numerator.
- The chosen configuration meets the target `M` and its
  rationale is a defense against chapter 5's decision list, not
  a preference.
- At least three real failure modes are documented.
- No invented numbers; if a row failed to run, the report
  documents it.

## Stretch goals

- **Length-distribution sensitivity.** Repeat the packing
  measurement on a synthetic dataset with a shifted length
  distribution (e.g., all sequences equal to `max_seq_len`).
  Show that packing has zero benefit in that regime.
- **Selective-AC schedule ablation.** Selective AC has multiple
  variants (recompute attention only, recompute MLP only,
  recompute norms only). Measure each and pick the best MFU-per-
  GB combination.
- **CPU-offload comparison.** Enable FSDP2's `cpu_offload` for
  parameters or optimizer state. Compare peak HBM and MFU
  against the AC-based configurations at the same effective
  batch. Under what circumstances is CPU offload the right
  answer? (Usually: very-few-GPU setups where HBM is the hard
  bind. On a full cluster it is almost never worth it.)
- **Memory regression alert.** Fold peak HBM into the exercise-1
  CI job. A framework upgrade that accidentally raises peak HBM
  is a slow way to run out of budget at scale; catch it on the
  day it happens.
