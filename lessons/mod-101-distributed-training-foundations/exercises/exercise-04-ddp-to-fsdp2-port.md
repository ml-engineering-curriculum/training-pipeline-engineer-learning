# exercise-04: DDP → FSDP2 Port

**Estimated effort:** 4 hours

## Objective

Take a working ~1B parameter DDP training loop and port it to FSDP2,
measuring the memory and step-time changes at each stage. Deliver a small
codebase and a short write-up that reproduces the DDP → FSDP2 transition
from chapter 6, with real numbers from your own hardware.

## Prerequisites

- Chapters 2, 3, and 6 (this module).
- ≥ 2 GPUs on one host, and ideally 4 or 8. Two H100 80 GB or two A100
  80 GB is plenty; smaller cards work if you shrink the model.
- PyTorch ≥ 2.4 (for `torch.distributed.fsdp.fully_shard`).

## Problem statement

The training-platform team is preparing to move all mid-size runs from
DDP to FSDP2 as the default. Before that switch is rolled out, you owe
the team a concrete DDP-vs-FSDP2 comparison on a representative model,
with the numbers to justify the change.

## Requirements

1. **Baseline DDP loop.** Build (or reuse from exercise 1) a decoder-only
   transformer of ~1B parameters (`hidden ≈ 2048`, `layers ≈ 24`, `seq
   ≈ 2048`, vocab ≈ 32k) with AdamW, BF16 autocast, and DDP over your
   world size. It reads a synthetic or pre-tokenized dataset — the
   dataset is not the point of this exercise.
2. **Record baseline metrics.**
   - Wall-clock per step (mean over 20 steps after 10 warmup).
   - Peak GPU memory (`torch.cuda.max_memory_allocated()` reset every step,
     averaged).
   - Tokens / second.
   - GPU utilization from `nvidia-smi` or DCGM.
3. **Port to FSDP2.**
   - Introduce a 1-D `DeviceMesh` on the `cuda` device with a `"dp"`
     axis.
   - Apply `fully_shard` to each transformer block, then to the outer
     model, with a `MixedPrecisionPolicy` that uses BF16 params, BF16
     compute, and FP32 reductions.
   - Remove `DistributedDataParallel` — FSDP replaces it.
   - Confirm the training loop body is otherwise unchanged.
4. **Record FSDP2 metrics** on the same batch size, sequence length, and
   step count. Compare against baseline.
5. **Then double the batch size (or the sequence length) under FSDP2**
   and repeat the measurement. The claim is that FSDP2's memory savings
   let you fit more work per step; you must demonstrate it and quantify
   it.
6. **Write the comparison report** — 300–600 words plus a table. It must
   include:
   - The table of `(DDP baseline, FSDP2 same-batch, FSDP2 larger-batch)`
     with peak-memory, step-time, and tokens/sec columns.
   - One paragraph on where the memory savings came from
     (parameters / gradients / optimizer state) with rough arithmetic
     showing you understood which line item shrank by what factor.
   - One paragraph on where the step time changed and why, referring to
     chapter 3's `3(N-1)/N · S` wire volume.
   - A recommendation: default to FSDP2 or not, and under what
     conditions.

## Starter guidance

- Use the FSDP2 example in the PyTorch documentation
  (`torch.distributed.fsdp.fully_shard`, DeviceMesh, DTensor) as the
  reference — do not copy an FSDP1 example, since the migration story
  is different.
- If loss curves diverge between DDP and FSDP2, the most common cause is
  RNG or initialization state — seed all ranks identically and
  initialize parameters on `meta` device where possible.
- Use `torch.profiler` on a handful of steps to confirm that the
  all-gather of layer `i+1` is overlapping with the compute of layer
  `i`. If it is not, throughput will be much worse than the model
  predicts; investigate `forward_prefetch` and related knobs.
- Save the FSDP2 sharded state dict via `torch.distributed.checkpoint`
  (DCP) or gather to rank 0 with `full_state_dict`. If your only save
  path is a naive `state_dict()`, rank 0 will OOM on the model you just
  chose specifically for its memory profile.

## Acceptance criteria

- The codebase runs both DDP and FSDP2 configurations from a single
  script or a small CLI (e.g., `python train.py --strategy ddp` vs
  `--strategy fsdp2`).
- The measurement table shows a peak-memory reduction at the
  same-batch-size FSDP2 configuration consistent with sharding the
  optimizer state across ranks (roughly `1/N` on the sharded pieces,
  activations unchanged).
- The larger-batch (or larger-seq) FSDP2 configuration is documented to
  fit in memory when the DDP baseline could not.
- The write-up correctly attributes step-time changes to the extra
  all-gather on backward (chapter 3), not to FSDP being "faster" or
  "slower" in a vague sense.
- Loss on a fixed synthetic task tracks between DDP and FSDP2 within a
  small band across 100+ steps. If it does not, the port is buggy and
  the exercise is not complete.

## Stretch goals

- Convert the same code to **HSDP** with a 2-D DeviceMesh
  (`("shard", "replicate")`) and repeat the memory / step-time table.
  Show at what world size HSDP starts to win on step time.
- Enable **activation checkpointing** on every block and re-measure.
  Confirm the trade: memory drops, compute rises. Report the ratio.
- Compare FSDP2's per-parameter sharding to FSDP1's flat-parameter
  sharding on the same model. Do not spend more than 30 minutes on this
  — the goal is to see the DTensor-based sharded state directly.
