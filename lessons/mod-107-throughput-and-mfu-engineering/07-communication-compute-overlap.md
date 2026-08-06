# Communication–Compute Overlap: All-Gather Prefetch and All-Reduce Overlap

Every FSDP2 step includes collective communications: all-gather to
materialize each layer's full parameter shard before its forward,
reduce-scatter to reduce and re-shard each layer's gradients after
its backward. On a well-tuned fabric (mod-105), the *bandwidth* of
these collectives is not the problem — the problem is whether they
run *concurrently with compute* or *in series behind it*.

The metric this chapter targets is the **overlap ratio**: the
fraction of the collective's duration that is hidden inside
compute rather than exposed as its own step-time contribution. A
model with 100% overlap pays nothing at all for communication in
wall-clock terms. A model with 0% overlap adds the full collective
duration to every step. Real systems land in between; the goal is
to push toward 1 and measure honestly.

## The two overlap opportunities in FSDP2

FSDP2's forward-backward has two distinct collective classes, each
with its own overlap pattern:

- **Forward all-gather** (before each layer's forward): gather the
  sharded parameters of layer `L+1` onto every rank so `L+1`'s
  forward can compute. The prefetch opportunity is to *start*
  layer `L+1`'s all-gather while layer `L`'s forward is still
  running. If the fabric's bandwidth × the gather time fits inside
  layer `L`'s compute, the collective is fully hidden.
- **Backward reduce-scatter** (after each layer's backward):
  reduce-scatter the gradients of layer `L` while layer `L-1`'s
  backward is running. Same shape as forward prefetch, mirrored in
  time.

FSDP2 exposes both as configurable behaviors — the defaults
usually enable them, but the actual overlap ratio depends on
whether compute times exceed comm times *per layer*.

The overlap arithmetic per layer is:

```
overlap_fraction = min(t_compute_L, t_comm_L+1) / t_comm_L+1
exposed_comm     = max(0, t_comm_L+1 - t_compute_L)
```

If per-layer compute is longer than per-layer comm, you get 100%
overlap and pay zero. If comm is longer than compute, you pay the
excess. The larger the model per layer (bigger MLP, bigger attention),
the more compute there is to hide behind — this is why "make the
per-layer compute bigger" (larger micro-batch, longer sequence)
often raises MFU: it raises the compute-to-comm ratio and improves
overlap.

## Turning overlap on in FSDP2

FSDP2's `fully_shard` API auto-enables the standard overlap
patterns. The knobs to know about (see the FSDP2 tutorial for the
current names in your PyTorch version; the API has stabilized over
2.4→2.5+):

- **Forward prefetch.** Whether to issue layer `L+1`'s all-gather
  before layer `L` finishes forward. Enabled by default. Disabling
  it costs you the forward-side overlap and is only useful for
  debugging.
- **Backward prefetch.** Analogous knob for the reduce-scatter side.
  Enabled by default.
- **`limit_all_gathers`.** Bounds how many all-gathers can be in
  flight simultaneously. Higher values improve overlap but grow the
  peak HBM footprint (more unsharded param buffers alive
  concurrently). Default is a small number (1–2); raise cautiously
  and watch memory.
- **CPU offload.** If parameters or optimizer state are offloaded to
  CPU (FSDP2's optional `cpu_offload=...`), each forward now has a
  PCIe copy on the critical path. Overlap becomes a three-way race
  (PCIe copy, all-gather, compute); the analysis in this chapter
  still applies but with an extra term.

The best way to see whether prefetch is doing its job is a
profile — see the measurement section below. Do not rely on the
knob defaults; measure.

## Turning overlap on in DDP (for context)

Legacy DDP overlaps the backward all-reduce with backward compute
via the "gradient bucket" mechanism (`DistributedDataParallel`'s
`bucket_cap_mb`). Each parameter gradient, once computed, is
enqueued into a bucket; when the bucket fills, an async all-reduce
fires. The overlap ratio depends on bucket size:

- **Small buckets:** more all-reduces, more chances to overlap with
  the tail of backward.
- **Large buckets:** fewer all-reduces, better fabric utilization
  per collective, less overlap head.

DDP is not the FSDP2 story, but if you inherit a DDP codebase, the
same measurement discipline applies: profile the timeline, compare
exposed comm time to compute time, tune bucket size.

## Measuring the overlap ratio

The overlap ratio is not a single number the framework reports; you
compute it from a profile. Two paths:

### 1. `torch.profiler` timeline

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(3):
        out = model(input_ids)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

prof.export_chrome_trace("trace.json")
# open trace.json in chrome://tracing or perfetto.dev/
```

In the resulting timeline, look for:

- **NCCL kernels** — `ncclAllGather`, `ncclReduceScatter`,
  `ncclAllReduce`. Their duration is the collective's wall time.
- **Compute kernels on the same stream** overlapping in time.

The overlap ratio for a single collective is:

```
overlap = (compute_kernel_time_intersecting_with_collective) / (collective_duration)
```

For the whole step, sum numerator and denominator across every
collective. A step with 90% overlap is "nearly hidden"; a step
with 40% overlap has half the collective time exposed.

### 2. Nsight Systems

`nsys profile` around a training step produces a fuller trace
including CUDA streams, NVTX ranges, and NIC utilization. Prefer
Nsight for cross-node bandwidth questions and multi-GPU views;
prefer `torch.profiler` for quick, in-code single-node analysis.

### 3. Simple compute-vs-comm sanity check

Before profiling, compute the theoretical numbers:

- `t_comm_per_layer ≈ shard_size / bandwidth_per_rank`. Shard size is
  `param_bytes_per_layer / world_size`; bandwidth is what mod-105
  chapter 5 measured with `nccl-tests` for `all_gather` on your
  actual fabric.
- `t_compute_per_layer ≈ layer_FLOPs / (P · efficiency)`. Layer
  FLOPs from chapter 1; `P` from the datasheet; `efficiency` from
  the same measurement.

If `t_compute_per_layer > t_comm_per_layer`, full overlap should be
achievable. If the profile shows exposed comm, something is
misconfigured — usually `limit_all_gathers=1` when it could be 2, a
CUDA-stream serialization from an errant `.item()` call in the loop,
or the framework simply not issuing the prefetch. If
`t_comm > t_compute`, you cannot hide the excess without changing
the compute/comm ratio (bigger layer, less sharding).

## Design levers that shift the ratio

If your measured overlap is chronically below what you want, the
levers are:

- **Grow per-layer compute.** Larger micro-batch or longer sequence
  raises the compute side of the ratio. This is often the cheapest
  fix — a well-packed batch (chapter 5) increases compute-per-comm
  by definition.
- **Reduce comm bytes.** Reducing gradient reduction precision from
  FP32 to BF16 (chapter 3) halves the reduce-scatter payload. Only
  do this at scales where the numerical impact is negligible; at
  multi-thousand ranks, prefer FP32.
- **Choose fewer, larger FSDP2 units.** Every FSDP2 unit becomes an
  independent all-gather. Wrapping every layer with its own
  `fully_shard` is fine; wrapping every MLP sub-layer separately
  creates too many small collectives and the fabric's per-launch
  overhead dominates. Wrap at the Transformer block level, not at
  the sub-op level. See the FSDP2 tutorial's wrap-policy discussion.
- **Enable SHARP if your fabric supports it.** In-network reduction
  (mod-105 chapter 3) removes bytes from the fabric on reduce-
  scatter. This changes the comm-side arithmetic; re-measure.

Do *not* try to hide comm by "just doing more work on the CPU" or
by other cargo-cult tricks. The right lever is one of the above.

## Common overlap bugs

Failures observed in the field that hurt overlap without making
the bug obvious:

- **Synchronous `.item()` or `.cpu()` calls inside the training
  loop.** Every `.item()` forces a CUDA sync, which prevents the
  next kernel launch from overlapping with the collective in flight.
  Log tensor values to metric collectors *between* steps, not inside
  the loop; use non-blocking `to("cpu", non_blocking=True)` when
  moving data.
- **Gradient clipping computing the norm on the wrong stream.**
  `clip_grad_norm_` under FSDP2 must be the sharded version; the
  vanilla version can serialize behind the reduce-scatter and cost
  you the overlap.
- **Profiler enabled unintentionally.** `torch.profiler` with an
  aggressive schedule adds instrumentation that widens kernel
  intervals and can appear to reduce overlap. Profile in a
  representative but bounded window (`schedule=schedule(wait=1,
  warmup=2, active=3, repeat=1)`), and remove profiling in
  benchmark measurements.
- **A blocking dataloader `next()`.** If the loader is slower than
  the previous step, the loader's `next()` becomes a
  synchronization point on the main thread. Use `pin_memory=True`
  and `prefetch_factor` (mod-103 chapter 5).
- **`torch.distributed.barrier()` calls.** Any explicit barrier
  serializes everything. Remove them from the loop; use them only
  in test harnesses.

## How to publish an overlap number

An overlap ratio without context is useless. Publish:

- The framework and version (PyTorch 2.X, FSDP2 API).
- The wrap policy (block-level; whether the model was regionally
  compiled; the FSDP2 knobs used).
- The mixed-precision policy (chapter 3): `param_dtype`,
  `reduce_dtype`.
- The world size, per-rank micro-batch, and sequence length.
- The measured `t_comm` and `t_compute` per layer (or averaged if
  layers are uniform), and the resulting overlap ratio.
- The exposed comm time per step (`step_time_measured -
  step_time_predicted_from_compute_only`).
- The fabric measurement it was built on top of (link back to
  mod-105 chapter 5's `nccl-tests` numbers).

Anything less and the number is not comparable to another team's
number or to a next-quarter measurement of the same job.

## Summary

- FSDP2's forward all-gather and backward reduce-scatter both
  support overlap with compute. The metric to publish is the
  overlap ratio: the fraction of collective time hidden inside
  compute rather than exposed on the critical path.
- Forward prefetch and backward prefetch are on by default in
  FSDP2. `limit_all_gathers` bounds how many all-gathers are in
  flight and trades memory for overlap.
- Measure with `torch.profiler` (single-step, quick) or Nsight
  Systems (fuller, cross-node). Do not trust the framework knobs
  without profiling.
- Levers to raise the overlap ratio: grow per-layer compute
  (larger batch or packed sequence), shrink comm bytes (BF16
  reduction if safe), coarsen the FSDP2 wrap unit (block-level),
  enable in-network reduction (SHARP).
- Kill the loop-internal synchronizations that silently break
  overlap: `.item()`, blocking `.cpu()`, sync barriers, ad-hoc
  logging. These are the most common causes of a "surprising"
  low overlap ratio.
- Publish overlap numbers with context: framework version, wrap
  policy, mixed-precision policy, world size, batch, seq len, and
  the fabric measurement backing them.
