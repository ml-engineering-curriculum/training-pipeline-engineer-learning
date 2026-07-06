# exercise-01: DDP Forward + Backward Trace

**Estimated effort:** 3 hours

## Objective

Produce a whiteboard-style trace of one training step of a small
DDP-wrapped model that shows, at every moment in time, which parameter is
active on which rank, when its gradient becomes ready, when its bucket is
enqueued for all-reduce, and when the optimizer step observes the averaged
gradient. The deliverable is the artifact that proves you understand what
`loss.backward()` actually does under DDP — chapter 2 in prose form, but
grounded in a real graph and a real trace.

## Prerequisites

- Chapters 1 and 2 of this module.
- Working PyTorch install with CUDA on ≥ 2 GPUs (or a multi-GPU dev
  environment). Two GPUs on one host is enough; 4 or 8 is better because
  you can *see* the ring structure in the trace.
- Passing familiarity with the `torch.profiler` output view or, if you
  prefer, `torch.autograd.profiler` + Chrome trace viewer.

## Problem statement

You are on-call for a training platform team. A colleague ships a DDP job
that hangs 30% of the time at `optimizer.step()` and asks you to help
debug. You cannot fix what you cannot explain, so before touching their
code you first prove — on paper — that you can trace one clean step of a
minimal DDP job end-to-end.

## Requirements

1. **Build a small reference model.** Something like a 4-layer MLP with
   parameters `[W1, b1, W2, b2, W3, b3, W4, b4]` on a synthetic
   `x → y` task. It must have at least 4 named parameters so bucketing
   becomes meaningful. Do *not* use a real transformer here — the point
   is that you should be able to reason about every single tensor.
2. **Wrap it in `DistributedDataParallel`** with an intentionally small
   `bucket_cap_mb` (e.g., 1 MB) so that the model spans multiple buckets.
   Record which parameter lands in which bucket by inspecting DDP's
   reducer state or by using the `_get_bucket_sizes` API (or by
   deliberately printing at the DDP hook).
3. **Instrument the step.** Use `torch.profiler.profile(...)` with
   `activities=[CPU, CUDA]` and record one forward + backward +
   optimizer step. Export the Chrome trace and open it.
4. **Author the trace document** (submit as `trace.md` or a slide deck /
   diagram in your solutions repo). It must include:
   - A diagram of the model with buckets labelled.
   - A timeline (drawn or annotated screenshot of the profiler) showing:
     - Forward compute on each rank.
     - Autograd's *reverse-order* completion of each parameter's gradient.
     - The moment each bucket becomes ready and the all-reduce is enqueued.
     - The overlap between backward compute and all-reduce
       communication.
     - The optimizer step.
   - A short narrative (200–400 words) explaining what happens if
     parameter ordering is non-deterministic across ranks (e.g., you
     wrap in DDP after conditionally instantiating a layer on rank 0
     only). Predict the failure mode and, ideally, demonstrate it in a
     second run.

## Starter guidance

- Do not overthink the model. A stack of `nn.Linear` layers is fine.
  What matters is the *pattern* of the trace.
- If you have never used the PyTorch profiler at CUDA granularity
  before: start from the tutorial ("Profiling your PyTorch Module")
  and add `ProfilerActivity.CUDA` explicitly.
- Set `NCCL_DEBUG=INFO` on one run and paste the collectives it prints
  into the trace document; it will show you the algorithm NCCL picked
  and the payload sizes.
- Consider running two configurations: `bucket_cap_mb=1` (many small
  buckets) and `bucket_cap_mb=200` (one giant bucket). The overlap
  story is very different, and having both is a good learning artifact.

## Acceptance criteria

- The trace clearly shows overlap between backward compute and NCCL
  all-reduce for at least one bucket. Both are visible on the CUDA
  timeline with time-overlapping spans.
- Every named parameter in the model is placed in a specific bucket in
  the diagram, and the buckets are consistent with what the DDP reducer
  reports.
- The narrative correctly explains that DDP relies on identical
  parameter iteration order across ranks and correctly predicts the
  failure mode (a hang or a deadlock at the next all-reduce) if that
  order diverges.
- The write-up cites either the DDP paper (Li et al., 2020, "PyTorch
  Distributed") or the PyTorch DDP documentation for at least one
  claim it makes.

## Stretch goals

- Reproduce the same trace with `find_unused_parameters=True` after
  conditionally skipping a layer during forward. Compare the two
  timelines and explain the additional overhead.
- Reproduce the trace using `no_sync()` for gradient accumulation over
  4 micro-batches and confirm that all-reduce only fires on the final
  micro-batch.
- Force `NCCL_ALGO=Ring` vs. `NCCL_ALGO=Tree` at small payload sizes and
  compare the collective wall times against the α + β model in
  chapter 5.
