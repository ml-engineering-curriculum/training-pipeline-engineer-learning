# exercise-05: Megatron-Style Tensor-Parallel Port

**Estimated effort:** 4 hours

## Objective

Take the FSDP2 version of the ~1B model from exercise 4 and add a
Megatron-style tensor-parallel axis on top of it, producing a 2-D
`(tp, dp)` DeviceMesh. Demonstrate that the port is correct (loss curves
match) and that the intra-node tensor-parallel all-reduces land on
NVLink, not on cross-node fabric. This is the exercise that turns
"I have read the Megatron paper" into "I have run a Megatron-style
strategy end-to-end on my own hardware".

## Prerequisites

- Exercises 1, 3, and 4.
- Chapters 4 and 6 (parallelism design space; DDP → FSDP2 → TP ladder).
- A single node with ≥ 4 GPUs connected by NVLink (an 8-GPU DGX-class
  host is ideal, but any 4-way NVLink node works).
- PyTorch ≥ 2.4 with `torch.distributed.tensor.parallel` available.
  (You may alternatively use Megatron-LM directly, but the write-up
  must justify the choice.)

## Problem statement

The 1B model port to FSDP2 landed. Now the team wants to know what
happens when they layer tensor parallel on top so they can run a bigger
model per node. You will do the port, keep the DDP-shaped training loop
intact, and document what the collective pattern actually looks like on
the wire.

## Requirements

1. **Rebuild the model with parallelism-aware layers.**
   - Either import `torch.distributed.tensor.parallel` and apply a
     `parallelize_plan` that uses `ColwiseParallel` on the up-projection
     of each MLP and QKV, `RowwiseParallel` on the down-projection and
     the output projection of attention, and `SequenceParallel` on the
     layer norms; **or** use Megatron-LM's `ColumnParallelLinear` and
     `RowParallelLinear` layers directly. Do not mix the two paths.
   - Verify that each transformer block introduces exactly the four
     activation collectives predicted by chapter 4 (two on forward, two
     on backward).
2. **Set up a 2-D DeviceMesh.**
   - `mesh = init_device_mesh("cuda", (dp_size, tp_size),
     mesh_dim_names=("dp", "tp"))`.
   - Choose `tp_size ≤ 8` and keep the whole TP process group on a
     single node. Confirm this by checking that all ranks in the TP
     group share a hostname (or by inspecting `NCCL_DEBUG=INFO`'s
     topology output).
3. **Compose with FSDP2 on the `dp` axis.**
   - Apply `fully_shard(block, mesh=mesh["dp"])` after applying the
     tensor-parallel plan.
   - Confirm that parameters are DTensors with the expected
     `(Replicate|Shard(x))` placements on each axis.
4. **Correctness check.**
   - Fix the RNG. Run a short training on a synthetic task (`shape
     matmul` teacher, or a tiny in-memory language-modeling task) with
     `tp_size=1, dp_size=N` and `tp_size=N, dp_size=1` and confirm the
     loss curves match within numerical tolerance for the first 100
     steps. If they diverge, the port has a bug.
5. **Measure the collective landscape.**
   - Capture a `torch.profiler` trace of one training step with the
     final `(dp × tp)` configuration.
   - In the trace, identify: TP activation all-reduces (many, small,
     intra-node), FSDP all-gathers of layer parameters (fewer, larger,
     depends on the mesh), FSDP reduce-scatters on gradients.
   - Confirm from the profiler that the TP all-reduces stay inside the
     node.
6. **Write the port report** (300–600 words) covering:
   - The parallelism plan (which layer got which sharding annotation).
   - How you verified correctness (loss curves, DTensor placements).
   - The collective landscape observed in the profiler and how it
     matches chapter 4's prediction.
   - The step-time comparison against the FSDP2-only baseline. Explain
     which direction it moved and why.
   - The next axis you would add (PP? SP?) if you had to scale to a
     model that would not fit on this node.

## Starter guidance

- Start with `tp_size = 2` before going to 4 or 8. It is much easier to
  debug a mismatched shape or dropout-RNG issue at TP=2 than at TP=8.
- Do not try to be clever about the plan. The Megatron-style
  column-then-row pattern for MLPs and QKV+output for attention is
  the well-trodden path; the PyTorch docs' TP tutorials use exactly
  this plan.
- When running the correctness check, keep the batch size and sequence
  length small so 100 steps fits in a few minutes.
- If loss diverges: check that `nn.Dropout` uses tensor-parallel-aware
  RNG (Megatron's `get_cuda_rng_tracker` idiom, or the equivalent
  PyTorch `SequenceParallel` wrapper), and check that the LM head
  loss reduction is over the full vocab, not just the local shard.
- If TP=8 works but TP=16 falls off a cliff, you crossed the node
  boundary. Verify with `NCCL_DEBUG=INFO`. Do not "fix" this by tuning
  NCCL — TP across nodes is (almost always) the wrong strategy.

## Acceptance criteria

- A working 2-D `(dp, tp)` implementation of the ~1B model that trains
  for at least a few hundred steps without divergence.
- Loss curves for `(tp=1, dp=N)` and `(tp=N, dp=1)` match within
  numerical tolerance on the same synthetic task.
- Profiler evidence (a screenshot or a JSON excerpt) showing that TP
  all-reduces are intra-node and FSDP collectives run on the outer
  axis.
- The port report explains what changed in the memory footprint, in
  the step time, and in the collective pattern, referencing the
  cost-model formulas from chapter 5.
- At least one primary reference (Megatron-LM paper, `torch.distributed
  .tensor.parallel` docs, or torchtitan example) is cited.

## Stretch goals

- Add `SequenceParallel` on the layer norms and dropout to shard
  activations along the sequence dimension inside the TP group.
  Measure the reduction in activation memory.
- Increase the model size to 3B parameters and re-tune the mesh
  (e.g., `tp=4, dp=2` vs `tp=8, dp=1`). Report which configuration
  wins on tokens/sec and why.
- Compare against a **pure Megatron-LM** implementation of the same
  model (using `pretrain_gpt.py` as a starting point) and record any
  gap in throughput. Do not spend more than an hour on this — the
  point is a directional data point, not a benchmarking paper.
