# Communication-Compute Overlap

Chapter 1 named "communication exposed on the critical path" as a
5–25 pp MFU gap term — and, at scale, this is the single largest
lever remaining after FA, BF16/FP8, and PT2 are all in place. Every
byte of NCCL traffic that is not hidden behind compute lands directly
in `t_step`. This chapter is about the mechanisms each framework
exposes for hiding it, the profiler patterns that let you *see* the
gap in a trace, and the numbered playbook for closing it.

The two primary references:

- **PyTorch Profiler recipe (Kineto backend).**
  https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- **NVIDIA Nsight Systems.**
  https://developer.nvidia.com/nsight-systems

## The three overlaps that matter

Break the training step into a sequence of collectives and compute
regions per layer `i`:

- **Forward layer i.** `AG_i` (FSDP2 all-gather of layer i's
  parameters) → `Fwd_i` (compute) → optional `Free_i` (release the
  transient full params).
- **Backward layer i.** `AG'_i` (re-all-gather if
  `reshard_after_forward=True`) → `Bwd_i` (compute) → `RS_i`
  (reduce-scatter of layer i's gradients).

The three overlaps a well-tuned run achieves:

1. **Forward all-gather prefetch.** `AG_{i+1}` runs *while* `Fwd_i`
   is computing, so the gather completes before `Fwd_{i+1}` needs
   it. Same for backward: `AG'_{i-1}` runs while `Bwd_i` computes.
2. **Backward reduce-scatter overlap.** `RS_i` runs *while*
   `Bwd_{i-1}` computes, so the reduction completes without adding
   wall-clock time.
3. **Optimizer step overlap.** The parameter update happens as
   soon as `RS_i` lands; if you have async DCP saves or an
   optimizer-in-backward feature, the update can begin during
   `Bwd_{i-1}`. This is a mod-102/mod-106 concern; here we note that
   FSDP2 defaults to a serial optimizer step after all gradients
   are reduced.

The idealised picture: the sum of compute times equals the step
time. The comm time hides entirely behind the compute. Real runs
achieve this at ~80–95% overlap; the last 5–20% of comm is exposed
and directly hits MFU.

## FSDP2's overlap machinery

The three FSDP2 knobs that control overlap:

- **`reshard_after_forward`.** If `True` (default), the full params
  are freed after forward and re-all-gathered on backward. Costs one
  extra all-gather per layer per step. Set `False` on the *outermost*
  wrapped block only, to keep its parameters live between forward
  and backward — saves one AG at the cost of memory. Applied via
  `fsdp_module.set_reshard_after_forward(False)`.
- **Prefetching depth.** FSDP2 issues the next layer's all-gather
  as soon as it can. The depth (how far ahead) is controlled by
  `set_reshard_after_forward` combined with the underlying stream
  handling; recent PyTorch releases prefetch one layer ahead by
  default, with an explicit knob for deeper prefetch.
- **`set_requires_gradient_sync(False)`.** Skips the RS during
  gradient accumulation. Wrap this in `model.set_requires_gradient_sync(False)`
  for accumulation micro-steps and back to `True` for the sync
  step. Standard for effective-batch-size scaling; without it, RS
  fires on every micro-step and dominates.

<!-- needs-research: verify current FSDP2 prefetch depth default and the exact API name for the "look-ahead" knob in the latest PyTorch release. -->

The pattern in `torchtitan`'s reference config keeps
`reshard_after_forward=True` on every block except the outermost;
that layer keeps its params live, saving one AG per step on the
biggest single collective. See
https://github.com/pytorch/torchtitan for the canonical setup.

## DeepSpeed's overlap machinery

DeepSpeed's ZeRO-3 exposes:

- **`overlap_comm: true`.** Overlap gradient reduction with
  backward compute. Default `true` in modern configs.
- **`contiguous_gradients: true`.** Bucket gradients into a
  contiguous buffer so the reduction is one collective. Default
  `true`.
- **`reduce_bucket_size` and `allgather_bucket_size`.** Byte
  thresholds for bucketing. Too small: many small collectives, high
  latency term. Too large: one collective too big to hide behind
  one compute region.

Tuning DeepSpeed's overlap is mostly bucket-size tuning: pick a
size where each bucket's collective time roughly matches one
layer's compute time. The `nccl-tests` sweep from mod-105 gives you
`busbw` per payload size; combine with a per-layer compute time
from a profiler trace to pick the bucket.

## Megatron-LM pipeline schedule interaction

Under 3D-parallel with pipeline parallel (mod-101 chapter 4),
Megatron-LM interleaves 1F1B (Narayanan et al., 2021,
arXiv:2104.04473). The pipeline schedule and comm overlap interact:

- **1F1B minimises the bubble.** Bubble fraction is
  `(P - 1) / (M + P - 1)` for `P` stages and `M` micro-batches, per
  the GPipe / PipeDream analysis. Interleaved-1F1B further shrinks
  it by the interleave count.
- **PP send/recv overlaps with compute per micro-batch.** The
  cross-stage P2P send / recv completes while the next micro-batch's
  forward runs on the same stage. Megatron's schedule emits the
  correct order.
- **TP intra-node all-reduces on activations do not overlap.**
  Tensor-parallel activation all-reduces are on the critical path
  because the next matmul depends on the reduced tensor. This is
  why TP > 8 (crossing the NVLink boundary) hurts so much — the
  activation all-reduce is exposed and now on IB, not NVLink.

The mainstream production configuration for large models:
FSDP2 (or ZeRO-3) for DP + intra-node TP (≤ 8) + PP across
nodes. Overlap wins are per-axis:

- DP: prefetched AG + overlapped RS. Chapter's core focus.
- TP: no overlap. Just keep TP intra-node.
- PP: 1F1B / interleaved-1F1B schedule.

## Reading the trace

Two tools; use both.

### PyTorch profiler + Kineto

```python
import torch.profiler as prof

with prof.profile(
    activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.CUDA],
    schedule=prof.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=prof.tensorboard_trace_handler("./trace"),
    record_shapes=True,
    with_stack=True,
) as p:
    for step in range(6):
        train_step(model, batch)
        p.step()
```

Open the resulting trace in Perfetto or TensorBoard. The signals
you look for on each CUDA stream row:

- **NCCL stream.** One row of collective ops (`ncclAllGather`,
  `ncclReduceScatter`, `ncclAllReduce`). Every gap of dead time on
  the compute streams that lines up with active time on this stream
  is *exposed comm* — the collective is on the critical path.
- **Compute stream.** Matmul and elementwise kernel rows. A "sawtooth"
  gap pattern between kernels usually means launch overhead
  (chapter 6 owns that) or scheduler stalls.
- **FSDP2's collective annotations.** FSDP2 tags its collectives
  as `fsdp:all_gather_params_<layer_name>` and
  `fsdp:reduce_scatter_grad_<layer_name>`. The name-mapping lets you
  point at a specific block whose collective is not overlapping.

For quick tabular analysis, `torch.autograd.profiler.profile` has
key averages; for deep dives, the Kineto Chrome trace is the
canonical artefact.

### NVIDIA Nsight Systems (`nsys profile`)

`nsys` gives you a system-level view that Kineto does not: the
NCCL kernel plus the NIC IB traffic, all on one timeline.

```bash
nsys profile \
    -t nvtx,cuda,cudnn,cublas,nvme,osrt,nvlink,mpi \
    --gpu-metrics-device=all \
    --stats=true \
    -o training-nsys-trace \
    python train.py --steps 20
```

The report includes:

- CUDA API and kernel timelines per GPU.
- NVLink / NVSwitch traffic (if `--gpu-metrics-device` is set).
- NCCL calls annotated with the collective type and payload size.
- IB / RoCE NIC statistics per rank.

The typical use: run 20 steps, open the `.qdrep` in the Nsight
Systems GUI, zoom to one representative step, and read across the
GPU rows plus the NIC rows. If NIC utilisation is 100% during a
compute region, comm is on the critical path — the collective is
issuing traffic faster than the compute is finishing.

## The comm-exposure playbook

The numbered sequence for closing the exposed-comm term:

1. **Capture a trace.** Kineto for FSDP2-tagged collectives + `nsys`
   for NIC timeline. 20 steps is enough; skip the first 5 for
   warm-up.
2. **Identify the exposed collective.** Find the largest gap
   between the end of one compute region and the start of the next
   where the NCCL stream is active. That is the exposed collective.
3. **Attribute to a layer.** FSDP2's tagged names or Megatron's
   op names identify which layer's collective is exposed.
4. **Ask: is it prefetched?** If it is a `fsdp:all_gather_params_<n>`
   and it fires *at the start of* `Fwd_n`, prefetch is broken. Fix
   depends on the framework version — sometimes a `set_reshard_after_forward`
   change on the parent block, sometimes a
   `set_all_reduce_hook` / prefetch-depth knob.
5. **Ask: is the collective too big for one compute region?** If
   the collective is longer than the compute region it is trying
   to hide behind, no scheduling change will hide it. Options:
   split the parameter (finer-grained sharding), or split the
   compute (higher gradient-accumulation micro-batches so each
   compute region is longer).
6. **Ask: is the fabric bottlenecked?** `busbw` should be within 10%
   of nominal (mod-105 chapter 5). If not, the fix is at the fabric
   layer — NCCL algorithm, PXN, IB HCA assignment — not at the
   training layer.
7. **Re-measure MFU.** Prove the delta.

The single most common finding: `reshard_after_forward=True` on
the outermost wrapped block, causing one extra huge AG at the top
of backward that dominates. `set_reshard_after_forward(False)` on
that block only, keep it `True` elsewhere. Net memory cost: one
block's transient full parameters; MFU gain: several pp.

## Gradient accumulation and comm

For effective-batch-size scaling with gradient accumulation:

```python
model.set_requires_gradient_sync(False)  # skip RS this micro-step
for micro_batch in accum_batches[:-1]:
    loss = train_step(model, micro_batch)
    loss.backward()

model.set_requires_gradient_sync(True)   # sync on the last one
loss = train_step(model, accum_batches[-1])
loss.backward()

optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Without `set_requires_gradient_sync(False)` on the non-final micro-
steps, the RS fires N times per accumulation cycle, one per micro-
step. On a large model with a wide TP × PP axis this can turn a
32-micro-step accumulation into 32× the reduction cost.

DDP has the same idea via the `no_sync()` context manager.
DeepSpeed handles the accumulation-sync gate internally when
`gradient_accumulation_steps` is set.

## Interaction with mod-105 NCCL tuning

Every overlap improvement in this chapter assumes the underlying
NCCL is delivering near-line-rate `busbw`. If it is not — because
of PXN misconfiguration, wrong `NCCL_IB_HCA`, missing rail
alignment — no application-side overlap fix will close the gap.
Mod-105 chapter 5's tuning surface is the prerequisite; this
chapter's tools are the *demand*-side complement.

The mental model:

- Mod-105 chapter 5 tunes NCCL so that `busbw ≈ nominal_bw`.
- Mod-101 chapter 5 computes the *comm volume* per step from
  first principles.
- This chapter's tools close the *exposure* fraction of that
  volume by scheduling it behind compute.

All three have to be right for MFU to hit the target.

## Comm-compute ratio as the metric

The metric to defend: `T_comm_exposed / T_step`. For a well-tuned
FSDP2 run at multi-node scale, this should be under 10%. If it is
20%+, the run is comm-bound and adding more GPUs will hurt.

Compute it from a trace:

- `T_comm_exposed` = sum of dead time on compute streams that
  coincides with active NCCL time.
- `T_step` = wall-clock per step (chapter 1's `t_step`).

The ratio is a much better predictor of MFU than raw NCCL
`busbw`. `busbw` measures the fabric's health; the exposed ratio
measures the run's health.

## Summary

- Every byte of NCCL traffic not hidden behind compute lands in
  `t_step`. Overlap is the single largest MFU lever at multi-node
  scale after FA/BF16/FP8/PT2 are in place.
- FSDP2 overlaps forward AG prefetch, backward RS, and (via
  `set_requires_gradient_sync`) gradient accumulation. Owned knob:
  `reshard_after_forward` on the outermost wrapped block.
- DeepSpeed exposes `overlap_comm`, `contiguous_gradients`, and
  bucket-size knobs. Tune bucket sizes so each collective matches
  one layer's compute time.
- Megatron's 1F1B / interleaved-1F1B schedule minimises the PP
  bubble. TP activation all-reduces are *not* overlapped and are on
  the critical path — keep TP intra-node.
- Read overlap in traces. Kineto (`torch.profiler`) for
  FSDP2-tagged collectives; `nsys` for the system view including
  IB NIC statistics. Find the exposed collective, attribute to a
  layer, ask "is it prefetched?" and "is it too big for one
  compute region?"
- The metric: `T_comm_exposed / T_step`. Well-tuned target < 10%.
  Higher means comm-bound.
- Depends on mod-105's NCCL tuning being right first. Overlap is
  the demand side; NCCL tuning is the supply side.
