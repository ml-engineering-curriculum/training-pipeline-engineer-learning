# Offload Head-to-Head: FSDP2 CPU Offload vs. ZeRO-Infinity

Both FSDP2 and DeepSpeed can push training state off HBM to make a
model fit that otherwise would not. They do it differently, they
target different tiers of the memory hierarchy, and they pay for
that offload in different ways. This chapter puts them side by side
so that exercise 3 has a framework to compare its measurements
against.

The question we are actually answering is *not* "which is faster",
because the correct answer depends on the model, the cluster, and
which memory tier you were about to overflow. The question is
"which offload strategy is right for a specific `(model, cluster,
throughput target)` triple", and the analysis that follows gives
you the terms to reason with.

## The memory hierarchy that matters

Before comparing the two frameworks, name the tiers we can push
state onto, in order of decreasing bandwidth and increasing
capacity:

| Tier   | Bandwidth (per-GPU, order of magnitude) | Capacity per node (typical H100 box) | Latency  |
|--------|-----------------------------------------|--------------------------------------|----------|
| HBM (GPU memory)          | ~2 TB/s          | 8 × 80 GB = 640 GB                     | ~100 ns  |
| Host DRAM (system memory) | 400–800 GB/s aggregate; PCIe Gen4 x16 ≈ 32 GB/s per GPU | 1–2 TB          | ~100 ns from CPU, PCIe hop from GPU |
| NVMe SSD (local)          | Per drive: 3–7 GB/s read; array of drives can hit 20–50 GB/s | 4–30 TB per node | ~10–100 µs |
| Networked storage         | 10–100 Gb/s per node          | Effectively unlimited | ~ms |

The two useful ratios to remember:

- **HBM ↔ DRAM**: PCIe Gen4 x16 caps a single GPU's read/write to
  host at ~32 GB/s. That is roughly **1/60** of HBM bandwidth. So
  any per-step traffic that has to cross PCIe is dramatically
  cheaper if it can be overlapped with compute — and dramatically
  expensive if it can't.
- **DRAM ↔ NVMe**: an NVMe drive at ~5 GB/s is another ~1/100 of
  DRAM bandwidth. So the moment you push state onto NVMe, you are
  taking two orders of magnitude off the top of your available
  bandwidth for whatever traffic hits that tier.

Offloading buys memory ceiling; it charges throughput. Neither
framework can bend those physical bandwidths.

## What FSDP2 can offload

FSDP2 exposes CPU offload through `CPUOffloadPolicy`, passed to
`fully_shard`. There is no NVMe tier as of current PyTorch
releases:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy

for block in model.blocks:
    fully_shard(
        block, mesh=mesh,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                       reduce_dtype=torch.float32),
        offload_policy=CPUOffloadPolicy(pin_memory=True),
    )
fully_shard(model, mesh=mesh, offload_policy=CPUOffloadPolicy(pin_memory=True))
```

What that does:

- **Parameter shards** live on the CPU when not in use. Before a
  block's forward, its parameter shards are copied to GPU
  (H2D over PCIe), then all-gathered across the shard group.
- **Gradient shards** live on the CPU after they are computed.
  After the block's backward, the reduced gradient shard is
  copied D2H.
- **Optimizer state** (Adam moments, FP32 master parameters) lives
  on the CPU. The optimizer step runs on GPU, but the state is
  streamed through PCIe.

The optimizer step still runs *on the GPU* by default. You can
combine `CPUOffloadPolicy` with a CPU optimizer if you want the
step itself on CPU, but PyTorch does not currently ship a
CPU-optimized fused AdamW; you would use a stock `torch.optim.AdamW`
on the CPU DTensor shard, and that is generally slower per step
than DeepSpeed's `DeepSpeedCPUAdam`.

Design implication: FSDP2 CPU offload is essentially a swap-file
for HBM. Small models trained on modestly memory-constrained GPUs
benefit; extreme-scale models with 100B+ optimizer state do not
fit because host DRAM caps at ~2 TB.

## What DeepSpeed can offload

DeepSpeed's ladder has more rungs:

- **ZeRO-Offload** (CPU tier): shard optimizer state (+ optionally
  parameters) and place them in pinned host DRAM. Use
  `DeepSpeedCPUAdam` (an AVX/AVX-512 kernel) to run the Adam step
  on CPU, so the GPU does not idle during the update. Parameters
  and gradients still cross PCIe on every step.
- **ZeRO-Infinity** (NVMe tier): shard optimizer state and
  parameters to NVMe. `libaio` streams tiles of parameters into
  pinned host buffers and then to GPU HBM as compute advances,
  keeping the deepest tier of the pipeline saturated with async I/O.
  Memory-centric tiling means an individual layer can be *larger*
  than HBM as long as its tiles fit.

The DeepSpeed engine is aware of the tier at each level; the
partitioning of parameters, gradients, and optimizer state is
decided at `initialize` time and communicated to the ZeRO stage-3
partitioner.

## The direct comparison

Assume a 7B decoder-only transformer on an 8-GPU node with 80 GB
HBM each and 1 TB host DRAM. The parameter count of 7B implies:

- Params (BF16): 14 GB
- Grads (BF16): 14 GB
- Optimizer state (AdamW with FP32 master + FP32 moments): ~84 GB
  (6 bytes/param × 3 tensors + fp32 master = ~12 B/param without
  master, ~14 B/param with master; see mod-101 chapter 1 for the
  bookkeeping).

Total: ~112 GB of state per replica if unsharded. Sharded across 8
GPUs, that is ~14 GB/GPU — comfortably in-HBM already, so *this
size does not need offload*. The offload story becomes real at
70B+.

At 70B on the same 8-GPU node:

- Params (BF16): 140 GB → 17.5 GB/GPU sharded — still fits.
- Grads: 140 GB → 17.5 GB/GPU sharded — still fits.
- Opt state: ~840 GB → 105 GB/GPU sharded — **does not fit**.

Now you have three real options:

**Option A — FSDP2 with CPU offload of parameters + optimizer.**
Optimizer state and parameter shards live in pinned host DRAM
(needs ~840 GB + ~140 GB ≈ 1 TB, right at the ceiling for a typical
node). Each block's forward triggers a H2D copy of its ~140 GB / L
shard, followed by the intra-node all-gather. Step time is
dominated by PCIe traffic — expect a substantial slowdown vs. an
in-HBM run, but the job runs.

**Option B — DeepSpeed ZeRO-3 + ZeRO-Offload (CPU).**
Same DRAM budget as A, but the optimizer step itself runs on CPU
via `DeepSpeedCPUAdam`. Rough equivalence in memory footprint;
step time can be better than A because the CPU Adam kernel is
optimized, and the GPU does not spin idle during the update.

**Option C — DeepSpeed ZeRO-Infinity (NVMe).**
Optimizer state lives on NVMe (any ~1 TB drive), so DRAM is no
longer the ceiling — you could scale to a 200B model on the same
node. Step time drops significantly because the NVMe-to-DRAM
transfer is capped at ~5–7 GB/s per drive vs. PCIe's 32 GB/s.
Memory-centric tiling helps hide it, but does not eliminate it.

Concretely, the ordering you should predict *before* running is:

```
step_time(HBM-only)  <  step_time(FSDP2+CPU) ≈ step_time(DS+ZeRO-Offload)  <  step_time(ZeRO-Infinity NVMe)
memory_ceiling(HBM-only)  <  memory_ceiling(FSDP2+CPU) ≈ memory_ceiling(DS+ZeRO-Offload)  <  memory_ceiling(ZeRO-Infinity NVMe)
```

Exercise 3 asks you to measure the ordering on your own hardware
and confirm the numbers land in that shape.

## Overlap: the throughput saver

The step-time gap between "in-HBM" and "CPU-offloaded" is not the
raw PCIe bandwidth divided by the state size — that would be
catastrophic. Both frameworks work hard to *overlap* the offload
traffic with compute:

- FSDP2 prefetches block `i+1`'s parameters (both from CPU and via
  all-gather) while block `i` is still executing forward.
- DeepSpeed's `stage3_prefetch_bucket_size` and
  `overlap_comm=true` do the same thing, with the CPU stage
  additionally overlapped with the ZeRO stage-3 all-gather.
- ZeRO-Infinity's `libaio` layer keeps up to `queue_depth`
  outstanding NVMe reads at all times, so the effective throughput
  approaches the aggregate device bandwidth rather than a single
  request at a time.

When you profile these systems, the gap between "PCIe/NVMe is
bandwidth-limited" and "PCIe/NVMe is *step-time limiting*" is
almost entirely a question of overlap quality. A common failure
mode is `NCCL_SOCKET_NTHREADS` or `pin_memory=False` breaking the
overlap silently.

## Decision procedure

Given a model size `M`, per-GPU HBM `H`, per-node HBM `H_node`, per-node
DRAM `D`, per-node NVMe `V`, and a shard count `N`:

1. Compute the sharded per-GPU state footprint (params + grads +
   opt-state, in BF16 for params/grads and FP32 for opt-state).
   Call it `s_per_gpu`.
2. If `s_per_gpu ≤ 0.7 * H`, do **not** offload. FSDP2 or ZeRO-3
   in HBM is fastest.
3. Else compute the sharded per-node state footprint `s_per_node`.
   If `s_per_node ≤ 0.7 * D` (leaving DRAM for activations,
   dataloader, and the OS), offload to CPU. Pick **FSDP2 CPU
   offload** if your loop is Pytorch-native and the CPU Adam
   compute is not the bottleneck; pick **DS + ZeRO-Offload** if
   you specifically need the CPU-Adam kernel or if the model
   integrates through the DeepSpeed engine already.
4. Else, offload to NVMe with **ZeRO-Infinity**. Provision NVMe
   accordingly (5+ GB/s per drive, RAID/multiple drives for
   aggregate bandwidth). Accept the step-time cost.

Two edge cases worth flagging:

- **MoE experts.** Expert weights inflate parameters dramatically
  but the *active* expert count per batch is small. In principle,
  ZeRO-Infinity's memory-centric tiling handles this well because
  only active experts need to be paged in; DeepSpeed-MoE explicitly
  targets this pattern. FSDP2 + MoE with CPU offload is possible
  but hand-assembled.
- **LoRA / adapter fine-tuning.** Optimizer state now sits only on
  the trainable adapter parameters, which are 0.1–1% of the base
  model. Offload is usually unnecessary; keep the base model
  frozen and on HBM (or on a single GPU), and skip the offload
  ladder entirely.

## What you cannot rescue with offload

Offload does not help with:

- **Activation memory.** Activations for a decoder block scale with
  `batch_size * seq_len * hidden_dim`. Neither FSDP2 CPU offload
  nor ZeRO-Infinity reduces activation memory. Use activation
  checkpointing (mod-107) for that.
- **NCCL collective latency.** If your all-gather already fires
  every layer, offloading to CPU does not remove the collective;
  it only reduces the parameter memory it operates on. The
  collective time scales with parameter size and network α+β, not
  with where the parameter was resting before the collective.
- **Cross-node bandwidth.** If your training run is bottlenecked
  by IB/RoCE, offload to slower tiers only makes things worse.

## Summary

- Offload moves state off HBM to DRAM (FSDP2 CPU offload,
  DeepSpeed ZeRO-Offload) or NVMe (ZeRO-Infinity). Each tier
  swap costs roughly a 30–100× bandwidth reduction and a
  proportional step-time penalty.
- FSDP2's CPU offload is a targeted `CPUOffloadPolicy` knob; it
  is native, ergonomic, and stops at the CPU tier.
- DeepSpeed's ladder goes further: ZeRO-Offload adds a CPU Adam
  kernel; ZeRO-Infinity adds an NVMe tier with `libaio`-driven
  async I/O and memory-centric tiling.
- Overlap of offload traffic with compute is the throughput saver.
  Turn on prefetch, use pinned memory, and profile to confirm the
  gaps overlap.
- Decision procedure: fit in HBM if you can; go to CPU only when
  DRAM per node is enough; go to NVMe only when DRAM per node is
  not.
