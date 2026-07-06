# DeepSpeed and the ZeRO Ladder

DeepSpeed is Microsoft's training framework and the reference
implementation of the ZeRO family of memory optimizations
(Rajbhandari et al., 2020, "ZeRO"). It is the other major
PyTorch-family stack for sharded data-parallel training, and it is
still the default choice when you need extreme offload
(NVMe-tier ZeRO-Infinity) or when you are wrapping a stock
HuggingFace `transformers` model without editing its layers.

This chapter walks the DeepSpeed engine, its JSON configuration
surface, and the ZeRO ladder — ZeRO-1 → ZeRO-2 → ZeRO-3 →
ZeRO-Offload → ZeRO-Infinity — as a single coherent
memory-reduction design. Chapter 4 will put ZeRO-Infinity head-to-head
with FSDP2 CPU offload; this chapter is about knowing what each
level buys you.

## The DeepSpeed engine

Whereas FSDP2 asks you to keep your `nn.Module` and just call
`fully_shard(...)`, DeepSpeed asks you to hand your model, optimizer,
and dataloader (optional) to `deepspeed.initialize` and to configure
almost everything else in JSON:

```python
import deepspeed

model, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=[p for p in model.parameters() if p.requires_grad],
    config="ds_config.json",
)

for batch in loader:
    loss = model(batch).loss
    model.backward(loss)   # not loss.backward()
    model.step()           # not optimizer.step()
```

Two subtle differences from the PyTorch imperative style:

- `model.backward(loss)` and `model.step()` replace
  `loss.backward()` and `optimizer.step()`. The engine now owns
  the accumulation counter, the ZeRO partitioner, the gradient
  reduction, and the offload marshalling. You give up direct control
  in exchange for a stack you don't have to re-implement.
- Gradient accumulation, gradient clipping, learning-rate
  scheduling, mixed precision, and loss scaling all live in the
  JSON config. That is what "config-driven" means concretely.

Launch is `deepspeed --num_gpus=N train.py --deepspeed
--deepspeed_config ds_config.json`, or `torchrun`-launched with
DeepSpeed reading the `LOCAL_RANK` / `WORLD_SIZE` environment
variables that `torchrun` sets. Either way it initializes
`torch.distributed` and NCCL under the hood.

## The JSON config surface

A minimal ZeRO-3 config for a mixed-precision AdamW pretraining
job looks like:

```json
{
  "train_batch_size": 512,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 1.0,

  "bf16": { "enabled": true },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-4, "betas": [0.9, 0.95],
      "eps": 1e-8, "weight_decay": 0.1
    }
  },

  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0, "warmup_max_lr": 3e-4,
      "warmup_num_steps": 2000, "total_num_steps": 200000
    }
  },

  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true,
    "contiguous_gradients": true
  }
}
```

Three groups of knobs to know:

- **Global batch bookkeeping.** `train_batch_size = micro_batch *
  world_size * gradient_accumulation_steps`. DeepSpeed will refuse
  to start if these don't agree, which is a feature — it removes
  a class of "I thought my batch was 512 but it was actually 128"
  bugs.
- **Mixed precision.** `bf16` or `fp16` blocks pick the format.
  `fp16` needs `loss_scale` / `initial_scale_power`; `bf16`
  generally does not.
- **`zero_optimization`.** This is the ladder. `stage: 1|2|3`
  matches the ZeRO paper. The `stage3_*` knobs control how
  aggressively ZeRO-3 preserves live parameters (see below).

## The ZeRO ladder

The ZeRO paper decomposes per-parameter memory into three tiers:
optimizer state (`O`), gradients (`G`), and parameters (`P`).
Each stage shards more of them.

### ZeRO-1 — shard optimizer state

Per-GPU memory:

- Params: `1 * P`
- Grads: `1 * G`
- Opt state: `1/N * O`

Communication: identical to DDP (all-reduce of gradients), plus a
final broadcast/all-gather of parameter updates from the rank that
owns each shard. Useful when *only* optimizer state pushes you over
the memory ceiling (common for Adam on a small model — the two
FP32 moments + FP32 master copy dominate). Cheap to try. Most
LLM-scale training runs skip past it to ZeRO-2 or ZeRO-3.

### ZeRO-2 — shard optimizer state + gradients

Per-GPU memory:

- Params: `1 * P`
- Grads: `1/N * G`
- Opt state: `1/N * O`

Communication: DDP's all-reduce becomes a `reduce-scatter` (each
rank ends up with its shard of the reduced gradient), and no
additional collective is needed on the parameter side because
parameters are still replicated. Cheaper than ZeRO-3 in wire
volume, but you still pay the full-parameter memory footprint.
Good middle ground when parameters fit comfortably in HBM but
gradients + optimizer state are the pain point.

### ZeRO-3 — shard everything

Per-GPU memory:

- Params: `1/N * P`
- Grads: `1/N * G`
- Opt state: `1/N * O`

Communication: the FSDP2 pattern from mod-101 chapter 3 — one
`all-gather` on forward, one `all-gather` + `reduce-scatter` on
backward, per sharded unit. Total per-GPU wire volume of
`3 * (N-1)/N * P`, 1.5× DDP's `2 * (N-1)/N * P`. This is the
canonical "we cannot fit the model on one GPU" tier.

The `stage3_*` knobs are worth understanding because they are the
main perf handles once you've picked ZeRO-3:

- **`stage3_prefetch_bucket_size`** — how large a lookahead of
  parameter shards to `all-gather` before the compute for the
  current layer finishes. Bigger buckets hide latency better but
  cost transient memory.
- **`stage3_param_persistence_threshold`** — parameters smaller
  than this stay resident on every rank instead of being
  partitioned. Useful for embeddings and layer norms where the
  bookkeeping cost dwarfs the savings.
- **`stage3_max_live_parameters`** — cap on how many bytes of
  fully-materialized parameters can be alive simultaneously.
  Analogous to FSDP2's `reshard_after_forward`.
- **`stage3_gather_16bit_weights_on_model_save`** — save-time
  behavior only; whether to reconstitute the full model on rank 0
  when writing a checkpoint. Turn on for HuggingFace-compatible
  final weights; off for sharded ZeRO-native checkpoints (which
  DeepSpeed's `zero_to_fp32.py` script can later gather).

## ZeRO-Offload — CPU offload of optimizer state

ZeRO-Offload (Ren et al., 2021, USENIX ATC) adds a second axis to
the ladder: instead of only sharding across GPUs, offload the
sharded optimizer state to *host DRAM* and run the Adam update on
CPU with a CUDA-aware, SSE/AVX-optimized kernel (DeepSpeed's
`DeepSpeedCPUAdam`).

Config surface:

```json
"zero_optimization": {
  "stage": 2,
  "offload_optimizer": { "device": "cpu", "pin_memory": true }
}
```

Or, layered on top of ZeRO-3, adding parameter offload:

```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": { "device": "cpu", "pin_memory": true },
  "offload_param":     { "device": "cpu", "pin_memory": true }
}
```

Two things matter for performance:

- **PCIe becomes the bottleneck for optimizer updates.** The
  parameter shards travel host→device before forward and
  device→host after gradient reduction. Pinned memory + the
  `DeepSpeedCPUAdam` kernel make this survivable.
- **The optimizer step runs on CPU.** Fine for AdamW at moderate
  scale; a real cost for optimizers with per-parameter compute
  (Lion, Adafactor moment updates) or for very fast GPUs whose
  step-time is dominated by the CPU update.

ZeRO-Offload is the reason DeepSpeed can train a 13B-parameter
model on a single 24 GB gaming GPU. It is also why exercise 3 uses
DeepSpeed as one of its three configurations.

## ZeRO-Infinity — NVMe tier and memory-centric tiling

ZeRO-Infinity (Rajbhandari et al., 2021, SC21) extends the ladder
one more rung: offload to *NVMe SSD*, not just host DRAM. Combined
with two other techniques:

- **Memory-centric tiling.** Individual layer parameters can be
  tiled and prefetched piecewise, so a single layer's parameters
  do not need to fit in HBM at any moment.
- **Bandwidth-centric partitioning.** For very large models, a
  layer is partitioned across ranks and streamed from NVMe in a
  way that keeps the aggregate NVMe→HBM path saturated during
  compute.

Config surface:

```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": {
    "device": "nvme",
    "nvme_path": "/mnt/nvme/deepspeed_offload",
    "pin_memory": true, "buffer_count": 4, "fast_init": false
  },
  "offload_param": {
    "device": "nvme",
    "nvme_path": "/mnt/nvme/deepspeed_offload",
    "pin_memory": true, "buffer_count": 5, "buffer_size": 1e8, "max_in_cpu": 1e9
  },
  "aio": {
    "block_size": 1048576, "queue_depth": 8,
    "thread_count": 1, "single_submit": false, "overlap_events": true
  }
}
```

The `aio` block configures DeepSpeed's async I/O layer (`libaio`
under the hood). Real NVMe throughput requires it: naive `read()`/
`write()` cannot keep a modern NVMe device saturated.

Two guarantees worth noting:

1. ZeRO-Infinity moves the memory ceiling from *HBM* to *NVMe
   capacity per node*, in effect. A 1 TB NVMe drive per node
   comfortably holds a 100B+ parameter model's optimizer state.
2. Throughput drops. The gap between HBM (~2 TB/s) and NVMe
   (~5–7 GB/s per drive, tens of GB/s aggregated across a RAID
   or multiple drives) is 2–3 orders of magnitude. You are trading
   *the ability to run at all* for step-time.

Chapter 4 puts numbers on this trade-off and compares it against
FSDP2's CPU-offload path.

## What DeepSpeed does that FSDP2 does not

Several DeepSpeed features have no direct FSDP2 equivalent (as of
current PyTorch releases):

- **NVMe offload.** FSDP2 supports CPU offload but does not, at
  time of writing, offer NVMe offload of parameter/optimizer state.
- **Cross-optimizer offload primitives.** DeepSpeed's
  `DeepSpeedCPUAdam` is a targeted CPU kernel; PyTorch native
  optimizers do not include such kernels.
- **1-bit / gradient compression.** DeepSpeed ships 1-bit Adam,
  1-bit Lamb, and 0/1 Adam variants that trade compression for
  wire volume on slow interconnects.
- **Pipeline parallelism.** DeepSpeed has a native pipeline module;
  FSDP2 defers pipeline scheduling to a separate PyTorch API.
- **MoE.** DeepSpeed-MoE (Rajbhandari et al., 2022) ships an
  expert-parallel implementation. FSDP2 supports EP via mesh
  composition but you assemble it yourself.
- **Curriculum learning, ZeRO-Chat data pipeline.** DeepSpeed
  bundles opinionated data-pipeline features that PyTorch treats as
  out-of-scope.

## What DeepSpeed does that hurts

The trade-off cuts both ways:

- **Config sprawl.** A production DeepSpeed config commonly has
  30+ keys. It is easy to have "the config was wrong" become a
  diagnosis with no clear root cause.
- **`model.backward(loss)` shape.** Because DeepSpeed intercepts
  the backward pass, many PyTorch idioms (gradient hooks, custom
  autograd functions that touch grads, `torch.autograd.grad`) need
  adaptation.
- **DeepSpeed's optimizer wrapping** replaces your `torch.optim`
  optimizer with a wrapped one. If you build tooling around
  optimizer state dicts, that tooling must speak the wrapped form.
- **Version churn.** DeepSpeed's API surface has moved faster than
  PyTorch's `fully_shard`; upgrading DeepSpeed for a new feature
  often requires a config migration.

None of this is a reason not to use DeepSpeed — it is the reason to
weigh it against FSDP2 case-by-case (chapter 8).

## Summary

- DeepSpeed's engine is entered via `deepspeed.initialize`; the
  loop calls `model.backward(loss)` and `model.step()` rather than
  the PyTorch imperative forms. Most tuning is done in a JSON
  config file.
- ZeRO-1 / 2 / 3 form the memory-sharding ladder: shard optimizer
  state, then optimizer + grads, then all three. Wire volume grows
  from DDP's `2·(N-1)/N·P` to ZeRO-3's `3·(N-1)/N·P`.
- ZeRO-Offload adds a CPU tier (optimizer state, and optionally
  parameters), using a CPU Adam kernel. It enables 10B+ training
  on modest GPUs.
- ZeRO-Infinity adds an NVMe tier, memory-centric tiling, and
  bandwidth-centric partitioning. The memory ceiling moves from
  HBM to per-node NVMe capacity; throughput drops accordingly.
- DeepSpeed is the correct choice today when you need NVMe
  offload, 1-bit optimizers, or an off-the-shelf pipeline schedule
  — capabilities FSDP2 does not (yet) match.
