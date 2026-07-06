# exercise-03: ZeRO-Infinity vs. FSDP2 Offload

**Estimated effort:** 3 hours

## Objective

Take a model sized to *just barely not fit in HBM* on your
cluster and run it in three offload configurations — FSDP2 with
CPU offload, DeepSpeed ZeRO-3 with ZeRO-Offload (CPU), and
DeepSpeed ZeRO-Infinity (NVMe) — measuring peak HBM, peak host
DRAM, NVMe I/O rate, and tokens/sec/GPU for each. The deliverable
is a comparison table plus a short analysis that turns chapter
4's decision procedure into a claim you can defend from
observations: "on our hardware, the FSDP2-CPU / DeepSpeed-CPU /
ZeRO-Infinity trade-off looks like this."

## Prerequisites

- Chapters 3 and 4 of this module.
- Exercise 1 completed (you already have a working 3B FSDP2 and a
  working DeepSpeed ZeRO-3 configuration).
- A multi-GPU environment with:
  - Enough HBM per GPU that the 3B model would fit without
    offload (this is your baseline).
  - At least ~500 GB of host DRAM total across the node.
  - Fast local NVMe with ≥ 500 GB free, mounted at a known path
    (e.g. `/mnt/nvme/deepspeed`). If you do not have local NVMe,
    the ZeRO-Infinity leg is optional but the write-up must note
    that constraint.
- `iostat` or `iotop` for observing NVMe traffic; `nvidia-smi
  dmon` for HBM occupancy; the standard PyTorch profiler.

## Problem statement

Your team has been told they may need to train a 30B model on the
current 8-GPU node before the next-generation cluster is
provisioned. The 30B model will not fit in HBM even sharded across
8 GPUs. Before the request lands, you want to build the muscle for
the "when do we offload, and to what?" question by pushing your
existing 3B pilot into offload territory and comparing what
happens.

For that reason, you will *artificially* inflate the state
footprint of the 3B model (via longer sequence, larger optimizer,
or by simulating a bigger model with additional replicated
layers) until the in-HBM configuration is uncomfortable. Then you
run the offload comparison.

## Requirements

### Set-up: force the pressure

Pick one of the following ways to push the model into
memory-pressure territory (do not do all of them; pick what your
hardware makes convenient):

- **Increase sequence length** from 2048 to 8192 or 16384. This
  inflates activation memory dramatically. Combine with activation
  checkpointing off (temporarily) to make the pressure visible.
- **Use full-precision Adam** (`--fp32-master-weights` in Megatron
  or the equivalent in DeepSpeed/FSDP2). Optimizer state is now
  ~24 bytes/param instead of ~14, pushing you into offload.
- **Scale the model** from 3B to 7B or 13B by adding decoder
  layers. Simplest way to prove the point at the cost of a longer
  step.
- **Halve the GPU count** you use (run on 4 GPUs instead of 8) to
  double the per-GPU state footprint.

Confirm before running the offload legs that the *un-offloaded*
FSDP2 configuration OOMs or is uncomfortably close to it. If it
still fits comfortably, push harder.

### The three configurations

Run the same forced-pressure workload in three configurations. In
each, run for at least 50 steps of a synthetic task.

#### Configuration A — FSDP2 + CPU offload

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy

policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                              reduce_dtype=torch.float32)
offload = CPUOffloadPolicy(pin_memory=True)

for block in model.blocks:
    fully_shard(block, mesh=mesh, mp_policy=policy, offload_policy=offload)
fully_shard(model, mesh=mesh, mp_policy=policy, offload_policy=offload)
```

Use PyTorch's stock `AdamW` for the optimizer.

#### Configuration B — DeepSpeed ZeRO-3 + ZeRO-Offload (CPU)

DeepSpeed config:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param":     { "device": "cpu", "pin_memory": true },
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "bf16": { "enabled": true },
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.1 }
  }
}
```

DeepSpeed will select its CPU Adam kernel (`DeepSpeedCPUAdam`)
because `offload_optimizer.device == "cpu"`; verify by looking at
the engine's startup log.

#### Configuration C — DeepSpeed ZeRO-Infinity (NVMe)

DeepSpeed config:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/mnt/nvme/deepspeed",
      "pin_memory": true, "buffer_count": 4
    },
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/mnt/nvme/deepspeed",
      "pin_memory": true, "buffer_count": 5,
      "buffer_size": 1e8, "max_in_cpu": 1e9
    },
    "aio": {
      "block_size": 1048576, "queue_depth": 8,
      "thread_count": 1, "single_submit": false, "overlap_events": true
    }
  },
  "bf16": { "enabled": true },
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.1 }
  }
}
```

Confirm the NVMe path is writable and has enough free space.
Confirm `libaio` support at startup — DeepSpeed will complain
loudly if the runtime is missing.

### Measurements per configuration

For each of the three runs, capture:

- **Peak HBM per GPU** — `torch.cuda.max_memory_allocated`.
- **Peak host DRAM** — sample `psutil.virtual_memory().used` at
  step boundaries; record the maximum.
- **NVMe I/O rate** (config C only) — `iostat -x 5` averaged
  over steady-state, or `nvidia-smi` counters if your I/O path
  reports through it. Both read and write MB/s.
- **PCIe traffic** — approximate from `nvidia-smi dmon -s p`
  (PCIe RX/TX in MB/s) averaged over steady state.
- **Tokens/sec/GPU** — median of steady-state steps.
- **Step time** — mean and P99 of steady-state steps.
- **Loss curve first 50 steps** — all three should track. If any
  diverges, that configuration has a bug.

### The comparison table

Fill in and include the following table in your report:

| Metric                       | FSDP2 + CPU offload | DS ZeRO-3 + ZeRO-Offload | DS ZeRO-Infinity (NVMe) |
|------------------------------|:-------------------:|:------------------------:|:-----------------------:|
| Peak HBM / GPU               |                     |                          |                         |
| Peak host DRAM (node total)  |                     |                          |                         |
| NVMe read rate (MB/s)        |     n/a             |         n/a              |                         |
| NVMe write rate (MB/s)       |     n/a             |         n/a              |                         |
| PCIe traffic per GPU (GB/s)  |                     |                          |                         |
| Tokens/sec/GPU               |                     |                          |                         |
| Step time (median)           |                     |                          |                         |
| Step time (P99)              |                     |                          |                         |
| Loss @ step 50               |                     |                          |                         |

### The write-up

Deliver a 400–700-word `offload-comparison.md` covering:

1. **The forced-pressure setup.** Which lever you pulled (sequence
   length / precision / model size / GPU count) and what the
   pre-offload memory picture looked like.
2. **The results table.** With actual numbers.
3. **Chapter-4 prediction vs. observation.** Chapter 4 predicts the
   ordering:
   `step_time(FSDP2+CPU) ≈ step_time(DS+ZeRO-Offload) <
   step_time(ZeRO-Infinity)`. Did that match your observation? If
   not, why not?
4. **PCIe or NVMe bottleneck evidence.** Cite one specific
   configuration where a link saturates and explain how you saw it
   in the numbers.
5. **When would you actually pick each?** For your specific team
   and workload, articulate the regime for each configuration.

## Acceptance criteria

- Three working configurations that reach step 50 without OOM.
- Loss curves match across configurations to within numerical
  tolerance for the first 50 steps.
- Peak HBM, peak DRAM, and either PCIe or NVMe traffic are
  measured and recorded — not left blank.
- The write-up correctly identifies at least one bandwidth-limited
  transfer in one of the configurations and connects it to
  chapter 4's PCIe / NVMe bandwidth discussion.
- Cites either the ZeRO-Offload paper (Ren et al., 2021), the
  ZeRO-Infinity paper (Rajbhandari et al., 2021), or the current
  DeepSpeed docs on offload.

## Starter guidance

- **Start with FSDP2 + CPU offload.** It is the least new
  machinery. If it does not OOM, you have not applied enough
  pressure and configuration B/C will be even less interesting.
- **Turn on `pin_memory=True` everywhere.** Without pinned memory,
  the CPU offload path is much slower and the comparison numbers
  become noise.
- **DeepSpeed CPU-Adam has a first-run compilation step.** The
  first run may spend a couple of minutes JITing the CPU kernel.
  Do not count that in the throughput; profile steady state.
- **ZeRO-Infinity needs `libaio-dev`.** Missing it manifests as a
  fallback to `pread`/`pwrite` and NVMe throughput collapses. Check
  DeepSpeed's startup log.
- **NVMe path capacity.** Optimizer state alone for a 7B model at
  FP32 precision is ~80 GB. Provision the NVMe path with at least
  2–3× headroom.
- **Do not compare across configurations that OOM.** A run that
  died at step 3 is not a data point.

## Stretch goals

- **Turn off overlap** (`overlap_comm: false` in DeepSpeed;
  `set_reshard_after_forward(True)` and no prefetch tuning in
  FSDP2) and observe how much throughput you lose. This isolates
  the "prefetch buys back throughput" claim from chapter 4.
- **Add a fourth leg with FSDP2 + CPU offload + a CPU AdamW**
  (running the optimizer on CPU DTensors directly). This is
  probably slower than DeepSpeed's `DeepSpeedCPUAdam`. Confirm
  that and characterize the gap.
- **Compare ZeRO-Infinity NVMe throughput to the raw NVMe
  hardware bandwidth** (`fio` benchmark on the same drive). If
  ZeRO-Infinity is far below the drive's ceiling, examine the
  `aio` config knobs (`queue_depth`, `thread_count`) and see
  whether they close the gap.
