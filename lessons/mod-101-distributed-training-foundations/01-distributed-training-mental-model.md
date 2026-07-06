# The Distributed-Training Mental Model

Before we can trace a single DDP or FSDP step, we need a shared vocabulary for
what "distributed" actually means in a training job. This chapter builds the
mental model that the rest of the module hangs on: what a *world* is, what a
*process group* is, what *collectives* are, and how those primitives map to the
GPUs, NICs, and switches you actually have. Everything after this — DDP,
FSDP2, ZeRO-3, tensor / pipeline / sequence / expert parallel — is just a
recipe on top of these primitives.

## Motivation: why a single GPU is not enough

A single H100 has ~80 GB of HBM. A modern dense model in BF16 with the
Adam optimizer needs, at a bare minimum, roughly `16 * P` bytes of GPU memory
just to hold parameters, gradients, and optimizer state (2 bytes for BF16
parameter, 2 bytes for BF16 gradient, 4-byte FP32 master copy, and two 4-byte
FP32 Adam moments per parameter — 12 bytes if you skip the FP32 master, 16 if
you do not; ZeRO's paper walks through this bookkeeping in detail).
See the ZeRO paper's memory decomposition ("Memory Consumption",
Rajbhandari et al., 2020) for the definitive breakdown.

That means:

- A **7 B** parameter model with a stock optimizer already saturates a
  single 80 GB GPU before you have booked any memory for activations, the KV
  cache during training, or the data-loader buffer.
- A **70 B** or **175 B** parameter model cannot fit on a single GPU
  *at all*, no matter how carefully you tune. You have to shard.

So distribution is not a performance choice — for anything past a few
billion parameters it is a *correctness* choice. You either shard or the job
does not run.

## Ranks, world size, and process groups

The `torch.distributed` runtime is a small MPI-style abstraction. The pieces
you will use every day:

- **Process** — one Python interpreter, typically pinned to exactly one GPU.
- **Rank** — the global integer id of that process inside a *process group*.
  Rank `0` is conventionally the coordinator.
- **World size** — the total number of processes in the default process
  group. For a training job on 4 nodes × 8 GPUs, world size is 32.
- **Local rank** — the rank *inside* one node. You use this to pick which
  GPU on the box the process owns (`torch.cuda.set_device(local_rank)`).
- **Process group** — a subset of ranks that can exchange collectives with
  each other. The default group is "all of them"; you can carve
  sub-groups for tensor-parallel shards, pipeline stages, or expert routers.

The canonical launch pattern (from the PyTorch elastic docs) looks like:

```bash
torchrun \
  --nnodes=4 \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$HEAD_NODE:29500 \
  train.py
```

Inside `train.py`, every process calls `torch.distributed.init_process_group`
with a backend (`nccl` on GPU, `gloo` on CPU, `mpi` where an MPI runtime is
already present). NCCL is what you use for GPU-to-GPU collectives and is the
only backend this module cares about. (See the NCCL user guide, chapter
"Collective Operations", and the PyTorch `torch.distributed` overview.)

## Collectives: the primitives everything else is built out of

A **collective** is a communication operation involving every rank in a
process group. NCCL and PyTorch expose the same MPI-derived vocabulary. You
should be able to draw each of these on a whiteboard before moving on:

- **broadcast(src)** — one rank pushes a tensor; every rank receives it.
- **reduce(dst, op)** — every rank contributes a tensor; one rank ends up
  with the reduction (typically `SUM`).
- **all-reduce(op)** — every rank contributes a tensor; every rank ends up
  with the same reduced result. This is what DDP uses on gradients.
- **all-gather** — every rank contributes a shard; every rank ends up with
  the full concatenation of all shards. FSDP uses this to reconstitute a full
  parameter shard for a forward pass.
- **reduce-scatter** — every rank contributes a full tensor; every rank ends
  up with a *reduced shard* of it. FSDP uses this on the backward pass so
  each rank only keeps the piece of the gradient that corresponds to its
  parameter shard.
- **all-to-all** — every rank sends a distinct chunk to every other rank.
  This is how Mixture-of-Experts routing works (expert parallelism).
- **barrier** — synchronize all ranks; no data moves.

Two useful identities you will reuse constantly:

1. **all-reduce ≡ reduce-scatter + all-gather.** NCCL's ring all-reduce is
   literally implemented as one pass of reduce-scatter followed by one pass
   of all-gather. That is why FSDP's total wire volume is roughly the same
   as DDP's — it just splits the two halves so it can shard the parameters.
2. **broadcast + reduce ≡ all-reduce (via a root).** Slower, more
   convenient for logging metrics.

## The cluster shape

The "cluster shape" you design around has four dominant links, in decreasing
speed:

1. **Intra-GPU HBM** — hundreds of GB/s to a few TB/s. Basically free.
2. **Intra-node NVLink / NVSwitch** — hundreds of GB/s between GPUs on the
   same host. This is what makes tensor-parallel viable.
3. **Inter-node RDMA fabric** — InfiniBand (HDR ~200 Gb/s, NDR ~400 Gb/s per
   port) or RoCEv2. Roughly an order of magnitude slower than NVLink per
   GPU. This is what makes data-parallel and pipeline-parallel viable but
   makes cross-node tensor-parallel painful.
4. **Storage / dataloader fabric** — object store, parallel filesystem
   (Lustre, WEKA), NFS. Milliseconds per fetch, so must be hidden behind
   prefetch.

The mental picture that will save you in every future chapter: a modern
training cluster is a **hierarchy of comm domains** — NVLink islands stitched
together by IB/RoCE. Every parallelism strategy in this track is fundamentally
a choice of "which collective goes over which link", and the strategies that
look complicated (HSDP, 3D-parallel) look complicated because they map more
than one collective to more than one link on purpose.

## What we mean by "1B" through "175B" in this module

Because we will keep name-checking these throughout, the anchor points
this module uses:

- **~1 B parameters** — the DDP baseline you can fit on a single 80 GB GPU
  and scale to 2–8 GPUs. Exercise 4 targets this scale.
- **~7 B** — right at the edge of a single GPU. FSDP / ZeRO-3 is the first
  strategy that becomes essentially mandatory.
- **~70 B** — needs tensor parallel (typically TP = 8 inside a node) plus
  FSDP or pipeline parallel across nodes.
- **~175 B and up** — full 3D-parallel: DP × TP × PP, sometimes with
  sequence-parallel and expert-parallel layered on.

## Summary

- Distributed training is about mapping *collectives* onto a *hierarchy of
  network links*. Every strategy is a variation on that theme.
- A process group + a backend (NCCL for GPUs) is the only runtime
  abstraction you need to reason about launch.
- The six collectives — broadcast, reduce, all-reduce, all-gather,
  reduce-scatter, all-to-all — cover essentially all of DDP, FSDP2,
  Megatron, DeepSpeed, and JAX's `pjit`. The rest of this module opens up
  each of these one at a time.
