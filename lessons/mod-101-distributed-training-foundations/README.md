# mod-101 — Distributed-Training Foundations: DDP, Parallelism Strategies, and the Communication-Cost Model

**Estimated effort:** 18 hours

This module is the foundation of the Training Pipeline Engineer track.
It teaches you to trace, design, and cost the distributed-training
strategies every subsequent module builds on: DDP, FSDP2 / ZeRO-3, and
the DP / TP / PP / SP / EP / HSDP design space, grounded in the NCCL
collective cost model.

By the end of the module you should be able to (a) whiteboard the
forward + backward pass of any of these strategies, (b) pick a strategy
for a given `(model, cluster)` pair and defend it in an RFC, (c) predict
the comm-to-compute ratio of that strategy from first principles, and
(d) read the Llama 3, BLOOM, and OPT-175B papers as design records
rather than as opaque reports.

## Learning objectives

- Trace forward and backward through DDP (all-reduce), FSDP
  (reduce-scatter and all-gather), and a ZeRO-3 partitioned step.
- Design a parallelism strategy (DP / TP / PP / SP / EP / HSDP /
  3D-parallel) from first principles for a target model + cluster shape.
- Reason about NCCL collective costs (bandwidth × latency × topology)
  and derive the communication-to-compute ratio for each strategy.
- Run a 1B-parameter DDP training job on 2–8 GPUs, then port it to
  FSDP2 and to Megatron-style tensor parallel.
- Read the Llama 3, BLOOM, and OPT-175B reports well enough to reason
  about *why* each shop chose its parallelism strategy.

## Chapters

1. [The Distributed-Training Mental Model](01-distributed-training-mental-model.md) —
   ranks, world size, process groups, and the collectives (broadcast,
   reduce, all-reduce, all-gather, reduce-scatter, all-to-all) that
   every strategy is built out of. The cluster shape as a hierarchy of
   comm domains.
2. [DDP and the All-Reduce Backward Pass](02-ddp-and-the-all-reduce-backward.md) —
   how `DistributedDataParallel` overlaps a bucketed all-reduce with the
   tail of `backward()`, the `2(N-1)/N · S` per-GPU wire volume, and
   the correctness invariants (identical iteration order, seeded RNG,
   `DistributedSampler`).
3. [FSDP2 and the ZeRO-3 Partitioned Step](03-fsdp2-and-zero-partitioned-step.md) —
   the all-gather + reduce-scatter dance layer-by-layer, why the
   optimizer step is local, and how per-parameter DTensor sharding
   composes with tensor parallel via a 2-D DeviceMesh.
4. [The Parallelism Strategy Design Space](04-parallelism-strategy-design-space.md) —
   DP, TP, PP, SP, EP, HSDP as six axes; a top-down decision procedure
   for picking a strategy; a preview of what Llama 3 / BLOOM / OPT-175B
   chose.
5. [The NCCL Collective Cost Model](05-nccl-collective-cost-model.md) —
   the α + β cost model, algorithm selection (ring vs. tree), effective
   bandwidth vs. nominal bandwidth, and how to compute a
   comm-to-compute ratio you can defend.
6. [From Code to Cluster: Running a 1B Job, then Porting to FSDP2 and
   Megatron-TP](06-from-code-to-cluster.md) — the DDP → FSDP2 → TP
   ladder in code, with the memory and step-time observations you need
   at each transition. This chapter is the guide for exercises 4 and 5.
7. [Case Studies: Reading Llama 3, BLOOM, and OPT-175B](07-case-studies-llama3-bloom-opt.md) —
   a reading protocol for extracting strategy, forcing constraints, and
   documented failure modes from each paper.

## Exercises

- [exercise-01 — DDP forward + backward trace](exercises/exercise-01-ddp-forward-backward-trace.md) (3 h)
- [exercise-02 — Parallelism-strategy selection rubric](exercises/exercise-02-parallelism-strategy-selection-rubric.md) (4 h)
- [exercise-03 — NCCL collective cost model](exercises/exercise-03-nccl-collective-cost-model.md) (3 h)
- [exercise-04 — DDP → FSDP2 port](exercises/exercise-04-ddp-to-fsdp2-port.md) (4 h)
- [exercise-05 — Megatron-style tensor-parallel port](exercises/exercise-05-megatron-tensor-parallel-port.md) (4 h)

## Labs and quizzes

- `labs/` — a `lab-01` Llama-3 parallelism teardown lands here on the
  next autonomous cycle.
- `quizzes/` — one knowledge check lands here on the next autonomous
  cycle.

## Resources

- [resources.md](resources.md) — primary papers, NCCL / PyTorch / Megatron
  / DeepSpeed docs, and a recommended reading order.

## How the module fits together

Chapters 1–3 give you the vocabulary and the two canonical algorithms
(DDP and FSDP2/ZeRO-3). Chapter 4 opens the full parallelism design
space you will actually use in production. Chapter 5 makes the choices
in chapter 4 quantitative. Chapter 6 makes them executable. Chapter 7
grounds them in what real large-scale training runs have actually
looked like. Exercises 1, 3, 4, and 5 are hands-on; exercise 2 is a
design-doc exercise. Do them in order — later exercises reuse the
codebase from earlier ones.

## What this module deliberately does not cover

- **Kernel-level performance authoring** (FlashAttention v2/v3 internals,
  CUDA kernels) — owned by
  [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning).
- **Cluster orchestration (SLURM / Kueue / Volcano / KubeRay)** — owned
  by mod-104.
- **NCCL tuning at fabric depth (IB / RoCE / GPUDirect / PXN)** — owned
  by mod-105. Chapter 5 name-checks these; mod-105 goes deep.
- **Distributed checkpointing and elastic training** — owned by mod-106.
- **MFU engineering, activation checkpointing, `torch.compile`** — owned
  by mod-107. Chapter 5 sets up the "why"; mod-107 does the "how".
