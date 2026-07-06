# mod-102 — Training Frameworks Deep Dive: PyTorch FSDP2 / DeepSpeed / Megatron-LM / NeMo / JAX

**Estimated effort:** 22 hours

mod-101 taught the algorithms and the collective-cost model. This module
teaches the *five stacks* that actually implement those algorithms in
production and the *judgment* required to pick between them. You will
run the same ~3B-parameter training job through PyTorch FSDP2,
DeepSpeed ZeRO-3, and Megatron-LM 3D-parallel; port a Megatron-LM job
into NeMo; run a JAX/Flax + `jit` (formerly `pjit`) + `shard_map`
version on the same hardware; and end with a written decision doc that
picks a stack for a specific team and training regime.

The through-line is that these frameworks are not interchangeable
implementations of the same thing. They embed different assumptions
about who writes the code (research scientist vs. platform engineer),
what "the model" is (a `nn.Module` you own vs. a pre-built recipe),
where sharding is *decided* (imperative Python vs. a compiler), and how
you observe a failing run. The tour is deliberately hands-on: by the
end you should be able to walk into a training-platform team and
defend a stack choice from evidence you gathered yourself.

## Learning objectives

- Run the same 3B-model training job in three stacks: PyTorch FSDP2
  (torchtitan-style), DeepSpeed ZeRO-3, and Megatron-LM 3D-parallel.
- Explain FSDP2's per-parameter sharding and DTensor integration vs.
  FSDP1's flat-parameter sharding.
- Compare DeepSpeed ZeRO-Infinity + ZeRO-Offload memory / throughput
  trade-offs against FSDP2 + CPU offload.
- Port a Megatron-LM job into NeMo (Megatron-Core + PyTorch Lightning),
  reasoning about the additional abstractions.
- Run a JAX/Flax + `jit(in_shardings=...)` / `shard_map` / GSPMD
  training job on 8 GPUs and compare mental models against PyTorch.
- Author a decision doc: "which stack should team X use for training
  regime Y" with evidence.

## Chapters

1. [The Training-Framework Landscape](01-training-framework-landscape.md) —
   why five stacks exist, what each was designed for, and the common
   substrate (`torch.distributed`, NCCL, XLA) that they all sit on.
2. [FSDP2, DTensor, and Per-Parameter Sharding](02-fsdp2-per-parameter-and-dtensor.md) —
   the `fully_shard` API, how per-parameter DTensor sharding replaces
   FSDP1's `FlatParameter`, and why that unlocks composition with
   tensor-parallel through a 2-D DeviceMesh.
3. [DeepSpeed and the ZeRO Ladder](03-deepspeed-and-the-zero-ladder.md) —
   the DeepSpeed engine, the JSON config surface, and the
   ZeRO-1 → ZeRO-2 → ZeRO-3 → ZeRO-Offload → ZeRO-Infinity ladder as
   a coherent memory-ladder design.
4. [Offload Head-to-Head: FSDP2 CPU Offload vs. ZeRO-Infinity](04-offload-fsdp2-vs-zero-infinity.md) —
   what each system offloads, to where, at what bandwidth cost, and how
   to choose between them for a fixed model + cluster.
5. [Megatron-LM and 3D Parallelism in Practice](05-megatron-lm-3d-parallel.md) —
   `ColumnParallelLinear` / `RowParallelLinear` /
   `VocabParallelEmbedding`, the `pretrain_gpt.py` recipe, TP × PP × DP
   composition, interleaved-1F1B, and Megatron-Core as the underlying
   library.
6. [NeMo: Megatron-Core Wrapped in PyTorch Lightning](06-nemo-megatron-core-lightning.md) —
   NeMo's abstraction ladder (Trainer / Strategy / Model / Data),
   how it maps onto Megatron-Core, YAML/Hydra configuration, and when
   the extra abstractions help vs. hide.
7. [JAX, Flax, and the GSPMD Sharding Model](07-jax-flax-and-gspmd.md) —
   `jax.Array`, `Mesh`, `NamedSharding`, `jax.jit(in_shardings=...)`,
   `shard_map`, and how the XLA compiler produces the collectives
   PyTorch users insert by hand. Contrast in mental model, debugging,
   and observability.
8. [Choosing a Stack: The Decision Doc](08-choosing-a-stack.md) —
   the rubric behind exercise 6-style decision docs: team, regime,
   ecosystem, observability, cost, exit ramps. This chapter is the
   guide for the write-up that ends this module.

## Exercises

- [exercise-01 — Three-stacks same-model bake-off](exercises/exercise-01-three-stacks-same-model-bake-off.md) (6 h)
- [exercise-02 — FSDP2 vs. FSDP1 per-parameter sharding](exercises/exercise-02-fsdp2-vs-fsdp1-per-param-sharding.md) (3 h)
- [exercise-03 — ZeRO-Infinity vs. FSDP2 offload](exercises/exercise-03-zero-infinity-vs-fsdp2-offload.md) (3 h)
- [exercise-04 — Megatron-LM → NeMo port](exercises/exercise-04-megatron-to-nemo-port.md) (4 h)
- [exercise-05 — JAX `jit` + `shard_map` training on 8 GPUs](exercises/exercise-05-jax-pjit-shard-map-training.md) (4 h)

## Labs and quizzes

- `labs/` — a long-form lab that reuses the bake-off's Megatron
  configuration to explore interleaved-1F1B pipeline scheduling will
  land here on a subsequent authoring cycle.
- `quizzes/` — one knowledge check will land here on a subsequent
  authoring cycle.

## Resources

- [resources.md](resources.md) — primary papers, framework
  documentation, and reference codebases (torchtitan, DeepSpeedExamples,
  Megatron-LM, NeMo, MaxText) that the chapters and exercises are
  grounded in.

## How the module fits together

Chapter 1 is a map. Chapters 2 and 3 do the two dominant PyTorch stacks
in depth (FSDP2 and DeepSpeed) so the ZeRO-3 vs. FSDP2 comparison in
chapter 4 has real substance. Chapter 5 introduces Megatron-LM's
3D-parallel implementation, which chapter 6 then re-wraps as NeMo so
you can see the same collective pattern through two very different
abstraction layers. Chapter 7 pulls you out of PyTorch entirely: JAX
lets a compiler infer the sharding you have been writing by hand,
which changes what "debug the shape" and "measure the collective"
mean. Chapter 8 is the write-up rubric.

Exercise 1 is the anchor — running the same job in three stacks and
measuring what actually happens is what turns the chapters from
narration into muscle memory. Exercises 2 and 3 zoom in on specific
comparisons (FSDP1 vs. FSDP2 sharding; FSDP2 offload vs. ZeRO-Infinity).
Exercises 4 and 5 close the tour with NeMo and JAX. Do them in order.

## What this module deliberately does not cover

- **Kernel-level performance authoring** (FlashAttention, custom CUDA
  kernels, Triton) — owned by
  [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning).
- **Cluster orchestration (SLURM / Kueue / Volcano / KubeRay)** —
  owned by mod-104.
- **NCCL tuning at fabric depth (IB / RoCE / GPUDirect / PXN)** —
  owned by mod-105.
- **Distributed checkpointing and elastic training** — owned by
  mod-106. This module treats a checkpoint as an opaque blob.
- **MFU engineering, activation checkpointing, `torch.compile`
  micro-benchmarking** — owned by mod-107. Chapters here measure
  step time and memory, but do not tune them.
- **Training observability, TensorBoard/W&B pipelines, W&B alerting**
  — owned by mod-108.
