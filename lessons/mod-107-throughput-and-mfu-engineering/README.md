# mod-107 — Throughput and MFU Engineering: Kernels, Mixed Precision, and Communication Overlap

**Estimated effort:** 20 hours

mod-101 through mod-106 gave you correctness and reliability: the
model converges, the checkpoint survives, the fabric does not drop.
This module is about the *efficiency* of the training step — the
fraction of the accelerator's theoretical peak you actually turn
into training tokens. That number is **Model FLOPs Utilization
(MFU)**, and the eight chapters here are the systematic pass that
takes a naive baseline (typically 20–30% MFU on a modern LLM at
Hopper scale) up toward the 40–55% band that a well-tuned dense LLM
run reaches with BF16 + FlashAttention, and further with FP8 on
matmul-heavy workloads.

The through-line is a five-bucket gap-to-peak budget introduced in
chapter 1: kernel/dtype, memory-bandwidth, recompute, communication,
and everything-else. Each subsequent chapter owns one bucket. The
final chapter draws the boundary with the AI-infra performance-
engineering role — you *integrate and measure* kernels; you do not
author them.

## Learning objectives

- Compute MFU (Model FLOPs Utilization) correctly for a training
  step and reason about the gap to peak.
- Integrate FlashAttention v2 (Ampere) and v3 (Hopper + FP8) and
  measure the throughput lift.
- Apply BF16 mixed precision across FSDP2, and layer FP8 with NVIDIA
  Transformer Engine on H100.
- Combine activation checkpointing, gradient checkpointing, and
  sequence packing to hit a memory target.
- Use `torch.compile` (PT2) + Triton-authored kernels correctly with
  FSDP2 and DeepSpeed.
- Design communication–compute overlap (backward all-reduce, forward
  all-gather prefetch) and measure the overlap ratio.
- Frame the boundary with the ai-infra-performance-engineer role:
  integrate published kernels; do not author them.

## Chapters

1. [MFU and the Gap to Peak](01-mfu-and-the-gap-to-peak.md) — the
   PaLM-paper definition, the `6·N·B·S + attention` numerator, the
   dense-vendor-peak denominator, HFU vs. MFU, and the five gap
   buckets that map onto the rest of the module.
2. [FlashAttention v2 (Ampere) and v3 (Hopper + FP8)](02-flashattention-v2-and-v3.md)
   — which variant for which hardware, three integration paths
   through PyTorch, the four-step measurement protocol, and the FP8-
   attention caveat.
3. [BF16 Mixed Precision with FSDP2](03-bf16-mixed-precision-with-fsdp2.md)
   — the modern default policy, `MixedPrecisionPolicy` on FSDP2,
   what stays in FP32 and why, and the equivalence-test discipline
   that catches subtle bugs.
4. [FP8 with NVIDIA Transformer Engine on H100](04-fp8-with-transformer-engine.md)
   — E4M3/E5M2, the DelayedScaling recipe, layering FP8 matmuls on
   top of the chapter-3 BF16 policy, and the two-step verification
   (profile + equivalence run) before shipping.
5. [Activation Checkpointing, Gradient Checkpointing, and Sequence
   Packing](05-activation-checkpointing-and-sequence-packing.md) —
   activations as the dominant HBM term, the compute-for-memory
   trade of AC, and the free-MFU story of packing. Includes the
   memory-target decision list.
6. [`torch.compile` (PT2) and Triton with FSDP2 / DeepSpeed](06-torch-compile-and-triton-with-fsdp2.md)
   — Dynamo + AOTAutograd + Inductor, the FSDP2 composition rules,
   consuming published Triton kernels, and the recompile-storm /
   cache-thrash failure modes.
7. [Communication–Compute Overlap: All-Gather Prefetch and All-
   Reduce Overlap](07-communication-compute-overlap.md) — the
   overlap-ratio metric, how to measure it with `torch.profiler`
   and Nsight, and the design levers that shift the compute-to-
   comm ratio.
8. [The Boundary With the AI-Infra Performance-Engineering
   Role](08-boundary-with-performance-engineering.md) — what the
   performance engineer owns, what you own, the escalation
   contract, and the anti-patterns to avoid.

## Exercises

- [exercise-01 — MFU calculation and gap analysis](exercises/exercise-01-mfu-calculation-and-gap-analysis.md) (3 h)
- [exercise-02 — FlashAttention v2/v3 integration](exercises/exercise-02-flashattention-v2-v3-integration.md) (4 h)
- [exercise-03 — BF16 and FP8 mixed precision](exercises/exercise-03-bf16-and-fp8-mixed-precision.md) (4 h)
- [exercise-04 — Checkpointing and packing to a memory budget](exercises/exercise-04-checkpointing-and-packing-memory-budget.md) (3 h)
- [exercise-05 — `torch.compile` with FSDP2](exercises/exercise-05-torch-compile-with-fsdp2.md) (3 h)
- [exercise-06 — Communication–compute overlap measurement](exercises/exercise-06-comm-compute-overlap-measurement.md) (3 h)

## Labs and quizzes

- `labs/` — a `lab-01` end-to-end MFU-lift study (baseline →
  FlashAttention → BF16 policy audit → sequence packing → compile
  → overlap tuning) lands here on the next autonomous cycle.
- `quizzes/` — one knowledge check lands here on the next
  autonomous cycle.

## Resources

- [resources.md](resources.md) — primary papers (PaLM MFU
  definition, FlashAttention v1/v2/v3, FP8 formats), vendor docs
  (NVIDIA H100 architecture, Transformer Engine, NCCL, Nsight),
  framework references (PyTorch FSDP2, `torch.compile`, DeepSpeed),
  and a recommended reading order.

## How the module fits together

Chapter 1 gives you the yardstick and the diagnostic budget.
Chapters 2–4 attack bucket 1 (kernels and dtypes) from three
angles: attention (chapter 2), BF16 storage/compute (chapter 3),
FP8 matmul on top (chapter 4). Chapter 5 attacks bucket 3
(recompute) and its free-MFU cousin sequence packing. Chapter 6
attacks buckets 1 and 2 together via graph-level fusion. Chapter 7
attacks bucket 4 (comm). Chapter 8 clarifies whose job the "make
this specific kernel faster" work is and how you escalate. The
exercises march roughly in order; do them in sequence.

## What this module deliberately does not cover

- **Distributed-training semantics** (DDP, FSDP2, TP, PP, 3D-
  parallel) — owned by mod-101. This module *uses* FSDP2 as the
  parallelism baseline for the mixed-precision and overlap
  chapters, but does not re-derive the sharding math.
- **Framework internals** (Megatron, DeepSpeed, torchtitan
  internals) — owned by mod-102. This module treats them as
  consumers of the same underlying kernels and precision policies.
- **Data pipeline design and shard formats** — owned by mod-103.
  The sampler and loader that feed the packing pattern of chapter
  5 are mod-103's problem; this module treats the loader as a
  black box that must not stall the step.
- **Scheduler and topology hints** — owned by mod-104.
- **Fabric configuration and NCCL tuning** — owned by mod-105.
  This module measures overlap against a fabric assumed correctly
  tuned; when the fabric itself is wrong, mod-105 chapter 8 is
  the right runbook.
- **Checkpointing, elastic training, goodput SLO** — owned by
  mod-106. This module owns MFU when the run is up; mod-106 owns
  everything that happens when it stops.
- **Kernel authorship** — owned by the AI-infra performance-
  engineering role. Chapter 8 is the boundary; every other chapter
  respects it.
- **Cost accounting and capacity planning** — owned by mod-109.
  MFU is the input; mod-109 turns it into a dollar number.
