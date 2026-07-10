# mod-107 — Throughput and MFU Engineering: Kernels, Mixed Precision, and Communication Overlap

**Estimated effort:** 20 hours

mod-101 taught you the parallelism strategies and the α + β cost
model; mod-102 taught you the frameworks (FSDP2, DeepSpeed, Megatron)
that implement them; mod-105 taught you to make NCCL move bytes at
line rate over the fabric. This module is where those pieces
translate into a single number a training-run report can defend:
**MFU** — the fraction of the datasheet peak your training run is
actually achieving. Every optimisation in this module is a named
attack on a named term in the MFU-gap accounting: FlashAttention v2
and v3 attack the attention-kernel term; BF16 across FSDP2 and FP8
via Transformer Engine attack the dtype term; activation
checkpointing and sequence packing attack the memory and pad-token
terms; `torch.compile` attacks the Python-dispatch term; and
communication-compute overlap attacks the exposed-comm term. The
final chapter draws the boundary with the peer track
`ai-infra-performance-learning` — kernel authoring lives there, not
here.

By the end of the module you should be able to (a) compute MFU
correctly on your own run and decompose the gap to peak into named
contributions, (b) integrate FlashAttention v2 or v3 and measure the
throughput lift with an A/B, (c) apply BF16 across FSDP2 and layer
FP8 with NVIDIA Transformer Engine on H100 without breaking
convergence, (d) combine activation checkpointing and sequence
packing to hit a memory target while raising MFU, (e) install
`torch.compile` under FSDP2 and DeepSpeed without regressing on
graph breaks or shape recompiles, (f) design and measure
communication-compute overlap in a torch profiler + `nsys` trace,
and (g) frame the escalation contract with the performance-engineer
peer track — what stays in-house, what escalates, what evidence to
attach.

## Learning objectives

- Compute MFU (Model FLOPs Utilization) correctly for a training
  step and reason about the gap to peak in named contributions.
- Integrate FlashAttention v2 (Ampere) and v3 (Hopper + FP8) and
  measure the throughput lift.
- Apply BF16 mixed precision across FSDP2, and layer FP8 with
  NVIDIA Transformer Engine on H100.
- Combine activation checkpointing, gradient checkpointing, and
  sequence packing to hit a memory target.
- Use `torch.compile` (PT2) + Triton-authored kernels correctly
  with FSDP2 and DeepSpeed.
- Design communication-compute overlap (backward all-reduce,
  forward all-gather prefetch) and measure the overlap ratio.
- Frame the boundary with `ai-infra-performance-engineer`:
  integrate published kernels; do not author them.

## Chapters

1. [MFU: Calculation and the Gap to Peak](01-mfu-calculation-and-gap-to-peak.md) —
   the definition (Chowdhery et al., 2022), the `6·N·D` FLOP count
   (Kaplan et al., 2020), H100 peak by dtype, the instrumentation
   you need, HFU vs. MFU, and the gap decomposition that structures
   the rest of the module.
2. [FlashAttention v2 and v3: Integration, Not Authoring](02-flashattention-v2-and-v3-integration.md) —
   the tiled fused kernel, v2 (Dao, 2023, arXiv:2307.08691) on
   Ampere, v3 (Shah et al., 2024, arXiv:2407.08608) on Hopper with
   WGMMA async + FP8, the three integration paths (direct
   `flash_attn_func`, PyTorch SDPA backend, HuggingFace
   `attn_implementation`), and the A/B measurement protocol.
3. [BF16 Mixed Precision Across FSDP2](03-bf16-mixed-precision-across-fsdp2.md) —
   BF16's FP32-range advantage over FP16, FSDP2's
   `MixedPrecisionPolicy` (`param_dtype` / `reduce_dtype` /
   `output_dtype`), what stays FP32 (Adam master, LN, softmax
   variance), the FP32-gradient-reduction trade at scale, and the
   torchtitan reference config.
4. [FP8 with NVIDIA Transformer Engine](04-fp8-with-transformer-engine.md) —
   E4M3 / E5M2 formats (Micikevicius et al., 2022,
   arXiv:2209.05433), the `DelayedScaling` recipe, TE's `Linear` /
   `LayerNormLinear` / `TransformerLayer` drop-ins, which ops go
   FP8 and which stay BF16, composition with FSDP2 (amax sharding)
   and with FA v3.
5. [Activation Checkpointing and Sequence Packing](05-activation-checkpointing-and-sequence-packing.md) —
   the memory-for-compute trade (Chen et al., 2016), selective
   checkpointing (Korthikanti et al., 2022), `torch.utils.checkpoint`
   and `apply_activation_checkpointing`, sequence packing via
   FA's `flash_attn_varlen_func` and `cu_seqlens`, and the MFU
   accounting when both are on.
6. [`torch.compile` with FSDP2 and DeepSpeed](06-torch-compile-with-fsdp2-and-deepspeed.md) —
   PT2 = Dynamo + AOT Autograd + Inductor, the three modes,
   composition with FSDP2 (per-block compile, PyTorch ≥ 2.4),
   dynamic-shape handling, graph-break diagnosis, DeepSpeed's
   `.compile()` caveats, and Triton as the escape hatch.
7. [Communication-Compute Overlap](07-communication-compute-overlap.md) —
   the three overlaps (forward AG prefetch, backward RS,
   optimizer step), FSDP2's `reshard_after_forward` and
   `set_requires_gradient_sync`, DeepSpeed's `overlap_comm` and
   bucket-size knobs, the Kineto + `nsys` trace-reading playbook,
   and the `T_comm_exposed / T_step` metric.
8. [The Boundary with the Performance Engineer](08-boundary-with-performance-engineer.md) —
   what stays in-house (integration, policy, dashboarding) vs.
   what belongs to `ai-infra-performance-learning` (kernel
   authoring in CUDA / CUTLASS / Triton, FA internals, custom FP8
   recipes), the three escalation triggers, the evidence pack, and
   the reverse-direction consumption contract.

## Exercises

- [exercise-01 — MFU calculation and gap analysis](exercises/exercise-01-mfu-calculation-and-gap-analysis.md) (3 h)
- [exercise-02 — FlashAttention v2 / v3 integration](exercises/exercise-02-flashattention-v2-v3-integration.md) (4 h)
- [exercise-03 — BF16 and FP8 mixed precision](exercises/exercise-03-bf16-and-fp8-mixed-precision.md) (4 h)
- [exercise-04 — Checkpointing and packing memory budget](exercises/exercise-04-checkpointing-and-packing-memory-budget.md) (3 h)
- [exercise-05 — `torch.compile` with FSDP2](exercises/exercise-05-torch-compile-with-fsdp2.md) (3 h)
- [exercise-06 — Comm-compute overlap measurement](exercises/exercise-06-comm-compute-overlap-measurement.md) (3 h)

## Labs and quizzes

- `labs/` — a `lab-01` MFU uplift report (baseline → chapter-by-
  chapter delta, with traces attached) lands here on the next
  autonomous cycle.
- `quizzes/` — one knowledge check lands here on the next autonomous
  cycle.

## Resources

- [resources.md](resources.md) — primary papers (Kaplan, PaLM, FA v2,
  FA v3, FP8, activation-recomputation, sublinear memory), framework
  docs (PyTorch FSDP2, `torch.compile`, `torch.utils.checkpoint`,
  `torch.profiler`, NVIDIA Transformer Engine, DeepSpeed,
  FlashAttention repo, Triton, CUTLASS), profiling tooling (Nsight
  Systems, Kineto/HTA, NCCL debug), and a recommended reading order.

## How the module fits together

Chapter 1 fixes the metric and decomposes the gap; every subsequent
chapter attacks one term. Chapter 2 closes the attention-kernel
term; chapter 3 closes the FP32-to-BF16 dtype term; chapter 4 closes
the BF16-to-FP8 dtype term on Hopper. Chapter 5 handles the memory
and pad-token terms so the batch size can grow into the compute
regime the previous chapters unlock. Chapter 6 removes Python-side
dispatch overhead so what remains is real compute and real comm.
Chapter 7 hides the comm behind the compute using
FSDP2 / DeepSpeed / Megatron overlap machinery, measured in Kineto
and Nsight Systems traces. Chapter 8 draws the boundary with the
peer performance-engineer track so both roles know where their
depth ends. Exercises 1–6 march in the same order and end with an
MFU uplift report — the same report a real pretraining run would
attach to its post-mortem.

## What this module deliberately does not cover

- **Parallelism-strategy basics** (DP, TP, PP, SP, EP, HSDP, the
  α + β cost model) — owned by mod-101. This module *uses* mod-101's
  cost model as the analytic ceiling every MFU number is compared
  against.
- **Framework internals** (FSDP2 sharding mechanics, DeepSpeed
  ZeRO stages, Megatron-LM tensor parallel implementation) — owned
  by mod-102. This module *uses* the frameworks as installed.
- **NCCL tuning at fabric depth** (`NCCL_ALGO`, PXN, rail
  alignment, `nccl-tests` benchmarking) — owned by mod-105.
  Chapter 7's overlap analysis assumes NCCL is delivering
  near-line-rate `busbw`; mod-105 makes that so.
- **Distributed checkpointing and elastic training** — owned by
  mod-106. Chapter 4 mentions DCP for FP8 amax state save/resume;
  mod-106 owns the correctness and recovery semantics.
- **Training-run dashboards, DCGM, W&B / TensorBoard / MLflow at
  scale, reproducibility bundles** — owned by mod-108. This module
  produces the MFU number; mod-108 owns how it lands on a
  dashboard.
- **Cost, capacity, and cluster economics** (Chinchilla,
  reserved-vs-spot, MFU-uplift-to-dollars) — owned by mod-109.
- **Multi-tenant platform architecture and cross-team leadership
  contracts** — owned by mod-110.
- **Kernel authoring** (CUDA, CUTLASS, Triton, FA internals, custom
  FP8 recipes, occupancy tuning at SASS level) — owned by the peer
  track [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning).
  Chapter 8 formalises the boundary.
