# Resources for mod-107 — Throughput and MFU Engineering: Kernels, Mixed Precision, and Communication Overlap

Primary papers first (they define the metrics and the kernels), then
vendor hardware docs and libraries (they define the peak), then
framework references (they are the surface you actually touch), then
profilers, then adjacent modules. The four references you should be
able to reach for by memory by the end of the module are the PaLM
paper (MFU definition), the FlashAttention v2/v3 papers, the FP8-
formats paper, and the FSDP2 tutorial.

## Foundational papers (metric, kernels, precision, compiler)

- **Chowdhery, A., et al., 2022. "PaLM: Scaling Language Modeling
  with Pathways."** arXiv 2204.02311.
  https://arxiv.org/abs/2204.02311 — appendix B is the origin of the
  MFU definition used throughout chapter 1. Every MFU number you
  publish traces back to this appendix; read it before writing an
  MFU report.
- **Kaplan, J., et al., 2020. "Scaling Laws for Neural Language
  Models."** arXiv 2001.08361. https://arxiv.org/abs/2001.08361 —
  the source of the `6·N·B·S` dense-Transformer FLOP accounting
  in chapter 1's numerator.
- **Dao, T., et al., 2022. "FlashAttention: Fast and Memory-
  Efficient Exact Attention with IO-Awareness."** arXiv 2205.14135.
  https://arxiv.org/abs/2205.14135 — the original FlashAttention
  paper; the IO-awareness argument that motivates chapter 2.
- **Dao, T., 2023. "FlashAttention-2: Faster Attention with Better
  Parallelism and Work Partitioning."** arXiv 2307.08691.
  https://arxiv.org/abs/2307.08691 — v2, the Ampere / early-Hopper
  target and the current SDPA default backend on those platforms.
- **Shah, J., et al., 2024. "FlashAttention-3: Fast and Accurate
  Attention with Asynchrony and Low-Precision."** arXiv 2407.08608.
  https://arxiv.org/abs/2407.08608 — v3, built for Hopper WGMMA /
  TMA, with FP8 attention. Figure 1 is the reference point for the
  "FA3 near 75% of H100 BF16 peak" claim in chapter 2's kernel-
  isolation section.
- **Micikevicius, P., et al., 2018. "Mixed Precision Training."**
  arXiv 1710.03740. https://arxiv.org/abs/1710.03740 — the original
  mixed-precision paper (FP16-era, pre-BF16), cited in chapter 3 as
  the ancestor of the modern policy.
- **Micikevicius, P., et al., 2022. "FP8 Formats for Deep Learning."**
  arXiv 2209.05433. https://arxiv.org/abs/2209.05433 — the paper
  that defined E4M3 and E5M2. Chapter 4 assumes you have read the
  format section.
- **Chen, T., et al., 2016. "Training Deep Nets with Sublinear
  Memory Cost."** arXiv 1604.06174.
  https://arxiv.org/abs/1604.06174 — the original activation-
  checkpointing paper; chapter 5's cost model.
- **Ansel, J., et al., 2024. "PyTorch 2: Faster Machine Learning
  Through Dynamic Python Bytecode Transformation and Graph
  Compilation."** ASPLOS'24.
  https://dl.acm.org/doi/10.1145/3620665.3640366 — the PyTorch 2
  paper covering TorchDynamo, AOTAutograd, and TorchInductor;
  chapter 6's architectural reference.
- **Rasley, J., et al., 2020. "ZeRO: Memory Optimizations Toward
  Training Trillion Parameter Models."** arXiv 1910.02054.
  https://arxiv.org/abs/1910.02054 — the DeepSpeed / ZeRO
  foundation cited in chapter 6's DeepSpeed composition section.

## Frontier-scale training reports (published MFU anchors)

- **Grattafiori, A., et al., 2024. "The Llama 3 Herd of Models."**
  arXiv 2407.21783. https://arxiv.org/abs/2407.21783 — section 3.4
  reports the observed BF16 MFU during Llama 3 pretraining. This is
  the public H100-BF16 anchor for the "40–55% MFU band" claim in
  chapter 1 and the exercise-1 report template.
- **Le Scao, T., et al., 2022. "BLOOM: A 176B-Parameter Open-Access
  Multilingual Language Model."** arXiv 2211.05100.
  https://arxiv.org/abs/2211.05100 — section 3.3 is chapter 3's
  citation for BF16 becoming the default mixed-precision format at
  frontier scale.

## NVIDIA hardware, kernels, and libraries

- **NVIDIA H100 Tensor Core GPU product page and datasheet.**
  https://www.nvidia.com/en-us/data-center/h100/ — the canonical
  source for the 989 TFLOP/s BF16 / 1979 TFLOP/s FP8 peak numbers
  used as the MFU denominator in chapter 1. Use the *dense* number
  (not the sparsity-doubled one) as chapter 1 emphasizes.
- **NVIDIA H100 Tensor Core GPU Architecture whitepaper.** Linked
  from the H100 product page above — describes the Hopper FP8
  tensor-core path (chapter 4) and the WGMMA / TMA units FA3 targets
  (chapter 2).
- **NVIDIA A100 Tensor Core GPU product page.**
  https://www.nvidia.com/en-us/data-center/a100/ — Ampere BF16 /
  FP16 peak (312 TFLOP/s dense) for the same MFU denominator on A100
  fleets.
- **NVIDIA H200 product page.**
  https://www.nvidia.com/en-us/data-center/h200/ — same tensor-core
  math throughput as H100, higher HBM bandwidth. Confirms the
  chapter-1 note that MFU numerator and denominator are unchanged
  from H100.
- **NVIDIA Blackwell architecture / B200 product page.**
  https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
  — consult for current-generation FP4/FP8/FP16 peak and MXFP8 block-
  scaling recipes; the chapter-4 FP8 recipe class differs on Hopper
  vs. Blackwell. Confirm the version-specific numbers before using
  them.
- **NVIDIA Transformer Engine documentation.**
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/
  — canonical reference for the FP8 integration used in chapter 4:
  `fp8_autocast`, `DelayedScaling`, the `Format.HYBRID` recipe, and
  the module analogues (`te.Linear`, `te.LayerNormLinear`,
  `te.LayerNormMLP`, `te.DotProductAttention`).
- **NVIDIA Transformer Engine (GitHub).**
  https://github.com/NVIDIA/TransformerEngine — the library source
  with PyTorch / JAX / TensorFlow bindings. The release notes are
  the source of truth for the Hopper-vs-Blackwell recipe matrix.
- **FlashAttention repository (Dao-AILab).**
  https://github.com/Dao-AILab/flash-attention — the reference
  implementation for FA v1/v2/v3, plus `flash_attn_func` and
  `flash_attn_varlen_func`. Chapter 2's "which variant for which
  hardware" table and chapter 5's variable-length packing path both
  point here. The build matrix in the README is authoritative for
  what CUDA + PyTorch combinations get you v3.
- **NCCL user guide.**
  https://docs.nvidia.com/deeplearning/nccl/user-guide/ — timeouts,
  `NCCL_DEBUG`, environment variables, and the reference for the
  `ncclAllGather` / `ncclReduceScatter` kernels chapter 7 asks you
  to look for in a profile.
- **CUTLASS.** https://github.com/NVIDIA/cutlass — the CUDA GEMM
  template library the performance-engineering role builds on
  (chapter 8). You will read about it more than you write it.

## Kernel languages and profilers

- **Triton (kernel language).**
  https://github.com/triton-lang/triton — the OpenAI-originated
  Python-embedded GPU kernel language TorchInductor emits and that
  library authors write in. See also the Triton project page at
  https://openai.com/index/triton/. Chapter 6 covers *consuming*
  Triton kernels; authoring is chapter 8's boundary.
- **NVIDIA Nsight Systems (`nsys`).**
  https://developer.nvidia.com/nsight-systems — timeline profiler
  used in chapter 7's overlap-measurement path 2 and in chapter 2's
  kernel-name verification. Prefer over `torch.profiler` for
  cross-node and NIC-utilization questions.
- **NVIDIA Nsight Compute (`ncu`).**
  https://developer.nvidia.com/nsight-compute — kernel-level
  profiler (warp occupancy, memory throughput, roofline).
  Chapter 8's boundary section — the training-pipeline engineer
  reads Nsight Systems; the performance engineer reads Nsight
  Compute.
- **PyTorch Profiler documentation.**
  https://docs.pytorch.org/docs/stable/profiler.html — the
  `torch.profiler.profile` API and `ProfilerActivity` used in
  chapter 2's measurement protocol and chapter 7's overlap-ratio
  computation. Traces can be viewed in Chrome tracing
  (`chrome://tracing`) or at https://ui.perfetto.dev/.

## PyTorch framework references

- **PyTorch `torch.compile` documentation.**
  https://docs.pytorch.org/docs/stable/torch.compiler.html — API
  surface for chapter 6: `torch.compile`, `mode`, `fullgraph`,
  `dynamic`, `TORCH_LOGS`. Read alongside the PyTorch 2 paper for
  the architecture.
- **PyTorch `torch.compile` tutorial.**
  https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
  — the walk-through of common `torch.compile` pitfalls (graph
  breaks, dynamic shapes, cache). Do it end-to-end before starting
  exercise 5.
- **PyTorch FSDP2 tutorial.**
  https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
  — the current reference for `fully_shard`, `MixedPrecisionPolicy`,
  wrap policies, and the prefetch behavior chapter 7 discusses.
  Check the FSDP2 / `fully_shard` sections; older docs cover the
  legacy `FullyShardedDataParallel` class.
- **PyTorch `torch.nn.attention` / SDPA documentation.**
  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
  and https://docs.pytorch.org/docs/stable/backends.html#torch.backends.cuda.sdp_kernel
  — the API surface for chapter 2's `sdpa_kernel(FLASH_ATTENTION)`
  dispatch pattern and the `SDPBackend` enum.
- **PyTorch activation checkpointing (`torch.utils.checkpoint`).**
  https://docs.pytorch.org/docs/stable/checkpoint.html — the
  `checkpoint(use_reentrant=False, ...)` API used in chapter 5's
  per-block wrapping and in chapter 6's compile-compatibility note.
- **PyTorch `FlopCounterMode`.**
  https://docs.pytorch.org/docs/stable/generated/torch.utils.flop_counter.FlopCounterMode.html
  — the FLOP-counting utility exercise 1 uses as a cross-check
  against the analytic `6·N·B·S` estimate.
- **PyTorch CUDA memory instrumentation.**
  https://docs.pytorch.org/docs/stable/torch_cuda_memory.html and
  the Memory Snapshot Viewer at https://pytorch.org/memory_viz —
  `max_memory_allocated`, `reset_peak_memory_stats`, and the
  `_record_memory_history` / `_dump_snapshot` snapshotting used in
  chapter 5's instrumentation snippet and in exercise 4.

## Framework references beyond PyTorch

- **DeepSpeed documentation.** https://www.deepspeed.ai/ — chapter
  6's composition-with-DeepSpeed section is grounded here. Also
  covers ZeRO stages 1/2/3 semantics referenced in the chapter.
- **DeepSpeed on GitHub.**
  https://github.com/microsoft/DeepSpeed — the source and release
  notes. Check current-release notes for `torch.compile` +
  ZeRO-3 compatibility before enabling.
- **torchtitan repository.** https://github.com/pytorch/torchtitan
  — a PyTorch-native reference training stack that uses FSDP2 and
  demonstrates chapter 2's SDPA integration, chapter 3's mixed-
  precision policy, and chapter 4's TE wiring in one codebase.
  Useful to read alongside all four chapters.
- **Megatron-LM.** https://github.com/NVIDIA/Megatron-LM — a
  distinct reference stack with its own fused-kernel and selective
  activation-checkpointing implementations. Chapter 5's "selective
  AC" reference implementation lives here.

## Adjacent modules and cross-references

- **mod-101** — distributed training foundations. FSDP2 sharding
  semantics underlie every mixed-precision, memory, and overlap
  chapter. Chapter 7's overlap discussion assumes mod-101's all-
  gather / reduce-scatter shape math is familiar.
- **mod-102** — training frameworks. Megatron, DeepSpeed, and
  torchtitan internals are mod-102's territory; this module treats
  them as consumers of the same underlying kernels and precision
  policies.
- **mod-103** — data pipelines. Chapter 5's sequence-packing sampler
  and loader are mod-103's responsibility; this module assumes the
  loader is fast enough not to stall the step (chapter 7's blocking-
  `next()` failure mode).
- **mod-104** — cluster orchestration. Scheduler topology hints and
  gang scheduling live there; this module measures MFU on top of a
  correctly-scheduled gang.
- **mod-105** — networking and storage. Chapter 7's overlap
  arithmetic is measured against a fabric assumed correctly tuned;
  mod-105 chapter 5's `nccl-tests` numbers are the input, and
  mod-105 chapter 8 is the runbook when the fabric itself is wrong.
  SHARP (chapter 7's design lever) is documented under mod-105
  chapter 3.
- **mod-106** — checkpointing and fault tolerance. Chapter 5 notes
  DCP async save does not affect the HBM budget; chapter 7 notes
  that checkpoint writes on the critical path show up in bucket 5.
- **mod-108** — observability. This module *emits* the counters
  (MFU, HFU, overlap ratio, per-layer step time) that mod-108 turns
  into dashboards.
- **mod-109** — cost accounting. MFU is the input; mod-109 turns it
  into a dollar-per-token figure. Chapter 1 explicitly does not
  cover the translation.

## Recommended reading order for a first pass

1. Chapter 1 + PaLM appendix B + Llama 3 §3.4 (2 h). Fixes the
   vocabulary and gives you a public anchor for the 40–55% band.
2. Chapter 2 + skim the FA v2 and v3 abstracts + `Dao-AILab/flash-
   attention` README (1.5 h). Enough to reason about kernel choice
   before starting exercise 2.
3. Chapter 3 + FSDP2 tutorial's `MixedPrecisionPolicy` section +
   Micikevicius 2018 §2–3 (2 h). Then start exercise 3's BF16 half.
4. Chapter 4 + TE user guide's "Getting Started" and the FP8-
   formats paper §2 (2 h). Enough for exercise 3's FP8 half on
   H100.
5. Chapter 5 + Chen et al. 2016 abstract + FA `flash_attn_varlen_func`
   reference (1.5 h). Then start exercise 4.
6. Chapter 6 + PyTorch 2 paper §3 + the `torch.compile` tutorial
   end-to-end (2 h). Then start exercise 5.
7. Chapter 7 + PyTorch Profiler tutorial + Nsight Systems quick-
   start (2 h). Then start exercise 6 with mod-105 chapter 5's
   `nccl-tests` numbers already in hand.
8. Chapter 8 (0.5 h). Skim it before your first design conversation
   with a performance engineer.

Exercise 1 is the prerequisite for every other exercise — do it
first with the vocabulary from item 1. Exercises 2–6 can then be
tackled in the module order or reordered by the gap-to-peak budget
you built in exercise 1's report (its own final section is exactly
this reordering).
