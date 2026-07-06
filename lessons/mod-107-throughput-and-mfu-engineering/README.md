# mod-107-throughput-and-mfu-engineering: Throughput and MFU Engineering: Kernels, Mixed Precision, and Communication Overlap

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 20 hours

## Learning objectives

- Compute MFU (Model FLOPs Utilization) correctly for a training step and reason about the gap to peak
- Integrate FlashAttention v2 (Ampere) and v3 (Hopper + FP8) and measure the throughput lift
- Apply BF16 mixed precision across FSDP2, and layer FP8 with NVIDIA Transformer Engine on H100
- Combine activation checkpointing, gradient checkpointing, and sequence packing to hit a memory target
- Use `torch.compile` (PT2) + Triton-authored kernels correctly with FSDP2 and DeepSpeed
- Design communication-compute overlap (backward all-reduce, forward all-gather prefetch) and measure the overlap ratio
- Frame the boundary with ai-infra-performance-engineer: integrate published kernels; do not author them

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
