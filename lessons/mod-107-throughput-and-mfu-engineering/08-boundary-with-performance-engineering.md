# The Boundary With the AI-Infra Performance-Engineering Role

Every chapter of this module has told you to *integrate* published
kernels rather than author them. This chapter says why, spells out
the boundary explicitly, and gives you the vocabulary to work
alongside the ai-infra-performance-engineer role without stepping on
their work — or ceding parts of the throughput problem you should
own.

The short version: the performance-engineer role authors and tunes
the kernels that make bucket 1 of the gap-to-peak budget (chapter 1)
achievable. You, the training-pipeline engineer, assemble those
kernels into a training loop, choose the framework glue, pick the
mixed-precision policy, and measure the end-to-end MFU. Both jobs
are real; neither replaces the other; and the interface between
them is the topic of this chapter.

## What the performance-engineer role owns

The archetypal performance engineer in the modern AI-infra org is
the person who:

- Writes CUDA, Triton, or CUTLASS kernels for a specific op that
  the framework's default is too slow at.
- Tunes existing kernels — schedule, tile size, pipeline depth,
  register allocation — for a specific accelerator generation.
- Builds and maintains libraries like FlashAttention, xFormers,
  Transformer Engine's fused SwiGLU / attention paths, ThunderKittens,
  and the CUTLASS/CUDA-C++ backends behind them.
- Profiles at the SASS / warp / SM level. Reads Nsight Compute
  timelines. Owns the "occupancy is 63%; here's why" conversation.
- Answers "what does this kernel need to change for Blackwell tensor
  cores" ahead of the framework catching up.

They are the reason FlashAttention v3 exists. They are the reason
Transformer Engine's fused block runs at the FP8 tensor-core peak.
They are why `torch.compile`'s TorchInductor produces credible
Triton in the first place — the compiler team is deeply overlapping
with the performance-engineer discipline.

Their tools:

- **NVIDIA Nsight Compute (`ncu`).** Kernel-level profiler: warp
  occupancy, memory throughput, instruction mix, roofline analysis.
  https://developer.nvidia.com/nsight-compute.
- **NVIDIA Nsight Systems (`nsys`).** Timeline profiler; overlaps
  with your training-loop world but goes deeper into per-kernel
  timing.
- **CUTLASS.** NVIDIA's template library for CUDA GEMM primitives.
  https://github.com/NVIDIA/cutlass.
- **Triton.** OpenAI's Python-embedded kernel language.
  https://openai.com/index/triton/ and
  https://github.com/triton-lang/triton.
- **Roofline analysis.** The classical "peak FLOPs vs. HBM
  bandwidth" framework for reasoning about whether a kernel is
  compute-bound or memory-bound.

## What you own as the training-pipeline engineer

Bucket by bucket from chapter 1:

- **Bucket 1 (kernel/dtype gap).** You *integrate*: FlashAttention
  v2/v3 through SDPA (chapter 2), Transformer Engine's fused
  modules under `fp8_autocast` (chapter 4), TorchInductor's
  generated Triton by turning `compile` on (chapter 6). You
  *measure* the win with the kernel-isolation and full-step
  protocols from chapter 2. You do *not* author new kernels; you
  do escalate to the performance-engineer role when the published
  kernels are missing a shape or dtype your model needs.
- **Bucket 2 (memory-bandwidth gap).** You choose the mixed-
  precision policy (chapters 3–4) that lowers per-tensor bytes;
  you enable `torch.compile` to fuse away extra HBM round trips
  (chapter 6). The performance engineer might rewrite the
  norm-linear-activation-dropout stack as one kernel; you
  configure whether it dispatches.
- **Bucket 3 (recomputation gap).** Entirely yours (chapter 5).
  Activation checkpointing and packing are training-loop
  decisions; the performance engineer does not have a lever here.
- **Bucket 4 (communication gap).** Yours in the FSDP2 config and
  wrap policy sense (chapter 7). NCCL tuning and fabric config
  are owned by mod-105 and are shared with the platform/network
  team; the performance engineer is rarely in this loop.
- **Bucket 5 (everything else).** Yours. Loader, checkpoint I/O
  path (mod-106), Python overhead per step, sync bugs (chapter 7).

The dividing line, cleaner: the performance engineer owns *inside
one kernel or op*; you own *between kernels* and *end-to-end*.

## What "don't author kernels" actually means

It does *not* mean "never write Triton". It means: do not silently
absorb kernel-authoring into your role. Concretely:

- **Do not fork FlashAttention** to fix a shape your model needs.
  File an issue on the upstream repo; if urgent, work with your
  performance-engineering counterpart to author a patch. A forked
  kernel is a maintenance liability that lives forever.
- **Do not paste a Triton snippet from a blog post into
  production.** Blog-post Triton is un-tested, un-tuned, and
  version-drifts against the `triton` package. Take a dependency
  on a library instead.
- **Do not add "custom fused" kernels to the training loop to
  chase small wins.** A 2% MFU win from a hand-fused kernel that
  breaks on the next CUDA update is a bad trade. The performance
  engineer's kernels ship with a test suite and a support
  contract; a training-engineer's one-off does not.
- **Do prototype kernels for experiments** when you are exploring
  a new architecture. Move them into the performance-engineer's
  library the moment they need to be shipped.

The rule of thumb: if the kernel appears in a `git log --follow`
of your training repo more than twice a quarter, it should live in
a kernel library and be owned by someone whose job it is.

## The two shared surfaces you will design together

Two interfaces sit between the roles and require joint design:

### 1. The op registration and shape/dtype matrix

A performance engineer writing a new fused kernel needs to know:

- What shapes and dtypes it must support (your model's actual
  configurations, not hypothetical ones).
- Whether it needs to compose with `torch.compile` (register as
  `torch.library` op with a `meta` implementation) or is called
  directly.
- The correctness reference — usually your existing non-fused
  path — and the tolerance the fused version must match to.
- Whether backward is required and, if so, whether autograd's
  automatic differentiation is acceptable or a hand-written
  backward is needed.

You *specify* those requirements; they *implement* the kernel.
Neither of you own both halves.

### 2. The measurement contract

When they hand you a new kernel, you re-run the chapter-1 MFU
measurement end-to-end. When you find a regression that looks
kernel-shaped (a shape or dtype the fast path stopped hitting), you
hand back a minimal reproducer with the shape/dtype and the
observed vs. expected step time. This is the same handoff pattern
mod-106 chapter 5 uses for incident escalation to the platform
team: your job is a well-formed reproducer, not a diagnosis of the
kernel internals.

## When to escalate to the performance engineer

Cases where you should not attempt to fix it yourself, and the
signal:

- **A published kernel silently falls back for your shape.** SDPA
  went to `MATH` or `EFFICIENT` instead of `FLASH_ATTENTION`; the
  FA3 FP8 path is not dispatching under `te.fp8_autocast`. First
  verify configuration (chapter 2 measurement path); if the
  configuration is right and the shape is in the FA supported
  matrix, escalate.
- **A new hardware generation (Blackwell after Hopper) does not
  yet have the kernel you need.** File the request; do not paper
  over with a slow fallback and forget.
- **Your MFU is stuck at HFU while HFU is already high.** You are
  in a "recompute is expensive but there's no kernel to make it
  cheaper" situation; the performance engineer may be able to
  design a memory-efficient variant of the block. This is a
  research-adjacent conversation.
- **Nsight Compute shows a specific kernel below 30% of roofline
  on your shape.** Even if the framework thinks it dispatched
  correctly, the kernel might be a poor fit for the shape. The
  performance engineer owns "make this kernel faster on this
  shape".

Cases where you should *not* escalate:

- **Your MFU is low because comm is exposed.** That's chapter 7.
- **Your MFU is low because activations do not fit and you're
  full-AC'd.** That's chapter 5 (try selective AC or packing) or
  a mod-101 conversation about parallelism.
- **A specific op is slow but the fix is "use `torch.compile` and
  let TorchInductor fuse it".** Try that first.

## Career-adjacency notes

The two roles overlap; some organizations combine them at small
scale. But the *skill sets* differ enough that at multi-team scale
the split is worth defending:

- **The training-pipeline engineer's superpower** is end-to-end
  system reasoning: which chapter of this module is the right one,
  how the loader interacts with the sampler interacts with the
  fabric interacts with the checkpoint tier. Your artifact is a
  training run that stays converged and stays close to peak on a
  real cluster.
- **The performance engineer's superpower** is per-op depth:
  reading a Nsight Compute report and knowing which tile size is
  the bottleneck. Your artifact is a kernel that other people can
  drop into their training loops.

If you find yourself frequently writing kernels, either you should
switch roles or your team is under-staffed on that role. Both are
solvable; the important thing is to not silently absorb the second
role into the first.

## Summary

- The training-pipeline engineer *integrates and measures*
  kernels; the performance engineer *authors and tunes* them.
  The interface between them is a shape/dtype matrix on one side
  and an MFU measurement on the other.
- Every chapter of this module has been about the integration and
  measurement side: MFU accounting (1), FlashAttention (2), BF16
  and FP8 (3–4), memory levers (5), compile and Triton
  consumption (6), overlap (7).
- Do not fork upstream kernels or paste blog-post Triton into
  production. Depend on the libraries; escalate the missing
  cases; move successful prototypes into the performance-
  engineer's library.
- Escalate when a kernel does not dispatch, a new hardware
  generation lacks support, or Nsight Compute shows a specific
  kernel below its roofline on your shape.
- Do not escalate configuration bugs or missing chapter-5/chapter-7
  work — own those first.
- If you find yourself authoring kernels every week, that is a
  signal that either the role split is unbalanced or the team is
  understaffed on performance engineering. Solve it explicitly,
  not by drift.
