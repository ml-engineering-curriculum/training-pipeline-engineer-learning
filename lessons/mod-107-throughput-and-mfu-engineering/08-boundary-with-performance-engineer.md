# The Boundary with the Performance Engineer

Every previous chapter in this module has been an *integration*
chapter. FlashAttention v2 and v3, BF16 in FSDP2, FP8 via Transformer
Engine, `torch.compile`, comm-compute overlap — you installed
published components and measured the delta. This chapter names what
you *do not* own: the kernel authoring itself. That work belongs to
the peer track `ai-infra-performance-learning`. This chapter
codifies the hand-off contract: what stays in-house, what escalates,
and what evidence you attach to an escalation so the performance
engineer can act on it.

The three peer references you should keep visible:

- **Peer track:** `ai-infra-performance-learning` (kernel authoring
  in CUDA, CUTLASS, and Triton).
- **NVIDIA CUTLASS.** https://github.com/NVIDIA/cutlass — the
  building-block library the performance engineer composes into
  custom kernels.
- **Triton.** https://triton-lang.org/ — the DSL used for the
  higher-altitude kernel authoring in most modern open-source
  work.

## What this track owns

The training-pipeline engineer's MFU-engineering surface, per every
prior chapter in the module:

| Concern                                | Owned by this track |
|----------------------------------------|---------------------|
| MFU calculation and dashboarding       | Yes (ch. 1)         |
| FA v2 / v3 API integration             | Yes (ch. 2)         |
| BF16 mixed-precision policy under FSDP2 | Yes (ch. 3)         |
| FP8 recipe selection (`DelayedScaling`, `MXFP8BlockScaling`) | Yes (ch. 4) |
| Which modules go into `fp8_autocast`   | Yes (ch. 4)         |
| Activation-checkpointing granularity   | Yes (ch. 5)         |
| Sequence-packing pipeline choice       | Yes (ch. 5)         |
| `torch.compile` mode selection + graph-break diagnosis | Yes (ch. 6) |
| FSDP2 wrapping strategy                | Yes (ch. 6, mod-102) |
| Comm-compute overlap analysis          | Yes (ch. 7)         |
| Bucket sizes on DeepSpeed / FSDP2      | Yes (ch. 7)         |
| Reading Kineto and `nsys` traces       | Yes (ch. 7)         |
| Attributing MFU gap to a term          | Yes (ch. 1)         |
| Choosing FA v2 vs. v3 for a workload   | Yes (ch. 2)         |

Everything in this list is measurable at the training-run level,
tunable via a config or a per-module policy, and does not require
reading tensor-core SASS.

## What the peer track owns

The performance engineer's authoring surface:

| Concern                                              | Owned by peer track |
|------------------------------------------------------|---------------------|
| CUDA / CUTLASS kernel authoring                      | Yes                 |
| Triton kernel authoring for novel ops                | Yes                 |
| FA internals (tile shapes, warp specialisation, WGMMA pipelining) | Yes |
| Custom FP8 recipes beyond what TE ships              | Yes                 |
| KV-cache micro-optimisation (inference-side)         | Yes                 |
| Occupancy tuning per SM                              | Yes                 |
| SASS-level analysis (nvcc, cuobjdump, ncu)           | Yes                 |
| Cutlass / cuBLASLt heuristic overrides               | Yes                 |
| Autotune schedules for a new Triton kernel           | Yes                 |
| New attention variants (ALiBi, RoPE variants that FA does not support) | Yes |

The rule: if the fix requires modifying tensor-core code, or
authoring a Triton kernel that ships to production, it is a
peer-track responsibility. The training pipeline engineer identifies
that the fix is needed and provides the evidence; the performance
engineer writes it.

## The three escalation triggers

Three specific patterns should trigger an escalation from this
track to the peer track. Each has a signature you should be able to
identify from a trace, and each has evidence you attach to the
hand-off.

### Trigger 1: A kernel is memory-bound below the arithmetic roofline

The roofline model (Williams et al., 2009) says a kernel's
achievable performance is `min(peak_FLOPS, arithmetic_intensity ×
peak_bandwidth)`. If a matmul or fused op sits well below both
ceilings — the kernel is neither compute- nor bandwidth-bound in the
sense the profiler expects — the kernel implementation is the
problem, not the fabric.

The signature in an Nsight Compute (`ncu`) report:

- SM active cycles < 60% at max occupancy.
- Tensor pipe active cycles < the SM active fraction (tensor cores
  idle even when the SM is running).
- L2 hit rate low with the working set clearly fitting in L2.

Evidence to attach:

- The Nsight Compute report.
- The specific kernel name and its input shapes.
- What the workload is (which layer of which model at which step).

### Trigger 2: A fused kernel is missing for a specific op sequence

You have identified with `torch.compile` + `TORCH_COMPILE_DEBUG=1`
that Inductor produced N separate kernels for an op sequence that
*could* fuse. You verified this is not a graph-break issue (chapter
6). You wrote a Triton prototype that fuses the sequence and it
outperforms the split version.

Two paths from here:

- **Upstream to PyTorch Inductor.** If the fusion belongs in
  Inductor's general lowering, file a PyTorch issue with the
  reproducer.
- **Peer-track hand-off.** If the fusion is workload-specific
  (a custom op sequence unique to your training stack), the
  peer track owns the kernel author and its production maintenance.

Evidence to attach:

- The FX graph dump showing the split.
- The Triton prototype (or a description of the fusion).
- The measured throughput delta at the workload-relevant shape.

### Trigger 3: A Triton kernel needs a new autotune schedule

Triton kernels are parameterised on block shapes and warp counts.
The `@triton.autotune` decorator picks among candidate configs at
first call and caches the winner. When you inherit a Triton
kernel that was tuned for A100 and are running it on H100, the
autotune space may not include the H100-optimal config; you get
sub-peak throughput even though the kernel is well-written.

Signature:

- The kernel runs at 60–70% of the equivalent cuBLAS or FA
  throughput at the workload shape.
- `ncu` shows SM active is high but tensor pipe utilisation is
  low.
- The autotune log
  (`TRITON_PRINT_AUTOTUNING=1`) shows only a handful of configs
  considered.

Evidence to attach:

- The kernel's `@triton.autotune` config list.
- The workload-relevant shape.
- The A/B numbers vs. the alternative (cuBLAS, FA, or a competing
  kernel).

## What stays in-house

Explicit non-escalations — the training pipeline engineer's
responsibility to solve without pulling in the peer track:

- **Activation checkpointing choice** (which blocks to checkpoint,
  selective vs. full). Chapter 5.
- **FSDP2 wrapping strategy** (per-block vs. per-two-block, ordering
  with checkpointing and compile). Chapters 5 and 6.
- **Sequence packing** (loader-side decisions, `cu_seqlens` layout).
  Chapter 5.
- **Comm-compute overlap tuning** (`reshard_after_forward`,
  DeepSpeed bucket sizes). Chapter 7.
- **MFU dashboarding** (the metric itself, its instrumentation,
  the run reports). Chapter 1 and mod-108.
- **Framework version pins** (which PyTorch, which TE, which
  DeepSpeed for a specific training run). Governed by
  mod-110's platform migration plan.
- **BF16 policy per module** (which modules stay FP32, which are
  BF16). Chapter 3.
- **FP8 recipe selection and per-module opt-in** (which layers get
  `fp8_autocast`, which are excluded because of numerical
  instability). Chapter 4.

If a peer-track engineer sends you a new kernel, your job is to
integrate it, measure the delta, and dashboard the MFU lift. Your
job is *not* to modify the kernel.

## The evidence pack for an escalation

Every escalation to the peer track ships with an evidence pack. The
minimum contents:

1. **The MFU / HFU delta.** Baseline MFU, HFU, and step time; the
   observed limitation and its attribution (chapter 1's gap
   decomposition).
2. **A trace.** Kineto Chrome trace or `nsys` report for the region
   in question. Named collectives / kernels highlighted.
3. **The workload shape.** Model architecture, block dimensions,
   batch, sequence length, dtype policy, parallelism strategy.
4. **The reproducer.** Minimum Python script that produces the
   observed behaviour on a specific commit of the training stack.
5. **The alternative you compared against.** If you are escalating
   a Triton kernel autotune issue, name the alternative kernel and
   its throughput.

Without the evidence pack the peer track cannot act. With it, the
hand-off is a well-defined engineering task.

## The reverse direction: consuming a new kernel

The other side of the hand-off. When the peer track ships a new
kernel:

- **API surface.** The kernel arrives as a Python-callable custom
  op (with a meta-kernel for Dynamo trace-ability), a wheel that
  installs into the training container, and API docs.
- **Integration cost.** You write the wiring in the model:
  substitute the op, verify shape / dtype compatibility with FSDP2
  / TP, add a config flag to switch it on and off.
- **Measurement contract.** A/B before / after MFU numbers, on the
  same workload shape, run three times each for variance. You own
  the report.
- **Maintenance contract.** The peer track maintains the kernel;
  you maintain the integration. If a kernel breaks on a framework
  upgrade, you file the issue upstream; the peer track fixes and
  releases.

This is the same contract mod-110 (platform architecture)
codifies in its "hand-off contract with peer tracks" section.

## Anti-patterns

The four patterns that show a boundary violation, and the fix:

- **You are writing a Triton kernel because Inductor is slow.**
  If the kernel ships to production, this is a peer-track
  hand-off. If it stays a debugging tool, fine. Ask: "is this
  kernel maintainable by this team past this quarter?"
- **You are modifying FlashAttention to add a new attention
  variant.** Peer track. FA has a small maintainer group; any
  fork ships to `ai-infra-performance-learning` for review
  or upstreams to `Dao-AILab/flash-attention`.
- **You are tuning Transformer Engine's FP8 recipe by editing its
  amax computation.** Peer track. The recipe is a numerical
  choice with correctness implications; escalate.
- **You are calling into CUTLASS directly to write a fused
  matmul + LN.** Peer track. CUTLASS is a kernel-author library,
  not an integration surface.

## The larger point

Level 35 for a training-pipeline engineer is depth on the
integration surface. It is *not* depth on the kernel-author
surface — that is the peer level-35 role. Both roles have
level-35 skill ladders; they are peers, not superiors and
subordinates.

The instinct to "just write the kernel" when a term is stubborn
is understandable and often produces short-term wins. It also
produces a maintenance liability the team cannot support: kernel
authoring skills are scarce, the code is fragile across CUDA
version bumps, and the on-call burden of a home-grown kernel is
high. Escalate.

The corresponding responsibility on the peer track: they do not
own MFU dashboarding, activation-checkpointing choice, or FSDP2
strategy. When the peer engineer proposes "let's raise MFU by 10 pp
by re-writing the RS reducer", that is *your* domain; the
appropriate answer is "here is the trace showing RS is already at
90% overlap; that is not the term to attack".

Chapter 8's job is to make both sides of that conversation
crisp. Chapter 1 through 7 gave you the vocabulary and the
measurements; this chapter gives you the boundary.

## Summary

- This module owns the *integration* surface for MFU
  engineering: FA integration, BF16 policy, FP8 recipe,
  checkpointing granularity, packing, `torch.compile` mode,
  comm-compute overlap.
- `ai-infra-performance-learning` owns the *authoring* surface:
  CUDA / CUTLASS / Triton kernel authoring, FA internals,
  custom FP8 recipes, occupancy tuning, ncu-level analysis.
- The three escalation triggers: memory-bound kernel below the
  roofline, missing fusion for a specific op sequence, or a
  Triton kernel needing a new autotune schedule.
- An escalation ships with an evidence pack: MFU/HFU delta,
  trace, workload shape, reproducer, and the compared
  alternative.
- The reverse direction (consuming a new kernel): peer track
  maintains the kernel, this track maintains the integration.
  Framework-upgrade issues are filed jointly.
- Boundary violations are visible: writing Triton kernels for
  production, forking FA for a new attention variant, editing
  TE's amax logic, or calling CUTLASS directly. Escalate
  instead.
- Both roles are level-35 peers with distinct depth. The
  boundary is a contract, not a hierarchy. Mod-110's platform
  architecture chapter formalises this alongside the other
  cross-track contracts.
