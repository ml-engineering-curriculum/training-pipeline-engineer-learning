# MFU and the Gap to Peak

Every previous module in this track let you build correctness — the
model converges, the checkpoint survives a crash, the fabric does not
drop. This module is about the *efficiency* number that lands on the
capacity-planning slide: what fraction of the theoretical peak of the
hardware you actually turned into training tokens. That number is
**MFU** (Model FLOPs Utilization). It is the yardstick every other
chapter in this module optimizes against.

The first job of this chapter is to nail down the definition. The
second is to fix the two numerator/denominator conventions that get
mixed up in the wild and cause plausible-looking MFU numbers that
disagree by 20% across teams. The third is to build the "gap-to-peak
budget" — the small table that tells you *where* your remaining
percentage points went and *which* of the later chapters is the tool
that gets them back.

## Why MFU is the right yardstick

Throughput alone (tokens per second) is not comparable across models
or hardware. A 7B model at 200k tokens/s on 64 H100s is not
straightforwardly better or worse than a 70B model at 40k tokens/s on
the same fleet — one might be at 55% MFU and the other at 20%. The
raw throughput number tells you neither how close you are to the
hardware ceiling nor how much headroom the next optimization can
recover.

PaLM (Chowdhery et al., 2022, arXiv 2204.02311, appendix B) introduced
MFU precisely to make cross-model, cross-hardware comparisons honest:

> **Model FLOPs utilization (MFU)** is the ratio of the observed
> throughput (tokens-per-second) to the theoretical maximum throughput
> of a system operating at peak FLOPs.

The two anchors are the *observed throughput* (the numerator) and the
*peak FLOPs* of the accelerator at the precision you actually train
in (the denominator). If you write both down correctly, MFU has three
useful properties:

- It is dimensionless and hardware-normalized: 40% MFU on an A100 and
  40% MFU on an H100 mean the same thing about "how well the code
  uses the silicon". The absolute tokens/s differ; the efficiency does
  not.
- It bounds the return on any single optimization. If you are already
  at 55% MFU, no single kernel swap gets you to 110%. The remaining
  gap gates how much any one chapter of this module can buy you.
- It surfaces regressions that raw throughput hides. A framework
  upgrade that adds a 5% overhead to the step but ships on a fabric
  1.1× faster looks like a wall-clock win and an MFU loss — the loss
  is the one that compounds.

## The formula, precisely

The numerator is the *model FLOPs per training step* — the count of
floating-point operations the model *definition* requires to compute
one training step at your batch size and sequence length. It does not
include operations the implementation adds (recomputation for
activation checkpointing, extra math in a Triton kernel that pads to a
tile boundary). Model FLOPs is a property of the model, not of the
kernel.

For a dense decoder-only Transformer with `N` non-embedding
parameters, `B` global batch size (in sequences), and sequence length
`S`, one forward+backward pass costs approximately

```
F_step ≈ 6 · N · B · S  +  attention_flops(B, S)
```

The `6·N·B·S` term is the standard dense-Transformer accounting from
Kaplan et al. (2020, arXiv 2001.08361): every non-embedding weight
does 2 FLOPs per token in the forward pass and 4 FLOPs per token in
the backward pass (one for the input gradient, one for the weight
gradient — each is a matmul the same size as the forward one). It
already covers all the MLP and attention-projection matmuls, because
those weights are counted in `N`.

The `attention_flops` term is the softmax/QK^T/AV block. For standard
multi-head attention across `L` layers with `H` heads and per-head
dimension `d_head`, one training step over `B` sequences of length `S`
adds roughly `12 · L · H · d_head · B · S²` FLOPs (the `12` is
`2 · (QK^T + softmax·V) · (forward + 2·backward)`; different papers
count these constants slightly differently, so verify against your
own model card before publishing an MFU number). This term is
quadratic in `S` and becomes the dominant cost for long-context
training even if `6·N·B·S` looks like the whole story.

If your training uses grouped-query attention, MoE, or a
non-Transformer architecture, redo the counting from the layer
definitions — do *not* copy the dense formula and hope. The most
common MFU mis-reports in the field come from applying `6·N` to an
MoE where only the active experts should be counted, or from
forgetting the quadratic attention term at long context.

The denominator is straightforward:

```
F_peak ≈ G · P · t_step
```

where `G` is the number of accelerators in the training run, `P` is
the vendor-published peak FLOPs of one accelerator *at the precision
you are training the matmuls in*, and `t_step` is measured wall-clock
step time in seconds. `P` numbers to keep close:

- NVIDIA A100 (SXM4, no sparsity): 312 TFLOP/s BF16/FP16.
- NVIDIA H100 (SXM5, no sparsity): 989 TFLOP/s BF16/FP16, 1979
  TFLOP/s FP8. Source: NVIDIA H100 Tensor Core GPU datasheet.
- NVIDIA H200 (SXM5): same math throughput as H100 at higher HBM
  bandwidth — MFU numerator identical, denominator identical.
- NVIDIA B200 (Blackwell): consult the current Blackwell datasheet;
  the FP4/FP8/FP16 numbers moved substantially from Hopper.

Always use the **dense** peak, not the sparsity-doubled marketing
number. Training does not use structured sparsity; the 2× "with
sparsity" figure is inference-only marketing. Dividing by that number
gives you a plausible 30% MFU that is actually 60%.

Then

```
MFU = F_step / (F_peak · G · t_step)
     = (6·N·B·S + attention) / (G · P · t_step)
```

That is the number to publish. Any other formulation belongs in an
appendix or a footnote.

## HFU: the sibling metric, and when to use it

If you have activation-checkpointed the model, some of the FLOPs the
GPU actually executes are recomputations — not model FLOPs. Recompute
grows your executed FLOP count by roughly 33% for full activation
checkpointing (an extra forward pass per checkpointed segment
during backward) and by a smaller fraction for selective schemes. To
separate "the code is efficient" from "the model requires
recomputation", the community uses a second metric:

```
HFU (Hardware FLOPs Utilization) = executed_FLOPs / (G · P · t_step)
```

HFU counts every FLOP the GPU actually performed (including
recomputation). It is always ≥ MFU. Two useful facts:

- **HFU tells you if your kernels are good.** If HFU is 65% on H100
  BF16, your matmul kernels, dtypes, and shapes are near optimal for
  the hardware. If HFU is 25%, you have a kernel or memory-bandwidth
  problem, not an algorithmic one.
- **MFU tells you if the model+recompute+overlap is good.** MFU can
  be much lower than HFU if you are re-executing a lot of
  computation. Whether the gap is acceptable is a memory/throughput
  trade-off chapter 5 formalizes.

Publish both when you can. If you only publish one, MFU is the number
that matters to capacity planning; HFU is the number that matters to
"is the kernel author done".

## The gap-to-peak budget

An MFU number is only actionable if you can decompose the missing
`(1 - MFU)`. The gap-to-peak budget is a small table that attributes
the shortfall to specific causes. Filling it in *before* optimizing
is what keeps this module from being a random walk through kernels.

The five buckets, in order of usual size for a new setup:

1. **Kernel and dtype gap.** Are matmuls running in the intended
   precision (BF16, FP8) and hitting near-vendor-peak on their shape?
   Attention is the largest single kernel; FlashAttention v2/v3
   (chapter 2) is usually the first several points of MFU. Non-tensor-
   core ops (layernorms, elementwise, softmax) eat cycles too, but
   individually smaller — chapter 6's `torch.compile` targets these.
2. **Memory-bandwidth gap.** Model FLOPs that are memory-bound (small
   matmuls, layernorm, softmax reduction) do not use the tensor cores
   at all. Their upper bound is HBM bandwidth, not FLOPs. If your
   arithmetic intensity is low, no kernel swap saves you — the fix is
   to fuse (chapter 6) or reshape (larger micro-batch, sequence
   packing, chapter 5).
3. **Recomputation gap.** Full activation checkpointing costs ~33% of
   the executed forward FLOPs, which shows up as HFU > MFU. Selective
   AC costs less. Chapter 5 quantifies the trade-off.
4. **Communication gap.** Any all-reduce, all-gather, or reduce-scatter
   the GPU spent waiting on instead of computing is directly subtracted
   from MFU. Chapter 7 is about reducing the *exposed* portion of
   communication by overlapping it with compute; mod-101 chapter 5 and
   mod-105 chapter 5 are the underlying references.
5. **Everything else** — loader stalls, checkpoint writes on the
   critical path (mod-106 chapter 2), Python overhead per step,
   scheduler jitter, thermal throttling. These are usually small on a
   healthy platform, but they compound; a 1% loss to a hot loader plus
   1% to a slow checkpoint plus 1% to Python overhead is 3% you will
   not spot in any single profile.

The rest of the module reduces to: "measure the gap, attribute it to
one of these buckets, apply the chapter that owns that bucket, re-
measure". Chapter 8 puts the boundary with the ai-infra-performance-
engineer role on top of this budget — you own applying the buckets;
they own authoring the kernels that make bucket 1 land.

## A worked example, end-to-end

Assume you are training a 30B-parameter dense decoder-only Transformer
on 128 H100 SXM5 GPUs in BF16, with a global batch size of 1024
sequences at sequence length 4096. You measure a mean step time of
`t_step = 3.20 s` on your instrumented training loop.

Numerator (model FLOPs per step):

- `6·N·B·S = 6 · 30e9 · 1024 · 4096 ≈ 7.55 · 10^17` FLOPs.
- Attention (assume `L·H·d_head` gives ~5 · 10^9 per token per layer,
  standard for a 30B decoder-only): `12 · L · H · d_head · B · S² ≈
  ~1 · 10^17` FLOPs. This is a rough attention estimate; use your
  model's true shapes for a real report.
- Total `F_step ≈ 8.5 · 10^17` FLOPs.

Denominator (theoretical peak in the same 3.20 s window):

- `G · P · t_step = 128 · 989e12 · 3.20 ≈ 4.05 · 10^17` FLOPs.

Wait — the numerator is *larger* than the denominator, which would
give an MFU > 1. That is the diagnostic: either the step time is
mismeasured, the parameter count includes embeddings that should not
have been counted, or the attention term is wrong. Recheck. In this
example, if the correct step time is actually `t_step = 6.40 s` (double
what was reported because the harness was averaging over half-steps),
`F_peak = 8.10 · 10^17` and `MFU ≈ 8.5/8.1 · 100% ≈ 105%` — still
> 1. Which means either the FLOP count is high (probably: the `12` in
the attention term counts backward twice; verify) or the parameter
count is off. Push through this arithmetic on real numbers before
publishing any MFU report; the failure mode is almost never "MFU
looks low because the model is slow", it is "the FLOP count is
wrong".

For a corrected 30B example that lands, the community range on H100
BF16 dense training with FlashAttention v2 and reasonable
sequence-length packing sits roughly in the 40–55% MFU band (see
Grattafiori et al., 2024, "The Llama 3 Herd of Models", arXiv
2407.21783, section 3.4 for one published anchor; the paper reports
observed BF16 MFU during Llama 3 pretraining). Below 30% and you have
low-hanging fruit; above 55% on Hopper without FP8 and you should
double-check your denominator.

## What this chapter is not

- **A kernel benchmarking guide.** That is chapter 2 (FlashAttention)
  and chapter 6 (torch.compile + Triton). The MFU formula does not
  care which kernel you use; it just tells you how much room a better
  kernel has.
- **A replacement for a profile.** MFU tells you "how much" you are
  leaving on the table; a Nsight Systems / `torch.profiler` trace
  tells you *where*. Both matter; MFU without a profile is a number,
  a profile without MFU is a list.
- **A dollar figure.** Cost-per-token and MFU are related but not the
  same — a cluster at 40% MFU on H100 might be cheaper per token than
  one at 60% MFU on A100 depending on lease terms. mod-109 owns that
  translation.

## Summary

- MFU is `F_step / (G · P · t_step)` with `F_step` = the FLOPs the
  model *definition* requires (dense-Transformer accounting: `6·N·B·S`
  plus a quadratic attention term) and `P` = the *dense* vendor-peak
  at the precision you train the matmuls in. Use these numbers,
  publish them, and cite the datasheet you used for `P`.
- HFU is the same denominator with an executed-FLOP numerator. It
  isolates "kernel efficiency" from "recompute cost". Publish both
  when possible.
- Common range for a well-tuned dense LLM on Hopper in BF16 sits in
  the 40–55% band (Llama 3 report is a public anchor). Above 55% on
  Hopper BF16 without FP8, double-check your numerator.
- The five gap buckets — kernel/dtype, memory bandwidth, recompute,
  communication, everything-else — map one-to-one onto the chapters
  that follow. Fill the budget before optimizing.
- Do the arithmetic before publishing an MFU number. An MFU > 100% is
  a FLOP-counting bug, not a hardware miracle.
