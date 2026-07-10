# MFU: Calculation and the Gap to Peak

Every optimisation in this module — FlashAttention, BF16, FP8,
activation checkpointing, `torch.compile`, comm-compute overlap — is
justified by moving one number: MFU, Model FLOPs Utilization. This
chapter defines MFU precisely, walks through the instrumentation you
need to compute it on a real run, and decomposes the *gap* between
what you measure and the datasheet peak into named contributions. The
subsequent chapters each attack one of those gap terms; this chapter
is the accounting frame that keeps them honest.

If you cannot compute MFU on your training run by the end of this
chapter, do not open chapter 2 — no downstream optimisation is
measurable without this number.

## What MFU actually measures

MFU is defined in the PaLM paper (Chowdhery et al., 2022,
arXiv:2204.02311, section 5.1) as:

```
MFU = achieved_model_FLOPs_per_second / peak_hardware_FLOPs_per_second
```

Two things are worth pinning down immediately.

- **The numerator counts *model* FLOPs, not *hardware* FLOPs.** The
  numerator is a theoretical count of the forward-plus-backward FLOPs
  the *model* would do at the mathematical description of the forward
  pass. It ignores anything the *implementation* adds — activation
  recomputation, redundant matmuls from a suboptimal fusion, wasted
  FLOPs from padded sequences. If your implementation recomputes
  activations, those recomputed forward FLOPs do *not* count in the
  numerator; they are pure overhead.
- **The denominator is the datasheet peak for the specific tensor
  dtype you are training in.** BF16 peak and FP8 peak are different
  numbers on the same GPU. If you are training in BF16, use BF16
  peak; if you have switched a subset of matmuls to FP8, MFU is
  ambiguous unless you commit to one denominator and document it.

PaLM also introduces a companion metric, **HFU** (Hardware FLOPs
Utilization), which uses the actual FLOPs executed by the hardware
(including recomputed activations, TP-induced redundant computation,
padding) in the numerator. HFU is always ≥ MFU. The two together tell
you two different stories:

- **HFU** tells you how well the *kernels* are being kept fed.
- **MFU** tells you how well the *model* is being trained per unit of
  hardware.

The gap between HFU and MFU is exactly the cost of things your
implementation does but the model does not need. Activation
checkpointing is the biggest lever there: a run with full activation
recomputation might see HFU 55% and MFU 45%, with the 10 pp gap being
the recomputation forward pass.

Chapter 5 revisits this trade-off explicitly.

## The `6 · N · D` FLOP count

The numerator's "model FLOPs" is not folklore — it comes from Kaplan
et al., 2020, "Scaling Laws for Neural Language Models"
(arXiv:2001.08361, Appendix B). For a dense decoder-only transformer
with `N` non-embedding parameters trained on `D` tokens, the
forward-plus-backward FLOP count is:

```
C ≈ 6 · N · D
```

The `6` breaks down as `2 · 3`:

- `2 FLOPs` per multiply-accumulate (one multiply and one add).
- `3` because backward is roughly `2×` the forward FLOPs
  (activation-gradient pass and weight-gradient pass), so forward +
  backward ≈ `1 + 2 = 3` times the forward-only matmul cost.

Per training *step* on a global batch of `B_tokens` tokens:

```
FLOPs_per_step ≈ 6 · N · B_tokens
```

Two adjustments you should be prepared to defend in a design review:

1. **Attention is not `6ND`.** The `6ND` count treats attention as
   linear in `N`. The attention matmuls (Q·Kᵀ and softmax·V) scale
   quadratically in sequence length, not linearly in parameters, so at
   long sequence lengths they add a second term:

   ```
   FLOPs_attn ≈ 6 · L · s · h · s      (per layer, per token pair)
   ```

   where `L` is layers, `s` is sequence length, `h` is hidden size. For
   `s` in the low thousands and dense models of a few billion
   parameters, the `6ND` term dominates and you can round attention
   into "counted separately or ignored." For long-context runs
   (`s ≥ 32k`), track the attention term explicitly. The
   Chinchilla paper (Hoffmann et al., 2022, arXiv:2203.15556)
   discusses this correction in its scaling-law appendix.
2. **Embedding parameters are outside `N`.** Kaplan's `N` is
   "non-embedding parameters." Include the embedding table only if you
   are consistent about it across the numerator and any downstream
   comparisons. Reproductions that quietly include or exclude the
   embedding table differ by a few percent, which matters when you are
   comparing an MFU claim in a paper to your own run.

## Peak FLOPs by device and dtype

The denominator numbers you should have memorised for the H100 SXM,
per NVIDIA's H100 datasheet (nvidia.com/en-us/data-center/h100/, "H100
SXM5 80GB" specifications):

| GPU               | Dtype      | Peak (TFLOPS) |
|-------------------|-----------|---------------|
| H100 SXM          | FP64      | 67            |
| H100 SXM          | TF32      | 989 (with sparsity 1979) |
| H100 SXM          | BF16 / FP16 | 989 (dense; 1979 with 2:4 sparsity) |
| H100 SXM          | FP8       | 1979 (dense; 3958 with 2:4 sparsity) |
| A100 SXM 80GB     | BF16 / FP16 | 312         |
| A100 SXM 80GB     | TF32      | 156           |

<!-- needs-research: verify H200 SXM peak-FLOPS numbers against NVIDIA's H200 datasheet before quoting them in a run report. -->

Three rules for using these numbers honestly:

- **Use dense, not sparse.** NVIDIA quotes both "dense" and "with
  structured sparsity 2:4" numbers on the datasheet. Training runs do
  not use 2:4 structured sparsity by default; use the dense number.
- **Use the dtype you are actually running.** BF16 peak on H100 is
  989 TFLOPS. If you switch a subset of matmuls to FP8 via Transformer
  Engine (chapter 4), you either pick one denominator and stay honest
  about which matmuls are actually FP8, or you compute two MFUs (one
  BF16-denominator, one FP8-denominator) and reconcile.
- **These are per-GPU.** Total cluster peak is `peak_per_GPU · N_GPUs`.
  For a 512-H100 cluster in BF16 the denominator is
  `512 · 989 TFLOPS = 506 PFLOPS`.

## The instrumentation you need

To compute MFU on a real run you need four numbers per step:

- `N` — non-embedding parameter count.
- `B_tokens` — tokens processed per step (global batch, summed across
  DP × TP × PP × CP replicas, at the tokens-per-step granularity your
  loss is computed at).
- `t_step` — wall-clock seconds per step, averaged over enough steps
  to smooth out the first few warm-up iterations and any
  checkpoint-save spikes.
- `peak_per_GPU` — datasheet peak for the training dtype.

Then:

```python
import torch

def count_non_embedding_params(model: torch.nn.Module) -> int:
    total = 0
    for name, p in model.named_parameters():
        if "embed" in name.lower() or "lm_head" in name.lower():
            continue
        total += p.numel()
    return total

N = count_non_embedding_params(model)  # local rank; sum across TP if sharded

# Loop-level instrumentation.
step_times = []
tokens_per_step = global_batch_size * seq_len  # or sum(len(seq)) for packed

for step, batch in enumerate(loader):
    if step >= warmup_steps:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    loss = train_step(model, batch)

    if step >= warmup_steps:
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

t_step = sum(step_times) / len(step_times)
flops_per_step = 6 * N * tokens_per_step
achieved_flops_per_sec = flops_per_step / t_step

peak_flops_per_gpu = 989e12  # BF16 H100 SXM
peak_flops_total = peak_flops_per_gpu * world_size

mfu = achieved_flops_per_sec / peak_flops_total
print(f"MFU = {mfu:.3f}")
```

Six subtleties that will bite you if you skip them:

- **`torch.cuda.synchronize()` before both `t0` and `t1`.** Without
  synchronisation, Python's `time.perf_counter()` measures kernel
  launch, not kernel completion. Every published MFU number that
  disagrees with a torch-profiler measurement fails this test.
- **Warm-up matters.** The first several iterations pay for CUDA
  graph construction, autotuning (cuBLAS heuristics, TorchInductor,
  Triton autotuning), and any deferred CUDA-context initialisation.
  Skip at least the first 20 steps.
- **Count `B_tokens` correctly with packed sequences.** If you packed
  eight short sequences into one 4096-token document (chapter 5), you
  processed 4096 tokens, not eight sequences worth. Instrumentation
  that counts "batch size × seq_len" over-reports tokens whenever
  padding is present.
- **Sum `N` correctly under TP.** Under Megatron-style tensor
  parallel, each rank owns only a shard of the parameters, so
  `sum(p.numel() for p in local params)` returns `N / TP`. Multiply
  back up or use `p.full_tensor().numel()` for DTensor parameters
  (mod-102 chapter 2). Under FSDP2, the same care is needed but the
  DTensor API makes it explicit.
- **Optimizer step time counts.** `t_step` is the wall-clock from
  `t0` at the start of `forward` to `t1` at the end of
  `optimizer.step()`, not just the forward+backward. If you
  measure only forward+backward and divide by that, you are computing
  a fictional "compute MFU" that overstates the training system.
- **PP bubble time counts.** Under pipeline parallel, `t_step`
  includes bubble time (mod-101 chapter 4). MFU accounts for it
  automatically because it lands in `t_step`; do not "correct it
  out" — the bubble is a real cost of your strategy.

## Sanity numbers you should have internalised

Order-of-magnitude MFU numbers reported in public work. Treat them as
targets, not entitlements.

| Class                          | Reported MFU (approx.)            |
|--------------------------------|-----------------------------------|
| PaLM 540B (TPU v4)             | ~46% (Chowdhery et al., 2022)     |
| MegaScale (Jiang et al., 2024) | 55.2% on Llama 175B, arXiv:2402.15627 |
| Llama 3 405B pretraining       | ~38–41% BF16 on H100              |
| A "tuned" open-source stack    | 40–55% BF16 on H100               |
| A "first-launch" open stack    | 15–30% BF16 on H100               |

<!-- needs-research: cross-check the Llama 3 MFU number against the Llama 3 herd-of-models paper's specific 4-D-parallel throughput tables. -->

If your first-launch MFU is 20% and you got to 45% after this module,
you are on the mainstream trajectory. If your first-launch MFU is 60%,
either you are on TPU v5 with well-tuned XLA or your instrumentation
is wrong; check `torch.cuda.synchronize()` first.

## Decomposing the gap to peak

An MFU of 40% means you are leaving 60 percentage points on the
floor. The gap decomposes, roughly additively, into named
contributions. Each subsequent chapter attacks one of them.

| Gap term                              | Typical cost | Owned by      |
|---------------------------------------|--------------|---------------|
| Attention kernel overhead             | 5–15 pp      | Chapter 2 (FlashAttention v2/v3) |
| Sub-optimal dtype (FP32 forward)      | 10–30 pp     | Chapter 3 (BF16) / chapter 4 (FP8) |
| Activation recomputation forward FLOPs | 5–15 pp     | Chapter 5 (checkpointing granularity) |
| Padded / short sequences              | 5–20 pp      | Chapter 5 (sequence packing) |
| Python-side dispatch / graph breaks   | 5–15 pp      | Chapter 6 (`torch.compile`) |
| Communication exposed on the critical path | 5–25 pp | Chapter 7 (overlap) |
| Kernel-level inefficiency (memory-bound matmul, cache thrash) | 2–10 pp | Peer track `ai-infra-performance-learning`; chapter 8 |

There are three ways to attribute a gap term:

1. **A/B measurement.** Turn the optimisation off, measure MFU, turn
   it on, measure again. Delta is the term. This is the gold
   standard when it is feasible (e.g., `attn_implementation="eager"`
   vs. `"flash_attention_2"`).
2. **Kineto / Nsight trace inspection.** Read a torch profiler or
   `nsys` trace and see the exposed time explicitly. Chapter 7
   details the trace patterns. Comm exposure and Python-dispatch
   overhead are almost always attributed this way.
3. **Analytic.** Compute the expected cost of the term. For example,
   full-block activation checkpointing adds one forward pass, so the
   theoretical HFU–MFU gap is `~1/3` (the forward is `1/3` of
   forward+backward, so the recomputed forward is `~33%` extra FLOPs).
   Chapter 5 does this arithmetic.

Any gap term you cannot attribute by at least one of these three
methods is a red flag. In practice the residual "unexplained gap" on
a well-tuned run is 3–8 pp; if it is larger, something is happening
you have not modelled.

## The module contract

Every subsequent chapter in this module attacks one specific gap term:

- **Chapter 2 — FlashAttention v2/v3.** Closes the attention kernel
  overhead term. Also raises the ceiling on long-context runs where
  the quadratic-in-`s` attention FLOPs would otherwise dominate.
- **Chapter 3 — BF16 across FSDP2.** Closes the "sub-optimal dtype"
  term at the FP32-forward → BF16-forward jump, without needing FP8
  hardware.
- **Chapter 4 — FP8 via Transformer Engine.** Doubles the peak
  denominator on Hopper: the *ceiling* moves from 989 TFLOPS
  (BF16) to 1979 TFLOPS (FP8) for the matmul-heavy fraction of the
  step.
- **Chapter 5 — Activation checkpointing and sequence packing.**
  Manages the memory–FLOPs trade so the batch size can grow (raising
  the MFU denominator's *usefulness*, not the peak) and eliminates
  padding waste.
- **Chapter 6 — `torch.compile` and FSDP2.** Removes the
  Python-dispatch and graph-break contribution.
- **Chapter 7 — Communication-compute overlap.** Closes the exposed
  comm term — the single largest MFU contribution at multi-node
  scale.
- **Chapter 8 — Boundary with the performance-engineer track.**
  Names what stays in-house and what to escalate when the residual
  gap is kernel-level.

Every one of those chapters ends with a measurement you should be
able to add to a run report: "we integrated X, MFU went from A to B,
here is the trace evidence."

## Summary

- MFU = `achieved_model_FLOPs / peak_hardware_FLOPs`. Numerator is
  the mathematical `6 · N · D` count (Kaplan et al., 2020, Appendix
  B). Denominator is the datasheet peak for the *dtype you are
  actually running in*.
- H100 SXM: 989 TFLOPS BF16 dense, 1979 TFLOPS FP8 dense (NVIDIA
  H100 datasheet). Cluster peak is per-GPU × world size.
- HFU vs. MFU (PaLM paper). HFU counts recomputed FLOPs; MFU does
  not. The HFU–MFU gap is the recomputation cost.
- Instrument tokens/step, seconds/step, non-embedding param count,
  and dtype. Synchronise CUDA before and after `t_step`; skip
  warm-up; count packed-sequence tokens correctly.
- Public reference points: PaLM ~46%, MegaScale 55%, a tuned open
  H100 stack 40–55% BF16. First launches often start at 15–30%.
- The gap to peak decomposes into: attention overhead, sub-optimal
  dtype, activation recomputation, padding, Python dispatch, exposed
  comm, and kernel inefficiency. This module attacks the first six;
  the seventh is `ai-infra-performance-learning`'s domain.
- Every subsequent chapter closes one term. Chapter 8 draws the
  boundary with the performance-engineer track.
