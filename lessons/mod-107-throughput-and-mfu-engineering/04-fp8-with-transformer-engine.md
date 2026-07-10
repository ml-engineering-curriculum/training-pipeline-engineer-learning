# FP8 with NVIDIA Transformer Engine

Chapter 3 gave you the BF16 baseline; the peak FLOPS ceiling on H100
was 989 TFLOPS. This chapter doubles the ceiling to 1979 TFLOPS
(dense) by moving the matmul-heavy fraction of the step to FP8, using
NVIDIA Transformer Engine (TE). The goal is not to reproduce the FP8
math — the paper does that better than this chapter can — but to
teach you what the integration engineer needs to know: which
recipe, which modules, which knobs, what breaks, and how FP8
composes with FSDP2, FA v3, and `torch.compile`.

The two primary references, keep them open:

- **Micikevicius, P., et al. (2022). "FP8 Formats for Deep
  Learning."** arXiv:2209.05433. The E4M3 / E5M2 format definition
  and NVIDIA's tensor-scaling recipes.
- **NVIDIA Transformer Engine documentation and repo.**
  https://docs.nvidia.com/deeplearning/transformer-engine/ and
  https://github.com/NVIDIA/TransformerEngine

## What FP8 gives you, and what it costs

Two formats, per Micikevicius et al., 2022:

| Format | Sign | Exponent | Mantissa | Max value  | Where TE uses it |
|--------|------|----------|----------|------------|-------------------|
| E4M3   | 1    | 4        | 3        | 448        | Forward tensors (activations, weights on forward) |
| E5M2   | 1    | 5        | 2        | 57344      | Backward tensors (gradients) |

The split matches the tensor statistics:

- **Forward tensors** have narrower dynamic range (activations after
  a norm, weights after training). E4M3 (more mantissa, less range)
  gives better precision.
- **Backward tensors** span more decades (gradient magnitudes vary
  wildly across layers and steps). E5M2 (more range, less precision)
  keeps them representable.

The FP8 matmul on Hopper produces an FP32 accumulator: `A_FP8 · B_FP8
→ C_FP32`. You never store an FP8 accumulator. The output is then
either cast down to FP8 for the next layer or kept BF16/FP32 for a
non-FP8 op (LayerNorm, softmax).

Two things you buy:

- **~2× peak arithmetic.** H100 SXM: 989 TFLOPS BF16 dense →
  1979 TFLOPS FP8 dense (NVIDIA H100 datasheet). This is the top of
  the MFU-numerator ceiling.
- **~2× HBM traffic reduction for weights.** The all-gathered
  parameters in the FP8 path are half the bytes, so FSDP2's
  all-gather latency also drops for that fraction of the model.

Two things you pay:

- **Amax bookkeeping.** FP8 needs a *scaling factor* per tensor per
  op per step, tracked over a rolling window of steps. TE calls this
  the "recipe" — most commonly `DelayedScaling`. The overhead of the
  bookkeeping is small in FLOPs but non-trivial in code complexity.
- **Numerical hygiene.** FP8 is not a drop-in dtype change. Loss
  curves can look identical to BF16 for a thousand steps and then
  spike; the recipe has to be right, the excluded modules have to be
  excluded, and the amax history has to be sharded correctly under
  FSDP2.

Typical throughput lift on H100 for a well-integrated dense
transformer, per the TE user guide's benchmark tables: 30–50%
end-to-end MFU delta over BF16, more on very large batch sizes,
less on step recipes dominated by norms/attention. See
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/

## The TE integration surface

TE ships as `pip install transformer_engine`. The three modules you
touch:

- `transformer_engine.pytorch.Linear` — an FP8-aware
  `nn.Linear` drop-in. Same signature, adds FP8-execution paths
  under an `fp8_autocast` context.
- `transformer_engine.pytorch.LayerNormLinear` — a fused
  LayerNorm + Linear that reads the LayerNorm output in FP8
  directly. Removes a BF16 → FP8 cast between LN and the following
  Linear.
- `transformer_engine.pytorch.TransformerLayer` — a fully assembled
  transformer block (self-attn + MLP + norms + residuals) with FP8
  paths on all matmuls. If you already have your own block, use the
  finer-grained modules; if you are starting fresh or porting from a
  vanilla PyTorch model, `TransformerLayer` is the fast path.

Additionally:

- `transformer_engine.pytorch.fp8_autocast(enabled=True,
  fp8_recipe=recipe)` — the context manager that turns FP8 on for
  every TE module inside. Outside this context, TE modules run as
  BF16-only.

A minimum-viable FP8 forward:

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,   # E4M3 forward, E5M2 backward
    margin=0,
    amax_history_len=1024,
    amax_compute_algo="max",
)

class Block(torch.nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.norm1 = torch.nn.RMSNorm(hidden_dim)
        self.attn_qkv = te.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.attn_out = te.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm2 = torch.nn.RMSNorm(hidden_dim)
        self.mlp_up = te.LayerNormLinear(hidden_dim, ffn_dim, bias=False)
        self.mlp_down = te.Linear(ffn_dim, hidden_dim, bias=False)

model = torch.nn.Sequential(*[Block(4096, 14336) for _ in range(32)]).cuda()

# Training step.
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    loss = model(x)
loss.backward()
optimizer.step()
```

Two things are worth calling out:

- The norms are outside TE (`torch.nn.RMSNorm`) but the Linear
  right after them is `LayerNormLinear`, which folds an internal
  norm. Choose one convention per block: either everything is
  vanilla PyTorch norm + `te.Linear`, or you use
  `te.LayerNormLinear` and skip the outer norm.
- The attention itself is not shown. TE ships a fused-attention
  path (FA v3 wrapped for FP8) — use `te.attention.DotProductAttention`
  or wire in the `flash-attn` FP8 path directly. Chapter 2's FA v3
  discussion is what makes this end-to-end FP8.

## The `DelayedScaling` recipe

FP8's dynamic range is narrow. Every FP8 tensor is stored as `(FP8
values) × (FP32 scale)`; the scale is chosen so the tensor's max
absolute value lands near the top of the FP8 range without
saturating. The problem: choosing the scale requires knowing the
tensor's max before you can quantise it, which is a chicken-and-egg
step.

`DelayedScaling` (per Micikevicius et al., 2022, section 4 and the TE
user guide) solves this by using the scales from *past* steps:

- Track the running max absolute value (amax) of every FP8 tensor
  over a rolling window of steps (`amax_history_len`, typically
  1024).
- On step `t`, use the max amax observed in the history window to
  compute a scale that leaves a margin (`margin`, in bits) between
  the tensor's expected max and the FP8 saturation.
- The `amax_compute_algo` selects how to aggregate the history:
  `"max"` uses the maximum observed; `"most_recent"` uses the last
  step's amax.

The recipe knobs and their sensible defaults:

- `fp8_format=Format.HYBRID` — E4M3 forward, E5M2 backward. This is
  the default and the right choice for dense transformer training.
- `margin=0` — no additional headroom bits. Increase to 1 or 2 if
  you observe FP8 overflow amax spikes.
- `amax_history_len=1024` — 1024 steps of amax history. On short
  runs (fine-tuning) drop to 32–128.
- `amax_compute_algo="max"` — conservative; uses the maximum amax
  from the window. `"most_recent"` is faster to react to
  distribution shift but more prone to under-scaling.

TE also ships `MXFP8BlockScaling` and `Float8CurrentScaling`
recipes for newer platforms; on Blackwell (B200) the block-scaling
recipe is expected to be the default. On Hopper, `DelayedScaling` is
the reference.

<!-- needs-research: verify the recommended default recipe for Blackwell B200 once NVIDIA publishes the Blackwell TE user guide, and update the guidance here. -->

## Which ops FP8 replaces, which stay BF16

The rule from the TE docs and Micikevicius et al., 2022:

- **FP8:** the matmul in `Linear` (QKV projection, attention output
  projection, FFN up and down projections). These are the
  matmul-heavy ~90% of transformer FLOPs.
- **BF16 (or FP32):** everything else. LayerNorm / RMSNorm.
  Softmax (inside attention). Residual adds. GELU / SwiGLU
  activation functions. Bias adds.

FA v3 in its FP8 mode extends the FP8 fraction into the two matmuls
inside attention (Q·Kᵀ and softmax·V). If you are on FA v2 or on
Ampere, attention stays BF16 and only the projections around it are
FP8.

## FSDP2 composition: sharding the amax history

The subtlety unique to FSDP2 + FP8: the amax history is a
per-parameter *tensor* (one running max per FP8 tensor, over
`amax_history_len` steps). When FSDP2 shards the parameter, the amax
history has to shard consistently, and cross-rank amax reduction has
to happen once per step.

The good news: TE ships explicit FSDP2 integration. The relevant
patterns from the TE + FSDP2 example
(https://github.com/NVIDIA/TransformerEngine/tree/main/examples):

- **Amax reduction.** After the backward, the amax history has to be
  reduced across the DP group so all ranks agree on next step's
  scale. TE registers hooks that do this reduction on the
  gradient-reduction NCCL communicator. You do not have to wire it
  yourself, but if you build a custom NCCL communicator layout,
  you have to tell TE about it via
  `te.pytorch.fp8.fp8_model_init(fp8_group=my_dp_group)`.
- **Ordering with `fully_shard`.** Apply `te.fp8_model_init` *before*
  `fully_shard`; TE stashes the FP8 group on the parameters, and
  FSDP2 respects the annotation.
- **`param_dtype`.** Set FSDP2's `param_dtype=torch.bfloat16` (per
  chapter 3). TE handles the BF16 → FP8 cast internally on the
  forward under `fp8_autocast`; FSDP2 does not need to know FP8
  exists.
- **Checkpointing.** Distributed Checkpoint (DCP, mod-106) saves the
  FP32 optimizer state and BF16 parameters as usual, and TE's amax
  history is saved as a normal buffer. On resume, TE picks up the
  amax history and continues without needing a warm-up window.

The full pattern:

```python
import transformer_engine.pytorch as te
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
import torch

mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

model = build_transformer_with_te_modules().cuda()

# Register FP8 process group before FSDP2 sharding.
te.pytorch.fp8.fp8_model_init(fp8_group=mesh.get_group("dp"))

# FSDP2 shard.
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
for block in model.blocks:
    fully_shard(block, mp_policy=mp_policy, mesh=mesh)
fully_shard(model, mp_policy=mp_policy, mesh=mesh)

# Training loop uses fp8_autocast.
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    loss = model(x)
loss.backward()
```

## Interaction with `torch.compile`

TE's FP8 modules are `torch.compile`-compatible in recent releases,
but with two caveats:

- **`fp8_autocast` is a context manager.** `torch.compile` traces
  through the context and hoists the FP8 mode into the graph. This
  works for `mode="default"`; some `mode="reduce-overhead"` /
  CUDA-graphs paths have historically had issues with the amax
  update. Check the TE release notes.
- **Dynamic shapes and FP8 amax bookkeeping.** The amax history is
  keyed per FP8 tensor; if `torch.compile` recompiles due to a
  shape change (e.g., last-batch-in-epoch shorter than the rest), the
  amax bookkeeping stays consistent because it is state on the TE
  module, not on the compiled graph.

<!-- needs-research: confirm the latest torch.compile + TE compatibility matrix from the NVIDIA/TransformerEngine release notes before quoting specific version support in a run report. -->

Chapter 6 goes deeper on `torch.compile` composition.

## Interaction with FlashAttention v3

FA v3 (chapter 2) accepts FP8 QKV inputs and produces an FP8 output.
The pipeline stays fully FP8 through the transformer block:

- QKV projection (FP8 Linear) → Q/K/V FP8 tensors.
- FA v3 (FP8 mode) → attention output FP8 tensor.
- Output projection (FP8 Linear) → residual add (upcast to BF16).
- LayerNorm (BF16 or FP32) → FP8 recast for the next block.

This is the maximum-FP8 configuration on H100 and reproduces the
throughput ceiling the FA v3 paper and TE user guide quote. If you
are on FA v2, the attention step stays BF16 and the whole block's
throughput is bottlenecked there; the FP8 lift shrinks accordingly.

## Debugging FP8

The three symptoms and their diagnoses:

- **Loss spikes ~500 steps in.** FP8 amax history was warm (took
  its first `amax_history_len` steps to fill the window) and the
  saturation started biting. Fix: increase `margin` by 1, or
  raise `amax_history_len`, or switch a specific layer back to BF16.
- **NaNs from a specific block.** One layer's FP8 amax is
  overflowing. Instrument
  `te.pytorch.fp8_get_fp8_context_id()` output and inspect the
  per-tensor amax logs the TE debug flag prints
  (`NVTE_DEBUG=1`).
- **MFU delta from FP8 is smaller than 20%.** Something on the
  non-FP8 path is a bottleneck. Common cause: attention is still
  BF16 (FA v2). Fix: upgrade to FA v3 or accept the ceiling.

The TE user guide's "Debugging" section documents `NVTE_DEBUG` and
its subsystems.

## The scope boundary

This chapter is the integration engineer's view of FP8. What is
*not* here:

- The FP8 math derivation (Micikevicius et al., 2022).
- The tensor-core WGMMA FP8 kernel implementation (see FA v3 paper
  and NVIDIA CUTLASS).
- The trade-off between `DelayedScaling`, `Float8CurrentScaling`,
  and `MXFP8BlockScaling` at the numerical-analysis level (TE user
  guide, and Blackwell-generation whitepapers).

If you find yourself needing to author a new recipe (say, a
sub-tile block-scaled variant that TE does not ship), that is a
peer-track hand-off to `ai-infra-performance-learning`. Chapter 8
formalises the boundary.

## Summary

- FP8 doubles the peak arithmetic on H100 (989 → 1979 TFLOPS dense)
  by moving matmuls to E4M3 (forward) / E5M2 (backward). Everything
  else — norms, softmax, residuals — stays BF16.
- NVIDIA Transformer Engine (github.com/NVIDIA/TransformerEngine) is
  the reference integration. Use `te.Linear`,
  `te.LayerNormLinear`, or `te.TransformerLayer` as
  drop-ins, and wrap the training step in `te.fp8_autocast(recipe)`.
- `DelayedScaling` is the standard Hopper recipe (Micikevicius et
  al., 2022). Defaults: HYBRID format, `amax_history_len=1024`,
  `amax_compute_algo="max"`, `margin=0`. Blackwell prefers block
  scaling.
- FSDP2 + FP8 works. Call `fp8_model_init(fp8_group=...)` before
  `fully_shard`. Amax history reduces across the DP group;
  Distributed Checkpoint saves it transparently.
- FA v3 FP8 mode extends the FP8 fraction into attention. FA v2
  keeps attention BF16 and caps the FP8 lift.
- `torch.compile` composes with TE in recent releases; verify
  against the TE + PyTorch version pair. Chapter 6 details the
  compilation story.
- Expected end-to-end throughput lift over BF16: 30–50% on a
  well-integrated H100 stack.
- Kernel authoring (new recipes, custom scaled matmuls) belongs to
  the peer track. Chapter 8 owns the escalation contract.
