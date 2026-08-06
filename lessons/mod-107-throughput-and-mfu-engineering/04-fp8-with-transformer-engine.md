# FP8 with NVIDIA Transformer Engine on H100

Hopper's tensor cores expose FP8 at roughly 2× the BF16 FLOP peak.
On paper that is a 2× MFU headroom; in practice, done correctly, it
lands as a 30–50% end-to-end step-time win for the matmul-dominated
parts of an LLM. Done incorrectly, it silently degrades convergence.
This chapter is about applying it correctly through NVIDIA
Transformer Engine (TE) — the library that owns the scale-factor
bookkeeping FP8 requires.

Prerequisite: chapter 3's BF16 policy is measured, stable, and
working. Do not enable FP8 on top of a shaky BF16 policy.

## Why FP8 needs a library, not a flag

BF16 has FP32's exponent range and needs no dynamic scale management
— you can drop-in swap FP32 for BF16 and the numerics survive. FP8
does not. Its two variants:

- **E4M3** (1 sign, 4 exponent, 3 mantissa): max ~448, min ~2e-3.
  Used for forward activations and weights where the value range is
  bounded.
- **E5M2** (1 sign, 5 exponent, 2 mantissa): max ~6e4, min ~6e-5.
  Used for gradients, where dynamic range dominates precision.

Neither range covers a typical Transformer's activation or gradient
magnitudes end-to-end. To use FP8, each tensor's values must be
*scaled* into the FP8 representable range on the way in and *un-
scaled* on the way out, and the scale factor must be recomputed each
step because activation and gradient magnitudes evolve during
training. That bookkeeping — the choice of scaling recipe, the
history-based smoothing, the fused kernel launches — is what TE
provides. Writing it yourself is not the return-on-effort trade you
want; that is the ai-infra-performance-engineer role (chapter 8).

Primary sources:

- **Micikevicius et al., 2022. "FP8 Formats for Deep Learning."**
  arXiv 2209.05433. The paper that defined E4M3 and E5M2.
- **NVIDIA H100 Tensor Core GPU Architecture** whitepaper (from the
  NVIDIA H100 product page,
  https://www.nvidia.com/en-us/data-center/h100/). Describes the
  Hopper FP8 tensor-core path.
- **NVIDIA Transformer Engine documentation.**
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/.
  The canonical reference for FP8 integration.
- **Transformer Engine on GitHub.** https://github.com/NVIDIA/TransformerEngine.
  The library itself, with PyTorch, JAX, and TensorFlow bindings.

## What TE actually does

TE ships two things you consume:

- **`transformer_engine.pytorch` module.** Drop-in replacements for
  common Transformer sub-modules: `te.Linear`, `te.LayerNorm`,
  `te.LayerNormLinear`, `te.LayerNormMLP`, `te.TransformerLayer`,
  `te.DotProductAttention`. These wrap FA v3 for attention and emit
  FP8 matmuls under a scoped context manager.
- **`te.fp8_autocast(...)` context.** A context manager that, when
  entered, tells the TE modules inside it to use FP8 for their
  matmul dtypes. Outside the context, the same modules run in
  whatever `param_dtype` was configured (BF16 in the usual
  policy).

The pattern is:

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,          # E4M3 fwd, E5M2 bwd/grads
    amax_history_len=1024,
    amax_compute_algo="max",
)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    logits = model(input_ids)          # matmuls inside TE modules run in FP8
loss = loss_fn(logits, labels)         # loss in FP32 as usual
loss.backward()                        # gradients as usual; TE knows FP8 backward
```

Key knobs on the recipe:

- **`fp8_format=Format.HYBRID`** is the standard choice: E4M3 for
  forward activations and weights, E5M2 for gradients. The
  alternative `Format.E4M3` (all E4M3) is only appropriate for
  inference; for training use HYBRID.
- **`amax_history_len`** controls how many past steps' absolute-max
  values TE keeps to smooth the per-tensor scale. Longer history →
  more stable scale, slower to adapt to a distribution shift.
  Defaults (~16 in some releases, longer in others) are usually
  fine.
- **`amax_compute_algo`** picks between `max` (peak of history) and
  `most_recent` (last-step's amax only). `max` is the safer default.

The library's `DelayedScaling` recipe is the one to start with. The
newer `MXFP8`-style block-scaling recipes (see the TE user guide for
the current release matrix on Blackwell hardware) target Blackwell
tensor cores; on Hopper stick with DelayedScaling.

## The mixed-precision policy with FP8 layered on

FP8 is added to the chapter-3 BF16 policy; it does not replace any
piece of it. The correct layered policy is:

- **Parameter storage (FSDP2 `param_dtype`):** BF16. Not FP8. FP8
  parameter *storage* is a research topic; TE does not use it by
  default.
- **All-gather traffic (FSDP2 forward comm):** BF16. FP8 all-gather
  is a research topic; in the standard TE path, the sharded params
  are all-gathered in BF16 and *then* cast to FP8 by TE inside its
  matmul.
- **Matmul precision inside `te.fp8_autocast`:** FP8 (E4M3 fwd, E5M2
  bwd).
- **Activations returned by TE modules:** BF16. TE unscales on the
  way out, so downstream code sees BF16.
- **Gradient reduce (FSDP2 `reduce_dtype`):** FP32 (or BF16 on small
  clusters; the chapter-3 discussion carries over).
- **Optimizer state:** FP32 master weights. TE has FP8-optimizer
  research work; for a production run, keep the optimizer in FP32.

If you visualize the tensor as it flows through a layer: BF16
parameter in HBM → all-gathered BF16 → TE casts to E4M3 for the
matmul → tensor-core matmul in FP8 → TE unscales the output to BF16
→ next layer sees BF16.

The delta from chapter 3 is exactly: matmuls happen in FP8 instead
of BF16 for TE-wrapped modules. Everything else is the same.

## Wiring TE into an FSDP2 model

The correct order of composition is: replace your `nn.Linear` and
`nn.LayerNorm` etc. with the TE analogues, *then* wrap with FSDP2's
`fully_shard`. TE modules are `nn.Module`s and FSDP2 shards them
just like any other module.

```python
import transformer_engine.pytorch as te
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

class Block(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        # TE analogues, not torch.nn
        self.norm1 = te.RMSNorm(dim)
        self.attn = te.DotProductAttention(...)
        self.qkv  = te.Linear(dim, 3 * dim)
        self.proj = te.Linear(dim, dim)
        self.norm2 = te.RMSNorm(dim)
        self.mlp  = te.LayerNormMLP(dim, ffn_dim, activation="swiglu")

    def forward(self, x, ...):
        # unchanged
        ...

model = ...  # decoder-only Transformer built from Blocks
for block in model.blocks:
    fully_shard(block, mp_policy=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    ))
fully_shard(model, mp_policy=MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
))

# training loop
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    logits = model(input_ids)
loss = loss_fn(logits, labels)
loss.backward()
optimizer.step()
```

Common integration bugs:

- **`te.fp8_autocast` in the wrong scope.** If you put the context
  around only the attention and not the MLP, only the attention
  matmuls will be FP8. Almost always you want the context around
  the whole `forward`. Symmetrically, do not put it around the loss
  computation — the loss is FP32 and the context does nothing there.
- **Mixing `torch.nn.Linear` and `te.Linear` in the same block.**
  Only the TE modules respond to `fp8_autocast`. If half the matmuls
  in the block are `nn.Linear`, half your matmuls are still BF16 and
  the throughput lift is muted. Standardize on TE modules for the
  whole block.
- **Forgetting `te.DotProductAttention` for FP8 attention.** Using
  `F.scaled_dot_product_attention` inside `fp8_autocast` runs the
  attention in BF16 (SDPA does not know about the FP8 recipe). If
  you want FP8 attention (via FA3), use `te.DotProductAttention`
  which wires FA3-FP8 correctly.

## Checking that FP8 actually turned on

Two verifications, both in a short debug run:

### 1. Profile for FP8 kernel names

`torch.profiler` or Nsight Systems around one step, with FP8
enabled. Look for kernel names containing `fp8`, `e4m3`, `e5m2`, or
`hopper_fp8_gemm`. If you see only `bf16_gemm` / `cutlass_bf16`,
FP8 did not dispatch — check the `fp8_autocast` scope and the module
types.

### 2. Convergence equivalence run

Run the model in BF16 (no `fp8_autocast`) and in FP8 with the same
seed for a few thousand steps. Loss curves should track closely;
the FP8 run may be slightly noisier and can diverge by a small
constant offset that closes as `amax_history` fills. If the FP8
curve visibly separates and drifts upward, something is
misconfigured — commonly the recipe (`Format.E4M3` instead of
`HYBRID`), or a non-TE `nn.Linear` still running in BF16 breaking
the layered flow.

Publish both curves in any FP8 rollout — silence about numerics is
how bad FP8 policies ship.

## Measured lift, honestly

The lift from BF16 → FP8 depends heavily on model shape. Rough
rules of thumb from the TE user guide and community measurements:

- Matmul-heavy dense LLMs (7B–70B) on H100: 30–50% step-time
  reduction end-to-end, converting to 30–50% MFU rise on the matmul
  buckets. Kernel isolation shows the matmuls themselves near 2×;
  the end-to-end number is diluted by everything not in the FP8
  path (loader, comm, elementwise ops).
- MoE and attention-dominated regimes: smaller end-to-end wins,
  because a large fraction of step time is already outside the
  matmul path.
- Very small models on H100: the FP8 kernel launch overhead can
  dominate and the win vanishes. FP8 is a large-model tool.

Report the isolation-kernel win, the end-to-end step-time delta, and
the MFU delta from chapter 1. Cite the TE version, the recipe, and
whether attention was FA3-FP8 or FA-BF16.

## When FP8 is not the right tool

- **Non-Hopper hardware.** FP8 tensor cores are Hopper (H100/H200)
  and later. On Ampere/A100, TE loads but FP8 kernels do not
  dispatch — you get BF16, and TE is only useful for its module
  factoring.
- **Fine-tuning where numerics matter.** Runs that need exact
  convergence to a reference (RLHF critic, ablations against a BF16
  baseline) are cleaner in BF16.
- **A shaky BF16 baseline.** If your BF16 numbers are not stable,
  every FP8 problem will look like a scale-factor bug when it is
  actually the underlying policy. Fix BF16 first.

## FP8 on Blackwell

Blackwell (B200) changes the FP8 story: new tensor-core generation,
MXFP8 block-scaling as a native format, and an updated TE recipe.
The high-level architecture — TE owns the scale bookkeeping — is
unchanged; the specific recipe class and defaults are not. Consult
the TE release notes for the version you are on before assuming
Hopper-era knobs transfer.

## Summary

- FP8 requires per-tensor dynamic scale management. Transformer
  Engine owns that bookkeeping — do not roll your own.
- The correct policy layers FP8 matmuls on top of the chapter-3
  BF16 policy: BF16 parameter storage, BF16 all-gather, FP8 matmul
  inside `te.fp8_autocast`, BF16 output, FP32 gradient reduction,
  FP32 optimizer.
- Start with `DelayedScaling(fp8_format=Format.HYBRID)`. E4M3 for
  forward, E5M2 for gradients. `Format.E4M3` all-round is inference-
  only.
- Replace `nn.Linear` / `nn.LayerNorm` / attention modules with the
  TE analogues; wrap the whole `forward` in `fp8_autocast`; then
  `fully_shard` the model. Mixing `nn.Linear` inside the FP8 scope
  mutes the win.
- Verify with a profile (FP8 kernel names) and a convergence
  equivalence run against BF16 (loss curves should track within
  small tolerance).
- Expected end-to-end lift for matmul-heavy dense LLMs on H100:
  30–50% step-time reduction. Cite version, recipe, and whether
  attention is FA3-FP8.
- Do not enable FP8 on top of an unstable BF16 baseline; do not
  bother on non-Hopper hardware; Blackwell has its own recipe
  matrix.
