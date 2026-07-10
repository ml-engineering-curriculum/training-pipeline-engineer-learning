# BF16 Mixed Precision Across FSDP2

Chapter 1 named "sub-optimal dtype" as a 10–30 pp MFU gap. This
chapter closes the FP32-to-BF16 half of that gap — the half that
does not require FP8 hardware and pays off on every generation from
A100 forward. Chapter 4 closes the remaining half (BF16 → FP8) on
Hopper.

The essence: parameters, activations, and the matmul math run in
BF16 in the forward and backward, while the optimizer's master
weights and Adam moments stay in FP32. NCCL reductions run in
whichever precision reconciles range vs. bandwidth for your model.
FSDP2's `MixedPrecisionPolicy` is the one API that expresses all of
these choices per parameter, per module.

The two primary references:

- **NVIDIA "Mixed Precision Training" blog and the Micikevicius et
  al. (2018) paper (ICLR).** The foundational mixed-precision
  reference; introduces FP16 loss-scaling.
  https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/
- **PyTorch FSDP2 `fully_shard` API doc.**
  https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
  including `MixedPrecisionPolicy`.

## Why BF16, and why not FP16

BF16 (Google Brain's bfloat16 format, also documented in NVIDIA's
Ampere whitepaper) and FP16 both use 16 bits, but their exponent /
mantissa splits differ:

| Format | Sign | Exponent | Mantissa | Dynamic range          |
|--------|------|----------|----------|------------------------|
| FP32   | 1    | 8        | 23       | ~10⁻³⁸ to ~10³⁸        |
| FP16   | 1    | 5        | 10       | ~10⁻⁵ to ~65 504       |
| BF16   | 1    | 8        | 7        | ~10⁻³⁸ to ~10³⁸        |
| FP8 E4M3 | 1  | 4        | 3        | ~10⁻⁷ to ~448          |
| FP8 E5M2 | 1  | 5        | 2        | ~10⁻⁷ to ~57344        |

The design consequence:

- **BF16 has the same dynamic range as FP32.** Gradients that
  under-flow in FP16 (activations of small softmax outputs, gradients
  through deep transformer stacks) do *not* under-flow in BF16
  because the exponent is unchanged.
- **BF16 has less precision than FP16.** Only 7 mantissa bits (8
  effective with the implicit leading bit) vs. FP16's 10. In
  practice, tensor-core matmuls accumulate in FP32 regardless of
  operand dtype, so the precision loss shows up only in the operand
  storage, not in the accumulation. For transformer training this is
  a good trade.
- **No loss scaling required.** FP16 mixed precision needs a
  loss-scale multiplier (typically dynamic, 2^7 → 2^15) to shift
  gradients above the FP16 subnormal range. BF16 skips this
  entirely because the small end of the exponent range matches FP32.

The practical consequence for FSDP2: `MixedPrecisionPolicy` with BF16
parameters does not need `torch.cuda.amp.GradScaler`; the training
step is `loss.backward(); optimizer.step()` with no scaling
gymnastics. This alone eliminates a class of bugs (double-scaling,
scale drift on resume, scale desync across ranks under gradient
accumulation) that FP16 mixed precision inherits.

On H100 the BF16 and FP16 peak FLOPS are identical (989 TFLOPS
dense), so there is no performance reason to prefer FP16. Every
mainstream open-source large-scale training run since 2022 uses BF16
plus (optional) FP8, not FP16.

## The FSDP2 `MixedPrecisionPolicy` surface

FSDP2 owns three precision knobs, applied per shard-unit:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
import torch

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    output_dtype=None,        # inherit; usually leave None
    cast_forward_inputs=True,
)

for block in model.blocks:
    fully_shard(block, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

The three dtypes:

- **`param_dtype`.** The dtype the parameters live in *inside*
  the sharded unit. On forward, the all-gathered full parameters
  are cast to this dtype before use. Activations produced by these
  parameters inherit the dtype (transformer matmul: BF16 params ×
  BF16 activations → BF16 activations).
- **`reduce_dtype`.** The dtype the gradient reduce-scatter runs
  in. Default `None` uses `param_dtype`. Setting it to
  `torch.float32` upcasts on the reduction: each rank casts its
  local gradient shard to FP32 before the reduce-scatter and casts
  back to BF16 after. The cost is 2× wire volume on the
  reduce-scatter; the benefit is one fewer source of BF16
  round-off during the gradient sum.
- **`output_dtype`.** The dtype the module's output is cast to
  when it leaves this sharded unit. Usually `None` so downstream
  modules see whatever `param_dtype` produced.

A fourth flag:

- **`cast_forward_inputs=True`.** Cast any inputs to the module to
  `param_dtype` on entry. Turn this on unless you have upstream
  reasons to want mixed-dtype inputs; leaving activations at FP32
  on the boundary between an FSDP-wrapped block and its caller
  produces a redundant BF16 → FP32 → BF16 round-trip.

## What stays in FP32

Even under `param_dtype=torch.bfloat16`, several things must remain
FP32. In FSDP2 they stay FP32 automatically, but you need to know
*why*:

- **Optimizer master weights.** Adam's `param_data` copy inside
  the optimizer state stays FP32. On `optimizer.step()`, Adam reads
  FP32 moments, FP32 master params, applies the update in FP32, and
  writes back FP32. Then, on the next forward, FSDP2 casts the
  updated FP32 master to BF16 for the all-gather. The FP32 master
  is what preserves precision across many small updates that would
  round to zero in BF16 (a 7-mantissa-bit `p + lr · g` with
  `lr · g ≈ 1e-4 · p` rounds to `p` in BF16 half the time).
- **Adam moment estimates (`m`, `v`).** These are second-moment
  variances; storing them in BF16 would compound rounding across
  steps. Keep FP32.
- **LayerNorm / RMSNorm parameters and their reductions.** LN
  is a mean-and-variance reduction across the hidden dimension. In
  BF16 the variance computation drifts numerically; PyTorch's
  `nn.LayerNorm` casts inputs to FP32 internally for the reduction
  when its parameters are FP32. If you have set the LayerNorm
  parameters to BF16, the internal reduction runs in BF16 and you
  can see loss instability on very large models.

Two ways to enforce LayerNorm-in-FP32 under FSDP2:

- Do not wrap the LayerNorms in `fully_shard`. Because they are
  outside a wrapped unit, they keep their construction dtype
  (typically FP32).
- Wrap them but pass a different `mp_policy` with
  `param_dtype=torch.float32`. FSDP2's per-unit policy lets you do
  this at the granularity of a submodule.

The common convention in production is: transformer block →
`param_dtype=BF16`; norms and embeddings → FP32.

## `reduce_dtype`: the honest trade

The default `reduce_dtype=param_dtype` (BF16) is the throughput
choice: gradient reduce-scatter runs at BF16 wire volume, roughly
half the FP32 volume. The instability case is:

- A very large number of ranks summing gradients. The BF16
  add-tree round-off compounds; the final gradient may differ from
  the FP32 truth by more than the natural gradient scale.
- Very small gradients on a small parameter (final unembedding
  bias, for example) drifting below BF16 precision.

Two mitigations:

- **`reduce_dtype=torch.float32`.** Pay the 2× wire cost, get
  numerically clean reductions. Standard on multi-thousand-GPU
  runs.
- **Stochastic rounding on BF16 accumulator.** Not FSDP2's
  default; requires a NCCL build with stochastic-rounding support
  and an explicit knob. If you find yourself considering this, it
  is a signal to just use FP32 reductions.

<!-- needs-research: verify current default for `reduce_dtype` in the latest torchtitan reference config (BF16 vs FP32) and update guidance if the community consensus has shifted. -->

The Llama 3 pretraining report notes FP32 gradient reductions
explicitly; MegaScale's paper (Jiang et al., 2024, arXiv:2402.15627)
notes the same choice. Take these as evidence that the extra wire
cost is worth it at scale.

## Interaction with gradient clipping

`torch.nn.utils.clip_grad_norm_` computes the global gradient norm
by summing per-parameter squared L2 norms across all parameters. On
FSDP2 with sharded parameters you use
`torch.distributed.fsdp.FSDP.clip_grad_norm_` or (in the
`fully_shard` API) call `clip_grad_norm_` after the
reduce-scatter has landed. Two constraints:

- The clipping norm must run on FP32 gradients, not BF16, to avoid
  under-flow when squaring small values. If `reduce_dtype=BF16` you
  cast to FP32 *for the norm*.
- The clipping fires *after* reduce-scatter, so the norm sees the
  post-reduction gradient shard. FSDP2 handles the cross-rank
  all-reduce of the squared-norm inside its `clip_grad_norm_`
  helper.

The correct pattern:

```python
loss.backward()
# FSDP2 clip helper handles the cross-rank reduction.
from torch.distributed.fsdp import fully_shard
# fully_shard-wrapped model gains .clip_grad_norm_
grad_norm = model.clip_grad_norm_(max_norm=1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

For hand-rolled clip logic, cast to FP32 before the square:

```python
squared = sum((p.grad.detach().float() ** 2).sum() for p in params)
```

## Interaction with the optimizer

`torch.optim.AdamW` (and the Apex fused variant) understands mixed
precision: it reads the BF16 parameter, keeps FP32 master state
internally, and produces FP32 updates. When the parameter is a
DTensor (which is what FSDP2 produces), the optimizer's per-parameter
state (`m`, `v`, `master_param`) is a DTensor too — sharded on the
same mesh axis. `AdamW` runs local per-shard; there is no cross-rank
optimizer collective.

Two subtle things:

- **`foreach=True`.** The `foreach` fused AdamW path fuses per-tensor
  loops into per-tensor-list kernels. On a large parameter list
  (many transformer blocks) this is a substantial speedup. Verify
  it is on:

  ```python
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                foreach=True, fused=True)
  ```

  The `fused=True` variant goes further and uses the CUDA fused
  Adam kernel; supported for `dtype=torch.float32` params, which is
  what the optimizer sees.
- **State-dict loading across dtype changes.** If you save a run
  in BF16 params + FP32 optimizer state and reload with the
  policy changed to `param_dtype=torch.float32`, the optimizer's
  FP32 state loads cleanly but the parameters need explicit casting.
  Distributed Checkpoint (DCP, mod-106) handles this via its own
  dtype coercion; ad-hoc `state_dict` juggling does not.

## Debugging BF16 precision issues

The three symptoms and their diagnoses:

- **Loss diverges after several thousand steps, no NaN in the first
  few.** Likely the LayerNorm running in BF16 accumulating a variance
  that drifts. Check: what dtype are the LN params under FSDP2's
  policy? Fix: keep LNs FP32.
- **Loss curve is noticeably noisier than an FP32 reference.**
  Likely BF16 gradient reductions on many ranks. Check
  `reduce_dtype`. Fix: `reduce_dtype=torch.float32`.
- **`optimizer.step()` produces NaNs.** Rare in BF16 (the range is
  wide); if it happens, a gradient exploded and clipping was skipped
  or applied after the NaN was already in `p.grad`. Check the
  clip-norm order and cast to FP32 before squaring.

The `TORCH_DISTRIBUTED_DEBUG=DETAIL` and
`TORCH_NCCL_DESYNC_DEBUG=1` environment variables give you actionable
output on distributed-precision divergences across ranks. See the
PyTorch distributed debugging docs at
https://pytorch.org/docs/stable/distributed.html.

## Sample torchtitan-style policy

The reference `torchtitan` configuration for a Llama-style
decoder-only transformer:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
import torch

block_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    cast_forward_inputs=True,
)

for block in model.transformer.layers:
    fully_shard(block, mp_policy=block_policy, mesh=dp_mesh)

# Root model gets its own policy; embeddings + norms outside blocks
# remain FP32 in construction and stay FP32 here.
root_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
fully_shard(model, mp_policy=root_policy, mesh=dp_mesh)
```

Every choice here follows from the rules above: BF16 params for the
matmul-heavy transformer blocks, FP32 reductions on all
gradient-reduce-scatter collectives, LN and embeddings excluded from
the wrapped units so they stay FP32 as constructed. See
https://github.com/pytorch/torchtitan for the canonical
implementation this pattern is drawn from.

## Summary

- BF16 has FP32's exponent range and 7 mantissa bits. No loss
  scaling is required (unlike FP16). Tensor-core accumulation stays
  FP32; only operand storage is 16-bit.
- FSDP2's `MixedPrecisionPolicy` owns three dtypes:
  `param_dtype` (usually BF16), `reduce_dtype` (BF16 for
  throughput, FP32 for scale-honest reductions), `output_dtype`
  (usually inherit).
- Optimizer master weights and Adam moments stay FP32. LayerNorm
  parameters stay FP32; exclude them from the FSDP2-wrapped unit
  or use a per-unit FP32 policy.
- FP32 gradient reductions are the standard for very-large-scale
  runs (Llama 3, MegaScale). BF16 reductions are correct at
  smaller scale and half the wire cost.
- Clipping computes norms on FP32-cast gradient shards; FSDP2's
  `clip_grad_norm_` handles the cross-rank sum. AdamW with
  `foreach=True, fused=True` is standard.
- The three debug signatures: LN drift → keep LNs FP32; noisy loss →
  FP32 reductions; NaN → clip order and dtype.
- Chapter 4 layers FP8 on top of this BF16 baseline via Transformer
  Engine. The FSDP2 policy for the BF16 layer is unchanged; FP8 is
  a per-module opt-in inside `fp8_autocast`.
