# BF16 Mixed Precision with FSDP2

BF16 is the modern-default training precision on Ampere and Hopper.
It gives you the FLOP peak of the tensor cores without the numerical
knife-edges of FP16 (no loss scaling, no dynamic-scale storms) and
composes cleanly with FSDP2's parameter sharding. This chapter
covers the "why BF16, why now" story, the mixed-precision policy
FSDP2 wants you to set, and the pieces of the training loop that
must stay in higher precision for correctness.

FP8 is a separate machine and lives in chapter 4. This chapter is the
strong baseline every FP8 experiment should start from.

## The three floating-point formats you will use

Every training step touches at least these three formats:

| Format | Bits (sign / exp / mant) | Dynamic range           | Tensor-core support         |
| ------ | ------------------------ | ----------------------- | --------------------------- |
| FP32   | 1 / 8 / 23               | ~1e-38 to 3e38          | Yes (limited on new HW)     |
| FP16   | 1 / 5 / 10               | ~6e-5 to 6e4            | Yes (Ampere/Hopper)         |
| BF16   | 1 / 8 / 7                | Same as FP32            | Yes (Ampere/Hopper/Blackwell) |
| FP8-E4M3 | 1 / 4 / 3              | ~2e-3 to 448            | Yes (Hopper+; chapter 4)    |
| FP8-E5M2 | 1 / 5 / 2              | ~6e-5 to 6e4            | Yes (Hopper+; chapter 4)    |

The single most important row is BF16: it keeps FP32's exponent
range while cutting the mantissa. That trade means:

- **No loss-scaling machinery.** FP16 needs dynamic loss scaling
  because gradients underflow at 6e-5. BF16 does not — the exponent
  range covers gradient magnitudes without help. This alone removes
  a whole class of "scale storm" incidents FP16 mixed-precision
  training used to see.
- **Precision costs where it hurts.** BF16 has 7 mantissa bits
  vs. FP16's 10. Sums of many small values lose precision faster;
  large reductions (batch-norm running stats, optimizer moments)
  must stay in FP32 or diverge.
- **Same peak FLOPs as FP16 on tensor cores.** A100/H100 tensor
  cores expose BF16 and FP16 at the same TFLOP/s. The BF16 story is
  entirely about numerical robustness, not compute peak.

BF16 became the default in mainstream large-model training with the
Big Science BLOOM run (Scao et al., 2022, arXiv 2211.05100 §3.3) and
subsequent GPT-NeoX / OPT / Llama runs; the Kalamang paper (Micikevicius
et al., 2018, "Mixed Precision Training", arXiv 1710.03740) is the
original mixed-precision reference, though it predates BF16 as a
first-class training dtype.

## The mixed-precision policy: which tensor lives in which format

"Mixed precision" is a *policy*: for each class of tensor in the
training step, which precision does it live in? The canonical BF16
policy for a modern LLM is:

- **Model weights (the sharded parameters):** BF16. This is the
  "parameter dtype" — what FSDP2 stores per rank.
- **Activations (forward-pass intermediates, saved for backward):**
  BF16. Everything the matmul kernels consume and produce.
- **Gradients (computed in the backward pass, all-reduced by
  FSDP2):** BF16 during compute; the reduction can happen in BF16 or
  FP32 depending on your policy (see `reduce_dtype` below).
- **Optimizer state (Adam moments, LR-scheduler state):** FP32. Full
  precision. This is the "master weights" pattern — the optimizer's
  parameter view is FP32, and the update is applied in FP32 before
  being cast to BF16 for the next step's forward.
- **Loss and any reductions across a large number of terms:** FP32.
  Casting to BF16 too early in a reduction (softmax normalizer, MoE
  routing sums, `logsumexp`) is a classic silent-divergence bug.
- **Layernorm / RMSNorm running stats and the epsilon:** FP32.
  Otherwise the reduction underflows on long sequences.

The optimizer keeps FP32 master weights because Adam-family updates
sum many small increments per step; done in BF16, the
`param += -lr * m / (sqrt(v) + eps)` step loses information when
`lr * m` is small compared to `param`. Keep the optimizer in FP32,
downcast to BF16 for the forward, and this is a non-issue.

## FSDP2's `MixedPrecisionPolicy`

FSDP2 is the current PyTorch fully-sharded data-parallel API
(`torch.distributed.fsdp.fully_shard`), introduced in PyTorch 2.4
and refined in 2.5+; see the FSDP2 tutorial at
https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
(check for the FSDP2/`fully_shard` sections — earlier docs cover
the legacy `FullyShardedDataParallel` class). The mixed-precision
knob is `MixedPrecisionPolicy`:

```python
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,     # sharded parameter storage
    reduce_dtype=torch.float32,     # gradient reduction dtype
    output_dtype=None,              # activation dtype (inherits param_dtype)
    cast_forward_inputs=True,       # cast module inputs on the way in
)

for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

The three knobs to understand:

- **`param_dtype=torch.bfloat16`.** Parameters are stored, all-
  gathered, and used in matmuls in BF16. This is the setting that
  actually turns on BF16 training.
- **`reduce_dtype=torch.float32`.** The gradient reduce-scatter
  happens in FP32. This is a numerical-safety choice: reducing
  gradients across many ranks accumulates enough small values that
  BF16's 7 mantissa bits cause visible loss curve degradation at
  scale. The cost is 2× network traffic on the reduction. On small
  clusters (single-node, dozens of GPUs) you can often set this to
  BF16 with no observed loss quality delta; on multi-thousand-GPU
  runs, keep it in FP32. Measure before deciding.
- **`output_dtype`.** The dtype the wrapped module returns. Leaving
  this as `None` means "same as `param_dtype`", which is usually
  what you want.

FSDP2 handles the master-weight pattern behind the scenes: even
though `param_dtype` is BF16, the optimizer sees a FP32 view of the
parameters if you use `torch.optim.AdamW` with the standard
`fully_shard`-wrapped model — the optimizer step is applied in FP32
and the result is cast back to BF16 storage. This is the "master
weights" model, mediated by the fused optimizer's state, and it is
the reason you do not need a separate FP32 shadow copy.

## Legacy `torch.cuda.amp.autocast` still works, sort of

The pre-FSDP2 pattern was to wrap the forward pass in
`torch.cuda.amp.autocast(dtype=torch.bfloat16)`. This still works and
is not wrong for simple setups, but under FSDP2 it is redundant with
`MixedPrecisionPolicy` and can produce confusing behavior (double-
casts, unexpected outputs at module boundaries). The recommendation:

- Under FSDP2, set `MixedPrecisionPolicy` and do *not* use
  `autocast`. FSDP2 owns the dtypes.
- Under plain single-GPU or DDP, `autocast(dtype=torch.bfloat16)` is
  still the correct entry point.

If you inherit a codebase that uses `autocast` inside an FSDP2-
wrapped model, remove the `autocast` context; keep the
`MixedPrecisionPolicy`.

## Pieces that stay in FP32 (and how to enforce it)

FSDP2's mixed-precision policy applies to the *module* — activations
and outputs are cast at module boundaries. Some operations must
stay in FP32 regardless; explicitly cast their inputs when the
casting is not automatic:

- **Cross-entropy loss / `log_softmax`.** These already run in FP32
  in most PyTorch versions when given a BF16 input, but verify with
  your version. If not, cast the logits to FP32 before the loss.
- **Softmax normalizers inside a research variant of attention.**
  If you are outside FlashAttention and doing softmax by hand, do
  the exponent in FP32.
- **Any custom reduction over more than ~1000 values.** RMS layer
  norms, MoE router sums, any `torch.sum` over a long axis. The
  standard `nn.LayerNorm` / `nn.RMSNorm` in modern PyTorch already
  computes in FP32 internally; verify against your framework's
  implementation.
- **Learning-rate scheduler and step counter.** Not affected by
  mixed precision — but a reminder to check nothing accidentally
  downcast them.

The right check is a targeted equivalence test: run the model
forward once in FP32 and once in BF16-under-FSDP2 with identical
weights and inputs, and check that the losses agree to a small
tolerance (typically ~1e-3 relative). Larger disagreement points
directly at a reduction that should have stayed in FP32.

## Interaction with FlashAttention (chapter 2)

FlashAttention v2 and v3 both support BF16 inputs directly. When you
enable `MixedPrecisionPolicy(param_dtype=bfloat16)` and call
`scaled_dot_product_attention` on BF16 Q/K/V, FA runs in BF16 with
its internal accumulations in FP32 — the exact behavior you want.
No additional configuration is needed. Verify with a profile that
you see the BF16 FlashAttention kernel and not a fallback path.

## Interaction with gradient clipping

Gradient clipping (`torch.nn.utils.clip_grad_norm_`) computes the
global norm of the gradients and rescales. Under FSDP2 with
`reduce_dtype=torch.float32`, the gradients arrive at the clip step
in FP32 and the norm is computed correctly. Under
`reduce_dtype=torch.bfloat16`, the norm calculation is one more
place where BF16 mantissa loss shows up on large parameter counts;
prefer FP32 reduction here as well, or use the FSDP2-native
`clip_grad_norm_` that handles the sharded-parameter case.

## Signals a BF16 policy is subtly broken

Symptoms that show up in the first few thousand steps of a bad
mixed-precision policy and how to read them:

- **Loss plateau at higher value than expected.** A reduction is
  running in BF16 that should be FP32 — commonly the softmax
  normalizer in a hand-written attention variant.
- **Loss diverges after N thousand steps.** Optimizer moments are in
  BF16, not FP32. Confirm your optimizer is applying updates in
  FP32; if you rolled your own, the master-weight pattern is
  missing.
- **NaN grad in a single layer.** An activation blew up past the
  BF16 range in one specific block — often an unusually-scaled
  learned bias. Check the layer's forward, and if the value is
  legitimately outside BF16 range, upcast that block explicitly.
- **Loss curve is 2–4x noisier than the FP32 reference at the same
  seed.** Some noise is normal — mixed precision changes reduction
  order. If the noise is orders of magnitude off, one of the above
  bugs is present.

## The FP8 preview

The precise policy you land in this chapter (BF16 storage, FP32
reduction, FP32 optimizer) is the *baseline* the FP8 path in
chapter 4 layers on top of. FP8 does not replace BF16; it augments
it — FP8 for matmuls only, BF16 for parameter storage, and FP32
still for the optimizer. Chapter 4 makes that concrete. If your BF16
policy is not stable and measured, do not turn on FP8.

## Summary

- BF16 is the modern-default training precision on Ampere and
  Hopper: same tensor-core peak as FP16, same exponent range as
  FP32, no loss scaling required. This is your baseline.
- The canonical policy: BF16 parameters + activations, FP32 gradient
  reduction (`reduce_dtype=torch.float32`) at multi-thousand-GPU
  scale, FP32 optimizer state and master weights. FSDP2's
  `MixedPrecisionPolicy` expresses this in three lines.
- Do not layer `torch.cuda.amp.autocast` on top of FSDP2's
  MixedPrecisionPolicy; the policy owns the dtypes.
- Cross-entropy, softmax normalizers, and large reductions must
  stay in FP32. Verify with a targeted FP32-vs-BF16 equivalence
  run.
- FlashAttention v2 and v3 handle BF16 correctly with FP32 internal
  accumulations. No additional config needed.
- FP8 (chapter 4) is layered on top of a working BF16 policy, not
  a replacement for it.
