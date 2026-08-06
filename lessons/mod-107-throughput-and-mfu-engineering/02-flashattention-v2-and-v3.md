# FlashAttention v2 (Ampere) and v3 (Hopper + FP8)

The single largest kernel in a Transformer training step is
attention. On any generation of hardware, the first big MFU jump
comes from replacing the framework's default attention with a
FlashAttention kernel: v2 on Ampere/A100 and Hopper, v3 on Hopper
with FP8. This chapter is about picking the right variant for your
hardware, integrating it correctly through PyTorch, and measuring the
throughput lift — not about the kernel internals, which are the
performance-engineer role (chapter 8).

## Why attention is worth its own chapter

For a Transformer trained at sequence length `S`, attention FLOPs
scale as `S²` while the rest of the model's FLOPs scale as `S`. Long
past `S ≈ 2048` on a modern LLM, attention starts dominating step
time. It is also *memory-bound* in its naive implementation: the
softmax and the `QK^T` intermediate blow up to `O(S²)` HBM traffic,
so the kernel becomes bandwidth-limited long before it becomes
FLOP-limited. FlashAttention fixes both problems in the same kernel:

- **Tiling the softmax** so intermediate `S²` blocks live in SRAM,
  not HBM. Turns the algorithm from HBM-bandwidth-bound to
  arithmetic-intensity-limited, i.e. tensor-core-bound.
- **Fusing the forward and backward passes into single kernels**, so
  the recompute of the softmax on backward is cheap (small SRAM tile)
  rather than needing to re-materialize the full `S²` attention
  matrix.

The primary sources are:

- **Dao et al., 2022. "FlashAttention: Fast and Memory-Efficient
  Exact Attention with IO-Awareness."** arXiv 2205.14135. The
  original paper.
- **Dao, 2023. "FlashAttention-2: Faster Attention with Better
  Parallelism and Work Partitioning."** arXiv 2307.08691. v2 on
  Ampere / early Hopper.
- **Shah et al., 2024. "FlashAttention-3: Fast and Accurate
  Attention with Asynchrony and Low-Precision."** arXiv 2407.08608.
  v3, built for Hopper's WGMMA and TMA units, with FP8 support.

The implementations live at https://github.com/Dao-AILab/flash-
attention. FlashAttention v2 is what PyTorch's `torch.nn.functional
.scaled_dot_product_attention` (SDPA) reaches for on Ampere by
default; v3 is available via the same repo and, on new-enough
PyTorch, through SDPA when the backend selects it.

## Which variant for which hardware

The version-to-hardware pairing is not negotiable — using the wrong
one silently costs you 30% of your attention throughput or breaks
entirely.

| Hardware      | Recommended | Notes                                                                                                                                      |
| ------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| V100 / T4     | (neither)   | Neither v2 nor v3 targets pre-Ampere. Use SDPA's `math` or `efficient` backend and expect low MFU on attention. Upgrade the hardware.      |
| A100 (SM80)   | v2          | v2 targets Ampere tensor cores. v3 is Hopper-only (WGMMA/TMA), not supported on A100.                                                      |
| H100 (SM90)   | v2 or v3    | v2 works but does not use Hopper-specific units. v3 uses WGMMA + TMA and adds FP8; measurably higher throughput. Prefer v3 when available. |
| H200          | v2 or v3    | Same math throughput as H100; v3 preferred for the same reasons.                                                                           |
| B200 (SM100)  | check repo  | Blackwell requires kernels retargeted to the new tensor-core generation. Consult the FlashAttention repo for the current status.           |

Two operational subtleties:

- FlashAttention v3 requires a recent CUDA + PyTorch combination and
  is packaged separately from v2. Follow the repo's build matrix; do
  not assume `pip install flash-attn` gives you v3.
- Some SDPA backends silently fall back to the "efficient" xFormers-
  style path if the head dimension or dtype is not supported. Verify
  which backend actually ran (see the SDPA `enable_flash` /
  `enable_mem_efficient` / `enable_math` context managers and
  `torch.backends.cuda.sdp_kernel`).

## Integrating through PyTorch: three paths, one recommendation

You can call FlashAttention from PyTorch three ways. The
recommendation is to use `scaled_dot_product_attention` and let SDPA
dispatch, unless you have a specific reason to bypass it.

### 1. `torch.nn.functional.scaled_dot_product_attention` (recommended)

```python
import torch
import torch.nn.functional as F

# q, k, v shape: (batch, heads, seq, head_dim), dtype bf16 or fp16
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,           # causal via is_causal=True
    dropout_p=0.0,
    is_causal=True,
)
```

Under the hood, PyTorch tries several backends in order. On a modern
build with FlashAttention installed, `flash` is the first choice; on
Hopper with recent PyTorch + FA3, v3 is dispatched. Force the choice
with the context manager if you want to be sure:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Verify what actually ran with `torch.backends.cuda.flash_sdp_enabled()`
and by profiling — see the measurement section below.

### 2. Direct `flash_attn` API (for features SDPA does not expose)

For features SDPA has not yet plumbed (variable-length sequences via
`flash_attn_varlen_func`, ALiBi, sliding-window attention, block-
sparse masks), call the `flash_attn` package directly:

```python
from flash_attn import flash_attn_func, flash_attn_varlen_func

# fixed-length
out = flash_attn_func(q, k, v, causal=True)

# packed variable-length (sequence packing, chapter 5)
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal=True,
)
```

The `varlen` path is what chapter 5 leans on for sequence packing —
it avoids materializing a padded attention mask when your batch is a
concatenation of variable-length sequences.

### 3. Framework-native (Transformer Engine, Megatron, torchtitan)

NVIDIA's Transformer Engine (chapter 4) wraps FlashAttention v3 in a
`DotProductAttention` module that plays nice with its FP8 pathway.
Megatron-LM and torchtitan integrate FlashAttention through their own
attention modules. If you are on those frameworks, use their
integration — they already picked the right kernel and mask type for
their fused blocks.

## Measuring the throughput lift honestly

The most common way to lie to yourself about a FlashAttention
integration is to swap the kernel and *only* look at end-to-end
tokens/s. The kernel might be a huge win in isolation and a small
win end-to-end if attention was already only 15% of your step time
(short sequences). Or it might be a small win in isolation and a
huge win end-to-end because it unlocks a larger micro-batch that
raises MFU elsewhere.

Measure four things, in this order:

### 1. Kernel isolation microbench

Before touching the training loop, benchmark the attention op in
isolation for the exact `(batch, heads, seq, head_dim, dtype,
causal)` your model uses. A minimal harness:

```python
import torch, time
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

def bench(backend, iters=100, warmup=10):
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    with sdpa_kernel(backend):
        for _ in range(warmup):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters

t_math   = bench(SDPBackend.MATH)
t_effic  = bench(SDPBackend.EFFICIENT_ATTENTION)
t_flash  = bench(SDPBackend.FLASH_ATTENTION)
```

Report all three. If `t_flash` is not materially better than
`t_effic`, either FlashAttention was not dispatched (check the
backend availability flags) or your shapes hit an unsupported case
(check the FA repo's supported-shape matrix).

Compare to the attention roofline: attention on Hopper BF16 with
FA v3 should sit at a large fraction of the 989 TFLOP/s peak on
long-enough sequences; the FA v3 paper (arXiv 2407.08608, figure
1) reports FA3 reaching ~75% of H100 BF16 peak on long-sequence
forward, and near-peak FP8 with a dedicated FP8 path. Verify
against your measured `t_flash` and the attention-FLOP formula
from chapter 1.

### 2. Full-step lift measurement

Now measure end-to-end training step time with the old attention and
with FA. Two runs, identical everything except the SDPA backend:

```
step_time_baseline:  attention on the SDPA math or efficient backend
step_time_flash:     attention on FLASH_ATTENTION (v2 or v3)

lift_step_time = step_time_baseline / step_time_flash
lift_MFU       = MFU_flash - MFU_baseline
```

Publish both. The step-time ratio is intuitive; the MFU delta is
what capacity planning uses.

### 3. Verify with a profile

Run `torch.profiler` or Nsight Systems with `record_shapes=True`
around a single step in both configurations. Check that:

- The attention kernel name changed. Look for `flash_fwd_kernel`,
  `flash_bwd_kernel` for v2; `flash_fwd_hopper` / `flash_bwd_hopper`
  or similar for v3. If the kernel name still contains `sdpa_math`
  or `mem_eff`, FA was not dispatched.
- Attention now occupies less of the step time. The delta between
  old and new attention time should closely match the FLOP-adjusted
  ratio you predicted from the kernel isolation microbench.
- Nothing else regressed. It is easy for a kernel swap to change the
  layout of nearby tensors and slow an adjacent kernel.

### 4. Correctness gate before shipping

FlashAttention is *exact* (up to floating-point associativity), not
approximate. But subtle differences will show up:

- Loss curves in the first few hundred steps of a fresh training run
  will diverge slightly between backends due to reduction order.
  This is expected; do not treat "not bit-identical" as a bug.
- Some masks that work in the reference implementation are not
  supported by the fused kernels (odd shapes, some ALiBi
  configurations, some `attn_mask` broadcastings). SDPA falls back
  silently — profile to confirm you actually got the fast path.
- Dropout in attention is *seeded differently* between backends. If
  your setup depends on a specific dropout pattern (rare), you have
  to reproduce it manually.

Run at least one short (a few hundred steps) equivalence run with an
identical seed and both backends, and check that the loss curve
diverges only within expected floating-point tolerance.

## Adding FP8 with FlashAttention v3

FA v3's headline feature on Hopper is native FP8 attention. It is not
"turn on a flag" — you have to be running the surrounding matmuls in
FP8 through Transformer Engine for the numerical scale management to
land correctly. Chapter 4 covers TE end-to-end; here, the two things
to know:

- FA3's FP8 forward matches BF16 to a small numerical tolerance on
  well-scaled inputs. The FA3 paper (arXiv 2407.08608) reports
  reduced accumulated error compared to naive FP8 attention through
  its "block quantization" scheme.
- Do not enable FA3 FP8 attention while the rest of the model runs in
  BF16. The scale factors coming into attention will not line up with
  what FA3 expects, and you will see numerical drift with no
  throughput win to justify it.

The right pattern is: use FA3 in BF16 mode when the surrounding
model is BF16; use FA3 in FP8 mode when Transformer Engine has
already switched the model to FP8. Chapter 4 shows how.

## When not to use FlashAttention

Two cases where FA is not the right tool:

- **Very short sequences (`S` a few hundred).** Attention is a small
  fraction of the step and the kernel launch overhead of the fused
  block dominates. Use the SDPA `math` backend and put your effort
  into other buckets of the gap-to-peak budget.
- **Custom attention patterns not in the supported matrix.** Learned
  positional biases with unusual shapes, cross-attention with a very
  large key dimension, or research variants like linear attention.
  Consult the FA repo for the current supported-shape list;
  otherwise you are back on the fallback SDPA backend.

For everything else on Ampere or Hopper, FA is the default. On
Blackwell, use the FA version the repo indicates for SM100 support.

## Summary

- FlashAttention v2 for Ampere / A100 and as a Hopper fallback; v3
  for Hopper's WGMMA/TMA and for FP8 attention. Do not use v3 on
  A100 (Hopper-only) and do not use v2 on Hopper if v3 is available
  and packaged for your PyTorch build.
- Prefer `torch.nn.functional.scaled_dot_product_attention` with
  explicit `sdpa_kernel(FLASH_ATTENTION)` in the training loop; call
  `flash_attn_varlen_func` directly when you need sequence packing
  (chapter 5).
- Measure the lift in four steps: kernel-isolation microbench, full-
  step training measurement, profiler verification that the right
  kernel dispatched, and a short equivalence run against the
  previous backend for numerical sanity.
- FA3 FP8 attention is only correct when the surrounding model is
  also in FP8 through Transformer Engine (chapter 4). Do not mix
  FP8 attention into a BF16 model.
- Very short sequences and non-standard attention masks are the two
  cases where the FA fast path does not apply; SDPA falls back
  silently, so profile to confirm.
