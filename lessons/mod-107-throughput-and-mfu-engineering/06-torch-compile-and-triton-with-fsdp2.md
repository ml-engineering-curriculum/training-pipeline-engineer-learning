# `torch.compile` (PT2) and Triton with FSDP2 / DeepSpeed

`torch.compile` is PyTorch's opt-in graph compiler. It takes eager-
mode Python and produces fused kernels (via TorchInductor and, for
GPU targets, generated Triton). For a well-behaved LLM training
step, `compile` typically reclaims 10–20% of MFU on top of BF16 +
FlashAttention, mostly by fusing the small elementwise ops (norms,
biases, activations, dropout) that would otherwise each be a
separate HBM round-trip.

This chapter is about how to turn `compile` on correctly under
FSDP2 (and DeepSpeed), how to *consume* published Triton kernels
without authoring them, and how to keep the compile cache from
becoming a productivity tax.

## What `torch.compile` actually does

PyTorch's Just-In-Time compilation stack has three layers (see the
PyTorch 2 paper: Ansel et al., 2024, "PyTorch 2: Faster Machine
Learning Through Dynamic Python Bytecode Transformation and Graph
Compilation", ASPLOS'24):

- **TorchDynamo** intercepts Python bytecode during execution and
  captures a graph of tensor ops. When it can't (unsupported
  Python), it produces a "graph break" and falls back to eager for
  that fragment.
- **AOTAutograd** captures both forward and backward graphs
  ahead-of-time, so the compiler sees the full computation.
- **TorchInductor** is the default backend. It lowers the captured
  graph to fused kernels — Triton on GPU, C++ on CPU — cached to
  disk keyed on graph identity, input shapes, and stride patterns.

The user-facing entry point is a one-line wrapper:

```python
model = torch.compile(model, mode="default", fullgraph=False, dynamic=None)
```

Key knobs:

- **`mode`.** `"default"` for general use; `"reduce-overhead"` uses
  CUDA graphs for reduced launch overhead (small models, small
  batch); `"max-autotune"` spends more time picking kernel schedules
  and typically wins another few percent on large models — measure
  before committing.
- **`fullgraph`.** `False` (default) allows graph breaks; `True`
  errors if a break would happen. For training LLMs, use `False`
  during bring-up and inspect the break points; try `True` after
  the code is clean, to catch regressions.
- **`dynamic`.** Controls dynamic-shape handling. `None` lets
  Inductor decide based on observed shape variation; explicit `True`
  or `False` forces the choice. Sequence-length variation from
  packing is a dynamic-shape case (see below).

## Composing with FSDP2

FSDP2 and `torch.compile` compose, but the *order* and *granularity*
matter. The current guidance (see the FSDP2 tutorial and the
"Compiling FSDP2" section of the PyTorch docs) is:

- **Wrap FSDP2 first, then compile the top-level module.** Since
  PyTorch 2.4+, compiling the whole FSDP2-wrapped model works with
  the "regional compile" that respects FSDP2's `_pre_forward` /
  `_post_forward` hooks. Earlier versions required you to compile
  block-by-block.

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

for block in model.blocks:
    fully_shard(block, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                       reduce_dtype=torch.float32))
fully_shard(model, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                   reduce_dtype=torch.float32))

# Then compile.
model = torch.compile(model, mode="default")
```

- **Prefer block-level compile as a fallback.** If top-level compile
  produces graph breaks you cannot eliminate (custom Python hooks
  that Dynamo cannot trace, control-flow branches), compile each
  block individually:

  ```python
  for block in model.blocks:
      fully_shard(block, ...)
      block._forward_impl = torch.compile(block._forward_impl)
  fully_shard(model, ...)
  ```

  This gives you `L` cached compiled units instead of one, at the
  cost of losing whole-model fusion opportunities across block
  boundaries (usually small).

- **Do not compile the whole model *before* wrapping with FSDP2.**
  `fully_shard` mutates the module hierarchy; compiling first
  produces a graph tied to the un-sharded shape, which Dynamo will
  invalidate on the first FSDP2 all-gather.

- **Check-list for a clean compile:**
  - `torch._dynamo.config.suppress_errors = False` during bring-up
    to see the actual break reasons instead of silent falls-back.
  - `TORCH_LOGS="graph_breaks,recompiles"` env var to see when and
    why a recompile fires.
  - `TORCHINDUCTOR_MAX_AUTOTUNE=1` (equivalent to `mode="max-
    autotune"`) if you want to try the more aggressive schedule
    search.

## Composing with DeepSpeed

DeepSpeed (Rasley et al., 2020, arXiv 1910.02054 for the original
ZeRO paper; https://www.deepspeed.ai/ for docs) predates
`torch.compile` and has its own wrapping model. Compiling under
DeepSpeed works but the composition surface is thinner than under
FSDP2:

- **ZeRO stages 1 and 2 compose with `torch.compile` naturally**:
  DeepSpeed hooks live outside the compiled region. Wrap first,
  compile the underlying `nn.Module` after `deepspeed.initialize`.
- **ZeRO stage 3 is where issues appear.** Stage 3 shards
  parameters and inserts pre-forward/pre-backward hooks similar to
  FSDP2. These hooks currently interact poorly with Dynamo tracing
  in some versions; check the DeepSpeed release notes for your
  version and be prepared to compile block-by-block instead of at
  the top level.
- **DeepSpeed's own kernel fusions.** DeepSpeed ships fused
  layernorm, fused Adam, and fused kernels for other common ops.
  These already do the fusion `torch.compile` would do; measure
  whether `compile` still buys you anything on top or whether the
  overlap makes it net-neutral.

The rule of thumb: use `torch.compile` opportunistically on top of
DeepSpeed; make FSDP2 the default choice when possible. FSDP2 was
designed with compile composition in mind; DeepSpeed was not.

## Consuming Triton kernels (do this — do not author them)

Chapter 8 draws the boundary sharp: authoring novel Triton kernels
is the ai-infra-performance-engineer role. What *you* do as a
training-pipeline engineer is integrate published kernels
correctly. Two patterns:

### 1. TorchInductor-generated Triton (automatic)

Whenever `torch.compile` fuses ops on GPU, it *emits* Triton for
them. You do not read or write that Triton — Inductor manages the
cache. Your job is to make sure the graph is compilable so the
fusion happens. See the `TORCH_LOGS` guidance above.

### 2. Explicit Triton kernels from `triton` (manual integration)

Kernels from the Triton ecosystem — the `triton` PyPI package's
tutorial suite, `xformers`, `TransformerEngine`, `unsloth`,
`flash-attn` variants — are Python functions you import and call.
Integration pattern:

```python
# Example: use a published fused SwiGLU kernel from some library
from some_kernel_lib import swiglu_fused

class SwiGLU(nn.Module):
    def forward(self, x, w1, w2):
        return swiglu_fused(x, w1, w2)   # Triton kernel under the hood
```

Two rules for these integrations:

- **Wrap them as `torch.compile`-compatible ops.** If the kernel is
  registered as a `torch.library` op with a "meta" implementation
  (shape/dtype inference), `torch.compile` can include it in a
  larger graph without a break. If not, it becomes a graph break.
- **Do not smuggle Triton code into your codebase.** If a library's
  Triton kernel is what you want, depend on the library; do not
  copy the kernel and modify it. Kernel authors update the code for
  correctness and performance regularly; a fork rots.

## Compile-time and recompile costs

`torch.compile` is not free: the first time a graph runs, Inductor
compiles it, which takes tens of seconds to minutes depending on
graph size. Subsequent runs hit the cache and are near-instant.
Three failure modes:

- **Recompile storm.** Every time the input shape changes, Dynamo
  invalidates the graph and recompiles. On a training loop where
  micro-batch shape or sequence length varies, this is fatal — the
  loop compiles every step. Fix by either (a) padding shapes to a
  small set of canonical shapes, (b) `torch.compile(model,
  dynamic=True)` to enable dynamic-shape compilation, or (c)
  bucket-batching the loader.
- **Cache thrash across ranks.** Every rank compiles independently
  by default. On a 128-rank job, that is 128 concurrent Inductor
  compilations at step 1, hammering the filesystem and slowing the
  first step to minutes. Set `TORCHINDUCTOR_CACHE_DIR` to a per-
  rank local NVMe path (not a shared parallel FS) or pre-populate a
  shared cache once from one rank.
- **Graph breaks silencing the win.** If half the block has a
  graph break in the middle, Inductor fuses two smaller sub-graphs
  and misses the cross-break fusion. Use `TORCH_LOGS=graph_breaks`
  to find and eliminate breaks.

The right cadence for a production loop: compile once, log the
compile time per rank, cache to local NVMe, and expect the second
step onward to run at the compiled cost. If step 1 takes 3 minutes
and step 2 takes 200 ms, that is correct behavior.

## Interaction with FlashAttention and TE

- **FlashAttention:** SDPA's FlashAttention backends are already
  fused kernels; `torch.compile` treats them as opaque ops and does
  not try to fuse anything inside them. No conflict, no additional
  win from compile on the attention block itself.
- **Transformer Engine (chapter 4):** TE modules are also opaque to
  compile (they call into their own C++/CUDA path). Compile still
  helps for anything outside the TE modules — data pipeline,
  loss computation, custom layers.
- **Activation checkpointing (chapter 5):** `torch.utils.checkpoint`
  works under `torch.compile`, but you must use
  `use_reentrant=False`. The reentrant path breaks Dynamo tracing.

## Signals that compile is (or is not) doing its job

- **Step time drops on step 2 vs. step 1** and continues at the new
  level. Compile worked and the cache is warm.
- **Every step recompiles.** Dynamic shapes are firing.
  `TORCH_LOGS=recompiles` will tell you which guard failed.
- **Step time on compile is identical to eager.** Every op is
  probably in a graph break. `TORCH_LOGS=graph_breaks` to find them,
  or Inductor is generating kernels that are no better than the
  eager path for this shape (rare but possible for very small
  kernels).
- **First step takes many minutes on a large model.** Cache is not
  shared or is being hit by many ranks at once. Set
  `TORCHINDUCTOR_CACHE_DIR` correctly.

## Summary

- `torch.compile` wraps TorchDynamo + AOTAutograd + TorchInductor,
  generates fused kernels (Triton on GPU), and caches them keyed
  on graph identity and shapes. Turn it on with a one-liner *after*
  FSDP2 wrapping, not before.
- Under FSDP2 (PyTorch 2.4+), top-level compile works. Fall back to
  block-level compile if top-level produces persistent graph
  breaks. Under DeepSpeed ZeRO-3, block-level is often the safer
  starting point.
- Consume published Triton kernels; do not author them (chapter 8).
  Prefer libraries whose kernels register as `torch.library` ops
  so they compose with compile.
- Recompile storms and shared-FS cache thrash are the two operational
  failure modes. Use dynamic shapes or shape bucketing for the
  former; local-NVMe cache directories for the latter.
- Compile stacks on top of FlashAttention, TE, and AC; it does not
  replace them. The typical incremental win is 10–20% MFU on top
  of a well-tuned BF16 + FA setup.
