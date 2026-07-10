# `torch.compile` with FSDP2 and DeepSpeed

Chapter 1 named "Python-side dispatch / graph breaks" as a 5–15 pp
MFU gap. That number is what an eager PyTorch training loop pays for
every op it dispatches through the Python interpreter: kernel launch
overhead, autograd graph construction, dtype coercion, memory
allocation. `torch.compile` (PyTorch 2's TorchDynamo + TorchInductor
stack, PT2 for short) fuses ops into larger CUDA kernels and
eliminates most of that overhead. This chapter is about installing it
correctly on top of FSDP2 and DeepSpeed, avoiding the two failure
modes (graph breaks and dynamic-shape recompiles), and knowing when
to escape into a Triton-authored kernel.

The primary references:

- **PyTorch `torch.compile` docs.**
  https://pytorch.org/docs/stable/torch.compiler.html
- **PyTorch 2.0 announcement / TorchDynamo overview.**
  https://pytorch.org/get-started/pytorch-2.0/
- **DeepSpeed `.compile()` doc.**
  https://www.deepspeed.ai/tutorials/compile/
- **Triton docs.** https://triton-lang.org/

## What PT2 actually does

Three components run in sequence when you call
`compiled = torch.compile(model)`:

- **TorchDynamo.** Intercepts Python bytecode at the frame level;
  when your compiled function is called, Dynamo traces the Python
  path taken and produces an FX graph of ATen ops. Anything Dynamo
  can not trace (a Python call it can not represent as an op — I/O,
  custom C extensions, some data-dependent control flow) causes a
  *graph break* that falls back to eager Python.
- **AOT Autograd.** Takes the forward FX graph, runs autograd
  symbolically, and produces a joint forward+backward graph. This is
  where the backward computation gets lifted into the compiled
  region instead of being generated per-op at runtime.
- **TorchInductor.** The backend that lowers the joint graph into
  fused Triton kernels (on CUDA) or C++ kernels (on CPU). Kernel
  fusion is where the actual speedup comes from — pointwise ops that
  used to be N separate CUDA kernels become one Triton kernel.

Two things are worth pinning down:

- **PT2 is a *compiler*, not a jit-wrap.** It produces a persistent
  compiled graph per input shape / dtype / device. Recompiles happen
  when the *shape* changes (or when Dynamo's guards fire); a
  well-configured PT2 pipeline recompiles a handful of times during
  a training run and stays cached from then on.
- **Kernel fusion is the mainstream win.** TorchInductor's fusion
  passes eliminate intermediate HBM writes between pointwise ops.
  For a transformer this hits every residual add, every activation
  function, every LN input/output, every dropout mask. The
  aggregate is often 1.2–1.5× step-time on the fused fraction.

Reported speedups from the PT2 announcement: 43% average on a
diverse benchmark set, higher on transformer training. Real-world
training runs typically see 10–20% step-time reduction from PT2 on
top of a well-tuned eager baseline; larger on smaller models where
launch overhead is a larger fraction of the step.

## The three modes

```python
compiled = torch.compile(model, mode="default")
compiled = torch.compile(model, mode="reduce-overhead")
compiled = torch.compile(model, mode="max-autotune")
```

- **`"default"`.** Fusion but no CUDA graphs, no autotuning. Safe
  starting point. Compatible with FSDP2 and dynamic shapes.
- **`"reduce-overhead"`.** Uses CUDA graphs to eliminate per-launch
  overhead. Best for small models where launch dominates step time.
  Requires static shapes; incompatible with FSDP2's all-gather
  because the collective buffer sizes change each step
  under some workloads. Do not use with FSDP2 as a first pass.
- **`"max-autotune"`.** Additionally autotunes matmul and conv
  kernels by benchmarking candidate implementations. Long compile
  time (minutes), highest steady-state throughput. Standard for
  production pretraining once the config is stable.

For a large-model training loop under FSDP2 the mainstream choice is
`mode="default"` first, then `mode="max-autotune"` once the config
is frozen and you can afford a long first-step warm-up.

## FSDP2 composition

FSDP2's `fully_shard` was designed with PT2 in mind. As of PyTorch
2.4, `fully_shard`-wrapped models are `torch.compile`-compatible with
some caveats:

- **Per-layer compilation.** The recommended pattern is to compile
  the *inner* blocks, not the outer root module. This keeps the
  compile-time graphs small (one graph per unique block, cached) and
  isolates recompiles.
- **Full-graph compilation.** PyTorch ≥ 2.4 supports full-graph
  compilation of the whole FSDP2-wrapped model in some
  configurations, but the compile times are long and any graph break
  falls back to per-op eager. Full-graph mode is worth measuring
  once your config is otherwise stable.

The per-layer pattern:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
import torch

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
)

for block in model.transformer.layers:
    fully_shard(block, mesh=dp_mesh, mp_policy=mp_policy)
fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)

# Compile after sharding. Per-block compile is the safe default.
for block in model.transformer.layers:
    block.compile(mode="default")
```

Two subtleties:

- **Order.** `fully_shard` first, then `compile`. Compile inspects
  the sharded module's `forward`; if you compile before sharding,
  Dynamo traces the pre-shard forward and the resulting graph is
  wrong at runtime.
- **`fullgraph=True`.** Passing `fullgraph=True` disables graph
  breaks — Dynamo will *error* rather than falling back. This is a
  debugging tool: run it once on a small config to surface every
  break, fix them (or explicitly allow-list them), then remove
  the flag for production.

<!-- needs-research: verify the exact PyTorch minimum-version requirement for full-graph FSDP2 compilation with the current release notes; the story has moved fast between 2.4, 2.5, and 2.6. -->

## Dynamic shapes

Training loops usually have one shape axis that varies: the
sequence length in the last batch of an epoch, or a variable
per-batch `cu_seqlens` under packing (chapter 5). Dynamo tracks
shape guards and *recompiles* whenever a shape leaves its
current bucket. Without care, a run can recompile once per unique
shape and pay compile time repeatedly.

Three defences:

- **`dynamic=True`.** Tell Dynamo up front to treat shapes as
  symbolic:

  ```python
  compiled = torch.compile(block, mode="default", dynamic=True)
  ```

  Dynamo generates one graph that is shape-agnostic; runtime
  guards fire only if a *rank* or dtype changes. Trades peak
  fused throughput (some shape-specialised fusions are unavailable
  when shapes are symbolic) for compile-time stability.
- **Bucket the dynamic axis.** Pad every batch to the nearest
  bucket boundary (`s ∈ {512, 1024, 2048, 4096, 8192}` for
  example). Dynamo compiles one graph per bucket; steady state
  is small and stable. This is how the `torchtitan` reference
  handles variable-length data.
- **Pack to a fixed length.** Under sequence packing (chapter 5)
  the packed length is fixed at `max_seqlen`; only the
  `cu_seqlens` layout varies. `cu_seqlens` is a shape guard on the
  attention op, so if the number of documents per pack varies you
  still get recompiles at the FA op. Passing
  `dynamic=True` on the FA op region is the mainstream fix.

## Graph breaks and how to diagnose them

Dynamo prints a warning per graph break by default. To turn it into
an error during development:

```python
import torch
torch._dynamo.config.error_on_graph_break = True
```

Or run with `TORCH_LOGS=graph_breaks python train.py` to see the
break sites without erroring.

Common break causes in training loops and the fixes:

| Cause                                          | Fix |
|-----------------------------------------------|-----|
| `print()` or `logger.info()` inside forward   | Move outside the compiled region |
| `if step % N == 0: do_something()` inside forward | Move the sampling logic outside |
| `.tolist()` / `.item()` on a tensor           | Replace with an in-graph op if possible |
| Data-dependent Python control flow            | Rewrite as `torch.where` or `torch.cond` |
| A custom op without a meta-kernel             | Register a meta-kernel or exempt the op |
| `contiguous()` on a mid-graph tensor          | Usually fine; Dynamo may still trace through |

The frequency of these is high on a first PT2 install. Budget an
hour to walk them out before comparing throughput to eager.

## DeepSpeed's `.compile()` wrapper

DeepSpeed ships its own `.compile()` method that wraps
`torch.compile` and reconciles it with ZeRO's parameter sharding.
See https://www.deepspeed.ai/tutorials/compile/. The main
integration points:

- **`engine.compile()` after `deepspeed.initialize()`**. Same
  order rule as FSDP2: shard first, compile second.
- **ZeRO-3 vs. compile.** ZeRO-3 partitions parameters and
  gathers them on demand; the gather / release calls are custom
  DeepSpeed ops. In recent DeepSpeed releases these are Dynamo-
  traceable, but the traceability has historically lagged
  fully_shard. Check the release notes for your version.
- **DeepSpeed CPU offload + compile.** Offload hooks fire on
  the module boundary; compile within blocks works, compile across
  the offload boundary does not.

If you own a DeepSpeed codebase and are just starting with PT2,
compile inner blocks first (same pattern as FSDP2). Full-model
compilation on ZeRO-3 remains fragile at time of writing;
expect graph breaks around the ZeRO gather ops.

<!-- needs-research: pull the current DeepSpeed + torch.compile compatibility caveats from the latest DeepSpeed release notes and update the recommendation. -->

## Triton as the escape hatch

TorchInductor lowers to Triton kernels on GPU. When Inductor's
fusion misses an op sequence you know is fusable, or when a specific
kernel is memory-bound and you want to hand-tune the tile shapes,
you can drop to Triton:

```python
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, w_ptr, y_ptr, n_cols, eps,
    BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    y = x * rstd * w
    tl.store(y_ptr + row_idx * n_cols + col_offsets, y, mask=mask)
```

You then wrap the kernel in a custom `torch.autograd.Function` and
register it as a custom op so Dynamo can trace through it (or leave
it as an opaque op that graph-breaks with an explicit boundary).

Triton is the right hammer when:

- You have identified a specific fusable region Inductor is
  missing. Inspect its output with `TORCH_COMPILE_DEBUG=1
  python train.py` (writes fused Triton kernels to disk) and see
  what it did or did not fuse.
- You have a memory-bound custom op (a fused optimizer step, a
  novel activation, a masked matmul) that no existing library
  ships.

Triton is the *wrong* hammer when:

- You want a matmul faster than cuBLAS. cuBLAS + Transformer
  Engine + FA are hand-tuned and better than any first-pass
  Triton matmul.
- You want a new attention variant. FA is the reference; extend
  FA before writing a new attention kernel.

Chapter 8 codifies the escalation contract: if you find yourself
authoring a Triton kernel that ships to production, the maintenance
burden lands on `ai-infra-performance-learning`, not on this track.

## Measuring the PT2 delta

The A/B protocol:

1. Eager baseline. Measure MFU, HFU, step time. Save the profiler
   trace.
2. Turn on `torch.compile(mode="default")` on the inner blocks
   only. Re-measure after skipping compile-time warm-up steps.
3. Turn on `dynamic=True` if there is a variable axis, or bucket
   the axis.
4. If the eager baseline had large launch overhead (small model,
   many small ops), try `mode="reduce-overhead"` off FSDP2 (single
   GPU or DDP only). Not usable with FSDP2.
5. For frozen production configs, try `mode="max-autotune"`.
   Compile time in minutes; steady-state throughput is the
   ceiling.

Expected steady-state lift on a well-tuned transformer training
step: 8–15% step-time reduction from `mode="default"`, another
2–5% from `mode="max-autotune"`. If you see less, you either had
a very well-fused eager baseline (rare) or Dynamo is graph-breaking
somewhere important. Check with `TORCH_LOGS=graph_breaks`.

If MFU *dropped* after enabling compile, the almost-certain cause is
recompiles from a shape you did not bucket. Look for repeated
"recompiling" log lines.

## The measurement surface

Two profiler hooks pay off for PT2 debugging:

- **`TORCH_COMPILE_DEBUG=1`.** Writes the Inductor-lowered Triton
  kernels and the FX graph to a `torch_compile_debug/` directory.
  You can read the kernels and see exactly what got fused.
- **`torch._logging.set_logs(dynamo=logging.INFO,
  inductor=logging.INFO)`.** Turns on per-graph compile-time
  logging.

For step-time attribution, the standard profiler recipe from
https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
still works; PT2 marks its compiled regions in the trace.

## Summary

- `torch.compile` (PT2 = Dynamo + AOT Autograd + Inductor) fuses
  pointwise ops into Triton kernels and eliminates Python-side
  dispatch overhead. Typical training-step lift: 8–15% at
  `mode="default"`, 10–20% at `mode="max-autotune"`.
- Compose with FSDP2 by sharding first, then compiling per block.
  PyTorch ≥ 2.4 for full-graph FSDP2 compilation. Use
  `fullgraph=True` during development to surface graph breaks.
- Dynamic shapes cause recompiles. Fix with `dynamic=True`,
  bucket the axis, or pack sequences to fixed length. Watch
  `TORCH_LOGS=graph_breaks` and `torch_compile_debug/`.
- `mode="reduce-overhead"` uses CUDA graphs; incompatible with
  FSDP2 in most configurations. `mode="max-autotune"` is the
  steady-state ceiling but has minutes of compile time.
- DeepSpeed's `engine.compile()` follows the same rules. ZeRO-3 +
  compile is more fragile than FSDP2 + compile; check the
  DeepSpeed release notes for the current caveats.
- Triton is the escape hatch when Inductor misses a fusion or you
  need a hand-tuned memory-bound kernel. If a Triton kernel ships to
  production, that is an `ai-infra-performance-learning`
  hand-off (chapter 8).
- Expect launch-overhead to matter less on H100 than on smaller
  GPUs; the MFU lift from compile is real but smaller than the lift
  from FA v3 or FP8 on the same run.
