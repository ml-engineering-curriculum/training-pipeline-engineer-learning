# JAX, Flax, and the GSPMD Sharding Model

Every stack up to this point in the module has been imperative:
you (or a config file) place a collective at a specific point in
the code, and NCCL runs it. JAX inverts the model. You write a
pure function of parameters and inputs, annotate its arguments
with *where they live*, and the XLA compiler — via GSPMD
(Xu et al., 2021, "GSPMD: General and Scalable Parallelization for
ML Computation Graphs") — rewrites the program so that the
required collectives happen at the right points. You never write
`all_gather`. The compiler does.

This chapter builds the JAX mental model, walks the current API
for distributed arrays (`jax.Array`, `Mesh`, `NamedSharding`,
`jax.jit(in_shardings=...)`, `shard_map`), and contrasts the JAX
approach with PyTorch so the mode-switch in exercise 5 doesn't
land cold.

## The mental model in one sentence

*Parallelism in JAX is a property of tensors, and the compiler is
responsible for materializing the collectives that make the
computation valid.*

Everything else follows from that. If a matmul's operands are
sharded on incompatible axes, the compiler inserts an
`all-gather` (or a `reduce-scatter`, or an `all-to-all`) to make
them compatible. If two `jit`-compiled functions consume the same
sharded array, the compiler propagates the sharding through the
graph. You describe the *data placement*; the compiler figures
out the *communication*.

## The pieces

### `jax.Array` — unified device-agnostic array

Since JAX 0.4 (2023), the single array type is `jax.Array`. A
`jax.Array` can be:

- **Committed to a device** — lives on one accelerator, behaves
  like a `torch.Tensor` on `cuda:0`.
- **Sharded across devices** — a "global array" whose value is
  a collection of local shards on multiple devices, described by
  a `Sharding` object. From a Python perspective, it *looks like
  one array* with a global shape; you index it, `jnp.sum` over
  it, `jax.jit` compiles functions over it. The compiler and
  runtime take care of the distributed storage.

This unification replaced JAX's older split between
`DeviceArray`, `ShardedDeviceArray`, and `GlobalDeviceArray`.

### `Mesh` — the device grid

`jax.sharding.Mesh` is the JAX analogue of PyTorch's
`DeviceMesh`. You name the axes and assign devices to a
multi-dimensional grid:

```python
import jax
from jax.sharding import Mesh
import numpy as np

devices = np.array(jax.devices()).reshape(2, 4)  # 8 GPUs → (dp=2, tp=4)
mesh = Mesh(devices, axis_names=("dp", "tp"))
```

The mesh axes correspond exactly to the parallelism axes you would
name in PyTorch: `("dp", "tp")`, `("dp", "tp", "pp")`,
`("dp", "fsdp", "tp")`, and so on. The compiler uses the mesh to
choose which collective goes on which axis.

### `PartitionSpec` and `NamedSharding` — where the data lives

`PartitionSpec` describes how an array's dimensions are sharded
onto mesh axes:

- `PartitionSpec("dp", "tp")` — dim 0 sharded on `dp`, dim 1
  sharded on `tp`.
- `PartitionSpec("dp", None)` — dim 0 sharded on `dp`, dim 1
  replicated across all mesh axes.
- `PartitionSpec(None, ("tp", "dp"))` — dim 0 replicated, dim 1
  sharded on the composition of both mesh axes (a "sub-mesh").
- `PartitionSpec()` — fully replicated.

`NamedSharding(mesh, pspec)` combines the mesh and the
partition-spec into a placement you can attach to a `jax.Array`:

```python
from jax.sharding import PartitionSpec as P, NamedSharding

sharding = NamedSharding(mesh, P("dp", "tp"))
array = jax.device_put(np.zeros((8, 16)), sharding)
```

That `array` is now distributed across the 8 devices, with the
first dim sharded on the `dp` axis and the second on `tp`.

### `jax.jit(in_shardings=..., out_shardings=...)` — the GSPMD entry point

The former `pjit` API (still available as an alias in current
JAX) is now folded into `jax.jit`. You compile a pure function
with explicit input and output shardings, and the compiler
propagates through the function body:

```python
@partial(jax.jit,
         in_shardings=(NamedSharding(mesh, P(None, "tp")),  # weight sharded on TP
                       NamedSharding(mesh, P("dp", None))), # input sharded on DP
         out_shardings=NamedSharding(mesh, P("dp", "tp")))
def matmul(w, x):
    return x @ w
```

Two things happen when this compiles:

1. **GSPMD figures out where the collectives go.** If the matmul
   requires the `tp`-sharded operand to be reconstructed before
   the local computation, XLA inserts an `all-gather` on the `tp`
   axis. If the result is a partial sum requiring reduction, XLA
   inserts an `all-reduce` (or `reduce-scatter`, depending on
   `out_shardings`).
2. **The function is specialized to the mesh.** Recompilation
   happens if you call it with the same shapes but a different
   `mesh` or `sharding`. That is the JAX version of "recompile
   for a new configuration".

Consequence: the `jit`ted function is a distributed program even
though its Python code looks single-device. When you profile the
run, you see collectives on the timeline; when you read the code,
you see `x @ w`.

### `shard_map` — the manual escape hatch

Some computations do not fit the GSPMD model well: they require
per-device control flow, or the compiler's choice of collective
placement is wrong for the specific problem. For those,
`jax.experimental.shard_map` (post-0.4) lets you write a
per-device function that operates on *local* shards, with
explicit collectives:

```python
from jax.experimental.shard_map import shard_map

@partial(shard_map, mesh=mesh,
         in_specs=(P("dp", None), P(None, "tp")),
         out_specs=P("dp", "tp"))
def matmul_manual(x_local, w_local):
    # x_local has shape (batch/dp, hidden), w_local has shape (hidden, out/tp).
    # Both are local; no compiler-inserted collectives.
    return x_local @ w_local
```

Inside `shard_map`, `x_local` and `w_local` are the *local shard*
of what `x` and `w` were globally. If you want to communicate,
you call `jax.lax.psum` / `jax.lax.all_gather` /
`jax.lax.pshuffle` explicitly with a mesh axis. This is the JAX
equivalent of writing PyTorch collectives by hand.

Use `shard_map` when:

- You need a collective the GSPMD partitioner will not insert
  correctly (a specific `all-to-all` for expert-parallel MoE, a
  ring reduction, an `all-gather-then-permute`).
- You are writing a custom kernel and need explicit per-device
  computation.
- You want to hand-tune the overlap of compute and communication.

Use `jax.jit(in_shardings=...)` for everything else. Most modern
JAX training code (MaxText for GPUs, Flax's LM examples) uses
both, with GSPMD as the default and `shard_map` in a few surgical
places (attention, MoE routing).

## Flax and the model layer

Flax is the neural-network library sitting on top of JAX. Two
things to know:

- **`flax.linen`** (also called Linen) — the mature module system.
  Modules are `@dataclass`-style classes with a `__call__` method,
  and you use `.init(rng, x)` to obtain a parameter pytree.
- **`flax.nnx`** — the newer imperative module system introduced
  in 2024, with mutable state and closer feel to PyTorch. `<!--
  needs-research: NNX is evolving; confirm whether NNX is
  production-ready for 3B+ pretraining with the current Flax
  release before recommending it as the default for the module's
  exercise. -->`

A Flax parameter pytree is just a nested dict of `jax.Array`s.
Sharding a Flax model amounts to constructing a matching
`NamedSharding` pytree and calling `jax.device_put`:

```python
from flax import linen as nn

class DecoderBlock(nn.Module):
    hidden: int
    heads: int
    def __call__(self, x):
        ...

model = DecoderBlock(hidden=2048, heads=16)
params = model.init(rng, x_example)   # nested dict of jax.Arrays on host

# Build a matching shardings pytree.
def get_sharding(p):
    # e.g. shard weight[0] on "tp" if 2-D, replicate biases, etc.
    if p.ndim == 2:
        return NamedSharding(mesh, P(None, "tp"))
    return NamedSharding(mesh, P())

param_shardings = jax.tree_util.tree_map(get_sharding, params)
params = jax.device_put(params, param_shardings)
```

This is the JAX equivalent of PyTorch's `parallelize_module` +
`fully_shard` step: you have declared placement of every leaf.

## The training step

A JAX training step is a pure function of `(params, opt_state,
batch)`, returning updated `(params, opt_state)` and a loss:

```python
import optax

optim = optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.1)
opt_state = optim.init(params)

def loss_fn(params, batch):
    logits = model.apply(params, batch["input_ids"])
    return cross_entropy(logits, batch["targets"])

@partial(jax.jit,
         in_shardings=(param_shardings, opt_state_shardings, batch_shardings),
         out_shardings=(param_shardings, opt_state_shardings, ()))
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, new_opt_state = optim.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
```

Every call to `train_step(params, opt_state, batch)` returns
sharded outputs. The runtime keeps the parameters and optimizer
state on device across steps — `jax.jit` is smart about donation
so buffers are not copied. The `donate_argnums` argument, or
`in_shardings` with the `donate` field set, tells the compiler
which inputs it may overwrite in place.

You never wrote an `all-gather`. GSPMD did.

## Contrast with PyTorch

Side by side, on the same 3-billion-parameter transformer:

| Concern             | PyTorch (FSDP2 + TP)                                                       | JAX (jit + NamedSharding)                                     |
|---------------------|----------------------------------------------------------------------------|---------------------------------------------------------------|
| Where sharding is declared | On modules (`fully_shard`) and layer plans (`parallelize_module`) | On arrays (`NamedSharding`) and function inputs (`in_shardings`) |
| Where collectives are inserted | By the FSDP2 backend and TP layer wrappers (imperative Python) | By the XLA compiler (GSPMD partitioner)                       |
| Model definition    | `nn.Module` with parallelism-aware layers or `parallelize_module` plan     | Pure `flax.linen.Module` — parallelism is *not* baked into layers |
| Debugging placement | `param.placements` on DTensors                                             | `jax.debug.visualize_array_sharding(array)` or `array.sharding` |
| Compilation         | Eager by default, `torch.compile` optional                                 | Everything under `jit` is compiled by XLA                     |
| Recompilation triggers | Rare — dynamic shapes work                                               | Any shape/mesh/sharding change triggers a recompile           |
| Error surface       | Python stack + NCCL logs                                                   | HLO dumps + `jax.debug.print` traces                          |
| Time-to-first-step  | Fast — imperative                                                          | Slower — first `jit` compile dominates                        |
| Steady-state throughput | Excellent when overlap is tuned                                        | Excellent — GSPMD does aggressive collective fusion           |

Neither model is objectively better. The JAX model shines when
the parallelism strategy is regular (dense TP + DP + FSDP on a
transformer) because the compiler can globally optimize the
program. The PyTorch model shines when the parallelism is
irregular or when your team wants to introspect the loop step by
step in a familiar language.

## The observability shift

The single biggest source of engineer friction switching between
PyTorch and JAX is how you *see* what your program is doing.

- **Program shape.** In PyTorch you can `pdb` into `training_step`
  and inspect `param.grad`. In JAX, `train_step` is `jit`-compiled
  after the first call and there is no Python-level breakpoint
  inside; you use `jax.debug.print` (compiled-aware) or trace at
  the pre-jit level.
- **Sharding.** In PyTorch you read `param.placements`. In JAX
  you read `array.sharding` and use
  `jax.debug.visualize_array_sharding(array)` to see a color-coded
  layout in the terminal.
- **Collectives.** In PyTorch you see NCCL kernels tagged by
  `torch.profiler`. In JAX you see them in the HLO dump
  (`jax.jit(...).lower(...).compile()`
  and inspect `hlo_text`) and in traces via
  `jax.profiler.start_trace(...)` /
  `jax.profiler.stop_trace(...)` viewed in TensorBoard or Perfetto.
- **Errors.** A PyTorch shape mismatch is a Python `RuntimeError`.
  A JAX sharding mismatch is often an `XlaRuntimeError` referring
  to an HLO instruction. Reading HLO is a skill you have to build.

## The GPU story specifically

Historically JAX targeted TPUs first. On GPUs, the story has
matured but still has caveats worth naming:

- **Runtime.** On GPUs, XLA delegates collectives to NCCL. When
  you profile, you still see NCCL kernels on the timeline.
- **Precision.** Mixed precision in JAX is dtype-driven: cast
  parameters to `bfloat16`, keep the optimizer state in
  `float32`. There is no `MixedPrecisionPolicy`-style knob;
  precision is a property of the pytree.
- **`torch.compile`-style fusion.** JAX/XLA does this natively —
  matmul + bias + activation fuse without user intervention. On
  H100, XLA integrates with cuBLAS and TransformerEngine kernels
  for FP8 (via `jax.experimental.pallas` and TE's JAX bindings).
- **Reference codebases.** MaxText (`github.com/google/maxtext`)
  is Google's reference LLM pretraining implementation and runs
  on both TPU and GPU. Meta's Levanter and Meta's XLA-on-GPU work
  are other GPU-first JAX training examples.

## When to reach for JAX

- **You want compiler-driven sharding** and are willing to invest
  in reading HLO to debug it.
- **Your team already has JAX expertise** — Google-DeepMind alumni,
  research code that already targets JAX.
- **You are training on TPUs**, where JAX is the primary supported
  path and PyTorch/XLA is a second-class citizen.
- **Your workload benefits from XLA fusion** more than from
  imperative control — dense transformer training is a good fit.

## When *not* to reach for JAX

- **Your model has heavy dynamic control flow** — variable-length
  sequences without padding, dynamic MoE routing that GSPMD does
  not model well. `shard_map` can rescue some of this; not all.
- **Your team has zero JAX experience** and the training work is
  time-critical. The debugger, HLO, and compilation-cache
  ergonomics are a real learning curve.
- **You want to use HuggingFace models directly.** HF Transformers
  has partial Flax support but the PyTorch path is the maintained
  one; JAX ports lag.

## Summary

- JAX inverts the sharding model: parallelism is a property of
  tensors, and the XLA compiler (via GSPMD) figures out the
  collectives to insert.
- `Mesh` names the device grid; `PartitionSpec` + `NamedSharding`
  describes where each array's dimensions live;
  `jax.jit(in_shardings=..., out_shardings=...)` compiles a pure
  function under those constraints.
- `shard_map` is the manual escape hatch for cases where GSPMD's
  choice of collectives is wrong.
- The mental-model contrast with PyTorch is real: PyTorch is
  imperative and DTensor-first; JAX is compiler-driven and
  Array-first. Neither is better in general; they trade time-to-
  first-step for steady-state fusion quality.
- Debugging shifts from Python + NCCL logs to HLO dumps + JAX
  profiling traces. Plan for that learning curve when adopting
  JAX in a PyTorch-native team.
