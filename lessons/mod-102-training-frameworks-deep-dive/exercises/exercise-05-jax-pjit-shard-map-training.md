# exercise-05: JAX `jit` + `shard_map` Training on 8 GPUs

**Estimated effort:** 4 hours

## Objective

Build a JAX/Flax training loop for a small transformer, shard it
across 8 GPUs using `jax.jit(in_shardings=..., out_shardings=...)`
(the current form of what was `pjit`), and use `shard_map` on the
attention block to insert one collective by hand. Compare the
resulting program — mental model, debugging surface,
observability — against the PyTorch FSDP2 leg of exercise 1. The
deliverable is the artifact that lets you argue for or against
JAX in a decision doc from firsthand experience rather than from
Twitter takes.

## Prerequisites

- Chapter 7 of this module.
- A working JAX + Flax installation with GPU support and 8 visible
  GPUs. The JAX docs' "Distributed arrays and automatic
  parallelization" tutorial is a helpful pre-read.
- Comfort with pure functions and immutable pytrees. JAX is not a
  place to fight the paradigm.
- Exercise 1 completed (your FSDP2 leg is the PyTorch reference
  you will compare against).

## Problem statement

Your team is PyTorch-native but the next hardware refresh may
include TPUs, at which point JAX becomes the primary supported
path (chapter 7). Rather than face-plant into JAX two weeks
before the migration, you want to build one small-but-realistic
JAX training loop now — small enough to fit in an afternoon,
realistic enough that you understand what the compiler is
actually doing.

## Requirements

### The model

A tiny decoder-only transformer implemented in Flax (Linen is
fine; NNX also fine if your JAX version supports it well
enough). Suggested shape: `num_layers=6`, `hidden=512`,
`ffn=2048`, `heads=8`, `vocab=1024`, `seq_len=256`. Aim for
~10–30M parameters — small enough that a full step compiles fast
and you can iterate.

Requirements:

- Standard transformer components: multi-head attention with a
  simple causal mask, an MLP with SwiGLU or GELU, RMSNorm.
- Pure `flax.linen.Module` — no in-place state.
- A `loss_fn(params, batch)` that returns the mean cross-entropy
  loss over the batch.

### The sharding plan

Build a 2-D mesh over your 8 GPUs:

```python
import jax, numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

devices = np.array(jax.devices()).reshape(2, 4)
mesh = Mesh(devices, axis_names=("dp", "tp"))
```

Apply a sharding plan to the parameters:

- Attention `qkv` projections: `PartitionSpec(None, "tp")` — the
  `hidden` dim is replicated, the `heads*head_dim` dim is
  sharded on the TP axis.
- Attention output projection: `PartitionSpec("tp", None)` —
  input dim sharded on TP, output dim replicated.
- MLP up-projection: `PartitionSpec(None, "tp")`.
- MLP down-projection: `PartitionSpec("tp", None)`.
- Norms and biases: `PartitionSpec()` (fully replicated).

Apply a batch sharding to inputs: `PartitionSpec("dp", None)` on
the leading batch dim, replicated over TP.

### Part A — The GSPMD training step

Write and compile the training step:

```python
import optax
import functools

optim = optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.1)
opt_state = optim.init(params)

@functools.partial(
    jax.jit,
    in_shardings=(param_shardings, opt_state_shardings, batch_shardings),
    out_shardings=(param_shardings, opt_state_shardings, NamedSharding(mesh, P())),
)
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, new_opt_state = optim.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
```

Requirements:

- `params`, `opt_state`, and `batch` are placed on the mesh via
  `jax.device_put` before the first `train_step` call.
- Print `train_step.lower(params, opt_state, batch).compile()`
  HLO summary once. Identify at least one `all-gather` or
  `all-reduce` in the HLO (the compiler-inserted collectives).
- Run 100 steps on a synthetic dataset (random token IDs, fixed
  seed). Log loss per step.

### Part B — Add `shard_map` on attention

Rewrite the attention block using
`jax.experimental.shard_map.shard_map` so the softmax-over-heads
is expressed as explicit per-device computation, with one
explicit `jax.lax.psum` if you shard across the head dimension
(or one `jax.lax.all_gather` if you prefer to reconstitute
before the softmax; document your choice).

The point is not to make attention faster; it is to *understand*
what `shard_map` gives you that GSPMD does not. Requirements:

- Wrap the attention forward function in `shard_map` with
  `in_specs` and `out_specs` that match your `NamedSharding`
  plan.
- Include *exactly one* explicit collective inside the
  `shard_map` function. Choose it deliberately and justify in the
  write-up.
- Run 100 steps again and confirm the loss curve matches the
  Part A loss curve within numerical tolerance.

### Part C — Observability

Capture the JAX equivalents of what `torch.profiler` gave you in
exercise 1:

- Print `array.sharding` for `params`, `opt_state`, and `batch`.
  Also run `jax.debug.visualize_array_sharding(params['...'])`
  for one representative weight and paste the output.
- Emit a JAX profile trace with `jax.profiler.start_trace(dir)` /
  `jax.profiler.stop_trace()` over 20 steady-state steps. Open it
  in TensorBoard or Perfetto and identify the collectives on the
  timeline.
- Dump the compiled HLO for `train_step` (via
  `train_step.lower(...).compile().as_text()`) to a file. Search
  it for `all-gather`, `all-reduce`, and `reduce-scatter` and
  count how many appear per training step.

### Part D — The PyTorch/JAX contrast write-up

Deliver a 500–700-word `jax-vs-pytorch-reflection.md` covering:

1. **The mental-model shift.** Where in your code did you specify
   sharding, and where did the collectives *actually* appear
   (from the HLO)? Contrast with FSDP2 in exercise 1, where you
   applied `fully_shard` and the collectives appeared inside the
   FSDP2 machinery.
2. **Time-to-first-step.** Wall-clock time from `python train.py`
   to first steady-state step in JAX vs. FSDP2. JAX will lose
   this comparison; explain why (`jit` compilation).
3. **Debugging surface.** Take one failure you introduced
   intentionally (a wrong `PartitionSpec`, or a batch shape
   mismatch) and describe what the error looked like — an XLA
   HLO error message, a Python-level shape mismatch, or a silent
   miscompilation. Contrast with the equivalent PyTorch error.
4. **Observability.** How did viewing `jax.profiler` differ from
   viewing `torch.profiler`? What did you like; what did you find
   worse?
5. **Recommendation.** Under what circumstances would you argue
   for JAX in a training-team RFC? Under what circumstances
   would you argue against it? Reference chapter 8's rubric.

## Acceptance criteria

- A working JAX/Flax training loop that runs 100 steps on 8 GPUs
  without divergence.
- Both Part A (GSPMD) and Part B (`shard_map`) versions produce
  matching loss curves within numerical tolerance.
- HLO evidence (a code excerpt or a screenshot) showing at least
  one compiler-inserted collective.
- Sharding-visualization output from
  `jax.debug.visualize_array_sharding` for at least one
  parameter.
- Write-up cites at least one primary JAX doc page (JAX
  distributed arrays tutorial, `shard_map` docs, or the JAX SPMD
  guide).

## Starter guidance

- **Do not port your 3B model from exercise 1.** A small model
  compiles fast, iterates fast, and is easier to reason about.
  The point is the mental model, not the throughput.
- **`jax.device_put` returns a `jax.Array` in the correct
  placement.** Do this *once* per epoch of training; do not
  re-shard every step or you will pay compilation and transfer
  cost repeatedly.
- **`jax.jit` with `in_shardings` will recompile on shape or
  sharding change.** Do all your placement decisions *before* the
  training loop and keep shapes fixed inside it.
- **`shard_map` is stricter than `jit`.** It will not silently
  reshape; when you get an `in_specs`/`out_specs` shape error,
  the message is usually accurate — read it.
- **Read the compilation cache.** JAX caches compiled programs to
  a directory (`JAX_COMPILATION_CACHE_DIR`). Use it to avoid
  re-compiling on every rerun.
- **When in doubt, print `array.sharding`.** JAX is honest about
  where data lives; the friction is in mapping "where" to "which
  collective the compiler chose".
- **Do not compare throughput between JAX and PyTorch on a toy
  model.** The comparison is meaningless because the model is too
  small to hide compilation and startup costs. Focus the compare
  on the mental model and observability.

## Stretch goals

- **Compose the training loop with `jax.experimental.multihost_utils`**
  for a multi-node run. This introduces the notion of a "global
  array" spanning multiple hosts and is the piece that scales the
  paradigm to real cluster sizes.
- **Enable `flax.nnx`** (the newer, more imperative Flax API) and
  reimplement the model with it. Compare the ergonomics against
  Linen.
- **Enable XLA GPU FP8** via TransformerEngine's JAX bindings on
  H100 and record whether the loss curve stays within tolerance.
- **Read one `all-gather` in the HLO end-to-end.** Search the HLO
  for the specific instruction, trace back which parameter it
  gathered, and confirm from your sharding plan that GSPMD
  chose the collective you expected. This is the "I understand
  what the compiler did" milestone.
