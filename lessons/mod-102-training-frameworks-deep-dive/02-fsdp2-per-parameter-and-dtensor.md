# FSDP2, DTensor, and Per-Parameter Sharding

FSDP2 (`torch.distributed.fsdp.fully_shard`, introduced as the new
FSDP API in PyTorch 2.4) is a rewrite of FSDP1's sharding
implementation. The training loop looks nearly identical, the wire
volume is roughly identical, and — for a first-pass user — the API
surface is friendlier (one function call per module, no
`FullyShardedDataParallel` wrapper class). The changes underneath
are the interesting ones. This chapter explains what FSDP1's
`FlatParameter` was, what it broke, and how FSDP2's per-parameter
DTensor sharding fixes those breakages while composing with
tensor-parallel through a 2-D DeviceMesh.

## Recap: what FSDP has to do

From mod-101 chapter 3, one FSDP step does the following for each
sharded unit (typically a decoder block):

1. `all-gather` the unit's parameter shards → transient full
   parameters.
2. Forward-compute the unit; save activations.
3. Free the transient full parameters.
4. (Backward) `all-gather` again, backprop the unit,
   `reduce-scatter` the gradients, free the transient full parameters
   and full gradient.
5. The optimizer step is local per shard.

So FSDP's implementation problem is: *how do you represent "1/N of
every parameter" on each rank while letting the user still write
`self.linear.weight` and get sensible behavior?* FSDP1 and FSDP2
answer that question very differently.

## FSDP1's answer: `FlatParameter`

FSDP1 (the `FullyShardedDataParallel` class, still available in
PyTorch) worked by flattening. When you wrapped a submodule in
`FullyShardedDataParallel`, FSDP1 would:

1. Take every parameter inside the wrapped module.
2. Concatenate them into a single contiguous 1-D buffer — the
   `FlatParameter`.
3. Shard *that* buffer evenly across the process group.
4. On the module, replace every original `nn.Parameter` with a
   `_UnflatParameter` view into the flat buffer.
5. On forward, `all-gather` the flat buffer, view the concatenation
   as the original parameter shapes, run the layer, and free.

The flat buffer let FSDP1 do a *single* NCCL collective per
sharded unit instead of one collective per parameter, which was
important for small models where NCCL kickoff latency dominates.

The trouble was that the flattening leaked out through the rest of
PyTorch:

- **Mixed precision** — `FlatParameter` had one dtype for the whole
  buffer. Mixing dtypes within a wrapped unit (e.g. LayerNorm in
  FP32, Linears in BF16) required workarounds.
- **Optimizer state** — `torch.optim.AdamW` sees the parameter as
  the flat buffer, not as the individual `nn.Parameter`s. Any
  optimizer that maintains per-parameter state (LoRA, per-tensor
  learning rates, parameter-group configs) had to reason about
  flattened offsets rather than named parameters.
- **Freezing parameters** — mark one parameter as
  `requires_grad=False` and the flat buffer's gradient shape no
  longer matched. Parameter-efficient fine-tuning (LoRA, adapters)
  became fiddly.
- **DTensor composition** — TP-aware layers already produce
  DTensors. `FlatParameter` produced its own opaque sharded object.
  The two abstractions did not compose without special handling.
- **State-dict handling** — saving and loading had to translate
  between the flat buffer and the original named parameters.

FSDP1 shipped a family of workarounds (`use_orig_params=True`,
`ignored_parameters`, `sharded_state_dict`, and so on) that made
these problems tractable, but the underlying model — "one flat
buffer per unit" — was structurally at odds with the rest of the
PyTorch distributed stack. FSDP2 is the redesign that removes it.

## FSDP2's answer: per-parameter DTensor sharding

`fully_shard` shards each `nn.Parameter` *individually* as a
`DTensor`. There is no flat buffer. There is no
`_UnflatParameter`. After you call `fully_shard(module, mesh=...)`,
every parameter inside the module becomes a `DTensor` with a
`Shard(0)` placement on the sharding mesh axis.

Concretely:

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

model = build_transformer().cuda()
for block in model.blocks:
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)

for name, p in model.named_parameters():
    assert isinstance(p, DTensor)
    print(name, p.shape, p.placements)   # Shard(dim=0) on "dp"
```

A few consequences fall out of that design:

- **Parameters look native.** `param.shape` is the *logical* shape,
  as if it were unsharded. `param.to_local()` gives you your local
  shard. `param.full_tensor()` reconstitutes the whole tensor via
  `all-gather` (useful for saving state dicts, not for compute).
- **Per-parameter mixed precision.** `MixedPrecisionPolicy` applies
  per-parameter, and modules that must stay FP32 (e.g. LayerNorm
  weights, embedding tables, softmax) can be excluded selectively.
- **Optimizers see per-parameter DTensors.** `AdamW` runs on
  DTensors, updating each shard locally. The optimizer's state dict
  is naturally per-parameter, sharded.
- **Freezing individual parameters is fine.** No flat buffer means
  no alignment to break.
- **State dict is DTensor-native.** PyTorch DCP (distributed
  checkpointing, mod-106) handles DTensors directly.
- **Composition with TP is natural.** A tensor-parallel plan
  applied to a layer produces a DTensor with `Shard(k)` on the TP
  axis; `fully_shard` then adds `Shard(0)` on the DP axis. The
  parameter ends up with a 2-D sharding across the 2-D mesh, and
  the collectives run on the correct sub-mesh automatically. This
  is the substrate for chapter 5's 3D-parallel and for
  `torchtitan`'s reference implementation.

The `FlatParameter` optimization — one collective per unit — is
recovered inside `fully_shard`'s internal *bucketing*: parameters
of a unit are all-gathered together (grouped by dtype so each
collective is a single NCCL kernel). You do not have to think about
it, but you do not lose the throughput win either.

## The `fully_shard` API surface

The API is small, and the surface you touch is (in decreasing
order of frequency):

- `fully_shard(module, mesh=mesh, mp_policy=policy,
  offload_policy=None, reshard_after_forward=True)` — the main entry
  point. Apply to each transformer block, then to the enclosing
  module. Order matters: shard the leaves first, then the container.
- `MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype,
  cast_forward_inputs)` — the mixed-precision knobs. Common choice
  for a decoder-only transformer:
  `MixedPrecisionPolicy(param_dtype=torch.bfloat16,
  reduce_dtype=torch.float32)` — parameters live in BF16 in HBM,
  and gradient reductions accumulate in FP32 for stability.
- `CPUOffloadPolicy(pin_memory=True)` — passed as `offload_policy`
  to offload parameter and optimizer state to host memory.
  Chapter 4 goes deep on this.
- `set_reshard_after_forward(True|False)` — controls whether the
  transient full parameters are freed after forward. Setting `False`
  keeps them around and saves the backward's re-`all-gather`, at
  the cost of memory. Trade-off explored in chapter 4.
- `set_requires_gradient_sync(bool)` — the FSDP2 equivalent of
  DDP's `no_sync()` context manager, used to skip
  `reduce-scatter` during gradient accumulation.

Notice what is *not* in the API: no `sharding_strategy` argument.
FSDP2 always does the equivalent of `FULL_SHARD` (ZeRO-3). If you
want HSDP (shard inner, replicate outer), you build a 2-D mesh and
apply `fully_shard` on the inner axis only; PyTorch's docs and
`torchtitan` show the pattern.

## Migration from FSDP1

If you inherit an FSDP1 codebase, the mechanical migration is:

- Replace `model = FullyShardedDataParallel(model, ...)` with a
  per-block `fully_shard(block, mesh=mesh)` loop.
- Replace `MixedPrecision(...)` with `MixedPrecisionPolicy(...)`.
- Replace `sharding_strategy=ShardingStrategy.FULL_SHARD` with the
  default (FSDP2 is always full-shard).
- Replace `sharding_strategy=HYBRID_SHARD` with a 2-D mesh and
  `fully_shard` on the inner axis.
- Replace `use_orig_params=True` with… nothing. FSDP2 always
  exposes original parameters (as DTensors).
- Convert `sharded_state_dict()` → `torch.distributed.checkpoint`
  (DCP). FSDP1's `FullStateDictConfig` and friends are gone.
- Remove `ignored_parameters` — you now just don't call
  `fully_shard` on the modules you want left alone.

A subtle place to trip: FSDP1's `.summon_full_params()` context
manager has no direct FSDP2 equivalent. Under FSDP2 you call
`param.full_tensor()` on a specific DTensor at the boundary.

## Composing with tensor-parallel

The single most important consequence of DTensor-native sharding is
that you can compose with `torch.distributed.tensor.parallel`
(introduced for TP in PyTorch 2.x). The pattern that `torchtitan`
uses (and that chapter 5 revisits for the Megatron-LM comparison):

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel,
)

mesh = init_device_mesh(
    "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
)

# Apply Megatron-style TP plan.
tp_plan = {
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
    "attention_norm": SequenceParallel(),
    "ffn_norm": SequenceParallel(),
}
for block in model.blocks:
    parallelize_module(block, mesh["tp"], tp_plan)

# Then shard on DP.
for block in model.blocks:
    fully_shard(block, mesh=mesh["dp"])
fully_shard(model, mesh=mesh["dp"])
```

After this composition, each weight tensor is sharded on *both* mesh
axes. `attention.wq.weight`, for example, is a DTensor with
`Shard(0)` on the `tp` axis (Megatron's column-wise shard on the
output dim) and `Shard(0)` on the `dp` axis (FSDP2 shard on the
outermost dim of the local per-TP shard). All-gathers on the `dp`
axis fire during forward; the `tp` all-reduces fire on activations
between the column and row halves of each MLP and attention block
just as Megatron does. FSDP2 is picking up half of what mod-101
chapter 6 required Megatron for.

The composition is *not* limited to TP. FSDP2 + expert-parallel and
FSDP2 + context-parallel work through the same mesh pattern.
Pipeline-parallel needs additional plumbing (a pipeline schedule,
mod-101 chapter 4); FSDP2 does not provide the schedule, but its
per-rank state dict is compatible with a `PipelineSchedule` layered
on top.

## Debugging FSDP2 sharding

The most common FSDP2 debugging need is "which of my parameters are
sharded, and where?" — three tools:

- `for name, p in model.named_parameters(): print(name, p.shape,
  p.placements, p.device_mesh)`. This tells you whether each
  parameter is a DTensor and, if so, how it is placed.
- `torch.distributed.checkpoint.state_dict.get_state_dict_ops()` and
  `distributed_state_dict()`, which produce checkpoint-ready dicts
  of DTensor state.
- `torch.profiler` traces. FSDP2 tags its collectives clearly
  (`fsdp:all_gather_params`, `fsdp:reduce_scatter_grad`); missing
  overlap between block `i+1`'s all-gather and block `i`'s compute
  is the first thing to look for when throughput is
  disappointing.

For the classic "collective mismatch across ranks" hang, the
symptom is often a NCCL timeout without a Python stack trace. Set
`TORCH_NCCL_DESYNC_DEBUG=1` and `TORCH_DISTRIBUTED_DEBUG=DETAIL`
to make PyTorch's own collective-desync detection print. That
combination is far more actionable than raw NCCL logs.

## Summary

- FSDP1 sharded a flat concatenation of parameters. Efficient, but
  broke mixed precision, per-parameter optimizer state, freezing,
  and DTensor composition.
- FSDP2 shards each `nn.Parameter` individually as a DTensor. The
  training loop is unchanged, the wire volume is unchanged, and the
  above pain points disappear.
- The `fully_shard` API is small: apply to blocks and then to the
  root module. `MixedPrecisionPolicy` and `CPUOffloadPolicy` are the
  main knobs.
- FSDP2 composes cleanly with `torch.distributed.tensor.parallel`
  through a 2-D DeviceMesh. This is the substrate `torchtitan` uses
  and the substrate the exercise 1 bake-off targets.
- Debug per-parameter sharding by inspecting `DTensor` placements
  and by profiling the tagged FSDP2 collectives.
