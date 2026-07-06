# FSDP2 and the ZeRO-3 Partitioned Step

DDP pays for scale by burning memory: every rank stores the whole model, the
whole optimizer, and the whole gradient buffer. Once you cannot fit that on
one GPU, the fix is to shard the state and reconstitute it *just in time* for
the layer that needs it. That is exactly what FSDP (PyTorch's Fully Sharded
Data Parallel, particularly FSDP2) and DeepSpeed's ZeRO-3 do. They are, at
the collective level, the same algorithm.

This chapter traces one training step end-to-end so you can whiteboard
which tensors live where, when they are reconstituted, and where the
`reduce-scatter` and `all-gather` land.

## Motivation

The ZeRO paper (Rajbhandari et al., 2020, "ZeRO: Memory Optimizations
Toward Training Trillion Parameter Models") frames the state you have to
store per parameter as three tiers:

- **Optimizer states** — for Adam, two FP32 moments plus the FP32 master
  copy of the parameter. Dominant memory user in mixed precision.
- **Gradients** — one buffer of the same shape as the parameters.
- **Parameters** — the parameters themselves.

ZeRO-1 shards optimizer states. ZeRO-2 shards optimizer states + gradients.
ZeRO-3 shards all three. FSDP2 in PyTorch implements the equivalent of
ZeRO-3 with per-parameter sharding (FSDP1 sharded flat "FlatParameter" units;
FSDP2 shards individual parameters and integrates with DTensor).

You give up local memory savings only when the model needs to see a full
parameter — during forward and backward — and the point of the design is
that "when the model needs to see a full parameter" is a *small*
transient state that can be freed immediately after use.

## The step, traced

Assume `N` ranks, a decoder-only transformer with `L` layers, and one
FSDP2 shard per layer (this is what `fully_shard` on each block gives you
by default). Steady-state at the start of a step:

- **Parameter shards** — each rank holds `1/N` of every parameter,
  contiguously.
- **Gradient shards** — each rank holds `1/N` of every gradient (currently
  zeros).
- **Optimizer state shards** — each rank holds `1/N` of every Adam moment
  and FP32 master copy.

### Forward pass, layer `i`

1. **All-gather parameters of layer `i`** across the shard group. After the
   collective, every rank has the full parameters of layer `i` in a
   *transient* buffer.
2. **Compute the layer's forward**, saving activations for backward.
3. **Free the transient full parameters** for layer `i`. This is the whole
   point: only *one* layer's worth of parameters is fully materialized at
   a time.

Repeat for `i = 0..L-1`. At the top of the network, all ranks compute the
loss.

### Backward pass, layer `i`

1. **All-gather parameters of layer `i`** again (they were freed after the
   forward). ZeRO-3 and FSDP both eat this second all-gather; activation
   recomputation trades it for compute if you enable it (mod-107).
2. **Backprop the layer**, producing a *full* local gradient for its
   parameters.
3. **Reduce-scatter the gradients**: every rank sends its full local
   gradient; every rank ends up owning the *reduced shard* of the gradient
   that corresponds to its parameter shard. This is the FSDP analogue of
   DDP's all-reduce.
4. **Free the transient full parameters** and the transient full gradient
   for layer `i`.

At the end of backward, each rank owns `1/N` of the reduced gradient for
every parameter.

### Optimizer step

Each rank runs the optimizer on its own shard: it updates its `1/N` of the
FP32 master parameters using its `1/N` of the gradients and its `1/N` of the
Adam moments, then copies the result back into its BF16 parameter shard.

**No cross-rank communication is required in the optimizer step**, because
every rank owns everything it needs for its own shard. This is a big deal:
it means FSDP/ZeRO-3 can integrate CPU / NVMe offloading of optimizer state
without a fabric round-trip on every step. (See ZeRO-Offload, Ren et al.,
2021, and ZeRO-Infinity, Rajbhandari et al., 2021, for the offload ladders.)

## Where the wire volume goes

Let `S` be the model-parameter size in bytes.

- **Forward** — one all-gather of `S`, distributed across layers. Per-GPU
  send is `(N-1)/N * S`.
- **Backward** — one all-gather of `S` (again) plus one reduce-scatter of
  `S` (gradients). Per-GPU send is `2 * (N-1)/N * S`.
- **Total** — `3 * (N-1)/N * S`.

Compare to DDP's `2 * (N-1)/N * S`. FSDP/ZeRO-3 pays for the extra all-gather
on the backward pass. That is *the* trade: 1.5× more wire in exchange for
`N`× less memory. For large `N` and models that would otherwise not fit,
that trade is trivially worth it. For 8-GPU nodes running a small model
where DDP already fits, FSDP is wasted bandwidth.

## FSDP2's per-parameter sharding

FSDP1 flattened all parameters within a "unit" into one FlatParameter and
sharded that flat buffer. That was efficient for allreduce but leaked
abstraction all over the place: DTensor did not compose, mixed precision
handling was awkward, and things like parameter-freezing broke sharding
alignment.

FSDP2 (introduced in PyTorch 2.4+ under `torch.distributed.fsdp.fully_shard`)
shards *each parameter individually* as a DTensor. Practical implications:

- Each `nn.Parameter` after `fully_shard` is a DTensor with a `Shard(0)`
  placement across the shard mesh. `param.full_tensor()` reconstitutes it.
- You can compose FSDP2 sharding with tensor-parallel sharding by using a
  2-D device mesh (`Replicate()` on the TP axis, `Shard(0)` on the FSDP
  axis). This is how modern 3D-parallel setups are wired.
- You can freeze parameters (e.g. LoRA base weights) without breaking
  bucketing.

See the PyTorch docs for `torch.distributed.fsdp.fully_shard`, DTensor, and
DeviceMesh for the current API surface.

## Minimum working example — FSDP2

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

model = build_transformer().cuda()

policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,   # reduce-scatter in FP32 for stability
)

# Shard each decoder block on the "dp" mesh axis.
for block in model.decoder.layers:
    fully_shard(block, mesh=mesh, mp_policy=policy)
# Shard the outer module last so its unsharded params are also sharded.
fully_shard(model, mesh=mesh, mp_policy=policy)

optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    optim.step()
    optim.zero_grad()
```

The **shape of the training loop is identical to DDP**. The difference is
entirely internal: `loss.backward()` no longer runs a full all-reduce; it
runs the all-gather + backward + reduce-scatter dance layer by layer.

## HSDP: hybrid sharding across the mesh

For a large `N`, an all-gather across the whole world becomes cross-node
expensive. Hybrid Sharded Data Parallel (HSDP; see PyTorch docs, the
"HYBRID_SHARD" sharding strategy in FSDP1 and the 2-D mesh pattern in FSDP2)
splits the mesh:

- Shard *inside* a fast (usually intra-node NVLink) group so the all-gather
  is over NVLink.
- Replicate *across* the slower (inter-node RDMA) group, and do a DDP-style
  all-reduce across replicas at the end of the step.

You are trading a bit of memory (replicas instead of shards across the outer
axis) for a much cheaper all-gather. Chapter 5's cost model gives you the
crossover point.

## When to reach for which

- Model + optimizer state fits on one GPU, want throughput → **DDP**.
- Model + optimizer state does not fit on one GPU, but does fit within one
  node → **FSDP2** with a single-node shard group.
- Multi-node scale but a single node can hold a shard → **HSDP** (shard
  intra-node, replicate across nodes).
- Individual layers do not fit on one GPU even after sharding →
  **tensor-parallel** on top of FSDP (chapter 4).
- Depth is the bottleneck (very deep model, activation memory dominates) →
  **pipeline-parallel** (chapter 4).

## Summary

- FSDP2 and ZeRO-3 shard parameters, gradients, and optimizer state across
  ranks. A forward + backward reconstitutes each layer's parameters
  transiently via `all-gather`, then reduces gradients via `reduce-scatter`.
- Total per-GPU wire volume is `3 * (N-1)/N * S`, 1.5× more than DDP.
- The optimizer step is local — no fabric round-trip.
- FSDP2's per-parameter sharding composes with tensor-parallel through
  a 2-D DeviceMesh, which is the substrate for the parallelism strategies
  in chapter 4.
