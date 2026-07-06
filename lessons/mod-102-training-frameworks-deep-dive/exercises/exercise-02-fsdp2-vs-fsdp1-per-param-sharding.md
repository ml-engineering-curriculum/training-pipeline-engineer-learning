# exercise-02: FSDP2 vs. FSDP1 Per-Parameter Sharding

**Estimated effort:** 3 hours

## Objective

Wrap the same small transformer twice — once in FSDP1
(`FullyShardedDataParallel`) and once in FSDP2 (`fully_shard`) —
inspect the per-parameter placements each API produces, and
demonstrate on a concrete example that FSDP2's per-parameter
DTensor sharding composes with `torch.distributed.tensor.parallel`
while FSDP1's `FlatParameter` does not. The deliverable is the
artifact that turns chapter 2 into a claim you can defend at a
whiteboard: "here is exactly what changed between FSDP1 and FSDP2,
and here is the composition it unlocked."

## Prerequisites

- Chapter 2 of this module.
- A dev environment with PyTorch ≥ 2.4 (both APIs are still
  present in current PyTorch — FSDP1 as
  `torch.distributed.fsdp.FullyShardedDataParallel`, FSDP2 as
  `torch.distributed.fsdp.fully_shard`).
- At least 2 GPUs. 4–8 is better because you can inspect
  non-trivial shardings.
- Passing familiarity with `torch.distributed.tensor.DTensor` and
  `torch.distributed.device_mesh.init_device_mesh` (mod-101
  chapter 3).

## Problem statement

Your team's training codebase was written for FSDP1
(`FullyShardedDataParallel`) and now needs to add tensor-parallel
so a per-layer parameter fits in HBM. You have three days before
you need to decide whether to migrate to FSDP2 first or to layer
TP on top of FSDP1 as-is. Before you commit, you want to see with
your own eyes what each version of the sharding looks like on a
running module — and whether FSDP1's `FlatParameter` really is
incompatible with per-parameter DTensor sharding in the way
chapter 2 claims.

## Requirements

### The reference module

A tiny transformer block is enough. Something like:

```python
class MiniBlock(nn.Module):
    def __init__(self, hidden=512, ffn=2048, heads=8, vocab=1024):
        super().__init__()
        self.attn_q = nn.Linear(hidden, hidden, bias=False)
        self.attn_k = nn.Linear(hidden, hidden, bias=False)
        self.attn_v = nn.Linear(hidden, hidden, bias=False)
        self.attn_o = nn.Linear(hidden, hidden, bias=False)
        self.mlp_up   = nn.Linear(hidden, ffn, bias=False)
        self.mlp_down = nn.Linear(ffn, hidden, bias=False)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
    def forward(self, x):
        h = self.ln1(x)
        q, k, v = self.attn_q(h), self.attn_k(h), self.attn_v(h)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(h.size(-1)), dim=-1) @ v
        x = x + self.attn_o(a)
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.ln2(x))))
        return x
```

You do not need to train it well. You need to be able to inspect
its parameters after sharding.

### Part A — FSDP1 with `FullyShardedDataParallel`

1. Wrap `MiniBlock` in
   `FullyShardedDataParallel(mini_block, sharding_strategy=FULL_SHARD,
   use_orig_params=False)` and print, for each named parameter,
   `type(p)`, `p.shape`, and `p._flat_param` (the private handle
   that reveals FSDP1's flat buffer).
2. Wrap it again with `use_orig_params=True` and print the same
   fields. Note which parameters are "original" vs. views into
   the flat buffer, and record the difference.
3. In both configurations, try to serialize the state dict with a
   simple `mini_block.state_dict()` and inspect what you got. Also
   try `torch.distributed.checkpoint.state_dict.get_state_dict`;
   record what does and does not work.

### Part B — FSDP2 with `fully_shard`

1. Wrap `MiniBlock` with `fully_shard(mini_block, mesh=mesh)` for
   a 1-D DP mesh across your `world_size` GPUs.
2. Print, for each named parameter:
   - `type(p)` — should be `DTensor`.
   - `p.shape` — logical shape.
   - `p.placements` — the placement(s) on the mesh.
   - `p.device_mesh` — the mesh the parameter is sharded on.
   - `p.to_local().shape` — the local shard shape.
3. Materialize the full parameter via `p.full_tensor()` and
   confirm the shape equals `p.shape`.

### Part C — The composition test

Now apply a tensor-parallel plan to the same module using
`torch.distributed.tensor.parallel.parallelize_module`. Do this in
two configurations:

1. **FSDP1 + TP.** Try wrapping with `FullyShardedDataParallel`
   *first* and then applying `parallelize_module`. Then try the
   reverse order. Record what happens — expect an error, an
   uninformative behavior, or a workaround that requires
   `use_orig_params=True` and manual bookkeeping. Do *not* try to
   force it to work; the point is to document the friction.
2. **FSDP2 + TP.** Build a 2-D mesh
   `init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp",
   "tp"))`. Apply `parallelize_module(block, mesh["tp"], tp_plan)`
   first, then `fully_shard(block, mesh=mesh["dp"])`. Print
   `p.placements` for the same parameters as in part B and
   observe that each is now a `(Shard, Shard)` DTensor across the
   2-D mesh.

Include a small forward+backward pass on each configuration to
confirm parameters and gradients flow through, and no exception is
raised.

### Part D — The write-up

Deliver a 300–500-word `sharding-comparison.md` in your solutions
repo that covers:

1. **What FSDP1 stores per parameter** and what
   `use_orig_params=True` does to it. Illustrate with a specific
   parameter (e.g. `attn_q.weight`) from part A.
2. **What FSDP2 stores per parameter** — a `DTensor` with its
   placement, its mesh, and its local shard. Illustrate with the
   same parameter from part B.
3. **What breaks in FSDP1 + TP composition** — cite what you saw in
   part C.1, whether it was a hard error, a wrong-shape training
   run, or a workaround that shipped through.
4. **How FSDP2 + TP composes cleanly** — describe the 2-D
   placement of the same parameter from part C.2 and predict which
   collective (all-gather on which axis) fires for this parameter
   during forward.

## Acceptance criteria

- Working FSDP1 and FSDP2 wrappings of the same `MiniBlock`, both
  running one forward+backward without exception.
- Printed evidence of `_flat_param` in FSDP1 (with
  `use_orig_params=False`) and `DTensor` typing in FSDP2, for at
  least three named parameters.
- The write-up correctly identifies at least one specific FSDP1
  issue that FSDP2 fixes (mixed precision per-parameter,
  per-parameter freezing, or TP composition — pick the one you
  demonstrated).
- Part C.2 shows a 2-D-mesh DTensor with placements like
  `(Shard(0), Shard(0))` or `(Replicate(), Shard(0))`, depending
  on the layer. Screenshot or log excerpt suffices.
- The write-up cites the PyTorch `fully_shard` documentation and
  either the DTensor or DeviceMesh documentation.

## Starter guidance

- **Set `world_size = 2 * 2` if you have 4 GPUs** so the 2-D mesh
  has both axes populated.
- **`torch.distributed.tensor.DTensor` has helpful printing** —
  `print(param)` shows placements and mesh directly.
- **Do not test the sharding by training.** A tiny module with
  synthetic input for 1 step is enough to prove the invariants.
- **Expect FSDP1 + TP to be painful.** Do not fight it for hours;
  a clear "here is why this does not compose" is a valid
  deliverable.
- **`use_orig_params=True` is the FSDP1 escape hatch** the
  PyTorch team shipped to make composition possible. Try it in
  part A.2 to understand the workaround, but do not conclude
  from it that FSDP1 has parity with FSDP2 for TP — the
  workaround has known edge cases the FSDP2 rewrite eliminates.

## Stretch goals

- Save an FSDP2 sharded state dict with
  `torch.distributed.checkpoint.save` and reload it in a job with
  a different `world_size`. This exercises DCP resharding — the
  operational payoff of DTensor-native sharding.
- Freeze one parameter (`p.requires_grad_(False)`) inside FSDP2
  and confirm training still works. Do the same in FSDP1 with
  `use_orig_params=False` and observe what breaks (or what
  workaround `ignored_parameters` requires).
- Repeat part C but include `SequenceParallel` on the layer
  norms. Confirm the layer norm parameters end up with
  `Shard(0)` on the TP axis and `Replicate()` on the DP axis, and
  that a forward pass still runs.
