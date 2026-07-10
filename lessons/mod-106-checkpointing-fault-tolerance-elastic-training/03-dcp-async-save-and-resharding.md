# Async Save, Resharding, and the Stateful Data Loader

Chapter 2 gave you the synchronous DCP save/load pair. In a real
training loop that pair is not good enough for two reasons: (a)
every synchronous save stops the run for tens of seconds while the
storage tier drains, and (b) a resume that replays the last N
batches without knowing which batches the previous run consumed
silently reintroduces samples the model has already seen. This
chapter closes both gaps.

Three references stay open:

- **`torch.distributed.checkpoint` async API.**
  https://pytorch.org/docs/stable/distributed.checkpoint.html —
  the `async_save` entry point and the state-dict staging notes.
- **torchtitan `Checkpointer`.**
  https://github.com/pytorch/torchtitan (see
  `torchtitan/components/checkpoint.py`) — the reference
  async+DCP integration for a production training loop.
- **`torchdata` StatefulDataLoader.**
  https://github.com/pytorch/data — the resumable dataloader that
  makes "resume from step 12 345" mean the same batch sequence as
  the original run.

## Why sync save is not enough

Do the arithmetic. A 70B parameter model in BF16 is `70e9 · 2 =
140 GB` of parameters. Add optimizer state in FP32 (Adam: 8B per
parameter for FP32 master + `m` + `v` = 3 × 4B; some setups also
keep BF16 grads) and you land at roughly `70e9 · 12 = 840 GB` of
checkpoint payload per save.

On a well-provisioned parallel filesystem (mod-105 chapter 7) with,
say, 400 GB/s of aggregate write bandwidth to your training namespace,
that is a 2.1-second best-case write. But every checkpoint save also
includes:

- A collective barrier before the save so all ranks agree on the
  state.
- The staging step where DTensor shards are materialized into a
  contiguous buffer for writing.
- The write itself.
- A metadata index write on rank 0.
- A second barrier at the end.

Real production numbers land at **20–60 seconds per synchronous
save** for a 70B model on a healthy fabric — Llama 3-scale reports
land in that range too (Grattafiori et al., 2024, §6, discussion
of checkpoint frequency). At a save-every-30-minutes cadence and
30 seconds per save, sync save costs you `30 / (30 · 60) = 1.7%`
of your wall clock as pure checkpoint stall. Multiply by the run
duration and you have burned days of GPU time.

Async save eliminates most of that.

## `async_save`: hide the write behind the next step

`dcp.async_save` returns a `Future` immediately after the *staging*
step completes. The heavy lifting — the actual write to storage —
happens in a background thread while the trainer moves on to the
next iteration. The pattern looks like:

```python
import torch.distributed.checkpoint as dcp

# Track the outstanding save so we can wait on it before overwriting
# the staging buffer on the next save.
outstanding_save = None

def save_async(step, state_dict, root):
    global outstanding_save

    # If a previous async save is still running, wait for it to finish
    # before we mutate any of the state we are about to stage again.
    if outstanding_save is not None:
        outstanding_save.result()

    ckpt_dir = f"{root}/step-{step:07d}"
    outstanding_save = dcp.async_save(
        state_dict=state_dict,
        storage_writer=dcp.FileSystemWriter(ckpt_dir),
    )
```

Two invariants the async path relies on:

1. **The state dict is staged before `async_save` returns.** DCP
   copies (or references, depending on `cpu_offload` /
   `pin_memory` settings) the tensor payloads into a snapshot
   buffer inside the call. After that, the trainer is free to
   mutate its parameters and optimizer state on the next step
   without corrupting the checkpoint.
2. **Only one save can be in flight at once.** The staging buffer
   is per-checkpointer, not per-save. If you kick off save N+1
   before save N has written to storage, you overwrite the
   staging buffer. `torchtitan`'s `Checkpointer` gates every
   `async_save` on the previous save's `Future` for exactly this
   reason.

If you configure the save interval to be shorter than the actual
write time, you have effectively rebuilt a synchronous save — the
staging wait blocks the trainer. The rule is: **save interval must
be greater than the p99 write time**. mod-108's dashboards should
graph both.

### Staging costs and CPU offload

The staging step itself is not free. Each rank has to walk its
DTensor shards, allocate a snapshot buffer, and copy tensors into
it. For an 840-GB checkpoint, staging into GPU memory doubles the
peak memory budget of the trainer during the save window — a
memory bill you almost certainly cannot pay on a fully-loaded
FSDP2 run.

The escape hatch is CPU staging: DCP copies the tensors to pinned
CPU memory instead of another GPU buffer, and the write proceeds
from there. See `StateDictOptions(cpu_offload=True)` in the DCP
`state_dict` helpers, and torchtitan's checkpointer for how it
plumbs the option through. On a well-provisioned host, CPU staging
adds ~5 seconds of D2H copy time up front but frees the GPU
immediately.

The two-parameter trade-off you actually tune is:

| Setting | GPU peak memory during save | Time to first async return | Write concurrency |
|---------|-----------------------------|----------------------------|-------------------|
| GPU staging | 2× steady state | milliseconds | Full |
| CPU staging | 1× steady state | seconds (D2H copy) | Full |
| No staging (sync save) | 1× steady state | full write time | Full but blocking |

For any run large enough that memory is tight, CPU staging is the
answer. It is what torchtitan and Megatron-Core recommend at
production scale.

## Resharding: the whole point of the DCP format

Chapter 2 introduced the world-size-agnostic property. Resharding
is what you do with it. The three cases you must be able to handle
on-call:

1. **Reload after node loss (fewer ranks).** A node crashes, the
   scheduler cannot immediately return an identical replacement,
   and you want to resume on the surviving nodes rather than wait.
   New world size is smaller; DCP redistributes shards.
2. **Reload after capacity expansion (more ranks).** You saved on
   1024 GPUs, the scheduler has 1536 GPUs free, and you want to
   speed up the remaining training with the extra capacity. DCP
   redistributes shards in the other direction.
3. **Reload with a different parallelism topology.** You saved with
   `TP=8, PP=1, DP=64` (512 GPUs) and want to reload with
   `TP=4, PP=2, DP=64` (also 512 GPUs) because you discovered
   TP=4 is better for your model shape at inference-time. DCP
   handles this too, provided the model's DTensor sharding
   metadata still describes the same logical tensors.

The resharding contract in words:

> DCP can reload any checkpoint whose global tensor set is a
> superset of the new run's expected global tensor set, and whose
> per-tensor global shapes match the new run's expected global
> shapes. The physical layout — how tensors are chunked across
> ranks and files — is renegotiated by the load planner.

Two things that break resharding, both on the model side:

- **Renaming or restructuring layers.** If `layer.5.attn.qkv`
  became `layer.5.attn.q`, `layer.5.attn.k`, `layer.5.attn.v`,
  the metadata does not match. The fix is either a compatibility
  shim (a `state_dict` rename hook) or a one-off migration script.
- **Changing tensor shapes.** If you widened the model, or added
  MoE experts, or changed the vocabulary size, the global shape
  does not match. There is no automatic migration; either
  reinitialize the affected tensors or write a bespoke migration.

Neither of these is a DCP failure; they are model-code failures.
Chapter 5 puts "checkpoint schema drift" into the incident
taxonomy explicitly.

### What resharding does not do

Resharding does not know about non-tensor state. If your training
state — RNG state, LR scheduler position, sampler position — is
tied to the world size (a per-rank RNG per world index, say),
loading at a new world size will not do the right thing. The
recipe is to store per-rank RNG state as *per-logical-rank*
metadata that is meaningful regardless of world size, or to
reseed on load. torchtitan reseeds on load; it also stores the
tokens-consumed counter so the sampler resumes at the right
global offset.

## The stateful data loader

Consider the following pathology. Your run crashes at step 12 345.
The last DCP save landed at step 12 000. You reload the model +
optimizer at step 12 000, spin up a fresh `DataLoader`, and
resume. But your `DataLoader` starts iterating from batch 0 of the
epoch — 12 000 batches from where the model actually saw its last
gradient update. You have just:

- **Replayed 12 000 batches**, wasting hours of compute.
- **Reordered the epoch**, so the model sees the first 12 000
  samples twice in a row and misses whichever samples came
  between 12 000 and the epoch boundary.
- **Broken your seed contract**, because the shuffled order the
  model actually sees is no longer a function of the seed alone.

The `torchdata` **`StatefulDataLoader`** exists to fix this. It
extends the PyTorch `DataLoader` with `state_dict()` and
`load_state_dict()` methods that capture the iterator's position,
the shuffle-RNG state, the prefetch queue's high-water mark, and
(for sharded datasets) the per-worker cursor.

A minimal integration:

```python
from torchdata.stateful_dataloader import StatefulDataLoader

loader = StatefulDataLoader(
    dataset,
    batch_size=micro_batch,
    num_workers=8,
    shuffle=True,
)

# Add it to the DCP state dict:
def build_state_dict(model, optim, loader, train_state):
    model_sd, optim_sd = get_state_dict(model, optim)
    return {
        "model": model_sd,
        "optim": optim_sd,
        "loader": loader.state_dict(),
        "train_state": train_state,
    }

# On load, feed it back:
loader.load_state_dict(state["loader"])
```

Two subtle points worth understanding:

1. **The state dict includes the RNG.** Reseeding on load without
   the sampler state is not enough; the shuffle order depends on
   how far into the epoch you were when the RNG advanced. The
   stateful loader captures that RNG for you.
2. **It composes with sharded datasets.** For WebDataset / MDS /
   Ray Data loaders (mod-103), the stateful loader tracks the
   per-worker shard cursors, so resume points at the right file
   inside the right shard. When the loader replays, no sample is
   duplicated across the boundary.

Not every dataset backend supports `StatefulDataLoader` natively.
For raw `IterableDataset` implementations you may have to
implement a small `Stateful` wrapper that captures whatever
per-worker cursor the dataset uses. mod-103's dataset design
choices — sharded, offset-addressable formats like WebDataset and
MDS — are what make this cheap. mod-106 relies on that groundwork.

## Putting the pieces together: a training-loop skeleton

Here is what the assembled loop looks like. This is the shape the
exercises assume.

```python
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict, StateDictOptions,
)
from torchdata.stateful_dataloader import StatefulDataLoader

CKPT_ROOT = "/mnt/lustre/train/run-2026-07-09"
CKPT_INTERVAL_STEPS = 250      # tune from the SLO exercise (ch. 7)

train_state = TrainState(step=0, tokens=0, rng_state=torch.get_rng_state())
model = build_and_shard_model()
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
loader = StatefulDataLoader(dataset, batch_size=micro_batch,
                            num_workers=8, shuffle=True)

outstanding_save = None

def build_sd():
    m_sd, o_sd = get_state_dict(model, optim,
                                options=StateDictOptions(cpu_offload=True))
    return {"model": m_sd, "optim": o_sd,
            "loader": loader.state_dict(),
            "train_state": train_state}

def maybe_resume(root):
    latest = find_latest_step_dir(root)     # your naming policy
    if latest is None:
        return
    sd = build_sd()
    dcp.load(sd, storage_reader=dcp.FileSystemReader(latest))
    set_state_dict(model, optim,
                   model_state_dict=sd["model"],
                   optim_state_dict=sd["optim"])
    loader.load_state_dict(sd["loader"])
    train_state.load_state_dict(sd["train_state"].state_dict())

def save_step(step):
    global outstanding_save
    if outstanding_save is not None:
        outstanding_save.result()     # wait for previous save
    ckpt_dir = f"{CKPT_ROOT}/step-{step:07d}"
    sd = build_sd()
    outstanding_save = dcp.async_save(
        sd, storage_writer=dcp.FileSystemWriter(ckpt_dir),
    )

maybe_resume(CKPT_ROOT)

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    optim.step()
    optim.zero_grad()
    train_state.step += 1
    train_state.tokens += batch.numel()

    if train_state.step % CKPT_INTERVAL_STEPS == 0:
        save_step(train_state.step)

# Final drain
if outstanding_save is not None:
    outstanding_save.result()
```

Five properties this loop guarantees, each of which is one of the
run's insurance policies:

- **Async save.** Checkpoint I/O runs concurrently with the next
  iterations; the trainer only stalls if a save is still running
  at the next save interval.
- **Resharding on resume.** `dcp.load` reads the metadata index
  and redistributes shards to whatever world size the current
  process group has.
- **Stateful loader.** Resume sees the same batch sequence as the
  original run — no replay, no reorder, no seed-contract violation.
- **Train state and RNG persisted.** Reproducibility bundle is
  intact across restarts (mod-108 depends on this).
- **CPU staging.** Peak GPU memory during save equals steady-state,
  so the run does not OOM during a checkpoint save.

Chapter 4 wraps this loop in a `torchrun` launcher that also
handles rendezvous and elastic reshape when the node count actually
changes.

## What can still go wrong

Two failure modes specific to the async + resharding + sampler
combo you should know about:

- **The save Future silently fails.** `dcp.async_save` runs the
  write on a background thread. If the storage tier throws
  mid-write, the exception is stored in the `Future`, not raised
  to the trainer. You *must* call `.result()` (or check
  `.exception()`) at least once per save interval. torchtitan
  wraps this; homemade loops often forget. The failure mode is
  "run keeps training past the last successful save without
  knowing" — you find out on the next incident, when the load
  succeeds against a stale directory.
- **Sampler state depends on the previous world size.** For some
  dataset implementations, the shuffle order is a function of the
  world size (each worker computes its own shard). A save at
  world size N and a load at world size M can, in those cases,
  visit the same sample twice or skip samples entirely. The fix
  is either a world-size-invariant sampler (mod-103) or
  documenting the constraint on resharding.

Both of these belong on the on-call runbook in chapter 5.

## Summary

- Synchronous DCP save costs the run tens of seconds per
  checkpoint. `dcp.async_save` hides the write behind the next
  iteration and reduces the trainer stall to the staging step.
- CPU staging (`StateDictOptions(cpu_offload=True)`) keeps GPU
  peak memory at steady state during the save; use it for any
  large run.
- Resharding is a property of the DCP on-disk format: the same
  checkpoint can be loaded at a different world size or a
  different TP/PP/FSDP topology, provided the *logical tensor
  set* still matches.
- Renaming or reshaping tensors breaks resharding. The fix is on
  the model side (rename hooks, migration scripts).
- Resume without a stateful data loader replays or skips samples
  and breaks the seed contract. `torchdata.StatefulDataLoader`
  captures iterator, RNG, and per-worker cursor state.
- The training loop skeleton in this chapter — async save, CPU
  staging, stateful loader, resume-on-start — is the shape the
  exercises assume. torchtitan's `Checkpointer` is its reference
  implementation.
- Two silent-failure modes (async save exception dropped on the
  floor, sampler state that depends on world size) belong on the
  runbook (chapter 5).
