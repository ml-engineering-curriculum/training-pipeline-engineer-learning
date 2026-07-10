# DCP Mechanics: Save, Load, Planners, and the Storage Split

Chapter 1 argued that a training run is a stream of jobs stitched
together by checkpoints. This chapter is about the stitching. The
tool of record in the PyTorch world is `torch.distributed.checkpoint`
— **DCP** — and the goal of the chapter is to give you a mental
model precise enough that you can (a) write a save/load pair from
scratch, (b) reason about what happens on disk when the training
topology changes, and (c) debug the failure modes DCP owns (which
are different from the ones the storage tier owns — mod-105).

Two references live open on the desk while you read this chapter:

- **PyTorch DCP docs.**
  https://pytorch.org/docs/stable/distributed.checkpoint.html
- **torchtitan's `Checkpointer`.**
  https://github.com/pytorch/torchtitan — see
  `torchtitan/components/checkpoint.py`. This is PyTorch's own
  reference implementation of a real training-scale checkpoint
  loop built on DCP, and it is what the exercises assume.

## Why not just `torch.save(model.state_dict())`?

The first checkpointing story in every PyTorch tutorial is
`torch.save(model.state_dict(), path)`. That works fine for a
single-process fine-tune. It falls apart at any scale for four
reasons you should be able to recite:

1. **The state dict is sharded, not full.** With FSDP2 (chapter 3
   of mod-101), each rank holds a `1/N` shard of every parameter
   as a `DTensor` or `ShardedTensor`. Calling `state_dict()` on
   rank 0 does not have the other ranks' shards, so
   `torch.save(model.state_dict())` writes an incomplete file. A
   naive fix — gather everything to rank 0 first — costs an
   all-gather over the whole model at every save and OOMs rank 0
   for anything above ~10B parameters.

2. **The optimizer state has the same shape.** Adam's `m` and `v`
   moments are per-parameter and equally sharded. Saving the
   optimizer without gathering (or with a `torch.save` of
   `optim.state_dict()`) has the same pathology.

3. **The write is a single-writer bottleneck.** Even if the gather
   works, one process serialising ~800 GB of BF16 weights over
   one file handle takes minutes. During those minutes every rank
   is idle at the collective barrier that follows.

4. **The output is topology-locked.** A file written by a rank-0
   process on a `TP=8, PP=8, DP=16` world size is unreadable by a
   subsequent run at `TP=4, PP=4, DP=32`, because the flat layout
   assumes the old grouping. Restarts across cluster shapes
   become impossible.

DCP was designed to fix all four. The rest of this chapter is how.

## What DCP actually is

`torch.distributed.checkpoint` (introduced in stable form in
PyTorch 2.0 as `torch.distributed.checkpoint`; the module lives
at `torch/distributed/checkpoint/`) is:

- **A distributed, parallel save/load API** — every rank participates,
  every rank writes its own shards, no gather-to-rank-0.
- **A tensor-metadata-aware format** — the on-disk artifact
  describes each tensor by its logical name, global shape, and
  the offsets of the chunks that landed on disk. There is no
  "rank 0 to rank N" mapping in the file.
- **A planner + storage split** — the `SavePlanner` /
  `LoadPlanner` decide *what* each rank contributes, and the
  `StorageWriter` / `StorageReader` decide *where* those bytes
  go. Both halves are pluggable.

The two calls you will use 95% of the time:

```python
import torch.distributed.checkpoint as dcp

# Save
dcp.save(
    state_dict=state,
    checkpoint_id="/mnt/checkpoints/step-01000",
    storage_writer=dcp.FileSystemWriter("/mnt/checkpoints/step-01000"),
    planner=dcp.DefaultSavePlanner(),
)

# Load
dcp.load(
    state_dict=state,
    checkpoint_id="/mnt/checkpoints/step-01000",
    storage_reader=dcp.FileSystemReader("/mnt/checkpoints/step-01000"),
    planner=dcp.DefaultLoadPlanner(),
)
```

Everything else in the API is an extension of these two — different
planners for exotic layouts, different writers for object stores,
async wrappers (chapter 3) that hide the wait behind the next
iteration.

## The state-dict contract: `Stateful` and what to include

DCP does not know what your model is. It only knows how to
serialise a `state_dict` — a `dict[str, torch.Tensor]`-shaped
object where the values can be `DTensor`, `ShardedTensor`, or
regular `Tensor`. Anything that has state that must survive a
restart implements `torch.distributed.checkpoint.stateful.Stateful`:

```python
from torch.distributed.checkpoint.stateful import Stateful

class TrainState(Stateful):
    def __init__(self, step, tokens, rng_state):
        self.step = step
        self.tokens = tokens
        self.rng_state = rng_state

    def state_dict(self):
        return {
            "step": self.step,
            "tokens": self.tokens,
            "rng_state": self.rng_state,
        }

    def load_state_dict(self, sd):
        self.step = sd["step"]
        self.tokens = sd["tokens"]
        self.rng_state = sd["rng_state"]
```

A production `state_dict` for the run should include, at minimum:

- `model` — the model's `state_dict()` (FSDP2 returns a sharded
  `state_dict` by default; DCP handles the `DTensor` payload
  directly).
- `optim` — `torch.distributed.checkpoint.state_dict.get_optimizer_state_dict(...)`
  which gives a DCP-friendly, sharded view of the optimizer.
- `sampler` — the stateful dataloader's iterator position
  (chapter 3).
- `train_state` — step counter, total tokens, RNG state, LR
  scheduler state.
- `run_metadata` — the seed, the config hash, the tokenizer hash,
  the framework versions. mod-108 will treat this as the
  reproducibility bundle.

If any of those is missing, "resume" is not truly a resume — you
are silently drifting from the baseline run. The exercises verify
each of the five.

## Planners: what each rank contributes

The **SavePlanner** is the thing that runs on every rank at save
time and decides *what this rank should write*. The default
planner (`dcp.DefaultSavePlanner`) implements the simplest useful
policy:

- For each tensor in the state dict, look at its `DTensor` /
  `ShardedTensor` sharding metadata.
- Emit a **local plan** — a list of `WriteItem` records, one per
  shard this rank owns — with the tensor's fully-qualified name,
  the shard's offset in the global tensor, and a pointer to the
  local storage.
- Rank 0 collects every rank's local plan into a **global plan**
  and de-duplicates any overlapping writes (e.g., replicated
  buffers where all ranks hold the same bytes).
- Each rank then writes only its own `WriteItem`s.

The dual **LoadPlanner** runs on every rank at load time and does
the mirror job: given the metadata on disk and the state dict the
caller wants to fill, it computes which byte ranges *this* rank
needs to read and where in the local `DTensor` shard those bytes
land.

Two subtle-but-important consequences of the planner design:

1. **The number of ranks writing does not equal the number of
   ranks reading.** The metadata describes tensors, not ranks.
   You can save at world size 128 and load at world size 64
   because the load planner just recomputes the shard-to-byte
   mapping. This is the resharding property; chapter 3 unpacks it.
2. **You can extend the planner without touching DCP core.** If
   your model has a custom parameter type or an unusual sharding
   scheme (e.g., expert-parallel routing tables), subclass
   `DefaultSavePlanner` and add the missing metadata to
   `WriteItem`s. Megatron-Core and DeepSpeed do this for their
   pipeline-parallel state; torchtitan does not need to because
   it uses stock DTensor sharding.

If you don't have a custom parameter type, the default planner is
enough. Do not build a custom planner speculatively.

## Storage writers: where the bytes actually go

The **StorageWriter** side of the split turns "here are the bytes
this rank must write" into "these bytes land in this file, in this
directory, with this metadata index".

The two writers that matter in practice:

- **`dcp.FileSystemWriter`** — writes to a POSIX filesystem. This
  is what you use on a parallel filesystem (Lustre, WEKA, FSx for
  Lustre; see mod-105 chapter 7). Every rank writes its own
  shard file into the target directory, and rank 0 writes the
  `.metadata` index.

- **`dcp.FsspecWriter`** (available in recent PyTorch releases;
  see the current DCP docs for the exact class name and
  stability status) — writes through the `fsspec` abstraction to
  object stores (S3, GCS, Azure Blob). Object-store checkpoints
  matter because parallel filesystems are expensive; putting
  older checkpoints in S3 is a common cost-optimization pattern.

The on-disk layout under a checkpoint directory looks like:

```
step-01000/
  .metadata                    # global index (small, JSON-ish)
  __0_0.distcp                 # rank 0 shard file
  __1_0.distcp                 # rank 1 shard file
  ...
  __N_0.distcp                 # rank N shard file
```

The `.metadata` file is the whole point of the format: it describes
every logical tensor in the state dict — global shape, dtype,
sharding — and points at the byte ranges inside the `.distcp`
files that hold the actual data. A load at a different world size
reads `.metadata`, computes the new shard-to-file mapping, and
issues parallel reads.

**mod-105 owns the storage-tier throughput budget.** DCP will
saturate it if you let it — every rank writes concurrently, and a
1024-rank save at 200 MB/rank/s asks for 200 GB/s aggregate write
bandwidth. Sizing the storage tier for that peak is a mod-105
job. mod-106 is about how the checkpoint API composes with the
storage tier once it exists.

## World-size-agnostic on the wire

The property that makes DCP the right tool at scale is that the
on-disk artifact does not know how many ranks wrote it. All of
the following work with the same checkpoint directory:

- Save at 128 ranks, load at 128 ranks — the default path.
- Save at 128 ranks, load at 64 ranks — each of 64 readers pulls
  twice as many byte ranges.
- Save at 64 ranks, load at 128 ranks — each of 128 readers pulls
  half as many byte ranges.
- Save at `TP=8, DP=16, PP=1` (128 GPUs), load at `TP=4, DP=16,
  PP=2` (128 GPUs) — the DTensor sharding metadata carries the
  logical layout separately from the physical layout, so a
  reshape into a different mesh is a load-planner decision.

The last case is where DCP earns its keep. Chapter 3 goes into
resharding in depth; the point to internalize here is that the
format is a function of the *model*, not the *cluster*. That is
the same invariant that makes the DDP wire protocol usable across
cluster shapes; DCP applies it to checkpoints.

## Failure modes DCP owns (and what belongs elsewhere)

When something goes wrong with a checkpoint, the failure will look
like one of:

- **"metadata / data mismatch"** — the model on the current run
  does not have the same set of parameter names as the metadata
  in the checkpoint. Cause: someone renamed a module, or the
  config changed the number of layers. DCP owns detecting this;
  the fix is in your code.
- **"Load planner cannot cover shard"** — the checkpoint has a
  shape for tensor X that the current model does not expect.
  Cause: TP degree changed and a parameter was split differently.
  DCP raises; chapter 3 covers the resharding contract.
- **"Storage writer failed"** — the underlying filesystem rejected
  a write (out of space, quota, permission). DCP surfaces the
  exception unchanged; the fix is in the storage tier (mod-105).
- **"Metadata read timeout"** — object-store latency on the
  `.metadata` fetch. mod-105 owns this too; but DCP-side you can
  mitigate by keeping recent checkpoints on the parallel FS and
  older ones on object store (the tiered-checkpoint pattern in
  chapter 7).

A rule of thumb: if the error mentions `WriteItem`, `LoadPlanner`,
or `TensorProperties`, it is a DCP issue. If it mentions `EIO`,
`ENOSPC`, or a network transport, it is a storage / fabric issue.
Do not chase the wrong module.

## A minimum end-to-end example

Putting everything together, the smallest useful save/load pair
looks like this (adapt paths to your storage tier):

```python
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict, StateDictOptions,
)

def build_state_dict(model, optim, train_state):
    model_sd, optim_sd = get_state_dict(
        model, optim,
        options=StateDictOptions(cpu_offload=False),
    )
    return {
        "model": model_sd,
        "optim": optim_sd,
        "train_state": train_state,   # a Stateful subclass
    }

def save(step, model, optim, train_state, root):
    state = build_state_dict(model, optim, train_state)
    ckpt_dir = f"{root}/step-{step:07d}"
    dcp.save(
        state_dict=state,
        storage_writer=dcp.FileSystemWriter(ckpt_dir),
    )

def load(step, model, optim, train_state, root):
    ckpt_dir = f"{root}/step-{step:07d}"
    state = build_state_dict(model, optim, train_state)
    dcp.load(
        state_dict=state,
        storage_reader=dcp.FileSystemReader(ckpt_dir),
    )
    set_state_dict(model, optim, model_state_dict=state["model"],
                   optim_state_dict=state["optim"])
    train_state.load_state_dict(state["train_state"].state_dict())
```

Two things this snippet gets right that beginner code often gets
wrong:

- **`get_state_dict` / `set_state_dict`** wrap FSDP2's sharded
  state dict extraction and injection. They handle DTensor
  metadata correctly, which is what makes the checkpoint
  world-size-agnostic on load. See the PyTorch docs for
  `torch.distributed.checkpoint.state_dict`.
- **The optimizer is saved and loaded through the same helpers.**
  Optimizer state carries almost as many bytes as the model in
  mixed-precision training (FP32 master weights + Adam moments);
  forgetting it means "resume" silently loses your training
  trajectory.

Chapter 3 will graft `async_save` and a stateful sampler onto this
skeleton and turn it into the training loop you actually deploy.

## Summary

- DCP is `torch.distributed.checkpoint`. It is the tool of record
  for large-scale PyTorch training checkpoints.
- The design has two independent halves: the **planner** decides
  what each rank writes/reads, and the **storage writer/reader**
  decides where the bytes go. Both are pluggable.
- Every rank writes its own shards in parallel; there is no
  gather-to-rank-0 bottleneck.
- The on-disk artifact is tensor-metadata-driven: the `.metadata`
  file describes global shapes and shard offsets so the same
  checkpoint can be loaded at a different world size or a
  different parallelism topology.
- A production `state_dict` includes model, optimizer, sampler,
  train state, and run metadata. Omit any of them and "resume"
  is not really resume.
- DCP errors that mention `WriteItem` / `LoadPlanner` are DCP
  problems; errors that mention I/O or transports belong to
  mod-105. Do not chase the wrong module on-call.
- The full end-to-end save/load pair is ~30 lines. Chapter 3
  turns it into the async, resharding, stateful-sampler-aware
  version you deploy.
