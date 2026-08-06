# PyTorch Distributed Checkpoint (DCP)

Chapter 2 gave you the cost model. This chapter gives you the API
that lets you actually hit that model. PyTorch Distributed Checkpoint
(`torch.distributed.checkpoint`, called DCP from here on) is the
in-tree solution for parallel, shard-aware, resharding-friendly
checkpoints, and it is what every modern PyTorch training stack
(torchtitan, torchrec, torchtune, and the Meta production stack it
came from) uses as its foundation.

Keep the primary reference open while you read this chapter:

- **PyTorch Distributed Checkpoint documentation.**
  https://docs.pytorch.org/docs/stable/distributed.checkpoint.html —
  API reference for `save`, `load`, `async_save`, `Stateful`, the
  planners, and the storage backends.
- **PyTorch DCP tutorial.**
  https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
  — the canonical worked example. Read it before starting exercise 1.

## Why DCP exists

Before DCP, the two options in PyTorch were:

1. **Rank 0 gathers everything and writes it.** Correct, portable,
   O(1) parallelism, dies at scale (chapter 2's anti-pattern).
2. **Every rank writes its own `torch.save` file with its shard.**
   Fast, but the reader has to know exactly the same world size and
   shard layout as the writer. A resume onto a different world size
   was a manual re-shard.

DCP was designed to give you (a) the write throughput of "every rank
writes in parallel", (b) the correctness of "the state on disk is
semantically the model, not the shard layout of the machine that
wrote it", and (c) the ability to load onto a *different* world size
than the one that wrote — which is what makes elastic training in
chapter 4 actually work.

## The DCP data model

DCP stores a **`state_dict`**, which is a plain Python dict of
tensors, plus a metadata file that records the *global* shape of each
tensor and how it was sharded when written. On save, each rank
contributes its local shard of each tensor; DCP writes those shards
to storage and writes a `.metadata` file that describes the global
mapping.

On load, DCP reads the `.metadata` file, sees the current world's
sharding for each tensor, computes the shard-to-shard mapping between
"what's on disk" and "what this rank needs", and issues the reads to
fulfill exactly what's needed. The reshard is automatic.

Key consequence: **the on-disk layout is not tied to the training
world size**. A checkpoint saved on 512 ranks with FSDP-8 tensor
parallelism can be loaded on 256 ranks with FSDP-4 tensor parallelism
without any manual conversion. Exercise 1 is the drill for that.

## The minimal save / load pattern

The current DCP API centers on `save(state_dict, checkpoint_id, ...)`
and `load(state_dict, checkpoint_id, ...)`. `state_dict` is passed by
reference; `load` mutates the in-memory tensors in place. A minimal
sketch:

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
)

# --- save ---
model_sd, optim_sd = get_state_dict(model, optimizers=optimizer)
state_dict = {
    "model": model_sd,
    "optim": optim_sd,
    "step": step,
}
dcp.save(state_dict=state_dict, checkpoint_id=f"/ckpt/step-{step}")

# --- load ---
model_sd, optim_sd = get_state_dict(model, optimizers=optimizer)
state_dict = {
    "model": model_sd,
    "optim": optim_sd,
    "step": 0,
}
dcp.load(state_dict=state_dict, checkpoint_id="/ckpt/step-1000")
set_state_dict(model, optimizers=optimizer,
               model_state_dict=state_dict["model"],
               optim_state_dict=state_dict["optim"])
step = state_dict["step"]
```

`get_state_dict` / `set_state_dict` from
`torch.distributed.checkpoint.state_dict` are the current recommended
way to bridge FSDP2-sharded model and optimizer state into a
DCP-friendly `state_dict`; consult the linked docs for the current
signatures in the PyTorch release you are on.

## Making non-tensor state stateful: the `Stateful` protocol

The `state_dict` above only handled model, optimizer, and a scalar
step. To include sampler position, RNG state, LR scheduler, and
gradient scaler, DCP exposes the **`Stateful`** protocol:

```python
from torch.distributed.checkpoint.stateful import Stateful

class AppState(Stateful):
    def __init__(self, model, optimizer, lr_scheduler,
                 scaler, sampler, step):
        self.model, self.optimizer = model, optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.sampler = sampler
        self.step = step

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(self.model,
                                            optimizers=self.optimizer)
        return {
            "model": model_sd,
            "optim": optim_sd,
            "lr_sched": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "sampler": self.sampler.state_dict(),
            "step": self.step,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
        }

    def load_state_dict(self, sd):
        set_state_dict(self.model, optimizers=self.optimizer,
                       model_state_dict=sd["model"],
                       optim_state_dict=sd["optim"])
        self.lr_scheduler.load_state_dict(sd["lr_sched"])
        self.scaler.load_state_dict(sd["scaler"])
        self.sampler.load_state_dict(sd["sampler"])
        self.step = sd["step"]
        torch.set_rng_state(sd["rng_cpu"])
        torch.cuda.set_rng_state_all(sd["rng_cuda"])
```

Then `dcp.save({"app": AppState(...)}, checkpoint_id=...)` and
`dcp.load({"app": AppState(...)}, checkpoint_id=...)` handle every
piece of state above in one call. This is the pattern chapter 2's
inventory was building toward; every real production trainer wraps
its stateful surface in one `AppState`-like class.

## Async save

`dcp.async_save(state_dict, checkpoint_id, ...)` returns a `Future`.
The state is copied off-GPU to a staging buffer during the call; a
background thread does the disk write. The training loop continues as
soon as the staging copy completes — the disk write overlaps with the
next training steps.

The pattern is:

```python
future = None

for step in range(start_step, total_steps):
    train_step()
    if step % SAVE_EVERY == 0:
        # Wait for the previous async save before starting the next.
        if future is not None:
            future.result()
        future = dcp.async_save(
            {"app": app_state},
            checkpoint_id=f"/ckpt/step-{step}",
        )

# At end of training, drain the last one.
if future is not None:
    future.result()
```

Two rules for async save:

- **Wait for the previous save before starting the next.** Two
  concurrent saves compete for storage bandwidth and staging memory;
  you can end up with both slow and neither committed. The DCP
  documentation warns about this explicitly.
- **Do not mutate `AppState`'s underlying tensors between the
  `async_save` call and its `result()`.** DCP's staging copy usually
  happens in the initial synchronous prelude — but consult the
  version docs; some pieces of the copy happen after `async_save`
  returns.

For pure archival checkpoints (once a day, off the critical path),
use the synchronous `dcp.save`. For every-N-step "recent copy"
checkpoints, use `async_save`. Exercise 1 measures the step-time
delta between the two on your fabric.

## Resharding on load

The interesting behavior that makes DCP different from `torch.save`:
you can load onto a **different world size or sharding** than what
saved the checkpoint. The mechanism:

1. On save, each rank writes its shard along with metadata that
   records the tensor's *global* shape and this rank's *local*
   offsets into it.
2. On load, each rank looks at its current world's sharding for
   each tensor, computes which byte-ranges of the global tensor it
   needs, and reads exactly those byte-ranges — potentially spanning
   multiple on-disk shard files.

Practical consequences:

- **Elastic reshape works transparently.** If chapter 4's rendezvous
  brings up a job on `G' ≠ G` ranks, DCP reshards on the load path.
  No re-materialization or offline conversion step.
- **Cross-parallelism changes work.** Save on FSDP tensor-parallel
  = 8, load on FSDP tensor-parallel = 4. Save on `HYBRID_SHARD` at
  one hybrid-group size, load at another. The metadata carries
  enough to reconstruct.
- **You cannot silently change model architecture.** Adding a layer,
  changing a hidden dimension, or renaming parameters requires
  either a matching in-code migration or `strict=False`-style
  handling. DCP will reshard; it will not reshape a model.

Exercise 1's core drill is: save on world size N, kill everything,
launch on world size N/2, load, resume. If your `AppState` is
correctly written, that just works.

## Storage backends

DCP separates the *what* (planners + `state_dict` sharding) from the
*where* (storage backend). The current in-tree backend is
`FileSystemWriter` / `FileSystemReader`, which writes many small
files to a filesystem path. That path can be:

- A **parallel filesystem** mount (Lustre, WEKA, FSx for Lustre —
  see mod-105 chapter 7). This is the standard production choice.
- An **NFS mount** for small-scale experiments or debugging.
- An **S3-compatible object store** via a plugin. Ecosystem plugins
  (see `resources.md`) mount S3 as a DCP storage backend; the
  built-in filesystem backend also composes with FUSE mounts like
  AWS Mountpoint for Amazon S3.
- A **local NVMe scratch** as a staging tier before an async
  background upload to durable storage — the "tiered checkpoint"
  pattern documented in the CheckFreq and Gemini research papers
  (see `resources.md`).

The choice of backend is a mod-105 chapter 7 choice; DCP does not
much care.

### The tuning knobs that matter

- **Chunk size on the writer.** Larger chunks = fewer files, less
  metadata pressure on parallel FS, better S3 throughput at the
  cost of amplified write-per-mutation. Consult the current
  `FileSystemWriter` signature for the exact parameter name.
- **Thread count on the writer / reader.** Concurrent I/O per rank.
  On a saturated storage tier, more threads hurt; on a per-rank
  bandwidth-limited tier, more threads help until you saturate.
- **Planner overrides.** The default `DefaultSavePlanner` and
  `DefaultLoadPlanner` are correct; you touch them only when you
  need a custom sharding transformation on load (rare) or a custom
  compression / dedup step (also rare).

## The stateful sampler pattern

Chapter 2 identified sampler state as the most-commonly-missed piece
of checkpoint state. Wire it into DCP by making the sampler itself
implement `Stateful`, then include it in `AppState`:

```python
class StatefulShardedSampler(torch.utils.data.Sampler, Stateful):
    def __init__(self, dataset_size, world_size, rank, seed):
        self.dataset_size = dataset_size
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.position = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.dataset_size, generator=g).tolist()
        my_indices = perm[self.rank::self.world_size]
        for idx in my_indices[self.position:]:
            self.position += 1
            yield idx
        self.position = 0
        self.epoch += 1

    def state_dict(self):
        return {"epoch": self.epoch,
                "position": self.position,
                "seed": self.seed}

    def load_state_dict(self, sd):
        self.epoch = sd["epoch"]
        self.position = sd["position"]
        self.seed = sd["seed"]
```

Every rank has its own sampler state (its own `position` within its
own permutation). On save, each rank's sampler state ends up in the
per-rank shard of the DCP checkpoint. On load with a *different*
world size, you have two choices:

1. **Restart the epoch.** Simplest correct behavior; you re-see
   some data. Acceptable if `epoch >> 1`.
2. **Re-plan from `(epoch, seed)`.** Deterministically reconstruct
   the epoch's global permutation, compute the union of unseen
   indices across old ranks, redistribute across the new ranks.
   Correct but complex. mod-103 chapter 4 covers the sharded-loader
   plumbing that makes this tractable.

Exercise 1 asks you to implement option (1) first, then argue in
writing why or why not option (2) is worth the code in your setting.

## Compatibility and version pinning

Two hazards that trip production platforms:

- **DCP file-format compatibility.** DCP's on-disk format is
  documented in the module and has evolved with PyTorch releases.
  Checkpoints written by one PyTorch version are usually readable by
  the next, but the reverse is not guaranteed. Pin your training and
  recovery jobs to the same PyTorch version, and treat
  cross-version reads as a migration event, not a routine load.
- **`get_state_dict` / `set_state_dict` API churn.** The bridging
  helpers between FSDP2 and DCP have moved and been renamed several
  times. Pin your import path per PyTorch release and add a smoke
  test that a checkpoint written today loads on the version you
  intend to recover with.

## Things DCP is not

- **Not a portable inference format.** A DCP directory is meant to
  be re-loaded into the same (or a resharded) training job. For
  serving, export to a consolidated Safetensors / HuggingFace-style
  format — this is the "consolidate on demand" step chapter 2
  mentioned.
- **Not a git for models.** DCP does not deduplicate across
  checkpoints. If you want incremental / deduplicated storage, layer
  it in a storage plugin.
- **Not append-only.** A partially-written DCP directory is
  corrupt. Use atomic-rename semantics on your storage tier (write
  to a temp path, rename on success) so recovery never picks up a
  half-written checkpoint.

The atomic-rename pattern is a common recovery bug: a save was
interrupted at step S, the training job restarts and DCP scans
`/ckpt/`, sees a torn `step-S` directory alongside a good `step-S-1`,
picks the newest by mtime, and fails to load. Write to
`/ckpt/step-S.tmp/`, `rename` to `/ckpt/step-S/` on success, and
teach the load path to ignore `.tmp` suffixes.

## Summary

- DCP is the in-tree PyTorch solution for parallel, shard-aware,
  resharding-friendly checkpointing. Use it — do not roll your own
  atop `torch.save`.
- The two-line change to make a checkpoint correct: implement a
  `Stateful` `AppState` that includes model, optimizer, LR scheduler,
  step, RNGs, scaler, sampler; call `dcp.save({"app": app_state},
  checkpoint_id=...)`.
- `dcp.async_save` moves the disk write off the critical path. Wait
  for the previous save before starting the next; do not mutate the
  underlying state between call and completion.
- DCP reshards on load. Same code loads a checkpoint written at a
  different world size or sharding. This is what makes chapter 4's
  elastic reshape flow work at all.
- The sampler is a stateful object; wire it into `AppState`. Missing
  it is the single most common correctness bug at this layer.
- Version-pin the writer and reader; use atomic-rename semantics so
  a torn checkpoint never gets promoted to "latest".
