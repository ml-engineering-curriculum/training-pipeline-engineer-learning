# Checkpoint Anatomy and the Storage-Cost Model

Before we look at any specific checkpoint API, we need to agree on
two things: **what has to be in the checkpoint for the resume to be
correct**, and **how much it costs to write and read one**. Most
production incidents that trace back to checkpointing are one of two
mistakes — a missing piece of state, or a cost model that made the
platform team believe the checkpoint was "free" when it wasn't.

## What actually needs to be checkpointed

A resume is only correct if, after loading, every stochastic and
stateful component in the pipeline continues *as if the failure never
happened*. Anything you leave out silently changes the training
distribution.

The full inventory:

- **Model parameters.** Every rank's shard of every FSDP-sharded
  parameter, or the full parameter set if you are running DDP.
- **Optimizer state.** For Adam / AdamW / Lion / Muon and their
  variants, this is the running moments — the same shape as the
  parameters, sometimes 2× or 3× their size. This is often the
  largest single piece of state on disk.
- **Learning-rate scheduler state.** Cosine schedules, warm-up
  counters, LR-milestone state. Small in bytes; catastrophic to lose.
- **Global step / sample counter.** The single source of truth for
  "where in training am I". Sampler state and LR schedules both
  depend on it.
- **RNG state per rank.** PyTorch's CPU RNG, per-GPU CUDA RNG, NumPy
  RNG, and any Python `random` state you use. Missing this breaks
  reproducibility and, for augmentation-heavy pipelines, breaks the
  claim that resume is equivalent to no-crash.
- **Dataloader / sampler state.** The exact position in the dataset,
  including the current epoch's shuffle permutation. mod-103 chapter
  4 introduced the "stateful sampler" pattern; chapter 3 of this
  module wires it into DCP.
- **Gradient scaler (mixed precision).** The dynamic-loss-scale
  state for automatic mixed precision. Losing this triggers a
  scale-storm on resume.
- **EMA / shadow parameters.** If you keep an exponential moving
  average of the weights (common for diffusion, some LLM finetuning
  regimes), the EMA is a whole second copy of the model that has to
  be checkpointed and restored.

The two categories most commonly left out:

1. **Sampler / dataloader state.** Teams check `state_dict()` on the
   model and forget the loader is a stateful object. On resume they
   restart the epoch from index 0, silently over-sampling the data
   they already saw and under-sampling the tail.
2. **RNG state.** Teams write a checkpoint that reproduces on
   deterministic-mode CPU-only smoke tests and fails to notice that
   real training's augmentation, dropout mask, and initialization
   noise all diverge from the pre-crash trajectory.

The chapter 3 DCP `AppState` example wires all of these into one
`state_dict` interface.

## The cost model

A checkpoint has two costs: **wall-clock cost of the save** (how much
step time the GPU spends waiting for the write to finish) and
**wall-clock cost of the load** (how long recovery takes).

For a synchronous save on `G` ranks with total parameter+state size
`S` bytes and aggregate write bandwidth `B` to your storage tier:

```
T_save (sync) ≈ max(S / (B_per_rank · G), S / B_aggregate)
```

For a synchronous load spread across the same `G` ranks:

```
T_load ≈ S / min(B_read_aggregate, B_per_rank · G)
```

The bandwidth numbers come from mod-105 chapter 7: Lustre delivers
per-client throughput measured in ~1 to 10 GB/s and aggregate
throughput measured in ~100s of GB/s to ~few TB/s depending on
provisioning; S3 is measured in ~10s of Gb/s per prefix with
aggregate scale determined by request concurrency. Consult your
cluster's mod-105 baseline before assuming a number.

A concrete anchor: a 70B-parameter model in BF16 is about 140 GB of
parameters. AdamW optimizer state (two moments per parameter,
usually kept in FP32) adds ~560 GB — so the raw state is ~700 GB
before RNG, sampler, and EMA. At 10 GB/s aggregate write bandwidth,
a naïve synchronous consolidated save is 70 seconds. At 100 GB/s
sharded parallel write, it is 7 seconds. At a bad S3 config with
serial single-writer, it is ten minutes. The design decisions in
the next section are what determine which of those numbers you
actually get.

### Save cost: three axes to pick on

- **Sharded vs. consolidated.** Sharded writes have every rank
  stream its own shard in parallel; a `G`-rank job writes `G` files
  and the aggregate bandwidth is close to `G × B_per_rank`.
  Consolidated writes gather all shards onto one rank before writing
  a single "portable" tensor. Consolidated saves are convenient for
  downstream inference or for handoff to another team, but they are
  strictly slower during training. Split the two use cases: shard
  during training, consolidate on demand.
- **Sync vs. async.** A sync save blocks the training step until the
  write completes. An async save copies the state to a staging buffer
  (usually CPU pinned memory) and returns; a background thread does
  the actual disk write. The GPU keeps training. Async is nearly
  always the right choice for the intra-run "keep a recent copy"
  checkpoint; sync is fine for the once-a-day "long-term archive"
  checkpoint. Chapter 3 wires DCP's `async_save` into this pattern.
- **Full vs. incremental.** Only the pieces of state that changed
  need to be rewritten. In practice, *all* of the model and
  optimizer changes each step, so incremental checkpointing rarely
  helps for training checkpoints. It is a research direction (see
  MSFT's "Just-in-Time Checkpointing" paper referenced in
  `resources.md`), not a shipped default.

### Load cost: what determines it

- **Number of shards vs. reading world size.** DCP reshards on read;
  chapter 3 covers the algorithm. If the saved checkpoint has 512
  shards and you resume on 256 ranks, each rank reads two shards.
  Even in the best case, aggregate read bandwidth caps recovery
  time.
- **Read locality.** A cross-region S3 bucket to a compute-region
  cluster adds tens of ms per object of latency; multiplied by
  thousands of shards, that becomes a significant fraction of the
  load time. Stage checkpoints on the same-region storage tier that
  training reads.
- **Metadata scaling.** A DCP checkpoint is a directory of many
  files. If your storage tier has slow metadata (S3 LIST, Lustre MDS
  hot-spot), the initial "figure out what's here" phase dominates
  load time. Chapter 3 covers the DCP metadata layout and the
  operators that break under it.

## How often to checkpoint: the failure-window formula

For a job with per-hour failure rate `f` and checkpoint interval
`Δt`, the *expected work lost per failure* is `Δt / 2`. The
*fraction of wall-clock spent* checkpointing is roughly
`T_save / Δt`. Total expected wasted-time fraction as a function of
`Δt` is:

```
W(Δt) = (T_save / Δt) + f · (Δt / 2 + T_load) / (1 hour)
```

Minimizing `W` yields (for the leading terms)
`Δt* ≈ sqrt(2 · T_save / f)`, ignoring `T_load` and other constants.
For plausible production numbers — `T_save = 20 s` async and
`f = 0.25 failures/hour` — the optimum is on the order of tens of
minutes. That is a first-order estimate: use it to set a starting
point, then adjust based on measured recovery time and how bursty
your failure rate actually is.

A useful sanity check: if your team's answer to "how often do you
checkpoint" is "every hour" but the answer to "how often do you
crash" is "twice a day", you are throwing away about six hours of
work per day just to a mis-tuned interval. Reset with the formula.

## Two anti-patterns

The two designs that show up in newly-set-up training platforms and
should be replaced before scaling:

- **"Rank 0 writes everything."** All ranks all-gather their state
  to rank 0, rank 0 writes one giant file. Every rank blocks on rank
  0's write bandwidth; the CPU on rank 0 melts under a 100+ GB
  gather. This design is O(1) write concurrency; DCP is O(G).
- **"Save every N steps, N chosen without measurement."** Either N
  is small and you spend real time in the save path, or N is large
  and every failure eats hours. Use the failure-window formula
  above; then tune based on the actual measured `T_save` and
  observed failure rate.

Both are fixable with a `torch.distributed.checkpoint` (DCP) rewrite
and a real measurement of `T_save` on your storage tier. Chapter 3 is
that rewrite.

## Storage-side considerations from mod-105

The checkpoint tier is one of the four link tiers from mod-105
chapter 1 — tier 4, storage. All the caveats from that chapter apply:

- **Parallel filesystem hot metadata servers.** Millions of small
  files (a pathologically-configured DCP directory) will hammer
  Lustre's MDS. Batch small tensor writes into fewer, larger files;
  chapter 3 covers the DCP `FileSystemWriter` chunk parameters.
- **S3 request rate limits.** A 5 000-shard checkpoint saved to a
  single S3 prefix will run into per-prefix rate limits. Hash-prefix
  or use multiple prefixes.
- **GPUDirect Storage for checkpoint reads.** Cold-start checkpoint
  loads on a very-large model can be materially accelerated by GDS;
  see mod-105 chapter 6. The gain matters most when you are
  restarting frequently.
- **Object-store consistency semantics.** S3 provides strong
  read-after-write consistency; older EOS-style stores did not.
  Always confirm the semantics before assuming a completion barrier
  is durable.

Chapter 3's DCP recipe uses a `FileSystemWriter` targeting either a
parallel FS mount or an S3 prefix — same code, different storage
backend.

## Summary

- A correct checkpoint includes model, optimizer, LR scheduler, step
  counter, RNGs on every rank, sampler/dataloader position, mixed-
  precision scaler, and EMA if you have one. Missing sampler and
  RNG state are the two most common bugs.
- The save cost model is `S / B_effective`. The three axes are
  sharded vs. consolidated, sync vs. async, and full vs. incremental
  — pick sharded + async + full for training checkpoints, save
  consolidated on demand for handoff.
- The load cost model is symmetric but constrained by reshard math
  and metadata scaling. Cold-start recovery time is what chapter 8's
  goodput SLO cares about.
- The optimal save interval is roughly `sqrt(2 · T_save / f)`.
  Measure both terms before setting the interval; the number teams
  guess is often off by an order of magnitude.
- "Rank 0 writes everything" and "N chosen without measurement" are
  the two design anti-patterns to eliminate before scaling. Chapter
  3 replaces them with DCP.
