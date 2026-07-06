# WebDataset: A Tar-Shard Pipeline That Saturates 8 GPUs

WebDataset is the archetype of the "big sequential shard" data-loading
strategy. It replaces "one file per sample on a POSIX filesystem" with
"one POSIX tar file per shard", read as a sequential byte stream and
demuxed into samples inside the loader. That single change turns the
worst-case object-store workload — millions of small random-access
GETs — into the best-case one — a few thousand large sequential GETs
that saturate the network. This chapter shows how the tar-shard model
works, how to design a shard layout that keeps 8 GPUs fed, and where
the pipeline can still stall.

## Motivation: I/O patterns your object store loves and hates

Object storage (S3, GCS, MinIO) is optimized for throughput on large
sequential reads and is *deliberately* not optimized for latency on
small random reads. The public S3 performance guidance calls out
"3 500 PUT/COPY/POST/DELETE and 5 500 GET/HEAD requests per second per
prefixed partition" as the target ceiling for concurrent requests
against a prefix, and steers users toward multi-megabyte object sizes
to hit peak throughput per request.
<!-- needs-research: cite the current AWS S3 "Best practices design patterns: optimizing Amazon S3 performance" doc for the exact numbers when authoring, so this stays fresh. -->

Two rules of thumb follow:

- **Object sizes** below a few MB waste most of each request's cost on
  metadata and TLS setup.
- **A single reader** trying to fan out to millions of small keys will
  hit prefix-level rate limits long before it hits the bucket-level
  bandwidth ceiling.

If your corpus is a billion JPEG or NPZ files (image / audio /
multimodal training), a naive `DataLoader` reading them individually
will spend most of its time in HTTPS handshake and none of it moving
bytes. The WebDataset design is a targeted fix.

## The WebDataset format

The format is deliberately trivial: a **shard** is a POSIX tar file,
and a **sample** is a group of tar members that share a common basename.
For example a shard containing two samples might look like:

```
00000123.jpg
00000123.cls
00000123.json
00000124.jpg
00000124.cls
00000124.json
```

The reader iterates the tar sequentially, groups members by basename,
and yields a Python `dict` per sample keyed by extension. Sequential
tar reads mean the underlying I/O pattern is one large streaming GET
per shard, which is exactly the shape S3 and GCS were built for.

Two shard-format properties are worth internalizing:

1. **No random access.** You cannot seek within a shard efficiently.
   Any operation that would require it (e.g., "give me sample 42 of
   shard 7") is designed out of the API. Sampling is done by shuffling
   shards and streaming through them.
2. **No embedded metadata.** The shard itself does not know how many
   samples it contains, what tokenizer produced them, or how they were
   ordered. That knowledge lives in a **manifest** you own (chapter 7).

Reference: the `webdataset` library README and the "WebDataset — A
Format for Deep Learning" white paper describe the format
formally.<!-- needs-research: cite the current github.com/webdataset/webdataset README and the accompanying paper by Aizman et al. when authoring. -->

## Shard-size arithmetic

The dominant design decision is **shard size**. Two constraints pull
against each other:

- **Large enough** that each shard read amortizes the object-store
  request cost and lets a single GET saturate the per-connection
  bandwidth. In practice this puts the floor at roughly 100 MB.
- **Small enough** that (a) a shard fits in the shuffle buffer with
  headroom, (b) losing or refetching one shard is cheap, and (c) the
  number of shards per epoch exceeds the total number of loader workers
  in the fleet by at least an order of magnitude, so the shard shuffle
  has room to be random.

The community-conventional range for pretraining and imaging workloads
is roughly **100 MB – 1 GB per shard**, with 512 MB a common default.
Below 100 MB you are back in the small-object regime; above a few GB
a single unlucky worker holds up a rank for too long. Pick a shard
size, write it down in the runbook, and don't drift.

Given a corpus size `C` in bytes and a chosen shard size `S`, the
epoch has `C / S` shards. For a 100 B token pretraining corpus with
2-byte token IDs, `C ≈ 200 GB`. With `S = 500 MB` you get ~400 shards
— enough to shuffle, small enough that any individual shard's read
finishes in a second on a decent link.

For a multimodal corpus at 5 KB / sample and 2 B samples, `C ≈ 10 TB`.
With `S = 500 MB` you get 20 000 shards — plenty of room for a fleet
of ranks × workers to see distinct shards each.

## The reader pipeline

The canonical WebDataset pipeline is a chain of composable operations
that each act on the shard-URL stream or the sample stream:

```python
import webdataset as wds

pipeline = (
    wds.WebDataset(
        urls="s3://corpus/train/{00000..00399}.tar",
        shardshuffle=True,
        resampled=False,          # deterministic epoch
        nodesplitter=wds.split_by_node,   # split shards across ranks
        workersplitter=wds.split_by_worker,  # then across DataLoader workers
    )
    .shuffle(1000)                # per-worker in-memory shuffle buffer
    .decode("pil")                # decode image bytes → PIL
    .to_tuple("jpg", "cls")       # (image, label) tuples
    .map_tuple(transform, identity)
    .batched(batch_size)
)

loader = wds.WebLoader(pipeline, num_workers=8, batch_size=None)
```

A few pieces of that pipeline are worth naming, because they are what
the exercise measures:

- **`split_by_node` and `split_by_worker`** — the two-level shard
  distribution that gives each `(rank, worker)` pair a disjoint slice
  of the shard list per epoch. This is how coverage (chapter 1's
  Invariant 2) is preserved across the fleet.
- **`shuffle(N)`** — the local shuffle buffer, sized in samples. This
  is the *only* randomness after `shardshuffle` picks the shard
  order. Its size is what stands between you and shuffle collapse
  (chapter 6).
- **`resampled=False`** — the default is a "fresh" epoch: every
  shard is seen exactly once. Setting `resampled=True` switches to
  infinite sampling with replacement, which trades determinism for
  simpler bookkeeping.
- **`WebLoader`** — a thin wrapper around `torch.utils.data.DataLoader`
  that knows the WebDataset yields IterableDataset batches. Do not use
  `DataLoader(..., num_workers=W, batch_size=B)` with a map-style
  wrapper; you will collide with the tar iterator's own batching.

## Saturating 8 GPUs on one node

The exercise target is: on an 8×GPU node, sustain the per-step token
throughput required by a Llama-3-class training loop *without I/O
stalls*. Two knobs dominate.

### Worker fanout

The dataloader worker count `W` must be sized so `W ≥` "the number of
concurrent shard reads you need to keep the network pipe full". Since
each worker holds one shard open at a time, and each shard read
takes roughly `S / link_bandwidth` seconds, you need enough workers
that the aggregate outstanding-shard time covers the compute time.

For an 8-GPU node with, say, 25 Gbps of effective object-store egress
per rank and `S = 500 MB` shards:

- Time to fully read a shard from S3: `500 MB / 25 Gbps ≈ 0.16 s`.
- If step time is 0.3 s and each shard has ~2 K samples at batch 32,
  each rank consumes a shard every `2000 / (32 · steps_per_second) ≈
  20 steps ≈ 6 seconds`. Two overlapping workers per rank keeps the
  queue full with margin.

The exact numbers depend on your object store, VPC endpoint, and
sample-decode CPU cost. The exercise makes you measure them.

### Prefetch and shuffle depth

`prefetch_factor` (in the underlying `DataLoader`) controls how many
batches each worker keeps queued for the main process. The default of
2 is almost always too shallow at training scale — a single stall on
one worker's shard read empties the queue.

The shuffle buffer size trades:

- **Memory** — `shuffle(N) × sizeof(sample)` bytes per worker.
- **Effective randomness** — samples within a window of size `N`
  can be reordered; samples outside it are locked into shard order.

Common defaults: `shuffle(1000)` for tokenized pretraining shards,
`shuffle(2000-5000)` for image workloads where sample-similarity
correlation within a shard is high. Chapter 6 shows what "too small"
looks like on a loss curve.

## Composing with DDP / FSDP

Two invariants when composing WebDataset with a distributed training
strategy from mod-101:

1. **Each rank sees a disjoint shard set per epoch.** `split_by_node`
   handles this if the total shard count is a multiple of world size.
   If it isn't, a few shards get dropped or duplicated depending on
   your version of the splitter — verify explicitly and cite the fix
   in the runbook.
2. **The epoch boundary must be a synchronization point.** Ranks
   with different worker fanout can finish their share at slightly
   different times; without a barrier the fast ranks start epoch N+1
   while the slow ones finish epoch N, and your global batch order
   diverges from the plan. `WebLoader.with_epoch(nsamples)` fixes the
   per-epoch sample count explicitly so this is deterministic.

For IterableDataset-backed loaders, `DistributedSampler` is **not**
what you want (it is for map-style datasets). WebDataset does its own
sharding.

## Common failure modes (preview)

Two failure modes from chapter 6 show up first on WebDataset pipelines
because their surface area is largest:

- **Silent shard corruption** — one tar file has a truncated member.
  Reader raises inside a decoder, worker recovers by skipping the
  sample, and coverage silently degrades. Fix by hashing every shard
  in the manifest and rejecting any shard whose live hash disagrees.
- **Shuffle collapse** — `shuffle(N)` is set too small relative to
  intra-shard sample correlation. The loss curve gets a periodic
  wobble that lines up with shard boundaries. Fix by increasing `N`
  until the wobble disappears and documenting the new value.

## Summary

- WebDataset converts an object-store-hostile "millions of small
  files" workload into an object-store-friendly "few thousand large
  sequential reads" workload.
- Shard size 100 MB – 1 GB; number of shards should be >> loader
  worker count in the fleet.
- Worker fanout and prefetch depth are the two knobs that keep an
  8-GPU node fed; size them from measured step time and shard read
  time.
- `split_by_node` + `split_by_worker` + `with_epoch` are how
  coverage and determinism are preserved across the fleet.
- The shard manifest and per-shard hashes (chapter 7) are what make
  the pipeline *auditable* — everything else is throughput.
