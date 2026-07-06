# The Staging Layer: S3 / GCS → Lustre / WEKA, on a Compute-Matched I/O Budget

WebDataset and MosaicML Streaming both assume the shards live "near"
the training fleet in a sense that has to be made precise. In practice
"near" means a **staging layer** — a warm on-cluster cache in front of
the cold object store — sized to match the per-step compute budget so
the loader never sees an S3 or GCS round trip on the hot path. This
chapter is about designing that layer.

## Motivation: why object storage is on the wrong side of the network

For a modern training node:

- **Object store egress** — S3, GCS, blob storage — provides
  large aggregate throughput but at *round-trip latencies* on the
  order of 10–50 ms per request over the public-internet or VPC
  endpoint path, and at unpredictable tail latencies (p99 can be
  seconds). Enough for a batch job; not enough for a hot training
  loop.
- **Parallel filesystem** — Lustre or WEKA mounted on the compute
  nodes — provides much lower latency (sub-millisecond per open,
  a few-ms per read) and predictable tail. Bandwidth per client is
  typically limited by the client's NIC.
- **Node-local NVMe** — order of a GB/s per drive, sub-100 μs
  latency, but capacity-bounded (a few TB per node).

The training loop can afford the parallel-filesystem latency budget
but not the object-store one. So somewhere between the cold store
and the hot loop there has to be a fast-tier cache. The staging
layer is that cache.

## The compute-matched I/O budget

Before you design any cache, you compute the byte-rate budget the
loader has to sustain. Pull the units from chapter 1:

- **`t_step`** — wall-clock per training step (from mod-101).
- **`B_step`** — bytes of samples consumed per step.

Budget: the staging layer must deliver `B_step / t_step` bytes per
second, per rank, with tail latency low enough that the deepest
downstream queue in the pipeline (prefetch / shuffle buffer) does not
drain during a burst.

Concrete example.

- 8×H100 node, Llama-3-class 7 B model, step time 0.30 s, global
  batch = 8 · 4 · 2048 = 65 536 tokens (with `micro_batch=4`,
  `seq_len=2048`).
- 2 bytes/token pretokenized: `B_step ≈ 131 KB` per node per step.
- Throughput requirement: `131 KB / 0.30 s ≈ 440 KB/s` per node.

**That per-node token feed is trivial.** The reason the staging layer
is a real subsystem is not the average byte rate — it's:

1. **Tail latency on individual shard fetches** — a single shard
   miss that goes to S3 stalls the worker for tens of ms to
   seconds.
2. **Non-token modalities** — a multimodal training job with 100 KB
   images at the same step rate needs 50 MB/s per node, and a
   VideoDataset-style pipeline at MB-scale samples needs GB/s per
   node.
3. **Compute-bound tokenization at read time** — if you skipped
   chapter 4 and are still tokenizing in the loader, per-node CPU
   demand goes up 10–100×.

For everything past "small pretokenized shard", staging is a real
budget item.

## The three-tier storage hierarchy

The stack in most training platforms:

```
tier 0 — object store  (S3, GCS, Azure Blob) — "cold, canonical"
             │
             ▼
tier 1 — parallel filesystem  (Lustre, WEKA, GPFS, FSx) — "warm, shared"
             │
             ▼
tier 2 — node-local NVMe  (/local, /tmp, /nvme) — "hot, ephemeral"
             │
             ▼
      loader workers → prefetch buffer → GPU
```

Not every deployment has all three. Some skip tier 1 (small
corpora, one-node runs). Some skip tier 2 (Lustre-only shops).
Rules of thumb for choosing:

- **Corpus fits on the node's local NVMe** (e.g., 400 GB corpus, 8×
  1.9 TB NVMe drives): just stage to tier 2 at run start. Simplest,
  fastest, most predictable.
- **Corpus fits on Lustre / WEKA share** but not on any one node's
  NVMe: stage to tier 1, read from tier 1. Do not use tier 2. The
  parallel filesystem was designed for this workload.
- **Corpus does not fit on any hot tier**: stream from tier 0
  through a *rolling* per-node cache in tier 2. WebDataset and
  StreamingDataset both support this out of the box; the loader's
  `predownload` and shuffle buffer are what keep the ratio of hot
  to cold reads high enough.

The exercise makes you build the third case for a corpus you
deliberately size larger than the fleet's hot tier.

## Prefetch semantics

The staging layer's job is not just "copy the shard" — it is "have
the shard local **before the training worker asks for it**". Three
common designs:

### 1. Bulk pre-stage at job start

The simplest possible design: before the training job starts, a
copy step drags the whole corpus from tier 0 to tier 1. The training
loop then reads exclusively from tier 1.

- **Pro**: no runtime I/O to object storage; predictable throughput
  across the run.
- **Con**: adds hours (at TB scale) to job launch; wastes storage
  if the corpus is larger than the run consumes; requires manual
  invalidation when the corpus changes.

Ideal for short reruns of a fixed corpus. `s5cmd` or `gsutil -m` are
the standard high-throughput copy tools; a parallel `rclone` also
works.

### 2. Rolling per-node cache

Each node runs a small cache manager (or leans on the loader's own,
as in `StreamingDataset`'s `local` + `cache_limit`). The manager:

- Fetches shards on demand as the loader requests them.
- Evicts on an LRU or size-bounded policy.
- Pre-fetches N shards ahead of the read cursor (loader-driven or
  cache-manager-driven).

This is the default pattern for `StreamingDataset`. It composes
naturally with the reader; the cache directory is just a filesystem
path the loader treats as writable.

- **Pro**: bounded storage cost; no launch-time delay.
- **Con**: tail latency during warm-up; cache-miss storms if the
  shuffle causes the working set to cycle faster than the cache
  can refill.

### 3. Coordinated prefetch daemon

At larger scale (hundreds of ranks, tier 1 shared cache) a
dedicated **prefetch daemon** runs alongside the training job. It
watches the loader's declared shard schedule (from `index.json` +
current sample cursor) and drives an out-of-band copy from tier 0
to tier 1, ahead of the loader.

- **Pro**: highest hit rate; tier 1 becomes the single warm cache
  across the fleet.
- **Con**: another coordinated system to run and monitor; the
  daemon becomes another thing that can fail.

`fsspec` + `smart_open`, `Alluxio`, and cloud-vendor caching layers
(AWS Mountpoint for Amazon S3 with local cache, GCP Filestore /
GKE-CSI FUSE) are the productionized versions of this.

## Cache eviction and the working-set model

Any per-node cache has to answer: what does it evict, and when?

- **LRU** is the safe default. Works well when the loader visits
  shards in a mostly-monotonic order (which is what happens after a
  `shardshuffle` fixes the order for the epoch).
- **Explicit pinning** for a *reference* set of shards — say, an
  eval or held-out set that gets read every step — prevents the
  training loop from evicting them.
- **Size limit** must be strictly enforced. The
  `StreamingDataset(cache_limit=...)` knob is the pattern: when the
  cache is full, the next fetch triggers an eviction *before* the
  fetch, not after (which would risk running out of disk mid-write).

The **working set** is the shard set the loader touches over one
"cache period" (typically one epoch). If working-set > cache size,
you thrash: every shard is fetched more than once per epoch, and
your effective throughput drops toward tier-0 speed. Chapter 6
lists the signature of this failure.

## S3 / GCS access patterns you actually want

Two configuration mistakes to avoid on the object-store side.

1. **Request concurrency per prefix.** S3 rate limits per prefix
   (chapter 2). If your entire corpus lives under `s3://corpus/train/`,
   thousands of concurrent workers can hit the per-prefix limit even
   though the bucket is nowhere near full utilization. Distribute
   shards across many prefixes (e.g., `s3://corpus/train/aa/`,
   `bb/`, `cc/`, ...) so the request budget grows with the shard
   count. `MDSWriter` does this by default with hashed subdirs;
   verify yours does the same.
2. **Multi-part reads.** For shards > ~8 MB, use the S3 client's
   multi-part GET so a single shard's throughput is not
   single-TCP-stream-bound. `boto3`'s
   `s3.download_file(..., Config=TransferConfig(...))` and the
   AWS SDK's default transfer manager both do this; `pyarrow.fs` +
   `S3FileSystem` also do it. Do not use `urllib` or `requests`.

## The staging layer's runbook artifact

The runbook (chapter 7) requires the staging layer to expose two
per-shard properties as observable metrics:

- **First-fetch time** — wall-clock from "cache miss" to "shard
  visible in the cache". This is the tail-latency signal.
- **Serve time** — wall-clock from "loader open()" to "loader
  read() completes". This is the hot-path signal.

Both are exported from the cache manager or the loader itself
(`StreamingDataset` has hooks) into your metrics pipeline (mod-108
covers the observability half). A p99 for either that starts to
climb is your leading indicator that either the object store is
throttling or the cache is thrashing — long before the loss curve
notices.

## When the staging layer is *not* the answer

Two anti-patterns to name explicitly:

1. **Adding staging to fix a slow tokenizer.** If your loader worker
   is CPU-bound on tokenization, more storage bandwidth cannot help.
   Fix the tokenizer (chapter 4) before you look at storage.
2. **Adding staging to fix an undersized shard.** If your shards are
   1 MB each, staging can't paper over the request-count problem.
   Repack into larger shards.

The staging layer is the answer when you have a right-sized shard
format, a right-sized tokenizer, and the observed stall shape is
"loader queue drained during shard fetch".

## Summary

- The staging layer's byte-rate contract is derived from step time
  and per-step byte demand; compute it before designing the cache.
- Three tiers: object store (cold), parallel filesystem (warm),
  node-local NVMe (hot). Pick the smallest hot tier that fits the
  working set.
- Three prefetch designs: bulk pre-stage, rolling per-node cache,
  coordinated daemon. `StreamingDataset`'s built-in rolling cache is
  the default for LLM pretraining.
- LRU + explicit size limit + reference-set pinning is the standard
  eviction policy. Working-set > cache = thrash.
- On the S3 / GCS side: distribute across prefixes and use
  multi-part GETs.
- The staging layer's per-shard first-fetch and serve times are
  first-class runbook metrics.
