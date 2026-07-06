# Parallel Filesystems for Training: Lustre, WEKA, FSx for Lustre, and S3 Staging

Chapter 6 gave you the GPU-side of the storage path — GDS, `cuFile`,
per-step budget. This chapter gives you the *filesystem* side. The
storage tier a training pod runs on is one of the two or three most
consequential platform-architecture choices you make; it is also one
of the hardest to change once training is running against it. This
chapter arms you to defend the choice.

The comparison here focuses on three parallel filesystems commonly
deployed in training clusters — DDN / open-source Lustre, WEKA, and
AWS FSx for Lustre — plus the S3-staging pattern that ties any
object-store-first workflow to any of them. Consult the vendor docs
below for the numbers and features that move between releases:

- **Lustre wiki.** https://wiki.lustre.org/
- **DDN AI400X and EXAScaler.** https://www.ddn.com/
- **WEKA documentation.** https://docs.weka.io/
- **Amazon FSx for Lustre.**
  https://docs.aws.amazon.com/fsx/latest/LustreGuide/
- **AWS Mountpoint for Amazon S3.**
  https://github.com/awslabs/mountpoint-s3

## The parallel-filesystem shape

Every parallel filesystem deployed for training has (at minimum) three
kinds of server:

- **Metadata server(s) (MDS).** Owns the namespace: directory
  hierarchy, file inode metadata, permissions. Every `open`,
  `readdir`, `stat` goes to the MDS. Metadata-heavy workloads
  (many-tiny-file datasets, WebDataset shard indexes) live and die
  by MDS performance.
- **Object storage server(s) (OSS).** Owns the file data. Files are
  striped across many OSS instances so a single large read can
  parallelize across the storage fabric.
- **Client.** Runs on the training host. Talks to MDS for namespace
  operations and to OSS for data operations. Both flows are RDMA on
  the storage fabric on a well-provisioned cluster.

Every filesystem in this chapter has some version of this
decomposition. The interesting differences are (a) how they handle
metadata scaling, (b) how they schedule the striping / caching, and
(c) how tightly they integrate with object storage.

## Lustre (open source and DDN EXAScaler)

Lustre is the reference-quality parallel filesystem for HPC and, by
inheritance, for on-prem training clusters. Its architecture:

- **One or more MDS + MDT (Metadata Target).** The MDS process runs on
  the MDS host; the MDT is the block device that stores the metadata.
  Modern Lustre supports Distributed Namespace (DNE) to shard the
  namespace across multiple MDTs, but the default single-MDS
  deployment is still common on smaller clusters.
- **Many OSS + OST (Object Storage Target).** OSSes are the horizontal
  scale-out layer. A file is striped across `stripe_count` OSTs; on a
  well-tuned cluster you can drive multi-GB/s per client just by
  reading a striped file.
- **Client kernel module.** `lustre` client that mounts the FS.

The knobs that matter to a training platform engineer:

- **`stripe_count` and `stripe_size`.** How aggressively files are
  striped. For large sequential shard reads (WebDataset tar shards,
  MDS parquet files), a stripe count of 4–8 and a stripe size of 1–4
  MiB is a common starting point. `lfs setstripe` sets these; verify
  with `lfs getstripe`. Consult the Lustre wiki's "Optimizing Lustre"
  section for the current guidance.
- **DoM (Data on MDT).** Small files (say < 64 KiB) can be stored
  directly on the MDT, avoiding OSS round-trips. Useful for small
  configuration / label files scattered through a dataset.
- **Client-side read-ahead.** Lustre clients read ahead speculatively;
  the parameters are tunable per mount. For training workloads doing
  streaming sequential reads, the defaults are usually adequate but
  worth verifying.

Lustre's strengths: mature, well-understood, ecosystem-native for
HPC. Its weaknesses: metadata is historically the bottleneck (though
DNE helps), and small-file workloads punish it. If your dataset is
"one giant packed shard per file" you are fine; if it is "millions of
tiny JPEGs" you are not, at least not without DoM or a shard-repack.

## WEKA

WEKA is a distinct, commercial parallel filesystem architected
specifically for high-concurrency low-latency workloads, including
AI/ML. Its differences from Lustre worth understanding:

- **Single-tier NVMe backend.** WEKA stores data on locally-attached
  NVMe drives, with distributed erasure coding for durability. No
  separate MDS/OSS distinction is exposed to the operator; all servers
  run all functions.
- **User-space client.** The WEKA client runs in user space (with a
  small kernel shim). This lets it bypass much of the kernel VFS
  overhead and drive very high per-client IOPS.
- **POSIX and object interfaces.** Same data accessible via POSIX
  filesystem mount and via S3 API. This matters for the staging
  pattern below.
- **GDS integration.** WEKA integrates with `nvidia-fs` for GDS
  support; check the current WEKA docs for supported versions.
- **Tiered storage.** WEKA supports transparent tiering of cold data
  to object storage (S3, Azure Blob, GCS). The hot tier is NVMe on
  the WEKA servers; cold data lives in object storage and is faulted
  in on access.

WEKA's strengths: extremely high metadata rates (well-suited to
many-file datasets), simpler operational model than Lustre's
MDS/OSS split, integrated object-store tiering. Its weaknesses:
proprietary and commercial, less HPC-community history.

## FSx for Lustre

Amazon FSx for Lustre is AWS's managed Lustre offering. Architecturally
it is Lustre; the operational surface is very different:

- **Provisioning is API-driven.** You declare a filesystem via
  CloudFormation / Terraform / the AWS console; AWS provisions the
  MDS/OSS servers, stripes, mount targets.
- **Throughput per TiB.** FSx exposes throughput as a per-TiB rate you
  choose (e.g., 125 / 250 / 500 / 1000 MB/s per TiB). Total throughput
  scales with capacity. Verify the current tier and rate options in
  the AWS docs (https://docs.aws.amazon.com/fsx/latest/LustreGuide/).
- **S3 linking (Data Repository Association).** An FSx filesystem can
  be linked to an S3 bucket so that files in S3 appear in the FSx
  namespace, are lazy-loaded on first read, and written back to S3
  either explicitly or automatically depending on the policy. This is
  the specifically-supported version of the staging pattern.
- **Persistent vs. scratch.** FSx offers persistent (durable) and
  scratch (short-lived, higher-throughput) filesystem types. Scratch
  is the standard choice for training runs where the dataset origin
  is S3.
- **RDMA and GDS.** As of recent generations FSx for Lustre supports
  Amazon EFA (the RoCEv2-adjacent RDMA-over-Ethernet) as a transport
  and GDS with supported EC2 instance types.
  <!-- needs-research: verify the current FSx-EFA and FSx-GDS support matrix in AWS docs when authoring. -->

FSx's strengths: managed, S3-integrated, elastic. Its weaknesses:
per-TiB throughput pricing means "cheap capacity with fast throughput"
is a hard combo, and you inherit the general Lustre small-file
behavior.

## The comparison matrix

The trade-off table you can defend in a design review:

| Criterion | Lustre (self-managed / DDN) | WEKA | FSx for Lustre |
|-----------|-----------------------------|------|----------------|
| Deployment model | On-prem, self-managed or vendor-managed | On-prem or cloud, self- or managed | AWS-managed |
| Metadata perf on small files | Baseline; DNE / DoM help | Very high | Baseline |
| Sequential large-file throughput | Very high per client with striping | Very high per client | Scales with per-TiB throughput tier |
| GDS support | Vendor-dependent; consult docs | Yes | On supported instance types |
| Object-store integration | External tools (HSM policies) | Native tiering | Native (Data Repository Association) |
| Operational complexity | High (own the MDS/OSS) | Medium (own the WEKA cluster) | Low (managed) |
| Multi-tenancy story | Basic ACLs | Rich, native | Per-filesystem isolation |
| Cost model | Capex + operations | License + operations | Per-GiB + per-throughput-tier |
| Ecosystem maturity in HPC | Highest | Growing | High through Lustre inheritance |
| Ecosystem in cloud | Requires DIY on IaaS | Available on all major clouds | AWS only |

## Choosing between them: honest heuristics

Not a decision tree — a set of overlapping considerations:

- **You are on AWS with S3-origin datasets → FSx for Lustre with a
  Data Repository Association.** The S3 linking is the specifically
  supported version of the staging pattern and it eliminates a whole
  class of glue you would otherwise write yourself.
- **You are on-prem with a large HPC team → self-managed Lustre with
  DDN hardware.** Follow the DGX SuperPOD Reference Architecture's
  storage recommendations.
- **You are on-prem with a small ops team and can afford the license
  → WEKA.** The single-tier NVMe design and simpler operator
  interface trade capex for opex.
- **Your dataset is many-small-files → WEKA (or Lustre + DoM + shard
  repack).** Do not just throw it at classic Lustre; you will spend
  the next quarter debugging metadata contention.
- **Your dataset is packed shards (WebDataset / MDS) → any of the
  three, choose on the axes above.** Shards make the small-file
  problem moot.

## The S3 staging pattern

Regardless of the parallel-filesystem choice, most modern training
datasets *originate* in an object store (S3, GCS, Azure Blob) because
that is where the data-preparation pipeline (mod-103 chapter 5) wrote
them. The pattern that hides object-store latency behind training:

### The three tiers

1. **Cold tier: object storage.** Ground truth for the dataset.
   High-latency (100s of ms per request), scale-out bandwidth, cheap
   per byte. Never on the training critical path.
2. **Warm tier: parallel filesystem.** Whatever you chose above.
   Sub-ms latency, tens of GB/s per client. On the training critical
   path.
3. **Hot tier: local NVMe or page cache.** The current epoch's
   working set. µs latency. Absorbs random-access reads and the
   loader's prefetch buffer.

The engineering question is: how does data flow from tier 1 → tier 2
→ tier 3 without the trainer ever waiting?

### Movement patterns

Three canonical patterns; you almost always deploy some blend of
them:

- **Pre-stage.** Before the run starts, copy the working set from
  object storage to the parallel filesystem. Simple; the whole set
  must fit and the copy takes wall-clock time up front.
- **Lazy fetch on first read.** The parallel filesystem (WEKA, FSx
  with DRA, or an Alluxio / Mountpoint layer) faults data in on the
  first client access. First-epoch reads are slow; later epochs are
  fast because the data is now hot.
- **Coordinated prefetch daemon.** A sidecar service reads N shards
  ahead of the training loop's cursor, warming the filesystem or a
  local cache. Requires the shard order to be predictable enough that
  the sidecar knows what to prefetch. This is the pattern mod-103
  chapter 5 goes deep on.

For a canonical multi-epoch pretraining run, the composite pattern is:

- Pre-stage a small "always-hot" subset (curated eval, high-value
  subshards).
- Lazy-fetch the bulk on first-epoch touch.
- Run a coordinated prefetch daemon for the second epoch onward so
  shards are warm on the parallel filesystem when the loader wants
  them.

### Mountpoint, Alluxio, and the FUSE alternatives

Sometimes you want to skip the parallel filesystem entirely for a
subset of the workload and just mount object storage directly. Two
production options:

- **AWS Mountpoint for Amazon S3.** A FUSE client for S3 with
  training-friendly defaults: high read parallelism, no local
  writeback cache, optimized for sequential large-object reads.
  Suitable for shard-based training reads where each shard is a
  large sequential object.
- **Alluxio.** A caching layer that presents object storage as a
  POSIX-style filesystem with a distributed cache tier. A
  productionized version of the coordinated-prefetch daemon
  pattern.

Neither replaces a parallel filesystem for the highest-throughput or
smallest-latency slice of the workload; both are good ways to keep
the loader running while a longer-term storage architecture is
planned.

## The metadata problem

One recurring failure mode is worth calling out before the exercise:

Small-file datasets — a million small JPEGs, a million small JSON
labels — punish metadata servers. Every `open`, `readdir`, `stat`
becomes a synchronous round trip. On a many-worker DataLoader
(mod-103 chapter 1), the aggregate metadata rate can saturate the
MDS long before the OSS is doing any work.

Two fixes:

- **Repack into shards.** Convert to WebDataset tars or MosaicML MDS
  shards so each file the FS sees is large and sequential. This is
  the mod-103 chapter-2 / chapter-3 argument, and it applies here
  from the storage side.
- **DoM or high-metadata-perf FS.** For workloads where shard
  repacking is not an option (interactive random access to specific
  files), pick a filesystem that handles metadata well or configure
  Lustre with DoM for the small-file subset.

## What to measure on a new storage tier

Every time you bring up a new storage tier, run this sequence:

1. **Per-client sequential read bandwidth.** Large sequential reads
   at various stripe / block sizes. `fio` on local NVMe, `ior` on
   Lustre, `weka benchmark` on WEKA. Compare against the nominal
   per-client wire limit.
2. **Per-client random read IOPS.** Small random reads at 4 KiB.
   Bounds how badly small-file workloads will perform.
3. **Metadata rate.** `mdtest` or the FS-vendor's metadata benchmark.
   Bounds the parallel-DataLoader worker count you can support.
4. **Cold-vs-warm read latency.** Time a first-touch read and a
   subsequent-read of the same object. The delta bounds first-epoch
   time.
5. **Concurrent-client scaling.** Repeat 1–3 with 2, 4, 8, 16, 32
   concurrent clients. Confirm aggregate scales roughly linearly to
   the fabric ceiling.
6. **GDS end-to-end.** `gdsio` from the `nvidia-fs` package validates
   the GPU-side path against the filesystem.

Save every one of those numbers into the same runbook you built in
chapter 5. Every regression on this tier compares against those
baselines.

## Summary

- Every parallel filesystem shares the MDS / OSS / client decomposition
  but they differ on metadata scaling, cache tiering, cloud
  integration, and operational model.
- Lustre (self-managed or DDN) is the HPC reference; WEKA is the
  high-metadata-rate NVMe-tier commercial option; FSx for Lustre is
  the managed AWS answer with native S3 linking.
- Choose based on team, cloud, dataset shape, and cost — not on
  spec-sheet bandwidth alone. The matrix in this chapter is the
  starting point.
- The S3 staging pattern has three tiers (object, parallel FS, hot
  local) and three movement modes (pre-stage, lazy fetch, coordinated
  prefetch). Almost every production training run uses a blend.
- Small-file datasets are a metadata problem, not a bandwidth
  problem. Repack into shards or pick a filesystem architected for
  the workload.
- Measure everything on bring-up. Exercise 5 walks the full
  selection-and-benchmark cycle.
