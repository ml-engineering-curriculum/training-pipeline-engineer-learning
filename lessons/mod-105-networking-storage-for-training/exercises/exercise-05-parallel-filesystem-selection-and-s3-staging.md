# exercise-05: Parallel Filesystem Selection and S3 Staging

**Estimated effort:** 3 hours

## Objective

Produce a design document that (a) selects a parallel filesystem for a
specified training platform and defends the choice against the two
serious alternatives, and (b) specifies an S3-staging pattern that
lets the platform's training runs read data whose ground truth lives
in object storage without the loader ever waiting on it. The
deliverable is the storage-architecture section of a training-platform
RFC: precise enough to hand to a storage vendor for a quote and to a
platform team for implementation.

Where exercise-04 was per-workload arithmetic, this is per-platform
architecture — one storage tier serves many workloads, and the choice
has to accommodate the range you expect to see.

## Prerequisites

- Chapter 7 of this module in depth. Chapter 6 for the GDS
  considerations that fold into the choice.
- Exercise 4 done (or at least a rough loader-side budget for the
  workloads the platform will serve). You need to know the order-of-
  magnitude bytes per step and per epoch for the workload mix.
- Chapter 5 of mod-103 (data pipeline design at scale) if you have
  it — otherwise chapter 7's summary of the staging pattern is
  enough.
- Access to (or vendor docs for) at least one candidate parallel
  filesystem. If you cannot bring one up, work the exercise as a
  paper design grounded in vendor documentation, clearly labelled.

## Problem statement

Your organization is standing up a new training pod (or refreshing an
existing one) and the storage tier is the open architecture question.
The platform will serve a mix of workloads over its lifetime, and the
storage tier has to be operable for years without a re-architecture.
The RFC needs your section: filesystem selection plus staging
pattern, with the arithmetic and the operational plan.

Pick one of the following platform contexts (or substitute your
team's actual one):

- **Platform P1: on-prem SuperPOD, 512 H100 GPUs, HPC-native ops
  team.** Datasets originate in on-prem object storage (Ceph
  RadosGW or similar), ~2 PB active working set, mixed text and
  vision-language workloads.
- **Platform P2: AWS-hosted training on EC2 P5, up to 1024 H100
  GPUs across capacity blocks.** Datasets in S3 (single region),
  ~5 PB active working set, mostly text pretraining with occasional
  multimodal.
- **Platform P3: on-prem, 128 H100 GPUs, small ops team, small-ish
  workloads.** Datasets originate on S3 (cold) and are worked
  through on the pod. ~500 TB active working set, dataset shape
  varies from packed shards to many-small-files.
- **Platform P4: cloud (Azure ND-series or GCP A3), 256 H100 GPUs,
  video pretraining dominant.** Per-sample bytes are large;
  loader-side is potentially critical-path.

Name the platform in the first line of your document. Every choice
downstream refers back to its constraints.

## Requirements

Deliver one document (`storage-architecture-<platform>.md`) with the
following sections.

### 1. Workload profile

Summarize the workload mix the platform will host, in the terms
chapter 7 uses:

- **Dataset shapes.** Packed shards (WebDataset / MDS), many-small-
  files, or a mix. The metadata story hinges on this.
- **Active working set size** (per epoch that must be readable at
  training-time bandwidth).
- **Cold set size** (ground truth, may exceed active working set by
  10–100×).
- **Read pattern.** Sequential shard-scan, random-access small file,
  or a mix.
- **Checkpoint cadence and total size.** For fault-tolerance
  planning; chapter 7 flags this as a distinct workload the FS must
  serve.
- **Concurrent-job count** the platform will typically see. This
  bounds metadata rate and per-client throughput requirements.

Cite the workload-arithmetic memo(s) from exercise-4 where they
apply.

### 2. Candidate matrix

Instantiate chapter 7's comparison matrix against your platform.
Rows are Lustre (self-managed or DDN), WEKA, and FSx for Lustre (for
AWS platforms) or your cloud's equivalent for GCP / Azure. Columns
are the criteria chapter 7 lists, with two additions specific to your
platform:

- The workload mix from section 1 that each candidate handles well
  and the mix each candidate handles poorly.
- The operational load your team can actually carry.

Do not just copy chapter 7's matrix — evaluate each candidate
*against your platform*. A cell that says "very high metadata rate"
in chapter 7's abstract matrix should say "handles our 10^6-worker
DataLoader baseline; measured at N ops/s in vendor doc <URL>" in your
concrete one.

### 3. Selection and forcing argument

State the chosen filesystem and write a 400–600 word forcing
argument. The argument must:

- Name the top two or three constraints from section 1 that drove
  the choice.
- Name the alternative you seriously considered and reject it in one
  paragraph with a specific reason (not "it costs more" — say what
  operational or performance property lost).
- Address the operational-load axis explicitly. Chapter 7 makes the
  point that Lustre is highest-ops-load, FSx is lowest, WEKA is in
  the middle. Which is your team ready for?
- Address the small-file / metadata story explicitly. Even if your
  workloads are packed-shard today, the platform will see small-file
  workloads at some point.
- Address the GDS story explicitly (does the chosen FS support it,
  will the platform enable it now or later — refer back to
  exercise-4's ship/defer recommendation).

### 4. Capacity and throughput sizing

Concrete numbers. For the chosen filesystem, size:

- **Total capacity** — active + hot cache + growth headroom.
- **Aggregate throughput target** across all clients at peak
  concurrent-job count. Justify with the per-node loader bandwidth
  from section 1 or exercise 4.
- **Metadata rate target** — ops/s at the peak-concurrent-DataLoader
  baseline.
- **Number of servers** (Lustre: MDS + OSS count; WEKA: server
  count; FSx: throughput-tier configuration).
- **Storage-fabric NIC count and generation** per server. Chapter 7
  and chapter 2 together: storage fabric must be sized so the FS
  can hit the throughput target without ever contending with the
  compute fabric.

For each number, one line of arithmetic. `~40 GB/s per node × 64
concurrent clients / 8 servers = ~320 GB/s per server`, for
instance. If the servers you can buy do not hit that number, that
is a finding — the platform will be storage-bound and the RFC must
say so.

### 5. Staging architecture

Design the object-store → parallel-FS → hot-tier pipeline for this
platform. Cover:

- **Which of the three chapter-7 movement modes** — pre-stage, lazy
  fetch on first read, coordinated prefetch — the platform uses,
  and in what blend. For a WEKA + native S3 tier or FSx with a
  Data Repository Association, name the vendor mechanism.
- **How the working set is refreshed.** New datasets or new epochs
  imply new object-store objects; how do they reach the parallel
  FS, and how quickly? What is the SLA the training team can
  expect?
- **How eviction works.** Every hot tier fills; how do stale
  objects get removed to make room? Automatic (LRU) or
  operator-driven?
- **Consistency and correctness.** If the object store is updated
  underneath the FS, when do training runs see the new bytes?
  Chapter 7 does not go deep on this; make an explicit choice and
  document it.
- **Cost model.** Per-GB and per-request costs on the object side;
  per-TB or per-throughput-tier costs on the FS side. The training
  team needs an estimate to plan.

If your platform uses AWS Mountpoint for S3 or Alluxio directly
(bypassing a parallel FS for some workloads), specify which
workloads, why, and what the loader-side performance implication is
for those workloads specifically.

### 6. Operational plan

The RFC has to end with a plan. Enumerate:

- **Who owns bring-up.** Named team, deliverable timeline.
- **Baselines.** Which of the chapter-7 "measure everything on
  bring-up" drills (sequential BW, random IOPS, metadata rate,
  cold-vs-warm, concurrent-client scaling, GDS end-to-end) will run
  on bring-up. When they will re-run.
- **Monitoring.** The three or four metrics that determine
  "storage tier is healthy": per-client throughput vs. baseline,
  metadata ops/s vs. cap, cache hit rate on the hot tier,
  first-touch latency for lazy-fetched objects.
- **Failure-mode budget.** Cross-reference chapter 8's failure
  modes for the storage-fabric-side ones (silent NIC drop on the
  storage fabric, checkpoint burst congestion). What is the on-call
  runbook entry?
- **Sunset criteria.** When would the platform re-open the FS
  choice? A capacity ceiling, a workload-shape change, a vendor
  end-of-life. Naming the criteria now prevents lock-in.

## Starter guidance

- Do not chase the "best" FS. Chase the one that fits *this*
  platform. A great FS on paper that your team cannot operate is a
  worse choice than an adequate one your team can.
- The staging pattern is a blend, not a mode. Almost every real
  platform pre-stages some critical objects, lazy-fetches the bulk,
  and prefetches ahead for the second-epoch-onward loader.
- FSx for Lustre with a Data Repository Association is the boring,
  well-supported answer on AWS for good reason. If you are on AWS
  and choosing against it, the RFC has to say specifically why.
- WEKA's single-tier NVMe design plus native S3 tiering is powerful
  when the ops team is small; the license is not free, so cost the
  license as part of section 4.
- Self-managed Lustre wins on flexibility and community; it loses
  on operational overhead. Do not underestimate the second — every
  Lustre-native shop has stories.
- Distributed-checkpoint writes will use the storage fabric.
  Chapter 6 warned about this; size for both loader reads and
  checkpoint bursts, not for the average of the two.
- The "consistency and correctness" section of the staging design
  is where subtle data-pipeline bugs hide (a shard updates in S3
  while an epoch is mid-flight, and half the ranks see the old
  bytes). Do not skip it.

## Acceptance criteria

- Section 1 has numeric workload inputs and cross-references the
  exercise-4 memos.
- Section 2 has an evaluated matrix, not a generic one. Every cell
  is grounded in the platform, with citations.
- Section 3 has an explicit filesystem choice, a named rejected
  alternative, and an operational-load statement.
- Section 4 has sized numbers with arithmetic for capacity,
  aggregate throughput, metadata rate, server counts, and storage-
  fabric NIC counts.
- Section 5 specifies the movement modes, refresh flow, eviction,
  consistency, and cost model. Vendor mechanisms are named when
  they replace custom staging code.
- Section 6 has owned bring-up, baselines, monitoring metrics,
  failure-mode runbook cross-refs, and sunset criteria.
- Every quantitative claim is derived (arithmetic shown) or cited
  (vendor doc URL and figure).
- Nothing is invented. Storage tiers you cannot benchmark are
  labelled datasheet-only.

## Stretch goals

- Extend the design to a second candidate: rewrite the RFC as if
  you had selected the losing alternative from section 3. What
  changes in sections 4–6? What operational costs get transferred?
  This is what a real RFC review will make you do; better to have
  done it once yourself first.
- Model a 3-year cost comparison: capex + license + operations for
  each of the top two candidates, integrated over the platform's
  expected lifetime. Do not pretend to more precision than your
  input numbers warrant, but the exercise clarifies which choice is
  economically dominant.
- Add a "multi-region" section: if the platform grows to two or
  more regions (or two or more pods in the same region), how does
  the storage tier extend? Does object storage become the
  cross-region ground truth with per-region FS caches, or does the
  FS replicate directly?
- Prototype the section-5 staging design against a small local
  setup: MinIO for the object store, a small Lustre or a
  MountpointS3 mount for the FS side, `torch.utils.data.DataLoader`
  as the client. Confirm the movement pattern works before
  committing at scale.
- Fold the exercise-4 loader budget for one of your platform's
  workloads into section 4 and show that the sized aggregate
  throughput satisfies it with margin. If the arithmetic does not
  close, iterate the FS choice or the platform-tier configuration
  until it does.
