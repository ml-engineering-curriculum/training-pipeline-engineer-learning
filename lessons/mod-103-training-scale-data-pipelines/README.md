# mod-103 — Training-Scale Data Pipelines: WebDataset, MosaicML Streaming, Ray Data, and Tokenization at Scale

**Estimated effort:** 20 hours

mod-101 and mod-102 taught the algorithms and the frameworks that turn
GPUs into a training system. This module teaches the subsystem that
keeps that system *fed*: a data pipeline that must sustain a byte-rate
matched to the accelerator fleet, for the weeks a modern pretraining
run lasts, across restarts, without ever silently changing the sample
stream.

By the end of the module you should be able to (a) design a shard
layout and loader configuration that keeps an 8-GPU node saturated
without I/O stalls, (b) convert a raw 100 B-token corpus into a
resumable, deterministic streaming dataset that survives elastic
rescale, (c) run the offline tokenization + dedup job that produced
that corpus on Ray Data, (d) design the S3-to-Lustre staging layer
between them, (e) recognize and defend against the four canonical
data-pipeline failure modes, and (f) ship a runbook a colleague can
operate without your presence.

## Learning objectives

- Build a WebDataset tar-shard pipeline that saturates 8 GPUs'
  compute budget without I/O stalls.
- Convert a raw corpus (~100 B tokens) into a MosaicML
  StreamingDataset with resumable, deterministic, shard-aware
  sampling.
- Run a distributed tokenization job on Ray Data over Parquet
  input with dedup at shard boundaries.
- Design a staging layer that pre-fetches from S3 / GCS to Lustre /
  WEKA with I/O budgets matched to per-step compute.
- Diagnose the four canonical data-pipeline failure modes at
  scale: shuffle collapse, epoch drift, tokenizer drift, silent
  shard corruption.
- Ship a data-pipeline runbook (shard manifest, hash trail, dedup
  log, per-epoch throughput baseline).

## Chapters

1. [The Training-Data Mental Model: Feed Rate, Step Time, and the
   I/O Budget](01-training-data-pipeline-mental-model.md) — the
   loader as a compute-competing subsystem; the two invariants
   (determinism, coverage); the pipeline as a stack of queues; why
   a naive `DataLoader` fails at scale.
2. [WebDataset: A Tar-Shard Pipeline That Saturates 8 GPUs](02-webdataset-tar-shard-pipeline.md) —
   the tar-shard format, shard-size arithmetic,
   `split_by_node`/`split_by_worker`, prefetch and shuffle depth,
   and the two knobs that stand between you and I/O stalls.
3. [MosaicML StreamingDataset: Deterministic, Resumable,
   Shard-Aware Sampling for a 100 B-Token Corpus](03-mosaicml-streamingdataset.md) —
   the MDS format, `num_canonical_nodes`, `shuffle_block_size`,
   `state_dict()` for resume, multi-stream mixes, and
   `merge_index`.
4. [Ray Data: Distributed Tokenization and Dedup Over Parquet](04-ray-data-tokenization-and-dedup.md) —
   the offline tokenization job, actor-pool tokenizers, tokenizer
   hash pinning, global dedup at shard boundaries, sequence
   packing.
5. [The Staging Layer: S3 / GCS → Lustre / WEKA, on a
   Compute-Matched I/O Budget](05-staging-layer-s3-to-lustre.md) —
   the three-tier storage hierarchy, bulk pre-stage vs. rolling
   cache vs. coordinated daemon, LRU + explicit size limits,
   object-store prefix and multi-part patterns.
6. [The Four Canonical Data-Pipeline Failure Modes at Scale](06-four-failure-modes.md) —
   shuffle collapse, epoch drift, tokenizer drift, silent shard
   corruption. For each: signature, root cause, defense, metric to
   alarm on.
7. [The Data-Pipeline Runbook: Manifest, Hash Trail, Dedup Log,
   Throughput Baseline](07-data-pipeline-runbook.md) — the four
   artifacts every corpus must ship with, the loader asserts that
   consume them, and what "handoff done" actually means.

## Exercises

- [exercise-01 — WebDataset tar-shard pipeline](exercises/exercise-01-webdataset-tar-shard-pipeline.md) (4 h)
- [exercise-02 — MosaicML StreamingDataset conversion](exercises/exercise-02-mosaic-streamingdataset-conversion.md) (4 h)
- [exercise-03 — Ray Data tokenization and dedup](exercises/exercise-03-ray-data-tokenization-and-dedup.md) (4 h)
- [exercise-04 — Staging layer S3 → Lustre](exercises/exercise-04-staging-layer-s3-to-lustre.md) (4 h)
- [exercise-05 — Data-pipeline failure-mode drills](exercises/exercise-05-data-pipeline-failure-mode-drills.md) (4 h)

## Labs and quizzes

- `labs/` — a long-form lab that stitches the exercise output
  into a "one corpus, three loaders" bake-off will land here on a
  subsequent authoring cycle.
- `quizzes/` — one knowledge check will land here on a
  subsequent authoring cycle.

## Resources

- [resources.md](resources.md) — primary papers, library
  documentation, and reference codebases (WebDataset, mosaicml-streaming,
  Ray Data, mosaicml/composer, torchtitan) that the chapters and
  exercises are grounded in.

## How the module fits together

Chapter 1 defines the byte-rate contract and the two
non-negotiable invariants (determinism and coverage) every
subsequent chapter is evaluated against. Chapters 2 and 3 build the
two dominant loader toolchains — WebDataset for image/multimodal
and MosaicML Streaming for LLM pretraining — because you should
understand the *format* before you understand the job that
produced it. Chapter 4 then walks the Ray Data offline job that
produces the shards those loaders consume. Chapter 5 pulls back
and asks where those shards live between the cold object store
and the hot loader, and how to size the staging layer to the
per-step compute budget. Chapter 6 catalogues the four failure
modes that any of the previous four chapters can produce when
misconfigured. Chapter 7 turns everything into a runbook a
colleague can operate.

Exercise 1 is the anchor — an 8-GPU WebDataset saturation
experiment where you measure step time, worker count, and
prefetch depth. Exercise 2 converts a real corpus to MDS and
proves resumability. Exercise 3 builds the Ray Data job that
produced it. Exercise 4 designs and measures the staging layer.
Exercise 5 is the failure-drill: deliberately induce each of the
four failure modes and detect them from your metrics. Do them in
order; each reuses artifacts from the previous.

## What this module deliberately does not cover

- **The training loop itself** — DDP, FSDP2, TP/PP composition —
  owned by mod-101 and mod-102. This module assumes you can
  already run a distributed training step; it feeds it.
- **Cluster orchestration (SLURM, Kueue, Volcano, KubeRay)** —
  owned by mod-104. This module runs on top of whatever scheduler
  the cluster gives you.
- **NCCL and network-fabric tuning** — owned by mod-105. Chapter 5
  name-checks the storage-side network story; mod-105 owns the
  compute-side.
- **Distributed checkpointing and elastic training** — owned by
  mod-106. Chapter 3 hooks the loader-side state into that story
  but does not define it.
- **MFU engineering** — owned by mod-107.
- **Training observability, W&B alerting, dashboarding** — owned
  by mod-108. Chapter 6 names the metrics; mod-108 owns the
  pipeline that ships them.
- **Long-term corpus governance, licensing, and legal** — out of
  the training-pipeline engineer's scope; noted in chapter 7 as a
  handoff to the data governance function.
