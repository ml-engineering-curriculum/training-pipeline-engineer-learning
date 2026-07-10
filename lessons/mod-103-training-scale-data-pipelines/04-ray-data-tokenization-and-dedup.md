# Ray Data: Distributed Tokenization and Dedup Over Parquet

Chapters 2 and 3 assumed the corpus was already tokenized. This chapter
is about the job that produces it: a **distributed tokenization
pipeline** that reads a raw text corpus in Parquet, applies a pinned
tokenizer, deduplicates near-duplicate documents, and writes shards
that a downstream `MDSWriter` (chapter 3) or WebDataset packer
(chapter 2) can consume. We use **Ray Data** as the execution
substrate; it is the modern choice for petabyte-scale batch pipelines
that need to run on the same GPU cluster as the training job.

## Motivation: why tokenize in a separate job at all

A pretraining corpus at 100 B tokens implies roughly 200–400 GB of
tokenized data on disk (chapter 3's arithmetic) coming from
substantially more raw text — a common ratio is 5–10× before
compression, so the input Parquet can easily be 1–2 TB. Two properties
force this to be an **offline** step:

1. **Tokenizer determinism.** On-the-fly tokenization in the loader
   worker means the tokenizer version and its byte-level
   normalization *must* match the exact vocab the model was trained
   with. In practice a HuggingFace tokenizer's behavior can shift
   subtly across releases (whitespace handling, regex changes,
   special-token defaults). If the tokenizer drifts mid-run, the
   model's `<vocab>` no longer means the same thing as it did at
   epoch 0 and the loss curve breaks in a way that is very hard to
   diagnose. Chapter 6 catalogues this failure mode. Tokenizing
   once, with a pinned version, and hashing the vocab file, is the
   defense.
2. **Cost.** Tokenization is CPU-bound (or, for BPE-heavy corpora,
   RegEx-bound). Doing it inside the training loader steals CPU from
   `.to(device)` and prefetch. Doing it offline, on a right-sized
   fleet of CPU workers, lets you scale each independently.

The rest of the chapter is about how to run that offline job at
100 B-token scale on a Ray cluster.

## Why Ray Data

Ray Data is a distributed dataset abstraction sitting on top of Ray
Core. Its differentiators for this specific job:

- **First-class Parquet + Arrow** reader (`ray.data.read_parquet`)
  that streams row-groups instead of materializing full files.
- **Actor-based UDFs** with configurable resource requirements
  (`num_cpus`, `num_gpus`, `concurrency`) — so a tokenizer actor
  that holds a large `tokenizers` object in memory is instantiated
  once per worker, not once per row.
- **Streaming execution** — Ray Data schedules the pipeline as a
  DAG that produces output blocks incrementally rather than
  materializing intermediates. This matters when the intermediate
  is 1 TB and you don't have room for it on disk.
- **Native S3 / GCS support** via `pyarrow.fs`, with the same
  read-write throughput characteristics as a raw `pyarrow` job.

Reference: the Ray Data user guide covers the Dataset API,
`map_batches`, and `groupby`.
<!-- needs-research: cite the current https://docs.ray.io/en/latest/data/data.html landing page when authoring. -->

Alternatives worth naming so you can defend the choice:

- **Spark** — mature, ubiquitous, no GPU story if you also want to
  fuse dedup with embedding-based similarity.
- **Beam / Dataflow** — same, plus a native windowing model you do
  not need here.
- **A hand-rolled `multiprocessing`** — fine at 1 B tokens, breaks
  at 100 B on any pipeline that needs cross-file dedup.

## The pipeline shape

The tokenization + dedup job is a DAG that looks like:

```
[read_parquet]  --row-groups-->  raw text rows
       │
       ▼
[normalize + hash]   sha1 of normalized text → duplicate key
       │
       ▼
[dedup by hash]  keep first occurrence per key
       │
       ▼
[tokenize (actor pool)]   text → List[int]
       │
       ▼
[pack into fixed-length sequences]   L=2048 or L=4096
       │
       ▼
[write parquet shards]   one Parquet file per output shard
```

In Ray Data pseudocode:

```python
import ray
from ray.data import read_parquet

ds = read_parquet("s3://corpus/raw/*.parquet", override_num_blocks=8192)

ds = ds.map_batches(normalize_and_hash, batch_size=1024,
                    num_cpus=1)

ds = ds.groupby("dup_key").map_groups(dedupe_keep_first,
                                      num_cpus=1)

ds = ds.map_batches(
    TokenizerActor,
    fn_constructor_args=(tokenizer_path,),
    concurrency=(64, 256),           # (min, max) tokenizer actors
    batch_size=64,
    num_cpus=2,
)

ds = ds.map_batches(pack_sequences, batch_size=256, num_cpus=1)

ds.write_parquet("s3://corpus/pretokenized/",
                 num_rows_per_file=2**16)
```

Each `map_batches` stage runs across the cluster; Ray Data handles
work distribution, block coalescing, and back-pressure.

## Tokenizer pinning

The tokenizer object must be **hash-pinned** across every worker.
Idiomatic pattern:

- Store the tokenizer artifact (`tokenizer.json`,
  `special_tokens_map.json`, and `tokenizer_config.json` — everything
  HF's `AutoTokenizer.from_pretrained` needs) at a fixed URL:
  `s3://artifacts/tokenizers/<sha>/`.
- Compute `sha = sha256(tokenizer.json)` once and record it in the
  runbook (chapter 7).
- The tokenizer actor loads from that URL at startup and asserts on
  the hash before processing any row.

```python
class TokenizerActor:
    def __init__(self, tokenizer_path: str, expected_sha: str):
        vocab = load_from(tokenizer_path)
        actual = sha256(vocab.serialize())
        if actual != expected_sha:
            raise RuntimeError(f"tokenizer drift: {actual} != {expected_sha}")
        self.tok = HFTokenizer(vocab)

    def __call__(self, batch):
        return {"input_ids": self.tok.encode_batch(batch["text"])}
```

The sha in the job's arguments is what future-you cites in the
incident postmortem when a colleague asks "did we ever change the
tokenizer".

## Deduplication at shard boundaries

The specific dedup problem this chapter cares about is **exact and
near-exact duplicate documents**. The rationale (from the Gopher and
Llama-family dataset papers) is that memorized-verbatim duplicates
in the training data inflate reported metrics and can leak eval
data. There are heavier dedup techniques — MinHash / LSH near-dup,
substring-level dedup à la CCNet, semantic-embedding-space dedup —
that are worth understanding but out of scope for this chapter; a
useful survey lives in Lee et al., "Deduplicating Training Data
Makes Language Models Better" (2022).
<!-- needs-research: cite Lee et al. 2022 and the CCNet / Common Crawl dedup pipeline when authoring. -->

The specific challenge Ray Data adds: **duplicate detection must
survive shard boundaries**. If shard 42 and shard 87 each contain
the same document independently, the naive per-shard dedup misses
the collision.

Two common patterns:

### Global hash-set dedup

Assign each document a canonical hash key `k`:
```
k = sha1(unicode_normalize(strip_boilerplate(text)))
```
Then use `ds.groupby("dup_key")` to bring all rows sharing a key
into one worker and keep only the first. Ray Data implements groupby
as a **shuffle**, which is expensive at TB scale but bounded (the
cost is proportional to corpus size, not to corpus-squared). Budget
for it in the job's step time.

### MinHash-LSH near-dup (stretch)

For "same document with minor formatting differences" you need
MinHash bands + LSH bucketing. Ray Data can host the MinHash actor
identically to the tokenizer actor; the bucketing pass is another
groupby. The Llama 2 and Falcon paper appendices describe the
canonical parameters (typically 5–13 hash bands over shingled
n-grams).<!-- needs-research: verify the actual bands / shingle sizes cited by Touvron et al. 2023 (Llama 2) and Almazrouei et al. 2023 (Falcon) when authoring. -->

**The dedup log.** However you dedup, the job must emit a per-shard
log of `(input_shard_id, input_row_offset, dup_key, kept_or_dropped)`.
That log is a mandatory artifact in the runbook (chapter 7) and the
evidence you cite when someone asks "why does the corpus have 8.4 B
rows instead of 9.1 B".

## Ordering, packing, and shard determinism

Two conventions to fix in the job's spec:

1. **Sequence packing.** LLM training almost always wants
   fixed-length sequences (2 K or 4 K tokens). Packing concatenates
   multiple tokenized documents into each `L`-token sequence with an
   `<eos>` between them, and (optionally) an `attention_mask` that
   marks the document boundaries so cross-document attention is
   masked out. Do this in the offline job; do not do it in the
   loader.
2. **Output ordering.** For downstream determinism you want the
   output shard order to be a stable function of the input, not
   dependent on Ray's block scheduling. Sort each output block by
   its stable input key (input shard id + row offset) before
   `write_parquet`. This is the shard shuffle's "reference order" —
   `StreamingDataset`'s `shuffle_seed` will shuffle on top of it,
   but the underlying order must be reproducible.

## Sizing the Ray job

Rough sizing template for a 100 B-token corpus:

- **Tokenizer throughput** — a fast Rust-backed tokenizer
  (HuggingFace's `tokenizers` crate) processes on the order of a few
  hundred thousand to a low-million tokens per CPU-second, depending
  on vocab and normalization. Concrete number: expect ~1 s of CPU
  per 1 M tokens as a starting estimate, and *measure it* on your
  own tokenizer before extrapolating.
- **Total CPU-hours** — `1e11 tokens / 1e6 tokens per CPU-second
  ≈ 1e5 CPU-seconds ≈ 28 CPU-hours`. On a 64-core CPU node that's
  half an hour of pure tokenize time.
- **Add the dedup shuffle** — often 2–5× the tokenize cost at
  scale.
- **Add the read + write** — TB-scale reads from object storage
  typically bottleneck on egress rate, not CPU. Provision multiple
  worker nodes with independent network paths.

The exercise makes you measure and defend these numbers instead of
inheriting mine.

## Composing with training (chapter 3)

The job's output shards feed directly into `MDSWriter`. Idiomatic
pattern is a two-step conversion:

```
Ray Data job → s3://corpus/pretokenized/*.parquet
                    │
                    ▼
     MDSWriter conversion job (per input shard)
                    │
                    ▼
        s3://corpus/mds/train/*.mds + index.json
```

Each Ray Data output Parquet becomes one MDS output subdir; the
subdirs are merged with `streaming.util.merge_index` at the end,
per chapter 3.

If you prefer to skip Parquet entirely you can write MDS directly
from the Ray job's final stage. The exercise pins the two-step
pattern because it makes the intermediate inspectable — you can
open the Parquet in pandas, sanity-check the token distribution,
and re-run just the MDS step without re-tokenizing.

## Summary

- Tokenization is an offline job because determinism and cost
  demand it.
- Ray Data is the modern substrate: Parquet in, actor-pool
  tokenizers, native shuffle for dedup, output blocks that stream
  into `MDSWriter`.
- Pin the tokenizer by sha and assert on it in every worker; drift
  is one of the four failure modes in chapter 6.
- Dedup must be global, not per-shard; the dedup log is a
  mandatory runbook artifact.
- The output shard order must be a stable function of the input;
  the shuffle happens at training time, not here.
