# exercise-03: Ray Data Tokenization and Dedup Over Parquet

**Estimated effort:** 4 hours

## Objective

Build the offline job that produced the corpus you converted in
exercise-02. Read raw Parquet documents, dedup them at global shard
boundaries, tokenize with a pinned tokenizer, pack into fixed-length
sequences, and emit Parquet suitable for MDS conversion. Prove that
the job is deterministic — running it twice on the same input
produces byte-identical output — and that the dedup log accounts
for every input row.

The deliverable is the Ray Data job code, the intermediate output,
a hash trail (chapter 7), and a short report on measured
throughput.

## Prerequisites

- Chapter 4 of this module.
- `ray[data]`, `pyarrow`, `zstandard`, and `tokenizers`
  (HuggingFace's Rust-backed library) installed.
- A local or remote Ray cluster you can run against — 1 head
  node is fine for the scale in this exercise; a real cluster is
  better if you have one.
- A pinned tokenizer artifact (`tokenizer.json` + config files) at
  a known path or URL.

## Problem statement

You have a raw corpus at `s3://your-bucket/exercise03/raw/`
consisting of Parquet files with a single `text: string` column.
Approximate scale: 500 M documents, ~50 GB of text after zstd
compression. This is a proxy for the module's 100 B-token target
— an order of magnitude smaller so you can run it locally in an
afternoon.

Produce a pretokenized, packed, deduplicated Parquet corpus
suitable for exercise-02's `MDSWriter`.

## Requirements

1. **Tokenizer pinning.**
   - Compute `sha256(tokenizer.json)` for your chosen tokenizer.
   - Vendor the artifact at a fixed URL or path and check the
     hash into the runbook.
   - Every Ray Data tokenizer actor asserts on the hash at
     init; a mismatch fails the job.
2. **Read + normalize.**
   - `ray.data.read_parquet(...)` the input.
   - Compute a `dup_key` per document: `sha1` of the
     Unicode-normalized, whitespace-collapsed text. Any
     documented normalization is fine as long as it is
     deterministic and versioned.
3. **Dedup at shard boundaries.**
   - Use `ds.groupby("dup_key").map_groups(...)` (or an
     equivalent Ray Data primitive) to keep the first
     occurrence of each key and drop the rest.
   - Emit a **dedup log**: one row per input row, columns
     `(input_shard_id, input_row_offset, dup_key, kept,
     reason)`. Write to
     `s3://your-bucket/exercise03/dedup_log/`.
4. **Tokenize.**
   - Actor pool of `TokenizerActor`s, sized to your cluster's
     CPU count.
   - Each actor asserts the tokenizer sha at init and processes
     rows in `map_batches` with `batch_size=64` or larger.
5. **Pack into fixed-length sequences.**
   - Concatenate tokenized documents into `L=2048`-token
     sequences separated by `<eos>`.
   - Emit one row per packed sequence with
     `input_ids: uint16[2048]`.
6. **Stable output ordering.**
   - Sort each output block by a stable input key
     (`input_shard_id`, `input_row_offset`) before writing.
   - Write to `s3://your-bucket/exercise03/pretokenized/`.
7. **Determinism proof.**
   - Run the job twice on the same input. Compute sha256 of the
     concatenation of output Parquet files' shas (order-stable).
     Both runs must produce the same top-level hash.
8. **Hash trail.**
   - Emit a `hash_trail.json` (chapter 7's schema) capturing
     input manifest sha, tokenizer sha, dedup algo version,
     packing config, output manifest sha, code commit, seed,
     and Ray version.

## Starter guidance

- Start with a **very small** slice of the input (a few hundred
  MB) while iterating on the pipeline. Ray Data's streaming
  execution makes it easy to accidentally submit a many-hour job
  before your pipeline is correct.
- The tokenizer actor's `__init__` is where you pay the setup
  cost; make sure `concurrency` is set to a range that reuses
  actors rather than respawning them per batch.
- `write_parquet` accepts `num_rows_per_file` and
  `sort_key`-style options depending on Ray version; check the
  docs for your pinned version.
- The dedup groupby is a shuffle. Expect it to be the slowest
  stage. Budget for it explicitly in your throughput report.
- For a first pass do **exact** dedup only. MinHash-LSH near-dup
  is a stretch goal.
- Instrument the job with wall-clock timers per stage; the
  deliverable's throughput numbers come from these.

## Acceptance criteria

- The tokenizer sha assertion is enforced in the actor and
  demonstrated to fail correctly if you swap in a different
  tokenizer.
- Two independent runs of the pipeline on the same input
  produce output whose top-level manifest sha256 is identical.
- The dedup log accounts for every input row exactly once. A
  small script asserts this: for every input `(shard, offset)`,
  exactly one row exists in the dedup log.
- The output Parquet files are sorted by
  `(input_shard_id, input_row_offset)` within each file.
- `hash_trail.json` includes every field listed in Requirement
  8, and its `input_manifest_sha256` and `output_manifest_sha256`
  match the actual inputs and outputs at the URLs cited.
- Throughput report includes: tokens tokenized per CPU-second
  (steady state), wall-clock per stage, and the total
  CPU-hours consumed. Numbers include the dedup shuffle.

## Stretch goals

- Add MinHash-LSH near-dup detection at a documented parameter
  set (bands, shingle size). Emit a second dedup pass log
  alongside the exact-dup log.
- Compose the job's output directly into `MDSWriter` calls (one
  writer per output block) instead of going through Parquet.
  Compare wall-clock and disk cost.
- Parametrize the packing scheme: also try `bos + eos +
  attention_mask` per document boundary and check that a
  reference reader honors it.
- Run the job on 10× the data with a Ray cluster and confirm the
  determinism proof still holds.
- Add a per-source-shard token-count summary to the runbook so
  future-you can spot corpus-drift over time.
