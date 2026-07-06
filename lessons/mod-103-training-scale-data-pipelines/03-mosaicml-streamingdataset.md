# MosaicML StreamingDataset: Deterministic, Resumable, Shard-Aware Sampling for a 100 B-Token Corpus

WebDataset (chapter 2) solves the object-store I/O pattern for image
and multimodal training. It is fine for LLM pretraining too, but there
is a different toolchain that has become the default for that use
case: **MosaicML Streaming**. Streaming was designed around the LLM
pretraining problem statement — a corpus in the tens to hundreds of
billions of tokens, replayed for a fractional number of epochs, on a
fleet that must resume exactly where it stopped after any interruption
— and it embeds solutions to shuffle collapse, epoch drift, and
resume determinism directly in the format.

This chapter walks the **MDS** (Mosaic Data Shard) format, how the
`StreamingDataset` iterator preserves determinism across restarts,
and how you convert a raw pretokenized corpus (~100 B tokens) into it.

## Motivation: what LLM pretraining data looks like at scale

Concrete anchor numbers for the rest of the module:

- **Corpus** — 100 B tokens. At 2 bytes/token (a `uint16` vocab under
  ~65 K entries) this is ~200 GB on disk; at 4 bytes/token (`int32`
  with a larger vocab) this is ~400 GB.
- **Fleet** — hundreds to low thousands of GPUs (e.g., 512 or 1024
  H100s) for a modern 7 B–70 B pretraining run.
- **Epoch fraction** — most large runs consume 1–5 "epochs" of the
  full corpus. That is, the sampler must be able to declare
  `fractional_epoch=0.7` and honor it, because "run to convergence"
  is described in tokens seen, not in loops around the corpus.
- **Run duration** — days to weeks of wall clock, across dozens to
  hundreds of interruptions and restarts (SLURM preemption, node
  failure, deliberate rescale). Every restart must resume the sample
  stream exactly, not "close enough".

Any tooling that can't honor those numbers is out of scope for
pretraining.

## The MDS format

An MDS shard is a **binary column-oriented file** with an explicit
index header, chosen so:

- **Random access within a shard is cheap** — the header lets the
  reader seek to sample `k` without scanning the shard. This is the
  key difference from WebDataset's tar layout.
- **Sample size is known ahead of time** — the format stores the
  dtype and shape of each field, so the iterator can decode without
  a Python-side JSON parse per sample.
- **One shard has an integer sample count**, and the shard set for
  a stream has a total sample count that is small enough to fit in
  memory (typically millions of shards is the practical ceiling).

Alongside the shards there is an **`index.json`** that lists every
shard, its byte size, sample count, hash, and compression codec.
This is the manifest the rest of the pipeline (chapter 7) treats as
the source of truth.

The reader (`streaming.StreamingDataset`) treats a **stream** as
`(remote_url, local_cache_dir, index.json)`. Multiple streams can
be composed with per-stream weights so you can, say, hold a 60% code
/ 40% web mix without physically merging the corpora.

Reference: the `mosaicml-streaming` documentation covers the MDS
format spec, the writer API (`MDSWriter`), and the reader API
(`StreamingDataset`).
<!-- needs-research: cite the current https://docs.mosaicml.com/projects/streaming/ MDS format page and the mosaicml/streaming GitHub repo when authoring. -->

## The three guarantees Streaming makes

The value proposition of Streaming over "WebDataset for LLM" is not
throughput. WebDataset would move the bytes just fine. It is the
three guarantees below, each of which prevents a specific failure
mode chapter 6 catalogues.

### 1. Deterministic sampling from `(seed, world_size, num_canonical_nodes)`

The sample order across the full run is fully determined by:

- The `seed`.
- The **canonical world size** `num_canonical_nodes` — the
  reference fleet size the sample order was defined against.
- The actual world size at runtime.

Concretely, Streaming shuffles at the **shard** level first
(deterministically), then partitions shards across `world_size`
ranks (deterministically), then, inside each partition, applies a
per-worker in-memory shuffle. The `num_canonical_nodes` trick lets
you resume on a different fleet size than you trained on — as long
as `num_canonical_nodes` matches the original, the sample order is
preserved up to how it interleaves with the new rank layout.

That last property is what makes elastic training (mod-106) tractable.

### 2. Resumable state across restarts

`StreamingDataset` exposes `state_dict()` / `load_state_dict()` on the
iterator. The state is a small (bytes to low-KB) object holding the
epoch index, the sample index within the epoch, the shuffle epoch,
and the sample seen counters per shard. On resume, the iterator
reconstructs its position exactly.

The invariant is that the state is checkpointed **atomically with the
model** — at the moment `torch.save(model, opt, dataloader_state)`
snapshots, the next training step will consume exactly the sample that
was next in the iterator when the snapshot fired. Mod-106's
checkpointing chapter walks the coordination story; this chapter
provides the loader-side piece.

### 3. Shard-aware sampling with bounded working set

Streaming's shuffle is designed to run over a corpus that does not
fit in memory. Two configuration axes decide the trade:

- **`shuffle_algo`** — the shuffling algorithm (`py1s`, `py1b`,
  `py1br`, `py1e`, `py2s`, etc. — the names are historical). Each
  is a specific point on the "cost vs. quality vs. resume complexity"
  trade. `py1br` and later are the modern defaults: block-based
  shuffles that maintain resumability while keeping the shuffle
  buffer bounded.
- **`shuffle_block_size`** — the block size (in samples) that each
  worker holds in memory at once. Sample order within a block is
  free; sample order across blocks is fixed by shard order and shard
  shuffle.

Shuffle collapse (chapter 6) is what you get when
`shuffle_block_size` is set too small and adjacent training steps
see samples from the same shard, which are typically highly
correlated (same source document, same time slice, same crawl
partition). The fix is measured, not guessed: increase the block size
until the loss curve stops wobbling at shard boundaries.

## Converting a 100 B-token corpus with `MDSWriter`

The offline conversion job takes a raw pretokenized corpus (Parquet,
JSONL of token IDs, or the output of the Ray Data tokenization job in
chapter 4) and writes MDS shards. In its simplest form:

```python
from streaming import MDSWriter

columns = {"tokens": "ndarray:uint16"}
compression = "zstd:6"        # per-shard compression codec
hashes = ["sha1", "xxh64"]    # per-shard hashes recorded in index.json
size_limit = "512mb"          # target shard size

with MDSWriter(
    out="s3://corpus/mds/train/",
    columns=columns,
    compression=compression,
    hashes=hashes,
    size_limit=size_limit,
) as writer:
    for row in iter_pretokenized_rows():
        writer.write({"tokens": row["input_ids"]})
```

`MDSWriter` handles shard rollover at `size_limit`, computes hashes,
uploads to the remote URL, and appends to `index.json`. It is
single-process; distributed conversion is done by running one writer
per input shard (or per Ray Data output block) and merging the
`index.json` files at the end. The idiomatic pattern:

- Partition the input by a stable key (input shard index, Parquet
  row group, etc.) so each writer produces a distinct output subdir.
- After all writers finish, run `streaming.util.merge_index` to
  produce a single unified `index.json` at the root of the stream.

The merge step is the one bookkeeping trap: if you skip it, each
subdir looks like its own valid stream, but the reader sees N tiny
streams instead of one, and shard shuffling only ever mixes within a
subdir. Shuffle quality collapses; loss curve wobbles. Check that
your conversion job always ends with the merge.

## Reading with `StreamingDataset`

```python
from streaming import StreamingDataset

dataset = StreamingDataset(
    local="/mnt/local-nvme/mds-cache/train",   # per-node cache dir
    remote="s3://corpus/mds/train/",
    shuffle=True,
    shuffle_seed=1234,
    shuffle_algo="py1br",
    shuffle_block_size=1_000_000,              # samples
    num_canonical_nodes=64,                    # canonical fleet size
    predownload=8,                             # shards to prefetch per worker
    cache_limit="200gb",                       # per-node cache ceiling
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=micro_batch,
    num_workers=8,
    prefetch_factor=4,
    persistent_workers=True,
)
```

Two knobs that show up in the exercise:

- **`local`** — the on-node staging directory. Almost always
  local NVMe (fast, ephemeral) or a per-node Lustre mount, sized to
  hold the working shard set plus predownload. Chapter 5 covers the
  staging story in depth.
- **`predownload`** — how many shards ahead of the read cursor to
  fetch. This is Streaming's answer to WebDataset's prefetch: it
  hides the shard read behind current-shard consumption so an S3
  hiccup doesn't stall the GPU.

## Multi-stream (data-mix) composition

Modern pretraining runs mix multiple streams at controlled ratios
("40 % web crawl, 30 % code, 20 % books, 10 % Wikipedia"). Streaming
composes streams via `streaming.StreamingDataset`'s `streams` argument
(or by wrapping each `StreamingDataset` and using a proportional
sampler). Each stream has:

- Its own `remote`, `local`, `index.json`.
- Its own `proportion` or `repeat` (fractional or integer).

Coverage now applies per stream — every shard of each stream is
consumed the promised number of times per epoch. The mixture is a
deterministic function of the seed and the mix ratios; the runbook
in chapter 7 pins them both.

## Composing with mod-106's elastic training

Two things Streaming and the mod-106 elastic story need to agree on:

1. **Canonical world size** — set `num_canonical_nodes` at the start
   of the run and treat it as immutable. If you rescale from 32 to 64
   ranks midway, Streaming still shards against the canonical 32 and
   the sample order is preserved; new ranks pick up half-shards of
   the original partition.
2. **Loader state in the checkpoint** — `dataset.state_dict()` must
   be part of every checkpoint. mod-106 covers the coordination; this
   chapter is where the state lives. Skipping this line is the most
   common resume bug in the wild.

## When Streaming isn't the right choice

- **Multimodal image/video corpora** — WebDataset's tar format is
  more common there; the format is a wash for throughput, WebDataset
  has broader per-decoder support.
- **Purely local single-node training** — Streaming still works but
  MDS conversion is overhead you don't need if your corpus fits on
  local disk and won't be shared.
- **Non-token modalities you want to inspect with a text editor** —
  MDS is a binary format; JSONL is easier to grep.

For everything that looks like modern LLM pretraining, Streaming is
the default. The exercise makes you convert a corpus and prove
resumability.

## Summary

- MDS is a binary, indexed, shard-hashed format designed for
  determinism first and throughput second.
- `StreamingDataset` gives you deterministic sampling, resumable
  state, and shard-aware shuffle out of the box — the three
  guarantees that WebDataset does not.
- `num_canonical_nodes` decouples sample order from fleet size,
  which is what makes elastic training possible.
- `shuffle_block_size` is the knob that stands between you and
  shuffle collapse; the `merge_index` step is the one that stands
  between you and a broken shuffle.
- The stream's `index.json` — with its per-shard hashes — is the
  loader-facing half of the runbook chapter 7 makes canonical.
