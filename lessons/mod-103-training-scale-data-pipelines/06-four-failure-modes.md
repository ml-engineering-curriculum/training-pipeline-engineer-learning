# The Four Canonical Data-Pipeline Failure Modes at Scale

Every previous chapter alluded to failure modes that this chapter now
catalogues in full. These four are the ones the training-pipeline
engineer is on-call for. Each has a **signature** you can spot in a
loss curve or a metric, a **root cause** that you can name in the
postmortem, and a **defense** you can bake into the pipeline before
the first incident.

The names are conventional in the community; the diagnosis protocols
are the point of this chapter.

## Failure mode 1 — Shuffle collapse

**Signature.** The loss curve develops a periodic wobble whose period
matches the shard consumption rate. On a smoothed loss chart it looks
like small sawtooth ripples aligned with `step % (samples_per_shard /
batch_size)`. On a per-shard token-distribution chart the histogram
per shard is visibly non-uniform (heavy skew toward one source,
one time slice, or one language).

**Root cause.** The **shuffle buffer / block is smaller than the
intra-shard correlation length**. A shard is not a random sample from
the corpus — it is a contiguous chunk of one input file, one crawl
partition, one document collection. Samples within a shard share
topic, style, language, or vocabulary distribution. If the loader's
shuffle window only reorders within a shard or across a few
shards, adjacent training batches see correlated data. The gradient
signal picks up that correlation as a periodic bias.

**Defenses.**

- **Increase the shuffle window.** For WebDataset, raise
  `shuffle(N)` until the wobble disappears. For MosaicML Streaming,
  raise `shuffle_block_size` until the wobble disappears. The right
  size is empirical — it depends on how "sticky" your corpus's
  intra-shard correlation is — but the diagnostic loop is the
  same: change the knob, retrain 500–1000 steps, look at the
  wobble.
- **Diverse shards at ingest.** During the Ray Data conversion
  (chapter 4), stripe input rows across output shards by a
  round-robin hash key so each output shard is already a mixture.
  This is cheap defense and independent of the loader knob.
- **Two-level shuffle.** WebDataset's `shardshuffle=True` + a
  large `shuffle(N)` is a two-level shuffle. Streaming's `py1br`
  and `py2s` algorithms are two-level shuffles. Both are strictly
  better than a single-level shuffle at the same buffer size.

**What "collapse" means.** In the pathological limit — `shuffle(1)`
or `shuffle_block_size = samples_per_shard` — you are training on
one shard at a time in shard order. That is *not* uniform sampling
from the corpus; it is a curriculum you did not design.

## Failure mode 2 — Epoch drift

**Signature.** The samples-seen counter (from your training
observability, mod-108) diverges from the theoretical
`step × global_batch`. Some ranks report fewer samples per epoch
than others. On a very long run the token count at "epoch 3"
disagrees with `3 × corpus_tokens` by a percent or more.

Loss curves usually **do not** show this cleanly — the divergence
is diffuse. You typically see it first in eval metrics that assume
a specific number of tokens seen, or in a reproducibility exercise
where you rerun the same job and get a different final loss.

**Root cause.** Any of a family of small bookkeeping mistakes that
add up to "coverage is not what you think":

- **`drop_last=True` on a sharded sampler** that drops the tail of
  each rank's shard slice — over many epochs, some samples get
  dropped systematically.
- **Silent worker restart** — a loader worker crashes on a
  corrupted sample, `DataLoader` respawns it starting from the
  next shard, and the samples between "last known good sample" and
  "next shard boundary" are lost.
- **`shardshuffle` racing across ranks** — the ranks disagree on
  the shard-order permutation because the seed depends on wall
  clock, not on epoch.
- **Sampler resume drift** — after a restart, the loader resumes
  by sample count but not by shard cursor, so the same shards get
  re-consumed and the tail of the previous epoch is skipped.
- **World size change without `num_canonical_nodes`** (MosaicML
  Streaming) — resuming on a different fleet size without pinning
  `num_canonical_nodes` reshuffles the sample-to-rank mapping and
  the resume state is meaningless.

**Defenses.**

- **Every checkpoint records the exact sample state.** For
  Streaming, this is `dataset.state_dict()`. For WebDataset,
  it's the shard cursor and per-worker sample index. For any
  other loader, the invariant is: on resume, the same next sample
  is served.
- **Pin `num_canonical_nodes`** at run start and treat it as
  immutable across the entire training run.
- **`drop_last=False`** unless the model architecture actually
  cannot handle a partial batch (rare — even then, drop on the
  training side, not the sampler side, so coverage is still
  visible).
- **Per-shard "seen" counter** — the runbook (chapter 7)
  requires a metric that ticks per shard as it is consumed. Sum
  it at epoch end and assert `== total_shards × completed_epochs`.
- **Corrupted-sample handling is opinionated.** The loader must
  either (a) raise and stop the run, so you can fix the shard, or
  (b) log the specific `(shard, offset)` and continue with the
  known-loss counted. Silently skipping and moving on is what
  causes epoch drift.

## Failure mode 3 — Tokenizer drift

**Signature.** A specific one, and the reason it deserves its own
category: **you restart a training run** (job died, resume from
checkpoint) and the loss abruptly jumps. It doesn't diverge
smoothly, it doesn't wobble — it steps. Often the step is small
(a few % on the loss), but it's clearly *not* natural noise. Some
runs manage to recover; some drift for hundreds of steps before
someone notices.

**Root cause.** The tokenizer on the resumed job is not the exact
tokenizer used on the pretokenized corpus. Common failure paths:

- **Loader-side tokenization** with `AutoTokenizer.from_pretrained("some-model")`
  and *no version pin*. The HuggingFace hub updated the tokenizer
  artifact upstream. Now `<|im_start|>` gets one token; last week
  it got three.
- **Different tokenizer artifact** in the resumed job's
  environment — a different `tokenizer.json` sha under
  `/opt/model/tokenizer/`.
- **The vocab size on the model side changed** — someone added
  special tokens, the embedding table grew, but the corpus was
  tokenized against the old vocab. Now `<pad>` and `<eos>` swap
  meaning on old rows.
- **Silent normalization change** — `nfd_normalization` on/off,
  `add_bos_token` on/off, `add_special_tokens` on/off. Each
  changes the byte-to-token boundary somewhere.

**Defenses.**

- **Tokenize once, offline.** Chapter 4's Ray Data job is the
  defense: produce pretokenized shards, and never call the
  tokenizer again on the hot path.
- **Pin the tokenizer artifact by sha** in the runbook and assert
  that the training environment matches the corpus's tokenizer
  sha at launch. If they disagree, refuse to start.
- **Never invoke `AutoTokenizer.from_pretrained("...")`** in a
  training script. Always pass an explicit local path to a
  vendored tokenizer artifact.
- **Vocab-size assertion.** At model init, assert
  `model.config.vocab_size == corpus.tokenizer.vocab_size`. This
  catches "someone added special tokens" instantly.
- **Sanity roundtrip.** On startup, tokenize a fixed 100-sample
  reference set with the current tokenizer and compare the
  resulting IDs to a checked-in golden file. Any mismatch fails
  the launch.

## Failure mode 4 — Silent shard corruption

**Signature.** A specific worker consistently stalls at a specific
shard offset. Sometimes the whole job hangs; more often, the
worker raises an internal `DecodeError`, `DataLoader` swallows it
into a warning, the worker skips ahead, and no user-facing metric
changes. Loss looks fine. Coverage silently degrades. The dedup
log looks fine. But your run is now training on 99.6% of its
intended corpus, and the missing 0.4% is a systematically
skewed slice (whatever the corrupted shards were about).

**Root cause.** Any of:

- **Truncated shard** — an upload to S3 failed partway; the
  key exists at some byte length less than the writer intended.
- **Bit rot** — media corruption during a copy step or during a
  long-lived cache lifetime.
- **Writer bug** — the shard was produced by a version of
  `MDSWriter` (or your custom shard builder) with a bug that
  corrupted a specific sample offset.
- **Compression codec mismatch** — the reader has an incompatible
  codec version installed and quietly returns garbage bytes.

**Defenses.**

- **Per-shard hash at write time.** `MDSWriter` records
  `sha1` and `xxh64` per shard in `index.json`. WebDataset needs
  you to compute this yourself — the shard manifest (chapter 7)
  is where you keep it.
- **Verify the hash on first fetch, always.** The staging layer
  (chapter 5) is the natural place. If the hash mismatches,
  refuse to serve the shard, refetch from tier 0, and alarm.
- **Byte-size sanity check** — the manifest records exact
  shard size; the cache manager compares.
- **Do not swallow decode errors.** The loader must fail the
  run on shard-level errors and let the operator make the
  call. The runbook's incident playbook covers "how to
  quarantine a shard" (chapter 7).
- **Hash the vocab and the special-tokens map.** Same principle
  applied to the tokenizer's on-disk artifact prevents "someone
  patched the tokenizer file in place" silently.

## The diagnostic protocol

When a training run misbehaves and the model / optimizer are your
current suspects, the data pipeline is your next suspect. The
five-minute protocol:

1. **Wobble?** Look at the loss on a per-step scale, not the
   smoothed macro chart. If there is a periodic wobble aligned
   with shard consumption, suspect shuffle collapse. Increase the
   shuffle buffer.
2. **Loss step at resume?** Suspect tokenizer drift. Diff the
   tokenizer sha at last-known-good against the current env.
3. **Coverage disagreement?** Sum the per-shard "seen" counter
   and compare to `epochs × total_shards`. Any drift → epoch
   drift.
4. **Worker consistently stalls at a specific shard?** Suspect
   silent shard corruption. Check the manifest hash against the
   live shard.
5. **None of the above?** Now suspect the training loop.

## Composing with mod-108 (observability)

Every failure mode above has a **metric** that can be alarmed
on. mod-108 covers the end-to-end training-observability story;
this chapter names the specific data-pipeline metrics that must
exist:

- `data.shard.seen{shard_id=...}` counter — one per shard per
  epoch; sum for coverage.
- `data.shard.hash_mismatch` counter — should always be 0.
- `data.tokenizer.sha` gauge — string label; alarm on change.
- `data.loader.stall_seconds` histogram — per worker, per
  fetch. p99 climbing → staging trouble.
- `data.shuffle.buffer_depth` gauge — samples in the shuffle
  buffer at each moment. Bottoming out → collapse.

None of these are expensive; all four failure modes above are
caught by them. Write them into the pipeline from day one, not
after the first incident.

## Summary

- Four failure modes: shuffle collapse, epoch drift, tokenizer
  drift, silent shard corruption.
- Each has a signature you can read from a chart or a metric;
  each has a defense you bake in ahead of the incident.
- The five-minute diagnostic protocol is: wobble → drift step →
  coverage → stalls → other.
- The runbook (chapter 7) turns these defenses into artifacts
  and metrics you can audit, not just habits.
