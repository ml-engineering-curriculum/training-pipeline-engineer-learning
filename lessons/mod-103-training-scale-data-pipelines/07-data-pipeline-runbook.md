# The Data-Pipeline Runbook: Manifest, Hash Trail, Dedup Log, Throughput Baseline

Chapters 2–6 built the individual components. This chapter is about the
**operational contract** the training-pipeline engineer signs when they
ship a corpus to the training team. It is a short set of artifacts —
the runbook — that:

- Lets any engineer on the team run the corpus without you present.
- Lets the incident-response engineer, at 3 AM in the middle of a
  hardware failure, decide in five minutes whether the corpus is
  the suspect.
- Lets the platform's compliance / audit path answer "what data did
  we train on, and how do we know" without a two-week
  archaeological dig.

If the pipeline exists but the runbook doesn't, you have not shipped
a pipeline; you have shipped an incident waiting to happen.

## The four required artifacts

Every corpus this module ships with must have these four artifacts,
all versioned alongside the corpus itself in the same object-store
prefix or a linked one:

1. **Shard manifest** — the canonical list of every shard, its size,
   its hash, and its sample count.
2. **Hash trail** — the cryptographic chain from source-of-truth
   inputs to every derived shard, so a corpus's provenance is
   verifiable.
3. **Dedup log** — the record of what the offline job kept and what
   it dropped, and why.
4. **Per-epoch throughput baseline** — the reference throughput
   numbers a run should meet, with the units used to compare.

Each is small — a JSON file, a Parquet, at most a few MB — and each
prevents a specific class of incident.

## Artifact 1 — Shard manifest

The shard manifest is a machine-readable index of the corpus.

For MosaicML Streaming corpora, the manifest is largely already
there: `index.json` at the root of each stream includes every
shard's basename, byte range, sample count, and per-shard hashes.
For WebDataset corpora you write it yourself; JSON or Parquet, one
row per shard, with columns:

```json
{
  "shard_id": "train/aa/00000123.tar",
  "n_samples": 4096,
  "bytes": 524288000,
  "sha256": "1a2b3c...",
  "xxh64": "7e8f9a...",
  "written_at": "2026-06-14T18:22:04Z",
  "written_by": "mds_writer@1.7.3",
  "input_shards": ["raw/2024-11/00000042.parquet"],
  "input_row_range": [0, 4096],
  "tokenizer_sha256": "d4e5f6..."
}
```

`input_shards` and `tokenizer_sha256` are the fields that make the
manifest useful for incident response — they let you trace any
shard back to its input and its tokenizer version instantly.

Two invariants:

- **Every shard the loader sees at runtime must be present in the
  manifest.** No orphan shards. The loader's shard enumeration
  should be the manifest, not a `LIST` of the bucket prefix.
- **Every shard in the manifest must have a hash and a byte
  size.** No `null`s. If a shard exists but was not hashed, treat
  it as unshippable.

The manifest is what future-you diffs when someone asks "did we
change the corpus between run A and run B". A one-line `diff` on
the manifest either says yes and shows which shards, or says no
and closes the question.

## Artifact 2 — Hash trail

The hash trail is a **DAG of hashes** rooted in the input corpus
and terminating in each output shard. For a Ray Data
tokenization job with an `MDSWriter` back end, the DAG is:

```
input parquet (sha256 per file)
      │
      + tokenizer artifact sha256
      + dedup key algorithm version
      + sequence packing config
      │
      ▼
  intermediate parquet (sha256 per file, sha per row)
      │
      + MDSWriter version
      + compression codec + level
      │
      ▼
  output MDS shards (sha256 per shard, already in the manifest)
```

Concretely, a hash-trail record is a JSON that captures every
transform in the pipeline plus the inputs' and outputs' hashes:

```json
{
  "job_id": "tokenize-2026-06-14",
  "input_manifest_sha256": "aaa111...",
  "tokenizer_sha256": "d4e5f6...",
  "dedup_algo": "sha1_normalized_v1",
  "sequence_length": 2048,
  "packing": "eos_delimited",
  "code": {
    "repo": "corpus-tools",
    "commit": "abc123def",
  },
  "output_manifest_sha256": "bbb222...",
  "reproducibility": {
    "seed": 42,
    "ray_version": "2.11.0",
    "python_version": "3.11.7"
  }
}
```

Two properties matter:

- **A hash trail lets you *prove* a corpus is what the runbook
  says.** Recompute the input manifest sha; compare. Recompute the
  output manifest sha; compare. Both match → this corpus is the
  one the runbook describes. Any mismatch → someone or something
  edited state under you.
- **A hash trail lets you *reproduce* the corpus.** If the trail
  is well-typed you can re-run the exact same job on the exact
  same inputs and get the exact same outputs. If you can't
  reproduce, some transform is nondeterministic and you need to
  fix that before shipping.

This is the artifact compliance and audit teams care about most.
Without it, "what data did the model train on" has no formal
answer.

## Artifact 3 — Dedup log

Chapter 4 introduced dedup as a stage of the offline job. The
dedup log is the record of what it did.

For each input document, one row:

```
(input_shard_id, input_row_offset, dup_key, kept, reason)
```

`kept` is `true` for the surviving copy of a duplicate group,
`false` for the dropped copies; `reason` is a short enum
(`exact_dup`, `minhash_near_dup`, `retained_as_first`, ...).

The log lets you answer, without recomputing:

- **How many rows were dropped as duplicates?** Sum
  `kept == false`.
- **Was any specific source overrepresented in dropped rows?**
  Groupby `input_shard_id` → dropped fraction. A source whose
  dropped fraction is much higher than the corpus average is
  either heavily boilerplate (which you may or may not want to
  keep) or has a preprocessing bug.
- **Did any specific hash key have a suspicious duplicate
  count?** Extremely large duplicate groups often indicate a
  scraping error (e.g., the crawler hit a paginated page
  incorrectly and produced 10 000 copies of a boilerplate).

A summary row per corpus goes into the runbook itself
(`before_dedup: 12.8 B docs → after_dedup: 8.4 B docs, ratio
0.66`). The full log lives alongside the corpus for anyone who
needs to drill into a specific source.

## Artifact 4 — Per-epoch throughput baseline

The throughput baseline is the shortest of the four, but it is
the one on-call cites when the run "feels slow".

For a fixed reference training job (fixed model, fixed step
config, fixed fleet size), record:

- **Wall-clock per epoch** — measured over the first
  representative epoch of a golden run.
- **Steady-state step time** — measured across steps 500–1500
  (excluding warmup), median and p95.
- **Tokens per second per GPU** — derived from the above and the
  batch config.
- **Shard fetches per epoch** — from the loader's `shard.seen`
  counter.
- **Cache hit rate at steady state** — from the staging layer.
- **Loader worker CPU utilization** — from the training host's
  metrics.

Store this as a small JSON per corpus + fleet-shape combination:

```json
{
  "corpus_sha256": "bbb222...",
  "fleet": "8xH100_1node",
  "model": "llama3-7b-reference",
  "step_time_seconds": {"p50": 0.302, "p95": 0.318},
  "tokens_per_gpu_per_second": 43000,
  "shards_per_epoch": 400,
  "cache_hit_rate": 0.98,
  "loader_cpu_pct": 55
}
```

The baseline is what the on-call engineer diffs against when a
new run reports 0.42 s step time on 0.32 s reference:
`step_time p50` is up 40 %, everything else is nominal → the
corpus is fine, the training loop changed. Without a baseline
you cannot make that inference; you have to bisect from scratch.

## The loader-side asserts

The runbook artifacts are only useful if the pipeline actively
uses them. Three asserts every launch should run before the
first step:

1. **Corpus manifest hash matches expected.** Compute the
   manifest's sha256; compare to the value pinned in the training
   config.
2. **Tokenizer sha matches the corpus's `tokenizer_sha256`.**
   From the manifest.
3. **`num_canonical_nodes` matches the value in the corpus's
   throughput baseline** (if using MosaicML Streaming).

Any mismatch fails the launch loudly, with a specific error
message that points to which artifact disagreed. This is the
runbook's teeth: the corpus can only be run one way.

## The rescale story

When a run rescales (adds nodes, loses nodes) or resumes on a
different fleet size, the runbook is what tells you whether
that's safe:

- **`num_canonical_nodes` is pinned.** The Streaming state is
  portable. Safe.
- **Shard set is stable** (same manifest sha). Safe.
- **Tokenizer sha is stable.** Safe.
- **New `world_size` is not a divisor of `num_canonical_nodes`?**
  Streaming's sharding scheme allows this but the per-rank shard
  slice may be unbalanced. Note it in the runbook and check the
  first epoch's coverage explicitly.

The runbook is where the answers live so on-call doesn't have to
derive them from first principles at 3 AM.

## Handoff: what "done" looks like

You have shipped a data-pipeline runbook when a colleague who has
never seen your corpus can, from the runbook alone:

- Run the corpus in a training job on their fleet.
- Reproduce every artifact (manifest, hash trail, dedup log,
  baseline).
- Explain what would break if they changed the tokenizer, the
  seed, or the fleet size.
- Point on-call at a specific metric for each of the four
  failure modes in chapter 6.

Anything short of that and you are still the runbook. That is
not a place a training platform can scale from.

## Summary

- Four artifacts: shard manifest, hash trail, dedup log,
  throughput baseline. Each is small; each prevents a specific
  class of incident.
- The manifest is the canonical shard list with per-shard
  hashes; the hash trail chains transforms so provenance is
  auditable; the dedup log answers "what did the pipeline
  drop"; the baseline is what on-call diffs against.
- The loader asserts on the artifacts at launch, and refuses to
  start if any disagree.
- Runbook done ≠ "docs page written". Runbook done = "someone
  else can operate the pipeline without me".
