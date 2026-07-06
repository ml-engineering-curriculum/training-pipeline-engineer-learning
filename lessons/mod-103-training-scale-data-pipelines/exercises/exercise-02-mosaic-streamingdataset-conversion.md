# exercise-02: MosaicML StreamingDataset Conversion with Deterministic Resume

**Estimated effort:** 4 hours

## Objective

Convert a pretokenized text corpus into a MosaicML `StreamingDataset`
(MDS format) and prove — end to end — that the reader satisfies the
three guarantees from chapter 3: deterministic sampling, resumable
state across restarts, and a bounded shuffle working set. The
deliverable is a corpus, a runbook stub, and a resume test that
demonstrates bit-identical sample sequences before and after a
simulated restart.

## Prerequisites

- Chapter 3 of this module (chapter 1 for the invariants).
- `mosaicml-streaming` installed. Version pinned in your
  `requirements.txt`.
- Access to an object-store bucket. The exercise runs at a scale
  that easily fits in low-cost storage; a few tens of GB.
- A single-node dev environment with 2–8 GPUs (only used to run
  the reader; no training loop required).

## Problem statement

You have a pretokenized text corpus at
`s3://your-bucket/exercise02/parquet/` — either produced by
exercise-03 (Ray Data tokenization) or by any pretokenizing script
you write. Total corpus size ≈ 1 B tokens (a scaled-down proxy for
the chapter's 100 B; the pattern is identical). Column layout: one
row per document with an `input_ids: List[int]` field (`uint16`).

Convert it to an MDS stream that a training platform can consume,
then prove the reader is deterministic and resumable on your fleet.

## Requirements

1. **Write the conversion job.**
   - Read the source Parquet with `pyarrow` or `pandas`.
   - Emit `int_uint16_array` (or `ndarray:uint16`) rows through
     `MDSWriter`.
   - `size_limit = "256mb"`, `compression = "zstd:6"`,
     `hashes = ["sha1", "xxh64"]`.
   - Run distributed if the input has more than a few files. One
     writer per input Parquet, each writing to a distinct output
     subdir.
   - After all writers finish, run `streaming.util.merge_index`
     to produce a single unified `index.json` at the stream root.
2. **Build the reader.**
   - `StreamingDataset(remote=..., local="./mds-cache",
     shuffle=True, shuffle_seed=1234, shuffle_algo="py1br",
     shuffle_block_size=<value you defend>,
     num_canonical_nodes=<value you defend>, predownload=8,
     cache_limit="10gb")`.
   - Wrap in a `torch.utils.data.DataLoader` with
     `num_workers=4`, `persistent_workers=True`,
     `batch_size=32`.
3. **Prove determinism.**
   - Run the reader for 500 batches, hash each batch's
     `input_ids`, log the sequence of hashes.
   - Restart the process (fresh Python interpreter, same seed
     config) and confirm the first 500 batches produce the same
     hash sequence.
4. **Prove resumability.**
   - Run the reader for 250 batches, snapshot
     `dataset.state_dict()`, and record the last batch's hash.
   - Kill the process. In a fresh process, `load_state_dict()`
     the snapshot and confirm the next batch matches the batch
     251's hash from the original run.
5. **Prove elastic-safe rescale.**
   - Repeat the resume test but with a different world size
     between the two runs (e.g., snapshot on 4 workers, resume
     on 8, or vice versa). With `num_canonical_nodes` pinned,
     the sample sequence should be a stable function of
     canonical rank layout — document what stays identical and
     what merely stays *coverage-preserving*.
6. **Runbook stub.** Ship a `README.md` under the stream that
   captures:
   - Stream URL and manifest hash (chapter 7).
   - Tokenizer sha256 the corpus was tokenized with.
   - `num_canonical_nodes`, `shuffle_algo`,
     `shuffle_block_size`, and `shuffle_seed`.
   - Total sample count and expected samples-per-epoch.

## Starter guidance

- The `streaming.util.merge_index` call is the single most
  commonly forgotten step. Add a test that fails if the root
  `index.json` shard count disagrees with the sum of subdirs'
  shard counts.
- Pick `shuffle_block_size` from your corpus: it should be
  large enough that many shards' worth of samples are in the
  shuffle at once. Start at 1 M samples and defend your choice.
- Pick `num_canonical_nodes` as a power of 2 that matches the
  fleet size class you plan to train on (32 or 64 for a
  1024-GPU-class run; 4 or 8 for this exercise). Pin it before
  you write any MDS data.
- Use `hashlib.sha256` on the concatenated bytes of each
  batch's `input_ids` as the batch hash. Any collision-resistant
  hash works.
- Do **not** persist the cache directory between runs unless
  you also want to test cold-start behaviour separately.
- If you're on a shared cluster, put `local` under a per-run
  directory so parallel exercises don't share cache state.

## Acceptance criteria

- The MDS stream at your S3 URL has a single root `index.json`
  whose shard count equals the sum of the writer-subdir shard
  counts. Confirmed by a script, not a screenshot.
- The determinism test runs twice, in fresh Python processes,
  and produces bit-identical batch hash sequences over 500
  batches.
- The resumability test hands off through `state_dict()` /
  `load_state_dict()` and produces a batch-hash sequence that
  matches the reference for at least 250 continuation batches.
- The rescale test documents which invariants hold across the
  world-size change and which do not.
- The runbook stub captures every field listed in Requirement 6
  and is versioned alongside the stream (same prefix or a
  linked one).
- The `mosaicml-streaming` version is pinned in the
  requirements file.

## Stretch goals

- Compose two streams at a fixed proportion (say, 70 / 30) using
  `StreamingDataset(streams=...)` and verify the mixture ratio
  holds over 10 000 batches.
- Delete a single MDS shard from the stream and observe how the
  reader fails; document the failure mode and what a good
  detection metric would be (relates to chapter 6's silent
  shard corruption).
- Run the determinism test with `shuffle_algo="py1s"` (an older
  algorithm) and `py1br` and compare the resume behavior;
  document the difference.
- Toggle `predownload` between 1 and 16 and measure the
  cold-start time-to-first-batch and steady-state cache miss
  rate.
