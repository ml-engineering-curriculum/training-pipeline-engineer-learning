# exercise-05: Data-Pipeline Failure-Mode Drills

**Estimated effort:** 4 hours

## Objective

Deliberately induce each of the four canonical data-pipeline failure
modes from chapter 6 — shuffle collapse, epoch drift, tokenizer
drift, silent shard corruption — on a running training-style
pipeline, and detect each one from the metrics you would ship in
production. The point is muscle memory: by the time this exercise is
done you should be able to look at a chart from a real incident and
call the failure mode in under a minute.

The deliverable is four short drill reports (one per failure mode)
plus a consolidated "5-minute diagnostic protocol" runbook page.

## Prerequisites

- Chapters 6 and 7 of this module.
- The MDS stream from exercise-02 (or an equivalent WebDataset
  corpus). You'll intentionally corrupt copies of it during the
  drills; work off a duplicate prefix so you don't wreck the
  original.
- The runbook artifacts from exercise-02 and exercise-03
  (manifest, hash trail, tokenizer sha).
- Ability to run a synthetic training loop that computes a loss
  proxy per batch (a deterministic scalar function of
  `input_ids` — mean of ids, sum of a hashed subset, whatever
  reads clean on a chart).

## Problem statement

Your on-call rotation just started. Overnight there will be four
incident scenarios injected into your training runs. For each,
you have to (a) recognize the failure mode from metrics alone,
(b) confirm it with a targeted diagnostic, (c) file a short
postmortem citing the root cause, and (d) note the defense that
would have prevented it.

You'll build the drills yourself, then run them as if they were
adversarial.

## Requirements

Set up a **synthetic training loop** that reads batches from
`StreamingDataset`, computes a deterministic per-batch scalar
("loss proxy") from the tokens, and logs:

- `loss_proxy` per step.
- `step_time` per step.
- `shard_seen{shard_id}` counter per shard.
- `tokenizer_sha` (as a label / gauge).
- `worker_stall_seconds` histogram.

Then run each of the four drills:

### Drill 1 — Shuffle collapse

- **Induce it.** Rerun the loader with `shuffle_block_size` set
  to a value much smaller than the samples-per-shard (e.g., 10 %
  of it). Keep everything else identical to the baseline.
- **Detect it.** From the `loss_proxy` chart, identify the
  periodic wobble. Confirm its period matches
  `samples_per_shard / batch_size`.
- **Prove it.** Per-shard token-distribution histogram before
  and after the fix; the "before" should show clear per-shard
  skew feeding into consecutive batches.
- **Fix.** Restore `shuffle_block_size` to a value that
  eliminates the wobble and defend the new value.

### Drill 2 — Epoch drift

- **Induce it.** Pick one of these and document your choice:
  - Set `drop_last=True` in the sampler.
  - Simulate a worker crash mid-shard by killing one worker
    every ~50 shards; loader respawns skip to the next shard
    boundary.
  - Set a fresh `shuffle_seed` (or don't pin `num_canonical_nodes`)
    across a simulated restart.
- **Detect it.** Sum `shard_seen` at the end of one epoch;
  compare to `total_shards`. Any drift is the failure.
- **Prove it.** Diff per-shard seen counts between the drifting
  run and a clean baseline. Identify which shards were
  under- or over-counted.
- **Fix.** Apply the corresponding defense from chapter 6 and
  confirm the drift is zero on a rerun.

### Drill 3 — Tokenizer drift

- **Induce it.** Set up a resumed run where the training
  environment has a *different* `tokenizer.json` than the one
  the corpus was tokenized with (e.g., a HuggingFace hub
  version bump, or a special-tokens edit).
- **Detect it.** The `loss_proxy` will step up (or down) at
  the resume point. Confirm by diffing `tokenizer_sha` at
  last-known-good against the current env.
- **Prove it.** For one held-out reference sample, tokenize
  with the two tokenizers and show the ID sequences differ.
- **Fix.** Add the launch-time tokenizer-sha assertion from
  chapter 7 and demonstrate it refuses to start with the
  mismatched tokenizer.

### Drill 4 — Silent shard corruption

- **Induce it.** Pick one shard from the MDS stream, download
  it, truncate the last few KB, and re-upload it in place. Do
  **not** update the `index.json` hash.
- **Detect it.** Run the loader. Depending on your setup you
  will either see a specific worker stall, a `DecodeError`
  swallowed as a warning, or a coverage drop on that shard.
- **Prove it.** Compute the live shard's hash and compare to
  the `index.json` value.
- **Fix.** Add the on-fetch hash verification from chapter 5
  and demonstrate the loader refuses to serve the corrupted
  shard.

Finally, write the **5-minute diagnostic protocol** as a one-page
runbook: given a misbehaving training run, which chart do you
look at first, which second, and which chart-to-failure-mode
mapping holds.

## Starter guidance

- The `loss_proxy` doesn't have to be a real loss. Any
  deterministic function of `input_ids` that changes when the
  sample distribution changes will show the wobble in Drill 1.
- For Drill 2, `shard_seen` is the single most important
  metric in this module. If you have not implemented it yet,
  do so now — it stays useful for the rest of your career.
- For Drill 3, the tokenizer sha is a gauge whose *value* is
  what matters; log it as a string label, not as a number.
- For Drill 4, keep the pre-corruption shard cached so you can
  A/B compare the loader's behavior on good vs. bad copies.
- If you cannot instrument the loader deeply, wrap it in a
  small proxy that emits the four metrics above.

## Acceptance criteria

- All four drills run to completion with clear before / after
  comparisons.
- For each drill, the report contains:
  - The chart or metric that first exposed the failure.
  - The confirming diagnostic (a specific script or query).
  - The chapter-6 defense that would have prevented it.
- The 5-minute diagnostic protocol names the chart or metric
  for each of the four failure modes, in the order chapter 6
  recommends.
- The tokenizer-sha assertion, the on-fetch hash verification,
  and the `shard_seen`-based coverage assertion are all
  implemented and demonstrated.
- Every drill uses a copy of the corpus, not the original.

## Stretch goals

- Chain two failure modes: induce Drill 1 and Drill 4
  simultaneously and confirm the diagnostic protocol still
  disambiguates them.
- Add an alarm rule per failure mode (in whatever alerting
  DSL you have — Prometheus / Datadog / CloudWatch). Trigger
  each drill and confirm the alarm fires.
- Run Drill 2's rescale variant with `num_canonical_nodes`
  pinned vs. unpinned; document the sample-order difference
  in each case.
- Implement a "watchdog" that periodically re-hashes a sample
  of shards from the manifest against the live corpus and
  alarms on mismatch, and integrate it with Drill 4.
- Write a "corpus health check" script that runs all three
  runbook-level asserts (manifest hash, tokenizer sha,
  `num_canonical_nodes`) and can be run as a pre-flight
  before every job launch.
