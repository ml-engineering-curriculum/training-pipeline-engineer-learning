# exercise-01: DCP Async Save, Stateful Sampler, and Reshard

**Estimated effort:** 4 hours

## Objective

Turn chapter 3's DCP walkthrough into working code that (a) saves the
full training-loop state — model, optimizer, LR scheduler, RNGs,
sampler position, and step — via `dcp.async_save`, (b) demonstrates
that a checkpoint written at world size `N` loads correctly onto
world size `M ≠ N`, and (c) produces a numeric before/after report on
the step-time cost of async vs. synchronous save on your fabric.
The deliverable is a small trainer plus a written measurement report.

## Prerequisites

- Chapters 2 and 3 of this module.
- mod-101 (DDP / FSDP2) enough that you can spin up a small
  `FullyShardedDataParallel` model.
- mod-105 chapter 7 for context on the storage tier you write the
  checkpoint to. Any of Lustre, WEKA, FSx for Lustre, or a
  local-NVMe scratch works for the exercise; note the tier in the
  report.
- A PyTorch environment (pin the version in your report) with
  `torch.distributed.checkpoint` available. The DCP tutorial at
  https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
  should run on this environment.
- Access to at least 4 GPUs, arranged as either 2 nodes × 2 GPUs or
  1 node × 4 GPUs. You need at least two ranks and you need to be
  able to change the world size between runs.

## Problem statement

Your team currently checkpoints via a "rank 0 gathers everything"
pattern (chapter 2's anti-pattern). You need to migrate to DCP.
Before the migration you have three questions to answer with numbers:

1. **What is in the checkpoint?** Enumerate every stateful piece the
   trainer touches, and prove — via a resume-from-crash equivalence
   test — that none was silently omitted.
2. **What does async save cost, if anything?** Compare per-step
   wall-clock time with no save, with synchronous save, and with
   async save at the same interval. Is async a wash, a win, or a
   loss on your storage tier?
3. **Does resharding work end-to-end?** Save on world size `N`, load
   on world size `M`, verify the resumed loss curve matches the
   pre-crash trajectory.

## Requirements

Ship one small repository (or one branch of an existing repo) plus a
`report.md` in the same directory. The repository must contain:

### 1. A minimal DCP-based trainer (`trainer.py`)

Requirements:

- Uses FSDP2 (`fully_shard` / `FullyShardedDataParallel`) or DDP
  as the parallelism. Either works; pick one and be consistent.
- Trains a small model (a random-init Transformer of a few hundred
  million parameters is plenty; the exact model is not important as
  long as the state has a non-trivial optimizer footprint).
- Uses `torch.utils.data.DataLoader` with a **`Stateful` sampler**
  as sketched in chapter 3. Any dataset works — a synthetic
  in-memory dataset is fine, as long as the sampler position
  matters for what the next batch looks like.
- Wraps its state in an `AppState` class implementing the
  `Stateful` protocol from `torch.distributed.checkpoint.stateful`.
  The `state_dict` must include, at minimum: model params,
  optimizer state, LR scheduler state, step counter, per-rank CPU
  and CUDA RNGs, sampler state, and mixed-precision scaler state
  if you use AMP.
- Provides a CLI toggle for `--save-mode {none, sync, async}` and a
  `--save-every N` argument.
- On startup, checks for the newest DCP checkpoint at a configured
  path and, if present, loads it (chapter 4's re-entrant contract).
- Writes to a checkpoint path via the atomic-rename pattern from
  chapter 3 (`step-<N>.tmp/` → rename to `step-<N>/` on success).

### 2. A reshape driver (`reshape_test.sh`)

A shell script (or Python driver) that:

1. Launches the trainer under `torchrun` at world size `N` (pick
   whatever your hardware supports; document the choice) with a
   fresh checkpoint directory.
2. Trains for `S_1` steps, taking a checkpoint every `k` steps.
   Records per-step loss to `trace-N.jsonl`.
3. Sends SIGTERM to the trainer.
4. Launches the trainer again with the same checkpoint directory
   but at world size `M = N / 2` (or `M = N × 2` if you can get
   more hardware). No arguments other than world size change.
5. Trains for `S_2` more steps, recording per-step loss to
   `trace-M.jsonl`.

### 3. A measurement harness (`bench.sh`)

Runs three back-to-back experiments on the same allocation with the
same trainer code:

- Baseline: `--save-mode none` for 500 steps. Records mean and P99
  step time.
- Sync: `--save-mode sync --save-every 100` for 500 steps. Records
  mean and P99 step time (both the "saving" and "non-saving" steps
  should be separately reported).
- Async: `--save-mode async --save-every 100` for 500 steps.
  Same breakdown.

Save the raw per-step traces so the report can be reproduced.

## The `report.md`

Answers, in order, with numbers:

### 1. State inventory

A table listing every field in `AppState.state_dict()` with a
one-line description and a note on why it must be checkpointed
(cross-reference chapter 2's inventory). If your trainer omits any of
chapter 2's items, defend the omission.

### 2. Resume equivalence

Run the reshape driver. Show:

- The loss curve from `trace-N.jsonl` (the first N-rank run).
- The loss curve from `trace-M.jsonl` (the resumed M-rank run).
- The concatenation. It should be visually smooth across the
  crash-and-resume boundary.

Also run a **control**: launch a fresh training run at world size
`M` from step 0 with a fresh seed for `S_1 + S_2` steps and plot its
loss curve. It should have the same general shape (not exact
numbers — the two runs saw different rank-decomposition and NCCL
non-determinism) but should not have the discontinuity that a broken
resume would produce.

Note: due to floating-point non-determinism in FSDP collectives, the
post-resume curve will not be bitwise identical to a hypothetical
"no crash" run at the same world size. That is expected and is the
subject of chapter 6's discussion of cross-rank agreement. Report
what you *did* see and interpret it.

### 3. Save-cost measurement

For each of `none / sync / async`, publish:

- Mean step time (non-saving steps only).
- Mean step time (saving steps only).
- P99 step time overall.
- Amortized wall-clock overhead per save
  (`total_wall_clock - baseline_wall_clock / n_saves`).

Answer, in one paragraph each:

- Is async materially cheaper on the critical path than sync? By
  how much?
- What was the storage tier and its measured write bandwidth
  during the async save? Cross-reference mod-105 chapter 7.
- What is the recommended `--save-every` interval on this
  fabric, from chapter 2's failure-window formula, given the
  measured `T_save` and an assumed failure rate you defend in
  writing?

### 4. Failure modes you hit

Document at least three real issues you encountered — could be
missing state in the first `AppState`, an atomic-rename bug, a torn
checkpoint on abrupt SIGTERM, a stale future not awaited before the
next save, a DCP version mismatch, an OOM on the staging copy. For
each: what the symptom was, how you diagnosed it, and how you fixed
it.

## Starter guidance

- **Start with the DCP tutorial.** Get the tutorial code running
  end-to-end against a single-node 2-GPU setup before wiring in
  your model / optimizer / sampler. Confirm you can save and load
  the tutorial state.
- **Add fields to `AppState` one at a time.** For each new field,
  write a smoke test: save with the field, delete the in-memory
  state, load, assert the field round-tripped correctly. This is
  the fastest way to catch API surprises (e.g., some optimizer
  state dicts are not vanilla tensors).
- **Use the current recommended `get_state_dict` / `set_state_dict`
  bridge from `torch.distributed.checkpoint.state_dict`** for the
  FSDP2 model and optimizer. The old `FSDP.state_dict_type(...)`
  context manager still works on many versions but is not the
  recommended path. Pin the exact import and note the PyTorch
  version in the report.
- **For the stateful sampler**, implement chapter 3's sketch. Make
  the `epoch` and `position` fields round-trip; verify on a
  synthetic dataset that after loading, the next batch is what the
  pre-crash sampler would have produced next.
- **Do not add pre-training-set staging.** The exercise is about
  DCP mechanics, not data-loader design. mod-103 covers loader
  correctness.
- **Log everything to JSONL, not stdout.** Post-hoc analysis is
  much easier when the trace is machine-readable. `report.md` can
  embed small tables computed from the JSONL.

## Acceptance criteria

- `AppState` includes at minimum: model, optimizer, LR scheduler,
  step, per-rank CPU + CUDA RNG, sampler, scaler (if AMP). Any
  omission is justified in the report.
- The reshape driver runs to completion at `N → M` and produces
  three loss traces that support a resume-equivalence claim in the
  report.
- The measurement harness produces reproducible numbers for
  `none / sync / async` step-time distributions.
- The report answers all four sub-sections above with numbers, not
  narratives.
- The `--save-every` interval recommendation in section 3 is
  derived from an explicit formula (chapter 2), not asserted.
- The three failure modes in section 4 are real — a report that
  hits no issues at all across four hours of exercise is either
  suspicious or copying somebody else's work.
- No invented numbers. If a drill did not run (e.g., you only have
  4 GPUs and cannot demonstrate a 4→8 reshape), the report says so
  and does not fabricate a data point.

## Stretch goals

- Implement chapter 3's option-2 sampler reshape (deterministic
  re-plan from `(epoch, seed)` across the new world size). Verify
  that no index is repeated or skipped across the boundary.
- Add a second storage backend (e.g., swap FSx-for-Lustre for a
  local NVMe scratch, or add an S3-based DCP backend via a
  plugin). Re-run the measurement harness on both. The delta is
  what chapter 2's storage-cost model predicts.
- Add a "long-term consolidated" save path: after every K
  intra-run DCP saves, produce a consolidated Safetensors export
  suitable for handoff to an inference stack. Measure the
  wall-clock cost of the consolidation and confirm it is off the
  training critical path.
- Add periodic reads of the just-written checkpoint by a separate
  process (e.g., an eval runner). Measure whether the read
  interferes with the trainer's read/write of the next save.
