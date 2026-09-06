# exercise-03: W&B / TensorBoard / MLflow at Scale

**Estimated effort:** 3 hours

## Objective

Turn chapter 4 into a working experiment-tracking integration that
survives the volume math of a real multi-rank training run.
Deliverable: a pick-one tracker (W&B, TensorBoard, or MLflow)
wired into a small distributed training loop with the three
volume-control patterns from chapter 4 (rank-0 emission,
reduce-across-ranks, sub-sampled cadences), an offline-sync
demonstration, and a written comparison of the design against the
naive "log everything" baseline.

## Prerequisites

- Chapter 4 of this module. Do not start without it — the volume
  math and the three logging options are the spec this exercise
  implements.
- Chapter 1 (three-audience framing) and exercise 1 (the
  Prometheus-side dashboard); the tracker is the researcher-tier
  counterpart to the on-call dashboard and both need to coexist.
- A distributed training loop you can run with at least 2 ranks
  (2 GPUs on one node is enough; 8 across two nodes is better for
  the reduce-across-ranks pattern). torchtitan's Llama-3 config,
  nanoGPT-DDP, or an internal small model all work.
- At least one of:
  - A W&B account (free tier is sufficient) or a self-hosted W&B
    Server instance you can reach.
  - Python packages `tensorboard` and `torch` (TensorBoard needs
    no server).
  - An MLflow tracking server (a local `mlflow server` on a
    laptop is sufficient).

Pick one tracker for the main implementation. The comparison
section of the writeup covers the other two.

## Problem statement

Chapter 4's argument is that "just call `wandb.log`" fails at
pretraining scale because the volume math gets out of hand. A
`10^4`-rank run at `10^-1` step frequency emitting `10^1`
per-rank metrics generates `10^11` events over a run — enough to
break any tracker's ingest.

This exercise is the operational cash-out of chapter 4's design.
You will build the *right* integration from the start: rank-0
emission for shared scalars, reduce-across-ranks for per-rank
distributions, sub-sampled cadences for high-cardinality metrics,
and an offline-mode + sync path for network-restricted
environments. Along the way you will demonstrate why the naive
integration fails.

## Requirements

Ship one directory:

- `experiment_tracking/`
  - `README.md` — how to run everything, top to bottom.
  - `tracker.py` — the reusable integration module (rank-0
    helpers, reduce helpers, cadence helpers, offline-sync
    helpers). Tracker-agnostic where possible; tracker-specific
    behind a small adapter interface.
  - `train_ddp.py` — a small DDP training loop that imports
    `tracker.py` and drives it in the pattern the real trainer
    would. Runs against a synthetic dataset for reproducibility.
  - `naive_baseline.py` — a deliberately-wrong "log everything
    from every rank every step" implementation, for the writeup's
    comparison numbers.
  - `sync_offline.sh` — the offline-mode → hosted-tracker sync
    script (chapter 4 pattern 3).
  - `metric_schema.yaml` — the metric-naming convention (chapter
    4 "metric naming: a schema, not a habit") the team will
    adopt.
  - `design_comparison.md` — the written comparison of the three
    trackers and the defense of the volume-control design.

### 1. The tracker integration (`tracker.py`)

A thin module the trainer imports. At minimum:

- **Rank-0 emission helper** (chapter 4, option 1) — logs a
  scalar to the tracker only from rank 0; a no-op on other ranks.
  Takes the global step as an argument, not the wall clock or the
  local iteration count.
- **Reduce-across-ranks helper** (chapter 4, option 2) — takes a
  per-rank scalar tensor, computes `max / min / mean / dispersion`
  via `dist.all_reduce`, logs the aggregates from rank 0. One
  call site per metric group.
- **Sub-sampled cadence helper** (chapter 4, option 3) — logs a
  metric every N steps (configurable per metric) rather than
  every step. Used for per-layer gradient norms and any other
  high-cardinality set.
- **Sharded per-rank logging helper** (chapter 4, sharded
  section) — writes to a per-rank subdirectory or per-rank tag
  when per-rank detail is load-bearing (per-rank loss for SDC
  detection, per-rank profile traces). Not the default.
- **Prometheus double-emit helper** (chapter 4, "coupling to
  Prometheus") — a single call that emits a scalar to both the
  tracker and a `prometheus_client` gauge so the on-call
  dashboard from exercise 1 sees the same value.
- **`run_id` handling** — the trainer passes a canonical `run_id`
  (chapter 7 scheme, e.g. `<yyyymmdd>-<team>-<recipe>-<seq>`);
  every logged event carries it. No auto-generated names.
- A tracker-adapter interface with at least one concrete
  implementation (W&B or TensorBoard or MLflow — the one you
  chose). Second and third adapters are stretch.

### 2. The training loop (`train_ddp.py`)

A DDP training loop that:

- Launches with `torchrun --nproc_per_node=2` (or more) on a
  synthetic dataset. No real model required; a two-layer MLP or
  a nanoGPT-scale transformer is enough to prove the plumbing.
- Uses `tracker.py`'s helpers in the right places:
  - `train/loss`, `train/lr`, `train/gradient_norm_global` — via
    rank-0 emission every step.
  - `system/step_time_ms_{max,mean,dispersion}` — via
    reduce-across-ranks every step.
  - `grad/per_layer_norms` — via sub-sampled cadence every `N`
    steps.
  - `train/loss_per_rank` — via sharded per-rank logging (a
    small overhead-controlled sample; the point is to demonstrate
    the pattern for SDC detection, not to spam).
- Emits both to the tracker and to `prometheus_client` for the
  scalars that belong in both tiers.
- Runs long enough to generate a plottable curve on the tracker
  UI (a few hundred steps is enough).

### 3. The naive baseline (`naive_baseline.py`)

Deliberately wrong. Every rank calls `tracker.log(...)` on every
metric on every step. Do not try to make it fast or correct — it
exists to be measured against `train_ddp.py` and to make the
volume math from chapter 4 concrete. Instrument it enough that
you can report:

- Events emitted per second per rank.
- Total events emitted over a fixed number of steps.
- Wall-clock time the tracker call added per step (measure with
  `torch.cuda.synchronize()` + `time.perf_counter`).
- Whether the tracker's ingest fell behind (visible in the
  tracker UI as a lagging plot, or in the server's queue depth
  if you self-host).

This is the "why chapter 4 exists" section of your writeup.

### 4. Offline-sync demonstration (`sync_offline.sh`)

Take the training loop and run it in offline mode
(`WANDB_MODE=offline` for W&B, or write to a local `runs/`
directory for TensorBoard, or run MLflow with a local file store).
Then run `sync_offline.sh` after the run finishes, and confirm
the events appear in the hosted tracker (or the shared TensorBoard
directory / remote MLflow instance).

This is chapter 4's "pattern 3: offline mode + periodic sync"
made concrete. If you cannot reach any hosted tracker from your
environment, sync between two local directories to demonstrate
the pattern.

### 5. Metric-naming schema (`metric_schema.yaml`)

The chapter 4 "metric naming" section is a schema, not a habit.
Author `metric_schema.yaml` with the prefix hierarchy the team
will adopt, one metric per line, with:

- The metric's canonical name (e.g., `train/gradient_norm_global`).
- Its units.
- Its emission cadence (every step, every N steps, checkpoint
  boundary).
- Which reduction (rank-0, all-reduce max/mean, sharded).
- Whether it also emits to Prometheus.

Both `tracker.py` and `train_ddp.py` should consult this file (or
be lint-checked against it) so metric names never drift from the
schema.

### 6. Design comparison (`design_comparison.md`)

A short written document (2–3 pages) with:

- **Volume math for your loop**, plugged into chapter 4's
  formulas. Include: ranks used, metrics per rank per step, steps
  per second, projected events per hour naive vs. designed. Show
  the two-order-of-magnitude gap the design closes.
- **Measured numbers** from the naive baseline vs. the designed
  integration — per-step wall-clock overhead, events per second,
  and any tracker-side lag observed.
- **A comparison of the three trackers** (W&B, TensorBoard,
  MLflow) against chapter 4's "which tracker for which situation"
  matrix. Cover: hosted vs. self-hosted, air-gapped viability,
  cross-run comparison UX, artifact support, and the boundary
  with chapter 7's metadata store. Justify your choice of the
  main tracker; name the situation where a different one would
  be right.
- **The artifact vs. scalar boundary** for your loop — what you
  log as scalars, what you log as artifacts (checkpoint URIs,
  sample generations, eval reports, config snapshot), and why.
- **The alert on the tracker's own health**. What signal tells
  you the tracker is falling behind reality? What do you page
  on?

## Starter guidance

- **Do the volume math on paper before writing code.** If you
  discover the naive integration would be fine for your loop
  size, scale the loop up (add ranks, add metrics per rank) until
  the math actually breaks. The exercise is worthless if you
  never see the failure mode chapter 4 exists to prevent.
- **Do not fight the tracker's step semantics.** Every tracker
  requires monotonic global step counters; passing wall clock or
  local iterations there produces confusing plots. Chapter 4 is
  explicit — global step, everywhere.
- **All-reduce for reduction, not gather + Python.** Reducing
  per-rank scalars with `dist.all_reduce(op=MAX)` and
  `dist.all_reduce(op=SUM)` is one small collective per metric.
  Doing it in Python by all-gathering to rank 0 and computing
  aggregates there does not scale to `10^4` ranks — the payload
  size grows linearly with world size.
- **Test the sub-sampling with the incident-signature lens.** If
  you log per-layer norms every 100 steps, can you still see a
  divergence signature (chapter 6 signature 1) forming? If the
  answer is no, log a lower-cardinality summary (e.g., top-N
  layers by norm) every step in addition. Sub-sampling is a
  trade-off; document what you gave up.
- **Emit a `run_id` from the very first log.** Retrofitting a
  `run_id` label onto a tracker that was configured without one
  is annoying. Chapter 7's `run_id` scheme is the input.
- **Verify offline sync end-to-end.** Kill the network on the
  training node during the run (or start the run with no network
  reachable at all) and confirm that the run keeps running and
  the sync step catches everything up. A sync path that has never
  been tested is a sync path that will fail on the day you need
  it.

## Acceptance criteria

- `train_ddp.py` runs successfully on ≥ 2 ranks with `torchrun`
  and produces a live view in the chosen tracker's UI.
- Rank-0 emission, reduce-across-ranks, and sub-sampled cadence
  are each implemented in `tracker.py` and each has at least one
  call site in `train_ddp.py`.
- Metrics that belong in both tiers (loss, step time) appear in
  both the tracker *and* Prometheus (the exercise-1 dashboard
  can render them if pointed at this loop).
- The `run_id` label is attached to every event; runs never
  auto-generate names.
- `naive_baseline.py` runs, and `design_comparison.md` reports
  measured overhead numbers against the designed integration
  showing the difference chapter 4's volume math predicts.
- `sync_offline.sh` moves events from an offline run to a
  hosted-tracker view; the writeup demonstrates the transfer
  worked.
- `metric_schema.yaml` exists and every metric emitted by
  `train_ddp.py` is present in it (verify with a small linter or
  by manual audit; document either way).
- `design_comparison.md` covers the three-tracker matrix, defends
  the primary-tracker choice, names the artifact vs. scalar
  boundary, and defines the tracker-health alert.
- Sub-sampled metrics were tested against at least one incident
  signature from chapter 6 — the writeup names which signature
  and confirms it remains observable at the reduced cadence.

## Stretch goals

- Implement a second tracker adapter (e.g., you built against
  W&B; also build against TensorBoard) and prove that
  `train_ddp.py` swaps between them via configuration only.
  Chapter 4's argument that "the tracker is a discovery layer,
  not the canonical store" is much easier to defend when your
  own code can swap them.
- Wire an MLflow model registry entry for each checkpoint the
  training loop writes: register the checkpoint URI, its
  reproducibility bundle URI (from exercise 4 or a placeholder),
  and the tracker URL. This is the seam into chapter 7 the
  fine-tuning team will pull from.
- Instrument tracker-health signals from the trainer: track
  events queued vs. flushed inside `tracker.py`, expose them as
  Prometheus gauges, and alert when the queue depth exceeds a
  threshold. Chapter 4 named "the trainer runs faster than the
  tracker can ingest" as a common failure mode; this is how you
  catch it before it silently drops.
- Add per-rank sharded logging for a *deliberately-poisoned*
  rank (multiply that rank's loss tensor by a small perturbation
  factor) and confirm the sharded per-rank view surfaces the
  drift. This is the SDC detection cadence chapter 6 signature 5
  depends on.
- Add a `chapter-7` sidecar that writes the run into a small
  metadata store (SQLite is fine for the exercise; Postgres is
  the honest production choice) with the fields from chapter 7's
  `run` table populated. Now the tracker, the reproducibility
  bundle, and the metadata store all agree on `run_id`.
