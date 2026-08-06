# exercise-01: Training-Run Dashboard Design

**Estimated effort:** 3 hours

## Objective

Turn chapter 2 into a Grafana dashboard, checked into version
control, that renders a live small pretraining run in three rows
(liveness, efficiency, hardware). Deliverable: a `dashboard.json`
plus a written design brief that defends every panel's presence
against the goodput lens, and the trainer-side emission code that
populates the metrics.

## Prerequisites

- Chapter 1 of this module (audiences, goodput framing) and
  chapter 2 (the panel catalog). Do not start the exercise
  without them.
- A running Prometheus + Grafana stack, either local (Docker
  Compose is fine) or on your cluster. If you need to stand it
  up, exercise 2 walks you through it — you can do exercise 2
  first if convenient.
- A small training loop you can run for at least a few dozen
  steady-state steps (nanoGPT-scale, torchtitan's Llama-3, or an
  internal model). It does not need to be at cluster scale for
  this exercise; a single node with 1–8 GPUs is enough to prove
  the plumbing.
- The `prometheus_client` Python package installed in the trainer
  environment.

## Problem statement

Chapter 2's argument is that a dashboard is a design artifact,
not a random pile of graphs. Your task is to *design* one — pick
the panels, pin the refresh cadences, split alert from display,
and defend each choice — then implement it in Grafana JSON with a
minimum viable trainer-side emitter. The result should be
reviewable by a teammate who has not read the chapter and be
enough for the training team to adopt as the on-call landing
page.

## Requirements

Ship one directory:

- `training_dashboard/` — the artifacts.
  - `dashboard.json` — Grafana dashboard export.
  - `emitter.py` — the trainer-side `prometheus_client`
    integration.
  - `train_smoke.py` — a small training loop that imports
    `emitter.py` and drives the dashboard for demo purposes.
  - `docker-compose.yml` or a Kubernetes manifest that stands up
    Prometheus + Grafana with the dashboard pre-provisioned.
- `design_brief.md` — the written defense of the design.

### 1. The dashboard (`dashboard.json`)

Export a Grafana dashboard with the three rows from chapter 2:

- **Row 1 (liveness)** — loss curve (rank-0, high-resolution),
  global gradient norm, median step time, world-size sanity
  panel, steps-since-last-checkpoint counter, run header.
- **Row 2 (efficiency)** — MFU with target band overlaid,
  per-step-time dispersion across ranks, comm-time as a fraction
  of step-time, GPU-utilization histogram.
- **Row 3 (hardware)** — GPU temperature top-N, ECC-delta
  heatmap, NVLink / NIC error rate per node, throttle-reason
  count. If DCGM is not wired up yet, stub these panels with
  placeholder queries that render "no data" gracefully.

Requirements on the JSON itself:

- **Template variable `$run_id`** at the top of the dashboard,
  populated from a Prometheus label-values query.
- **Refresh cadences per row**: 5s / 15–30s / 30–60s (chapter 2).
  Set explicitly.
- **Every panel has a description** that names the metric, its
  units, and the on-call action if it goes out of band. This is
  what makes the dashboard readable at 3 AM.
- **Every panel links** to the appropriate mod-106 chapter 5
  runbook entry via a description hyperlink.

### 2. The emitter (`emitter.py`)

A thin module the trainer imports. At minimum:

- A `prometheus_client` HTTP server started on rank 0.
- Gauges/counters for: `train_loss`, `train_lr`,
  `train_gradient_norm_global`, `system_step_time_ms_median`,
  `system_step_time_ms_dispersion`, `system_comm_time_fraction`,
  `system_gpu_util_median` (from within the trainer, not DCGM —
  DCGM covers Row 3), `run_world_size`,
  `train_steps_since_last_checkpoint`.
- The `run_id` label attached to every metric.
- A helper for the reduce-to-rank-0 pattern from chapter 4
  (all-reduce for max/mean/dispersion; log from rank 0 only).

### 3. The smoke test (`train_smoke.py`)

A synthetic training loop (not a real model — this is about
proving the plumbing, not measuring MFU) that:

- Emits fake-but-plausible metric values at a steady cadence.
- Calls `emitter.py`'s helpers in the same pattern the real
  trainer will.
- Occasionally injects a synthetic incident: a loss spike, a
  dispersion excursion, a step-time cliff. Enough to verify the
  dashboard *renders* the shapes from chapter 6.

Runnable via `python train_smoke.py`; produces the metrics
Prometheus scrapes; the dashboard renders live.

### 4. The design brief (`design_brief.md`)

A short (roughly 2–3 pages) written document that:

- **Lists every panel** with its metric, its refresh cadence,
  its axis choice (log vs. linear), its threshold band, and the
  on-call runbook it links to.
- **Justifies every panel against goodput** (chapter 1) — the
  single-sentence answer to "why is this panel on the page".
- **Names every panel you chose *not* to include** and why. Being
  explicit about omissions is as important as the inclusions.
- **Documents the alert–display split** — which metrics are
  alerted on (via Alertmanager or Grafana alerting) and which
  are display-only.
- **Includes screenshots** (or an ASCII sketch) of the rendered
  dashboard with fake data, so a reviewer can see the layout
  without standing up the stack.

## Starter guidance

- **Do not download a random Grafana dashboard from the web and
  edit it.** The design brief is what you are being evaluated on;
  start from a blank dashboard.
- **Version-control the JSON before iterating in the UI.** The
  Grafana web UI is a great editor and a bad source of truth.
  Commit early; commit after every UI change.
- **Templatize `$run_id` from the start.** Retrofitting a
  templatized dashboard is more painful than starting with one.
  See the Grafana template-variables docs for the `label_values`
  query type.
- **Use the fake-incident injection to test the dashboard.** A
  dashboard that never rendered a real spike will surprise you
  the first time one happens. Chapter 6's five signatures each
  need to be observable on your rendered dashboard; verify each
  before shipping.
- **Do not skip the panel descriptions.** They are the primary
  documentation the on-call reads. A panel without a description
  is a panel that will be misread.

## Acceptance criteria

- `dashboard.json` is committed and re-imports cleanly into a
  fresh Grafana instance via provisioning (not by hand).
- The dashboard renders the three rows in order (liveness /
  efficiency / hardware).
- `$run_id` is a template variable driving every panel; changing
  it filters the whole dashboard.
- Refresh cadences match chapter 2: 5 s / 15–30 s / 30–60 s per
  row. Set explicitly, not left at Grafana defaults.
- Every panel has a description that names the metric, its
  units, and the on-call action; every description links to the
  relevant mod-106 chapter 5 runbook entry.
- The trainer-side emitter is a real module with the reduce-to-
  rank-0 pattern implemented (chapter 4); the smoke test drives
  it and the dashboard responds live.
- `design_brief.md` justifies every panel against goodput,
  documents omissions, and includes the alert–display split.
- Chapter 6's five signatures (divergence, spike, cliff,
  straggler, silent corruption) are each observable on the
  rendered dashboard — inject each in the smoke test and confirm
  the dashboard makes it visible.
- No panel is a plain mean where a distribution would tell the
  truth. In particular, GPU utilization is a histogram or top-N,
  not a scalar.

## Stretch goals

- Add a second dashboard that is the *researcher-facing* variant
  (chapter 4) — per-layer gradient norms sub-sampled, sample
  outputs, validation slice losses. Different tool (W&B or
  TensorBoard) is fine; the point is to draw the boundary from
  chapter 1 explicitly.
- Wire Alertmanager to fire on: NaN loss, world-size mismatch,
  dispersion sustained above 1.20 for N minutes, DBE ECC event.
  Ship the rule files alongside the dashboard.
- Add a "run summary" panel to Row 1 that pulls from the
  metadata store (chapter 7) via a lightweight sidecar and
  renders the run's `bundle_uri`, `tracker_url`, and
  `owning_team` right on the dashboard. Removes the "which run
  is this and who owns it" question at 3 AM.
- Version the dashboard in a way that supports schema evolution:
  bump the dashboard version when you add a panel, and keep the
  N-1 version accessible for historical run comparison. Document
  the migration story.
- Fork the design brief into a "dashboard for a 100k-GPU fleet"
  variant. What breaks about the three-row layout at that scale?
  What panels have to move to a drill-down page? Defend in
  writing.
