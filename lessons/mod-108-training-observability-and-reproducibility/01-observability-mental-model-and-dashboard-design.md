# The Observability Mental Model and the One-Glance Dashboard

mod-105 taught you how to see a fabric misbehave. mod-106 taught you
how to catch a checkpoint mid-flight when a node dies. mod-107 taught
you how to squeeze the last percent of MFU out of a step. This module
is the surface that lets you *know* those systems are working — the
one your on-call rotation stares at during every run, the one your
research partners screenshot into their post-mortems, the one your
platform director points to during a budget review to argue the
fleet is not being wasted.

An observability layer for a training platform has to answer a very
narrow set of questions well. If it answers them slowly, your
research partners walk over to your desk instead of opening a
dashboard. If it answers the wrong ones, you burn GPU-hours you did
not need to. The rest of this chapter fixes the mental model and
the "one-glance dashboard" you build the rest of the module against.

## The two axes: model view and cluster view

Every training-run signal falls into one of two buckets:

- **Model-level view.** What the optimizer is doing to the weights.
  Loss, gradient norm, parameter norm, learning-rate schedule,
  tokens processed, MFU, evaluation loss / task metric. These are
  the signals your ML researcher partner cares about. They live in
  the experiment tracker (W&B, TensorBoard, MLflow) because they are
  per-run, per-step, and semantically tied to a specific
  training-run identity.
- **Cluster-level view.** What the hardware is doing while it runs.
  Per-GPU utilization, HBM temperature, ECC / XID counters, NCCL
  step time, NVLink and NIC bandwidth, node availability, storage
  read throughput. These are the signals your training-platform
  on-call cares about. They live in the fleet telemetry stack
  (DCGM + Prometheus + Grafana) because they are per-host,
  per-second, and shared across every run on the cluster.

Both views are necessary and neither is sufficient. A run that has
a beautiful loss curve on a cluster whose HBM is thermally
throttling is a run whose next step will hang. A cluster whose
per-GPU util panel is a perfect green wall while every job's
evaluation loss has stalled is a cluster that is silently corrupting
weights. Fluency in this module is fluency in bridging the two views
in the same three seconds of glancing at a screen.

The rest of the module assigns tools to axes:

- **Chapter 2** wires up the cluster-level view with DCGM +
  Prometheus + Grafana.
- **Chapter 3** wires up the model-level view with an experiment
  tracker (W&B / TensorBoard / MLflow) tuned for pretraining scale.
- **Chapter 4** builds the reproducibility bundle that lets both
  views be *re-derived* on a different day from the same inputs.
- **Chapter 5** teaches you to read shapes across both views as a
  single run-time signature.
- **Chapter 6** exports the metadata contract so downstream tracks
  can consume the run.

## The three questions your dashboard answers in five seconds

The point of the dashboard is not "show every metric". Every metric
is available on drill-down. The point is: at 3 a.m., paged by the
scheduler, before you have coffee, you should be able to look at
one URL for five seconds and answer three questions:

1. **Is the loss going down?** (Model view.) If it is not, either
   the training loop is broken or your data / config changed. If it
   is, the run is still doing useful work regardless of hardware
   noise.
2. **Is the fabric healthy?** (Cluster view.) If XID counters are
   ticking, HBM is above the throttle threshold, or NCCL step time
   is drifting, the run is one bad step away from a hang. If they
   are quiet, whatever is going wrong is a code / data problem, not
   a fabric problem.
3. **Is throughput on target?** (Both views.) The right number is
   tokens/s/GPU at steady state and it should match the number your
   run's launch banner promised. If it is off, you are burning
   money — go find the panel that shows why.

Every panel on the top-level dashboard has to earn its place by
serving one of those three answers. Everything else lives on a
drill-down view.

## Panel budget: eight panels, one screen

A useful convention is to fit the entire top-level dashboard on a
single 1920×1080 screen without scrolling. That constrains you to
roughly eight panels. A budget that has held up across three
generations of training clusters:

| # | Panel | Answers | Time-series unit |
|---|-------|---------|------------------|
| 1 | Training loss (log-y) | Q1: loss going down | tokens seen |
| 2 | Gradient-norm p50 / p95 | Q1: numeric stability | step |
| 3 | Tokens/s/GPU (steady state) | Q3: on target | step |
| 4 | MFU (%) | Q3: on target | step |
| 5 | Per-rank step-time histogram (heatmap) | Q2 + Q3: straggler | step × rank |
| 6 | DCGM SM util + HBM temp (heatmap by node) | Q2: fabric health | second × node |
| 7 | NCCL comm time / total step time overlay | Q2 + Q3: comm bound | step |
| 8 | XID / ECC / node-down count | Q2: fabric health | second |

Each panel is one Grafana panel or one W&B panel. Each panel
carries a threshold line — the target value from the run's launch
banner, or the last-known-good baseline. A panel without a
threshold line does not help someone glancing at it under stress;
they need to see "current vs. expected" at a glance, not just
"current".

Two panels are worth spelling out because they are where the two
views compose:

- **Panel 5 (per-rank step-time histogram).** The x-axis is step
  number, the y-axis is rank, the color is step time. A healthy run
  is a uniform blue; a straggler is a single yellow row that
  darkens over time; a fabric event is a vertical stripe. This
  panel alone diagnoses ~60% of "the run feels slow" pages. See
  chapter 5.
- **Panel 7 (NCCL comm-time overlay).** Two series on the same
  axis: the total step time and the collective time inside that
  step (measured with `torch.cuda.Event`s or the PyTorch profiler's
  Kineto trace, sub-sampled to one step every 100). The delta is
  compute time. If total = comm, you are comm-bound and the answer
  lives in mod-105 / mod-107. If total ≫ comm, you are compute-
  bound and the answer lives in mod-107.

Anything you cannot fit on the top screen goes into a drill-down
dashboard. The drill-down dashboards are unlimited; the top screen
is not.

## Naming and label discipline

Prometheus is unforgiving on cardinality mistakes and Grafana is
unforgiving on inconsistent naming. Both problems compound at
cluster scale. Two conventions to lock in on day one:

- **Metric naming follows the Prometheus best-practice
  guide.**<!-- needs-research: verify current URL, was
  https://prometheus.io/docs/practices/naming/ --> Lowercase
  snake_case, unit suffix (`_seconds`, `_bytes`, `_total`), and a
  namespace prefix per team (`training_`, `dcgm_`, `nccl_`).
- **Labels are bounded.** Never label with an unbounded set:
  no `run_id` as a Prometheus label (put it in W&B), no `step`
  (put it in a histogram bucket), no per-tensor names. Bounded
  labels are `node`, `rank`, `gpu`, `pod`, `namespace`, `job`,
  `role`. Everything else lives in the experiment tracker or the
  logs.

The trade-off is deliberate. Prometheus is your low-cardinality
fleet metric store; the experiment tracker is your high-cardinality
per-run metric store. Confuse the two and Prometheus will fall
over at ~500 concurrent runs while the tracker gets underused.

## The Grafana / experiment-tracker split

A useful mental picture of "which system holds which panel":

- Panels 1, 2, 3, 4 live in the experiment tracker (W&B / MLflow /
  TensorBoard). They are per-run, per-step, and part of the run's
  archived record. Chapter 3 shows how to keep this tracker from
  collapsing at pretraining scale.
- Panels 5, 6, 7, 8 live in Grafana off Prometheus. They are
  per-node / per-rank / per-second, and shared across every run on
  the cluster. Chapter 2 wires this up.

Both systems export links to each other. Every W&B run's page has a
Grafana link filtered by `job_id`; every Grafana panel has a
click-through into the currently-scheduled run's tracker page. That
cross-link is what lets the on-call glance move seamlessly between
"model went wrong" and "hardware went wrong" in the same triage
minute.

## Grafana docs and Prometheus best practices as the ground truth

Two references you should keep open while building this out:

- **Grafana documentation** — dashboarding, variables, template
  variables, alerting.
  <!-- needs-research: canonical Grafana docs base is
  https://grafana.com/docs/grafana/latest/ but verify sub-page URLs
  before citing directly. -->
- **Prometheus documentation and best practices** — instrumentation,
  labels, naming, alerting rules, cardinality control.
  <!-- needs-research: canonical Prometheus docs base is
  https://prometheus.io/docs/ but verify sub-page URLs before
  citing directly. -->

Both projects publish opinionated guidance that has, in practice,
survived the transition from "monitoring a web service" to
"monitoring an ML training cluster" better than most bespoke
tooling. Learn their idioms before you invent your own.

## Two dashboard smells to reject

Two anti-patterns you will encounter and should push back on:

- **The 40-panel wall.** Someone assembles a dashboard with every
  Prometheus metric they could find. It looks impressive at a demo
  and is unusable in an incident. Ask: "which three questions does
  this answer in five seconds?" If the author cannot answer, cut
  the panels down until they can.
- **The runs-are-metrics conflation.** Someone puts per-run
  training loss into Prometheus with a `run_id` label. Prometheus
  falls over at ~500 concurrent runs. The right home for per-run
  metrics is the experiment tracker; Prometheus is for fleet
  metrics. This mistake is the single most common way a
  training-observability stack collapses in its second year.

Both patterns are gently corrected by anchoring on "answer the
three questions in five seconds" as the acceptance test for every
new panel.

## Summary

- Training-run observability has two axes: the model view (loss,
  gradient, MFU) and the cluster view (per-GPU util, HBM, NCCL,
  XID). Both are necessary; neither is sufficient.
- The one-glance dashboard answers three questions in five
  seconds: is the loss going down, is the fabric healthy, is the
  throughput on target.
- Eight panels fit one screen: loss, gradient norm, tokens/s/GPU,
  MFU, per-rank step-time histogram, DCGM SM/HBM heatmap, NCCL
  comm-time overlay, XID / ECC / node-down count.
- Model-view panels live in the experiment tracker (W&B / MLflow /
  TensorBoard). Cluster-view panels live in Grafana off
  Prometheus. Cross-link the two.
- Prometheus is low-cardinality fleet metrics; the experiment
  tracker is high-cardinality per-run metrics. Never confuse them
  with a `run_id` label on a Prometheus metric.
- Reject the 40-panel wall and the runs-as-metrics conflation.
  Every top-screen panel earns its place by serving one of the
  three glance questions.
