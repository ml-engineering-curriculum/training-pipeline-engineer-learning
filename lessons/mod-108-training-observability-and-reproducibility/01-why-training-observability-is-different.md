# Why Training Observability Is Different

Every previous module in this track produced signals — MFU numbers,
per-step times, NCCL events, checkpoint durations, ECC counters. This
module is about the layer that catches those signals, stores them,
plots them, and turns them into decisions. Without it, everything
mod-101 through mod-107 built is measurable only in a live terminal
by whoever launched the run.

The problem observability solves at training scale is not the same
one it solves for a web service. A production API has thousands of
requests per second, each independent, and observability's job is to
pull outliers out of that stream. A pretraining run is a single
long-lived process across thousands of ranks, all running the *same*
step in lockstep, for weeks. What you are looking for is not an
outlier request — it is a slow drift in a distribution, a per-rank
divergence that separates from the pack over hundreds of steps, a
kernel that was fast on framework version A and is 8% slower on
framework version B. The signals that matter are structurally
different, and so is the tooling built around them.

The seven chapters here are the design of that tooling: the metric
catalog, the GPU-telemetry stack, the experiment tracker, the
reproducibility bundle, the run-time signature catalog, and the
metadata store the downstream teams pull run provenance from. Chapter
1 frames why the problem is a scale problem in its own right and
draws the boundary with the modules that emit the raw signals.

## The three audiences a training observability stack serves

Three audiences read a training-run dashboard, and they want different
things from it. If you design the stack for one of them and treat
the others as afterthoughts, the other two will silently work around
it — and silent workarounds become the reason nobody can reproduce
last quarter's best run.

### 1. The on-call platform engineer

At 3 AM, on-call needs three things in the first thirty seconds:

- **Is the run healthy right now?** Loss finite, gradient norm within
  its usual band, step time within its usual band, no ranks stuck.
- **If not, which class of incident?** The five classes from mod-106
  chapter 5 (loss spike, NaN, NCCL timeout, hardware fault, silent
  corruption). Each has a different runbook.
- **What changed recently?** A code deploy, a config diff, a fabric
  event, a node drain. The correlation between a signal and the
  most recent change is often the answer.

On-call reads a *dashboard*. Their tolerance for chasing signals
across three tools is zero. If the loss curve is in W&B, the GPU
temperature is in Grafana, and the fabric counters are in a
vendor-specific portal, on-call will only look at the first two.

### 2. The ML researcher running the run

The researcher wants a very different set of numbers, on a different
cadence:

- **Loss and validation curves** at the resolution they need to reason
  about warm-up, learning-rate schedule inflection points, and
  convergence.
- **Per-layer gradient norms** to diagnose optimization pathology in
  the model itself (chapter 6's divergence signature).
- **Sample outputs** from the model at checkpoint boundaries — text
  generations, per-class accuracies, whatever the eval harness
  produces.
- **Hyperparameter sweep views** that compare a run against its
  siblings.

The researcher reads an *experiment tracker* — W&B, TensorBoard,
MLflow. They rarely open Grafana. If throughput drops 10%, they may
not notice; if the loss curve is subtly off, they notice within
minutes.

### 3. The downstream team consuming the run

Fine-tuning-engineer and model-evaluation-engineer teams (see the
sibling roles in this org) do not run the pretraining job. They
consume its outputs. Their questions look nothing like the first
two audiences':

- **Which specific checkpoint** was this base model trained from?
- **What was the exact data mix** at that checkpoint step?
- **Which tokenizer** and which vocabulary version?
- **What was the seed, config, and container digest** that produced
  the weights?
- **Is this run reproducible** to bitwise equivalence, to
  loss-curve equivalence, or not at all?

They read the *metadata store*. Chapter 7 is the schema that answers
these questions; chapter 5 is the bundle that populates it.

The three audiences share almost no infrastructure and share almost
no metric definitions. The observability stack's job is to serve all
three from one authoritative source of truth — not three copies that
drift.

## The metric surface: what a training run emits

A single training step emits roughly four groups of signals. Each
group needs a different storage tier because the volumes, retention
requirements, and query patterns differ by orders of magnitude.

### Model-level scalars, per step

Loss, validation loss, learning rate, gradient norm (global and
per-layer), weight norm, optimizer state statistics. These are what
the ML researcher plots. Volume is small — one Python float per
metric per step, a few tens of metrics per step. A run of `10^6`
steps with 50 metrics is `5 × 10^7` scalars, comfortably absorbed
by any experiment tracker.

The one gotcha: **per-layer gradient norms** on a `10^2`-layer model
turn a "few tens of metrics" into a few thousand. Chapter 4's
sub-sampling patterns are for this class.

### System-level scalars, per rank per step

Per-rank step time, per-rank data-loader wait time, per-rank
communication time, per-rank max HBM allocated, per-rank peak
temperature during the step. These are what on-call plots for
straggler detection (mod-106 chapter 6). Volume is `world_size ×
metrics_per_rank_per_step`. A `10^4`-GPU run at `10^-1` step
frequency emits `10^3` samples per second per metric. This is high
enough that a naive "log everything to W&B" costs money and
saturates network — the design in chapter 4 is the answer.

### Hardware telemetry, continuous

DCGM's field IDs: SM clock, memory clock, GPU temperature, power
draw, HBM ECC error counters, NVLink error counters, PCIe error
counters, throttle-reason bitmask. These are what DCGM's exporter
emits into Prometheus at a poll interval you configure (typically
`1–10 s`). Volume is `world_size × ~30 metrics × 1/poll_interval`.
Retention is *long* — you want six months of ECC deltas to spot a
GPU on its way out. Prometheus + a long-term store (Cortex, Thanos,
Mimir, or a managed service) is the pattern.

### Events, ad-hoc

Xid events, NCCL warnings, checkpoint save events, rendezvous
events, LR schedule inflections. These are discrete, low-volume,
and the tags on them (rank, step, node, reason) matter more than
any numeric value. They belong in a log/event store (Loki,
Elasticsearch, or whatever the platform standardizes on) and in
the run's *logbook* — a curated append-only file the researcher and
on-call both write to, modeled after the OPT-175B chronicles
(mod-106 chapter 7).

Notice how "log everything into one place" fails: the per-step
metrics and the continuous hardware telemetry alone differ by two
orders of magnitude in volume, and by three orders of magnitude in
retention requirements. Multiple tiers, one *canonical* run
identifier that ties them together. The `run_id` is the join key
that makes the stack coherent; chapter 7 makes that formal.

## Goodput as the north-star metric

Every observability decision in this module reduces to "does this
number help us understand or defend goodput?" mod-106 chapter 8
defined goodput as tokens per wall-clock hour that made it into a
persisted checkpoint. It is the ratio the platform team's
performance is graded on, and it is the metric the finance
conversation in mod-109 turns into a dollar number.

You cannot dashboard goodput directly step-by-step — it is a lagging
metric that only crystallizes at checkpoint boundaries. What you
*can* dashboard is its components:

- **Uptime** — is the job running? Not stalled, not looping in a
  restart cycle. On-call's first-glance signal.
- **Throughput** — when it is running, tokens per second. Chapter 2
  covers plotting this against its long-run mean.
- **Recovery cost** — when it *stopped* running, how many tokens
  and how many wall-clock minutes did we lose? Feeds into the
  reduction of goodput week over week.
- **MFU** — chapter 2 of mod-107 defined this; chapter 2 of this
  module dashboards it against the run's target band.

Every panel on the dashboard exists because it explains a component
of goodput. If you can't defend a panel's presence on that basis,
delete it — a dashboard with 40 panels is a dashboard nobody reads.

## The three anti-patterns this module exists to prevent

Named, so you can catch them in review.

### Anti-pattern 1: "The trainer log is our observability"

The training loop's stdout has the loss and the step time. Some
teams stop there. This works exactly until:

- The log grows past `10^6` lines and grep is not a query engine.
- The researcher wants to compare this run against a run from last
  month and last month's log has been rotated.
- The on-call wants the last five minutes of loss values on their
  phone at 3 AM.
- Two different runs' log lines got interleaved on the shared
  filesystem and now nobody can tell which loss belongs to which
  run.

An experiment tracker (chapter 4) is not optional at pretraining
scale. The trainer log is a *complement* to it — the record of what
the process actually did — not a substitute.

### Anti-pattern 2: "We check GPU metrics if there's a problem"

DCGM is a background always-on signal, not an on-demand diagnostic.
By the time you notice HBM ECC errors are climbing on rank 47, that
rank has already been silently producing wrong numbers for hours.
The always-on stream is what lets you catch fail-slow and silent-
corruption incidents (mod-106 chapter 6) *before* they show up as
loss divergence.

Chapter 3 makes DCGM the default state.

### Anti-pattern 3: "We can reproduce the run by re-running the config"

The config is one of six things a reproducibility bundle needs (seed,
config, dataset hash, tokenizer hash, framework versions, container
digest, hardware manifest — chapter 5 is the full list). A run from
six months ago on an image that no longer builds, against a data
shard that was silently re-materialized, is not reproducible from
the config alone. Every mature training platform has been burned by
this at least once; chapter 5 is how you stop being burned again.

## What this module owns vs. what it does not

The observability stack sits at the intersection of every other
module in this track. The dividing line matters because signals
without ownership become nobody's problem.

- **Distributed-training semantics** (DDP, FSDP2, TP, PP, 3D) — owned
  by mod-101. This module *emits and stores* per-rank metrics
  produced by those parallelism strategies; the strategies
  themselves are mod-101.
- **Framework internals** (Megatron, DeepSpeed, torchtitan) — owned
  by mod-102. This module treats them as sources of per-step
  scalars.
- **Data pipeline shard formats and sampler** — owned by mod-103.
  Loader-stall signals are surfaced here; the loader design is
  mod-103.
- **Scheduler and topology** — owned by mod-104. Node-level events
  (drain, quarantine, reschedule) are surfaced here; the scheduler
  plumbing is mod-104 chapter 5.
- **Fabric configuration and NCCL tuning** — owned by mod-105.
  NCCL warnings, per-collective times, and NIC-side counters are
  displayed here; the *why* they are what they are is mod-105.
- **Checkpointing, elastic training, incident classification, and
  goodput SLOs** — owned by mod-106. This module hosts the
  detectors' output; mod-106 chapter 5 owns the classification
  and mod-106 chapter 8 owns the SLO itself.
- **MFU, HFU, kernel/comm efficiency** — owned by mod-107. This
  module plots MFU on the dashboard; mod-107 chapter 1 defines it.
- **Cost accounting** — owned by mod-109. Goodput is the input;
  mod-109 turns it into a dollar figure.
- **Platform architecture and org contracts** — owned by mod-110.
  This module produces the artifacts (metadata schema, run bundle
  spec) that mod-110's cross-team contracts reference.

The one-line summary of the boundary: **other modules produce
signals; this module makes them discoverable, comparable, and
durable**.

## Summary

- Training observability is not web-service observability. The
  signals are lockstep, long-lived, and drift-oriented; the tooling
  built around them (metrics tiers, experiment tracker, DCGM
  pipeline, reproducibility bundle) is shaped by that difference.
- Three audiences read the stack: on-call platform engineers,
  ML researchers, and downstream fine-tuning / evaluation teams.
  Each wants a different tool and a different metric set; the
  stack has to serve all three from one authoritative `run_id`.
- The metric surface splits four ways: model-level scalars per
  step, system-level scalars per rank per step, continuous hardware
  telemetry, and discrete events. Each goes into a different tier;
  the join key that makes them coherent is the `run_id`.
- Goodput is the north-star metric — you cannot dashboard it
  directly, but every panel on the training-run dashboard exists
  because it explains a component of it.
- Three anti-patterns to catch in review: "trainer log is our
  observability", "we check GPU metrics on demand", "we can
  reproduce from the config alone". The rest of this module is the
  concrete fix for each.
- This module owns the *plumbing* — dashboards, metric storage,
  reproducibility bundle, metadata store. It does not own the
  distributed-training strategy (mod-101), the fabric (mod-105),
  the fault-tolerance mechanism (mod-106), or the efficiency
  metric definitions (mod-107). It makes their signals
  discoverable and durable.
