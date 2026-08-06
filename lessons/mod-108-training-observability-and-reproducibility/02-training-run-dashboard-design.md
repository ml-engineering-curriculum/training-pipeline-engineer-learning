# The Training-Run Dashboard: Metric Catalog and Panel Design

Chapter 1 said every panel has to defend its presence against goodput.
This chapter is the operational cash-out: the specific metrics that
belong on a training-run dashboard, why each one is there, how they
compose into a page an on-call can read in thirty seconds, and what
gets left off. It is the design brief exercise 1 asks you to build
against.

The dashboard is a *design artifact*, not a random pile of graphs. A
well-designed page fits on a monitor, is readable from a phone with
minor zoom, and answers the three questions from chapter 1's on-call
paragraph in the top row.

## The three rows that make up the page

Structure the dashboard as three horizontal rows, top to bottom in
order of decreasing urgency:

- **Row 1 — is the run alive?** Fast-glance panels that answer "is
  the loss finite, is the step time normal, is the world size the
  size it should be?"
- **Row 2 — is the run efficient?** MFU, per-step time distribution
  across ranks, communication overhead, GPU utilization histograms.
- **Row 3 — is the hardware healthy?** DCGM roll-ups per node:
  temperature, ECC, throttle reasons, NVLink and NIC errors.

Row 1 is what the on-call reads first. Rows 2 and 3 are what they
read when Row 1 is not obviously bad but something is off. The
researcher lives in a different tool (chapter 4); the metadata-store
consumer lives in a different tool again (chapter 7). This page is
for the on-call.

## Row 1: liveness

Six panels, each pinned to the top row.

### Loss curve (rank-0, full resolution)

The single most important panel. Plot the training loss as a function
of step (not wall-clock; step is the invariant the researcher and
the recipe are aligned on). Two rules that matter more than the
plot library:

- **Do not downsample.** A two-step loss spike is the entire
  divergence signature; a dashboard that averages loss over the
  last 1 000 steps hides it. Use a raw scatter or a `min/mean/max`
  band per bucket if you must aggregate.
- **Show validation loss on the same axes.** A crossing between
  training and validation loss is a diagnostic; two panels side by
  side hides that. Overlay them.

Rank-0 emits the loss; you do not need per-rank loss here (that's
Row 2 for silent-corruption detection).

### Gradient norm (global)

Global L2 norm of gradients per step. On a healthy run this is a
smooth curve with a slow decay over training; a spike is a chapter
6 signature. Log-scale y-axis, always. A linear-scale gradient-norm
plot is unreadable because the dynamic range across a run spans
several orders of magnitude.

### Step time (median across ranks)

Median per-step time in milliseconds, over a window of the last
`10^3` steps or so. A creeping increase in median step time is a
throughput-cliff early warning (chapter 6). Absolute value matters
less than the trend — always plot with the run's rolling mean
baseline overlaid.

### World-size sanity check

A single big number: current `world_size`. If the number does not
match the expected number for the run, something rescheduled and
you have an incident (mod-106 chapter 4 covers elastic reshape;
the *panel* is here so on-call notices the reshape happened).

### Steps since last checkpoint

How long since the last DCP save succeeded (in steps). If this
number is climbing past its usual value, the checkpoint job is
stuck and your recovery window is growing (mod-106 chapters 2 and
3 own the checkpoint plumbing; the panel is the surface that tells
on-call to check).

### The current run's `run_id` and start time

Not a metric — a header. Every dashboard link, every log query,
every metadata-store lookup uses this `run_id`. Making it visible
at the top removes the "which run am I looking at" question that
otherwise wastes the first thirty seconds of every incident.

## Row 2: efficiency

Four panels. This is where the platform engineer works during a
performance regression.

### MFU with target band

Chapter 1 of mod-107 defined MFU. Plot it per step (or per short
window), with the run's target band (e.g., "40–55% for BF16 on
Hopper without FP8" from mod-107 chapter 1) overlaid as a shaded
region. A dip below the band is a chapter 8 (mod-107 chapter 7) or
loader problem; a dip *and* a slow step time is a straggler.

Also plot HFU on the same axes when the run uses activation
checkpointing. The MFU–HFU gap is the "how much are we paying for
recompute" signal (mod-107 chapter 5).

### Per-step-time dispersion across ranks

The straggler signal from mod-106 chapter 6. Compute `max_rank_step /
mean_rank_step` per step; plot it. A healthy homogeneous cluster
sits close to `1.0`; a dispersion walking above `1.10–1.20`
sustained for many steps is a chapter 6 straggler runbook trigger.

Colour a red band above your team's threshold (`1.10` or `1.20`,
whichever your straggler detector uses). This is one of the most
diagnostic panels on the page.

### Communication time as a fraction of step time

From `torch.profiler` (or NCCL's own profiling counters) exported to
Prometheus. Plot `comm_time / step_time` per rolling window. This is
the overlap signal from mod-107 chapter 7; a rise here without a
change in step time is a good day (overlap is working), a rise
*with* a step-time rise is a communication-cliff incident.

### GPU utilization histogram

DCGM's `DCGM_FI_DEV_GPU_UTIL` per rank, aggregated. Do not plot
"average GPU utilization" — the average is meaningless because it
hides the ranks that are stalling. Plot a histogram (or a rank-by-
rank strip chart) and let on-call see the *shape* of the
distribution.

## Row 3: hardware

Four panels, roll-up-oriented. Individual DCGM-per-GPU views live
in a drill-down page linked from these; the top-level panels are
"is any of the fleet unhealthy?"

### GPU temperature top-N

Plot the top 10 hottest GPUs' temperature over time. If they
cluster together (same node, same rack), it is a facilities
issue; if they are scattered, it is per-GPU. Persistent
temperatures above the vendor throttle threshold are a chapter 6
straggler signal.

### ECC-error rate (double-bit and single-bit deltas)

Per-rank per-hour delta of DCGM's ECC counters (both single-bit
correctable and double-bit uncorrectable — mod-106 chapter 6 lists
the field IDs). Plot as a stacked bar or heatmap over the fleet.
Any DBE event is a page; a rising SBE rate on a specific GPU is a
"GPU on its way out" flag.

### NVLink / NIC error counters, per node

Per-node delta of the NVLink error counters (from DCGM) and the
NIC-side counters (from `mlnx_perf`/`ethtool -S`, exported into
Prometheus — mod-105 chapter 8 covers the tool side). Persistent
non-zero deltas indicate a fabric-side problem the on-call needs
to escalate to the fabric team.

### Throttle-reason count

DCGM's throttle-reason bitmask, aggregated across the fleet.
Plot the count of GPUs currently reporting each reason
(`THERMAL`, `POWER`, `HW_SLOWDOWN`, etc.). A cluster reporting
`THERMAL` throttling all at once is a facilities incident; a
single GPU reporting `POWER` is a per-GPU incident.

## The alert–display split

Not every metric on the dashboard deserves an alert; not every alert
metric belongs on the dashboard. The default rule:

- **On the dashboard**: everything on-call needs to *understand* the
  current run.
- **Alerting**: only the metrics whose crossings are unambiguous
  incidents — NaN loss, DBE ECC event, world-size mismatch, NCCL
  timeout, per-step-time dispersion sustained above threshold, loss
  spike above threshold sustained beyond one step.
- **On the dashboard AND alerting**: the "you want to look at it if
  it goes off" metrics — MFU dropping out of target band, checkpoint
  age growing past threshold, GPU temperature above throttle
  threshold.

The failure mode you are guarding against is *alert fatigue*: 40
pages a day for signals that are informational becomes zero pages
that get acknowledged. Alert only on decisions; display everything
else.

## Refresh cadences

A common source of dashboard confusion: two panels on the same page
with different refresh rates give the same on-call two different
"current" values. Set them explicitly:

- **Row 1 (liveness)**: refresh every `5 s`. On-call needs the
  freshest read.
- **Row 2 (efficiency)**: refresh every `15–30 s`. These are
  windowed metrics; a faster refresh gains you nothing.
- **Row 3 (hardware)**: refresh every `30–60 s`. DCGM's default
  poll interval is `1–10 s` per GPU; more frequent dashboard
  refresh than that is wasted queries against Prometheus.

Document the cadences in the dashboard's own description panel so
the reader is not left guessing.

## Where the data flows in

For orientation, the sources that feed each row:

- **Row 1**: the trainer emits scalars to the experiment tracker
  (chapter 4) and, in parallel, into Prometheus via a small
  `prometheus_client` gauge/counter set. Rank 0 owns the emission
  of shared scalars (loss, LR, world size); each rank owns its own
  per-rank scalars (step time, HBM).
- **Row 2**: Same trainer emission. MFU comes from a computed
  gauge that combines the FLOP counter (mod-107 chapter 1) with
  the step-time gauge.
- **Row 3**: `dcgm-exporter` on every training node, scraped by
  Prometheus at its configured poll interval. Chapter 3 covers the
  install and scrape configuration.

The join key is the `run_id`; it must appear as a label on every
metric so the dashboard can filter to one specific run. Prometheus'
`external_labels` and the trainer's `prometheus_client` gauges both
have to carry it. Chapter 7 formalizes the ID.

## Panel-by-panel pitfalls

A handful of specific mistakes worth naming:

- **Plotting throughput without step time.** Tokens/sec is a
  composite of `global_batch * seq_len / step_time`. If you plot
  the composite and the step time changes, you cannot tell whether
  the batch changed or the step slowed. Plot both.
- **Averaging loss over huge buckets.** Chapter 6's spike
  signature is often two to five steps wide. If your dashboard
  bucket is 1 000 steps, a 500× spike averages to a 2× bump — you
  will miss it. Downsample only for old data past the current run's
  live window; keep the last 10 000 steps at full resolution.
- **Mixing "since start" and "since last checkpoint" counters
  silently.** Both are useful; both need to be labeled. A
  "checkpoint save duration" that is really "save duration since
  process start divided by save count" is a slow-motion lie during
  a job that recently restarted.
- **A single GPU-util number for the whole fleet.** Loses the
  shape. Use a histogram or a top-N slow-rank list.
- **Per-layer gradient norms on the on-call page.** They belong on
  the researcher's tracker page (chapter 4), not on the on-call
  dashboard. There are too many of them to read at a glance.

## The dashboard as a design artifact

The dashboard is code. Grafana dashboards live in JSON; commit them
to version control alongside the training code. Two consequences:

1. **Review dashboards in code review.** A dashboard change that
   removed the ECC panel is a code review question, not a "who
   deleted the panel" post-mortem after the next incident.
2. **Templatize per-run.** A dashboard should take `run_id` as a
   template variable, not be hand-copied per run. Grafana's
   `${run_id}` variable is the mechanism; the training-side
   emission has to carry the label. Chapter 7's metadata-store row
   for the run has a link column that points at the templatized
   URL.

If you cannot render the dashboard for a historical `run_id` from
its metadata, you cannot post-mortem historical incidents; that is
one of the most common versions of anti-pattern 3 from chapter 1.

## What the dashboard is not

- **Not the experiment tracker.** The researcher's per-layer
  gradient-norm plot, the sample-output panel, the hyperparameter
  sweep view — those live in W&B / TensorBoard / MLflow (chapter
  4). Do not try to cram them into Grafana.
- **Not the run's logbook.** The append-only text file of
  operator decisions ("bumped LR down 30% at step 152 000 after
  the spike") is a chapter 5 / mod-106 chapter 7 artifact; the
  dashboard *links to it* but is not it.
- **Not the incident playbook.** mod-106 chapter 5's playbook
  tells on-call what to do; the dashboard is where the symptoms
  live. Link from every panel description to the playbook entry
  it corresponds to.

## Summary

- The dashboard is three rows: liveness (is the run alive), then
  efficiency (is it efficient), then hardware (is the fleet
  healthy). Order matters — on-call reads top to bottom.
- Row 1 minimum: loss curve, gradient norm, step-time median,
  world size, steps since last checkpoint, run header. All at
  high resolution; do not downsample the loss.
- Row 2 minimum: MFU with target band, per-step-time dispersion
  across ranks, comm-time fraction, GPU-utilization histogram.
- Row 3 minimum: temperature top-N, ECC deltas, NVLink/NIC error
  counters, throttle-reason count. Roll-up first; drill-down
  linked.
- Alerting is a separate policy from display. Alert only on
  decisions; display everything else. Set refresh cadences
  explicitly per row.
- The dashboard is code — Grafana JSON in the repo, templatized
  on `run_id`, reviewed in code review, renderable for any
  historical run. If you cannot recover a past dashboard, you
  cannot post-mortem the incident.
- The dashboard is not the tracker, is not the logbook, is not
  the playbook — it links to all three.
