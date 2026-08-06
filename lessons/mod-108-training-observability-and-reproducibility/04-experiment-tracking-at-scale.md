# Experiment Tracking at Pretraining Scale

Chapter 2's dashboard serves the on-call. This chapter is the tool
the researcher lives in: the experiment tracker. At single-workstation
scale, "install W&B and call it a day" is fine. At pretraining scale
— thousands of ranks, millions of steps, hundreds of runs in a
comparison view — that same install pattern generates enough log
volume to break the tracker, the trainer, or both. This chapter is
the design that keeps the researcher's tool useful without paying an
unreasonable throughput tax.

The three tools you will pick between (and sometimes combine):

- **Weights & Biases (W&B).** https://docs.wandb.ai/ — hosted or
  self-hosted; strong sweep-comparison UI; the tracker most
  frontier-lab research teams use.
- **TensorBoard.** https://www.tensorflow.org/tensorboard — the
  original; open-source; runs against a local `events.out.tfevents.*`
  directory; PyTorch integration via
  `torch.utils.tensorboard.SummaryWriter`. Free, minimal, and does
  not phone home.
- **MLflow.** https://mlflow.org/docs/latest/index.html — open-source;
  strong on the model-registry / experiment-metadata side; often the
  tool the fine-tuning team downstream is already using.

They are not mutually exclusive. A common configuration is TensorBoard
for the researcher's live plots, MLflow for the metadata store, and
W&B for cross-run comparisons — see chapter 7 for how the metadata
row ties them.

The tracker is *not* Prometheus. Prometheus is the systems-side metric
tier; the tracker is the researcher-side metric tier. Both are
persistent, both are indexed on `run_id`, but they answer different
questions on different cadences.

## What the tracker has to hold that Prometheus does not

Prometheus is great at numeric time series with low label cardinality
and short retention. It is bad at:

- **Rich artifacts.** Sample outputs, plots-as-images, model
  cards, per-checkpoint eval reports. Trackers can attach files
  and images to a step; Prometheus cannot.
- **Cross-run comparison views.** "Show me the loss curve for the
  five runs of hyperparameter sweep X" is one click in W&B. In
  Prometheus/Grafana, it requires bespoke templating and is
  painful.
- **Long-term run metadata.** Config, tags, notes, links. The
  tracker is the researcher's persistent record; Prometheus's
  labels are not meant to be a run's provenance store.
- **Per-layer histograms and distributions.** Weight histograms,
  gradient histograms per layer per checkpoint. Prometheus's data
  model does not support the shape; TensorBoard's `add_histogram`
  and W&B's `wandb.Histogram` do.

The clean division: **Prometheus for time-series that alertmanager
should page on; tracker for anything a researcher wants to
compare, plot, or annotate**. Both durability tiers coexist.

## Log-volume math: why "log everything" fails

Do the arithmetic before designing your logging cadence, not after.

For a run of `S = 10^6` steps and `M = 50` scalar metrics per step
(loss, LR, gradient norm, a few dozen per-layer norms), the tracker
receives `S × M = 5 × 10^7` scalar events. That is fine for any
tracker's storage.

For a run of `W = 10^4` ranks with `M_r = 10` per-rank metrics per
step, the equivalent is `S × W × M_r = 10^11` events. That is not
fine. It is 100 GB per run at 1 byte per event (impossible; a
tracker event is 100+ bytes with metadata), and it saturates any
tracker's ingest.

The two-order-of-magnitude difference is why chapter 4 is a design
chapter and not "just call `wandb.log`". You have three options and
you will use all three in combination.

### Option 1: log from rank 0 only

Loss, learning rate, and any scalar that is invariant across ranks
(shared model state) get logged by rank 0. The tracker never sees
the other ranks. Simplest pattern; sufficient for the researcher's
core plots.

```python
if dist.get_rank() == 0:
    tracker.log({
        "train/loss": loss.item(),
        "train/lr": scheduler.get_last_lr()[0],
        "train/step_time_ms": step_time_ms,
    }, step=global_step)
```

The `step` argument matters: pass the global step, not the wall
clock or the internal iteration count, because the researcher's
comparison view aligns on step.

### Option 2: reduce across ranks before logging

Per-rank metrics (per-rank step time, per-rank HBM) become
`max/min/mean/dispersion` scalars. One all-reduce per metric,
logged from rank 0. The dashboard sees the same numbers the
researcher does.

```python
step_time_tensor = torch.tensor([step_time_ms], device="cuda")
max_t = step_time_tensor.clone()
mean_t = step_time_tensor.clone()
dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
dist.all_reduce(mean_t, op=dist.ReduceOp.SUM); mean_t /= dist.get_world_size()

if dist.get_rank() == 0:
    tracker.log({
        "system/step_time_max_ms": max_t.item(),
        "system/step_time_mean_ms": mean_t.item(),
        "system/step_time_dispersion": max_t.item() / mean_t.item(),
    }, step=global_step)
```

This is the *reduction* pattern; it collapses a `W × 1` tensor to a
handful of scalars per step at a cost of one small all-reduce. Use
it for everything that has a natural aggregate summary — chapter 6's
straggler dispersion, per-rank max HBM, per-rank max temperature.

### Option 3: sub-sample the sampling cadence

Per-layer gradient norms every step multiplied by `10^2` layers is a
lot; you almost never look at every step. Log every N-th step
(`N = 10` or `N = 100`), or log a sub-sample of layers every step.

```python
LOG_EVERY = 100
if global_step % LOG_EVERY == 0 and dist.get_rank() == 0:
    per_layer_norms = {
        f"grad/layer_{i}": p.grad.norm().item()
        for i, p in enumerate(model.parameters())
    }
    tracker.log(per_layer_norms, step=global_step)
```

The trade-off is that a spike between logging points is missed by
the tracker's view. Mitigation: keep a full-resolution *scalar*
(global gradient norm, chapter 2) at every step, and the per-layer
detail sub-sampled. The full-resolution scalar catches the
spike-existence signal; you re-inspect at higher resolution offline
if you need to.

## Sharded logging: when rank 0 is not enough

For truly per-rank data (per-rank loss for silent-corruption
detection, per-rank profile traces), the reduce-to-rank-0 pattern
loses information. Two shard-friendly patterns:

### Per-rank subdirectories

Every rank writes its own `events.out.tfevents.<rank>.<host>` file
into a per-rank subdirectory under the run's root:

```
runs/<run_id>/rank_0000/events.out.tfevents.*
runs/<run_id>/rank_0001/events.out.tfevents.*
...
runs/<run_id>/rank_<N>/events.out.tfevents.*
```

TensorBoard loads all of them when pointed at the root directory
and shows them as separate runs (or a single run with rank tags,
depending on how you configure `SummaryWriter`'s `log_dir`). W&B's
group / job-type model does the same thing with `group=run_id,
job_type=f"rank_{rank}"`.

The gotcha: the shared filesystem the ranks write to has to
tolerate `W`-way concurrent open/append. mod-105 chapter 7 covers
parallel filesystems; a Lustre or WEKA mount handles this fine, a
naive S3 backend does not (S3 does not support append). If you are
S3-backed, buffer to node-local disk and periodically sync — the
`aws s3 sync` pattern.

### Chunked upload

For W&B in particular, the offline mode (`WANDB_MODE=offline`) has
each rank buffer to a local `wandb-*` directory; a background sync
process uploads chunks to the tracker. This decouples the training
loop from the tracker's ingest bandwidth. On a fabric where the
tracker is not reachable from every rank (private cluster, egress
firewall), offline mode + sync from a designated node is the only
option.

The design pattern documented at
https://docs.wandb.ai/guides/track/log/offline is the reference;
the equivalent for TensorBoard is a `SummaryWriter` writing to a
staging directory, followed by `gsutil rsync` (or the equivalent
for your cloud) to the persistent bucket.

## Offline sync patterns for restricted networks

Many production training clusters are network-isolated: the compute
fabric is on one VLAN with no direct egress to the internet, and
the researcher's laptop is on another network entirely. Trackers
that assume "you can hit `api.wandb.ai` from the training job" fail
in this environment. Three patterns that work:

### Pattern 1: on-cluster tracker

Self-host the tracker inside the cluster's network. W&B has a
self-hosted offering (W&B Server); TensorBoard runs anywhere; MLflow
is a Python + database stack you host yourself. The training job
writes to a tracker instance the researcher can reach via VPN.

The trade-off is you now operate the tracker. For a large team,
this is worth it (data sovereignty, latency, cost); for a small
team, use a hosted tracker.

### Pattern 2: proxy through an egress gateway

A single small proxy VM in a network zone that can talk to both
the training fabric and the internet forwards tracker traffic. All
`wandb`-issued HTTP calls are routed through it. Simpler to set up
than pattern 1, but the proxy is a single point of failure and a
bandwidth chokepoint at large fleet sizes.

### Pattern 3: offline mode + periodic sync

Ranks write locally; a scheduled job (cron on the head node,
Airflow, whatever) walks the offline directories and syncs them to
the hosted tracker on some cadence. The researcher gets a lagged
view — typically minutes to hours behind live. Acceptable for a
long pretraining run; less acceptable for a debug session.

The right choice depends on the network topology and the team's
operational appetite. Chapter 2's dashboard (running on Prometheus)
is independent of this choice and is what on-call uses in real
time.

## Artifacts, not just scalars

The tracker's biggest advantage over Prometheus is that it can hold
non-scalar artifacts and attach them to a step. Use it for:

- **Checkpoint pointers.** The tracker holds a URI (S3, GCS, Azure
  Blob) to the DCP checkpoint written at step X, not the checkpoint
  itself. Chapter 5's reproducibility bundle is this pointer's
  authoritative form.
- **Sample generations.** Every N steps (or every checkpoint
  boundary), dump a few dozen sample generations from the model
  and log them as a table or a text artifact. The researcher reads
  these to catch degeneration that loss alone would not surface.
- **Downstream eval reports.** When an eval job finishes against
  checkpoint X, it logs its report as an artifact against run X's
  step X. The tracker becomes the join between training and eval
  (chapter 7).
- **Config and code snapshots.** A tar of the code and config at
  run-start time, uploaded once. This is a component of chapter 5's
  reproducibility bundle. The tracker keeps it discoverable from
  the run's page.

The size limit is per-tracker; W&B and MLflow both let you register
external URIs when the artifact itself lives in object storage,
which is what you want for GB-scale checkpoints.

## Which tracker for which situation

There is no universally right answer. A defensible default matrix:

- **Frontier-lab-style pretraining, hosted W&B is acceptable.** Use
  W&B. Sweeps, artifacts, and cross-run comparisons work well out
  of the box. Egress-restricted variant: self-hosted W&B Server or
  offline-mode + sync (pattern 3).
- **Air-gapped or cost-sensitive.** Use TensorBoard for live
  plotting + MLflow for the metadata store. TensorBoard is free
  and does not phone home; MLflow's tracking server is a small
  Python service you host.
- **Model-registry / downstream-team boundary matters most.** Use
  MLflow's model registry. Chapter 7's metadata store often
  overlaps with MLflow's schema.
- **You inherited a stack.** Use what's there; don't rewrite. The
  cost of a tracker migration during a live pretraining run is
  higher than any tracker's differential UX advantage.

Whichever you pick, standardize the **run ID** and the **metric
naming convention**. Chapter 7 makes the ID formal.

## Metric naming: a schema, not a habit

The tracker's dropdown menus become impossible to navigate at
hundreds of metrics per run unless you impose a schema. A
production-grade convention:

```
train/loss
train/lr
train/gradient_norm_global

val/loss_slice_english
val/loss_slice_code

system/step_time_ms_median
system/step_time_ms_dispersion
system/gpu_util_median

hardware/temperature_max_celsius
hardware/ecc_dbe_delta_total

comm/allreduce_time_ms
comm/allgather_time_ms
```

Prefixes group metrics; the tracker's UI collapses them into
categories. Standardize the prefixes at the team level, not per
project. A new project's metrics that follow the schema slot into
the existing dashboards for free.

## Coupling to Prometheus: emit twice, cleanly

Some scalars belong in both tiers — loss and step-time in
particular. Emit twice: once to the tracker (for the researcher)
and once to `prometheus_client` (for the dashboard). Same value,
different tier. The wrong pattern is to make the trainer emit only
to the tracker and then have a separate scraper poll the tracker's
API to feed Prometheus — that adds a synchronization failure mode
you do not need.

A small helper:

```python
def log_scalar(name: str, value: float, step: int):
    if dist.get_rank() != 0:
        return
    tracker.log({name: value}, step=step)
    _prom_gauges[name].labels(run_id=RUN_ID).set(value)
```

Both tiers see the value; the trainer is not aware of either's
downstream consumers.

## Common failure modes

- **The trainer runs faster than the tracker can ingest.** W&B and
  MLflow will queue events and eventually fall behind or drop
  them. Symptom: the researcher's plot lags reality by hours. Fix:
  reduce log volume via reduction (option 2) and sub-sampling
  (option 3).
- **Rank-0 is not addressable from the tracker.** Common in air-
  gapped clusters. Fix: offline mode + sync.
- **Two runs with the same auto-generated name.** Some trackers
  assign a random human-readable name at run start (`snowy-
  meadow-42`). Two ranks with different clocks can generate the
  same name if you rely on the tracker's default. Fix: set the
  run ID explicitly from your `run_id` scheme (chapter 7).
- **The tracker becomes the source of truth for reproducibility.**
  It's a good discovery UI but a bad canonical store — deleting a
  run in the UI can wipe the artifacts. Chapter 5's bundle is the
  canonical store; the tracker is the *discovery layer*.
- **Per-layer gradient norms on the tracker page kill the render.**
  A `10^2`-layer model with per-step per-layer norms produces a
  chart with hundreds of series. Sub-sample or group by
  layer-block, not per-layer.

## Summary

- The tracker (W&B / TensorBoard / MLflow) is the researcher-tier
  metric store; it complements Prometheus, does not replace it.
- Volume math dictates the design: log `10^11` events per run and
  every tracker fails. Reduce ranks-to-scalars, sub-sample
  cadences, and shard the per-rank log streams when you must
  keep them.
- Rank-0 emission covers the shared scalars; all-reduce reductions
  cover system-level distributions; per-rank shards cover the
  cases where per-rank detail is load-bearing (loss for SDC
  detection, profile traces).
- Offline mode + periodic sync is the pattern for network-
  restricted clusters. Self-hosting the tracker is the alternative
  when the team is large enough.
- Artifacts — checkpoint URIs, sample generations, eval reports,
  config snapshots — are the tracker's real advantage over
  Prometheus. Use it for them.
- Standardize the metric-naming schema at the team level;
  standardize the run ID (chapter 7 formalizes it).
- The tracker is a discovery layer, not the canonical store; the
  reproducibility bundle in chapter 5 is the source of truth.
