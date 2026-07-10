# Experiment Tracking at Pretraining Scale

Chapter 2 wired up the cluster view. This chapter wires up the model
view — loss curves, gradient norms, learning rates, MFU, evaluation
metrics, artifact links. Most training teams first meet experiment
tracking on a laptop or a single 8×A100 node, where the tracker
"just works": `wandb.init()`, log every metric every step,
`wandb.finish()`. That same pattern, unchanged, will destroy your
tracker on a 1024-GPU pretraining run.

This chapter is about how to keep W&B / TensorBoard / MLflow useful
at pretraining scale — where a step is 0.3 seconds, a run has
thousands of ranks, network to the tracker service is slow or
absent, and the run may restart after a node failure and have to
re-attach to the same experiment.

Primary references:

- **W&B logging guide:** https://docs.wandb.ai/guides/track/log
- **W&B logging FAQ (rate limits, offline sync):**
  https://docs.wandb.ai/guides/track/log/logging-faqs
- **MLflow tracking:**
  https://mlflow.org/docs/latest/tracking.html
- **PyTorch TensorBoard integration:**
  https://pytorch.org/docs/stable/tensorboard.html
- **TensorBoard project:**
  https://github.com/tensorflow/tensorboard

## The three failure modes of naïve logging

Before the patterns, know what you are avoiding:

1. **Per-rank logging.** Every rank calls `wandb.log(...)` every
   step. A 1024-GPU run generates 1024 × steps × metrics HTTP
   posts; the tracker service either rate-limits you (W&B),
   silently drops writes (MLflow's file store), or falls behind
   and lags dashboards by minutes to hours. The tracker's *own*
   scaling problem is now your training run's problem.
2. **Per-step logging of everything.** Even from rank 0 alone,
   logging 40 metrics every step of a 100 000-step run produces
   4 million points. The tracker service handles it; the browser
   trying to render the run's page does not. Loss curves become
   opaque smears; researchers stop opening the page.
3. **Large artifact logging on every checkpoint.** Someone
   `wandb.save("checkpoint-*.pt")` on every save. On a
   400 GB checkpoint every 10 minutes, you saturate the tracker's
   artifact upload lane and consume the run's uplink budget. The
   artifact system was not designed to be a checkpoint store —
   your object store (S3 / GCS / on-prem) is. See chapter 4.

Each has a clean fix; the rest of this chapter is those fixes.

## Pattern 1: log only on rank 0

The single most important change. Everything else compounds off
this. In PyTorch:

```python
import os
import torch.distributed as dist
import wandb

def is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

# Initialize the tracker on rank 0 only.
if is_rank_zero():
    run = wandb.init(
        project="pretraining-3b",
        id=os.environ["JOB_ID"],  # see pattern 4 below
        resume="allow",
        config=cfg.model_dump(),
    )
else:
    run = None

# In the training loop:
if is_rank_zero():
    wandb.log({"loss": loss.item(),
               "grad_norm": grad_norm.item(),
               "lr": optim.param_groups[0]["lr"]},
              step=global_step)
```

Two subtleties:

- **Rank 0's metrics are not always representative.** Gradient
  norm on rank 0 is fine because gradients are all-reduced
  before you observe them (DDP / FSDP2 / Megatron all synchronize
  before the optimizer step). Per-rank quantities like
  data-loader p95 time are *not* representative from rank 0
  alone — do a `dist.all_reduce(..., op=ReduceOp.MAX)` first, or
  log them separately as a periodic dist reduction (see pattern 5).
- **Rank 0 must not block on the tracker.** Every W&B / MLflow /
  TensorBoard log call is non-blocking by default (they buffer
  in-process and flush on a background thread). Verify that
  before you turn logging on — a blocking `.log()` inside the
  step loop is a step-time regression waiting to be found.

MLflow's equivalent is `mlflow.log_metrics(...)`; TensorBoard's
is `writer.add_scalar(...)`. The rank-zero guard applies to all
three.

## Pattern 2: sub-sample metrics

Not every metric needs per-step resolution. A useful heuristic:

| Metric | Log every | Why |
|--------|-----------|-----|
| Loss | 1 step | The one signal every glance depends on |
| Learning rate | 100 steps | Schedule is smooth |
| Gradient norm | 10 steps | Enough resolution to catch a spike |
| Tokens/s/GPU | 100 steps | Averaged, per-step is noisy |
| MFU | 100 steps | Same |
| Per-rank step-time p95 | 100 steps | Reduced across ranks first |
| Evaluation loss | per eval | Every N steps, N ~ 1000 |
| Weight histograms | 10 000 steps | Huge payload, log rarely |

```python
if is_rank_zero():
    metrics = {"loss": loss.item()}
    if global_step % 10 == 0:
        metrics["grad_norm"] = grad_norm.item()
    if global_step % 100 == 0:
        metrics["lr"] = optim.param_groups[0]["lr"]
        metrics["tokens_per_gpu_per_sec"] = throughput
        metrics["mfu"] = mfu
    wandb.log(metrics, step=global_step)
```

A 100 000-step run under this schedule produces roughly 100 000
loss points, 10 000 gradient-norm points, and 1000 points for
throughput / MFU / LR — a total < 130 000 points per metric family,
which every tracker renders instantly. That is the target order of
magnitude.

## Pattern 3: sharded / offloaded artifact logging

Never `wandb.save(checkpoint_dir/**)` on a distributed checkpoint.
The DCP shards from mod-106 are already sharded across ranks; if
you point W&B at the checkpoint directory it will try to upload
every shard through the tracker's artifact lane. Two better
patterns:

- **Log a URI to the checkpoint, not the checkpoint.** The
  checkpoint lives on your object store (S3 / GCS / a POSIX
  mount) with a stable path. Log the path as a W&B artifact
  reference (`wandb.Artifact(..., type="checkpoint").add_reference(
  "s3://runs/<run_id>/step-1234/")`) or as a metadata field on
  the MLflow run. Downstream consumers (mod-106 restore, the
  registry — see chapter 6) can fetch it directly.
- **Log a hash manifest of the checkpoint, not the payload.**
  A JSON with one row per shard {shard_path, sha256, bytes}.
  Consumers can verify the checkpoint they fetched is what the
  run wrote without any tracker upload happening at all. Chapter
  4 formalizes this.

The tracker's job is to remember *what was written*; the object
store's job is to hold the bytes. Confuse the two and you will
learn what artifact-lane rate limits feel like at scale.

## Pattern 4: resumable runs on scheduler-driven restarts

mod-106 covered why elastic training reshapes the world size on a
node failure. The tracker has to survive that. Every tracker
supports a *fixed run identifier* that lets a restarted process
resume the same run instead of creating a new one:

- **W&B:** `wandb.init(id=fixed_id, resume="allow")`. The `id`
  should be your scheduler's job ID (SLURM `SLURM_JOB_ID`,
  Kubernetes job name, KubeRay actor name).
- **MLflow:** `mlflow.start_run(run_id=fixed_id)`. The `run_id`
  needs to be the MLflow-created UUID from the first attempt;
  store it in the object store alongside the checkpoint so
  restarts can find it. Alternatively use `mlflow.set_tag(
  "job_id", ...)` as a *searchable* proxy.
- **TensorBoard:** file-based, so "resume" is just writing to
  the same event-file directory. Nothing special.

Concretely:

```python
# WandB, restart-safe. JOB_ID is set by the scheduler and stays
# constant across restarts of the same job.
job_id = os.environ["SLURM_JOB_ID"]  # or KUBERNETES_JOB_NAME
if is_rank_zero():
    wandb.init(project="pretraining-3b",
               id=f"run-{job_id}",
               resume="allow",
               name=f"3B-{job_id}")
```

Without this, every torchrun elastic reshape creates a new W&B run
and your loss curve looks like a comb of independent segments
instead of one continuous line. See mod-106 chapter 4 for the
elastic-reshape flow this pattern hooks into.

## Pattern 5: offline sync for air-gapped or high-latency clusters

Many production training clusters live in a network that either
cannot reach the public tracker service (SaaS W&B) or reaches it
with painful latency (multi-region, on-prem). Every tracker has an
offline / file-backed mode:

- **W&B offline:** `WANDB_MODE=offline` or `wandb.init(mode=
  "offline")`. All metric events land in a local `wandb-run-*`
  directory. A separate `wandb sync <dir>` process from an
  egress-permitted host uploads them later.
- **MLflow file store:** point `MLFLOW_TRACKING_URI` at a POSIX
  path (e.g., `/mnt/lustre/mlflow-runs`). Every metric write is
  a file write. A separate ingest job from a networked host
  reads the file store and mirrors it into a canonical
  tracker (MLflow server, or DB-backed).
- **TensorBoard:** already file-based. Write event files to a
  shared FS; run TensorBoard behind a proxy that reads them.

The pattern to standardize on:

1. Every run writes metrics locally to a well-known path on the
   parallel filesystem, keyed by scheduler job ID.
2. A sidecar sync job on an egress host `rsync`s (or
   `wandb sync`s / MLflow ingests) the directory every N
   minutes.
3. Dashboards read from the canonical tracker; the local
   directory is the ground truth.

This decouples training-cluster network from tracker uptime.
Trackers go down; training runs do not stop.

## Pattern 6: separate the on-call dashboard from the researcher dashboard

The tracker's UI is optimized for the researcher's needs
(comparing runs, sweeping hyperparameters, looking at 40+ metrics
at once). The on-call dashboard from chapter 1 is optimized for
five-second-glance operations. They are different products; keep
them separate.

A useful split:

- The **researcher view** is the tracker's default project page.
  All the metrics land there. All the hyperparameter sweeps
  land there. It is a scroll-heavy page. No SLA on load time.
- The **on-call view** is a fixed Grafana dashboard (chapters 1
  and 2) with a link out to the tracker's per-run page for
  the currently-scheduled runs. The Grafana panels pull loss and
  MFU from the tracker's Prometheus exporter (W&B and MLflow
  both provide one) or from a hand-rolled scraper of the
  tracker's REST API.

Do not force the on-call to use the tracker UI for their
five-second glance. It has too many pixels. Do not force the
researcher to use the on-call dashboard for their comparison
work. It has too few.

## Pattern 7: the tracker as the audit log, not the metric database

At real pretraining scale a run's metric volume is small; the
tracker's real value is as the audit log:

- Which config was launched, hash-verified against the
  reproducibility bundle (chapter 4).
- Which container digest ran.
- Which dataset shards were seen (checkpointed alongside the
  DCP checkpoint, sha-linked from the tracker's metadata).
- Which final loss / eval metric the run achieved before it
  ended.
- Which downstream artifacts (checkpoints, evaluation reports,
  fine-tune inputs — chapter 6) came out of it.

Every one of those items is a *tracker tag or config field*, not
a per-step metric. When you get the tracker's role right,
`wandb.log(...)` in the step loop becomes cosmetic; the config
and tags are what actually make the run auditable.

## Bring-up checklist

Before you let a real pretraining run touch the tracker:

1. Rank-zero-only logging verified on a two-node dry run — no
   duplicate metric writes.
2. Sub-sampling schedule verified — the run's page loads in <2 s.
3. Resumable run IDs verified — kill a rank mid-run, let elastic
   reshape, confirm the same tracker run picks up.
4. Offline mode verified end-to-end — the sync job on the
   egress host ingests the offline directory.
5. Checkpoints logged as URIs / hashes, not as artifact payloads.

Only after all five are green does the tracker go into the
one-glance dashboard's model-view slot (chapter 1, panels 1-4).

## Summary

- Naïve per-rank / per-step / full-artifact logging collapses the
  tracker at pretraining scale. Every pattern below is about
  keeping the tracker healthy while still capturing the
  information a run needs.
- Log only on rank 0. Guard every tracker call with
  `dist.get_rank() == 0`.
- Sub-sample metrics: loss every step, gradient norm every 10,
  throughput / MFU every 100, weight histograms every 10 000.
- Never upload full checkpoints through the tracker's artifact
  system. Log a URI reference and a hash manifest instead. The
  bytes live in the object store.
- Use a fixed, scheduler-derived `run.id` (`SLURM_JOB_ID`, K8s
  job name) and `resume="allow"` so elastic reshapes and node
  failures re-attach to the same tracker run.
- Air-gapped / high-latency clusters use offline sync: W&B
  offline, MLflow file store, TensorBoard event files on a
  parallel FS, then a sidecar ingest job on an egress host.
- The tracker is the researcher view *and* the audit log. The
  on-call view is a Grafana dashboard (chapter 1 + 2) with
  cross-links to the tracker, not the tracker itself.
