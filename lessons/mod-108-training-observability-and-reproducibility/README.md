# mod-108 — Training Observability and Reproducibility at Scale

**Estimated effort:** 15 hours

The previous seven modules produced signals — MFU numbers, per-step
times, NCCL events, checkpoint durations, ECC counters. This module
is the layer that catches those signals, stores them, plots them,
and turns them into decisions. Without it, everything mod-101
through mod-107 built is measurable only in a live terminal by
whoever launched the run.

The through-line: three audiences read the observability stack —
on-call platform engineers, ML researchers, and downstream fine-
tuning / evaluation teams — and each wants a different tool from a
different tier. Chapter 1 sets the frame. Chapter 2 designs the
on-call dashboard. Chapter 3 wires the DCGM + Prometheus + Grafana
stack behind it. Chapter 4 is the researcher-side experiment tracker.
Chapter 5 is the reproducibility bundle — the seven-field artifact
that makes the run re-runnable six months later. Chapter 6 is the
runbook-side visual catalog of the five canonical run-time
signatures. Chapter 7 is the metadata store that makes all of it
discoverable by the downstream teams.

## Learning objectives

- Design a training-run dashboard covering loss curves, gradient
  norms, throughput, MFU, GPU utilization, thermals, and NCCL
  health.
- Wire DCGM into per-node GPU telemetry and Prometheus / Grafana
  for cluster-wide roll-ups.
- Integrate W&B / TensorBoard / MLflow at pretraining scale —
  including sharded logging, sub-sampled metrics, and offline
  sync patterns.
- Author a run reproducibility bundle: seed + config + dataset
  hash + tokenizer hash + framework versions + container digest
  + hardware manifest.
- Detect and diagnose the canonical run-time signatures:
  divergence, loss spike, throughput cliff, straggler, silent
  corruption.
- Design an experiment-metadata store that lets the fine-tuning-
  engineer / model-evaluation-engineer teams pull run provenance.

## Chapters

1. [Why Training Observability Is Different](01-why-training-observability-is-different.md)
   — the three audiences (on-call, researcher, downstream), the
   four-tier metric surface, goodput as the north-star, and the
   three anti-patterns this module exists to prevent.
2. [The Training-Run Dashboard: Metric Catalog and Panel Design](02-training-run-dashboard-design.md)
   — a three-row page (liveness / efficiency / hardware), the
   minimum panel set per row, the alert-vs-display split, refresh
   cadences, and the "dashboard is code" contract.
3. [DCGM and the Cluster Telemetry Stack](03-dcgm-and-the-cluster-telemetry-stack.md)
   — what DCGM is, which field IDs matter for a training run,
   `dcgm-exporter` deployment, Prometheus retention and recording
   rules, Grafana provisioning, and the named gotchas.
4. [Experiment Tracking at Pretraining Scale](04-experiment-tracking-at-scale.md)
   — W&B vs. TensorBoard vs. MLflow, the volume math that dictates
   sub-sampling and reduce-to-rank-0, sharded logging patterns,
   offline sync for restricted networks, and the artifact vs.
   scalar boundary.
5. [The Reproducibility Bundle](05-the-reproducibility-bundle.md)
   — three tiers of reproducibility, the seven required fields,
   the emit pattern (with `write_atomic` and rank-0 gather), the
   verify-at-emit and verify-for-rerun harnesses, and the team-
   level contract.
6. [Canonical Run-Time Signatures](06-canonical-run-time-signatures.md)
   — divergence, loss spike, throughput cliff, straggler, silent
   corruption. Each with curve shape, correlated metrics, a
   single-query disambiguation, and a pointer into mod-106
   chapter 5's runbook.
7. [The Experiment Metadata Store: A Contract with Downstream
   Teams](07-experiment-metadata-store-for-downstream-teams.md)
   — five tables (`run`, `checkpoint`, `dataset_snapshot`,
   `evaluation`, `incident`), the five query patterns downstream
   teams exercise, and the store-as-contract with the fine-tuning
   and evaluation roles.

## Exercises

- [exercise-01 — Training-run dashboard design](exercises/exercise-01-training-run-dashboard-design.md) (3 h)
- [exercise-02 — DCGM + Prometheus + Grafana integration](exercises/exercise-02-dcgm-prometheus-grafana-integration.md) (3 h)
- [exercise-03 — W&B / TensorBoard / MLflow at scale](exercises/exercise-03-wandb-tensorboard-mlflow-at-scale.md) (3 h)
- [exercise-04 — Reproducibility bundle spec + emitter](exercises/exercise-04-reproducibility-bundle-spec.md) (3 h)
- [exercise-05 — Run-time signature catalog](exercises/exercise-05-run-time-signature-catalog.md) (3 h)

## Labs and quizzes

- `labs/` — end-to-end observability + reproducibility lab (build
  the full three-tier stack against a small pretraining loop and
  simulate incidents against the signature catalog) lands here on
  the next autonomous cycle.
- `quizzes/` — one knowledge check lands here on the next
  autonomous cycle.

## Resources

- [resources.md](resources.md) — DCGM / NCCL / GPU-Operator docs,
  Prometheus / Grafana / MLflow / W&B / TensorBoard docs, the SDC
  and reproducibility papers, and a recommended reading order.

## How the module fits together

Chapter 1 frames the problem and names the three audiences.
Chapter 2 designs the on-call surface (dashboard); chapter 3 wires
the plumbing behind it. Chapter 4 covers the researcher-side tier
(the experiment tracker) that runs in parallel to chapter 2's
platform-side tier. Chapter 5 is the durability artifact — the
reproducibility bundle — that lets a re-run of the recipe six
months later actually reproduce. Chapter 6 is the pattern library
the on-call reads at 3 AM; each signature has a specific runbook
pointer back to mod-106 chapter 5. Chapter 7 is the metadata store
that ties every other tier together and makes it discoverable by
the downstream fine-tuning-engineer and model-evaluation-engineer
teams. The five exercises march in the same order.

## What this module deliberately does not cover

- **Distributed-training semantics** (DDP, FSDP2, TP, PP, 3D) —
  owned by mod-101. This module emits and stores metrics from
  those strategies; the strategies themselves are mod-101.
- **Framework internals** (Megatron, DeepSpeed, torchtitan) —
  owned by mod-102. This module treats them as sources of
  per-step scalars.
- **Data pipeline shard formats and sampler design** — owned by
  mod-103. Loader-stall metrics are surfaced here; the loader
  itself is mod-103.
- **Scheduler and topology** — owned by mod-104. Node-level events
  (drain, quarantine, reschedule) are surfaced here; the
  scheduler plumbing is mod-104 chapter 5.
- **Fabric configuration and NCCL tuning** — owned by mod-105.
  NCCL warnings and per-collective times are displayed; the *why*
  they are what they are is mod-105.
- **Checkpointing, elastic training, incident response, and the
  goodput SLO** — owned by mod-106. This module hosts detectors'
  output; mod-106 chapter 5 owns the classification and mod-106
  chapter 8 owns the SLO. Chapter 6 signatures point back to
  mod-106 chapter 5 runbook.
- **MFU, HFU, kernel/comm efficiency** — owned by mod-107. This
  module plots MFU on the dashboard; mod-107 chapter 1 defines
  it.
- **Cost accounting** — owned by mod-109. Goodput is the input;
  mod-109 turns it into a dollar figure.
- **Platform-architecture and cross-team org contracts** — owned
  by mod-110. This module produces the artifacts (metadata
  schema, bundle spec) that mod-110's contracts reference.
