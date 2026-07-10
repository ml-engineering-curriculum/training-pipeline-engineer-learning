# mod-108 — Training Observability and Reproducibility at Scale

**Estimated effort:** 15 hours

mod-105 taught you what a healthy fabric looks like. mod-106 taught
you how to catch a checkpoint when the fabric misbehaves. mod-107
taught you how to measure and improve MFU. This module is the
observability and reproducibility surface that ties all three
together: the dashboards your on-call watches, the experiment
tracker your researchers use, the reproducibility bundle every
completed run emits, and the metadata contract that hands the run
off to the fine-tuning-engineer, model-evaluation-engineer, and
model-registry peer tracks.

By the end of the module you should be able to (a) design a
training-run dashboard covering loss, gradient norm, throughput,
MFU, per-rank step time, GPU utilization, thermals, and NCCL
health that answers three questions in five seconds, (b) wire DCGM
into per-node GPU telemetry and roll it up via Prometheus +
Grafana without a cardinality explosion, (c) integrate W&B /
TensorBoard / MLflow at pretraining scale using rank-0 logging,
metric sub-sampling, sharded-artifact patterns, resumable run IDs,
and offline sync, (d) author a full reproducibility bundle for a
run — seed set, config, dataset hash, tokenizer hash, framework
versions, container digest, hardware manifest — that a peer team
can verify without your help, (e) recognize the five canonical
run-time signatures (divergence, loss spike, throughput cliff,
straggler, silent corruption) across the panels you built, and
(f) publish a `training_run.v1.json` metadata contract that
downstream fine-tune, evaluation, and registry consumers code
against.

## Learning objectives

- Design a training-run dashboard covering loss curves, gradient
  norms, throughput, MFU, GPU utilization, thermals, and NCCL
  health.
- Wire DCGM into per-node GPU telemetry and Prometheus / Grafana
  for cluster-wide roll-ups.
- Integrate W&B / TensorBoard / MLflow at pretraining scale —
  including sharded logging, sub-sampled metrics, and offline sync
  patterns.
- Author a run reproducibility bundle: seed + config + dataset
  hash + tokenizer hash + framework versions + container digest +
  hardware manifest.
- Detect and diagnose the canonical run-time signatures:
  divergence, loss spike, throughput cliff, straggler,
  silent-corruption.
- Design an experiment-metadata store that lets the
  fine-tuning-engineer / model-evaluation-engineer teams pull run
  provenance.

## Chapters

1. [The Observability Mental Model and the One-Glance Dashboard](01-observability-mental-model-and-dashboard-design.md) —
   the two axes (model view / cluster view), the three glance
   questions the top screen has to answer in five seconds, the
   eight-panel budget, and the naming / label discipline that
   keeps the stack scalable.
2. [DCGM, dcgm-exporter, Prometheus, and Grafana](02-dcgm-prometheus-grafana.md) —
   the cluster-view stack. Which DCGM fields to scrape, how to
   deploy `dcgm-exporter` on Kubernetes or SLURM, how to control
   cardinality, and the four standing alerts every training
   cluster runs (persistent low SM util, XID, HBM over 90 °C,
   uncorrectable ECC).
3. [Experiment Tracking at Pretraining Scale](03-experiment-tracking-at-pretraining-scale.md) —
   the model-view stack. Rank-0-only logging, metric
   sub-sampling, sharded / URI-referenced artifact patterns,
   resumable run IDs across elastic reshapes, offline sync for
   air-gapped clusters, and the researcher-view / on-call-view
   split.
4. [The Reproducibility Bundle](04-reproducibility-bundle-spec.md) —
   what "reproduce" means (bitwise / statistical / provenance),
   the bundle contents, the `run_manifest.json` schema, the full
   seed set, dataset / tokenizer hashes, container-digest
   pinning (OCI), and the hardware manifest (`nvidia-smi -q -x`,
   `dcgmi diag`, NCCL topology dump).
5. [The Run-Time Signature Catalog](05-run-time-signature-catalog.md) —
   the five signatures you learn to read: divergence, loss
   spike, throughput cliff, straggler, silent corruption. For
   each: which panels move, the discriminating panel, first
   mitigation, escalation to mod-105 chapter 8 or mod-106
   chapter 5.
6. [The Hand-Off Metadata Contract](06-hand-off-metadata-contract.md) —
   the `training_run.v1.json` contract that leaves this track.
   Storage protocol, consumer contracts (fine-tune, evaluation,
   registry), schema evolution rules, MLflow's `MLmodel` and
   W&B artifact schemas as reference formats.

## Exercises

- [exercise-01 — Training-run dashboard design](exercises/exercise-01-training-run-dashboard-design.md) (3 h)
- [exercise-02 — DCGM + Prometheus + Grafana integration](exercises/exercise-02-dcgm-prometheus-grafana-integration.md) (3 h)
- [exercise-03 — W&B / TensorBoard / MLflow at scale](exercises/exercise-03-wandb-tensorboard-mlflow-at-scale.md) (3 h)
- [exercise-04 — Reproducibility bundle spec](exercises/exercise-04-reproducibility-bundle-spec.md) (3 h)
- [exercise-05 — Run-time signature catalog](exercises/exercise-05-run-time-signature-catalog.md) (3 h)

## Labs and quizzes

- `labs/` — a `lab-01` hand-off metadata contract exercise (design
  a `training_run.v1.json` schema, wire it into a run, verify a
  downstream consumer parses it) lands here on the next
  autonomous cycle.
- `quizzes/` — one knowledge check lands here on the next
  autonomous cycle.

## Resources

- [resources.md](resources.md) — primary docs (NVIDIA DCGM +
  dcgm-exporter, Prometheus + Grafana, W&B / TensorBoard /
  MLflow, PyTorch reproducibility guide), container / hash
  tooling (OCI image spec, `docker inspect`, `crane`),
  recommended metadata schemas (MLflow `MLmodel`, W&B artifacts,
  HuggingFace Hub model-card metadata), case-study papers (OPT-175B
  logbook, Llama 3), and a recommended reading order.

## How the module fits together

Chapter 1 fixes the mental model — two axes, three glance
questions, eight-panel top screen. Chapter 2 wires the cluster
half of that dashboard (DCGM → Prometheus → Grafana). Chapter 3
wires the model half (W&B / TensorBoard / MLflow, tuned to survive
pretraining scale). Chapter 4 formalizes the reproducibility
bundle every completed run emits, so a run can be verified and
re-derived after the fact. Chapter 5 teaches you to read the
dashboard as a set of *signatures* — patterns that map to
specific incident classes and specific downstream runbooks.
Chapter 6 exports the whole surface as a machine-readable
contract that the fine-tune, evaluation, and registry peer tracks
consume. The exercises march in that order and the lab exercises
the full end-to-end.

## What this module deliberately does not cover

- **Parallelism strategy and the collective-cost model** — owned
  by mod-101. This module *uses* the collectives to derive comm
  time on panel 7 but does not teach how they are chosen.
- **Framework internals** (FSDP2 / DeepSpeed / Megatron internals)
  — owned by mod-102.
- **Data-pipeline manifests and shard-hash trails** — owned by
  mod-103. This module *consumes* mod-103's shard manifest as
  the dataset side of the reproducibility bundle.
- **Scheduler and launcher plumbing** — owned by mod-104. This
  module *uses* the scheduler's job ID as the resumable-run
  identifier.
- **Fabric failure-mode diagnosis** (NCCL timeout, silent NIC
  drop, congestion tree, PXN mis-config) — owned by mod-105
  chapter 8. The run-time signature catalog in chapter 5
  *escalates to* that runbook.
- **Checkpointing, DCP, elastic reshape, and incident-driven
  rollback** — owned by mod-106. The reproducibility bundle
  *co-locates* with mod-106's checkpoints under the run prefix;
  the signature catalog *fires* the rollback decision, but
  mod-106 chapter 5 owns how the rollback is done.
- **MFU engineering itself** — owned by mod-107. This module owns
  the *measurement* of MFU on panel 4; mod-107 owns how to
  raise it.
- **Cost, capacity, and cluster economics** — owned by mod-109.
  This module produces the metrics mod-109 turns into dollar
  figures.
- **Multi-tenant platform architecture and RFC-level design**
  — owned by mod-110.
