# Resources for mod-108-training-observability-and-reproducibility

Primary vendor and standards docs first, then framework and tool
references, then the research papers that ground the module's
claims. Every chapter and exercise is grounded in this list.

## NVIDIA GPU telemetry (DCGM, GPU Operator, Xid)

- **NVIDIA Data Center GPU Manager (DCGM) documentation.**
  https://docs.nvidia.com/datacenter/dcgm/latest/ — the API
  reference, field-ID catalog, `dcgmi` CLI, and policy engine
  documentation. Pin the version you install; field IDs and
  metric names can change across releases. Primary reference for
  chapter 3 and exercise 2.
- **NVIDIA Xid error reference.**
  https://docs.nvidia.com/deploy/xid-errors/ — every kernel Xid
  code, its meaning, its severity, and the recommended action.
  Chapters 3 and 6, and mod-106 chapters 5 and 6, both refer to
  this.
- **`dcgm-exporter` on GitHub.**
  https://github.com/NVIDIA/dcgm-exporter — the Prometheus
  exporter that wraps DCGM. README, configuration reference,
  default metrics list. Chapter 3 and exercise 2.
- **NVIDIA GPU Operator for Kubernetes.**
  https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/
  — installs the driver, container toolkit, DCGM, and
  `dcgm-exporter` as a coordinated bundle. The right entry point
  for a Kubernetes-based install.
- **NVIDIA Management Library (NVML) reference.**
  https://docs.nvidia.com/deploy/nvml-api/ — the underlying
  library DCGM talks to. Useful when writing custom telemetry
  outside DCGM's field catalog.
- **NVIDIA-Resiliency-Ext for large-scale training.**
  https://github.com/NVIDIA/nvidia-resiliency-ext — open-source
  detection utilities NVIDIA ships for straggler and health
  checks on top of DCGM. Cross-referenced from mod-106 chapter 6.

## Metric storage (Prometheus, long-term stores)

- **Prometheus documentation.** https://prometheus.io/docs/ — the
  canonical reference. Read the scrape-config, recording-rules,
  storage/retention, and Alertmanager chapters. Chapter 3 and
  exercise 2.
- **Prometheus Pushgateway.**
  https://github.com/prometheus/pushgateway — used by short-lived
  ranks or ranks not directly addressable from Prometheus.
- **Grafana Mimir.** https://grafana.com/docs/mimir/latest/ — the
  long-term Prometheus-compatible store from Grafana Labs.
- **Cortex.** https://cortexmetrics.io/ — a multi-tenant
  long-term store for Prometheus metrics.
- **Thanos.** https://thanos.io/ — another long-term Prometheus
  store; use whichever your platform team standardizes on.
- **OpenTelemetry Collector.** https://opentelemetry.io/docs/collector/
  — the buffering / aggregating collector used in chapter 3's
  metric-loss-during-incident mitigation.

## Grafana

- **Grafana documentation.** https://grafana.com/docs/ — the
  reference for dashboards, provisioning, alerting, template
  variables, and the data-source layer. Chapters 2 and 3.
- **Grafana Operator (for Kubernetes).**
  https://github.com/grafana/grafana-operator — declarative
  Grafana + dashboards + data-sources for Kubernetes clusters.
- **Grafana as code — Grafonnet.**
  https://github.com/grafana/grafonnet — the Jsonnet library for
  authoring Grafana dashboards in code; use whichever
  as-code approach you prefer, but pick one.

## Experiment trackers

- **Weights & Biases documentation.**
  https://docs.wandb.ai/ — hosted or self-hosted; artifacts,
  sweeps, groups, `wandb.log` reference. Chapter 4 and exercise
  3.
- **W&B Server (self-hosted) documentation.**
  https://docs.wandb.ai/guides/hosting — for air-gapped and
  cost-controlled deployments.
- **W&B offline mode.**
  https://docs.wandb.ai/guides/track/log/offline — the pattern
  chapter 4 recommends for restricted-network training clusters.
- **TensorBoard documentation.**
  https://www.tensorflow.org/tensorboard — open-source; free;
  runs against local event files; integrates with PyTorch via
  `torch.utils.tensorboard.SummaryWriter`.
- **PyTorch TensorBoard integration.**
  https://pytorch.org/docs/stable/tensorboard.html — the
  `SummaryWriter` API reference.
- **MLflow documentation.**
  https://mlflow.org/docs/latest/index.html — open-source
  experiment tracking + model registry. Chapter 4 and chapter 7
  both reference it.
- **MLflow tracking server backend stores.**
  https://mlflow.org/docs/latest/tracking.html#backend-stores —
  the tables MLflow uses, useful when layering chapter 7's
  metadata schema on top of an MLflow installation.

## Reproducibility (framework docs and standards)

- **PyTorch reproducibility notes.**
  https://pytorch.org/docs/stable/notes/randomness.html — the
  canonical list of RNG sources and non-determinism sources in
  PyTorch. Chapter 5 tier definitions and exercise 4.
- **NVIDIA NCCL documentation.**
  https://docs.nvidia.com/deeplearning/nccl/ — includes the
  determinism-relevant environment variables (`NCCL_ALGO`,
  `NCCL_PROTO`) and the note on non-associativity of collective
  reductions. Chapter 5 discussion of tier 1 vs. tier 2.
- **NVIDIA cuDNN reproducibility documentation.**
  https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html
  — cuDNN's `deterministic` flag and its performance
  implications.
- **The ACM Artifact Review and Badging policy.**
  https://www.acm.org/publications/policies/artifact-review-and-badging-current
  — background for what "reproducible" means as a formal
  discipline; the tier framing in chapter 5 corresponds to the
  community's "reproducible" / "replicable" / "results verified"
  tiers.
- **Papers With Code — reproducibility checklist and ML
  Reproducibility Challenge archive.**
  https://paperswithcode.com/rc2022 (annual editions —
  https://paperswithcode.com/rc2023, https://paperswithcode.com/rc2024,
  etc.) — worked examples of what a reproducibility bundle needs
  to contain in practice.

## Container image digests and content addressing

- **Open Container Initiative (OCI) image specification.**
  https://github.com/opencontainers/image-spec — the canonical
  reference for image manifests, digests, and content
  addressability. Chapter 5 field 6 references it.
- **`crane` CLI documentation.**
  https://github.com/google/go-containerregistry/tree/main/cmd/crane
  — the tool for inspecting registry manifests and resolving tag
  to digest.
- **`docker inspect` reference.**
  https://docs.docker.com/reference/cli/docker/image/inspect/ —
  the digest field.

## Training-side data / sampler references

- **PyTorch DataLoader reproducibility.**
  https://pytorch.org/docs/stable/data.html#data-loading-order-and-sample-order
  — the `generator` and `worker_init_fn` mechanics.
- **PyTorch `DistributedSampler`.**
  https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
  — the `seed` argument and its use in chapter 5's data-hash
  discussion.

## Silent data corruption

- **Dixit, H., et al. (2021). "Silent Data Corruptions at Scale."**
  https://arxiv.org/abs/2102.11245 — Meta's paper on SDC in
  production. Primary reference for chapter 6's signature 5.
- **Hochschild, P. H., et al. (2021). "Cores that don't count."**
  ACM HotOS 2021, https://dl.acm.org/doi/10.1145/3458336.3465297
  — Google's paper on CPU-side SDC; the reasoning transfers
  directly to GPU compute-unit SDC.
- **NVIDIA on H100 / Hopper silent-data-corruption diagnostics.**
  Consult the DCGM release notes for the field IDs and diagnostic
  tests specific to the hardware generation. Cross-referenced
  from mod-106 chapter 5.

## Large-model training logbooks and the incident taxonomy

- **Zhang, S., et al. (2022). "OPT: Open Pre-trained Transformer
  Language Models."** https://arxiv.org/abs/2205.01068 with the
  OPT-175B chronicles at
  https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf
  — the canonical logbook and the reference format for the
  chapter 5 / chapter 7 logbook artifact.
- **Grattafiori, A., et al. (2024). "The Llama 3 Herd of Models."**
  https://arxiv.org/abs/2407.21783 — section 3.3.2 documents the
  interruption taxonomy on a modern H100 pretraining run. The
  chapter 6 signatures are grounded in the categories this
  section names.
- **Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling
  with Pathways."** https://arxiv.org/abs/2204.02311 — introduced
  "goodput" (§5) and documents PaLM's loss-spike experience
  (appendix F). Chapter 1's goodput framing and chapter 6's
  signature 1.
- **BLOOM training logbook.**
  https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml
  — the BLOOM training team's public log; another data point for
  the incident-shape reference set.

## Related pointers into other modules of this track

- **mod-105 chapter 8** — fabric-side runbook that chapter 3's
  NIC-error and chapter 6's throughput-cliff signatures escalate
  into.
- **mod-106 chapter 5** — incident classification (five classes)
  that chapter 6's signatures map to. If you have not read it,
  read it before exercise 5.
- **mod-106 chapter 6** — straggler and silent-corruption
  detection primitives; the detectors emit into this module's
  observability tier.
- **mod-106 chapter 7** — OPT-175B / Llama 3 logbook analysis;
  the source for chapter 5's logbook artifact format.
- **mod-107 chapter 1** — MFU definition; the chapter 2
  dashboard panel is a direct consumer.

## Recommended reading order for a first pass

1. Chapter 1 of this module + section 3.3.2 of the Llama 3 paper
   (1.5 h). Enough to understand why the three-audience framing
   matters.
2. Chapter 2 + the Grafana docs' "dashboards" and "template
   variables" chapters (1.5 h).
3. Chapter 3 + the DCGM API reference (skim field-ID sections) +
   the `dcgm-exporter` README (2.5 h). Enough to run exercise 2.
4. Chapter 4 + the W&B or TensorBoard docs (whichever your
   organization uses) + the MLflow tracking docs (2.5 h). Enough
   to run exercise 3.
5. Chapter 5 + the PyTorch reproducibility notes + the OCI image
   spec's manifest chapter (2 h). Enough to run exercise 4.
6. Chapter 6 + skim the OPT-175B chronicles and the Llama 3
   incident section + skim the Dixit et al. SDC paper (2.5 h).
   Enough to run exercise 5.
7. Chapter 7 + the MLflow tracking-store backend reference (1.5
   h). Enough to design the metadata store's schema for your
   team.

The exercises assume the reading in the same order; do them in
sequence.
