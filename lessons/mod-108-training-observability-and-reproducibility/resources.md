# Resources for mod-108-training-observability-and-reproducibility

Primary docs first, then tooling, then reference schemas, then
papers. The chapters and exercises ground every technical claim in
one of the sources below; skim the framework docs as-needed while
working the exercises. The two case-study papers reward a full
read.

## Primary docs — cluster-view telemetry (chapters 1–2)

- **NVIDIA Data Center GPU Manager (DCGM).**
  https://docs.nvidia.com/datacenter/dcgm/
  Concepts, field catalog, `dcgmi` commands (`dcgmi diag`,
  `dcgmi health`), profiling APIs. The authoritative source for
  the DCGM field IDs the chapters scrape.
- **`dcgm-exporter`.** https://github.com/NVIDIA/dcgm-exporter
  DCGM → Prometheus text-format adapter. README covers the
  Kubernetes DaemonSet install, the config file syntax, and the
  default port (`:9400`).
- **NVIDIA GPU Operator (Kubernetes).**
  https://github.com/NVIDIA/gpu-operator
  How `dcgm-exporter` is normally deployed on Kubernetes
  clusters, alongside the driver DaemonSet and device plugin.
- **NVIDIA XID error reference.**
  <!-- needs-research: canonical URL is
  https://docs.nvidia.com/deploy/xid-errors/ — verify. -->
  The mapping from XID code to failure class; needed to write
  meaningful alert rules and post-mortem notes.
- **Prometheus documentation.** https://prometheus.io/docs/
  Data model, PromQL, alerting, recording rules, best
  practices. The chapter-1 naming / labeling discipline
  follows the "instrumentation" and "naming" sub-pages
  directly.
  <!-- needs-research: verify sub-page URLs for
  /docs/practices/naming/, /docs/practices/instrumentation/. -->
- **Prometheus Operator (Kubernetes).**
  https://github.com/prometheus-operator/prometheus-operator
  `ServiceMonitor` and `PodMonitor` CRDs, referenced by
  chapter 2's scrape config example.
- **Grafana documentation.**
  https://grafana.com/docs/grafana/latest/
  Dashboard variables, panel types, alerting. The eight-panel
  design in chapter 1 is expressed as Grafana panel types
  directly.
  <!-- needs-research: verify current version-agnostic docs base
  URL. -->

## Primary docs — experiment-tracker layer (chapter 3)

- **Weights & Biases logging guide.**
  https://docs.wandb.ai/guides/track/log
  The `wandb.log(...)` API, step semantics, media logging.
- **Weights & Biases logging FAQ (rate limits, offline sync).**
  https://docs.wandb.ai/guides/track/log/logging-faqs
  Rate-limit thresholds, offline mode, resumable-run behavior.
  The chapter-3 patterns are the productionized answer to the
  scenarios listed here.
- **Weights & Biases artifacts.**
  https://docs.wandb.ai/guides/artifacts
  Artifact schema, reference vs. embed, artifact versions.
  Referenced by the sharded / URI-referenced artifact pattern
  in chapter 3 and the metadata-schema discussion in chapter 6.
- **MLflow tracking.**
  https://mlflow.org/docs/latest/tracking.html
  File store vs. server-backed, `mlflow.log_metrics`,
  `run_id` re-attach semantics.
- **MLflow models (`MLmodel` file format).**
  https://mlflow.org/docs/latest/models.html
  Model flavour, signature, environment. Referenced by chapter
  6 as a smaller-scope predecessor of the training-run
  contract.
- **PyTorch TensorBoard integration.**
  https://pytorch.org/docs/stable/tensorboard.html
  `SummaryWriter`, `add_scalar`, `add_histogram`. The
  TensorBoard side of the tracker discussion in chapter 3.
- **TensorBoard project.**
  https://github.com/tensorflow/tensorboard
  Event-file format, `tensorboard` CLI, projector.

## Primary docs — reproducibility (chapter 4)

- **PyTorch reproducibility guide.**
  https://pytorch.org/docs/stable/notes/randomness.html
  Seeds, deterministic algorithms, cuDNN determinism knobs,
  the honest limits of "reproduce". The canonical reference for
  the seed set in chapter 4.
- **HuggingFace Tokenizers.**
  https://github.com/huggingface/tokenizers
  `tokenizer.json` canonical form, `AutoTokenizer.save_pretrained`
  output. The tokenizer hash in chapter 4 hashes this file.
- **HuggingFace Hub model-card metadata.**
  https://huggingface.co/docs/hub/model-cards
  The YAML-front-matter schema for model cards. Referenced by
  chapter 6 as a companion metadata schema for the eventual
  published model.
- **`nvidia-smi` reference.**
  https://developer.nvidia.com/nvidia-system-management-interface
  Command syntax, `-q -x` XML dump used in the chapter-4
  hardware manifest.
  <!-- needs-research: verify canonical URL; the tool's docs
  also live under docs.nvidia.com/deploy/. -->
- **`dcgmi diag` documentation.**
  Under https://docs.nvidia.com/datacenter/dcgm/. Diagnostic
  levels (`-r 1` short, `-r 3` extended), JSON output format
  (`--json`).

## Container / hash tooling (chapter 4 + chapter 6)

- **OCI image specification.**
  https://github.com/opencontainers/image-spec
  Canonical digest scheme (`sha256:...`), manifest format,
  content-addressable identity. This is why container digests
  are portable across registries.
- **`docker inspect`.**
  https://docs.docker.com/reference/cli/docker/inspect/
  How to resolve `RepoDigests` at runtime for the chapter-4
  container manifest.
- **`crane` (go-containerregistry).**
  https://github.com/google/go-containerregistry/tree/main/cmd/crane
  Resolve a registry digest without pulling the image;
  scriptable digest resolution for CI.
- **`skopeo`.** https://github.com/containers/skopeo
  Alternate registry client with the same functionality.
- **`sha256sum` (GNU coreutils) / `openssl dgst -sha256`.**
  Standard tooling for hashing artifact files. Referenced
  throughout chapters 4 and 6 for tokenizer, shard-manifest,
  and checkpoint-manifest hashing.

## Reference metadata schemas (chapter 6)

- **MLflow `MLmodel` format.**
  https://mlflow.org/docs/latest/models.html — see also
  https://mlflow.org/docs/latest/model-registry.html for the
  registry surface that consumes the format.
- **Weights & Biases artifact schema.**
  https://docs.wandb.ai/guides/artifacts (referenced above).
- **HuggingFace Hub model-card metadata YAML.**
  https://huggingface.co/docs/hub/model-cards.
  All three are studied as reference formats; the
  training-run contract in chapter 6 layers on top of them.

## Papers — training-observability case studies

- **Zhang, S., et al. (2022). "OPT: Open Pre-trained Transformer
  Language Models."** arXiv:2205.01068.
  Read together with the **OPT-175B logbook** (the accompanying
  training log), the definitive public record of large-scale
  training-time incidents, restarts, and run-time signatures. If
  you read one thing from this list, it is the logbook. It is
  the reason chapter 5's signature catalog is shaped the way it
  is.
- **Grattafiori, A., et al. (2024). "The Llama 3 Herd of
  Models."** Meta AI technical paper. Section 6.3 discusses
  training reliability and observability — the classification
  of hardware failures Meta ran into across the 54-day 405B
  training run.
  <!-- needs-research: verify the exact section number and
  official URL for the Llama 3 technical report; the anchor
  section is "training reliability" / hardware-failure
  breakdown. -->
- **Le Scao, T., et al. (2022). "BLOOM: A 176B-Parameter
  Open-Access Multilingual Language Model."** arXiv:2211.05100.
  Additional operational data on a public pretraining run;
  their Jean Zay run's logs are worth skimming for a second
  vantage on the same failure classes.

## Adjacent-module runbooks the signature catalog escalates to

- **mod-105 chapter 8** —
  [Diagnosing Fabric Failure Modes: A Runbook](../mod-105-networking-storage-for-training/08-fabric-failure-modes-runbook.md).
  Where signatures 3 (throughput cliff) and 4 (straggler)
  escalate on the fabric side.
- **mod-106 chapter 5** — incident-classification playbooks
  (loss spike, NaN, NCCL timeout, hardware fault,
  silent-corruption from the *rescue* side). Where signatures
  1 (divergence) and 5 (silent corruption) route for rollback
  decisions.
- **mod-103 chapter 7** —
  [The Data-Pipeline Runbook](../mod-103-training-scale-data-pipelines/07-data-pipeline-runbook.md).
  Owns the shard manifest / hash trail / dedup log / throughput
  baseline that chapter 4's reproducibility bundle consumes on
  the dataset side.

## Recommended reading order for a first pass

1. Chapter 1 of this module + Prometheus best-practice /
   naming pages (1 h).
2. Chapter 2 + DCGM API concepts page + `dcgm-exporter`
   README (2 h).
3. Chapter 3 + W&B logging FAQ + MLflow tracking overview
   (1.5 h).
4. Chapter 4 + PyTorch reproducibility guide + OCI image spec
   (skim) (2 h).
5. Chapter 5 + OPT-175B logbook (2 h) — this is the
   longest-payoff item on the list.
6. Chapter 6 + MLflow `MLmodel` docs + W&B artifacts docs
   (1.5 h).

The exercises assume you have done at least items 1–4 before
starting. Item 5 pays off during exercise 5 (run-time signature
catalog).
