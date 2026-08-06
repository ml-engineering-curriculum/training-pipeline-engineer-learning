# Hand-Off Contracts with Peer Platform Tracks

The training platform does not live alone. It is one of six
AICG platform tracks that together deliver a working model
lifecycle: pretraining, fine-tuning, serving/registry, CI/CD
and post-training, kernel authorship, and security. Each
adjacent track has its own on-call, its own artifacts, its own
opinions about the checkpoint format and the provenance
manifest. The training platform's outputs are the neighbouring
platforms' inputs.

This chapter is the artifact for making those interfaces
explicit. The scheduler and the storage tier are technical
plumbing; the *hand-off contract* is a written agreement between
your team and the peer team about what flows across the
boundary, in what shape, with what SLA, and who to page when it
breaks. Without the hand-off contract, every cross-team
incident becomes a re-negotiation. With it, ownership is
mechanical.

## The five peer tracks

The training-pipeline-engineer role sits at the centre of the
AICG platform stack. It has five explicit interfaces you must
own:

- **`fine-tuning-engineer`** — the primary *consumer* of the
  platform. Runs SFT, DPO/PPO/GRPO, LoRA/QLoRA, adapter
  training, continued pretraining on domain corpora. Consumes
  the platform's compute, checkpoints (base models), and
  scheduler.
- **`ai-infra-ml-platform-engineer`** — owns the *serving side
  and the model registry*. Consumes your checkpoints; publishes
  serving-ready artifacts.
- **`ai-infra-mlops-engineer`** — owns the *CI/CD pipeline and
  the post-training workflow* (eval sweeps, RLHF orchestration,
  release gating). Consumes your run metadata and checkpoints;
  produces release-tagged artifacts.
- **`ai-infra-performance-engineer`** — owns the *kernel and
  low-level optimisation surface*. Publishes attention kernels,
  fused optimisers, communication primitives; the training
  platform consumes them.
- **`ai-infra-security-engineer`** — owns the *training-data
  provenance and cluster-boundary security*. Consumes your
  provenance manifest and admission logs; produces the security
  approval that gates production runs.

Each interface is a document. Publish them all in one place
(the "platform interfaces" repo or subtree); every consuming
team's onboarding link points to them.

## The hand-off document template

Every peer-track interface follows the same eight-section
template.

```
# Interface: <this-platform> ⇄ <peer-platform>

Owners:       <our lead> (this side); <their lead> (peer side)
Effective:    from <date>; superseded on RFC-<N>
Version:      v<N>

## 1. What flows across the boundary
- Direction, artifact, format, cadence.

## 2. Format specification
- Concrete schema (JSON schema, protobuf, filesystem layout).
- Versioning rule (semver? content-addressed digest?).

## 3. SLA
- Availability, freshness, correctness, size.

## 4. Compatibility window
- How long the current format remains supported after a change.

## 5. Failure handling
- What is the ack path when a consumer receives bad data?
- What is the rollback path?

## 6. On-call and escalation
- Who to page on each side; response times.

## 7. Change process
- Which RFC template governs a change to this interface.

## 8. Sign-off
- Named individuals on both sides; date; version.
```

Two properties every interface doc must satisfy:

- **Owners on both sides.** Never a one-sided contract. The
  peer team's lead signs; their name is in the header.
- **Versioned.** Changes to the interface bump the version; the
  history is preserved. The current version is what production
  runs against.

The rest of this chapter walks each of the five interfaces at
sketch-level. A real interface doc for your platform lands at
2–4 pages; the sketches below are the *skeleton* each fills in.

## Interface 1 — Fine-Tuning Engineer (the consumer)

The fine-tuning-engineer role is your platform's *primary
consumer*. If this interface is wrong, everything downstream
breaks; if it is right, most of the other four interfaces are
easier.

### What flows across

- **Platform → fine-tuning.** Compute allocation (GPUs, queue
  time, priority), base checkpoints (available for fine-tuning),
  data loaders (shared or documented), scheduler API, logging
  and observability endpoints.
- **Fine-tuning → platform.** Submitted workloads (Job manifests,
  Slurm scripts), reproducibility bundles per run, incident
  reports for run failures, feature requests against the
  roadmap.

### Format specification

- **Workload manifest.** A specific PyTorchJob / MPIJob / Kueue
  Workload schema (or Slurm sbatch script). Documented with
  every required field.
- **Base checkpoint format.** Distributed Checkpoint (DCP) with
  mod-108 chapter 5 bundle attached; discoverable via mod-108
  chapter 7's metadata store.
- **Reproducibility bundle.** The seven-field artifact from
  mod-108 chapter 5.

### SLA

- **Queue time.** Guaranteed capacity within 5 min p50, 30 min
  p95. Borrowed capacity best-effort.
- **Checkpoint availability.** Every checkpoint listed in the
  metadata store is readable within 60 s of publication.
- **Observability endpoints.** 99.9% availability; if metrics
  are down, the run continues but no dashboard.

### Compatibility window

- **Workload manifest schema.** Two-quarter deprecation window
  for breaking changes; announced by RFC (chapter 3).
- **Checkpoint format.** Base checkpoints remain loadable in the
  format they were produced in; no forced re-materialisation
  during the compatibility window.

### Failure handling

- **Fine-tuning receives a bad checkpoint.** They report via
  incident channel; on-call verifies the checkpoint integrity
  hash (mod-106 chapter 5); if corrupted, platform restores from
  redundant copy and posts a Sev-2 postmortem.
- **Rollback.** If a platform change breaks fine-tuning
  workloads, the affected teams can pin to the previous
  workload-manifest schema version until root-cause is
  identified.

### On-call and escalation

- **Fine-tuning-side on-call** handles submission failures and
  "my job is not launching" first.
- **Platform on-call** is paged when the fine-tuning on-call
  confirms it is not user error and the runbook (mod-106
  chapter 5) requires platform intervention.

### Change process

- Workload-schema changes: RFC (chapter 3), two-quarter window.
- Checkpoint-format changes: RFC + migration plan (chapter 7).
- Priority-class / quota changes: RFC + capacity-contract
  re-sign (chapter 2).

## Interface 2 — ML-Platform Engineer (serving + registry)

The ML-platform-engineer role consumes your checkpoints and
publishes them to serving infrastructure and a model registry.
This interface is where the "trained model" hand-off happens.

### What flows across

- **Training → serving.** Model checkpoints (both mid-run and
  final), tokenizer artifacts, model config, mod-108 bundle,
  serving-format conversion metadata (if any — e.g., FP8
  quantisation calibration data).
- **Serving → training.** Serving-side incident reports that
  trace back to training decisions ("this model outputs garbage
  at long context"); feedback on checkpoint format compatibility
  with the serving stack.

### Format specification

- **Checkpoint.** DCP for training-side; either the same DCP or
  a converted safetensors artifact for serving, per the
  serving-side stack. Naming convention:
  `<run_id>/step-<N>/{model.safetensors, config.json,
  tokenizer.json, bundle.json}`. Cite the mod-108 chapter 7
  metadata schema.
- **Provenance.** Every published checkpoint carries a manifest
  linking to the training run's reproducibility bundle. The
  serving team's registry indexes on both the checkpoint digest
  and the bundle digest.

### SLA

- **Publication latency.** Trained checkpoints appear in the
  metadata store within 60 s of the mod-106 chapter 2 save
  completion.
- **Format stability.** Breaking format changes require a
  two-quarter compatibility window in which both old and new
  formats are published.

### Compatibility window

- **Checkpoint format.** Two quarters; both formats co-published
  during transition.
- **Tokenizer format.** Same, but longer (three quarters) —
  tokenizer changes are the highest-blast-radius change in the
  stack.

### Failure handling

- **Serving discovers an incompatible checkpoint.** Sev-2 on
  both sides; joint bridge call; postmortem cites the last
  interface-changing RFC.
- **Rollback.** Serving pins to the previous checkpoint; the
  training team investigates.

### On-call and escalation

- **Serving on-call** takes checkpoint-load failures.
- **Platform on-call** is paged on the joint bridge if the
  failure is platform-side.

### Change process

- Checkpoint-format changes: joint RFC (both teams sign) plus
  migration plan (chapter 7).
- Registry-schema changes: owned by the serving track; the
  training platform consumes and signs off.

## Interface 3 — MLOps Engineer (CI/CD + post-training pipeline)

The MLOps-engineer role owns the release pipeline that consumes
your run outputs and drives evaluation, RLHF, and eventual
release-gating. This interface is where a checkpoint becomes
"the release candidate".

### What flows across

- **Training → MLOps.** Run metadata (mod-108 chapter 7 schema),
  checkpoints per gate step, evaluation-trigger events, run
  incident summaries.
- **MLOps → training.** Evaluation results per checkpoint (for
  gating), release-tag notifications, requests for re-runs when
  a release fails eval.

### Format specification

- **Run metadata.** The five tables from mod-108 chapter 7
  (`run`, `checkpoint`, `dataset_snapshot`, `evaluation`,
  `incident`). Any change to the schema is a joint RFC.
- **Evaluation event.** A protobuf/JSON message with checkpoint
  digest, eval-suite id, score, decision (gate / continue).

### SLA

- **Metadata freshness.** Every completed run's `run` row is
  populated within 5 minutes of run completion; every
  checkpoint row within 60 s of save.
- **Evaluation-decision latency.** Not the training platform's
  SLA — owned by the MLOps side — but the training platform
  guarantees the artifacts required for eval are available
  within the metadata-freshness SLA.

### Compatibility window

- **Metadata schema.** One-quarter window for
  backwards-compatible additions (new nullable fields); two
  quarters for breaking changes.
- **Checkpoint format.** Same as interface 2.

### Failure handling

- **MLOps cannot find a checkpoint referenced in metadata.**
  Sev-2. Platform on-call verifies the storage tier; if
  restored, updates the metadata store; if lost, publishes a
  postmortem and revises the durability SLA.
- **Rollback.** MLOps pins to the previous release candidate;
  eval-side unaffected.

### On-call and escalation

- **MLOps on-call** takes "the release pipeline is stuck" first.
- **Platform on-call** paged on "the artifact is missing or
  malformed".

### Change process

- Metadata-schema changes: joint RFC.
- New evaluation triggers: RFC on the MLOps side; the training
  platform reviews for producer-side implications.

## Interface 4 — Performance Engineer (kernel authorship)

The performance-engineer role authors the low-level primitives
your platform runs — attention kernels (FlashAttention v2/v3),
fused optimisers, custom communication ops. This is the
*inbound* interface: the training platform is the consumer.

### What flows across

- **Performance → training.** Kernel releases (versioned Python
  wheels or Triton kernels), tested against a matrix of GPU
  SKUs, with a compatibility statement against a PyTorch and
  CUDA version range, and a measured performance envelope.
- **Training → performance.** Recipe-side integration reports
  ("kernel version 3.4 measured `μ_sus` improvement of 2.1% on
  Llama-3-8B at 128 GPUs"), production incidents traceable to
  a kernel version, feature requests against the roadmap.

### Format specification

- **Kernel release.** A versioned artifact (wheel or container
  image with digest). Semver: patch = bug fix, minor = new
  feature, major = ABI break.
- **Performance envelope.** A published measurement matrix:
  {SKU × recipe × sequence-length × precision → throughput,
  MFU, memory}. Freshness: measured at each release.

### SLA

- **Kernel release cadence.** Owned by the performance track; a
  release lands at most weekly, and requires a training-side
  ack of the compatibility statement before adoption.
- **Compatibility.** A kernel major version supported for at
  least two quarters after its release.

### Compatibility window

- **Kernel major version.** Two quarters.
- **Kernel minor version.** No compatibility break.

### Failure handling

- **Training discovers a regression.** Sev-2 on the performance
  side; training pins to the previous kernel version;
  performance team investigates and issues a patch release.
- **Kernel produces silent numerical drift.** Sev-1 on both
  sides (this is a mod-108 chapter 6 signature-5 SDC event
  masquerading as a kernel bug); joint postmortem.

### On-call and escalation

- **Training on-call** detects the regression via mod-108
  chapter 6 signatures.
- **Performance on-call** is paged with the incident summary
  and a pinned reproduction.

### Change process

- Kernel-API changes: RFC on the performance side; training
  platform signs off on the compatibility statement.
- Recipe-side adoption of a new kernel version: platform-side
  RFC if it becomes the default for all teams.

## Interface 5 — Security Engineer (data provenance + boundary)

The security-engineer role owns the training-data provenance
manifest and the cluster-boundary controls (network isolation,
image signing, secret management). This is the *governance*
interface.

### What flows across

- **Training → security.** Provenance manifest per run (dataset
  hashes, licensing metadata, opt-out compliance evidence),
  admission logs, container-image digest attestations.
- **Security → training.** Approved-image list (signed digests),
  approved-dataset list (hashes and licenses), incident
  reports for boundary violations, security-review sign-off
  gating production runs.

### Format specification

- **Provenance manifest.** JSON with fields: `run_id`,
  `dataset_snapshot_id`, `dataset_hash`, `licenses` (per-source
  license summary), `opt_out_compliance_evidence` (link to the
  filter's audit log), `image_digest`, `attestation_signature`.
- **Approved-image list.** A registry API endpoint returning
  {image name → approved digest set}.
- **Approved-dataset list.** A metadata-store query returning
  {dataset name → approved hash set}.

### SLA

- **Provenance manifest.** Emitted at run start and finalized at
  run completion; retained for the audit-retention window (per
  legal requirements — typically 7 years).
- **Approved-image / dataset lists.** Freshness: updated within
  1 business day of a security-team approval decision.

### Compatibility window

- **Manifest schema.** One-quarter window for
  backwards-compatible additions; two quarters for breaking
  changes; audit-retention rules dictate the *retention*
  window, not the *compatibility* window.

### Failure handling

- **Security discovers a run used an unapproved image or
  dataset.** Sev-1. Run halted; postmortem cites the admission-
  controller invariant (chapter 2) and identifies the bypass.
- **Rollback.** Not applicable for provenance; the manifest is
  append-only.

### On-call and escalation

- **Security on-call** takes boundary violations first;
  platform on-call paged jointly.
- **Escalation:** Sev-1 security incidents escalate to legal
  and executive leadership per the org's security policy —
  outside this chapter's scope but named as an ownership
  hand-off.

### Change process

- Manifest-schema changes: joint RFC (chapter 3); security
  team sign-off is a hard gate.
- Admission-controller invariant changes (chapter 2): platform
  RFC; security team is a required reviewer.

## The interface index

All five hand-off documents live in one directory, indexed by a
top-level table:

| Interface        | Peer team               | Version | Owner (ours) | Owner (theirs) | Last update |
|------------------|-------------------------|--------:|--------------|----------------|-------------|
| ft-consumer      | fine-tuning-engineer    |     3.2 | (name)       | (name)         | 2026-Q2     |
| serving-handoff  | ml-platform-engineer    |     2.4 | (name)       | (name)         | 2026-Q1     |
| mlops-metadata   | mlops-engineer          |     4.1 | (name)       | (name)         | 2026-Q3     |
| perf-kernels     | performance-engineer    |     1.7 | (name)       | (name)         | 2026-Q2     |
| security-provenance | security-engineer    |     2.0 | (name)       | (name)         | 2026-Q3     |

Every quarter the platform lead walks the table with the peer
leads; any interface not updated for two quarters is either
stable (fine, note it) or stale (fix it).

## Common failure modes

- **The unwritten hand-off.** "We always do X, everyone knows."
  Nobody joining the team next quarter knows. Write it.
- **Contract signed by the wrong role.** The peer team's IC
  signs, not their lead. When the IC changes teams the
  contract is orphaned. Signatures belong to leads.
- **SLA without measurement.** "Checkpoints available within
  60 s". Nobody measures it. Six months later the actual
  latency is 15 minutes and nobody noticed until the release
  pipeline broke. The SLA is only real if a metric backs it.
- **Compatibility window shorter than the peer team's release
  cycle.** If serving releases quarterly and your compatibility
  window is one quarter, they never catch up; every release is
  an emergency. Set the window against the *slowest* consumer.
- **Escalation ladder without joint runbook.** Two teams paged
  simultaneously; both wait for the other. Runbook must name
  who leads the joint call by default.
- **Interface change without a joint RFC.** One team's
  unilateral schema change breaks the other's parser. The RFC
  process (chapter 3) is where changes to a hand-off get
  sign-off from both sides.

## Summary

- The training platform has five peer-track interfaces:
  fine-tuning (consumer), ml-platform (serving), mlops (CI/CD),
  performance (kernels), security (provenance). Each is a
  written contract in an eight-section template.
- Every interface names owners on both sides (leads, not ICs),
  a format spec, an SLA with a measurement backing it, a
  compatibility window sized to the slowest consumer's release
  cycle, and a joint on-call escalation path.
- Interface changes go through the RFC process (chapter 3) and
  require sign-off from both sides. The compatibility window is
  sized in chapter 7's migration plan.
- The interface index lives in one directory; every consuming
  team's onboarding points to it. Quarterly the platform lead
  walks the index with peer leads.
- Unwritten hand-offs, missing measurements, and shorter-than-
  consumer-cycle compatibility windows are the common failure
  modes. Each has a specific fix in the template.
