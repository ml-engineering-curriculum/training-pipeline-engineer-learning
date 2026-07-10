# Hand-off Contracts with Peer Tracks

The Training Pipeline Engineer track owns the training platform.
It does not own model serving, it does not own CI/CD, it does not
own kernel authoring, it does not own registry, it does not own
security. Every one of those surfaces is a peer track's
responsibility, and every one of them either consumes something
your platform emits or produces something your platform consumes.
The interface between you is a contract, and the contract is a
concrete artifact you write down.

This chapter names the five peer contracts every training platform
owes, and specifies each as a small table of fields, types,
cadences, and on-break severities. When exercise-03 asks you to
publish these contracts, this chapter is the shape it grades
against.

Two organising principles before we get to the tables.

First: **name the direction of the arrow.** A hand-off contract is
always producer → consumer. If the arrow is ambiguous ("we and
they both use the manifest") the contract is not a contract; it
is a shared spreadsheet with a race condition. Every field in
every table below has a producer and a consumer, and only one of
them.

Second: **the on-break severity is a promise, not a hope.** If a
field's on-break severity is P0, you have committed to paging out
when it breaks. If it is P2, you have committed to a same-week
fix. Downstream teams plan against these promises. Do not publish
a P0 you cannot page for.

## Peer 1 — `fine-tuning-engineer-learning` (consumer)

The fine-tuning track is the primary *consumer* of this platform.
Their researchers submit fine-tuning jobs through your launcher
SDK, read your training-run metadata, and depend on your
checkpoint durability. Three sub-contracts.

### 1a. Training-run metadata (produced by us, consumed by them)

Every completed training run publishes a manifest. This is the
authoritative record of what the run did. Fine-tuners read it to
choose base checkpoints, reason about tokenizer compatibility, and
reproduce the environment.

| Field                | Type                              | Cadence               | On-break severity |
|----------------------|-----------------------------------|-----------------------|-------------------|
| `run_id`             | UUID                              | Emitted at submit     | P0                |
| `base_model_manifest`| URL to parent-run manifest (or `null` for pretraining) | At submit | P1 |
| `dataset_hash`       | SHA-256 of the shard manifest     | At submit             | P0                |
| `tokenizer_hash`     | SHA-256 of the tokenizer.json     | At submit             | P0                |
| `tokens_seen`        | int64 (tokens actually processed) | Updated every N steps | P1                |
| `container_digest`   | `sha256:...` of the training image| At submit             | P0                |
| `hardware_manifest`  | JSON: GPU count / gen / node IDs / fabric | At submit     | P1                |
| `final_loss`         | float64                           | At run completion     | P2                |
| `final_mfu`          | float64 (fraction)                | At run completion     | P2                |
| `checkpoint_path`    | URI to canonical last-good ckpt   | At run completion     | P0                |
| `run_status`         | enum: `success`, `failed`, `preempted`, `diverged` | Final | P1  |

Notes on why the severities look like this. `run_id`,
`dataset_hash`, `tokenizer_hash`, `container_digest`, and
`checkpoint_path` are P0 because a broken one of these silently
loses provenance — a fine-tuner cannot know what they inherited.
`hardware_manifest` and `run_status` are P1 because they are
diagnostic; a downstream team can proceed but is running blind on
reproducibility. `final_loss` / `final_mfu` are P2 because they
are informational; a wrong number is embarrassing but not
data-losing.

### 1b. Launcher-SDK API (consumed by them)

Fine-tuners submit jobs through your SDK. The SDK's `Job`
dataclass, its priority classes, and its scheduler-routing config
are the surface they write against.

| Surface                   | Owner (producer)  | Change process               | On-break severity |
|---------------------------|-------------------|------------------------------|-------------------|
| `Job` dataclass fields    | Platform          | RFC (chapter 2) + version bump | P0             |
| Priority-class semantics  | Platform          | RFC + migration window       | P1                |
| Router-config schema      | Platform          | RFC + migration window       | P2                |
| Handle-URI scheme         | Platform          | RFC (breaking)               | P1                |
| `submit()` / `status()` / `cancel()` semantics | Platform | Semver (mod-104 ch 9) | P0    |

The rule that keeps this contract stable: **breaking changes bump
the major version and land through a migration RFC (chapter 2 +
chapter 6).** Backwards-compatible additions bump the minor version
and land through the normal release process. Consumers can pin the
major version and know their `Job` YAML from six months ago will
still parse.

### 1c. Cluster + checkpoint SLA (produced by us, promised to them)

| SLO                                           | Target        | Measurement window | On-break severity   |
|-----------------------------------------------|---------------|--------------------|---------------------|
| Cluster availability (submitter-perceived)    | 99.5 %        | 30 days rolling    | P1 if < 99.0 %      |
| p99 submission latency                        | ≤ 10 s        | 24 h rolling       | P2 if breached      |
| Checkpoint durability (no loss of ckpt older than 24 h) | 100 % | Continuous        | P0 on any loss      |
| Rendezvous timeout floor (mod-104 ch 9)       | ≥ 300 s       | Continuous         | P1 if lowered w/o RFC |
| Preemption warning window                     | ≥ 120 s       | Continuous         | P1                  |

The 99.5 % availability target is a real number to defend. It maps
to roughly 3.6 hours of unavailability per month. Publish the
error budget explicitly; when you burn through it, publish that
too.

## Peer 2 — `ai-infra-ml-platform-learning` (registry / serving owner)

This peer owns the *model registry* and the serving surface. Your
platform produces a trained model artifact; they ingest it. The
critical piece: **they own the registry contract**, and your
platform emits artifacts that satisfy it.

### 2a. Trained-model artifact (produced by us)

A "trained model artifact" from this platform is a bundle, not a
single file:

| Field                  | Type                                  | Cadence           | On-break severity |
|------------------------|---------------------------------------|-------------------|-------------------|
| `weights`              | DCP shard directory + rank manifest, or consolidated `.safetensors` | Per-run, on request | P0 |
| `config.json`          | Model config (arch, dims, heads, layers) | Per-run          | P0                |
| `tokenizer/`           | Tokenizer files (spec, vocab, merges) | Per-run           | P0                |
| `run_manifest`         | Field 1a above                        | Per-run           | P0                |
| `training_args.json`   | The exact `Job` payload that submitted the run | Per-run    | P1                |
| `provenance_chain`     | Ordered list of parent `run_id`s      | Per-run           | P0                |
| `sbom.spdx.json`       | SBOM of the training image            | Per-run           | P1                |

The `provenance_chain` field is what links a pretraining run to
the fine-tuning runs that inherit from it. `ai-infra-ml-platform`
uses it to render the model-lineage graph in the registry. Break
this and their lineage graph goes blind.

The `weights` field allows two representations because different
consumers want different shapes. Serving on vLLM wants
`.safetensors`; further training wants the DCP shards. Publish
both if you can; publish DCP shards and ship a conversion utility
if you can only afford one.

### 2b. Registry ingestion contract (they own it, we satisfy it)

The registry team publishes a schema; you conform to it. The
important operational fields on the platform side:

| Field                     | Type              | Cadence          | On-break severity |
|---------------------------|-------------------|------------------|-------------------|
| `semantic_version`        | `MAJOR.MINOR.PATCH` per training regime | Per-run  | P1        |
| `provenance_run_id`       | Field 2a `run_id` | Per-run          | P0                |
| `signed_by`               | Cosign signature (mod-110 + security) | Per-run    | P0                |
| `registry_upload_status`  | enum: `pending`, `uploaded`, `failed` | Per-run  | P1                |

Semver on model artifacts is not the same as software semver. The
convention this track uses: `MAJOR` bumps for base-model changes
(new pretraining), `MINOR` for continued-pretraining / substantial
fine-tunes, `PATCH` for small SFT / DPO adjustments. Publish
whatever convention you pick; the important thing is that
`ai-infra-ml-platform` and `fine-tuning-engineer` both write
against the same convention.

## Peer 3 — `ai-infra-mlops-learning` (CI/CD post-training)

The MLOps peer owns the CI/CD pipeline that takes a trained model
through eval → canary → promote. Your platform *feeds* their
pipeline. Two sub-contracts.

### 3a. Eval-gate contract (consumed by them)

The eval gate is the first quality check a trained model hits after
your platform hands it off. It is *their* gate; your platform
publishes what it needs.

| Field                     | Type                        | Cadence                        | On-break severity |
|---------------------------|-----------------------------|---------------------------------|-------------------|
| `run_manifest_uri`        | URL to §1a manifest         | On run completion               | P0                |
| `weights_uri`             | URL to §2a weights          | On run completion               | P0                |
| `eval_dataset_pins`       | List of dataset hashes the run was trained against | On run completion | P1 |
| `expected_eval_regime`    | enum: `pretraining`, `sft`, `dpo`, `continued` | On completion | P2 |

`eval_dataset_pins` prevents an accidental eval on the same shards
the model was trained on. This is not just a hygiene concern; the
MLOps peer's canary logic uses it to select the right eval suite.

### 3b. Canary + promote hook (consumed by them)

The MLOps pipeline promotes models by name. Your platform publishes
webhooks / status endpoints that MLOps polls:

| Endpoint / event                | Direction        | Cadence           | On-break severity |
|---------------------------------|------------------|-------------------|-------------------|
| `POST /runs/{id}/complete`      | Platform → MLOps | On run success    | P1                |
| `POST /runs/{id}/failed`        | Platform → MLOps | On run failure    | P2                |
| `GET /runs/{id}/manifest`       | MLOps → Platform | Polled            | P1                |
| `POST /runs/{id}/tombstone`     | Platform → MLOps | On explicit delete| P2                |

Webhook failures should not block anything on the platform side —
the manifest is durable in your metadata service, so the MLOps
peer can always poll. Design for that: webhooks are best-effort,
polling is the authoritative path.

## Peer 4 — `ai-infra-performance-learning` (kernel authoring)

The performance peer authors kernels — FlashAttention variants,
fused adaptors, custom Triton kernels — that this platform
consumes. The contract goes both ways.

### 4a. Kernel escalation (Platform → Performance)

You escalate to the performance peer when a workload is
kernel-bound. Concrete triggers:

| Condition                                        | Diagnostic                                       | Escalation payload                              |
|--------------------------------------------------|--------------------------------------------------|-------------------------------------------------|
| Kernel is memory-bandwidth-bound at low AI       | Arithmetic intensity < HBM roofline for the layer| Layer name, shape, current impl, HBM BW measured|
| A "fused kernel is missing"                      | Two ops that could be fused show up back-to-back with a materialised intermediate | Op names, tensor shapes |
| MFU regression traced to a specific op           | mod-107 profiling identifies a hot kernel        | Profile trace, target uplift                    |
| Numerics anomaly under FP8                       | Loss spike correlated with FP8 kernel path       | Repro config, activation histograms             |

Every escalation is a ticket; every ticket has a payload with the
diagnostic evidence. The performance peer decides whether to
author, and if they author, they hand back through 4b.

### 4b. Kernel integration (Performance → Platform)

When the performance peer ships a kernel, the platform integrates
it. The hand-back contract:

| Field                     | Type                  | Cadence  | On-break severity |
|---------------------------|-----------------------|----------|-------------------|
| Kernel source + tests     | Repo / branch link    | Per-drop | P1                |
| Target framework(s)       | enum: `pytorch`, `triton`, `cuda`, `xla` | Per-drop | P2 |
| Numerics validation suite | Repro script + expected outputs | Per-drop | P0        |
| Perf baseline             | tokens/s/GPU at defined config | Per-drop | P1               |
| Kernel-version compatibility matrix | Table: framework version × CUDA version × arch | Per-drop | P1 |

The platform integrates behind a feature flag first, runs the
numerics validation and the perf baseline on the platform's own
CI (mod-108), and only then flips the flag as a default. A
numerics regression is P0 and rolls back immediately. Chapter 6's
migration-plan template applies to kernel rollouts too.

## Peer 5 — `ai-infra-security-learning` (data provenance + cluster boundary)

The security peer owns training-data provenance, checkpoint supply
chain, and the cluster boundary. Every one of these is a contract
this platform must respect, and one of the most common failure
modes at level 35 is skipping this contract. Do not skip it.

### 5a. Training-data provenance (they own, we conform to)

The security peer defines what "provenance" means for a training
dataset. Your platform publishes:

| Field                        | Type                          | Cadence         | On-break severity |
|------------------------------|-------------------------------|-----------------|-------------------|
| `dataset_license_inventory`  | Per-shard license spec        | On shard set    | P0                |
| `dedup_manifest`             | Hash-based dedup evidence     | On shard set    | P1                |
| `red_team_subset_flags`      | Bool: shard contains flagged content | On shard set | P0            |
| `pii_scrub_evidence`         | Link to scrub-pipeline run    | On shard set    | P0                |
| `crawl_source_manifest`      | URL / source-of-truth per shard | On shard set  | P1                |
| `data_freshness_window`      | ISO date range                | On shard set    | P2                |

The `dataset_license_inventory` and `red_team_subset_flags` are P0
because a broken one exposes the org to legal or safety risk. The
platform-side owner is the dataset service (mod-103); the review
happens on the security peer side. Every new shard set requires a
security review before it can be pinned as `production`.

### 5b. Checkpoint supply chain (co-owned)

| Requirement                                | Enforced by         | Cadence   | On-break severity |
|--------------------------------------------|---------------------|-----------|-------------------|
| All training-image digests are pinned      | Platform CI + admission controller | Continuous | P0 |
| All training images are signed (cosign)    | Platform CI + admission controller | Continuous | P0 |
| SBOM published for every image             | Platform CI         | Per-build | P1                |
| Checkpoint objects signed                  | Checkpoint service  | Per-save  | P1                |
| Weights export requires review             | Registry (peer 2) + security | On export | P0 |

The digest-pinning + signing pair is the load-bearing part. A
misconfigured admission controller that accepts unsigned images
undoes the whole supply chain. Verify this in the platform CI on
every deployment, not just on rollout.

### 5c. Cluster boundary contract

Who has what access, and how it rotates:

| Access                                    | Who                       | Rotation cadence | On-break severity |
|-------------------------------------------|---------------------------|------------------|-------------------|
| ssh to training nodes                     | Platform on-call rotation only | Quarterly     | P1               |
| sudo on training nodes                    | Named platform SREs (< 5) | Quarterly       | P0                |
| kubectl exec into training pods           | Platform + peer via break-glass | Per-incident | P1              |
| Read access to metadata service           | All engineering            | N/A             | P2                |
| Write access to metadata service          | Launcher SDK service account only | Yearly key rotation | P1        |
| S3 checkpoint bucket write                | Checkpoint service account only | Yearly       | P0                |
| Secret rotation cadence (in-cluster)      | 90 days                    | 90 days         | P1                |

The break-glass path for `kubectl exec` needs to be recorded and
auditable — every use of it produces an audit event that lands in
the security peer's log ingestion. Skip that and the peer track
will (correctly) reject sign-off on any subsequent RFC.

## Publishing the contracts

The contracts do not live in this chapter. They live in your
platform wiki, one page per peer track, with the tables above (and
your team's actual field types, actual URLs, actual on-break
severities). The chapter's job is to teach you what shape the
pages take.

A workable structure for the wiki page:

```
# Contract: Training Platform <-> <peer track>

## Owners
- Platform side: <team, on-call handle>
- Peer side: <team, on-call handle>
- Cross-team incident-review owner: <named human>

## Contracts
### 1. <sub-contract name>
Direction: Platform -> Peer
[table as above]

### 2. <sub-contract name>
...

## Change process
Any breaking change requires an RFC (chapter 2) with the peer's
sign-off. Non-breaking additions land through the normal
release process with 1 week notice on #train-platform-notify.

## Escalation
P0/P1 on-break: page platform + peer on-call simultaneously.
P2/P3: platform-side ticket, resolve within SLO.
```

Publish it, link it from every peer's docs, and reference it in
every RFC that touches the surface. When a peer track changes
lead, the contract is what the new lead reads on day one.

## Summary

- The training platform has five peer contracts:
  `fine-tuning-engineer-learning` (consumer),
  `ai-infra-ml-platform-learning` (registry / serving),
  `ai-infra-mlops-learning` (CI/CD),
  `ai-infra-performance-learning` (kernels),
  `ai-infra-security-learning` (provenance + boundary).
- Each contract is a set of tables of (field, type, cadence,
  on-break severity). The direction of each row is explicit:
  producer → consumer, never both.
- On-break severity is a promise. If a field is P0 you page for
  it; do not publish severities you cannot honour.
- Breaking changes to any contract go through the RFC process
  (chapter 2) with the peer track's active sign-off. Non-breaking
  additions land with notice.
- The contracts live in a wiki page per peer, linked from the
  peer's own docs. That page is what a new peer lead reads on
  day one and what your team defends in an RFC review.
- The security contract is the one most commonly under-invested.
  Digest-pinning, signed images, dataset license inventory,
  cluster-boundary access rotation — none of these are optional
  at level 35.
