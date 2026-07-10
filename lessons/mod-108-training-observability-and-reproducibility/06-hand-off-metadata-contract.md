# The Hand-Off Metadata Contract

Chapters 1–5 built the training-side observability and
reproducibility surface. This chapter is where that surface
becomes a *contract* — the machine-readable interface between
this track (which owns training) and the peer tracks that
consume its output. Two consumers matter most:

- **`fine-tuning-engineer-learning`.** The fine-tune team
  starts from a base model and needs to know exactly what it is
  — provenance, tokenizer, tokens seen, container to reproduce
  its behaviour, final training loss.
- **`model-evaluation-engineer-learning`** and
  **`ai-eval-engineer-learning`.** The evaluation team needs
  a stable identifier for the trained model plus the hardware
  and container manifest to reproduce its evaluation.

And one platform surface:

- **`ai-infra-ml-platform-learning`.** The model-registry /
  serving track that will eventually catalog the trained model
  for downstream inference. Its ingestion contract has to line
  up with what this module produces.

The reproducibility bundle from chapter 4 has all the raw
material. This chapter turns it into a stable, versioned
contract those consumers can code against.

Primary references:

- **MLflow `MLmodel` file format:**
  https://mlflow.org/docs/latest/models.html
- **W&B artifact schema:**
  https://docs.wandb.ai/guides/artifacts
- **HuggingFace Hub model-card metadata:**
  https://huggingface.co/docs/hub/model-cards
- **OCI image specification:**
  https://github.com/opencontainers/image-spec

## Why a contract, not just a bundle

The chapter-4 bundle is *complete* — it has every fact about the
run. Downstream consumers don't need every fact; they need a
*small, stable* subset in a *predictable* shape. Without a
contract:

- The fine-tune team writes ad-hoc parsers over `run_manifest.json`
  and breaks when you add a field.
- The evaluation team hard-codes S3 paths and breaks when you
  restructure the object-store prefix scheme.
- The registry team asks you to email the bundle every time.

With a contract, each consumer reads a stable document that says
"here is what you can rely on, forever, at this schema version;
here is what may change in the next version".

## The `training_run.v1.json` schema

The contract document is a single JSON per completed training
run, written to a fixed path in the object store:

```
s3://runs/<run_id>/training_run.v1.json
```

Its schema, in prose:

```json
{
  "$schema": "https://curriculum.example.org/schemas/training_run.v1.json",
  "schema_version": "training_run.v1",

  "run_id": "3b-pretrain-2026-07-09-1234",
  "produced_by": "training-pipeline-engineer",
  "produced_at": "2026-07-11T02:19:00Z",

  "model": {
    "family": "llama-3-style-decoder",
    "parameter_count": 3100000000,
    "architecture": {
      "layers": 32,
      "hidden_size": 3072,
      "attention_heads": 24,
      "kv_heads": 8,
      "context_length": 8192,
      "vocab_size": 128256
    },
    "config_sha256": "…",
    "config_uri": "s3://runs/3b-pretrain-2026-07-09-1234/config/training.yaml"
  },

  "tokenizer": {
    "family": "llama-3-bpe",
    "sha256": "…",
    "uri": "s3://runs/3b-pretrain-2026-07-09-1234/data/tokenizer.json"
  },

  "training_data": {
    "shard_manifest_uri": "s3://runs/3b-pretrain-2026-07-09-1234/data/shard_manifest.json",
    "shard_manifest_sha256": "…",
    "tokens_seen": 300000000000,
    "epochs": 1.03,
    "provenance_note": "See mod-103 chapter 7 shard-manifest schema."
  },

  "training": {
    "framework": "torchtitan-fsdp2-tp",
    "parallelism": {"dp": 8, "tp": 8, "pp": 1},
    "container": {
      "image": "nvcr.io/nvidia/pytorch:24.05-py3",
      "digest": "sha256:…"
    },
    "hardware": {
      "gpu": "H100-SXM5-80GB",
      "nodes": 64,
      "gpus_per_node": 8,
      "nccl_topology_uri": "s3://runs/3b-pretrain-2026-07-09-1234/hardware/topo.xml"
    },
    "seeds_uri": "s3://runs/3b-pretrain-2026-07-09-1234/seeds.json"
  },

  "checkpoint": {
    "type": "torch-distributed-checkpoint",
    "step": 100000,
    "uri": "s3://runs/3b-pretrain-2026-07-09-1234/checkpoints/step-100000/",
    "manifest_uri": "s3://runs/3b-pretrain-2026-07-09-1234/checkpoints/step-100000/manifest.json",
    "manifest_sha256": "…",
    "bytes": 12800000000
  },

  "metrics": {
    "final_train_loss": 1.834,
    "final_eval_loss": {"c4_val": 2.041, "wikitext_val": 2.187},
    "mfu_p50": 0.44,
    "wallclock_hours": 36.1
  },

  "artifacts": {
    "run_manifest_uri": "s3://runs/3b-pretrain-2026-07-09-1234/run_manifest.json",
    "tracker_run_url": "https://wandb.ai/acme/pretrain/runs/3b-pretrain-2026-07-09-1234"
  },

  "consumer_pointers": {
    "fine_tune_ready": true,
    "eval_ready": true,
    "registry_ingestible": true
  }
}
```

The contract is deliberately smaller than the full
`run_manifest.json`. Consumers get the pointers they need and
follow the URIs into the bundle for anything deeper. The full
bundle is the source of truth; the contract is the index.

## What each consumer promises to consume

Each peer track is on the hook for a specific subset of the
contract.

### The fine-tuning-engineer contract

The fine-tune consumer needs, at minimum:

- `model.config_uri` and `model.config_sha256` — same
  architecture at fine-tune time as at pretraining.
- `tokenizer.uri` and `tokenizer.sha256` — same tokenizer at
  fine-tune time.
- `checkpoint.uri` and `checkpoint.manifest_sha256` — the base
  weights to load, verifiable on fetch.
- `training.container.digest` — a container that can load the
  checkpoint (same framework major version, same NCCL). The
  fine-tune track may pin to a *different* container digest at
  fine-tune time, but it must be *compatible* with the training
  container. That compatibility check is a boundary condition
  between the two tracks.
- `training_data.tokens_seen` — needed by the fine-tune team's
  LR-schedule and dataset-mixing decisions.

The fine-tune track's contract is documented in
`fine-tuning-engineer-learning`. This module's promise: every
completed pretraining run publishes a `training_run.v1.json`
with those fields populated before the run is marked
`fine_tune_ready = true`.

### The model-evaluation-engineer contract

The evaluation consumer needs:

- `run_id` and `checkpoint.uri` — stable identity for the model
  under test.
- `tokenizer.uri` — evaluation loss requires the same
  tokenization.
- `training.container.digest` and `training.hardware.gpu` — to
  reproduce numerical behaviour if evaluation is done on the
  same fleet, or to document deltas if it is done elsewhere.
- `metrics.final_eval_loss` — the baseline the evaluation team
  compares its own results to.

Two subtleties. First, the evaluation team may run on different
hardware than training; the contract records what training used
so any behavioural delta is diagnosable. Second, evaluation
often needs the *tokenizer* alone without loading the
checkpoint (e.g., recomputing corpus statistics); a clean
tokenizer URI unblocks that.

### The ai-infra-ml-platform (registry) contract

The registry consumer ingests the contract into the model
registry. Its needs:

- `model.family`, `model.parameter_count`, `model.architecture`
  — the searchable / filterable model card fields.
- `checkpoint.uri` and `checkpoint.manifest_sha256` — the
  registry's own object-store pointer with integrity.
- `training.container.digest` — the reproducibility pin.
- `metrics.*` — the values that show up on the registry's
  model card.

Registry ingestion should be automatic once
`consumer_pointers.registry_ingestible = true`. The training
track sets that flag; the registry track polls for it.

## Storage protocol

Two rules on the object-store side, no exceptions:

1. **One run, one prefix.** `s3://runs/<run_id>/` (or the
   platform-equivalent path) contains everything about the run
   — bundle, checkpoint, metrics, logs. Nothing is stored
   outside the prefix. `run_id` is immutable; you never "move"
   a run.
2. **`training_run.v1.json` at the prefix root is the entry
   point.** Consumers list-by-prefix or head-key on the
   canonical path; they do not have to know the internal
   directory layout. If you restructure the internal layout in
   v2, the `training_run.v2.json` file changes; the entry-point
   convention does not.

Together those two rules mean the fine-tune / eval / registry
tracks can code against the contract without any coupling to
your internal object-store layout. You get to reorganize the
bundle whenever you want; they never notice.

## Versioning and evolution

The schema version is in the file name and inside the document
(`"schema_version": "training_run.v1"`). Evolution rules:

- **Additive changes are v1-compatible.** New optional fields
  in v1 are legal as long as every existing consumer is
  ignoring unknown fields (spec them to). Add fields freely
  when a downstream track needs a new pointer.
- **Removing or renaming fields requires a new schema
  version.** Publish `training_run.v2.json` alongside
  `training_run.v1.json` for a compatibility window (multiple
  quarters). Consumers migrate on their own schedule.
- **Never mutate a published contract file.** If a run's
  contract has a mistake, publish a corrected file with a new
  `run_id` (`<old_run_id>-corrected-1`) and mark the old one
  `status: "superseded"`. No in-place edits, ever. This is what
  keeps audits meaningful.

Chapter 4's `run_manifest.json` is a superset of the contract
for use *inside* this track. The contract is what leaves this
track. Keep both.

## MLmodel and W&B artifacts as reference formats

You are not the first shop to solve this problem. Two schemas
worth studying:

- **MLflow's `MLmodel` file.** Stores model flavour, signature,
  runtime versions, and a pointer to the underlying artifact.
  It is a much smaller surface than a training-run contract,
  but the pattern (a single YAML at the root of a directory
  containing pointers to the payload) is the same. See
  https://mlflow.org/docs/latest/models.html.
- **W&B artifact metadata.** W&B's artifact system stores a
  document per artifact version with typed metadata; every
  artifact has a hash-verifiable manifest. Study the schema in
  https://docs.wandb.ai/guides/artifacts and steal what
  applies.

Neither replaces the training-run contract because both are
designed for the model / artifact scale rather than the
training-run scale (they don't natively carry hardware manifest,
tokens seen, or a shard-manifest hash). The contract this
chapter defines *layers on top of* those systems: the contract
lives in the object store, MLflow / W&B carry the pointer to
the contract as their metadata.

## What a broken contract looks like

Two failure modes to defend against:

- **Field drift.** Someone renames `training_data.tokens_seen`
  to `training_data.tokens_processed` in the training code. The
  fine-tune team's parser breaks. Fix: schema validation runs
  in the run-manifest emission step and fails the run finalize
  if the contract is invalid. Use a JSON Schema validator on
  every produced document.
- **URI rot.** A run's `checkpoint.uri` points to a bucket that
  gets lifecycle-deleted 90 days later. Contract still validates;
  the pointer is dead. Fix: object-lifecycle rules that hold
  every referenced object as long as any published contract
  points at it, and periodic contract-validation sweeps that
  HEAD every URI and flag broken pointers.

Both failure modes are "the contract said something and the
world changed under it". Guard both with automation, not
convention.

## Bring-up checklist

The contract-emission code path every training run runs before
being marked complete:

1. `run_manifest.json` produced (chapter 4).
2. `training_run.v1.json` derived from `run_manifest.json`,
   validated against the schema.
3. Both files uploaded to the run prefix; the contract at the
   canonical root path.
4. The tracker's run entry updated with the contract URI.
5. `consumer_pointers.*` flags set based on which downstream
   consumers are eligible (fine-tune / eval / registry).
6. Registry poll sees `registry_ingestible = true`; ingests
   the contract; adds to the registry.

Once wired, downstream consumers become fully self-serve. That
is the point.

## Summary

- The reproducibility bundle (chapter 4) is complete. The
  contract is the small, stable subset downstream tracks
  code against.
- The contract is a single `training_run.v1.json` at the root
  of the run's object-store prefix. It carries model /
  tokenizer / data / training / checkpoint / metrics
  pointers with sha256s.
- Three consumers: fine-tuning-engineer (base model
  provenance), model-evaluation-engineer (stable identity +
  hardware manifest), ai-infra-ml-platform (registry
  ingestion).
- Storage protocol: one run, one prefix; entry point at the
  root; no in-place edits, ever. Consumers depend only on the
  canonical entry-point path, not the internal layout.
- Version by file name and inside-document field. Additive
  changes stay in v1; renames / removals mint v2 with a
  compatibility window. Never mutate a published contract.
- MLflow's `MLmodel` and W&B artifact schemas are worth
  studying — the training-run contract layers on top of the
  same idea at run scale.
- Automate validation: JSON Schema every emitted contract,
  lifecycle-hold every referenced object, HEAD-sweep every URI
  periodically. Convention alone will not keep the contract
  honest.
