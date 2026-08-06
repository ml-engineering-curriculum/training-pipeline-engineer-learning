# The Experiment Metadata Store: A Contract with Downstream Teams

Chapters 2 through 6 covered the tooling that serves the platform
engineer and the researcher — the two audiences physically working
on the training run. This chapter is about the third audience from
chapter 1: the downstream teams whose entire relationship with
training is *after the fact*. Fine-tuning teams (see the
`fine-tuning-engineer` role), evaluation teams (see the
`model-evaluation-engineer` role), safety / red-team consumers, and
inference-platform teams all pick up a checkpoint days or months
after it was written and need to know what they are picking up.

The tool that answers their questions is the **experiment metadata
store**. This chapter is its schema, its integration boundaries, and
the query patterns the downstream teams will exercise.

## What "the store" actually is

The metadata store is a small database (Postgres is the honest
default; MLflow's tracking store is a common instance of the same
pattern) that indexes every training run by ID and holds the row of
metadata that answers the downstream teams' questions. It is *not*
where any of the following live:

- **Checkpoint bytes** — they live in object storage; the store
  holds URIs.
- **Metric time-series** — they live in Prometheus and the tracker
  (chapters 3 and 4); the store holds pointers.
- **Log lines** — they live in the log store (Loki, Elasticsearch,
  or the trainer's raw stdout in object storage); the store holds
  pointers.
- **Bundle bytes** — they live in object storage per chapter 5;
  the store holds URIs.
- **Dashboards** — they live in Grafana; the store holds
  templatized URLs.

The store is the *index* that ties all of these together on `run_id`
and provides the query interface downstream teams use. Its value is
not the data it holds; it is the joins it makes cheap.

MLflow is a defensible open-source choice; the store's schema can
be layered on top of MLflow's own tables. W&B and some managed
tracker services offer equivalent metadata surfaces. Chapter 4's
tracker choice tends to determine whether the metadata store is
"MLflow standalone" or "the tracker's built-in metadata layer plus
a Postgres side-table for extension fields".

## The core schema: five tables

### `run`

The top-level object. One row per training run.

| Column | Type | Notes |
|---|---|---|
| `run_id` | text (primary key) | Canonical run identifier; the join key across every other tier. |
| `parent_run_id` | text nullable | For continuation runs (rank-count-changed resume, LR reset, cool-down). Nullable for fresh starts. |
| `run_type` | enum | `pretrain`, `sft`, `dpo`, `rlhf`, `cont_pretrain`, `eval`, etc. |
| `status` | enum | `queued`, `running`, `completed`, `failed`, `cancelled`. |
| `start_time_utc` | timestamptz | |
| `end_time_utc` | timestamptz nullable | |
| `bundle_uri` | text | Object-storage URI of the reproducibility bundle (chapter 5). |
| `dashboard_url` | text | Templatized Grafana URL for this `run_id`. |
| `tracker_url` | text | W&B / TensorBoard / MLflow URL for this `run_id`. |
| `logbook_uri` | text | Human-authored logbook artifact (mod-106 ch. 7). |
| `owning_team` | text | Team accountable for the run. |
| `notes` | text | Free-form description / research note. |
| `tags` | jsonb | Arbitrary key-value tags for filtering. |

The `run_id` scheme matters. A defensible convention:

```
<yyyymmdd>-<team>-<recipe_slug>-<seq>
```

E.g., `20260806-pretrain-llama3-70b-baseline-001`. Human-readable,
sortable, and grep-friendly. Whatever convention you pick,
standardize it at the team level and encode it in the trainer's
`run_id` generator — no hand-typed IDs.

### `checkpoint`

One row per persisted checkpoint. This is the primary object
downstream teams consume.

| Column | Type | Notes |
|---|---|---|
| `checkpoint_id` | text (primary key) | E.g., `<run_id>-step-<step>`. |
| `run_id` | text (foreign key) | Which run wrote it. |
| `step` | bigint | Training step count. |
| `epoch` | int nullable | If the loop is epoch-based. |
| `wall_time_utc` | timestamptz | When it was written. |
| `weights_uri` | text | Object-storage URI to the DCP checkpoint directory. |
| `weights_sha256` | text | Hash of a manifest of the DCP shards; used for integrity check on load. |
| `bundle_uri` | text | Copy of the run's bundle URI, for point-in-time lookup. |
| `train_loss` | double precision | Loss at step (or nearest logged step). |
| `val_loss` | double precision nullable | If a validation pass ran at this step. |
| `is_pinned` | boolean default false | Prevents cleanup for retention-policy purposes. |
| `notes` | text | E.g., "picked as base for fine-tuning-experiment X". |

The `is_pinned` column is what stops the training team from garbage-
collecting a checkpoint the fine-tuning team is about to use. The
storage cost of pinning is enough that "pin every checkpoint
forever" is not the default; explicit pinning by downstream
consumers is the pattern.

### `dataset_snapshot`

One row per dataset version referenced by any run. Small table
that a run's `run` row joins to via `run.tags.dataset_snapshot_id`.

| Column | Type | Notes |
|---|---|---|
| `dataset_snapshot_id` | text (primary key) | E.g., `<dataset_name>-<manifest_sha_short>`. |
| `dataset_name` | text | Logical name. |
| `manifest_uri` | text | Object-storage URI of the manifest. |
| `manifest_sha256` | text | Same value as the run bundle's `dataset.manifest_hash`. |
| `total_bytes` | bigint | Sum of shard sizes. |
| `total_tokens` | bigint nullable | After tokenization, if known. |
| `created_at_utc` | timestamptz | |
| `notes` | text | E.g., "excludes shards X-Y after quarantine". |

The join `run.tags.dataset_snapshot_id → dataset_snapshot.dataset_snapshot_id`
answers "which data did this run see?" The downstream fine-tuning
team uses this to avoid training against data the base model has
already memorized.

### `evaluation`

One row per evaluation performed against a checkpoint.

| Column | Type | Notes |
|---|---|---|
| `evaluation_id` | text (primary key) | |
| `checkpoint_id` | text (foreign key) | The checkpoint the eval was run against. |
| `eval_suite` | text | The eval harness / suite name. |
| `eval_version` | text | Version of the suite. |
| `results_uri` | text | Object-storage URI to the raw results. |
| `summary_scores` | jsonb | The headline numbers, denormalized for query. |
| `wall_time_utc` | timestamptz | |
| `run_id_of_eval` | text nullable | If the eval was itself a run in the store. |

This table is what the evaluation-engineer team writes to and the
fine-tuning-engineer team reads from when deciding which
checkpoint to start from. The `summary_scores` denormalization
enables fast queries ("which checkpoints for run X have MMLU >
0.7?") without loading the raw results.

### `incident`

One row per operator-noted incident on a run. Corresponds one-to-one
with entries in the run's logbook (mod-106 chapter 7).

| Column | Type | Notes |
|---|---|---|
| `incident_id` | text (primary key) | |
| `run_id` | text (foreign key) | |
| `step` | bigint nullable | Approximate step; nullable if incident was pre-run or between steps. |
| `wall_time_utc` | timestamptz | |
| `class` | enum | The five classes from mod-106 chapter 5. |
| `signature` | enum | The five signatures from chapter 6. |
| `severity` | enum | `low`, `medium`, `high`, `critical`. |
| `resolution` | text | Free-form. |
| `linked_prom_query` | text nullable | A Prometheus query URL that renders the incident. |

This table is what makes historical A/B analysis of runs possible.
A downstream consumer asking "was there an incident during the
window that produced my base checkpoint?" gets a direct answer.

## The query patterns downstream teams exercise

The schema is only as useful as the queries it makes fast. Five
patterns to design against; each maps to a downstream team
workflow.

### Pattern 1: "Give me the metadata for this checkpoint I have"

The fine-tuning engineer has a checkpoint URI (or ID) and wants to
know everything about it before starting a fine-tune.

```
SELECT c.*, r.*, ds.*
FROM checkpoint c
JOIN run r ON c.run_id = r.run_id
LEFT JOIN dataset_snapshot ds
    ON ds.dataset_snapshot_id = r.tags->>'dataset_snapshot_id'
WHERE c.checkpoint_id = :ckpt_id;
```

This is the primary query. Answer under 100 ms is easy on Postgres
with the right indexes; make it the "obvious" API of the store.

### Pattern 2: "Which checkpoint of run X had the best eval on suite Y?"

The evaluation engineer wants to pick a specific step.

```
SELECT c.checkpoint_id, c.step, e.summary_scores
FROM checkpoint c
JOIN evaluation e ON e.checkpoint_id = c.checkpoint_id
WHERE c.run_id = :run_id
  AND e.eval_suite = :suite
ORDER BY (e.summary_scores->>'primary_score')::float DESC
LIMIT 1;
```

The store as a discovery UI: the eval team writes rows; the
fine-tune team reads them.

### Pattern 3: "Find runs comparable to this one"

The researcher wants to A/B against a baseline.

```
SELECT run_id, notes
FROM run
WHERE tags @> :tag_filter
  AND owning_team = :team
  AND status = 'completed'
  AND start_time_utc > now() - interval '90 days';
```

`tag_filter` is a jsonb query that filters on model size, data
mix, LR schedule, etc. This is one of the queries the tracker's UI
is often better at; a store-side query is the fallback when the
tracker cannot express the filter.

### Pattern 4: "Was there an incident on run X in step-range S?"

The debugging pattern for post-mortem analysis.

```
SELECT step, class, signature, severity, resolution
FROM incident
WHERE run_id = :run_id
  AND step BETWEEN :s_start AND :s_end
ORDER BY step;
```

The linked Prometheus query URL in each row jumps the on-call
straight to the metric view (chapter 6's disambiguation queries).

### Pattern 5: "Which runs saw dataset snapshot Y?"

The safety / red-team consumer wants to know which models have
been exposed to a specific corpus (e.g., a corpus later marked
sensitive).

```
SELECT run_id, start_time_utc
FROM run
WHERE tags->>'dataset_snapshot_id' = :ds_id;
```

The reverse-join. Fast on the index; important for compliance.

## The contract with downstream teams

Every downstream team's read from the store is only as good as the
platform team's write into it. Formalize the contract:

- **Every run writes a `run` row at start.** The trainer refuses
  to start without emitting one (analogous to the bundle in
  chapter 5).
- **Every checkpoint writes a `checkpoint` row on successful
  save.** Load-time integrity check reads `weights_sha256` and
  fails hard on mismatch.
- **Every eval writes an `evaluation` row on completion.** The
  eval harness's job, not the training team's.
- **Every logbook entry writes an `incident` row.** The runbook
  in mod-106 chapter 5 references the store row.
- **Deletion is soft.** A hard-delete of a `checkpoint` row while
  downstream teams have URIs in flight breaks their runs. Soft-
  delete with a grace period, and require an explicit purge to
  reclaim storage.

Publish the contract as an API spec — either an OpenAPI document
or a set of gRPC service definitions. mod-110 chapter (the
platform architecture and leadership module) is where the cross-
team contract of which this is one instance formally lives; this
chapter is the schema side.

## The boundary with the fine-tuning-engineer role

The fine-tuning-engineer's job (as per the sibling role
`fine-tuning-engineer-learning` in this org) is to take a base
checkpoint and produce an aligned or specialized derivative. Their
questions of the store:

- "Which base checkpoint should I start from?" Pattern 2 above.
- "What data did the base model see? Do I need to avoid
  overlapping fine-tune data?" Pattern 5 above.
- "Is the base run reproducible? If I need to re-derive it, do I
  have what I need?" The `bundle_uri` column, verified against
  chapter 5's `bundle-diff` tool.
- "Was the base run clean, or were there incidents?" Pattern 4.

None of these questions are answerable from a checkpoint URI
alone. All of them are answerable from the store's schema.

## The boundary with the model-evaluation-engineer role

The evaluation engineer runs eval suites against checkpoints and
publishes results. Their write path is `evaluation` table rows;
their read path is `checkpoint` and `run` rows to find candidates.

The join `checkpoint × evaluation` is what turns "here are 500
checkpoints" into "here are the 12 checkpoints that scored above
threshold on the eval suite the fine-tune team cares about". The
store makes this a query; without it, it is a manual spreadsheet.

## The boundary with the inference-platform team

An inference platform picks up a checkpoint, converts it to an
inference-optimized format (quantized, sharded, TensorRT-LLM-
compiled, etc.), and serves it. Their questions of the store are
narrower:

- "What was the input format of this checkpoint (DCP, HF,
  Megatron)?" Should be a tag on the `checkpoint` row.
- "What tokenizer does this checkpoint use?" Recorded in the
  bundle at `bundle.tokenizer.uri`; can be denormalized onto the
  `checkpoint` row for query speed.
- "What is the model architecture?" From the config in the
  bundle; can also be a `run` tag.

The inference platform lives in a different module (out of scope
for this track); the store's schema has to be usable by them
anyway. Design the schema for the whole downstream surface, not
just the two adjacent roles.

## Failure modes and how the store guards against them

- **A checkpoint URI outlives its `checkpoint` row.** Downstream
  team loads weights that have no metadata. Fix: enforce the
  contract, and provide a fallback lookup by URI. Soft-delete
  never removes the URI-to-metadata pointer.
- **Two runs share a `run_id`.** Auto-generated IDs collide, or a
  restarted run reuses the same ID. Fix: append a monotonically-
  increasing suffix at the trainer, and add a unique constraint.
- **Bundle URI drift.** The bundle is moved or re-tiered without
  updating the store row. Fix: content-addressable storage
  (chapter 5), or a periodic reachability check that alerts on
  broken URIs.
- **Eval-suite version churn.** The eval was written against
  suite v1; a re-eval on v2 has incomparable numbers. Fix:
  `evaluation.eval_version` column; downstream teams filter by
  version when comparing.
- **Metadata store as the source of truth for the run itself.**
  If the store goes down, no run should stop. The store is a
  read-optimized index; the *source of truth* is the bundle in
  object storage. The store rebuilds from bundle URIs by
  scanning object storage.

## Operational sizing

The store is small. Even a large team writes only a few thousand
`run` rows per year, tens of thousands of `checkpoint` rows,
similar order of `evaluation` and `incident` rows. A single
Postgres instance with a `t3.medium`-equivalent handles this
volume through many years of use.

Do not over-engineer. Redundancy through backups and a warm read
replica is sufficient; a fully-managed multi-region deployment is
overkill for this workload.

## Summary

- The metadata store is the index that ties together every other
  observability tier — Prometheus, tracker, bundle, logbook,
  dashboard — on `run_id`. It holds pointers, not bytes.
- Five core tables cover the surface: `run`, `checkpoint`,
  `dataset_snapshot`, `evaluation`, `incident`. Postgres or
  MLflow's underlying store is the honest default.
- Five query patterns drive the schema: metadata-for-a-checkpoint,
  best-checkpoint-by-eval-score, find-comparable-runs, incidents-
  in-a-window, which-runs-saw-a-dataset. Design against them.
- The store is a contract with the downstream teams (fine-tuning,
  evaluation, inference, safety). Publish the contract as an API
  spec; make it unskippable in the training loop.
- The store is a read-optimized index, not the source of truth.
  Bundle URIs in object storage are the ground truth from which
  the store can be rebuilt.
- The schema is the platform team's leverage: it turns "the
  training team's spreadsheets" into "any downstream team can
  answer their question in a query". mod-110 formalizes the
  cross-team contract; this chapter is the schema.
