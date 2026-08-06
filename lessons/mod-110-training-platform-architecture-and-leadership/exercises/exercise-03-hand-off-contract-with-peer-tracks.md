# exercise-03: Hand-Off Contract with a Peer Track

**Estimated effort:** 3 hours

## Objective

Author a complete **hand-off contract** with one of the five
peer platform tracks named in chapter 4. Deliverable: one
Markdown document (`interface-<peer>-vN.md`) in the
eight-section template, one JSON Schema (or equivalent) that
formally specifies the artifact flowing across the boundary,
and one worked example artifact validated against the schema.

## Prerequisites

- Chapter 4 (hand-off contracts) of this module.
- The AICG role definitions or public role descriptions for
  the peer track you pick. Skim `role-description.md` in the
  peer track's learning repo if available.
- Mod-108 chapter 5 (reproducibility bundle) and chapter 7
  (metadata schema) if you pick interface 2 (serving) or
  interface 3 (MLOps) — those interfaces build on top of
  those artifacts.
- Familiarity with JSON Schema or protobuf (whichever you
  choose for the formal spec).

## Problem statement

Pick **one** peer track and author the full interface
document. The five options mirror chapter 4:

- **Interface 1 — fine-tuning-engineer (consumer).** The
  workload-manifest schema, base-checkpoint format, quota /
  queue-time SLA, incident-report format.
- **Interface 2 — ai-infra-ml-platform-engineer (serving +
  registry).** The checkpoint-publication format, tokenizer /
  config hand-off, provenance link into the registry.
- **Interface 3 — ai-infra-mlops-engineer (CI/CD + post-
  training).** The metadata-schema contract, evaluation-
  trigger event, release-tag notification.
- **Interface 4 — ai-infra-performance-engineer (kernels).**
  The kernel-release format (versioned wheels or Triton), the
  performance-envelope publication, the regression-report
  flow.
- **Interface 5 — ai-infra-security-engineer (provenance +
  boundary).** The provenance-manifest schema, the approved-
  image / approved-dataset list format, the audit-log
  contract.

Choose one. The rest of the requirements apply to whichever
interface you pick.

## Requirements

Ship one directory `mod-110-ex03/` with:

- `interface-<peer>-vN.md` — the interface document.
- `schemas/` — the formal schema for at least the primary
  artifact flowing across the boundary.
- `examples/` — one validated example artifact plus a short
  `validate.sh` (or equivalent) that runs the schema check.
- `escalation-runbook.md` — a short (~1 page) joint runbook
  for the cross-team incident case.

### 1. The interface document (`interface-<peer>-vN.md`)

Length target: 2–4 pages. All eight sections from chapter 4:

1. **What flows across the boundary.** Direction (ours→theirs,
   theirs→ours, or both), artifact, format, cadence.
2. **Format specification.** Concrete schema — JSON Schema,
   protobuf, filesystem layout, whatever fits. Versioning
   rule (semver, content-addressed digest, or explicit
   scheme).
3. **SLA.** Availability, freshness, correctness, size — each
   with a measurable threshold and the metric that backs it.
4. **Compatibility window.** Length, both-paths-produce-
   compatible-outputs statement, enforcement mechanism.
5. **Failure handling.** What is the ack path when a
   consumer gets bad data? What is the rollback path?
6. **On-call and escalation.** Who to page on each side,
   response-time targets, when does the incident go joint.
7. **Change process.** Which RFC template governs a change;
   which side owns the RFC; which side signs off.
8. **Sign-off.** Named individuals (roles are OK for the
   exercise), date, version.

Owners on both sides (leads, not ICs). Versioned; the
version bump is documented at the top.

### 2. The formal schema (`schemas/`)

For at least the primary artifact:

- **JSON Schema** (draft 2020-12 or newer) if the artifact is
  JSON.
- **Protobuf** schema if the artifact is a wire message.
- **A directory-layout spec** (Markdown table with expected
  file names, formats, and required fields) if the artifact
  is a filesystem layout (e.g., a checkpoint bundle).

The schema is complete enough that a consumer could implement
a validator from it. Required fields marked required;
optional fields marked with defaults; enumerations listed
inline; version field included.

### 3. Example artifact (`examples/`)

At least one worked example, filled with realistic-looking
(but not real) values:

- For interface 1: a workload manifest for a Llama-3-8B SFT
  run.
- For interface 2: a checkpoint bundle for a 34B pretraining
  step (directory + manifest JSON).
- For interface 3: a `run` row and its associated
  `checkpoint`, `dataset_snapshot`, `evaluation`, `incident`
  rows.
- For interface 4: a kernel release manifest with performance
  envelope.
- For interface 5: a provenance manifest for one training run.

Include `validate.sh` or an equivalent step that runs the
schema check (`jsonschema`, `protoc`, or a checked-in Python
script). It exits non-zero on schema violation.

### 4. The joint escalation runbook (`escalation-runbook.md`)

The chapter 4 template calls for a joint on-call bridge; write
its runbook. One page:

- **Who leads the bridge** by default (rotate between the two
  sides on odd/even weeks, or fix on one side — pick).
- **The bridge open command.** How the incident channel is
  created, who joins, what the first message contains.
- **The severity classification decision.** How the two
  teams reconcile if they disagree on the Sev level.
- **The postmortem ownership.** Who authors the joint
  postmortem (chapter 6) and who reviews.
- **The runbook cross-references.** Links to each side's
  domain-specific runbook (mod-106 chapter 5 on the training
  side; the peer's equivalent runbook).

## Starter guidance

- **Read the actual peer-track README/role description first.**
  Chapter 4's sketches are abstractions of real roles; the
  peer's own definition may name concerns you have not
  considered. Don't invent — read.
- **Pick the artifact whose schema you can pin down.** A
  workload manifest or provenance manifest is concrete; a
  "kernel release" is abstract. If your chosen interface's
  artifact is fuzzy, pick a different interface.
- **Anchor to existing formats where they exist.** If the peer
  is the ml-platform-engineer, the checkpoint layout is
  probably safetensors + config.json — do not invent a new
  format for the exercise.
- **The SLA is not real if no metric backs it.** For every
  SLA line, name the metric (mod-108 chapter 3 or peer-side
  equivalent) that measures it.
- **Do not skip the failure-handling section.** It is where the
  interface is stress-tested. If you cannot describe what
  happens when the consumer gets a corrupt artifact, the
  contract is not complete.
- **The escalation runbook is not the interface document.**
  Keep them separate — the interface doc is the contract, the
  runbook is the operational overlay.

## Acceptance criteria

- Interface document has all eight sections in order; each
  section is at least 3 sentences (no placeholders).
- Owners on both sides are named (leads, not ICs); role
  placeholders are OK for the exercise but the role name is
  filled in.
- Format specification is a real schema, not prose. Every
  required field, type, and version tag is expressed in the
  schema language.
- Every SLA line names the metric that backs it.
- Compatibility window is sized against the slowest
  consumer's release cycle and justified in one sentence.
- Failure-handling section covers both "consumer receives
  bad data" and "rollback of a bad producer release".
- Schema validates the example artifact (exit 0) and rejects
  a synthetic bad artifact (exit non-zero); include the bad
  example alongside the good one.
- Escalation runbook covers bridge lead, open command, Sev
  disagreement, postmortem ownership, and runbook cross-
  references.
- Sign-off block enumerates both sides' roles and the version
  bump reason.

## Stretch goals

- **Author all five.** Do the exercise once per interface;
  publish as a "platform interfaces" bundle. This is what a
  real platform lead delivers in their first quarter.
- **Version-bump exercise.** Bump the interface to v(N+1)
  with a breaking change (add a required field, drop a
  field). Write the migration plan (chapter 7 template) for
  the interface change, and the joint RFC (chapter 3
  template) that gates it.
- **Contract-test harness.** Ship a small test harness (in
  Python or Go) that both sides can run against a proposed
  artifact and get a pass/fail. This is the mechanism by
  which the interface's SLA becomes a CI check, not a
  document.
- **Interface index.** Author the top-level
  `platform-interfaces/README.md` from chapter 4 — the table
  that lists all five interfaces with versions, owners, and
  last-update dates. Ship it alongside your one interface as
  the index it will grow into.
- **Consumer-side integration.** Pretend to be the peer team
  and implement a minimal consumer against your schema
  (e.g., a small script that reads a workload manifest and
  submits a Kueue Workload). Confirms the contract is
  actually usable.
