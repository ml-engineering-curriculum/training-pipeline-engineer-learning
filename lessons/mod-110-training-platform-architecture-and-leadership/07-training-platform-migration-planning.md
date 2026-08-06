# Training-Platform Migration Planning

Chapter 3 (RFCs) is *whether* a platform change happens.
Chapter 7 is *how* it happens. Every large migration — a
framework major version, a hardware generation, a storage
tier, a scheduler cut-over — has the same anatomy: a phased
rollout with defined stages, an explicit rollback path per
stage, a compatibility window for consumers, and a versioning
scheme that lets old and new coexist during the transition.

This chapter is the migration-plan template every platform-
change RFC's §9 (rollout) links to. The two worked examples
are the migrations training platforms in 2025–2026 actually
run: FSDP1 → FSDP2, and Megatron-LM → torchtitan. Both are
open, both are ongoing across the industry, and both illustrate
the template's shape.

## Why a migration plan is a distinct artifact

The RFC covers the *decision*. The design doc covers the
*mechanism*. The migration plan covers the *transition*. Three
properties distinguish it:

- **Time-boxed.** Every migration has a start date, per-stage
  dates, and an end date at which the old path is removed.
- **Consumer-facing.** Every consuming team's action is enumerated,
  with a per-team completion date.
- **Rollback-shaped.** Every stage has a rollback trigger and a
  rollback procedure. A migration without rollbacks is a
  migration that eats an outage.

An RFC without a migration plan is a decision without a
delivery mechanism. A migration plan without an RFC is a rollout
with no sign-off. Both are needed for user-visible changes.

## The four stages every migration has

Every migration passes through four stages. The names, entry
gates, and exit gates below are the template.

### Stage 0 — Baseline lock

Before the migration begins, freeze the *current* state so
performance and correctness regressions become measurable
against a specific baseline.

- **Entry gate.** RFC accepted (chapter 3).
- **Action.** Snapshot the current platform's measurable
  properties for the workloads that will migrate: `μ_sus` per
  recipe, checkpoint round-trip time, elastic-recovery
  latency, kernel version, container digest, dependency
  versions. Publish to the migration doc.
- **Exit gate.** Baseline published; consumers acknowledge.

Duration: 1–2 weeks.

### Stage 1 — Pilot at 1/8 scale

Run the migration end-to-end on a small workload before any
consumer is asked to move. The 1/8 scale is a rule of thumb
from mod-109 chapter 7's pilot practice; adjust to your
platform's smallest realistic scale.

- **Entry gate.** Baseline locked.
- **Action.** Implement the change in a feature-flagged branch
  or a parallel installation. Run one recipe end-to-end on the
  new path at 1/8 the target cluster shape. Measure the same
  properties captured in stage 0.
- **Exit gate.** Success criteria met, or rollback executed
  and the RFC re-opened.

Success criteria are set in the RFC's test-plan section.
Typical: `μ_sus` within 2% of baseline, checkpoint round-trip
time within 20% of baseline, no new alerts firing, elastic
recovery drills pass.

Duration: 2–4 weeks. Extend if the pilot flags issues; do not
compress.

### Stage 2 — Opt-in rollout

Teams that want the new path early can opt in; the platform
supports both paths in parallel.

- **Entry gate.** Pilot success criteria met.
- **Action.** New path exposed as an opt-in flag or a parallel
  path. Consumer-side changes documented. Runbook updated. On-
  call trained on the new path.
- **Exit gate.** At least two teams have run successfully on
  the new path for at least two weeks; no Sev-2 incidents
  attributable to the new path.

The two-team, two-week rule is what distinguishes "our team
tested it" from "we know it works in production": at least one
consuming team other than the platform team has to succeed
before the mandatory rollout begins.

Duration: 2–4 weeks.

### Stage 3 — Mandatory rollout

The new path becomes the default. Consumers still on the old
path must migrate by the deprecation date.

- **Entry gate.** Opt-in exit gate met.
- **Action.** New path is the default; old path remains
  available via explicit flag through the compatibility
  window. Per-team migration dates published. Weekly
  check-ins on migration progress.
- **Exit gate.** All consumers migrated; old path removed;
  RFC status updated to Implemented.

The compatibility window is what makes this stage humane. The
next section sizes it.

## The compatibility window: three properties

Every migration has a *compatibility window* — the period
during which both old and new paths are supported. Three
properties define a sound window:

- **Length.** Set against the slowest consumer's release
  cadence. If serving releases quarterly (chapter 4 interface
  2), the window must span at least one full serving-release
  cycle, i.e. two quarters minimum. Framework major-version
  migrations (FSDP2, torchtitan) typically span two quarters.
  Hardware-generation cut-overs typically span three quarters
  (a full model-release cycle).
- **Behaviour.** Both paths produce compatible outputs during
  the window. Checkpoints written by the new path are readable
  by the old (and vice versa) *unless* the RFC explicitly
  declares this out of scope with a documented reason.
- **Enforcement.** The old path is removed on the announced
  date, not "when we get around to it". Slippage of the
  deprecation date is itself an RFC — the schedule change
  requires the same sign-off the original migration did.

The failure mode is a compatibility window that is announced
but never enforced. Teams that did not migrate stay on the old
path forever; the platform team maintains two implementations
in perpetuity; eventually the old path breaks and everyone is
surprised. Set the date; hold the date.

## Versioning and coexistence

During the window, both paths coexist. Two mechanisms make
coexistence work:

### Config-level versioning

Every consumer's config declares which path it uses:

```yaml
platform:
  parallelism: "fsdp2"      # or "fsdp1" during compatibility
```

Or:

```yaml
platform:
  framework: "torchtitan"   # or "megatron-lm"
```

The config version is checked at admission time (chapter 2
admission invariants). Deprecated paths remain valid submission
values until the deprecation date; after that, admission
rejects them.

### Artifact-level versioning

Checkpoints, containers, and provenance manifests carry version
tags:

- **Checkpoints.** Include a `platform_version` field in the
  mod-108 chapter 5 reproducibility bundle. Loaders check;
  cross-version loading is either supported (with a converter
  — see below) or explicitly rejected.
- **Container images.** Both old-path and new-path images
  co-published under distinct tags; both are digest-pinned.
- **Provenance manifests.** Include a `path_version` field so
  chapter 4 interface 5 (security) can audit which runs used
  which path.

### Converters

For checkpoint-format changes (FSDP1 → FSDP2, safetensors →
DCP), the migration ships a *converter*: a script that loads
a checkpoint in the old format and writes it in the new. Two
properties:

- **Idempotent.** Running the converter twice produces the same
  output as running it once.
- **Verified.** The converter includes a numerical-equivalence
  check: loading the converted checkpoint and running one
  training step must produce loss within a tight tolerance of
  the original.

The converter's existence is what lets teams migrate their
existing state without re-materialising from scratch. If a
migration cannot ship a converter, say so in the RFC — the
per-team migration cost is materially higher and the timeline
extends.

## Rollback per stage

Every stage has an explicit rollback procedure with a defined
trigger.

| Stage | Rollback trigger                              | Rollback action                                    |
|------:|-----------------------------------------------|----------------------------------------------------|
|     0 | Baseline not reproducible                     | Fix measurement; re-baseline; do not proceed       |
|     1 | Pilot success criteria not met                | Return to old path; re-open RFC; investigate       |
|     2 | Sev-2 incident on new path attributed to change | Withdraw opt-in flag; hold rollout; postmortem   |
|     3 | Multi-team Sev-1 or Sev-2 attributed to change | Restore old path as default; extend window; RFC amendment |

Two properties:

- **The rollback is a runbook, not a plan.** Write the
  step-by-step in a runbook page linked from the migration
  doc. The rollback is executed under pressure; nobody has
  time to read a page of prose.
- **The rollback preserves data.** No rollback ever discards
  checkpoints, manifests, or run metadata. The rollback is
  reversible; the migration is not — nothing that happens
  during a migration should destroy state.

## Worked example 1 — FSDP1 → FSDP2

FSDP2 is PyTorch's DTensor-based replacement for the original
FSDP; FSDP1 was deprecated in PyTorch 2.6. The migration is
representative of a framework-internal cutover with a working
converter path.

- **Baseline (stage 0).** `μ_sus` per recipe under FSDP1;
  checkpoint round-trip time; container/PyTorch version.
- **Pilot (stage 1).** Run Llama-3-8B at 64 GPUs under FSDP2
  with the same recipe. Success = `μ_sus` within 2% of
  baseline (typically FSDP2 is neutral or better; see
  https://pytorch.org/blog/fsdp2/), checkpoint round-trip
  within 20%, existing FSDP1 checkpoints load via converter.
- **Opt-in (stage 2).** Expose `platform.parallelism = "fsdp2"`
  in team configs. Runbook updated for FSDP2-specific failure
  modes (DTensor errors, `fully_shard` misuses).
- **Mandatory (stage 3).** FSDP2 becomes default. FSDP1
  supported for 6 months via `platform.parallelism = "fsdp1"`;
  after that, admission rejects.
- **Converter.** FSDP1 → FSDP2 checkpoint converter (`torch.
  distributed.checkpoint` supports cross-format load in most
  cases; verify per recipe).
- **Rollback.** Config-level flip returns any team to FSDP1
  during the window. After window closes, rollback requires an
  RFC amendment and window extension.

- **Compatibility window.** Two quarters. Matches the serving
  team's release cadence and PyTorch's own two-minor-version
  deprecation cycle.

## Worked example 2 — Megatron-LM → torchtitan

torchtitan is PyTorch's native training reference
implementation (pytorch/torchtitan). Migrating from Megatron-LM
is a whole-framework cutover — a strictly larger change than
FSDP1 → FSDP2 and typical of "we are re-basing the training
stack".

- **Baseline (stage 0).** For each recipe currently on
  Megatron-LM: `μ_sus`, wall-clock, checkpoint format, dataset
  format, tokenizer. Freeze.
- **Pilot (stage 1).** Port one recipe (typically the
  most-run) to torchtitan at 1/8 scale. Success = `μ_sus`
  within 5% of Megatron-LM baseline (framework rebases often
  produce larger initial gaps; the RFC sets the tolerance).
  Numerical-equivalence check on trained-weights outputs
  within a tight tolerance across N steps.
- **Opt-in (stage 2).** New teams and new recipes can adopt
  torchtitan; existing recipes remain on Megatron. At least
  two teams complete two-week runs on torchtitan before
  mandatory rollout.
- **Mandatory (stage 3).** New recipes default to torchtitan.
  Existing Megatron recipes migrate on a per-recipe schedule.
- **Converter.** Megatron sharded checkpoints → torchtitan DCP
  format converter. This is materially more involved than the
  FSDP1→FSDP2 case; may not exist for all parallelism
  configurations. If absent for a given config, the migration
  requires a full re-training or a warm-start from a
  cross-format load — costly, so document explicitly.
- **Rollback.** Per-recipe rollback via config flag. If a
  framework-wide Sev-1 fires on torchtitan, admission
  temporarily rejects new torchtitan submissions and default
  returns to Megatron.
- **Compatibility window.** Three quarters. Longer because the
  converter story is more complex and the per-team porting
  effort is higher.

Whole-framework migrations are the most expensive migrations
the platform runs; expect 6–12 months from RFC acceptance to
Megatron removal, and expect the schedule to slip once.
Migration plans that assume no slip are aspirational; leave a
buffer.

## Consumer-side compatibility

The migration plan is not complete without a per-consumer
action table. From chapter 4's interface docs, list every
consumer of the changing surface and what they must do.

| Consumer                       | Action required                                          | Owner (theirs) | Due (their date) |
|--------------------------------|----------------------------------------------------------|----------------|------------------|
| Foundations                    | Migrate 3 recipes to torchtitan; re-run baseline eval    | (name)         | 2027-01-15        |
| Fine-tuning                    | Migrate 8 recipes; update team runbook                   | (name)         | 2027-02-01        |
| Post-training                  | No action (consumes checkpoints; converter path)         | (name)         | —                 |
| Serving (ml-platform-engineer) | Verify torchtitan-produced checkpoint conversion         | (name)         | 2027-01-01        |
| MLOps                          | Update release pipeline metadata parsing (path_version)  | (name)         | 2026-12-15        |
| Security                       | Confirm provenance manifest schema unchanged             | (name)         | 2026-12-01        |

The table is published in the migration doc, updated weekly
during rollout. A team whose row is blank was not consulted —
fix before rollout begins.

## Weekly migration status format

During rollout, the migration is tracked in a weekly status
post:

```
# Migration: <name> — Week N status (YYYY-MM-DD)

Stage:              <current stage>
On track:           <yes / at-risk / no>
Consumers migrated: <count> / <total>

## Progress this week
- <what shipped>

## Blockers
- <named blocker, owner, ETA>

## Risks
- <risks with likelihood / impact / mitigation>

## Next week
- <what is planned>
```

Post to the incident/status channel and archive alongside the
migration doc. This is the artifact that tells leadership
whether to intervene.

## Common failure modes

- **No baseline.** The migration starts without a snapshot of
  the current state. When performance regresses, nobody can
  say by how much.
- **Pilot skipped.** The change goes to opt-in without a 1/8-
  scale run. The first Sev-2 on the new path lands on a
  consuming team, not on the platform.
- **Opt-in stage bypassed.** The platform team declares the new
  path ready and mandates it immediately. No consuming team
  has run in production; the rollout is the pilot.
- **Compatibility window without enforcement.** The date passes;
  nobody removes the old path. Two implementations, forever.
- **Rollback plan that is prose.** The rollback under pressure
  requires a runbook, not a doc that describes the rollback.
  Write the runbook page.
- **No converter, no acknowledgement.** The migration ships
  without addressing existing state. Consumers rediscover the
  gap during rollout and stall the mandatory stage.
- **Per-consumer action table missing.** Some team was not
  named; that team blocks the deprecation date; the migration
  ends by extending the window rather than by delivering.
- **Migration plan written after the change starts.** The plan
  is retroactive documentation, not a delivery mechanism.
- **Slippage without RFC amendment.** The dates slip
  informally; the sign-off in the RFC no longer reflects
  reality. Amend explicitly.

## Summary

- Every user-visible platform change lands with a migration
  plan alongside its RFC. The plan covers stages, compatibility
  window, versioning, converters, rollback, and per-consumer
  actions.
- Four stages, each with entry and exit gates: baseline lock
  (1–2 weeks), pilot at 1/8 scale (2–4 weeks), opt-in rollout
  (2–4 weeks, two-team / two-week rule), mandatory rollout
  (weeks to a full quarter).
- Compatibility window sized against the slowest consumer's
  release cadence: framework major = 2 quarters, hardware
  generation = 3 quarters. Old path removed on the announced
  date; date slippage is itself an RFC.
- Config-level and artifact-level versioning let both paths
  coexist during the window. Converters make existing state
  portable across formats; a missing converter is a
  documented, higher-cost migration.
- Every stage has an explicit rollback trigger and a rollback
  *runbook* (not prose). Rollback preserves data; a migration
  never destroys state.
- The per-consumer action table is the migration's checklist
  against chapter 4's interfaces; a team not named is a team
  who blocks the deprecation date.
- Weekly migration status posts keep leadership calibrated.
  Slippage without an RFC amendment is dishonest; own the
  amendment.
- FSDP1→FSDP2 is a framework-internal cutover with a working
  converter and a two-quarter window. Megatron-LM→torchtitan
  is a whole-framework rebase with converter gaps and a
  three-quarter window. Both are open industry examples; the
  template's shape holds for either.
