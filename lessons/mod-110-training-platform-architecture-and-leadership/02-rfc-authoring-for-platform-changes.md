# RFC Authoring for Platform Changes

Any change to the training platform that a downstream team can
notice deserves an RFC. That includes framework upgrades (FSDP1
→ FSDP2), hardware refreshes (A100 → H100 rotation), storage
migrations (Lustre → WEKA), scheduler-stack changes (SLURM →
Kubernetes), launcher-SDK API changes, quota-policy changes, and
any change to a hand-off contract with a peer track. If a
researcher runs their same YAML config next Tuesday and something
behaves differently, you owe them an RFC before Tuesday.

At level 35 the RFC is not a bureaucratic tax; it is the primary
technical artifact you produce. The RFC is where you argue the
change, invite peer review, publish the migration plan, and get
the sign-offs that let you push the change without paging four
engineering leads on the day of. This chapter is the template.

There is a lot of public prior art on how to run an RFC process
well: the Kubernetes Enhancement Proposals
(`github.com/kubernetes/enhancements`) and the Rust RFC repo
(`github.com/rust-lang/rfcs`) are the two canonical open examples,
and the IETF RFC series (`www.ietf.org/standards/rfcs/`) is the
historical grandparent. Read a couple of each; both processes have
been battle-tested at scale by shipping engineering orgs. You can
crib their structure straight into your team's wiki.

## Why bother with an RFC at all

Three reasons, in order of importance:

1. **It forces the analysis to happen once.** Writing "context,
   alternatives, rollback plan" separates the "we could do this"
   from the "we should do this" from the "we can back this out
   if it breaks." Ninety percent of the value is the author
   discovering, in section 4, that the migration is a two-quarter
   project, not a two-week project.
2. **It publishes the change to the people it will surprise.**
   The fine-tuning team, the security team, the finance team —
   none of them monitor your platform Slack. The RFC is the
   channel that reliably reaches them, because they explicitly
   sign it.
3. **It becomes the retrospective artifact.** Six months later,
   when a peer track asks "why is the tokenizer-hash field in
   the launcher SDK's `Job` struct?", the answer is "RFC-042 §3.2"
   and the answer takes 45 seconds instead of an hour.

The failure mode you are guarding against is the *silent surface
change*: a platform team ships an "internal cleanup" that turns
into a P1 for three downstream teams whose runs break the next
day. Every "internal cleanup" that touches an interface a
downstream team writes against gets an RFC. This is not optional.

## The RFC section template

A workable template — the one exercise-02 asks you to fill — has
seven sections. Skip any section only if you can justify skipping
it in a one-line note.

| Section                       | What it contains                                                                                             | Common failure mode                                     |
|-------------------------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| 1. Context                    | Problem statement, constraints, in-scope / out-of-scope, links to prior RFCs                                 | Author states the solution before the problem           |
| 2. Proposed change            | Technical detail — code paths, API diffs, config diffs, new / removed / renamed CRDs, resource impact         | Too abstract; reviewer can't tell what will actually run |
| 3. Alternatives considered    | 2–4 alternatives, why each was rejected                                                                       | Straw-man alternatives that were never real options     |
| 4. Rollout plan               | Phased schedule with named canaries, migration windows, EOL dates                                             | "We'll roll out on 2026-08-15" with no phases           |
| 5. Rollback plan              | The exact command / config change that reverts, and the conditions that trigger it                            | "We'll revert if needed" — not actually a plan          |
| 6. Risk register              | Regressions, breaking changes, capacity impact, cost impact, security implications                            | Missing security section; missing capacity section       |
| 7. Sign-off list              | Per peer track and per stakeholder team; each name must actively ack, not passively fail to object            | Sign-offs collected after the change ships             |

### Section 1: Context

Write in the shape "problem, constraint, decision needed." Do not
propose the solution. A context section that reads as "we should
upgrade to X" has jumped the analysis; a context section that reads
as "the current path has these limits under these conditions"
frames the problem correctly. Link every prior RFC that touches
the same surface — reviewers will want to know what constraints
those RFCs baked in.

In-scope and out-of-scope are load-bearing. If the FSDP2 upgrade
RFC does not explicitly say "we are not changing the checkpoint
format", downstream teams will assume the worst and object.

### Section 2: Proposed change

This is where the technical density goes. Show the code and config
diffs. Show the CRD schema before and after. Show the launcher SDK
API before and after, with the exact `Job` field renamed or added.
If the change touches a runtime shape (memory footprint, network
volume, storage QPS), estimate it with a number, not an adjective.

The Kubernetes KEP template's "Design Details" section is a good
reference here. Every KEP the K8s community accepts has a design
detail section that could be handed to another team's engineer and
implemented; that is the bar you are shooting for.

### Section 3: Alternatives considered

Two failure modes to avoid. One, straw-man alternatives ("we could
do nothing") — reviewers see through these immediately and lose
confidence in the rest of the document. Two, missing the
alternative that a peer track was quietly hoping for — talk to the
peer tracks *before* you write this section, so you know what they
would have preferred.

A workable alternatives section names three: the proposed change,
the "do nothing" option (with the cost of doing nothing quantified),
and the next-best change (with the reason it is not preferred). For
some RFCs there is a fourth alternative — the fully external option,
e.g. "migrate to managed platform X" — and that alternative deserves
its own paragraph because it flows into the build-vs-buy analysis
in chapter 4.

### Section 4: Rollout plan

The rollout plan is the section that separates a plausible RFC from
a real one. A workable rollout plan has named phases, named
canaries, and named exit criteria per phase.

A phased rollout for a framework upgrade typically looks like:

```
Phase 0 (week -2 to -1): dogfooding
  - Platform team runs internal smoke jobs on the new stack
  - Publish the migration guide draft

Phase 1 (week 0): canary team
  - Team X (identified low-risk team) runs one workload on the new stack
  - Publish comparative benchmark within 2 weeks
  - Exit criteria: MFU regression <= 3 %, throughput regression <= 5 %

Phase 2 (week 2 to 6): opt-in general availability
  - Any team can select the new stack via `Job(stack='fsdp2')`
  - Old stack remains default
  - Support both stacks; two weekly office hours slots

Phase 3 (week 6 to 10): flip default
  - New stack becomes the default; old stack requires an opt-out flag
  - Weekly office hours continue

Phase 4 (week 10+): EOL announcement
  - Publish EOL date for old stack (week 22)
  - After EOL: old stack becomes an exception-only opt-out
```

Chapter 6 goes deep on migration plans as a standalone artifact
type — the migration plan is what section 4 of the RFC references
in detail.

### Section 5: Rollback plan

The rollback plan answers two questions: what *command* rolls back,
and what *condition* triggers a rollback. Both must be explicit.

Example: for the FSDP1 → FSDP2 upgrade the rollback plan is "revert
the router config for `stack='default'` from `fsdp2` to `fsdp1`;
existing running jobs on FSDP2 continue to completion; new
submissions land on FSDP1 within 5 minutes." The trigger condition
is "MFU regression ≥ 10 % across two or more teams, or a
correctness bug reported by any team, on any workload."

Rollback plans that read "we will revert if things go wrong" are
not rollback plans. Reviewers should be able to look at your
rollback plan and know, in advance, at what point on the loss
curve or the MFU chart you personally will type the revert
command.

### Section 6: Risk register

The risk register lists the risks *you have thought of* and what
mitigates each. Reviewers will add more; that is expected. A
workable format is a small table:

| Risk                                  | Likelihood | Blast radius              | Mitigation                                                    |
|---------------------------------------|------------|---------------------------|---------------------------------------------------------------|
| MFU regression on team X's config      | Medium     | Single-team               | Canary in Phase 1; benchmark gate; rollback if > 10 %          |
| Checkpoint incompat on resume          | Low        | Multi-team, data loss risk| Dual-write during Phase 2; conversion utility shipped         |
| Capacity impact from lower MFU         | Medium     | Cluster-wide, cost impact | Provision +5 % headroom during Phase 2–3                      |
| Silent numerics drift under FP8        | Low        | Model quality             | Numerics validation suite in Phase 1; loss-curve monitor      |
| Security review not started            | High       | Sign-off blocker          | Owner: @sec on-call; kick off Phase 0 week -2                  |

The security-review risk is on this list intentionally. Every
platform-surface RFC has an implicit security dependency (chapter
3 covers this in depth); the risk register is where you name it
and schedule the review.

### Section 7: Sign-off list

Every peer track that consumes the surface signs off. For a
training-platform RFC that means, at minimum:

- **Platform team lead** — the RFC author's own team.
- **`fine-tuning-engineer-learning`** consumer — signs off that the
  launcher-SDK / metadata contract is preserved.
- **`ai-infra-ml-platform-learning`** — signs off that the
  trained-model artifact contract (registry ingestion) is
  preserved.
- **`ai-infra-mlops-learning`** — signs off that the CI/CD
  post-training hooks continue to work.
- **`ai-infra-performance-learning`** — signs off if the change
  touches kernel integration points.
- **`ai-infra-security-learning`** — signs off on cluster boundary,
  image supply chain, and data-provenance impact.

The rule that keeps sign-offs from becoming rubber-stamps: **active
ack, not passive silence.** Every sign-off name explicitly says
"reviewed, ack" or "reviewed, requesting changes." No sign-off ever
means "did not object in the time allotted." If a peer track's
sign-off does not come in on time, the RFC does not merge — you
escalate to that peer's lead and ask why, and the answer becomes
part of the record.

## Worked example: FSDP1 → FSDP2 framework upgrade

Sketch of an RFC. The full artifact is what exercise-02 asks you to
produce for one of these three canonical classes.

**Context.** The platform ships FSDP1 as the sharding stack for
teams targeting < 70B models. FSDP2 (`fully_shard`, per-parameter
sharding, DTensor-based) is the PyTorch-blessed successor; FSDP1
is on a deprecation path in torch. Constraints: (a) all in-flight
pretraining runs must complete on FSDP1; (b) checkpoint format
changes must have a conversion utility; (c) at least one team must
run an end-to-end pretraining run on FSDP2 before the default
flips.

**Proposed change.** Add `stack='fsdp2'` to `Job`, backed by a new
launcher backend that renders a PyTorchJob spec with a
`torch>=2.5` image. Add a `fsdp2` field to the router config so
per-team overrides are possible. Ship a `dcp-fsdp1-to-fsdp2`
conversion utility that resharded FSDP1 checkpoints into DTensor
format. Introduce a compatibility test suite (numerics + MFU +
checkpoint) that runs weekly on the platform's own CI.

**Alternatives.** (1) Stay on FSDP1 — cost is that FSDP1 is
deprecated upstream, so we accept an unbounded exposure to
upstream churn. (2) Skip FSDP2 and adopt `torchtitan` as the
whole training frontend — larger scope, requires researcher
retraining, delays the deprecation risk fix. (3) Adopt DeepSpeed
ZeRO-3 as the successor — abandons the PyTorch-native path this
platform is standardised on; larger switching cost.

**Rollout plan.** As described in section 4 above. Canary team is
Team A (small fine-tune, low blast radius, low team-side migration
cost). Benchmark gate: MFU regression ≤ 3 %, throughput regression
≤ 5 %.

**Rollback.** Router config revert from `fsdp2` to `fsdp1` for
`stack='default'`. Running FSDP2 jobs continue; new submissions
land on FSDP1 within 5 min. Trigger: MFU regression ≥ 10 % or any
correctness bug.

**Risk register.** As shown in section 6 above.

**Sign-off.** Platform lead, Team A lead (canary), fine-tuning
peer lead, ml-platform peer lead, security peer lead. Performance
peer sign-off not required (no kernel integration change).

## Worked example: A100 → H100 hardware refresh

**Context.** The cluster currently has 512 A100 (80 GB) GPUs across
64 nodes on an HDR IB fabric. A phased H100 (80 GB, NVLink4/NVSwitch3)
delivery brings 256 H100 GPUs online over Q3 as a second Scalable
Unit. The RFC decides how the two hardware pools are exposed to
users and how workloads migrate.

**Proposed change.** Split the cluster into two partitions
(SLURM) or two ClusterQueues (Kueue): `a100-legacy` and
`h100-primary`. Update the launcher SDK to accept
`Job(hardware='h100')` and default to `h100-primary` for new
pretraining runs. Update the NCCL topology file for the H100 SU
(NVLink4 semantics). Update Docker base images for H100
(CUDA 12.x, cuDNN 9.x). Publish per-hardware MFU baselines so
teams can budget correctly.

**Alternatives.** (1) Mixed pool — allow single jobs to span A100
and H100 nodes. Rejected: the collective performance is bounded by
the slower generation, and the mixed-precision recipes diverge
(FP8 on H100 vs. BF16 on A100). (2) Retire A100 immediately —
rejected: in-flight A100 runs would be forced to migrate mid-run.
(3) Keep A100 for interactive and research only — this is a subset
of the proposed change and is folded in.

**Rollout plan.** Phase 0: platform smoke jobs on the H100 SU.
Phase 1: canary team runs a 7B pretraining, publish MFU baseline.
Phase 2: all new `training-high` submissions default to
`h100-primary`. Phase 3: A100 partition drops in priority ceiling
from `training-high` to `training-normal`. Phase 4: A100 EOL date
published; migrate the last legacy runs.

**Rollback.** Router config revert; the two partitions coexist
throughout Phases 1–3, so rollback is a config change. Trigger:
H100 partition throughput below A100 baseline (indicates
misconfiguration).

**Risk register.** Capacity: H100 SU is smaller in count than
A100 SU; migration windows must be planned around this. Cost: H100
$/GPU-h is higher; provide updated cost model (mod-109). Security:
new base image, new CVE surface — coordinate with the security
peer track.

**Sign-off.** Platform lead, fabric SRE, storage SRE, all consumer
teams' leads, cost/finance owner, ml-platform peer lead
(different serving-side H100 image), security peer lead.

## Worked example: Lustre → WEKA storage migration

**Context.** The current cluster uses Lustre for the parallel
filesystem. WEKA has been benchmarked internally at 2× the
loader-side small-file throughput and 1.4× the checkpoint-write
throughput for our access patterns. Constraints: (a) the
migration must not stall any pretraining run; (b) checkpoint
integrity must be verifiable across the boundary.

**Proposed change.** Deploy WEKA as a second parallel tier at
mount `/wk`, alongside Lustre at `/scratch`. Update the dataset
service to stage new shard sets to `/wk` by default; keep the
Lustre mount for legacy shards. Update the checkpoint service to
default to `/wk` for new runs, with a `checkpoint_path` override
on `Job`. Publish a data-migration schedule per dataset owner.

**Alternatives.** (1) Retain Lustre and expand capacity — rejected:
throughput ceiling on our access pattern is a real limit, not a
capacity limit. (2) Adopt FSx for Lustre with S3 linking —
rejected: not applicable to the on-prem shape. (3) Move
checkpoints only to WEKA, keep datasets on Lustre — rejected:
splits the runbook and adds a second failure surface without
resolving the loader-side bottleneck.

**Rollout plan.** Phase 0: WEKA cluster stand-up + acceptance test
against the mod-105 chapter 8 fabric runbook. Phase 1: dataset
service supports both mounts; new datasets stage to WEKA. Phase 2:
checkpoint service supports both; new runs default to WEKA. Phase
3: dataset-owner-driven migration of hot datasets. Phase 4: Lustre
tier drains; publish EOL date.

**Rollback.** Router config revert to `/scratch` for both dataset
and checkpoint paths. In-flight runs continue on WEKA; new runs
land on Lustre. Trigger: WEKA availability below Lustre's baseline;
data-loss incident; sustained loader stall > 10 %.

**Risk register.** Availability: WEKA is a new failure surface;
require 4 weeks of steady-state before defaulting. Cost: WEKA
$/TB is different; folded into mod-109 cost model. Security:
new mount, new access-control surface — security peer sign-off
required. Migration windows must be reconciled with in-flight
runs; no run gets forced-migrated mid-checkpoint-cycle.

**Sign-off.** Platform lead, storage SRE (owner), fabric SRE
(load impact on IB), dataset service owner, checkpoint service
owner, security peer lead, cost/finance owner.

## Anti-patterns

The RFCs that get accepted but fail in production usually have one
of these tells:

- **Solution-first context.** The reviewer cannot separate "the
  problem" from "the author's preferred fix."
- **Rollout plan is a single date.** No phases, no canaries, no
  exit criteria — no way to know whether the rollout is going
  well until it either works or doesn't.
- **Rollback plan is aspirational.** "We will monitor and revert
  if needed" — no specific command, no specific trigger.
- **No cost / capacity impact.** Every framework or hardware
  change moves the MFU baseline; every storage change moves the
  $/TB. If your RFC does not update the mod-109 cost model,
  someone else will discover this during month-end and page you.
- **Sign-offs collected after merge.** The sign-off list is a
  gate, not a formality.
- **No mention of the security peer track.** If the RFC touches
  images, mounts, secrets, network policy, or data flow across
  the cluster boundary, security signs off. Full stop.

## Summary

- Any change to the training platform surface that a downstream
  team can notice gets an RFC. This is the primary technical
  artifact of level-35 work.
- The seven-section template (context, proposed change,
  alternatives, rollout, rollback, risks, sign-off) is your
  minimum. Skip a section only with an explicit one-line
  justification.
- The rollout plan is phased with named canaries and exit
  criteria. The rollback plan is a specific command and a
  specific trigger, not a wish.
- Every peer track that consumes the surface actively signs off.
  Passive silence is not a sign-off.
- Kubernetes KEPs and Rust RFCs are excellent public prior art.
  Steal freely from their structure.
- Three canonical RFC classes to practise on: framework upgrade
  (FSDP1 → FSDP2), hardware refresh (A100 → H100), storage
  migration (Lustre → WEKA). Chapter 6 goes deeper on the
  migration-plan artifact that section 4 references.
