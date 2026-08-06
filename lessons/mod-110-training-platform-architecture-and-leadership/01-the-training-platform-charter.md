# The Training-Platform Charter

The previous nine modules taught you to build a *training run* —
convergent, observable, priced, and near-peak. This module teaches
you to build the *platform* those runs live on, and to lead the
cross-team org that consumes it. That is a distinct discipline. A
platform is not a big training run; it is a product with users,
contracts, and a roadmap, whose users happen to be other engineers.

This chapter sets the frame every other chapter builds on: who the
platform's users are, what four contracts the platform owes them,
and what "level-35 altitude" (the AICG track's staff-engineer /
principal-engineer band for this role) actually looks like in
daily practice.

## Platform as product

A training platform is a product. It has:

- **Users** — the ML researchers, fine-tuning engineers, and
  post-training engineers who submit workloads. They do not care
  about your NCCL tuning; they care about "did my job launch, did
  it finish, did it reproduce, and did it cost what I was told".
- **A pricing surface** — GPU-hours per team, priced against
  guarantee vs. borrowed capacity, exposed to finance monthly.
  Chapters 3 and 4 of mod-109 own the arithmetic; this module
  owns the pricing *contract* with the consuming teams.
- **A roadmap** — the next framework upgrade, the next hardware
  generation, the next storage migration. Users need to see it
  before it lands; chapter 3 (RFCs) and chapter 7 (migration
  planning) are the mechanisms.
- **An SLA** — goodput floor, queue-time p50/p95, incident
  response time. Chapter 2 owns the multi-tenant fairness
  arithmetic; this chapter names the SLA as the contract.
- **A support surface** — on-call rotation, escalation ladder, a
  runbook that owns "the researcher's job died at 03:00, what
  now?". Chapter 6 (incident review) is the retrospective side
  of the same surface.

If you cannot answer "who are my users, what do I promise them,
and what happens when I break the promise", you do not have a
platform yet — you have a training cluster with a login shell.

## Who the users actually are

Every platform-facing artifact in this module names a user
segment. Four segments recur across the industry; know them by
name before writing an RFC.

- **Foundation / pretraining researchers.** Own the large runs.
  Care about `μ_sus`, elastic recovery, checkpoint durability,
  and the reproducibility bundle. Their unit of work is the
  multi-week pretraining run. They are the most expensive users
  and the loudest voice in the RFC review.
- **Fine-tuning engineers.** Own the derivative runs — SFT,
  DPO/PPO/GRPO, adapter training, continued pretraining on
  domain corpora. Their unit of work is a run of hours to days;
  they run many. The AICG `fine-tuning-engineer` role is the
  peer track that consumes this platform (chapter 4).
- **Post-training / evaluation engineers.** Consume checkpoints
  the platform produces, run evals, run RLHF loops. Their
  contract with the platform is checkpoint discoverability and
  format stability. The AICG `ai-infra-mlops-engineer` role
  owns the CI/CD side of this (chapter 4).
- **On-call platform engineers.** Your own team. Consume the
  platform through the incident and change-management surfaces.
  If the platform is hostile to on-call, everything else
  degrades.

Never write an RFC or a migration plan without a "users affected"
section listing which of the four segments moves and how.

## The four contracts

Every mature training platform owes its users four contracts.
Each contract has an artifact, an SLA, and an owner.

### Contract 1 — the guarantee (capacity contract)

The platform guarantees each team a floor of resources (H100s or
equivalent, storage IOPS, egress bandwidth) that they can plan
against, and a ceiling that bounds their runaway. Above the
floor, borrowable capacity is reclaimable by neighbours.

- **Artifact.** The quota table (chapter 2), signed off by
  Finance and by every consuming team's engineering director.
- **SLA.** Guaranteed capacity is available within a bounded
  queue time (typical: seconds when idle, minutes with
  preemption). Borrowed capacity is best-effort.
- **Owner.** Platform team, negotiated with each consuming team
  quarterly.

### Contract 2 — the hand-off (interface contract)

The platform sinks and sources artifacts from peer tracks:
checkpoints to the serving/registry team, dataset snapshots from
the data team, kernels from the performance team, provenance
manifests to the security team. Chapter 4 makes each hand-off
explicit.

- **Artifact.** A hand-off document per peer track, naming the
  format, the SLA, and the escalation path.
- **SLA.** Format stability across the compatibility window
  agreed in chapter 7's migration plan.
- **Owner.** Platform team + the peer team; both sign.

### Contract 3 — the incident contract

When the platform breaks, users need to know within a bounded
time, get a status page, and receive a postmortem. Chapter 6 is
the retrospective mechanism.

- **Artifact.** The incident postmortem, published within a fixed
  window (typical: five business days for a Sev-2 outage; two
  business days for a Sev-1).
- **SLA.** Detect within X minutes, page on-call within Y,
  publish status within Z, deliver postmortem within N business
  days. Named explicitly; measured against actuals monthly.
- **Owner.** Platform on-call, escalated to platform lead.

### Contract 4 — the roadmap (change-management contract)

Frameworks upgrade. Hardware refreshes. Storage migrates. Every
change to the platform surface has to land with a compatibility
window and a rollback, published *before* it lands. Chapter 3
(RFCs) and chapter 7 (migration plans) are the mechanisms.

- **Artifact.** RFC per change; migration plan per surface
  change; deprecation notice per removed capability.
- **SLA.** Named compatibility window (typical: two quarters
  minimum for a framework major version; three quarters for a
  hardware generation cut-over).
- **Owner.** RFC author (platform team) + reviewers from every
  affected consuming team.

These four contracts are the vocabulary of the rest of the
module. Every chapter refines one of them.

## The level-35 altitude

The AICG training-pipeline engineer role at level 35 is not a
level of technical difficulty; it is a level of *scope*. The
level-35 engineer is expected to make decisions that:

- Bind multiple downstream teams for multiple quarters.
- Move capital (hardware refresh, capacity commitments).
- Set the contract other engineers will be evaluated against.
- Are legible to non-engineering stakeholders (finance,
  legal, product leadership) without a translator.

Concretely, at level 35 you own:

- **The build-vs-buy decision** (chapter 5). Whether to run
  Megatron-LM in-house, integrate NeMo, sit on Databricks/Mosaic
  or Together's managed training — this is a level-35 call, not
  an IC-level tool choice. It is quoted to leadership, defended
  in a memo, and re-visited annually.
- **The framework major-version decision** (chapter 3, chapter 7).
  Cutting over from FSDP1 to FSDP2 or from Megatron-LM to
  torchtitan is a multi-quarter migration that binds every
  consuming team. The RFC gates it; the migration plan lands it.
- **The hardware-generation decision.** A Hopper-to-Blackwell
  cut-over touches every recipe, every MFU number, every dollar
  in mod-109's feasibility studies. The RFC is authored at this
  altitude even though the operational plumbing lives in mod-104
  and mod-105.
- **The incident review of a real outage** (chapter 6). Blameless,
  authoritative, action-item-bearing. Published to leadership;
  cited in future RFCs.
- **The org contract with peer platforms** (chapter 4). Which
  team owns what, what format flows across each interface, what
  the escalation path is when something breaks.

Below level 35, an engineer implements the pieces of the above.
At level 35, they own the argument for the whole.

## What "leadership" means in the platform context

Leadership on a training platform is not people-management (that
is EM). It is *technical direction* — the ability to author a
document that ten peer engineers agree to be bound by, and then
to hold them and yourself to it. In practice, the outputs are:

- **RFCs and design docs** — chapter 3.
- **Hand-off contracts with peer teams** — chapter 4.
- **Build-vs-buy memos** — chapter 5.
- **Incident reviews** — chapter 6.
- **Migration plans** — chapter 7.

Every one of these is a written artifact. A training-platform
lead who cannot write is a training-platform lead who cannot
lead. This module treats writing as a core engineering skill;
every exercise ships a written artifact.

## Anti-patterns the module is written against

Three patterns recur across the industry that this module exists
to prevent.

### Anti-pattern 1 — the invisible policy

The platform team makes a decision in Slack ("we're upgrading to
PyTorch 2.6 next week"). No RFC, no compatibility window, no
migration plan. Two teams break. Three months later, the same
pattern happens with FSDP2. There is no institutional memory of
the last cutover, so it hurts as much as the first one did.

The fix is chapter 3's RFC template and chapter 7's migration
plan. Every user-visible change lands as a document; the
document is discoverable a year later.

### Anti-pattern 2 — the heroic on-call

There is no runbook, no escalation ladder, no priority-class
policy. Every incident is resolved by paging the one senior
engineer who knows the cluster. When they leave, the platform
breaks in ways nobody else can debug.

The fix is chapters 2 and 6. Chapter 2 makes the policy
mechanical (priority classes and gang preemption resolve most
contention without a human). Chapter 6 makes every incident a
learning artifact that raises the on-call floor.

### Anti-pattern 3 — the shadow build

A team is not happy with the platform, so they stand up their
own SLURM cluster with their own container image and their own
NCCL tuning. Six months later, they hit an incident. Nobody on
the platform team can help; nobody on their team has the muscle
to run a large-cluster incident review. The org has
double-counted its GPU capacity and single-counted its
expertise.

The fix is contract 2 (hand-off) and contract 4 (roadmap). If
the platform's roadmap credibly serves the team's next quarter,
they do not fork. Chapter 4 defines the hand-off; chapter 5
(build-vs-buy) is the tool for checking whether the shadow
build's premise is actually correct.

## What this module does not cover

- **The individual training run.** All of mod-101 through
  mod-108. This module treats the run as the platform's unit of
  work, but does not re-derive the run's internals.
- **The cost arithmetic for a single run.** Owned by mod-109.
  This module consumes mod-109's per-run studies and aggregates
  them into quarterly capacity plans (chapter 2).
- **The scheduler implementation.** Owned by mod-104. This
  module names quotas, priority classes, and gang preemption as
  the four levers of the multi-tenant contract; the plumbing
  that implements them is mod-104 chapter 8.
- **People management, hiring, performance calibration.** Out of
  scope; this module is about *technical* leadership.

## Summary

- A training platform is a product with users, contracts, and a
  roadmap. Every chapter of this module refines one of the
  contracts.
- Four contracts: capacity (chapter 2), hand-off (chapter 4),
  incident (chapter 6), roadmap (chapter 3 + chapter 7). Each
  has an artifact, an SLA, and an owner.
- Level 35 owns the arguments that bind multiple teams for
  multiple quarters: build-vs-buy, framework major versions,
  hardware generations, cross-team org contracts, incident
  reviews. Every one is a written artifact.
- The three anti-patterns the module is written against —
  invisible policy, heroic on-call, shadow build — each have a
  chapter that supplies the counter-practice.
- Writing is a core engineering skill at this altitude. Every
  exercise in this module ships a written artifact, not code.
