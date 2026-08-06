# Multi-Tenant Training-Cluster Architecture

Mod-104 chapter 8 gave you the four levers of scheduler policy:
quotas, fair-share, priority, and gang preemption. This chapter
sits one level up. It is the platform architect's chapter: given
5–15 downstream teams sharing one training cluster, how do you
design the quota table, the priority-class ladder, the on-call
rotation, and the escalation ladder so that contention resolves
mechanically, capacity is fully committed but not
over-committed, and the platform team is not on Slack refereeing
every hour?

The scheduler primitives are the *building blocks*. The
multi-tenant *contract* is a different artifact: signed by
Finance, by every consuming team's engineering director, and by
the platform lead. It changes quarterly. It is what capacity
planning outputs, what the on-call desk reads at 03:00, and what
the incident review of chapter 6 cites when a run gets
preempted.

## The capacity contract in one page

The core artifact of this chapter is a *capacity contract* — a
one-page table plus a paragraph of policy per row. The table
looks like this:

| Team          | Guaranteed | Cap  | Cohort   | Default priority class | Preemptible above |
|---------------|-----------:|-----:|----------|------------------------|------------------:|
| foundations   |        192 |  384 | frontier | training-high          |               192 |
| finetune      |         64 |  256 | frontier | training-normal        |                64 |
| post-training |         32 |  128 | frontier | training-normal        |                32 |
| research      |         32 |  256 | research | research               |                 0 |
| platform-dev  |         16 |   64 | platform | preemptible            |                 0 |

*(Numbers are illustrative; real tables are keyed to specific
accelerator SKUs and topologies. This is a starter shape, not a
prescription.)*

Read the rows top-to-bottom before starting a policy discussion.
Every row is a team-quarter commitment; every column is a lever
someone on your team has to defend when a job gets preempted.

The paragraph of policy per row is the "what's borrowable, from
whom, under what conditions, and how much notice does the lender
get before being reclaimed" contract. Keep it short — a sentence
each — but write it. Verbal quota agreements are the source of
most Sev-2 platform incidents.

## Sizing the guarantee

The guarantee is the number a team plans against. It is set
against three inputs:

- **Historical demand.** The trailing-90-day 7-day-average GPU-
  hours the team consumed. Not their peak; the sustained.
  Mod-108's metadata store (chapter 7 of that module) is the
  source.
- **The next quarter's committed roadmap.** Every feasibility
  study (mod-109 chapter 7) the team has approved for the
  quarter. Sum the GPU-hours. This is the *forward* demand.
- **Cluster-wide headroom.** The sum of guarantees across all
  teams cannot exceed the cluster's steady-state deliverable
  capacity, which is (physical GPUs) × (goodput target) ×
  (uptime target). Typical goodput target: 0.85–0.95 for a
  mature training platform, per Llama 3 herd of models
  (Grattafiori et al., 2024, §3.3.2).

Two properties every guarantee should satisfy:

- **Sum of guarantees < physical capacity.** Never over-commit
  guarantees. A team whose guarantee cannot be filled during a
  routine incident lost the plot of what "guarantee" means.
- **Sum of caps > 2× physical capacity.** Caps should encourage
  borrowing. If every cap equals every guarantee, no team can
  ever borrow, and the cluster's average utilisation drops.

If the arithmetic does not work, do not fudge the guarantees; do
the mod-109 chapter 3 conversation with product leadership about
either buying more capacity or cutting a team's roadmap.

## Priority classes: the "what breaks first" contract

Priority classes are covered mechanically in mod-104 chapter 8.
At the platform-architecture altitude the question is different:
how many classes, and what does each guarantee.

Three or four classes is the useful range. Fewer and there is no
lever; more and the semantics collapse into "we sorted the
classes by number and half the org does not know what they
mean".

A workable set:

- `critical` — a demonstrable production or contractual deadline.
  Preempts every other class; cannot be preempted. Requires
  written sign-off from the platform lead per use. Kept scarce
  by process, not by policy — one abuse and the class is
  meaningless.
- `training-high` — the sanctioned large runs. Preempts normal
  and below. Preemptable only by critical. Typical use: the
  team's active pretraining or SFT run against a next-quarter
  ship date.
- `training-normal` — the default. Preemptable by high or
  critical.
- `research` / `preemptible` — free capacity for sweeps and
  experiments. Interruptible; expected to checkpoint.
  Everything above preempts.

The class is a *contract*, not a hint. The on-call knows what
each class implies without asking:

- **What guarantees the class carries** (queue-time target, no-
  preemption window if any).
- **Who authorised the class** (self-serve up to `training-
  normal`; PR from the team lead for `training-high`; platform
  lead per-use for `critical`).
- **What the preemption behaviour is** (auto-requeue after
  checkpoint restore is the standard; `PreemptCancel` should be
  reserved for `preemptible` only).

The class table lives in the platform docs next to the capacity
contract. Every team's onboarding doc points to it.

## Gang preemption at platform altitude

Gang preemption is a *correctness* property (mod-104 chapter 8:
partial preemption wedges NCCL). At platform altitude, three
questions matter that the scheduler chapter did not answer.

**Question 1 — what is the gang unit for each team?** For most
teams, it is one training run: one PyTorchJob, one Slurm job,
one Kueue Workload. For teams running sweeps, it may be a whole
sweep (Ray Train, Kubeflow Katib). Whatever the unit, the
platform team publishes the mapping — otherwise a sweep gets
partial-preempted mid-config and looks like corruption.

**Question 2 — what is the notice window?** Volcano's preempt
plugin and Kueue's `reclaimWithinCohort` are semantically
instant. In practice a training run needs a graceful window to
flush a checkpoint before dying. The standard is:

- **60 seconds** grace period on preemption. Trainer emits
  `SIGTERM`; trainer writes an async DCP checkpoint (mod-106
  chapter 2) to hot storage; trainer exits.
- If the trainer does not exit in 60 s, `SIGKILL`. The
  reproducibility bundle (mod-108 chapter 5) and the last
  successful sync-checkpoint are enough to resume from the
  previous saved state; the in-flight 60 s of work is lost.

Kueue supports `terminationSeconds` and `deletionTimestamp`;
Volcano supports `preemptGracePeriodSeconds`. Set both. The
Slurm equivalent is `KillWait=60`.

**Question 3 — what is the reclaim-priority order?** When
multiple lenders' capacity is being reclaimed at once, whose
gets evicted first? The default answer, in order:

1. `preemptible` jobs (regardless of team).
2. `research` jobs (regardless of team).
3. `training-normal` jobs from the team with the *smallest*
   deficit against its guarantee (i.e., the team least under-
   served takes the hit first).
4. `training-normal` jobs from other teams.
5. `training-high` jobs are only preempted by `critical`.

This ordering is a design choice, not a scheduler default —
Kueue's `reclaimWithinCohort: Any` will reclaim from
lower-priority within the cohort, but "which team first" is
policy you set with the borrowing-limit and lending-limit
knobs. Write the ordering in the platform docs so no team is
surprised when their job is the one that dies.

## On-call structure for 5–15 teams

At five teams you can run a single on-call rotation across the
whole platform. At fifteen you cannot. The failure mode of a
single rotation at scale is that the on-call has to be
simultaneously an FSDP expert, a NCCL expert, a storage expert,
and a Kubernetes expert. That person does not exist; if they
did, they would not be on a rotation.

The two workable structures:

### Structure A — Primary / secondary on-call

- **Primary.** Platform on-call. Owns the pager; is the
  first responder for every incident. Rotates weekly across the
  platform team.
- **Secondary.** Domain expert on-call. One rotation per domain
  (networking/storage, framework, scheduler). Paged by the
  primary when the incident lands in their domain.
- **Escalation.** Platform lead (business hours) or the
  on-call platform manager (after hours). Paged by the primary
  when the incident is Sev-1 or when the secondary cannot make
  progress in 30 minutes.

Works up to roughly 10 teams and a single-cluster platform.
Beyond that the primary spends most of their week
paging the secondary; the primary rotation is not carrying
knowledge.

### Structure B — Federated on-call

- **Team on-call.** Each consuming team runs its own on-call
  for jobs it owns. First responder for "my job died".
- **Platform on-call.** Runs the pager for cluster-wide
  incidents (scheduler down, storage down, network partition).
  Not paged for individual job deaths.
- **Domain rotation.** Owned by the platform team; consulted by
  team on-call.

Works at 10+ teams and multi-cluster platforms. Requires the
teams to have the muscle to on-call for their own jobs, and
requires the platform-side runbook (mod-106 chapter 5, mod-108
chapter 6) to be complete enough that the team on-call can use
it without a platform engineer.

## The escalation ladder

Every incident maps to a Sev level; every Sev level maps to a
notification, a response time, and an escalation trigger.
Publish this ladder before you need it.

| Sev | Definition                                                | Detect → page | Page → ack | Ack → mitigate | Escalate if… |
|----:|-----------------------------------------------------------|--------------:|-----------:|---------------:|--------------|
|   1 | Multi-team outage (scheduler, storage, network partition) |       ≤ 5 min |    ≤ 5 min |       ≤ 30 min | mitigate > 15 min |
|   2 | Single-team outage (their run cannot make progress)       |      ≤ 10 min |   ≤ 10 min |        ≤ 2 hrs | mitigate > 45 min |
|   3 | Degradation (goodput < SLO, but progress continues)       |         ≤ 1 h |     ≤ 1 hr |     next-day   | recurs 3× in 7 d  |
|   4 | Latent issue (SDC signal, ECC drift, no user impact yet)  |        ≤ 24 h |          — |     next-week  | trend worsens     |

Two properties this ladder must have:

- **Sev is set at page time, not later.** The on-call sets it
  from the runbook when acknowledging. Retro-adjusting Sev after
  the fact hides SLA breaches.
- **Every ladder rung names a person or role.** "Escalate to
  platform lead" is a role; the current holder is named in the
  platform docs and paged by a rotation. "Escalate to a senior
  engineer" is not a rung; that is a room to walk into.

The full incident retrospective machinery is chapter 6. This
chapter's job is to set the ladder that the retrospective
measures against.

## Cluster-wide invariants the platform enforces

Beyond the per-team contract, the platform holds a small set of
cluster-wide invariants. Every consuming team inherits them; if
a team's job violates them, the platform rejects the submission
at admission time.

- **Every job carries a `run_id` label** — the mod-108 chapter 7
  metadata store's key. Admission controller (Kueue's `Workload`
  webhook, Slurm's `job_submit` Lua plugin) rejects submissions
  without one.
- **Every job carries a team label matching an active cohort**
  — no ghost jobs from disbanded teams.
- **Every job carries an image digest, not a floating tag** —
  the OCI-digest constraint from mod-108 chapter 5. Prevents
  half-year-later "which image ran this?" archaeology.
- **Every job that requests `training-high` or `critical`
  attaches a reference to an approved feasibility study**
  (mod-109 chapter 7). No large-run capacity commitments without
  a written cost estimate.
- **Every job at gang size ≥ N attaches a checkpointing spec.**
  N is a platform-set threshold (typical: 8 nodes, 64 GPUs).
  Below it, opt-in; above it, mandatory. Prevents
  multi-day runs with no restart.

Admission-time enforcement is the point. A late-caught missing
`run_id` on day 12 of a run is a permanent data loss; a
rejected submission on day zero costs the submitter a minute.

## Composing the whole: a starter template

Below is what a working platform-architecture doc for a mid-
sized org looks like at the top level. It is deliberately short.

```
# Foundation Training Cluster — Multi-Tenant Contract v3.1

Owning team:      Training Platform
Effective:        2026-Q3 through 2026-Q4
Sign-off:         Platform Lead, Finance, Foundations EM,
                  Fine-Tuning EM, Research EM, Post-Training EM
Physical capacity: 512 H100 SXM5 GPUs, 64 nodes, NDR IB fabric
Goodput target:   0.90 (90th-percentile weekly)

## 1. Capacity table
(see table above; per-team rows and cohorts)

## 2. Priority classes
critical / training-high / training-normal / research /
preemptible.  Semantics per class linked to runbook §2.

## 3. Preemption policy
- Grace period: 60 s (SIGTERM → checkpoint → SIGKILL).
- Reclaim priority: preemptible → research → training-normal
  (from smallest-deficit team first) → training-high (critical
  only).
- Gang unit: one Workload / PyTorchJob / Slurm job.

## 4. On-call and escalation
Structure A (primary + secondary + domain rotation).
Sev ladder in runbook §4.  Rotation calendar in PagerDuty.

## 5. Cluster-wide invariants
run_id, team label, image digest, feasibility study for
training-high+, checkpointing spec for gang ≥ 64 GPUs.

## 6. Amendment process
Any change to §1 or §2 requires an RFC (chapter 3 of mod-110)
and a re-sign of the sign-off block. §3–§5 changes require
platform-team consensus and a two-week notice to consumers.
```

That is the whole contract. If your platform's contract does
not fit on one page, it is not a contract; it is prose.

## Common failure modes

- **Guarantees that sum above physical capacity.** When two
  teams simultaneously ask for their guarantee, one of them
  loses; the "guarantee" turned out to be marketing.
- **Priority classes without preemption semantics.** The class
  number sorts the queue but does not evict; a
  `training-high` job waits behind a `research` job that
  claimed the GPUs first. `training-high` is meaningless.
- **Gang preemption without a grace period.** The trainer dies
  mid-write; the last checkpoint is corrupt; the run restarts
  from an older checkpoint and loses hours. Set the 60 s window.
- **On-call rotation that pages the same person 60% of the
  time.** The rotation is a fig leaf; that person is the on-call.
  Fix the runbook, not the rotation.
- **Admission controller not enforcing invariants.** The
  `run_id` label is "recommended" and half the jobs miss it. Six
  months later mod-108's metadata store has holes; nobody can
  reconstruct history.
- **Capacity contract updated in Slack.** A team's guarantee
  moves in a DM. The Finance quarterly review does not match
  actual usage; the RFC of chapter 3 is skipped. Update the
  contract in a versioned doc, or do not update it.

## Summary

- The capacity contract is a one-page table plus a paragraph of
  policy per row. Signed quarterly by Finance and every
  consuming team's engineering director. Change requires an RFC
  (chapter 3).
- Sum of guarantees < physical capacity; sum of caps > 2×
  physical capacity. Otherwise the platform is
  over-committed or utilisation dies.
- Three or four priority classes with explicit preemption
  semantics; `critical` kept scarce by process. The class table
  is the "what breaks first" contract on-call reads at 03:00.
- Gang preemption at platform altitude has three knobs: what is
  a gang, what is the notice window (typical 60 s), and what is
  the reclaim-priority order across teams.
- On-call structure A (primary + secondary + domain) works up
  to ~10 teams; structure B (federated with per-team on-call)
  works beyond. Both need a written Sev ladder with named
  roles.
- The platform enforces a small set of cluster-wide invariants
  at admission time: `run_id`, team label, image digest,
  feasibility-study reference for large runs, checkpointing
  spec for large gangs.
- Every one of these is a written artifact. The scheduler
  primitives (mod-104 chapter 8) are the plumbing; this
  chapter's artifacts are the contract.
