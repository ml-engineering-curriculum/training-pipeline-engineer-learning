# exercise-01: Multi-Tenant Training-Cluster Architecture

**Estimated effort:** 3 hours

## Objective

Author the full **capacity contract** for a realistic multi-tenant
training cluster: quota table, priority-class ladder,
preemption policy, on-call structure, Sev ladder, and
admission-time invariants. Deliverable: one document
(`capacity-contract-v1.md`), one YAML bundle
(`platform-config/`) that implements the policy on either
Kubernetes (Kueue + Volcano) or Slurm, and a short defense
brief.

## Prerequisites

- Chapter 1 (four contracts) and chapter 2 (multi-tenant
  architecture) of this module.
- Mod-104 chapter 8 (quotas / fair-share / priority / gang
  preemption). If you have not read it, do so first — this
  exercise is the platform-altitude wrapper around its
  mechanics.
- A working knowledge of either Kubernetes (Kueue, Volcano,
  Kubeflow Training Operator) *or* Slurm (`sacctmgr`, QoS,
  `slurm.conf`). Pick one stack for this exercise; you do not
  have to author both.
- Access to a cluster to apply the config against is ideal but
  not required — a syntax-checked YAML/config bundle is
  acceptable.

## Problem statement

You are the training-platform lead at a company with the
following consuming teams and roadmap. The physical cluster is
**384 H100 SXM5 GPUs across 48 nodes** on an NDR InfiniBand
fabric.

| Team          | Headcount | Current use    | Q4 roadmap ask                             |
|---------------|----------:|----------------|--------------------------------------------|
| foundations   |         6 | 128 GPUs steady | 300B pretraining run, ~192 GPUs / 6 weeks |
| finetune      |        10 | 64 GPUs steady  | SFT + DPO sweeps on the new base           |
| post-training |         4 | 32 GPUs bursty  | RLHF pipeline against the base             |
| research      |         8 | 32 GPUs bursty  | LoRA sweeps + long-context experiments     |
| platform-dev  |         3 | 16 GPUs bursty  | migration testing (see exercise 2)         |

Total steady demand: **272 GPUs** against **384 physical**. The
Q4 roadmap ask exceeds the current use. Your job is to design
the contract that allocates the cluster fairly, encodes what
happens under contention, and gives every team a floor they
can plan against.

## Requirements

Ship one directory `mod-110-ex01/` with:

- `capacity-contract-v1.md` — the written contract.
- `platform-config/` — a YAML or config bundle that implements
  the contract on your chosen stack.
- `defense-brief.md` — a short (~1 page) defense.

### 1. The capacity contract (`capacity-contract-v1.md`)

Length target: 1–2 pages. Include every subsection listed in
chapter 2's "composing the whole" starter template:

- **Quota table.** One row per team. Columns: guaranteed, cap,
  cohort, default priority class, preemptible-above. Numbers
  in GPUs. Sum of guarantees must be strictly less than
  physical capacity; sum of caps should exceed 2× physical.
  Justify each number in a bullet under the table.
- **Priority classes.** 3–5 classes with names, integer
  priorities, preemption relations, and the "who can
  authorise" rule per class. Table format.
- **Preemption policy.** Gang unit, notice window (60 s
  default; justify if different), and the reclaim-priority
  ordering across teams and classes.
- **On-call structure.** Choose structure A (primary +
  secondary + domain) or structure B (federated). Justify
  against the 5-team size.
- **Sev ladder.** Same table format as chapter 2's ladder;
  every rung names a role (not a person). Detect→page,
  page→ack, ack→mitigate, escalate-if.
- **Cluster-wide invariants.** At minimum: `run_id`, team
  label, image digest, feasibility-study reference for
  training-high+, checkpointing spec for gang ≥ 64 GPUs.
- **Amendment process.** Which changes require an RFC; who
  signs the re-signed contract.
- **Sign-off block.** Named roles; the placeholders are fine
  (`<Platform Lead>`, `<Foundations EM>`, etc.), but every
  role needed for the amendment process is enumerated.

### 2. The platform config (`platform-config/`)

Implement the policy on either stack.

**If Kubernetes (Kueue + Volcano):**

- `priority-classes.yaml` — one `PriorityClass` per class.
- `cohorts-and-queues.yaml` — a `ClusterQueue` per team with
  `nominalQuota`, `borrowingLimit`, `lendingLimit`, and
  `preemption.reclaimWithinCohort`. Cohorts as designed in the
  contract.
- `volcano-scheduler-config.yaml` — the plugins enabled
  (`proportion`, `preempt`, `gang`, `drf`), one `Queue` per
  ClusterQueue, weights aligned with the fair-share intent.
- `admission-webhook/` — either a `ValidatingAdmissionPolicy`
  (CEL) or a real webhook stub that enforces at least three of
  the cluster-wide invariants (pick `run_id`, team label,
  image digest at minimum). Include a short README with the
  install command.

**If Slurm:**

- `slurm.conf` — the multi-factor priority section,
  `PriorityWeightFairshare`, `PriorityWeightQOS`,
  `PriorityDecayHalfLife`, `PreemptType`, `PreemptMode`.
- `sacctmgr-init.sh` — a script that creates the accounts, QoS
  definitions, and per-user associations from the quota table.
- `job_submit.lua` — a `job_submit` plugin that enforces the
  cluster-wide invariants (`run_id` present, team account
  matches submitter, image digest set, feasibility study
  reference for training-high+).

The YAML/config must be syntax-valid; run it through the
appropriate linter (`kubectl apply --dry-run=server` for
Kubernetes, `slurmd -C` and `scontrol reconfigure --dry-run`
for Slurm) before submitting.

### 3. The defense brief (`defense-brief.md`)

Length target: 1 page. Answer, in order:

- **Why these guarantee numbers.** Reference each team's
  current use and Q4 roadmap.
- **What happens when foundations, finetune, and research all
  submit their full ask at once.** Walk through the scheduler's
  decision.
- **What happens when a `training-high` job needs to preempt a
  `training-normal` job mid-flight.** Cite the grace period and
  the reclaim ordering.
- **What the on-call sees at 03:00 when a Sev-2 fires.**
  Reference the runbook and the escalation rung.
- **What breaks first if the physical cluster loses 8 nodes
  (16.7% capacity).** Which contract properties still hold;
  which need renegotiation.

## Starter guidance

- **Do the arithmetic first, in a scratch spreadsheet.** Sum
  guarantees; sum caps; check invariants. Only start writing
  the contract once the numbers work.
- **Do not start with priority-class internals.** Priority
  classes are the "what breaks first" contract; get the
  guarantee/cap numbers first, then decide who preempts whom.
- **Copy chapter 2's starter template verbatim, then edit.**
  The template's shape is the standard; deviating is fine but
  do so intentionally.
- **Run the config through a linter before writing the
  defense brief.** A broken YAML file is a broken contract.
- **Cite mod-104 chapter 8 for scheduler-primitive semantics.**
  This exercise does not re-explain what a `PriorityClass` is;
  it *uses* one.

## Acceptance criteria

- `capacity-contract-v1.md` includes every subsection listed
  above; the quota table's guarantees sum to less than 384;
  the caps sum to more than 768; each row justified.
- Priority classes: at least 3, at most 5; every class has an
  explicit preemption relation and authorisation rule.
- Preemption policy names the gang unit, the grace period
  (60 s or explicitly justified alternative), and the
  reclaim-priority ordering across teams and classes.
- On-call structure named (A or B) and defended against the
  5-team scale.
- Sev ladder table present with all four levels; every rung
  names a role.
- Cluster-wide invariants: at least 5 listed; at least 3
  enforced in the platform config.
- Platform config bundle passes syntactic validation on the
  chosen stack.
- Defense brief answers all five questions; the
  8-nodes-lost question explicitly identifies which contract
  properties fail.
- Sign-off block enumerates every role that must sign an
  amendment.

## Stretch goals

- **Cross-cluster.** Extend the contract to a second cluster
  in another region, with a `research` cohort that borrows
  across both clusters. What changes about the reclaim
  ordering when the two clusters are federated?
- **Spot mixed in.** Add 64 preemptible-only H100s (spot
  capacity) into the cluster. Which teams get access, and how
  do you keep spot instability from breaking guarantees?
- **Wire it up.** Deploy the config to a real dev cluster
  (kind/minikube for k8s; a small Slurm sim like `slurm-in-a-
  box`). Submit synthetic workloads matching each team's
  profile; verify contention resolves per the contract.
- **Quarterly review.** Author a two-page slide deck of what
  the Q4 → Q1 amendment would look like given the Q4 roadmap's
  actual delivery (invent plausible numbers). Simulate the
  amendment RFC.
- **Federated on-call transition.** Draft the plan for
  transitioning from structure A to structure B as the org
  grows to 10 teams. What runbook chapters have to be complete
  before the transition can succeed?
