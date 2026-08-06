# mod-110 — Training-Platform Architecture and Cross-Team Leadership

**Estimated effort:** 15 hours

The previous nine modules taught you to build a training *run* —
convergent, observable, priced, and near-peak. This module
teaches you to build the *platform* those runs live on and to
lead the cross-team organisation that consumes it. The unit of
work shifts: from the training loop, to the multi-team,
multi-quarter product whose users are other engineers.

The through-line: a training platform is a *product*, and
mature platforms owe their users four contracts —
**capacity**, **hand-off**, **incident**, and **roadmap**. Each
contract is a written artifact with an SLA and named owners.
Chapter 1 sets the frame and names the four contracts. Chapter
2 designs the multi-tenant capacity contract for 5–15
consuming teams. Chapter 3 is the RFC template every roadmap
change lands under. Chapter 4 is the hand-off contract with
the five peer platform tracks. Chapter 5 is the build-vs-buy
memo — a level-35 decision that binds capital and teams.
Chapter 6 is the incident review that closes the loop after
any outage. Chapter 7 is the migration plan that lands each
RFC.

## Learning objectives

- Architect a multi-tenant training-cluster platform for 5–15
  downstream teams: quotas, gang preemption, priority classes,
  on-call, escalation.
- Author RFCs and design docs for training-platform surface
  changes (framework upgrade, hardware refresh, storage
  migration).
- Design hand-off contracts with peer platform tracks:
  `fine-tuning-engineer` (consumer of the platform),
  `ai-infra-ml-platform-engineer` (owns serving + registry),
  `ai-infra-mlops-engineer` (owns CI/CD + post-training
  pipeline), `ai-infra-performance-engineer` (authors kernels
  the platform consumes), `ai-infra-security-engineer` (owns
  training-data provenance and cluster boundary).
- Run a build-vs-buy decision at level-35 altitude (Megatron-LM
  in-house vs. NeMo/torchtitan integration vs. Databricks/Mosaic
  hosted vs. Together-hosted training).
- Author an incident review for a real, open-report-anchored
  training-run outage (OPT-175B loss-spike class, BLOOM
  hardware-failure class).
- Design a training-platform migration plan (FSDP1 → FSDP2,
  Megatron-LM → torchtitan) with rollback, versioning, and
  researcher-side compatibility windows.

## Chapters

1. [The Training-Platform Charter](01-the-training-platform-charter.md)
   — platform as product, the four user segments, the four
   contracts (capacity / hand-off / incident / roadmap), the
   level-35 altitude, and the three anti-patterns the module is
   written against.
2. [Multi-Tenant Training-Cluster Architecture](02-multi-tenant-training-cluster-architecture.md)
   — the one-page capacity contract, guarantee-vs-cap sizing,
   priority classes at platform altitude, gang-preemption
   knobs, on-call structures A / B, the Sev ladder, and the
   admission-time cluster invariants.
3. [RFCs and Design Docs for Training-Platform Changes](03-rfcs-and-design-docs.md)
   — the nine-section RFC template, when to write one, the
   sponsor-signs-not-author-signs rule, worked example for a
   framework upgrade, and the RFC-repository conventions.
4. [Hand-Off Contracts with Peer Platform Tracks](04-hand-off-contracts-with-peer-tracks.md)
   — the eight-section interface template, the five peer-track
   interfaces (fine-tuning, ml-platform, mlops, performance,
   security), and the quarterly interface-index review.
5. [Build-vs-Buy at Level-35 Altitude](05-build-vs-buy-training.md)
   — the four canonical options (A: Megatron in-house; B: NeMo
   or torchtitan integrated; C: Databricks/Mosaic hosted; D:
   Together / specialist hosted), the seven-axis decision
   matrix, the two-year TCO decomposition, and the
   recommendation paragraph with re-review triggers.
6. [The Training-Run Incident Review](06-training-run-incident-review.md)
   — blameless discipline, the eight-section postmortem
   template, the five-class root-cause taxonomy, worked
   examples anchored to OPT-175B loss spikes and BLOOM hardware
   failures, and the monthly incident-trends review.
7. [Training-Platform Migration Planning](07-training-platform-migration-planning.md)
   — the four migration stages (baseline / pilot / opt-in /
   mandatory), compatibility-window sizing, config- and
   artifact-level versioning, converter requirements, per-stage
   rollback runbooks, and worked examples for FSDP1→FSDP2 and
   Megatron→torchtitan.

## Exercises

- [exercise-01 — Multi-tenant training-cluster architecture](exercises/exercise-01-multi-tenant-training-cluster-architecture.md) (3 h)
- [exercise-02 — Training-platform RFC authoring](exercises/exercise-02-training-platform-rfc-authoring.md) (3 h)
- [exercise-03 — Hand-off contract with a peer track](exercises/exercise-03-hand-off-contract-with-peer-tracks.md) (3 h)
- [exercise-04 — Build-vs-buy training decision](exercises/exercise-04-build-vs-buy-training-decision.md) (3 h)
- [exercise-05 — Training-run incident review](exercises/exercise-05-training-run-incident-review.md) (3 h)

## Labs and quizzes

- `labs/` — an end-to-end platform-charter lab (author the full
  capacity contract, RFC, one peer-track interface, and one
  migration plan for a realistic org shape, defended in a mock
  review) lands here on the next autonomous cycle.
- `quizzes/` — one knowledge check lands here on the next
  autonomous cycle.

## Resources

- [resources.md](resources.md) — RFC-process references
  (IETF/RFC 2119, Python PEP 1, Rust RFCs, Kubernetes KEPs),
  scheduler docs (Kueue, Volcano, Slurm), framework and
  hosted-training references (NeMo, torchtitan, MosaicML /
  Databricks, Together AI), the OPT-175B and BLOOM logbooks,
  SRE / incident-review references, and a recommended reading
  order.

## How the module fits together

Chapter 1 is the *frame* — the vocabulary of platform-as-
product and the four contracts. Chapters 2, 4, 6, and 3+7
each own one contract:

- Chapter 2 owns the **capacity contract** — the multi-tenant
  quota, priority-class, and on-call surface.
- Chapter 3 (RFC) and chapter 7 (migration plan) together own
  the **roadmap contract** — every user-visible change lands as
  an RFC with a migration plan.
- Chapter 4 owns the **hand-off contract** — the five peer-
  platform interfaces.
- Chapter 6 owns the **incident contract** — the postmortem
  template and the monthly trends review.

Chapter 5 (build-vs-buy) is orthogonal to the four contracts —
it is the *strategic* memo that periodically re-tests whether
the platform's premise is still correct.

The five exercises walk the same arc: 1 authors the capacity
contract, 2 authors an RFC, 3 authors a peer-track interface, 4
authors a build-vs-buy memo, 5 authors an incident review.
Every exercise ships a written artifact — this module treats
technical writing as a core skill.

## What this module deliberately does not cover

- **Distributed-training semantics** (DDP, FSDP2, TP, PP, EP,
  3D) — owned by mod-101. This module *runs* on top of them;
  the platform's RFC template governs their major-version
  changes but does not re-derive them.
- **Framework internals** (Megatron, DeepSpeed, torchtitan,
  NeMo) — owned by mod-102. Chapter 5 is a build-vs-buy memo
  *about* which framework to run; mod-102 covers *how* they
  work.
- **Data pipeline design** — owned by mod-103. This module's
  interface 5 (security) consumes the provenance manifest
  mod-103 emits; the manifest's contents are mod-103.
- **Scheduler implementation** — owned by mod-104. Chapter 2 is
  the platform-altitude quota/priority/preemption contract;
  mod-104 chapter 8 is the scheduler plumbing that implements
  it.
- **Fabric and storage configuration** — owned by mod-105. This
  module's chapter 3 templates a storage-migration RFC; mod-105
  covers the storage tiering.
- **Checkpointing, elastic training, incident detection** —
  owned by mod-106. Chapter 6 of this module is the *review*
  process; mod-106 chapter 5 is the *runbook* the reviews
  reference.
- **MFU and kernel/comm efficiency** — owned by mod-107. This
  module's chapter 3 templates a kernel-upgrade RFC; mod-107
  covers the kernel-integration mechanics.
- **Dashboards, metadata store, reproducibility bundle** —
  owned by mod-108. Chapter 4 (interface 3, MLOps) consumes
  mod-108's chapter 7 metadata store; the schema itself is
  mod-108.
- **Cost arithmetic per run** — owned by mod-109. This module
  aggregates mod-109's per-run feasibility studies into
  quarterly capacity plans; the individual studies are mod-109.
- **People management, hiring, performance calibration** — out
  of scope. This module is about *technical* leadership.

## The two-sentence recap every stakeholder learns

At the exit of this module, every training-platform lead should
be able to state: *"Every user-visible change to my platform
lands as an RFC with a migration plan, gets signed off by
named individuals from every affected team, and produces a
postmortem within the SLA if it breaks. The four contracts —
capacity, hand-off, incident, roadmap — are written down,
signed quarterly, and re-read at every review."* That
sentence, and the ability to hold themselves and their team
to it, is what makes a training platform sustainable across
multiple quarters and multiple teams.
