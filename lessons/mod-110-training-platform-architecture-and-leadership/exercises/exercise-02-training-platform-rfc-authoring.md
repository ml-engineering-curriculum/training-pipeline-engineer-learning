# exercise-02: Training-Platform RFC Authoring

**Estimated effort:** 3 hours

## Objective

Author a real, defensible RFC for a training-platform surface
change. Deliverable: one Markdown document
(`rfc-NNN-<slug>.md`) that fits the nine-section template from
chapter 3 and would stand up to review by a peer training-
platform lead. You also draft the accompanying migration plan
skeleton (chapter 7) that the RFC's §9 links to.

## Prerequisites

- Chapter 3 (RFC template) and chapter 7 (migration planning)
  of this module. Read both before starting; the RFC and
  migration plan are intertwined.
- Your exercise-01 capacity contract (or the chapter 2 starter
  template) as the platform state the RFC is proposed against.
- A working knowledge of at least one prior RFC-style
  document. Read one of the following as a shape reference:
  - **PEP 1** (https://peps.python.org/pep-0001/)
  - **rust-lang RFC 0002 — attributes** or any accepted RFC
    (https://github.com/rust-lang/rfcs/blob/master/text/)
  - **Kubernetes KEP-2000** or any recent KEP
    (https://github.com/kubernetes/enhancements/tree/master/keps)
  Note: read for *structure*, not content.

## Problem statement

Pick **one** of the three canonical RFC types from chapter 3
and author it end-to-end for your platform. All three are real
choices training platforms make in 2025–2026; pick the one
whose subject matter you know best.

### Option A — Framework major-version upgrade

**Migrate the platform's default parallelism from FSDP1 to
FSDP2** (PyTorch 2.6+). FSDP1 is deprecated upstream; the
migration is imminent for any PyTorch-based platform. Include:

- The MFU / memory / API-shape diffs from FSDP1 to FSDP2, with
  citations to PyTorch documentation and blog posts (linked in
  resources.md).
- A checkpoint-compatibility statement (see
  `torch.distributed.checkpoint` behaviour).
- A per-team migration schedule; foundations is the highest-
  blast-radius consumer.

### Option B — Hardware refresh

**Add a Blackwell (B200 / GB200) SKU next to the existing
Hopper (H100) SKU**, with a plan to retire H100 within 3
quarters (or extend the co-existence if the migration
demonstrates it is warranted). Include:

- The recipe-side implications: FP8 with Transformer Engine
  version bump, new attention kernels, new NCCL algorithms.
- The scheduler-side implications: new resource type
  (`nvidia.com/gpu-b200`), quota re-allocation, gang gang-
  affinity per SKU.
- The dollar arithmetic (cite mod-109 chapter 3 for the
  amortisation lens; you do not need real Blackwell prices —
  cite placeholder rates and mark `<!-- needs-research: ...
  -->`).

### Option C — Storage tier migration

**Migrate the primary checkpoint hot tier from an existing
storage tier to a new one** — for example, from POSIX-mount
NFS to a parallel filesystem (Weka, Lustre, DAOS) or from a
parallel filesystem to a GPU-Direct-Storage-optimised tier.
Include:

- The IOPS / throughput arithmetic showing the new tier meets
  the mod-105 chapter 5 chapter's checkpoint throughput
  budget.
- The migration protocol: in-place, dual-write, or
  lift-and-shift; how existing checkpoints in the old tier
  are handled during the compatibility window.
- The recovery-time objective during migration (RTO — the
  bounded window during which a checkpoint restore is slower
  than normal).

## Requirements

Ship one directory `mod-110-ex02/` with:

- `rfc-NNN-<slug>.md` — the full RFC (5–10 pages).
- `migration-plan-NNN-<slug>.md` — the migration plan skeleton
  (2–4 pages).
- `discussion-thread.md` — a simulated review thread with at
  least 3 comments from named reviewer roles and the author's
  responses (see below).

### 1. The RFC (`rfc-NNN-<slug>.md`)

Every one of chapter 3's nine sections is present in order:

1. **Metadata** — RFC number, title, author, reviewers,
   sponsor, status, target dates.
2. **Summary** — one paragraph, jargon-free.
3. **Motivation** — 2–4 paragraphs citing concrete drivers
   (measured limitations, incidents, roadmap dependencies).
   No hand-waving to "the industry".
4. **Design** — behaviour change side-by-side, API/format
   diffs shown as code blocks, compatibility guarantees,
   rollback path.
5. **Users affected** — one row per user segment (foundations,
   fine-tuning, post-training, on-call). Named consultation
   status per segment.
6. **Alternatives considered** — at least two real
   alternatives, each with a paragraph on why not.
7. **Risks and mitigations** — table with likelihood, impact,
   mitigation, owner. Include at least 4 risks.
8. **Test plan** — pre-rollout, during-rollout, post-rollout
   with measurable thresholds.
9. **Rollout plan and deprecation timeline** — week-by-week
   from acceptance to old-path removal. Every date
   calendar-anchored.

Length: 5–10 pages when rendered.

### 2. The migration plan (`migration-plan-NNN-<slug>.md`)

Fits chapter 7's four-stage template:

- **Stage 0 — Baseline lock.** What is measured before
  starting; how the baseline is snapshotted.
- **Stage 1 — Pilot at 1/8 scale.** What recipe, what cluster
  shape, what success criteria (measurable thresholds).
- **Stage 2 — Opt-in rollout.** How the new path is exposed;
  the two-team / two-week gate.
- **Stage 3 — Mandatory rollout.** Deprecation date; per-team
  migration schedule; end-of-support date.

Include the **compatibility-window** section (length, both-
paths-produce-compatible-outputs statement, enforcement date),
the **versioning** section (config-level and artifact-level),
the **converter** section (whether one is shipped, and if not
what the migration cost is), and the **per-stage rollback**
table with named triggers and runbook links.

The per-consumer action table (chapter 7) has one row per
consuming team with owner and due date.

### 3. The simulated discussion thread (`discussion-thread.md`)

The RFC is not merged by fiat; author a simulated review
thread showing the RFC being challenged and refined:

- **At least 3 comments** from named reviewer roles: one from
  a consuming team (e.g., foundations lead), one from the
  domain-adjacent platform (e.g., ml-platform-engineer or
  performance-engineer per chapter 4), and one from the
  on-call lead. Each comment either raises a concrete concern
  or requests a specific clarification.
- **The author's response to each**, either accepting
  (rewriting the relevant RFC section), rejecting with a
  reasoned argument, or promising a follow-up.
- **A final sign-off block** showing who approved (name and
  role) and any who abstained.

Comments should read like a real technical review: specific,
grounded in a section, not "looks good".

## Starter guidance

- **Start from the metadata block.** Filling in the metadata
  forces you to name the reviewers and target dates, which
  frames the whole document.
- **Write the summary and rollout plan first.** These are the
  two sections leadership reads. If you cannot summarise the
  change in three sentences, the design is not clear enough
  yet.
- **Every claim in the motivation section should have a
  citation.** A citation to a PyTorch blog post, a benchmark
  result, or an incident postmortem. Un-cited claims are
  where reviewers land.
- **Alternatives-considered is where most drafts are weakest.**
  Write two paragraphs on real alternatives, not one on "do
  nothing" and one on "the proposed thing". The alternatives
  section is what convinces skeptical reviewers.
- **The migration plan and the RFC's §9 must agree.** If the
  RFC's rollout says three quarters and the migration plan
  says two, one of them is wrong. Cross-check before
  submitting.
- **Use `<!-- needs-research: ... -->` for any fact you
  cannot verify.** Do not invent MFU numbers, dollar figures,
  or performance measurements. Placeholder marked comments are
  the correct treatment.

## Acceptance criteria

- RFC has all nine sections in order; no section is TBD.
- RFC metadata: numbered, dated, target dates are real
  calendar dates.
- Users-affected section: every one of the four user
  segments (foundations / fine-tuning / post-training /
  on-call) is either explicitly affected with a consultation
  status, or explicitly not affected with a one-line reason.
- Alternatives-considered section: at least two real
  alternatives, each with a paragraph on why not.
- Risks table: at least 4 risks with likelihood, impact,
  mitigation, owner.
- Migration plan has all four stages; each stage names its
  entry gate, exit gate, and (except stage 0) success
  criteria as measurable thresholds.
- Compatibility window length is set against the slowest
  consumer's release cadence and justified.
- Per-consumer action table: one row per consuming team, each
  with owner and due date.
- Discussion thread has at least 3 named-reviewer comments
  and 3 author responses; each response is either accept
  (with RFC-section reference), reject (with argument), or
  follow-up.
- Sign-off block is fully named (roles, not "TBD").

## Stretch goals

- **PR it.** Open a real PR against your internal or a
  personal RFC repo with the document; get someone to
  review; iterate. The real thing is a different exercise
  than the simulation.
- **Author the counter-RFC.** Pick the strongest alternative
  from your §6 and author *its* RFC. Compare. Which is more
  defensible?
- **Deprecation-notice draft.** Author the deprecation
  notice for the old path — the one that goes to the
  consuming teams' onboarding docs at the start of the
  compatibility window. This is the artifact users see; it
  should be readable without the RFC in hand.
- **Rollback runbook.** Turn the RFC's rollback plan into a
  step-by-step runbook (chapter 7's requirement). Every step
  is a copy-pasteable command or an unambiguous UI
  navigation.
- **Multi-RFC dependency chain.** If your chosen RFC depends
  on another RFC (e.g., FSDP2 depends on PyTorch 2.6
  adoption), sketch the dependency graph and the ordering
  constraints. What is the critical path?
