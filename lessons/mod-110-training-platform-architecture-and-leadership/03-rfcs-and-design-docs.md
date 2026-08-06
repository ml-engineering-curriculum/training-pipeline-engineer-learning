# RFCs and Design Docs for Training-Platform Changes

Every user-visible change to the training platform lands as a
written RFC. This is not bureaucracy; it is the mechanism that
prevents the anti-patterns of chapter 1 — invisible policy,
heroic on-call, shadow build — from silently accreting. A
platform whose changes are not written down is a platform whose
users cannot plan; a platform whose users cannot plan is one
that gets forked.

This chapter is the RFC template every training-platform change
uses. It is grounded in the RFC traditions of IETF (RFC 2119,
BCP 14), Rust (rust-lang/rfcs), Python (PEP 1), and Kubernetes
(KEPs) — cross-referenced in resources.md. The template is
adapted for training-platform surface changes specifically:
framework upgrades, hardware refreshes, storage migrations,
scheduler-policy changes, and capacity-contract changes.

## When to write an RFC

Not every change needs one. The threshold is *user-visible* and
*hard to reverse*. Concretely, write an RFC for:

- **Any change to the capacity contract** (chapter 2). New team,
  new priority class, quota re-allocation, cohort re-shape.
- **A framework major-version upgrade.** PyTorch 2.x → 2.y,
  FSDP1 → FSDP2, Megatron-LM major version, DeepSpeed major
  version, torchtitan cutover.
- **A hardware-generation refresh.** Adding a new SKU (Blackwell
  next to Hopper), retiring an older SKU, changing the fabric
  (IB → RoCEv2, chapter 3 of mod-105).
- **A storage-tier migration** (chapter 5 of mod-105). Moving
  from Lustre to Weka, adding a new S3 staging tier, changing
  the checkpoint hot-tier path.
- **A scheduler-policy change.** Enabling gang preemption where
  it was off, changing the fair-share decay window, introducing
  a new priority class, changing the admission-controller
  invariants.
- **A cross-team interface change** (chapter 4). Modifying the
  checkpoint format the serving team consumes, changing the
  provenance manifest the security team ingests, changing the
  registry API the MLOps team calls.
- **A build-vs-buy inflection** (chapter 5). Moving from in-house
  Megatron to NeMo, from self-hosted to Databricks/Mosaic-hosted,
  or vice versa.

Do *not* write an RFC for:

- **Bug fixes that preserve the contract.** File an issue; ship
  the patch.
- **Internal refactors invisible to consumers.** Container-image
  rebuilds, dashboard tweaks, log-format changes if no consumer
  parses them.
- **One-off operational actions.** Draining a node, moving a
  job. Those go in the runbook and the incident log.

The heuristic: if a consuming team would want to know before it
lands, or could be broken by it, write an RFC.

## The nine-section template

Every training-platform RFC has the same nine sections in the
same order. This is deliberate: reviewers should know where to
look without re-orienting. The order below is also the *authoring*
order — do not skip forward.

### 1. Metadata (5 lines)

- **RFC number.** Assigned from a running counter. Do not reuse
  numbers even if the RFC is withdrawn.
- **Title.** One line, imperative. "Migrate primary checkpoint
  format from `torch.save` to Distributed Checkpoint (DCP)".
- **Author, reviewers, sponsors.** The author drives the RFC.
  Reviewers are named individuals (not "the platform team")
  whose sign-off gates merge. Sponsor is the platform lead or
  their delegate.
- **Status.** Draft → Discussion → Accepted → Implemented →
  Superseded / Withdrawn. Set explicitly; date every
  transition.
- **Target dates.** Discussion end, accept-or-reject decision,
  implementation window, deprecation of the old behaviour
  (chapter 7 governs the window length).

### 2. Summary (1 paragraph)

Three-sentence description of the change and its effect on
users. This is what shows up in Slack when the RFC is announced.
If the summary requires jargon a first-year member of a
consuming team could not read, rewrite it.

### 3. Motivation (2–4 paragraphs)

*Why* the change is being proposed. Cite the drivers:

- **Concrete incidents.** If chapter 6's incident review flagged
  a class of failure this RFC prevents, link to the postmortem.
- **Measured limitations.** Numbers from mod-107 or mod-108:
  "torch.save writes stall training for 42 seconds at 512 GPUs
  per Llama 3 herd of models §4.5; DCP async save reduces this
  to 3 s". Cite the measurement source.
- **Roadmap dependencies.** If a downstream RFC or feasibility
  study requires this change, name it.
- **Cost or capacity drivers.** Mod-109 dollar figures if
  applicable.

Do not motivate with "the industry is moving to X". The industry
is not the reviewer; your platform's users are. Motivate with
their pain, not with the vendor's roadmap.

### 4. Design (the bulk of the RFC)

The technical proposal. Sub-sections vary by change type but
always include:

- **Behaviour change.** What the platform does *now* vs. what
  it will do *after*. Side-by-side is the clearest format.
- **API / format changes.** If a config key, path, or format is
  changing, show the diff.
- **Compatibility guarantees.** Which of the old
  behaviours are preserved during the compatibility window, and
  which are deprecated immediately. Chapter 7 owns the
  compatibility-window arithmetic; cite it.
- **Rollback path.** How is the change reverted if it does not
  work? "Rollback" is not the same as "we can turn it off"; a
  storage migration cannot be un-done by flipping a flag. If the
  rollback is irreversible, say so explicitly and require a
  higher bar of testing.

For a framework upgrade: name the framework version, the
per-team upgrade path, the recipe changes required, the MFU
regression risk, and the compatibility of existing checkpoints.

For a hardware refresh: name the SKU, the new topology, the
recipe-side implications (Transformer Engine version, new
attention kernels, new NCCL algorithms), the interaction with
existing runs, the amortisation math (mod-109 chapter 3).

For a storage migration: name the source and destination
storage, the migration protocol (in-place vs. dual-write vs.
lift-and-shift), the read-side compatibility of existing
checkpoints, the recovery-time objective during migration, the
capacity math.

### 5. Users affected and blast radius

For each of chapter 1's four user segments (foundation
researchers, fine-tuning engineers, post-training engineers,
platform on-call), name:

- **Whether they are affected.** Yes / No / Only if they use
  feature X.
- **What they have to do.** Nothing / opt in to new behaviour /
  migrate config / rebuild an image / re-run a benchmark.
- **What their timeline is.** By when.
- **Who from that team has been consulted.** Named individual;
  their sign-off status.

If a segment is not affected, say so explicitly. Silence in this
section is the RFC failure mode; a team that was not named will
discover they were affected during the rollout.

### 6. Alternatives considered

At least two, ideally three, with a paragraph each on why they
were not chosen. If the RFC is a build-vs-buy inflection
(chapter 5), the four canonical options — in-house / integrated
open source / hosted at hyperscaler / hosted at specialist —
each get a paragraph, with the build-vs-buy chapter's decision
matrix filled in.

The reason for this section is that reviewers who prefer a
different option should be able to point at it in your document
and see it addressed, rather than have to argue for it de novo.

### 7. Risks and mitigations

Every risk that could delay the implementation window or force
a rollback. For each: risk, likelihood (H / M / L), impact
(H / M / L), mitigation, owner. This is the RFC-scale analogue
of the mod-109 chapter 7 risk register.

Standard risks that appear in most training-platform RFCs:

- **Regression in `μ_sus`.** The new framework or hardware
  produces lower sustained MFU than the current one. Mitigation:
  measured pilot before commit.
- **Compatibility break for a downstream team.** A consuming
  team's config or checkpoints stop working. Mitigation:
  compatibility window; sign-off from that team.
- **On-call unfamiliarity.** The new system has failure modes
  the on-call has not seen. Mitigation: runbook update;
  paired-oncall week during rollout.
- **Rollback window closes.** A one-way migration crosses a
  point of no return. Mitigation: named checkpoint at which
  rollback becomes impossible; explicit "point of no return"
  decision in the timeline.

### 8. Test plan

How the platform team will validate the change before rollout,
during rollout, and after rollout.

- **Pre-rollout.** Pilot at what scale, on what recipe, for how
  long. Success criteria stated as measurable thresholds
  (`μ_sus` ≥ 0.35, checkpoint round-trip < 5 s, etc.).
- **During rollout.** What monitoring is added (mod-108 chapter
  2 panels, chapter 6 signatures). What signals abort the
  rollout.
- **Post-rollout.** How long the change is in "watch mode"
  before it is declared stable and the old path is deprecated.

Tests without measurable thresholds are not tests; they are
hope.

### 9. Rollout plan and deprecation timeline

The concrete week-by-week schedule from RFC acceptance to old-
path removal. Every date is calendar-anchored; every date has
an owner. Typical shape:

- **Week 0.** RFC accepted; work begins.
- **Weeks 1–4.** Implementation and pilot at 1/8 scale.
- **Week 5.** Rollout to opt-in teams. Pilot's success criteria
  must be met.
- **Weeks 6–12.** Rollout to remaining teams; compatibility
  window active.
- **Week 13.** Cut-over decision meeting; old path either
  deprecated or extended.
- **Week 26 (chapter 7's default window).** Old path removed.

Deprecation is a *date*, not a *stance*. "We plan to deprecate
X" is not a deprecation; "X is removed on 2027-01-15" is.

## Voting, sign-off, and merge criteria

The RFC is not merged by the author. It is merged by the
sponsor after:

- All named reviewers have sign-off comments in the discussion
  thread (approve / block-with-comment / abstain).
- Any block-with-comment reviewers have been resolved.
- The users-affected section has explicit confirmation from a
  named individual on each affected team.
- Any risk with impact H has a written mitigation, and its
  owner has ack'd.

The sign-off table lives at the top of the RFC alongside the
metadata block. It should be a fully-named table, not "the
platform team agreed". Individual names on a written record are
what makes the RFC durable; role names decay when people
change roles.

A common failure mode: the author merges their own RFC because
"nobody objected". If nobody engaged, the RFC does not have
consent — it has silence. Distinguish the two by requiring
active approve/abstain, not absence of block.

## Where the RFC lives

- **Source.** A monorepo directory `rfcs/`, one Markdown file
  per RFC, numbered. Or a GitHub project with the same layout.
- **Discussion.** A single PR against the RFC repo. Reviewers
  comment on the PR; the discussion is preserved. Slack and
  meeting notes are ephemeral and are not the record.
- **Index.** A top-level `README.md` listing every RFC by
  number, title, status, and date. Filterable by status.
- **Archive.** Withdrawn and superseded RFCs stay in the repo,
  marked; they are the institutional memory of what was
  considered.

The archive is more valuable than the current-open set. Six
months into a Blackwell migration, the RFC that got withdrawn a
year ago because "the launch dates slipped" is the document that
tells the new author why the current plan is what it is.

## Worked example: the framework-upgrade RFC

Not fully filled in, but the *shape* of an RFC to upgrade FSDP1
to FSDP2:

```
# RFC-042: Migrate primary parallelism from FSDP1 to FSDP2

Author:         Training Platform (author name)
Reviewers:      Foundations lead, Fine-tuning lead, Performance
                lead, On-call lead
Sponsor:        Platform Lead
Status:         Draft
Target dates:   Discussion ends 2026-09-15; decide 2026-09-22;
                pilot 2026-10-06 to 2026-11-03; rollout complete
                2027-01-15; FSDP1 removed 2027-04-15.

## 1. Metadata
(as above)

## 2. Summary
Migrate the platform's default parallelism from PyTorch FSDP1
(deprecated in 2.6) to FSDP2. Preserves existing recipes with
an opt-in flag during the compatibility window; targets a 3–5%
`μ_sus` uplift and simpler mixed-precision semantics.

## 3. Motivation
- FSDP1 is deprecated as of PyTorch 2.6 (link).
- Pilot on Llama-3-8B recipe at 64 GPUs shows +4.1% `μ_sus`
  (measurement date, source dashboard).
- Foundations team's Q4 pretraining depends on the
  DTensor-based composability that FSDP2 exposes.
- Incident POST-2026-08 (link to postmortem) traced a
  double-materialisation OOM to an FSDP1-specific corner case.

## 4. Design
- Behaviour change: `platform.parallelism = "fsdp"` in team
  config selects FSDP2 after cutover; `"fsdp1"` remains
  available during the compatibility window.
- API change: `FSDPConfig` fields renamed …
- Compatibility: FSDP1 checkpoints remain loadable via a
  translator (chapter 2 of mod-106); no re-materialisation is
  required.
- Rollback: setting `platform.parallelism = "fsdp1"` returns to
  the old path; supported through the end of the compatibility
  window.

## 5. Users affected
- Foundations: yes, must opt in during pilot; must migrate by
  2027-01-15. Consulted: (name).
- Fine-tuning: yes, no-op if using the default; consult
  (name) confirmed.
- Post-training: no (consumes checkpoints, format unchanged).
- Platform on-call: yes, new failure modes; runbook update in
  §8. Consulted: (name).

## 6. Alternatives considered
- Stay on FSDP1 through PyTorch 2.7: rejected; upstream will
  remove FSDP1 in 2.8, giving us three months to migrate later
  vs. six months now.
- Rewrite on DeepSpeed ZeRO-3: rejected; adds a second
  parallelism stack in the platform and does not resolve the
  DTensor composability driver.
- Wait for torchtitan cutover (RFC-050): rejected; RFC-050's
  own dependency is FSDP2, so this must land first.

## 7. Risks and mitigations
- MFU regression on Foundations recipe. Likelihood M, impact
  H. Mitigation: pilot at 1/8 scale for two weeks; abort if
  `μ_sus` shortfall > 2%.
- On-call unfamiliarity. Likelihood H, impact M. Mitigation:
  runbook update; paired-oncall week during rollout.

## 8. Test plan
- Pre: pilot at 64 GPUs on Llama-3-8B; success = `μ_sus`
  within 2% of FSDP1 baseline, checkpoint round-trip
  unchanged, resumability across restart proven.
- During rollout: mod-108 chapter 6 signatures continuously
  monitored; specific alerts on doubled-memory pattern (§4)
  and rank-drift.
- Post: two-week watch mode per team; alarms silent before
  declaring stable.

## 9. Rollout plan
(week-by-week from RFC acceptance)
```

That is the shape. A real RFC on a real change lands at 5–10
pages. The template's brevity in each section is intentional —
long RFCs are not read.

## Common failure modes

- **RFC as fait accompli.** The RFC is announced after
  implementation is complete. Reviewers are asked to rubber-
  stamp. The point of the RFC is missed. Discussion must
  precede implementation on any user-visible change.
- **Metadata drift.** The author is a person who has left the
  team; the sponsor slot is empty; the target dates are three
  quarters stale. Update the RFC or archive it; a stale RFC in
  the "Discussion" bucket is a landmine.
- **Users-affected section is a hand-wave.** "All teams may be
  affected". Name specifically or the rollout catches you.
- **Alternatives that are not alternatives.** "Do nothing" and
  "the proposed thing" is not a comparison. If you cannot name
  two real alternatives, the design space is too narrow — go
  read someone else's RFC on a similar change.
- **Rollout dates that are not calendar-anchored.** "After the
  next release" is not a date. Pick a date and re-plan if it
  slips; a slipping RFC is still a document, but an undated one
  is not.
- **The RFC that is really a design doc.** Chapter 4 of a
  parallelism strategy is a design doc; the *decision to adopt*
  that strategy is an RFC. Distinguish. Design docs live in
  each team's tree; RFCs live in `rfcs/` and are governed by
  this chapter.

## Summary

- Every user-visible, hard-to-reverse platform change lands as
  an RFC. The threshold is "a consuming team would want to
  know before it lands".
- The nine-section template — metadata, summary, motivation,
  design, users affected, alternatives, risks, test plan,
  rollout — is the invariant. Every RFC has all nine, in that
  order.
- RFCs are merged by the sponsor, not the author. Sign-off
  requires named individual approve/abstain, not silence. The
  users-affected section requires named per-team consent.
- The RFC repository is the institutional memory. Withdrawn
  and superseded RFCs stay; they explain why the current
  design is what it is.
- Framework upgrades, hardware refreshes, storage migrations,
  scheduler-policy changes, capacity-contract changes,
  cross-team interface changes, and build-vs-buy inflections
  each land as an RFC in this template's shape.
- The RFC is the *decision* artifact; the design doc is the
  *how*; the migration plan (chapter 7) is the *when*. Do not
  conflate the three.
