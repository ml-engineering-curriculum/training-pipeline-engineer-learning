# exercise-05: Goodput SLO and Availability Budget

**Estimated effort:** 2 hours

## Objective

Author your platform's **goodput SLO** and the derived availability
budget, following the SRE / SLO framing chapter 8 lays out. The
deliverable is a two-part document: a one-page SLO specification a
platform-leadership audience can sign off on, and an appendix of the
measurement wiring and the error-budget policy that engineering will
run against. You will not stand up any new detectors in this
exercise — the mechanisms live in exercises 1–4; this exercise is
the *objective* they will be optimized against.

Where exercises 1–4 were code-heavy, this one is a small amount of
math plus a written specification. It should be short, precise, and
land at "we would ship this to leadership on Monday".

## Prerequisites

- Chapter 8 of this module in depth. Chapters 1, 2, and 7 for the
  numbers that anchor the budget.
- Google SRE Book chapter 4 ("Service Level Objectives"),
  https://sre.google/sre-book/service-level-objectives/. Read it
  once before you write the SLO — the vocabulary (SLI, SLO, SLA,
  error budget, error-budget policy) is used verbatim in the
  deliverable.
- PaLM paper (Chowdhery et al. 2023, https://arxiv.org/abs/2204.02311)
  §5 for the "goodput" definition, and Llama 3 (Grattafiori et al.
  2024, https://arxiv.org/abs/2407.21783) §3.3.2 for a published
  frontier reference point.
- Enough exposure to your own platform (or the platform in your
  RFC) to know the step time, world size, per-step tokens, and
  either a rough historical failure rate or a plausible worst-case
  reference (chapter 1 defaults to Llama 3's numbers).
- Exercises 1–4 done, or at least skimmed. This exercise assumes
  you know what your platform *can* measure; the SLO reports on
  what it *should*.

## Problem statement

Your platform team runs weekly. Everyone knows individually that the
big pretraining run has been rough this week — restarts, a suspected
straggler, a rollback for a loss spike — but there is no single
number the team looks at to decide whether the platform is on-target
or off. Different stakeholders see different dashboards: the training
scientists watch loss, the on-call watches uptime, the capacity team
watches GPU-hours. None of those is *goodput*.

Your job is to author the goodput SLO that everyone reports against.
The deliverable has to survive two audiences:

1. **Leadership** — one page, decided targets, an unambiguous "are we
   green or not" answer. No jargon that is not defined in-line.
2. **Engineering** — the measurement wiring, the error-budget policy,
   and the connection back to the mechanisms in exercises 1–4.

## Requirements

Deliver a directory (or a single tree of markdown files) containing:

- `slo.md` — the one-page SLO specification. Leadership audience.
- `appendix-a-measurement.md` — how the SLI is measured; which
  emitters live in the trainer, which live in the scheduler, which
  live in the aggregator. Engineering audience.
- `appendix-b-error-budget-policy.md` — the pre-committed policy
  triggered by budget consumption. Engineering + team-lead audience.
- `appendix-c-worked-numbers.md` — the arithmetic that produced
  your specific `G*`, budget, and slice split. Show your work.

### 1. `slo.md` — the one-page SLO

The one page. Every heading below is required; keep each terse.

- **SLI definition.** One sentence, in your platform's vocabulary,
  that names the numerator ("training tokens applied and persisted")
  and the denominator ("allocated H100-hours" — or your specific
  accelerator × capacity unit).
- **SLO target.** The specific `G*` value, with units, and the
  fraction of `G_max` it represents. Include the rolling window
  (default 28 days; defend any other choice).
- **Rolling window.** The window over which the SLO is evaluated.
  Explicit and defended.
- **Scope.** Which jobs, which cluster, which time range. If the
  SLO covers only one flagship pretraining run and not the finetune
  fleet, say so.
- **Reporting.** Where the metric is published (dashboard link, if
  you have one; a placeholder if not), and the cadence (weekly is
  the default).
- **Exclusions.** Explicit list of things the SLI *does not* count.
  Chapter 8 defines exclusions for rolled-back tokens and
  in-flight-not-persisted tokens; add any platform-specific ones
  (e.g., "cluster-wide power event > 30 min excluded pending
  post-mortem").
- **Owner.** Which team owns the metric and the response to a burn.

Keep the page to one printed page. Anything longer belongs in an
appendix.

### 2. Deriving `G_max`, `G_actual`, and `G*`

Compute all three from your platform (or from the platform in the
RFC). Chapter 8's worked example is the template; the important
thing is that the numbers are *your* numbers, not the illustrative
ones from the chapter.

- **`G_max` (nominal maximum).** For your step time, world size,
  per-step tokens, and effective per-hour steps at zero downtime:
  `G_max = per_step_tokens · world_size / step_time · 3600`. Pin
  the exact recipe and world size used.
- **`G_actual` (historical baseline).** From your telemetry, the
  P50 tokens-per-hour over the last N weeks. If you do not yet
  have telemetry, use a defensible estimate anchored either to
  your own initial-week data or to Llama 3's published >90%
  effective-training-time as a stretch reference. Note the source
  explicitly.
- **`G*` (SLO target).** Chosen from `G_actual` per chapter 8's
  guidance: start slightly below `G_actual` so the platform ships
  with a small positive budget, and specify the ratchet-up
  schedule (chapter 8 suggests quarterly).

Show all three numbers in `slo.md`; show the arithmetic in
`appendix-c-worked-numbers.md`.

### 3. The availability budget and its split

In `appendix-c-worked-numbers.md`:

- Compute the total downtime budget from
  `D_budget = (G_max − G*) / G_max · window_length`. Express in
  both cluster-hours and GPU-hours. Chapter 8's worked example
  is the template.
- Split the budget across the three chapter-8 slices: unplanned
  incidents, planned maintenance, recovery overhead per incident.
  Publish the specific percentages; defend each against your
  platform's current maturity. New platforms lean heavily to
  unplanned; mature platforms shift toward planned + recovery
  overhead.
- Cross-check by scaling Llama 3's §3.3.2 numbers (`419 unexpected
  interruptions / 54 days / 16384 GPUs`) to your world size and
  time window; compare against your unplanned slice. If the
  scaled-Llama-3 estimate exceeds your unplanned slice, the SLO is
  aspirational for your first quarter — document that and set the
  ratchet-up plan accordingly.

### 4. `appendix-a-measurement.md` — the measurement wiring

Name the emitters, one per bullet, and where they live:

- **Per-step tokens applied.** Emitted from the trainer. Zero for
  a NaN-filtered or rolled-back step. Reference chapter 8's
  "gradient was applied" clause.
- **Wall-clock hours of allocated capacity.** Emitted by the
  scheduler (mod-104 chapter 4 patterns). Not derived from
  running-container-count.
- **Checkpoint persistence events.** Emitted from the trainer at
  every successful DCP `future.result()`; the timestamp bounds the
  "at-risk" window.
- **Incident stream.** Emitted from the exercise-4 aggregator and
  the exercise-3 playbook post-mortems; each event carries duration
  and category.

For each emitter, list: the emitter name, the metric name / type
(counter / gauge / event), the storage backend, and where the
aggregator that computes the SLI reads it. This is what a new
platform engineer implements when the exercise ships to production.

### 5. `appendix-b-error-budget-policy.md` — the policy

The policy is *what happens* when the budget is consumed. Follow
chapter 8's three-band template exactly:

- **Healthy (< 50% consumed).** Explicit list of what proceeds
  normally. Feature work, framework upgrades, capacity moves.
- **Stressed (50%–90% consumed).** The freeze. Name the specific
  categories of change that stop, the categories that continue
  (bug fixes, incident response), and the escalation path for
  exceptions (who signs off, on what evidence).
- **Exhausted (> 90% consumed).** Reliability-only mode. Name what
  work is allowed, what work is deferred, and the exit criterion
  (typically "budget recovers to < X% and stays there for N days").

Two things to nail explicitly:

- **Who calls each transition.** A policy that requires debate to
  invoke has zero teeth. Chapter 8 recommends the SLO computation
  itself be the trigger. Name the on-call role that owns the
  transition.
- **What "sub-budget breach" means.** If unplanned incidents alone
  have consumed 90% of that slice while the total budget is only
  at 50%, does the stressed policy fire? Chapter 8 leaves this to
  the team; make the choice and defend it.

## Starter guidance

- **Write `slo.md` last.** Get the numbers and the wiring right in
  the appendices first; the one-pager is a distillation, not a
  brainstorm.
- **Do the arithmetic on paper before touching the doc.** A goodput
  SLO where the numerator and denominator units do not match is
  the most common failure mode. Chapter 8's worked example is the
  reference; keep units on every intermediate quantity.
- **Do not invent measurement infrastructure you do not have.** If
  your platform emits allocated-capacity today, wire the SLO to it;
  if it does not, appendix A says "TODO: emit from scheduler",
  names the owner, and dates the follow-up. A `<!-- needs -->`
  placeholder is more honest than a fictional pipeline.
- **Set `SLO_frac` from measurement, not aspiration.** Chapter 8
  spells this out: setting `G* = 0.95 · G_max` on day 1 without
  the mechanisms to hit it means the policy trigger fires every
  week and is ignored by month 2. Start at `P50` of your last
  8 weeks (or the equivalent conservative anchor if you have no
  history yet) and ratchet up quarterly.
- **The exclusions list is where past incidents live.** If your
  platform had a data-center power incident last year that took
  the whole cluster down for an afternoon, the exclusion list
  either explicitly excludes that class of event (with a defense)
  or explicitly includes it (with a defense of why the SLO should
  absorb the hit). Do not leave it implicit.
- **Cross-reference chapters 2, 3, 4, 6 in the appendix.** Every
  number the SLO consumes is a lever on some mechanism from the
  earlier chapters. Naming the mapping is how the SLO stays a
  design tool rather than a scoreboard nobody acts on.

## Acceptance criteria

- `slo.md` fits on one printed page, defines every term it uses,
  and names an owner. A leadership reader could sign off without a
  glossary.
- `G_max`, `G_actual`, and `G*` are computed from your platform's
  actual step time, world size, per-step tokens, and either
  measured or defensibly-estimated baseline. No hand-waved numbers.
- The availability budget is derived from `G_max` and `G*` via
  chapter 8's formula, split across the three slices, and
  cross-checked against Llama 3's scaled numbers.
- `appendix-a-measurement.md` names every emitter, its storage
  backend, and where the aggregator reads it. Placeholders are
  labelled and dated.
- `appendix-b-error-budget-policy.md` specifies the three bands,
  the trigger authority for each, and the sub-budget-breach
  behavior. No case-by-case wording.
- Exclusions are explicit; every exclusion is defended in one
  sentence.
- No invented numbers. If your platform has no historical
  telemetry, the report says so and the anchor is the Llama 3 or
  PaLM published number, cited.
- Every claim tied to a source (chapter, SRE Book chapter, Llama 3
  section, PaLM section) has an in-line citation.

## Stretch goals

- **Per-workload SLO decomposition.** If your platform runs a
  pretraining run and a fleet of finetunes on the same cluster,
  each has a different `G_max` / step time / recovery cost.
  Author a per-workload SLO plus a rolled-up cluster SLO; discuss
  whether the aggregation is a weighted average or the min.
- **Historical burn-down chart.** Take the last 4 weeks of
  incident data (from exercise 3's post-mortem harvester if you
  built it, or from your existing incident tracker) and compute
  what the SLO *would have said* over that window if this policy
  had been in place. Publish the chart. Argue whether the
  observed burn rate would have triggered the stressed or
  exhausted band, and whether the response would have been
  correct.
- **Dollar-cost overlay.** Multiply consumed budget by your
  cluster's GPU-hour cost (mod-109 chapter 3 patterns) to
  produce a dollar-denominated error-budget report. Discuss
  whether a dollar overlay changes the policy trigger — it
  should not, but the conversation is a useful stress test on
  how leadership will read the SLO.
- **SLO for the recovery path itself.** Author a nested SLO on
  chapter 4's recovery time: "P95 recovery from a fail-stop
  event is under T seconds". Wire the exercise-2 drill to this
  SLO; a regression in the drill's timeline is a burn on this
  sub-SLO before it becomes a burn on the goodput SLO. Discuss
  the interaction between the two.
