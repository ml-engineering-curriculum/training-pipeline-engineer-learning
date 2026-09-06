# exercise-05: Feasibility Study for a Target Product

**Estimated effort:** 3 hours

## Objective

Turn chapter 7 into the module's capstone artifact: a real,
seven-section training-run feasibility study for a named product
target, defended against the two chapter-6 anchors (MPT-7B and
Llama 3), reviewable by a capacity team, and shipped with the
one-page front matter a stakeholder actually reads. Deliverable:
a `feasibility_studies/` directory with two full studies — one
mid-scale and one frontier-scale — plus the sign-off record, a
one-page front matter for each, and a written retrospective on
what surprised you during the exercise.

## Prerequisites

- Chapter 7 of this module — the seven sections, the one-page
  front matter, the "living contract" pattern, the kill
  criteria. You should be able to name the seven sections in
  order from memory before starting.
- Chapters 1–6 of this module and their exercises. This exercise
  composes everything.
- The `chinchilla_budget` tool from exercises 1–4 (all four
  subcommands: budget → cluster_cost → tier_study → mfu-uplift).
  The study is the *narrative* wrapper around the tool's output.
- Mod-108 chapters 2 and 4 as background — the operational
  instruments the "living study" section refers to.
- Mod-110's placeholder for the RFC / sign-off workflow —
  chapter 7 stops at the study schema; the workflow around it
  is mod-110.

## Problem statement

Chapter 7's argument is that the feasibility study is the *only*
artifact that survives from planning through post-mortem: the
recipe changes, the cluster changes, the roadmap changes, but
the study — versioned, dated, signed off, and re-priced weekly
— is the durable contract. The chapter also argues that a study
without a range, without kill criteria, without a defended
anchor ratio, or without a live update cycle is decorative
rather than operational.

Your task is to author two of those contracts end-to-end.
Because a mid-scale study and a frontier study exercise
different failure modes (mid-scale gets `μ_sus` right and
misprices side-terms; frontier gets `μ_sus` wrong and
underestimates goodput loss), you will author one of each and
compare what changed. You will also make a *live* pass on one
study — pretend a week of the run has happened, absorb an
adverse actual, and publish v1.1.

## Requirements

Ship one directory:

- `feasibility_studies/`
  - `midscale-34b-chat/` — the mid-scale study.
    - `study.md` — the seven-section study, v1.0.
    - `front-matter.md` — the one-page summary.
    - `inputs/` — the frozen input snapshot (JSON files that
      fed the `chinchilla_budget` runs).
      - `target.json` (exercise 1 input)
      - `cluster.json` (exercise 2 input)
      - `tiers.json` (exercise 3 input, applicable subset)
      - `roadmap.yaml` (exercise 4 input, if adopting any
        uplifts)
    - `outputs/` — the raw tool outputs the study cites
      (`budget_report.md`, `cost_report.md`, etc.).
    - `signoff.md` — the sign-off table with proposed
      reviewers and roles.
    - `weekly-reprice/v1.1.md` — the live-update pass.
  - `frontier-70b/` — the frontier study, same layout.
    - v1.1 is optional; v1.0 is required.
  - `RETROSPECTIVE.md` — a two-page retrospective comparing the
    two studies and what the exercise taught you.

### 1. The mid-scale study — `midscale-34b-chat/study.md`

Author the seven-section study for the 34 B chat product from
chapter 1 (Target B from exercise 1). Every section from chapter
7 is present in the same order:

1. **Product target and success criteria.** Name the product,
   the eval threshold, the accountable stakeholder. Do not
   skip; short does not mean absent.
2. **Recipe.** `N`, `D`, `D/N` ratio and its Chinchilla vs.
   inference-optimal justification, architecture, data mixture
   summary + manifest citation, sequence length, precision
   policy. Cite chapter 1's decision list explicitly for the
   `(N, D)` pick.
3. **Compute budget.** The `C = 6·N·D` arithmetic, the
   attention correction (compute it even if small), the
   `GPU_hours = C / (P · μ_sus · 3600)` arithmetic, the `η(G)`
   assumption, the HFU/MFU convention chosen. Cite the pilot or
   the public anchor that grounds `μ_sus`.
4. **Cluster shape and lease tier.** Cluster shape (SKU +
   count + fabric + node topology), tier (with mix if
   applicable), per-GPU-hour rate cited to a URL and date,
   goodput assumption for the tier, wall-clock estimate with
   `η(G)` shown, side-term budget itemised (no lumped "misc").
5. **Dollar range and schedule.** The chapter-7 table with
   best / central / worst cases and both anchor ratios
   (`C/C_MPT7B` and `C/C_Llama3_70B`) as sanity-check rows.
6. **Risk register and mitigations.** Every risk that moves
   the number > 10%; name the mitigation and the owner. At
   minimum: sustained-MFU shortfall, cluster availability /
   preemption, data-pipeline stall, checkpoint corruption /
   SDC, recipe drift, hardware refresh, product-target creep.
7. **Kill criteria and re-review triggers.** Mechanical
   thresholds, agreed in advance. At minimum the five chapter-7
   triggers.

Constraints:

- Total length: 2–5 pages of body (chapter 7's spec). Do not
  pad; do not skip.
- Every dollar figure is a range, not a point.
- Every rate cites a URL and a checked-on date; stale-rate
  regressions from exercise 2's tool are surfaced explicitly.
- Every `μ_sus` and `goodput` assumption cites a pilot
  measurement or a public anchor.
- The anchor-ratio rows are populated with real ratios; if
  either ratio is > 5× or < 0.2×, either revisit or write a
  paragraph explaining why the ratio is legitimate.
- Every failure mode from chapter 7's "failure modes" list is
  actively guarded against — walk the list and confirm.

### 2. The frontier study — `frontier-70b/study.md`

Same seven sections, for a Llama-3-70B-scale target: 70 B
parameters, 15 T tokens (or a defensible variant), on the
largest cluster you can plausibly imagine committing to (name
it: DGX Cloud multi-year commit, an on-prem SuperPOD, or a
hyperscaler reserved block).

Additional constraints specific to the frontier study:

- **`μ_sus` must reflect chapter 6's warning.** Small-scale
  MFU numbers are 3–5× too high at frontier scale. Reconcile
  against Grattafiori et al. (2024) §3.3 the way chapter 6
  walks through — cite the reconciliation, name the `μ_sus`
  you land on, publish both the naive (small-scale) and
  frontier-corrected GPU-hour figures side by side, and
  explain the delta.
- **Goodput and interruption budget dominate.** Cite Llama 3
  §3.3.2's interruption taxonomy explicitly; the risk register
  should mirror those categories with your own guardrails.
- **Lease tier is a strategic capital decision.** Chapter 4's
  "big planned frontier run" pattern applies; the tier
  discussion in § 4 should defend the split against on-prem
  amortisation, reserved multi-year, and capacity block.
- **The dollar figure is a very wide range and must be
  labelled so.** Frontier estimates are typically ±30–50% at
  study time; chapter 7's failure-mode "point estimate is
  wrong" is amplified at this scale.
- **The anchor ratio to Llama 3 70B should be within a factor
  of 2 of 1.** If it is not, walk it back or name the
  deliberate divergence.

### 3. The one-page front matter — `front-matter.md`

For each study, the chapter-7 front matter:

- Title, author, date, version.
- The four answers, one sentence each (what is being trained;
  what it will cost; when it will be ready; what could go
  wrong — top 3).
- The explicit go / no-go recommendation.
- Sign-off table with names / roles / dates.

Length: one page maximum. Chapter 7 is explicit: the front
matter is the artifact that gets forwarded and embedded in
slides. If it does not fit on a page, the study is not ready.

### 4. Sign-off record — `signoff.md`

The chapter-7 sign-off table with named roles (not names — this
is a course exercise, use role labels: `author`, `technical
reviewer`, `accountable stakeholder`, `capacity owner`).
Include, for each role:

- What they are being asked to sign off *on* (the specific
  paragraphs / tables / assumptions).
- What they can reject and what they cannot (chapter 7's
  guidance: reviewer challenges arithmetic; stakeholder owns
  success criteria; capacity owner owns feasibility of tier
  commit; author owns the whole document).
- The escalation path when a sign-off is refused.

### 5. Frozen inputs — `inputs/`

The JSON / YAML files that fed the `chinchilla_budget` tool
runs, committed alongside the study. Chapter 7's "frozen input
snapshot" is a *file*, not a prose paragraph — a reviewer
should be able to `python -m chinchilla_budget ... --target
inputs/target.json` and reproduce every number in the study.
If a change to any input file would change a study figure, the
study version must bump.

### 6. Tool outputs — `outputs/`

The raw markdown outputs from exercises 1–4's tool runs. The
study cites them; they live in the repo so reviewers can drill
into the arithmetic. If your `chinchilla_budget` tool does not
emit reproducible markdown, this is a bug to fix in the
earlier exercises before continuing.

### 7. The weekly re-price — `weekly-reprice/v1.1.md`

Chapter 7's "living study" pattern. Pick the mid-scale study
(the frontier one is optional here) and pretend one week of the
run has happened. Publish a v1.1 that:

- Records the actuals-to-date for `μ_sus`, `goodput`,
  `restart_rate`, and GPU-hours consumed (make up plausible
  numbers, but label them clearly as *illustrative* — this is
  the one place in the exercise where fabricated numbers are
  permitted, and they must be marked as such).
- Recomputes the central-case dollar figure against actuals.
- Names the drift (positive or negative) and, if it exceeds
  ±10%, drafts the escalation memo per chapter 7.
- Updates the risk register: which risks fired, which retired,
  which strengthened.
- Records whether any kill criterion fired and, if so, the
  mechanical decision that follows.

The v1.1 file should be shorter than v1.0 — it is a delta, not a
rewrite. Chapter 7's failure mode "a study never updated during
the run" is exactly what this deliverable guards against.

### 8. Retrospective — `RETROSPECTIVE.md`

Two pages, at most. Cover:

- **What the two studies have in common.** The seven-section
  schema, the frozen inputs pattern, the range-not-point
  discipline, the kill criteria.
- **Where the two studies diverge most.** Almost certainly on
  the `μ_sus` reasoning, the goodput / interruption budget, the
  tier discussion, and the width of the dollar range. Name the
  divergences concretely with paragraph references.
- **Which chapter-7 failure modes bit hardest in your first
  draft.** Almost every author of a first study trips on the
  side-term budget, the anchor ratio, or the kill criteria.
  Which ones tripped you?
- **What you would change in the exercise-1–4 tooling** based on
  what the study wanted from it. Which inputs were awkward?
  Which outputs did the study have to rewrite? The exercise-1–4
  tools are the plumbing behind the study; a study author is
  the tool's *user*.
- **The one number from chapter 7** — the "dollars per MFU
  point at this scale, and at the next-gen scale" line — as it
  applies to each study. This is what the module asks you to
  be able to state at exit.

## Starter guidance

- **Author the front matter last, but read it first.** Chapter
  7's front matter is a summary of the body; you cannot write it
  until the body exists. But writing it forces you to reduce the
  body to its four load-bearing sentences, which is exactly the
  review test.
- **Freeze the inputs before writing the prose.** The seven-
  section body has to cite the frozen numbers; if you edit the
  inputs while writing, the prose will drift. Freeze, then
  write.
- **Do the anchor-ratio math before writing § 5.** Chapter 7 is
  explicit that the two anchor rows are a sanity check the
  reader will scrutinise. If the ratio surprises you, revisit
  the arithmetic before publishing.
- **Do not extend the seven-section schema.** Chapter 7's
  "reviewers should be able to open any feasibility study in
  the org and know where to look" is not a stylistic preference;
  it is a review-throughput requirement. Add appendices if you
  must, but the seven sections stay put.
- **Push the frontier study's `μ_sus` down.** Chapter 6 warns
  explicitly that small-scale MFU numbers are 3–5× too high at
  frontier scale. If your frontier study's `μ_sus` looks like
  your mid-scale study's `μ_sus`, you have not internalised
  chapter 6.
- **Use the exercise-3 productive-rate discipline.** Every
  dollar in § 5 is a *productive* dollar (chapter 4). List-rate
  numbers are not acceptable in the final figure.
- **Name the accountable stakeholder in § 1.** Even as a role
  label. A study with no accountable stakeholder is a study no
  one signs off on.
- **Write the kill criteria before you write the risk
  register.** The kill criteria are the mechanical thresholds;
  the risk register is the narrative of what could push you
  across them. Building the mechanics first constrains the
  narrative in a healthy way.
- **Make v1.1 hurt.** Chapter 7's failure mode "study never
  updated during the run" only bites if the update is
  mechanical and forces an escalation. Set your illustrative
  actuals to trigger the ±10% escalation for the practice.

## Acceptance criteria

- Two full seven-section studies, each 2–5 pages of body,
  covering every chapter-7 section in order.
- Every dollar figure is a range with named drivers. No naked
  point estimates in § 5.
- Every rate, every anchor, every `μ_sus`, every `goodput`
  assumption has a citation to a URL, a paper section, or a
  pilot measurement. Anything unverifiable is flagged
  `<!-- needs-research: ... -->` per the module guardrails
  (not fabricated).
- Both anchor ratios (MPT-7B and Llama 3 70B) are present in §
  5 with a paragraph interpreting each. Ratios > 5× or < 0.2×
  are defended.
- The frontier study's `μ_sus` reflects chapter 6's frontier
  correction and shows both the naive and frontier-corrected
  numbers side by side.
- The risk register in each study has ≥ 5 items, each with a
  named mitigation and a named owner.
- The kill criteria section has ≥ 3 mechanical triggers per
  study, each with the action that follows.
- One-page front matter per study, fitting on a single page,
  containing all six front-matter elements from chapter 7.
- `signoff.md` names four roles, what each signs off on, and
  the escalation path.
- Frozen inputs are committed to `inputs/`; a reviewer can
  re-run the exercise-1–4 tools against them and reproduce
  every number in the studies.
- The weekly re-price (v1.1) for the mid-scale study exists,
  shows the drift arithmetic, and — where drift exceeds ±10%
  — includes the escalation memo. Illustrative actuals are
  clearly labelled as such.
- Retrospective is present and answers all five prompts.
- No fabricated numbers outside the explicitly-labelled v1.1
  illustrative actuals.

## Stretch goals

- **Add a third study for the MPT-7B-scale cost-optimised
  anchor** (chapter 6 anchor 1). Compare all three side by
  side; the exercise then explicitly walks the whole
  cost-optimised → mid-scale → frontier spectrum, and the
  retrospective can name what changes at each scale
  boundary.
- **Author a v2.0 of the mid-scale study three weeks in.**
  Illustrative actuals continue to accumulate; the roadmap
  ships an uplift or two; the cluster shape mutates in
  response to a chapter-4 tier decision (e.g., a capacity
  block cancellation). Chapter 7's "living contract" pattern
  gets exercised across a longer arc.
- **Wire the study into a CI check.** A repo-level test that
  fails when the study's frozen inputs are edited without a
  version bump, when a citation URL 404s, or when the anchor
  ratio drifts outside the guardrail range. Chapter 7's
  "living study" pattern needs mechanical enforcement to hold.
- **Draft the mod-110 RFC template that would commission a
  study of the same shape.** Chapter 7 stops at "the workflow
  around this is a mod-110 topic"; drafting the workflow
  proposes the connective tissue. This is the beginning of
  mod-110's org-contract chapter.
- **Publish a side-by-side comparison of your mid-scale study
  and MosaicML's MPT-7B blog post** (chapter 6 anchor 1). The
  MosaicML post is essentially a public-facing feasibility
  retrospective; your mid-scale study is the same artifact
  ex-ante. What did MosaicML include that you did not? What
  did you include that they did not? Which model reads better
  to a stakeholder?
- **Prototype a "study diff" tool** that takes v1.0 and v1.1
  (or v2.0) of a study and emits the delta as a markdown
  report — dollar drift, risk-register churn, kill-criterion
  status, sign-off deltas. Chapter 7's weekly re-price
  becomes a one-command operation instead of a manual pass.
- **Reconcile against Llama 3 §3.3 the way chapter 6 asks.**
  Take the paper's reported 39.3 M H100-hours for the 70 B
  model, back-solve for the implied `μ_sus`, and use that
  number as the input to your frontier study's § 3. Publish
  the reconciliation as an appendix to the frontier study.
