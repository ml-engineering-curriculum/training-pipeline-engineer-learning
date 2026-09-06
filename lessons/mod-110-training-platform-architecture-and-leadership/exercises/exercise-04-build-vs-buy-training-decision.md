# exercise-04: Build-vs-Buy Training Decision

**Estimated effort:** 3 hours

## Objective

Author a level-35 **build-vs-buy memo** for a training
platform. Rate the four canonical options (Megatron in-house
/ NeMo or torchtitan integration / Databricks-Mosaic hosted /
Together-hosted) against a shared seven-axis decision matrix,
produce a two-year TCO decomposition, and land with a
defensible recommendation paragraph that names its re-review
trigger. Deliverable: one memo (`build-vs-buy-memo-v1.md`),
one spreadsheet-shaped artifact (`tco.csv` or `tco.md` with
the arithmetic laid out), and one short appendix showing the
alternative that came closest.

## Prerequisites

- Chapter 5 (build-vs-buy at level-35 altitude) of this
  module.
- Mod-109 chapter 3 (cluster amortisation) and chapter 4
  (per-productive-GPU-hour lens) for the TCO decomposition.
  These are the two chapters chapter 5 cites; do not
  re-derive the arithmetic here.
- Chapter 2 of this module (the capacity contract you are
  costing against) or the exercise-01 output.
- Skim, before starting: the Megatron-LM README
  (https://github.com/NVIDIA/Megatron-LM), the NeMo Framework
  User Guide overview
  (https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html),
  the Databricks Mosaic AI Training / MPT-7B post
  (https://www.databricks.com/blog/mpt-7b), and the Together
  AI training documentation
  (https://docs.together.ai/docs/fine-tuning-overview). You
  are picking between real, currently-shipping platforms; skim
  each vendor's own words before rating them.

## Problem statement

You are the training-platform lead at a company with the
following shape. This is the same platform whose capacity
contract you authored in exercise-01; if you did not do
exercise-01, use the shape below as a fresh premise.

- **Team.** 4 platform ICs (you + 3), plus one platform-lead
  hire pending (5 ICs steady-state in 6 months).
- **Consumers.** 5 downstream teams (foundations,
  fine-tuning, post-training, research, platform-dev).
- **Current fleet.** 384 H100 SXM5 GPUs across 48 nodes on
  NDR InfiniBand, purchased under a 3-year on-prem lease.
  ~18 months of amortisation remaining.
- **2026-Q4 to 2028-Q2 roadmap.** One 300 B pretraining
  (~192 GPUs × 6 weeks), sustained fine-tuning and RLHF
  workload, research-team long-context experiments.
- **Board question.** Coming into 2027 planning, leadership
  has asked, in a written note, why the platform is not on
  Databricks or Together and what it would take to switch.
  The memo you author is the answer.

You are writing the memo for a leadership readout six weeks
out. The audience is the CTO, the VP of engineering, and the
CFO. Every claim needs a citation; every dollar figure needs
its arithmetic exposed.

## Requirements

Ship one directory `mod-110-ex04/` with:

- `build-vs-buy-memo-v1.md` — the memo itself (chapter 5's
  4–6 page skeleton).
- `tco.csv` or `tco.md` — the two-year TCO breakdown per
  option in table form.
- `appendix-closest-alternative.md` — 1 page on the
  alternative that came closest and the threshold at which it
  would have won.
- `re-review-triggers.md` — the standalone list of triggers,
  short (1 page), linked from §7 of the memo.

### 1. The memo (`build-vs-buy-memo-v1.md`)

Length target: 4–6 pages when rendered. All nine sections
from chapter 5's worked-example skeleton, in order:

1. **Executive summary.** One paragraph, ~150 words. Names
   the recommendation, the two binding axes, and the
   re-review trigger.
2. **The four options.** One paragraph per option (A / B / C
   / D) as chapter 5 defines them. Do not invent new options
   — the chapter's four are the universe.
3. **Decision matrix.** The full seven-axis table (TCO,
   time-to-first-run, headcount required, capability ceiling,
   lock-in and portability, roadmap control, auditability /
   provenance). Every cell filled; every rating either a
   number, a bounded range, or a categorical (High / Medium /
   Low) with a one-sentence justification underneath the
   table.
4. **Two-year TCO comparison.** References `tco.csv` /
   `tco.md`. In the memo itself, reproduce the four totals
   with a short paragraph on the assumption set that holds
   across options (recipe, target `μ_sus`, wall-clock,
   headcount cost).
5. **Capability ceiling for the roadmap.** The 300 B
   pretraining is the binding constraint. State, per option,
   whether the platform demonstrably runs a 300 B-class
   pretraining today. Cite public runs (Llama 3, Nemotron,
   Mosaic's MPT lineage, Together's public training runs) or
   mark `<!-- needs-research: ... -->` if you cannot
   confirm.
6. **Lock-in analysis.** Score each option on the three
   sub-axes (data-plane, format, API) from chapter 5. Bound
   the exit cost (labor-months to re-integrate on a different
   option) for each hosted option.
7. **Recommendation and re-review triggers.** The
   single-paragraph recommendation in chapter 5's shape:
   option, one sentence per binding axis, re-review trigger
   (date / scale / market event), sign-off list.
8. **Alternative that came closest.** Reference the
   appendix; in the memo, one paragraph naming which option
   and at what threshold it would have won.
9. **Sign-off.** Named individuals or roles (`<Platform
   Lead>`, `<VP Engineering>`, `<CFO>`) with target dates.

Cross-references to chapter 5, chapter 2 (the capacity
contract the memo is costing against), and mod-109 chapters
3–4 (the TCO decomposition) are required. Every dollar figure
in the memo either points into `tco.csv`/`tco.md` or is
inline with the arithmetic exposed.

### 2. The TCO table (`tco.csv` or `tco.md`)

Two-year total per option, decomposed exactly as chapter 5
lists:

- **Capex.** Hardware amortisation across 3–4 years per
  mod-109 chapter 3, prorated to 24 months. Options C and D:
  near zero. Cite the amortisation schedule.
- **Opex.** Electricity + cooling + network transit (on-prem
  options) or per-GPU-hour rate × sustained-productive-hours
  (hosted options). Reconcile using mod-109 chapter 4's
  per-productive-GPU-hour lens; the naïve per-hour rate is
  not comparable across options.
- **Labor.** Loaded cost per platform IC × FTE count × 24
  months. Options A and B: label the FTE assumption
  explicitly. Options C and D: label the (smaller)
  integration-team FTE assumption explicitly.
- **Risk-adjusted overhead.** Expected cost of one Sev-1
  outage × probability, per option. Options A and B: your
  team owns the risk. Options C and D: read the vendor SLA
  and derive the SLA-credit ceiling; state that the credit is
  not a substitute for the real cost.

Column layout: option / capex / opex / labor / risk /
two-year total, one row per option. Hold three inputs
constant across rows and state them at the top of the file:
the recipe (`N`, `D`, precision), the target `μ_sus`, and
the wall-clock target. Any input you cannot verify: mark
`<!-- needs-research: ... -->` and continue.

The table must be reproducible from its inputs. Any per-hour
rate, headcount rate, or amortisation schedule that is used
without a citation is treated as a fabricated number and
fails acceptance.

### 3. The closest-alternative appendix
(`appendix-closest-alternative.md`)

One page. Structure:

- Which option came closest.
- Which axis is the binding one — the axis on which the
  recommendation and the runner-up differ.
- The specific threshold at which the runner-up would have
  won (e.g., "if headcount cost rises by 25% or Together's
  per-H100-hour rate falls below $X / hr").
- The instrument by which you would notice the threshold has
  crossed (a monitoring script, a quarterly re-price, a
  vendor pricing page you re-check).

This appendix is what the *next* author of this memo reads
first. Its job is to point at the sensitive axis so they do
not re-derive the entire matrix from scratch.

### 4. The re-review triggers (`re-review-triggers.md`)

Named, dated, testable triggers that re-open the memo. At
least four:

- **A calendar trigger** — no later than 12 months from
  memo date.
- **A pricing trigger** — a specific vendor price move
  (up or down) that flips a matrix cell.
- **A capability trigger** — a competitor's public run at a
  scale that closes chapter 5's capability-ceiling gap.
- **An org trigger** — a headcount or budget change that
  flips the labor line materially.

Each trigger is one paragraph: the metric, the threshold, the
owner who watches it, the review cadence at which it is
checked.

## Starter guidance

- **Fill the decision matrix before you write anything else.**
  Chapter 5's seven axes are the invariant; the memo is a
  narrative over the matrix. If the matrix is empty, the memo
  will drift into vendor advocacy.
- **Do the TCO arithmetic in a spreadsheet, not in prose.**
  Prose hides the arithmetic; a spreadsheet exposes it.
  Copy the totals into the memo, keep the sheet as the
  reproducible artifact.
- **Hold the three inputs constant.** Recipe, `μ_sus`,
  wall-clock. If you vary them across options, the numbers
  are not comparable and the memo fails.
- **The labor line is where in-house memos are dishonest.**
  Chapter 5 flags "the build case ignores labor". Count them
  loaded (fully-burdened cost per IC × FTE count × 24 months)
  and put the number in the row.
- **The SLA line is where hosted memos are dishonest.**
  Chapter 5 flags "treating the provider's SLA as risk-free".
  Read the actual SLA credit language; a 24-hour outage
  credit does not cover a lost 6-week frontier run.
- **The phased hybrid is often the right answer.** Chapter
  5's worked example (Option B baseline + Option D research
  spike) is the canonical shape. If you find yourself
  choosing between "all A" and "all D", re-check whether a
  phased hybrid dominates both.
- **Use `<!-- needs-research: ... -->` for any number you
  cannot verify.** Placeholder-marked comments are the
  correct treatment; invented numbers are not. This is the
  memo leadership signs; the citation discipline matters.
- **Do not paste vendor marketing verbatim.** "Frontier-scale
  training platform" is not a rating. Extract the operating
  envelope (largest publicly-reported run, elastic-recovery
  SLO, FP8 support) and rate against it.

## Acceptance criteria

- Memo has all nine sections in order; no section is TBD.
- Executive summary is ≤ 200 words, names the recommendation,
  the two binding axes, and the re-review trigger.
- Seven-axis decision matrix is filled for all four options;
  every cell has a rating and a one-sentence justification.
- Two-year TCO table has all four rows and all four columns
  (capex / opex / labor / risk-adjusted overhead), plus a
  total column. The three constant inputs (recipe, `μ_sus`,
  wall-clock) are stated at the top.
- Every per-hour rate, headcount cost, and amortisation
  schedule in the TCO table has a citation URL or a
  `<!-- needs-research: ... -->` marker. No unattributed
  numbers.
- Capability-ceiling section names a concrete binding
  workload (the 300 B pretraining, in the default premise)
  and states per-option whether it is demonstrably supported.
- Lock-in analysis scores each option on data-plane / format
  / API and bounds the exit cost for options C and D in
  labor-months.
- Recommendation paragraph follows chapter 5's shape
  (option, binding axes, re-review trigger, sign-off
  list) verbatim in structure.
- Closest-alternative appendix names the option, the axis,
  the threshold, and the instrument that watches the
  threshold.
- Re-review triggers file has at least four triggers, each
  with metric, threshold, owner, and cadence.
- Sign-off block enumerates the roles required to commit the
  decision (Platform Lead + VP Engineering + CFO at
  minimum).

## Stretch goals

- **Simulate the readout.** Author a 5–8 slide deck of the
  memo — the version leadership actually sees. Which axis
  goes on slide 1? Where in the deck does the TCO table go?
  Where does the re-review trigger appear?
- **Counter-memo.** Pick the runner-up from your appendix and
  author its memo. Same seven axes, same TCO decomposition,
  opposite recommendation. What does the sensitivity analysis
  look like when you write both from the inside?
- **Vendor SLA teardown.** For each hosted option (C, D),
  read the vendor's published SLA in detail. Score:
  availability, checkpoint durability, elastic recovery,
  security/audit, exit-data-portability. Publish as
  `sla-scorecard.md` alongside the memo.
- **Live re-price.** Set up the recurring calendar entry
  described in your re-review triggers. Write the one-page
  quarterly re-price document that the calendar entry
  produces. What is the smallest artifact that keeps the memo
  alive without re-writing it every quarter?
- **The migration plan for the "no" answer.** If your memo
  recommends A or B (staying in-house), author the migration
  plan skeleton (chapter 7 template) that a *future*
  recommendation to switch to C or D would require. The
  exit-cost bound is only credible when the exit plan is
  outlined.
- **Cross-team review.** Circulate the memo to a peer — a
  fine-tuning lead, an ml-platform lead, a performance
  engineer — and collect written comments in the shape of the
  exercise-02 discussion thread. A build-vs-buy memo that has
  not been challenged has not been reviewed.
