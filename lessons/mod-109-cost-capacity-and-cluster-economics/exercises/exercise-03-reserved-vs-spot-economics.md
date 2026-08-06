# exercise-03: Reserved vs. Spot Economics

**Estimated effort:** 3 hours

## Objective

Turn chapter 4 into a tier-decision study for a training
workload portfolio. Build a small analysis that takes a
portfolio of planned runs (a mix of large planned, ongoing
experimental, occasional frontier, and evaluation workloads)
plus a set of lease-tier options, and produces a defensible
mix — reserved baseline, capacity-block for large runs,
on-demand and spot for the rest — with the dollar impact of
each tier quantified. Deliverable: a
`portfolio_tier_study.md` with the arithmetic, and a spot-
tier goodput measurement plan.

## Prerequisites

- Chapter 4 of this module (the lease-tier ladder, dollars per
  productive GPU-hour, dedicated vs. shared, on-prem vs.
  cloud).
- Exercise 2 completed (or its `cluster_cost` tool available).
- Bookmarked pricing pages for at least AWS on-demand,
  reserved, Capacity Blocks, and Spot; the same for GCP or
  Azure. Chapter 4 lists the URLs.
- mod-106 chapter 4 (elastic training) as background — you
  will reason about goodput on spot.

## Problem statement

Chapter 4's central claim is that comparing list prices across
tiers is misleading; the honest unit is *dollars per productive
GPU-hour*, which folds goodput in. Your task is to run that
comparison for a representative training-workload portfolio,
show that the naive comparison and the goodput-corrected
comparison give different answers, and recommend a portfolio
mix that a capacity team would sign off on.

You will also plan (not run — this is a design exercise) a spot
goodput measurement, because chapter 4's failure mode
"assuming spot goodput without measuring it" is otherwise
inescapable.

## Requirements

Ship a single directory:

- `portfolio_tier_study/`
  - `portfolio.json` — the workload portfolio (structure
    below).
  - `tiers.json` — the lease tiers to consider (SKU, provider,
    tier, per-hour rate with citation and date, expected
    goodput with citation and confidence).
  - `analyse.py` — a script that consumes both, computes
    dollars per productive GPU-hour for each `(workload, tier)`
    pair, and picks the best tier for each workload.
  - `report.md` — the written study.
  - `spot_measurement_plan.md` — the design of a spot goodput
    measurement.

### 1. The portfolio (`portfolio.json`)

A JSON list of workloads that represent a plausible training
programme. At minimum, include one of each pattern from
chapter 4:

- **Big planned main run.** A single ~$300 K–$3 M run, months
  of lead time, low preemption tolerance. Use one of the
  targets from exercise 1/2 (e.g., the 34 B chat model).
- **Continuous experimental workload.** ~50 small-to-medium
  experiments per month; median 8–32 GPUs for 4–48 hours each.
- **Occasional frontier run.** One ~$5–20 M run per year;
  weeks of wall-clock; on the largest cluster you have
  planned; very low preemption tolerance.
- **Recovery / restart / evaluation workload.** Small,
  interruption-tolerant jobs; ~10 × 8-GPU × 2-hour per week.

For each workload, at minimum:

- `name`, `gpu_hours_per_month_expected`,
  `cluster_shape_required`, `preemption_tolerance` (one of
  `none`, `low`, `moderate`, `high`),
  `wall_clock_predictability_required` (bool),
  `notes`.

### 2. The tiers (`tiers.json`)

A JSON list of lease-tier options. Include at minimum:

- On-demand H100 SXM5 (hyperscaler of your choice; cite the
  URL and date).
- Reserved H100 SXM5 1-year all-upfront.
- Reserved H100 SXM5 3-year all-upfront.
- Capacity Block / DGX Cloud short-duration equivalent.
- Spot / preemptible H100 SXM5.
- On-prem H100 SXM5 (use chapter 3's amortisation formula;
  cite assumed capex, PUE, utilisation, lifetime).

For each tier, at minimum:

- `sku`, `provider`, `tier`, `per_hour_usd`,
  `per_hour_source_url`, `per_hour_checked_on`,
  `expected_goodput_default`, `goodput_source`,
  `commit_period` (nullable).

### 3. The analysis (`analyse.py`)

For every `(workload, tier)` pair, compute:

- **Feasibility** — does the tier support the workload's
  cluster shape and predictability? A big planned run on
  spot with `preemption_tolerance = none` is infeasible;
  emit a flag, not a dollar number.
- **Naive dollars per GPU-hour** — the vendor's list price.
- **Productive dollars per GPU-hour** —
  `per_hour_usd / expected_goodput`.
- **Total monthly cost** for the workload at this tier
  (feasibility-permitting).

Emit a table per workload sorted by productive cost. Recommend
the cheapest feasible tier and produce a portfolio-level
monthly total.

Then run a **sensitivity pass**:

- If the assumed spot goodput is off by ±20 percentage points,
  does the recommendation change?
- If the reserved rate drops 15% (typical annual price
  compression), does the recommendation change?
- If capex on on-prem drops 30%, does on-prem enter or leave
  the recommendation?

### 4. The written report (`report.md`)

Section by section:

- **Portfolio summary.** Table of workloads, expected monthly
  GPU-hours per workload, total GPU-hours.
- **Tier catalogue.** Table of tiers with list rate,
  goodput assumption, source, and productive rate.
- **Naive comparison.** For each workload, cheapest tier
  under the naive comparison (list rate only). Emit the
  table.
- **Productive comparison.** Same table, corrected for
  goodput. Highlight the rows where the ranking changed.
- **Recommended portfolio.** Prose summary of the
  recommended mix, with the monthly and annual dollar
  totals. Include the on-demand fallback assumption for
  the burst above the reserved baseline.
- **Sensitivity findings.** Which assumptions move the
  recommendation the most, and what would you measure to
  reduce that uncertainty.
- **Failure modes flagged.** Explicitly name the chapter 4
  failure modes that this portfolio is at risk of, and how
  the recommendation guards against each.

### 5. The spot measurement plan (`spot_measurement_plan.md`)

A short (1–2 page) document that specifies how you would
actually measure spot goodput on a representative workload
before committing to it:

- **The pilot workload.** A concrete, small (e.g., 32-GPU) job
  that mimics the target workload's step time and I/O pattern.
- **The measurement period.** At least 7 days of wall-clock
  to catch preemption-frequency variability by day-of-week.
- **The instrumentation.** Which metrics (preemption events,
  restart wall-clock, checkpoint reload time, productive
  step count) with which source (mod-108 dashboards, cloud
  provider APIs).
- **The success criteria.** What goodput number would make
  spot viable for the workload it is planned for; what number
  would rule it out.
- **The escalation path.** If pilot goodput falls below the
  threshold, what does the study revert to (reserved,
  capacity block, on-prem)?

## Starter guidance

- **Do not compare a spot list rate to a reserved list rate
  and claim a discount.** Chapter 4's central lesson. Every
  comparison in your report must be productive-rate.
- **Cite everything.** Every rate, every goodput assumption,
  every capex figure. Sources are what make the report
  reviewable.
- **Do not assume spot preemption is uniform across regions,
  AZs, or SKUs.** It is not. Cite the region/AZ your rate
  assumption applies to.
- **Include a `dev budget` line even if it is small.** The
  continuous-experimental workload is where most training
  organisations spend most of their money over time; do not
  under-count.
- **Do not recommend "put everything on spot".** Chapter 4's
  worked example shows that this is nearly always wrong for
  the big planned run.
- **Do include an on-prem option even if you conclude cloud
  wins.** The arithmetic being explicit is what makes the
  conclusion defensible.

## Acceptance criteria

- The tool reads `portfolio.json` and `tiers.json` and produces
  a per-workload comparison table.
- Every rate and goodput assumption in `tiers.json` is cited
  with a URL / date / paper section.
- The comparison table has both a naive and a productive
  column; at least one workload has a different top-ranked
  tier between the two.
- The recommended portfolio names the tier for each workload
  and gives a monthly and annual dollar total (as a range).
- The sensitivity analysis names at least three assumptions
  that move the recommendation and quantifies each.
- The report calls out at least three chapter-4 failure modes
  the portfolio is at risk of and names the guardrail against
  each.
- `spot_measurement_plan.md` names a pilot workload, a
  measurement period, instrumentation, success criteria, and
  an escalation path.
- No naked list-price comparison. Every dollar in the
  recommendation section is productive-rate.

## Stretch goals

- **Model queue delay on shared / on-prem.** For a
  multi-tenant baseline, add a `queue_delay_expected_hours`
  field per workload and translate it to dollar impact via
  chapter 5's per-hour cost. When does queue delay flip the
  tier ranking?
- **Add a preemption-rate simulator.** Given a spot workload
  and an expected preemption rate `λ_p`, simulate a month of
  wall-clock and produce a distribution of goodput. Chapter
  4's productive-rate calculation currently uses a point
  estimate; a distribution is more honest.
- **Add mod-106 chapter 4 elastic-training assumptions.**
  Model the case where the training loop can survive a node
  loss without restart. Show how much the spot productive
  cost improves.
- **Build a "capacity commit optimiser".** Given the year's
  workload forecast and a set of reserved-tier options,
  compute the optimal 1-year vs. 3-year commit split for the
  reserved baseline. What is the cost of over-committing?
  What is the cost of under-committing?
- **Extend the study to include a hyperscaler alternative for
  the frontier run** — e.g., DGX Cloud vs. reserved
  `p5.48xlarge` vs. on-prem SuperPOD. Present a build-vs-
  buy comparison with a 3-year horizon; this is the seed for
  mod-110's build-vs-buy chapter.
