# The Feasibility Study

Chapters 1–6 built the machinery. This chapter is the *artifact*
that machinery produces: the **training-run feasibility study**,
the two-to-five-page document that gates the go / no-go decision
on a training run. It is the deliverable of this module, the
capstone exercise (exercise 5), and the primary artifact you
will hand to product and engineering leadership before any
capacity is committed.

The feasibility study answers exactly four questions:

- What is being trained? (`N`, `D`, architecture, recipe)
- What will it cost? (dollar range, GPU-hours, cluster shape)
- When will it be ready? (wall-clock schedule)
- What can go wrong? (risk register, mitigations, kill criteria)

Nothing else. It is not a design doc; it is not a research
proposal; it is not a status update. It is the document that
turns a strategic ask into a resourced commitment.

## Why a written feasibility study

Every large training run in the field has a story of a cost or
schedule surprise that would have been caught by a short written
study. The pattern is always the same: the run scope grew
gradually as the recipe took shape, the dollar figure was quoted
once verbally and never updated, and by the time the run was
half-way through, the actual cost was 2–3× the initial number
with no one accountable for the drift.

The feasibility study prevents that drift with three
mechanisms:

- **A frozen input snapshot.** `N`, `D`, per-hour rate, and
  assumed `μ_sus` are named at a specific date and quoted.
  Later changes are visible against the frozen baseline.
- **A defensible dollar range.** The output is a range, not a
  point. The drivers of the range are named; a reviewer can
  challenge each.
- **Explicit kill criteria.** Under what conditions is the run
  cancelled? The criteria are agreed *before* the run starts;
  during the run, the decision is mechanical instead of
  political.

## The seven sections

Every feasibility study has the same seven sections in the same
order. This is deliberate: reviewers should be able to open any
feasibility study in the org and know where to look.

### 1. Product target and success criteria (1 paragraph)

- What is the model for? Name the product / research programme.
- What does "done" look like? An evaluation threshold on a
  named suite ("above 0.70 on MMLU"), a downstream serving
  target ("supports the assistant product at Q4 volume"),
  or a research artifact goal ("SOTA on Long-Bench").
- Who is the accountable stakeholder? The person whose
  business the model belongs to.

This section is short but drives every subsequent choice. Do
not skip it.

### 2. Recipe: `N`, `D`, architecture (1 paragraph + table)

Directly from chapter 1's decision list:

- Parameter count `N`, with the sensitivity range (`N ± 20%`
  if not yet frozen).
- Data budget `D`, derived from the `D / N` ratio you picked.
- The `D / N` ratio and its justification: Chinchilla-optimal,
  Llama-1-style inference-optimal, or Llama-3-style aggressive
  inference-optimal. Cite the anchor.
- Architecture: dense Transformer, MoE (with active-parameter
  count), state-space, hybrid. If not dense Transformer, the
  chapter 1 accounting caveats apply.
- Data mixture: high-level composition (percentages by source),
  hash of the manifest if one exists. Full detail lives in
  mod-103; this section names the mixture and cites the
  manifest.
- Sequence length and context strategy. If long-context, name
  the attention correction (chapter 2).
- Precision policy: BF16, FP8 with Transformer Engine, other.
  Sets `P` in the denominator.

### 3. Compute budget: `C`, GPU-hours (1 paragraph + arithmetic)

The chapter 2 arithmetic, shown:

- `C = 6 · N · D` (plus attention correction, if applicable).
- `GPU_hours = C / (P · μ_sus · 3600)`.
- The assumed `μ_sus`, with the split into `μ_nom · goodput`
  and the source (measurement from a pilot, or citation from a
  public anchor).
- The scaling-efficiency `η(G)` at the target cluster shape,
  with citation to a small-scale measurement or a public
  anchor.
- If applicable, the HFU / MFU convention chosen.

Show the arithmetic — a reader must be able to re-derive every
number in the section from the inputs.

### 4. Cluster shape and lease tier (1 paragraph + table)

The chapters 3 and 4 arithmetic:

- Cluster shape: `G` GPUs of a specific accelerator SKU, in a
  specific topology (fabric, node count). Cite the vendor SKU
  name and URL.
- Lease tier: on-prem / reserved (with term) / capacity block
  / on-demand / spot. If mixed, name the split.
- Per-GPU-hour rate at each tier, cited to a pricing page and
  date.
- Goodput assumption for the tier, with citation to a pilot
  measurement or a public anchor.
- Wall-clock estimate at three cluster shapes (chapter 3), with
  `η(G)` explicit.
- Side-term budget: storage, staging, egress, checkpoints,
  evaluation, dev budget, staff (chapter 3's side-term list).

### 5. Dollar range and schedule (1 table)

The consolidated output. A single table:

| Scenario                | GPU-hours    | Cost           | Wall-clock    | Cluster shape       | Notes                          |
|-------------------------|--------------|----------------|---------------|---------------------|--------------------------------|
| Best case               | (chapter 2 low estimate) | $X    | Y days        | G GPUs              | Assumes μ_sus at target        |
| Central case            | (chapter 2 central)      | $Y    | Z days        | G GPUs              | Frozen assumptions             |
| Worst case              | (chapter 2 high)         | $Z    | W days        | G GPUs              | 20% MFU shortfall assumption   |
| Ratio to MPT-7B         | X×           | Y×             | —             | —                   |                                |
| Ratio to Llama 3 70B    | X×           | Y×             | —             | —                   |                                |

The last two rows are the chapter 6 sanity check. If the ratio
to either anchor is > 5× or < 0.2× and the reason is not
immediately obvious from the recipe, revisit.

### 6. Risk register and mitigations (1–2 paragraphs)

Every risk that could move the dollar or schedule number by
more than 10% goes in the register. Each risk names the
mitigation and the owner:

- **Sustained MFU shortfall.** If the pilot did not measure
  `μ_sus` at the target cluster shape, this is a real risk.
  Mitigation: run a 24-hour pilot at 1/8 the target cluster
  shape before committing. Owner: training-platform team.
- **Cluster availability / preemption.** If lease tier is
  spot or capacity blocks, the risk of interruption is
  material. Mitigation: elastic training (mod-106 chapter 4)
  proven for this workload. Owner: training-pipeline engineer.
- **Data pipeline stall.** The data loader is not paced with
  the compute (mod-103). Mitigation: pilot the loader at
  target throughput before committing. Owner: data-platform
  team.
- **Checkpoint corruption / silent data corruption.** Chapter 6
  of mod-106. Mitigation: SDC detectors deployed; checkpoint
  integrity checks on load. Owner: platform team.
- **Recipe change mid-run.** Product team pushes for a
  different `D` mid-way. Mitigation: freeze the recipe with
  written sign-off before the run starts; recipe changes
  require a new feasibility study. Owner: study author.
- **Hardware refresh mid-run.** A new generation lands that
  changes economics. Mitigation: name the amortisation window
  explicitly; run-in-flight is unaffected. Owner: platform team.
- **Product-team target creep.** Success criteria drift
  upwards. Mitigation: § 1 criteria are the frozen bar; new
  criteria require a new study. Owner: accountable
  stakeholder.

The register is not exhaustive; it captures the risks that
move the number by more than 10%.

### 7. Kill criteria and re-review triggers (1 paragraph)

Under what conditions is the run cancelled or the study re-
opened:

- **Pilot fails to hit `μ_sus` within 20% of the assumption.**
  Halt; re-price.
- **Preemption / restart rate exceeds the modelled `λ` by
  2×.** Halt; re-evaluate tier.
- **Actual cost through 25% of GPU-hours exceeds the central-
  case estimate by 20%.** Halt; re-price the remaining run.
- **Loss curve diverges beyond mod-108 chapter 6 signature 1
  tolerance and cannot be recovered in the mod-106 chapter 5
  runbook window.** Halt; recipe review.
- **The eval at intermediate checkpoints fails to trend toward
  the success criteria.** Trigger a research review; the run
  might continue but with revised success expectations.

The kill criteria are the mechanism that turns "we are
overspending" from a political conversation into a mechanical
one. Get them agreed *before* the run starts.

## The one-page front matter

The full study can run 3–5 pages. But every feasibility study
also has a one-page **front matter** — the summary that
stakeholders read and decide from. Its structure:

- **Title, author, date, version.** The date matters: pricing
  and MFU anchors move, and a stale study should be re-priced.
- **The four answers.** One sentence each:
  - What is being trained.
  - What it will cost (range).
  - When it will be ready (range).
  - What could go wrong (top 3 risks).
- **The go / no-go recommendation.** Explicit. "Recommend go
  at central-case budget of $X ± Y over T weeks" or "Recommend
  no-go pending pilot at 1/8 scale".
- **Sign-off table.** Author, technical reviewer, accountable
  stakeholder, capacity owner. Names and dates.

The front matter is the artifact that gets forwarded, embedded
in slides, and quoted. The body is the artifact that gets
challenged in review. Design both to be independently
defensible.

## Example: front matter for the 34 B chat model

*(All numbers illustrative — a real study cites pricing pages
and pilot measurements.)*

**Feasibility Study: 34 B Chat Model Pretraining**
**Author:** Training-Platform Team
**Date:** 2026-08-06
**Version:** v1.0

**What is being trained.** A 34 B-parameter dense decoder-only
Transformer chat model, pretrained on 680 B tokens (Chinchilla-
optimal) of the internal-web + code + curated-web-scrape mixture
`data/manifest-v42.json`. Sequence length 4096, BF16 with
Transformer Engine, FSDP2 parallelism.

**What will it cost.** Central-case `$293 K ± $60 K` (range
`$233 K–$353 K`) on 512 H100 SXM5 GPUs (64 × `p5.48xlarge`
reserved instances) for a wall-clock of 8–10 days. Side terms
(storage, staging, checkpoints, dev budget) add ~$45 K.

**When will it be ready.** 10 days from pilot completion (2 days
pilot + 8 days main run). Pilot start proposed 2026-08-20;
main-run completion target 2026-09-05.

**What could go wrong (top 3).**
1. Sustained MFU shortfall against the 0.40 assumption. Mitigation:
   pilot at 64 GPUs (1/8 scale) before commit.
2. Data-loader stall. Mitigation: run mod-103 chapter N loader
   under identical concurrency for 4 hours prior.
3. Recipe drift from product team. Mitigation: written sign-off
   before pilot; new studies for material changes.

**Recommendation.** Go, pending pilot at 1/8 scale meeting a 0.35
sustained MFU floor.

**Sign-off.**
- Author: (name), 2026-08-06.
- Technical reviewer: (name), 2026-08-07.
- Accountable stakeholder (Product): (name), 2026-08-07.
- Capacity owner (Platform): (name), 2026-08-07.

## Living the study during the run

The feasibility study is a *contract*, not a one-time artifact.
During the run:

- **Weekly re-price.** Update the central-case dollar figure
  against the actuals-to-date. If drift exceeds ±10%,
  escalate to the sign-off table.
- **Post the goodput dashboard.** Mod-108's dashboard (chapter
  2 of that module) is the operational instrument that shows
  whether the study's assumptions are holding.
- **Update the risk register when a risk fires.** A fired risk
  becomes an incident (mod-106 chapter 5); its dollar impact
  goes on the current-actuals-vs-central line.
- **Trigger the kill criteria when they fire.** Mechanically.
  A kill criterion that fires and is ignored is a study that
  did not do its job.

The training run's *end* is when the study is either signed off
as delivered (central-case ± range met) or amended with a
retrospective. Chapter 6 of mod-106 and chapter 5 of mod-108
are the source of the incident data the retrospective consumes.

## The interface with mod-110

The feasibility study is the artifact this module produces; the
*process* by which studies get commissioned, reviewed, and
signed off is a mod-110 topic (training-platform architecture
and cross-team leadership). The RFC template, the review
cadence, the authority to say no, and the archive of past
studies all live in mod-110's org contract.

The training-pipeline engineer authors the study. The
platform-team leader signs it. The product team commits to it.
The capacity team fulfills it. Each of those interfaces is a
mod-110 artifact.

## Failure modes

- **A study with no dollar range, only a point.** Every point-
  estimate is wrong. Ranges are the honest form.
- **A study with no `μ_sus` measurement or citation.**
  Speculation dressed up as a budget.
- **A study that omits the side-term budget.** The main-run
  hardware line is the biggest number, but the side terms move
  the total by 20–50%.
- **A study without a kill criterion.** No mechanism to stop a
  runaway run. The study is decorative, not operational.
- **A study never updated during the run.** The 10-day-old
  feasibility study on a 30-day run is a fiction after day
  one. Update weekly.
- **A study whose front matter and body disagree.** The
  stakeholder reads the front matter; the reviewer reads the
  body; if they disagree, one of them acts on the wrong
  information. Reconcile before publishing.
- **A study that misses the anchor sanity check.** A run that
  is 10× the ratio to MPT-7B or Llama 3 with no named reason
  is a run that has an error somewhere in the arithmetic.

## Summary

- The training-run feasibility study is the deliverable of
  this module. It answers four questions: what is trained,
  what it costs, when it is ready, what can go wrong.
- Seven sections in order: product target, recipe, compute
  budget, cluster shape, dollar range, risk register, kill
  criteria. A one-page front matter summarises the whole for
  stakeholders.
- Dollar figures are ranges, not points. Every input is cited
  to a pricing page, a public anchor, or a pilot measurement.
  The MPT-7B and Llama 3 ratios (chapter 6) are the sanity
  check.
- Kill criteria are mechanical. They are agreed before the
  run and fire during it; they turn "we're overspending"
  into a decision, not an argument.
- The study is a *living contract*. Re-price weekly against
  actuals; a study that drifts more than ±10% escalates.
- The process side of the study — commissioning, review,
  sign-off, archive — is a mod-110 topic. This chapter owns
  the schema; mod-110 owns the workflow.
- Exercise 5 is the capstone: author a feasibility study for a
  product target, defended against the two chapter-6 anchors,
  with full arithmetic and sign-off structure.
