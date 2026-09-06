# exercise-04: MFU Uplift to Dollars

**Estimated effort:** 2 hours

## Objective

Turn chapter 5 into a *business-facing* pricing tool for the MFU
and goodput work that mod-107 and mod-106 produce. Build a small
calculator that takes a proposed optimisation (an MFU delta, a
goodput delta, a checkpoint-cadence change, or an averted
incident) and returns its dollar value on the current run *and*
on the next-generation run — the two numbers that make kernel
and reliability work fundable. Deliverable: a
`mfu_uplift/` tool that extends exercise 2's calculator, an
`mfu_roadmap.md` for a realistic optimisation slate against a
named target, and a written interpretation.

## Prerequisites

- Chapter 5 of this module — the dollars-per-MFU-point identity,
  the goodput-to-dollars identity, Young's checkpoint-cadence
  formula, and the inference-lifetime dollar. You should be able
  to state `Δ$_run ≈ -$_run · (Δμ_sus / μ_sus)` from memory
  before starting.
- Exercise 2 completed (or its `cluster_cost` tool available).
  This exercise consumes `$_run` and `GPU_hours` from that tool.
- Chapter 6 as background — you will price uplifts on both a
  cost-optimised (MPT-7B-scale) and a frontier (Llama-3-scale)
  next-generation target.
- Mod-107 chapters 2, 4, 6, 7 and mod-106 chapters 3, 4, 6 as
  reference for the *anchor magnitudes* of typical uplifts. Do
  not fabricate uplift ranges; cite the source module chapter.

## Problem statement

Chapter 5's argument is that every technical optimisation has
two dollar numbers — its value on the current run and its value
on the next-generation run — and the second number is what makes
kernel and reliability investment fundable. Chapter 5 also
argues that the linear approximation `Δ$ ≈ -$ · Δμ/μ` breaks
above ~20% swings, that MFU and goodput uplifts are separate
levers that multiply, and that async DCP plus straggler / SDC
detection are catastrophic-event insurance whose per-incident
value often exceeds a whole per-optimisation budget.

Your task is to build the tool that computes both numbers
honestly (linear approximation *and* full re-arithmetic where the
swing is large), to price a realistic slate of proposals against
a named target, and to produce the MFU-roadmap table chapter 5
sketches — the artifact a platform lead takes into a funding
review.

## Requirements

Ship one directory:

- `mfu_uplift/`
  - `mfu_uplift/__init__.py`
  - `mfu_uplift/uplift.py` — the linear and full-arithmetic
    dollar calculators for MFU, goodput, cadence, and averted-
    incident uplifts.
  - `mfu_uplift/cadence.py` — Young's optimum-cadence solver
    plus a "sweep the cadence and price it" helper.
  - `mfu_uplift/roadmap.py` — the roadmap-table generator that
    consumes a list of `Uplift` proposals and emits the chapter-
    5 table.
  - `mfu_uplift/cli.py` — a `mfu-uplift` CLI subcommand added to
    the exercise-2 tool.
  - `tests/` — correctness tests, at minimum covering the
    small-Δ vs. large-Δ case, the goodput vs. MFU distinction,
    and Young's formula corners.
- `mfu_roadmap.md` — the written roadmap for the target run
  (exercise 1/2 Target B, the 34 B chat model, is the default)
  and its next-generation counterpart.
- `sensitivity.md` — a one-page appendix that stress-tests the
  roadmap's numbers.

### 1. The uplift calculator (`uplift.py`)

At minimum:

- `dollars_per_mfu_point(dollars_run, mu_sus)` — returns the
  linearised dollar value of one percentage point of sustained
  MFU (`$_run · (0.01 / μ_sus)`). Chapter 5's headline identity.
- `dollars_per_goodput_point(dollars_run, goodput)` — same for
  goodput. Documents that `goodput` and `μ_nom` compose into
  `μ_sus = μ_nom · goodput`, so the two levers do not double-
  count when priced together.
- `apply_mfu_delta(dollars_run, gpu_hours, mu_sus, delta_mu,
   convention="linear" | "exact")` — returns
  `(Δdollars, Δgpu_hours)` under either the linear
  approximation (chapter 5's shorthand) or the full arithmetic
  (`new = C / (P · (μ_sus + Δ) · 3600)`, re-derived). Raise a
  warning if `|Δmu / μ_sus| > 0.20` and `convention="linear"`.
- `apply_goodput_delta(...)` — same shape, but on the goodput
  factor.
- `averted_incident_value(cluster_size, hourly_rate,
   incident_hours, expected_events_per_run)` — chapter 5's
  straggler / SDC value formula:
  `expected_events · cluster_size · incident_hours · rate`.
- `next_gen_scaling(dollars_this_run, C_this_run, C_next_gen)`
  — returns `dollars_this_run · (C_next_gen / C_this_run)` as
  the naive scaling of a per-run dollar figure to the next-gen
  target, with a docstring caveat about `η(G)` and goodput not
  scaling linearly (chapter 6).

Every function has a docstring that cites the specific chapter
section (chapter 5 or, for the anchors, chapter 6) it derives
from.

### 2. Cadence solver (`cadence.py`)

At minimum:

- `optimal_cadence_seconds(t_write_s, restart_rate_per_s)` —
  `sqrt(2 · t_write / λ)`. Young 1974. Cite in docstring.
- `checkpoint_goodput_loss(t_write_s, t_cadence_s,
   restart_rate_per_s, restart_fixed_s)` — returns
  `(write_overhead_pct, restart_overhead_pct, total_pct)` per
  chapter 5's decomposition.
- `sweep_cadence(t_write_s, restart_rate_per_s, restart_fixed_s,
   cadences_s=None)` — sweep a range of cadences around the
  optimum and return a table of `(t_cadence, total_pct,
  dollars_lost_on_run)` given `dollars_run`. Use this to draw
  chapter 5's over-checkpoint / under-checkpoint asymmetry
  chart.

The chapter-5 worked example (`t_write = 30 s`, `λ = 3.3e-6/s`,
`t_restart_fixed = 600 s`) must be reproducible from your tool
in the tests, matching the chapter's numbers within rounding.

### 3. Roadmap generator (`roadmap.py`)

An `Uplift` dataclass with fields:

- `name` — e.g., `"switch to FlashAttention v3"`.
- `lever` — one of `mfu`, `goodput`, `cadence`, `averted_event`.
- `delta` — the size of the uplift (percentage points for
  `mfu`/`goodput`, a `CadenceChange` object for `cadence`, an
  `AvertedEventSpec` for `averted_event`).
- `ship_cost_eng_weeks` — engineering weeks to implement.
- `source_module_chapter` — the mod-107 / mod-106 chapter that
  documents the mechanism (mandatory; a missing citation is a
  linter failure in the roadmap generator).
- `anchor_uplift_range` — the typical uplift range from public
  data or the source module; a `(low, central, high)` tuple.

A `Roadmap` container that:

- takes a `RunSpec` (chapter 5's `$_run`, `μ_sus`, `goodput`,
  `C`, `gpu_hours`, `cluster_hourly_rate`) *and* a
  `NextGenSpec` for the next-generation target,
- iterates the `Uplift`s and emits the chapter 5 roadmap table:

| Optimisation | Lever | Expected Δ | Δ$ this run | Δ$ next gen | Ship cost | ROI (this run) | ROI (next gen) |

- computes cumulative Δ$ across the slate, correctly *not*
  double-counting when two uplifts touch the same lever (e.g.,
  two `μ_nom` uplifts must be combined multiplicatively, not
  added). Failing this test in unit tests is a bug.

### 4. CLI

```
$ python -m chinchilla_budget mfu-uplift \
    --run run.json \
    --next-gen next_gen.json \
    --roadmap roadmap.yaml \
    --output mfu_roadmap.md
```

Where `run.json` is the exercise-2 output for the current run
and `next_gen.json` is the exercise-2 output for the next-
generation target.

### 5. The written roadmap (`mfu_roadmap.md`)

Author the roadmap for the exercise-1/2 Target B run (the 34 B
chat model on 512 H100 SXM5) *and* its next-generation
counterpart (a plausible 34 B → 175 B upgrade, or a 34 B
retrained on 3× data — pick and justify).

Populate the slate with at least the following proposals, each
grounded in the named source-module chapter and the
`anchor_uplift_range` field:

- **Switch to FlashAttention v3** — mod-107 chapter 2. Uplift
  on `μ_nom`.
- **Enable `torch.compile`** — mod-107 chapter 6. Uplift on
  `μ_nom`.
- **Adopt FP8 with Transformer Engine on eligible ops** —
  mod-107 chapter 4. Uplift on `μ_nom`; note the accuracy-review
  cost as a `ship_cost` line.
- **Activation checkpointing + sequence packing** — mod-107
  chapter 5. Uplift on `μ_nom` (via larger effective batch);
  document the HFU/MFU-divergence caveat from chapter 5.
- **Comm-compute overlap** — mod-107 chapter 7. Uplift on
  `η(G)` — argue whether to model it as a goodput uplift or a
  correction to `μ_sus`; be explicit.
- **Async DCP** — mod-106 chapter 3. Cadence change; drives
  `t_write` toward zero. Rerun Young's formula with `t_write →
  0` and show the new optimum.
- **Elastic training on spot** — mod-106 chapter 4. Goodput
  uplift; the magnitude depends on the tier assumption from
  exercise 3.
- **Straggler / SDC detection** — mod-106 chapter 6. Averted-
  incident value at frontier scale; include the chapter-5
  16K-GPU / 6-hour worked example as a reference row.

For each row, publish the two dollar columns (`Δ$ this run`,
`Δ$ next gen`), the ship cost, and both ROIs. At the bottom,
publish a cumulative-uplift line (with the multiplicative
composition made explicit).

Then write a one-page interpretation:

- Which proposals top the ROI list on the current run?
- Which top it on the next-gen run? Do the two rankings agree?
- Which proposal has the largest next-gen leverage — the one
  where the two-column argument matters most?
- Which proposals are catastrophic-event insurance rather than
  steady-state uplifts, and how would you defend their line
  items to a reviewer who does not believe the underlying
  incident frequency?
- Which proposals are you *not* recommending, and why?

### 6. The sensitivity appendix (`sensitivity.md`)

Chapter 5's failure modes are largely sensitivity failures. Add
a one-page appendix that stress-tests the roadmap:

- Vary the current `μ_sus` from 0.30 to 0.50 in 5-point steps.
  Which proposals' ROI ranking flips?
- Vary the frontier-scale `μ_sus` assumption (chapter 6 says
  copying a small-scale MFU to a frontier budget is 3–5× low).
  Show the roadmap at `μ_sus_next_gen = 0.10` and `0.20`.
- Vary the assumed per-hour rate ±25%. Which proposals become
  or stop being fundable?
- For the averted-incident row, halve and double the assumed
  incident frequency. At what frequency does straggler
  detection stop paying for itself?

Every sensitivity number cites the input change and the
resulting recommendation change.

## Starter guidance

- **Do the two-column arithmetic by hand once.** Chapter 5's
  worked example (`3 pp` of MFU on the 34 B run) is the check
  that your tool gets the current-run number right; the frontier
  ratio is the check that your tool gets the next-gen number
  right.
- **Cite the source chapter for every uplift range.** An uplift
  range with no citation is a fabrication under this module's
  guardrails; use `<!-- needs-research: ... -->` if you cannot
  ground it.
- **Do not add uplifts that touch the same lever linearly.** Two
  `μ_nom` uplifts of `+4 pp` each are not `+8 pp`; they are
  approximately additive at small scales but multiplicative in
  the general form. Chapter 5's failure mode "applying the
  linear approximation to large `Δμ`" is exactly this trap.
- **Separate `μ_nom` uplifts from `goodput` uplifts.** Chapter 5
  is explicit that the two are different levers. A FlashAttention
  swap does not affect goodput; async DCP does not affect
  `μ_nom`. Compose them multiplicatively at the end.
- **Publish the ship-cost column in the same unit throughout
  (engineering weeks).** ROIs are only comparable when the
  denominator is the same.
- **Do not bury the caveats.** The tool's output should include a
  short "assumptions" block naming `μ_sus`, `goodput`, the
  per-hour rate, and the next-gen `C` — the four numbers whose
  drift makes the roadmap stale.
- **Test the cadence chart against the chapter-5 worked
  example.** If your `sweep_cadence` output for `t_write = 30 s`,
  `λ = 3.3e-6 /s` does not put the optimum near 71 minutes and
  the 12-hour under-checkpointing at ~23% run-cost, your solver
  is wrong.
- **Compute averted-incident value at the same cluster scale as
  the run being priced.** Copying the 16K-GPU worked example
  onto a 512-GPU run overstates the value by 30×.

## Acceptance criteria

- The tool computes both the linear approximation and the exact
  re-derived dollar impact and warns when the linear form is
  outside its region of validity (`|Δmu / μ_sus| > 0.20`).
- Every `Uplift` entry in the roadmap has a `source_module_chapter`
  citation; missing citations fail the roadmap-generator lint.
- The roadmap table has both the `Δ$ this run` and `Δ$ next
  gen` columns, and the cumulative-uplift line composes the
  same-lever uplifts multiplicatively, not additively.
- The cadence solver reproduces the chapter-5 worked example
  (optimum ≈ 71 minutes, ~0.8% total loss on the 34 B run,
  ~$2 300 in dollars) within rounding, and the over-/under-
  checkpointing asymmetry is documented in a chart.
- The averted-incident line is at the *run's* cluster scale, not
  at the chapter-5 frontier illustration's scale.
- The interpretation section names at least two proposals whose
  ROI ranking on this-run differs from the ranking on next-gen,
  and defends the divergence.
- The sensitivity appendix quantifies at least four assumption
  sweeps and names the recommendation changes each triggers.
- No fabricated uplift ranges; every number cites either a
  source-module chapter or the chapter-5 identity that produced
  it. Ranges from third-party sources are permitted only with a
  primary-source URL.

## Stretch goals

- **Add a "one-page business-case memo" generator.** From the
  roadmap YAML, emit a memo an engineering lead would forward
  to their VP: title, headline dollar (annual and next-gen),
  top three levers, ship cost, staffing ask, decision requested.
  Chapter 5's "the MFU roadmap is a business proposal" line is
  the design brief.
- **Model the interaction with the goodput SLO** (mod-106
  chapter 8). Given a team goodput target (e.g., `0.90`),
  compute the dollar cost of each point below target — and
  therefore the SLO's implicit dollar value. Chapter 5's
  goodput-to-dollars identity is the primitive; the exercise is
  translating the SLO into a fundable line.
- **Wire the roadmap into a CI job** that re-prices the slate
  weekly (per chapter 7's "living study" pattern) against the
  latest actuals for `μ_sus`, `goodput`, and the reserved rate.
  Publish a diff when the top-3 ranking changes.
- **Extend `averted_incident_value` to a distribution, not a
  point.** Model incident occurrences as a Poisson process with
  a cited rate `λ_incident` (mod-106 chapter 7 anchors), and
  publish the expected value plus the P50 / P90 dollar range.
  Chapter 5's straggler discussion currently uses a point
  estimate; the distribution is more honest.
- **Add the inference-lifetime dollar as a fourth column.** For
  each uplift that also lowers inference cost (FA v3, FP8),
  compute the additional serving-side savings using chapter
  5's inference-lifetime formula. Some uplifts pay off on both
  sides of the training / serving boundary; the tool should
  make that visible.
- **Publish an "MFU-roadmap-vs-anchor" panel.** For your
  recommended slate, plot the current-run and next-gen dollar
  totals against the MPT-7B and Llama 3 anchors from chapter
  6. Which proposals close the gap to a frontier-competitive
  cost profile the fastest?
