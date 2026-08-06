# exercise-01: Chinchilla-Scaled Compute Budget

**Estimated effort:** 3 hours

## Objective

Turn chapter 1 into a working budgeter: build a small Python
tool that, given a product target `(N, quality target, serving
profile)`, produces a defensible `(N, D, C)` for both the
Chinchilla-optimal and inference-optimal recipes and prints a
comparison table. Deliverable: a `chinchilla_budget/` directory
with the tool, a short `budget_report.md` for three worked
product targets, and a written interpretation of the results.

## Prerequisites

- Chapter 1 of this module (scaling laws, the compute ledger,
  the Chinchilla vs. inference-optimal distinction). You
  should be able to state `C = 6·N·D` and the "20 tokens per
  parameter" rule from memory before starting.
- The Hoffmann et al. (2022) Chinchilla paper, arXiv 2203.15556
  — read at minimum §1, §2, and §3.
- The Kaplan et al. (2020) paper, arXiv 2001.08361 — skim
  enough to see the older power-law fit; this exercise uses it
  only for comparison.
- Python 3.10+.

## Problem statement

Every training run in this module starts with a `(N, D)` pick
that has to be defensible against Chinchilla, the inference-
lifetime lens, and the two anchor recipes from chapter 6. Your
task is to build the tool that produces that pick, apply it to
three product targets that bracket the design space, and write
up what the three answers tell you about when each recipe wins.

## Requirements

Ship one directory:

- `chinchilla_budget/` — Python package.
  - `chinchilla_budget/__init__.py`
  - `chinchilla_budget/laws.py` — the scaling-law functions.
  - `chinchilla_budget/budget.py` — the (N, D) → C
    calculator and comparison generator.
  - `chinchilla_budget/cli.py` — a small CLI that takes a
    product-target JSON and prints the report.
  - `tests/` — unit tests that exercise at least the corner
    cases in the failure-modes section of chapter 1.
- `budget_report.md` — the written report for three worked
  product targets.

### 1. The scaling-law functions (`laws.py`)

At minimum:

- `chinchilla_optimal_D(N, tokens_per_param=20)` — return the
  Chinchilla-optimal token count. Default ratio 20; allow
  override.
- `compute_flops(N, D, seq_len=None, num_layers=None,
   num_heads=None, head_dim=None)` — return `C ≈ 6·N·D`,
  plus the attention correction (`+ 12·L·H·d_head·D·S`) when
  the layer/head arguments are supplied. Return the two
  terms separately so the caller can see the correction size.
- `kaplan_optimal_N(C)`, `kaplan_optimal_D(C)` — for
  comparison; use the exponents from Kaplan et al. (2020) and
  cite them in the docstring.
- `chinchilla_frontier(C)` — for a fixed `C`, return the
  loss-minimising `(N, D)` under the Chinchilla fit
  (`N ∝ C^0.5`, `D ∝ C^0.5`). Cite the paper for the
  constants used.

Every function has a docstring that cites the paper section it
comes from.

### 2. The comparison generator (`budget.py`)

A `Budget` dataclass and a `compare(product_target)` function
that, given:

- `product_target`: a small dict/dataclass with fields
  `name`, `N`, `quality_notes`, `serving_tokens_per_day`,
  `serving_days`, `seq_len`, `notes`.

produces a table with rows for at minimum:

- **Chinchilla-optimal** (`D / N = 20`).
- **Llama-1-style inference-optimal** (`D / N ≈ 140`, cite
  Touvron et al. 2023).
- **Llama-3-style aggressive inference-optimal**
  (`D / N ≈ 1800` for `N ≈ 8B`; scale similarly). Cite
  Grattafiori et al. 2024.
- **User-supplied ratios** (accept a list of extra ratios in
  the product target).

Each row shows: `D`, `C`, `attention_correction_fraction`,
inference-lifetime FLOPs `2 · N · T`, and the ratio of
inference-lifetime to training FLOPs.

### 3. The CLI (`cli.py`)

```
$ python -m chinchilla_budget --input target.json --output report.md
```

Reads a product-target JSON, prints the comparison table as
markdown to `report.md`.

### 4. Three worked product targets in `budget_report.md`

Run the tool on three targets:

- **Target A — small research model.** `N = 1.3 · 10^9`,
  serving-tokens-per-day negligible (research artifact only).
- **Target B — mid-scale product model.** `N = 34 · 10^9`,
  serving-tokens-per-day `~50 · 10^6`, serving-days `~365`.
  This is the chapter-1 worked example.
- **Target C — heavy-serving small model.** `N = 8 · 10^9`,
  serving-tokens-per-day `~10 · 10^9`, serving-days
  `~1000` (roughly Llama 3 8B's expected serving lifetime for
  a widely-adopted open-weight model).

For each target, publish the full comparison table plus a one-
paragraph interpretation:

- Which recipe wins on training-FLOPs alone?
- Which recipe wins once inference-lifetime FLOPs are
  included?
- What is the sensitivity to the serving-lifetime estimate
  (halve it, double it — does the winner change)?
- Which recipe would you actually recommend and why?

The three targets should give three different recommendations
(Chinchilla-optimal for A, Chinchilla-optimal for B,
inference-optimal for C). If your arithmetic gives the same
recommendation for all three, something is off — walk it back.

## Starter guidance

- **Do the arithmetic once by hand before writing the code.**
  A tool that computes numbers you did not first derive
  yourself is a black box.
- **Cite the source in every function.** Chinchilla vs.
  Kaplan constants come from *specific* equations in *specific*
  papers. Cite them.
- **Use SI-scaled prints (`1.39e23`, not `139000000000000000000000`).**
  Long integers hide arithmetic errors; scientific notation
  makes them visible.
- **Compute the attention correction even when it is small.**
  A `1%` correction is not "small enough to ignore"; publish
  it so the reviewer can see it is small.
- **Do not hard-code accelerator peaks or dollar figures.** That
  is exercise 2. This exercise stops at `(N, D, C)`.
- **Include a "sensitivity to `N`" panel for at least one
  target.** Recompute `C` at `N ± 20%` and show how much the
  answer moves.
- **Write the units out.** `D` is in tokens; `C` is in FLOPs;
  `T` is in tokens. Silent unit confusion is one of the failure
  modes.

## Acceptance criteria

- The tool runs from the CLI and produces a markdown table.
- Every scaling-law constant used is cited to a paper section
  (Chinchilla, Kaplan, or a specific Llama paper).
- Every function has unit tests, including at least one test
  that exercises the attention correction being non-trivial at
  long context (`S = 32 K`, `N = 30 B`).
- The report covers all three target types (research, mid-scale
  product, heavy-serving small model) and gives a defensible
  recommendation for each.
- The report includes at least one sensitivity panel (change
  `N` or `serving_days` and re-derive the answer).
- The report says explicitly which convention was used for the
  `N` in `6·N·D` (non-embedding vs. total) and why. Chapter 1's
  failure-mode list is the reference for what to disclose.
- The recommendations for A / B / C are not identical. If they
  are, walk back and find the error.
- No fabricated numbers. If a serving-lifetime estimate is a
  guess, mark it as an assumption to be verified.

## Stretch goals

- **Add an MoE recipe path.** MoE requires using *active*
  parameters in `6·N·D`. Extend the calculator so it takes
  `num_experts`, `experts_per_token`, and `total_params` and
  computes the active `N` correctly. Include a comparison
  between a dense 34 B and an MoE equivalent (e.g., 8-of-32
  70 B) at the same target quality.
- **Reproduce the Chinchilla paper's approach-1 fit** from the
  raw data in Hoffmann et al. (2022) figure 3, and show that
  your `chinchilla_frontier(C)` matches. This is a deeper
  reading exercise.
- **Add Muennighoff et al. (2023) data-constrained corrections**
  (arXiv 2305.16264): if `D > D_available`, apply the
  repeat-data correction. Publish the "at what `D`
  availability does the Chinchilla recipe need to switch to
  data-constrained mode" answer for a mid-scale target.
- **Add an `L(D)` and `L(N)` loss estimate.** The Chinchilla
  paper's approach 2 gives a functional form; use it to
  predict the loss delta between the Chinchilla-optimal and
  inference-optimal recipes for the same `C`. Show whether
  the inference-optimal recipe pays a training-loss penalty
  large enough to matter for the eval target.
- **Hook the tool into a CI job** that runs on any PR touching
  the module's chapters and re-generates the worked-target
  tables. Catches drift in the chapter's numbers.
