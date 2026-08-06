# exercise-02: GPU-Hours and Dollars for a Target Cluster

**Estimated effort:** 3 hours

## Objective

Extend the exercise-01 tool with the chapter-2 and chapter-3
arithmetic: given `(N, D, C)` from exercise 1 plus a target
cluster shape (accelerator SKU, count, lease tier, MFU
assumption), produce a defensible GPU-hours number, a dollar
range, and a wall-clock schedule. Deliverable: an extended
`chinchilla_budget` tool with a `cluster_cost` subcommand, a
`cost_report.md` for the three exercise-01 product targets on
two different cluster shapes each, and an anchor-ratio sanity
check against MPT-7B and Llama 3.

## Prerequisites

- Exercise 1 completed (or its output pre-provided). This
  exercise consumes `(N, D, C)`.
- Chapters 2 and 3 of this module. You should be able to state
  `GPU_hours = C / (P · μ_sus · 3600)` from memory before
  starting.
- Access to (or bookmarks of) the current vendor pricing pages
  for at least AWS (`p5.48xlarge`), GCP (`a3-highgpu-8g`), and
  Azure (`ND_H100_v5`) — see chapter 3 for URLs.
- NVIDIA H100 SXM5 and A100 SXM4 datasheets — bookmark the
  peak-FLOPs pages.
- Python 3.10+.

## Problem statement

Chapter 2 says every additional MFU point, every scaling-loss
percentage, and every unit slip changes the GPU-hours number by
a lot. Chapter 3 says every stale price number, every side-term
omission, and every conflated lease tier changes the dollar
number by a lot. Your task is to build the machinery that gets
both right, apply it to your exercise-1 product targets on
concrete cluster shapes, and produce cost reports that a
capacity team would sign off on.

## Requirements

Ship these additions to `chinchilla_budget/`:

- `chinchilla_budget/hardware.py` — the accelerator peak-FLOPs
  table and the per-hour rate table.
- `chinchilla_budget/cluster.py` — the GPU-hours and
  wall-clock calculators, including the `η(G)` scaling-loss
  function and the HFU/MFU convention split.
- `chinchilla_budget/dollars.py` — the dollar calculators for
  each lease tier, with the side-term budget.
- Extend `chinchilla_budget/cli.py` with a `cluster_cost`
  subcommand.
- Extend `tests/` with correctness tests for the arithmetic.

Plus:

- `cost_report.md` — the written report.

### 1. The hardware table (`hardware.py`)

A small module with at minimum:

- A `PEAK_FLOPS` dict: keys are SKUs (`"h100_sxm5_bf16"`,
  `"h100_sxm5_fp8"`, `"a100_sxm4_bf16"`, etc.), values are the
  dense peak FLOPs per accelerator per second, each with a
  `source_url` and `checked_on` date in a docstring.
- A `RATES` dict: keys are `(sku, provider, tier)` tuples,
  values are the current per-GPU-hour rate with the same
  `source_url` / `checked_on` metadata.
- A `check_rates_freshness()` helper that flags rates older
  than 90 days as stale.

Do not hard-code more than a handful of rates. This module is a
maintainable table, not a scraper. Ship it with the current
rates for at least reserved and on-demand H100 on AWS, GCP, and
Azure — cite the URL and date in the docstring.

### 2. The cluster calculator (`cluster.py`)

At minimum:

- `gpu_hours(C, peak_flops, mu_sus)` — the chapter-2 formula.
- `wall_clock_hours(gpu_hours, G, eta_curve=None)` —
  `gpu_hours / (G · η(G))`. `eta_curve` is a callable
  returning `η(G)`; default to `lambda G: 1.0` with a
  warning.
- `default_eta_curve(fabric="ib_400g", parallelism="fsdp2")`
  — a simple parametric curve calibrated from public anchors
  (Llama 3 §3.2 for large-scale, small-scale extrapolation for
  ≤1 K GPUs). Cite the anchors in the docstring.
- `attention_correction_fraction(N, D, seq_len, num_layers,
  num_heads, head_dim)` — reuses the exercise-1 function; here
  the correction is folded into the effective `C` before
  computing GPU-hours.
- Support both MFU and HFU conventions via a `convention`
  parameter; document what each does and enforce that the
  caller picks one.

### 3. The dollar calculator (`dollars.py`)

At minimum:

- `dollars_from_gpu_hours(gpu_hours, rate)` — the dominant
  term.
- `side_terms(target, cluster, days)` — returns an itemised
  breakdown of storage, staging, egress, checkpoint retention,
  evaluation, dev budget, and staff cost. Every line item is
  a formula, not a hardcoded number, and every line has a
  named assumption (e.g., "assumes 4 TB training data at
  $0.021/GB-month for 6 months").
- `range_for(central, low_shortfall_pct=0.15,
  high_shortfall_pct=0.25)` — returns a `(low, central, high)`
  tuple where `high` reflects a plausible MFU shortfall and
  `low` a plausible over-performance. Chapter 7 uses this for
  the range publication.

### 4. Extend the CLI

```
$ python -m chinchilla_budget cluster_cost \
    --target target.json \
    --cluster cluster.json \
    --output cost_report.md
```

Where `cluster.json` specifies SKU, count, tier, and the MFU /
goodput / `η(G)` overrides.

### 5. The written report `cost_report.md`

For each of the exercise-1 targets (A, B, C from that
exercise), pick *two* cluster shapes each:

- **A small research cluster** — 64–128 GPUs, reserved cloud
  or on-prem.
- **A mid-scale cluster** — 512–1024 GPUs, reserved cloud.

Or, for target C, a large cluster (2K–4K GPUs). For each
`(target, cluster)` pair, publish:

- The GPU-hours arithmetic, showing every input.
- The dollar range with side-term breakdown.
- The wall-clock schedule with the `η(G)` correction visible.
- The anchor-ratio sanity check against MPT-7B and Llama 3
  (chapter 6): `C / C_MPT7B` and `C / C_Llama3_70B`, with a
  paragraph interpreting how the target sits on the spectrum.

Then a section that answers:

- Which of the two cluster shapes is cheaper per productive
  GPU-hour for this target?
- Which wall-clock is achievable, and does either violate the
  target's success-criteria schedule?
- What is the sensitivity to a 5-point MFU shortfall?

## Starter guidance

- **Fetch pricing right before writing the report.** Cloud
  rates decay 15–30% per year; a rate two months old is
  already slightly wrong. Cite the date you checked.
- **Never mix accelerator generations in the same rate table
  row.** H100-hours and A100-hours are different units; the
  tool should never let you multiply an H100 rate by an
  A100-hours count.
- **Verify the dense peak vs. sparsity-doubled peak.** Chapter
  2's failure mode. The correct H100 SXM5 BF16 dense is
  989 TFLOP/s; the sparsity-doubled advertisement is
  1979 TFLOP/s and is for a different workload.
- **Do the arithmetic once by hand for one target-cluster
  pair.** Cross-check your tool's output; a matching pair
  builds trust in the whole tool.
- **Publish a range, not a point.** The chapter-3 recommendation
  is `(low, central, high)`; the CLI should default to
  emitting the range.
- **Cite `μ_sus` per pair.** The MFU you assume varies with
  cluster shape (small vs. large `G`), tier (reserved vs.
  spot), and workload (long-context vs. short). Do not use
  one number across all pairs.
- **Include the on-prem amortisation formula as a variant.**
  For target C at 2K–4K GPUs, computing an on-prem cost is
  interesting; chapter 3's amortisation formula is the
  reference.

## Acceptance criteria

- The tool takes an exercise-1 target and a cluster
  specification and produces a `cost_report.md` with the
  GPU-hours arithmetic shown.
- Every rate in the hardware table is cited to a vendor URL
  and dated; stale rates (older than 90 days) are flagged.
- Every peak-FLOPs value is cited to a vendor datasheet and
  is the dense (not sparsity-doubled) number.
- Every dollar figure is presented as a range `(low, central,
  high)` with the drivers named.
- The side-term budget is itemised — no lumped "misc" line.
- The wall-clock estimate includes the `η(G)` correction; if
  `η(G)` is defaulted to 1.0, the report calls this out as an
  assumption to verify.
- The anchor-ratio sanity check to MPT-7B and Llama 3 70B is
  present for every `(target, cluster)` pair, with a
  paragraph of interpretation.
- The report explicitly states the MFU vs. HFU convention
  chosen and shows the same GPU-hours number both ways as a
  consistency check.
- The unit tests pass; at minimum, they catch a stale-rate
  regression and a sparsity-doubled-peak regression.
- No fabricated numbers. If a rate is unknown, the tool
  refuses to publish rather than guessing.

## Stretch goals

- **Add a `wall_clock_schedule(target, cluster, launch_date)`**
  that emits a Gantt-style schedule (pilot / main-run /
  evaluation / retro) and highlights the delivery date.
  Chapter 7's front matter can consume this directly.
- **Fit an `η(G)` curve from public data.** Take Llama 3 §3.2's
  numbers plus at least one small-scale published anchor (e.g.,
  Meta's FSDP scaling papers) and fit a parametric form for
  `η(G, fabric, parallelism)`. Publish the fit and its
  residuals.
- **Build a sensitivity dashboard.** Vary `μ_sus` from 0.20 to
  0.55, per-GPU-hour rate ±30%, `η(G)` from 0.70 to 1.00, and
  plot the resulting dollar range as a heatmap. Chapter 7's
  risk register consumes this for the "what moves the number
  most" panel.
- **Add a spot / preemptible tier row** that accepts an
  expected preemption rate and a mod-106-chapter-4 elastic-
  training goodput assumption; compute the productive-GPU-hour
  cost per chapter 4. Exercise 3 will build on this.
- **Add a cross-region egress calculator** for the case where
  data lives in region A and compute is in region B. Chapter 3
  lists this as an under-counted side term; make it a first-
  class line.
- **Publish an "on-prem breakeven" calculator**: for a given
  fleet size, utilisation, and lifetime, at what per-hour
  cloud rate does cloud beat on-prem? Chapter 3's amortisation
  formula is the reference; make it a `breakeven(cluster, ...)`
  function.
