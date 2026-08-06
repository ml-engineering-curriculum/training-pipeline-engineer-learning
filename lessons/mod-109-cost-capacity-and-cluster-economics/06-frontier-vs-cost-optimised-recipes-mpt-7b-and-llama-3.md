# Frontier vs. Cost-Optimised Recipes: MPT-7B and Llama 3

The previous chapters gave you the arithmetic. This chapter gives
you two anchor recipes that let you calibrate that arithmetic
against public data. Every dollar figure in a feasibility study
should be reviewable against at least one of these two
references — "this proposal is 1.3× the per-token cost that
MosaicML reported for MPT-7B" is a defensible claim; "this
proposal is $X because our model computed it" is not.

The two anchors:

- **MPT-7B (MosaicML, 2023)** — the canonical *cost-optimised*
  recipe. A 7 B decoder-only model trained on 1 T tokens, with
  the full recipe and cost profile published openly. Represents
  the scaling-down (cost-first) end of the spectrum.
- **Llama 3 (Meta AI, 2024)** — the canonical *frontier* recipe
  at the scale where the compute bill is a strategic capital
  decision. 8 B / 70 B / 405 B models on ~15 T tokens each, on
  a 16 K-GPU H100 cluster. Represents the frontier end.

They bracket the design space for a modern training run. Every
concrete product target sits somewhere between them, and the
feasibility study's job is to name where.

## Recipe 1: MPT-7B (the cost-optimised anchor)

MosaicML released MPT-7B in May 2023 as a demonstration that a
Llama-1-comparable 7 B model could be trained end-to-end on the
company's platform for a publicly-quoted dollar figure. The
public artifacts are:

- The MPT-7B foundation-model card and blog post from
  MosaicML — https://www.databricks.com/blog/mpt-7b (Databricks
  acquired MosaicML in July 2023; the blog is preserved under
  the Databricks domain).
- The MosaicML `llm-foundry` repository, which contains the
  full recipe, dataset composition, and training loop.
- The subsequent MPT-30B blog and model card, which extended
  the recipe to 30 B parameters and published a comparable cost
  breakdown.

### The recipe

Consult the MPT-7B model card / blog post for the authoritative
numbers; the salient parameters as reported by MosaicML are:

- Model: 7 B parameter decoder-only Transformer with ALiBi
  positional embeddings and no bias terms in linear layers.
- Data: `~1 T` tokens from a curated mixture of C4, mC4, S2ORC,
  and web/code corpora — recipe detailed in the blog.
- Sequence length: 2048 during pretraining; extended in some
  variants.
- Hardware: MosaicML's H100 / A100 cluster (the blog reports
  training on 440× A100-40GB for the primary MPT-7B run and
  gives a per-generation cost figure).
- Framework: MosaicML `composer` + PyTorch FSDP.
- Wall-clock and dollar figure: MosaicML publicly quoted the run
  at "approximately $200 K on the MosaicML platform, in about
  9.5 days" for the primary MPT-7B variant. Consult the blog
  for the current version of that figure.

### The chapter arithmetic against MPT-7B

Let's price MPT-7B from first principles with this module's
formulas and see how the answer compares to MosaicML's public
figure.

- `N = 7 · 10^9`, `D = 1 · 10^12`.
- `C = 6 · N · D = 6 · 7e9 · 1e12 = 4.2 · 10^22` FLOPs.
- Hardware: A100-40GB SXM4. Dense BF16 peak `P = 312 · 10^12`.
  Source: NVIDIA A100 datasheet.
- Assume `μ_sus ≈ 0.40` — MosaicML reported strong MFU numbers
  on their cluster; consult the blog for the specific figure.

GPU-hours (chapter 2):

```
GPU_hours = 4.2e22 / (312e12 · 0.40 · 3600) ≈ 93 500 A100-hours
```

At MosaicML's roughly-A100-hourly rate for their platform at the
time (consult their published pricing history for the exact
number), the total lands in the low hundreds of thousands of
dollars — consistent with the publicly-quoted $200 K figure.
The exercise: reconcile your own arithmetic against the public
figure and identify where the discrepancy sits (usually
`μ_sus`, `η(G)`, or the effective platform per-hour rate).

Wall-clock: `93 500 / 440 ≈ 213 hours ≈ 8.9 days`, roughly
matching the published 9.5 days (the extra half-day is
consistent with a `η(440) < 1` and normal goodput losses).

### What MPT-7B teaches

- **Small-parameter, above-Chinchilla-D recipes are cost-
  competitive against the Chinchilla-optimal frontier for
  serving-heavy products.** MPT-7B trained on ~143 tokens per
  parameter — 7× Chinchilla — and got a model competitive with
  Llama 1 7B at a comparable cost.
- **A well-tuned, mid-scale run on a well-run cluster hits
  ~$200 K.** This is the number to remember for "small
  frontier-grade LLM" cost. Any proposal in the same shape
  that comes back with a $2 M estimate needs a very specific
  reason for the 10× discrepancy.
- **Publishing the recipe and the dollar figure is unusual.**
  MosaicML did this as a demonstration; most training programmes
  do not. The MPT-7B blog remains the single most useful
  public cost anchor at 7 B scale.
- **The A100 numbers date the anchor.** A modern equivalent run
  on H100 (roughly 3.2× the BF16 throughput per GPU) would
  finish in ~30% of the wall-clock or on ~30% of the GPU count
  — but at a higher H100 hourly rate. The dollar figure moves;
  the recipe shape does not.

## Recipe 2: Llama 3 (the frontier anchor)

Meta AI published the Llama 3 training report as Grattafiori et
al. (2024), "The Llama 3 Herd of Models", arXiv 2407.21783. It
is the most detailed public account of a frontier-scale
training run available as of this writing; every training-
pipeline engineer should read section 3 at least twice.

### The recipe

Consult the paper for the authoritative details. Key numbers
from Grattafiori et al. (2024):

- Models: 8 B, 70 B, and 405 B parameter decoder-only
  Transformers.
- Data: ~15 T tokens (all three models) — a mixture curated
  down from a much larger web corpus, with detailed
  deduplication and quality-filtering described in §3.1.
- Sequence length: extended in stages; the paper details the
  long-context extensions.
- Hardware: Meta's H100 clusters. The paper reports the primary
  training run on a 16 K-GPU cluster, with details of the
  parallelism (tensor, pipeline, data, sequence — a 4-D
  parallel decomposition) in §3.2.
- Framework: Meta's internal framework built on PyTorch with
  torchtitan-lineage components.
- Total compute for the 70 B model: `~39.3 M H100-hours` per
  the paper (§3.3, table 5 in some print versions — verify
  against the current arXiv PDF).
- Wall-clock: on the order of months on the 16 K-GPU cluster,
  with interruption / restart accounting in §3.3.2.

### The chapter arithmetic against Llama 3 70B

Price Llama 3 70B from first principles:

- `N = 70 · 10^9`, `D = 15 · 10^12`.
- `C = 6 · N · D = 6 · 70e9 · 15e12 = 6.3 · 10^24` FLOPs.
- Hardware: H100 SXM5, dense BF16 peak `P = 989 · 10^12`.
- Assume `μ_sus ≈ 0.40`. Llama 3 paper §3.4 reports its actual
  observed MFU; consult it for the exact figure and re-run
  the arithmetic with the paper's number.

GPU-hours:

```
GPU_hours = 6.3e24 / (989e12 · 0.40 · 3600) ≈ 4.42 · 10^6 H100-hours
```

Wait — that is 4.4 M, but Meta reports 39.3 M. What is the
discrepancy?

Two things:

- **Scaling loss at 16K GPUs.** The `η(G)` term at this scale
  is well below 1. If the effective sustained MFU on the
  paper's parallelism was ~0.20 rather than the small-scale
  0.40, the arithmetic multiplies by ~2×.
- **Attention correction at Llama-3's long context.** Llama 3
  extends context length during training; the attention-
  quadratic term becomes a substantial addition to `C` for the
  long-context portions.
- **Goodput and restart losses.** Meta's paper reports an
  interruption taxonomy over the run (see §3.3.2); the
  aggregate goodput sits well below `μ_sus = 0.40`'s implied
  ceiling.

The exercise for the reader (and for the module capstone,
chapter 7's feasibility study): read §3.3 of Llama 3 carefully
and reconcile the arithmetic against the reported 39.3 M
figure. The reconciliation will name a specific `μ_sus`
between 0.05 and 0.20 that produces the reported total; that is
the sustained MFU Meta actually achieved at frontier scale on
this workload. It is much lower than the small-scale MFU number
many teams quote — and *that* is the number that goes into a
frontier-run feasibility study.

### What Llama 3 teaches

- **At frontier scale, `η(G)` and goodput dominate `μ_nom`.**
  The nominal MFU on a well-optimised small run is 40–55% on
  H100 BF16; the sustained MFU on a frontier run is a
  substantially smaller number. Assume the small-scale number
  in a frontier budget and the estimate will be 3–5× low.
- **The interruption taxonomy is workload-shaped, not vendor-
  shaped.** Meta reports specific classes and rates in §3.3.2;
  chapter 4 of this module and chapter 5 of mod-106 both cite
  this taxonomy. A frontier feasibility study should model
  interruption rates against this reference, adjusted for the
  team's cluster scale.
- **Publishing the recipe and the interruption chronicle in
  detail is a public-good act.** As of 2026, Llama 3 is the
  best-documented frontier-scale training run in the open
  literature. Grattafiori et al. (2024) alongside the OPT-175B
  chronicles (Zhang et al. 2022 + the metaseq logbook) are the
  standard references.
- **The paper's total-cost lens is compute, not dollars.**
  Meta does not publish the dollar cost of Llama 3
  pretraining. To translate 39.3 M H100-hours to dollars,
  multiply by an assumed per-hour rate (chapter 3's
  arithmetic) — reserved-instance-level rates are the honest
  bracket. Public estimates put Llama 3 70B pretraining in
  the tens to low hundreds of millions of dollars, depending
  on the assumed per-hour rate; the paper itself does not
  commit to a figure.

## Bracketing your run between the anchors

The two recipes bracket a wide space. To locate a new proposal
in it:

- **Ratio your `C` to the two anchors.** MPT-7B is `4.2 · 10^22`
  FLOPs; Llama 3 70B is `6.3 · 10^24`. A 34 B Chinchilla-
  optimal run (`C = 1.39 · 10^23`) is roughly 3× MPT-7B and
  1/50 of Llama 3 70B — a mid-scale run. A 34 B inference-
  optimal run (`C = 9.71 · 10^23`) is 23× MPT-7B and 1/6 of
  Llama 3 70B — a large-but-not-frontier run.
- **Ratio your dollar cost to the two anchors' costs.** If the
  arithmetic says your run is $2 M for a `C` that is 5× MPT-
  7B's, you would expect roughly `5 · $200 K = $1 M` from a
  straight ratio. The 2× discrepancy needs to be explained: is
  it hardware generation, per-hour rate, sustained MFU, or
  something else? The reconciliation is what makes the number
  defensible.
- **Ratio your `η(G)` to the anchors.** MPT-7B ran on 440 GPUs;
  Llama 3 on 16 K. A proposal at 4 K GPUs sits between them —
  the applicable `η` should be closer to MPT-7B's than to
  Llama 3's, but not equal to either. Cite the ratio.
- **Ratio your `D / N` to the anchors.** MPT-7B is `143:1`
  (well above Chinchilla, cost-optimised for the parameter
  count). Llama 3 8B is `1875:1` (aggressively inference-
  optimised). Chinchilla-optimal is `20:1`. A proposal's ratio
  puts it on a spectrum with these three named points.

## Failure modes when using anchors

- **Copying an anchor's `μ_sus` without accounting for hardware
  differences.** MPT-7B was A100; Llama 3 was H100 with a
  different parallelism strategy. Sustained MFU is not a
  hardware property; it is a workload-on-hardware property.
  Take the anchor as a *bound*, not a *value*.
- **Copying an anchor's dollar figure without accounting for
  the per-hour rate at the time.** MosaicML's $200 K used
  their platform's rate in 2023; the same run on AWS reserved
  in 2026 has a different number. Redo the per-hour arithmetic.
- **Extrapolating linearly from a small anchor to a frontier
  proposal.** The scaling terms (`η(G)`, goodput, restart
  rate) do not scale linearly. A 100× compute jump does not
  imply a 100× dollar jump; it implies more.
- **Treating the anchors as prescriptive rather than
  illustrative.** MPT-7B and Llama 3 are examples of specific
  design choices at specific times. They are not "the right
  recipe"; they are references against which to defend a new
  proposal.

## The scaling-up vs. scaling-down spectrum, summarised

| Axis                          | MPT-7B (cost-optimised)                   | Llama 3 70B (frontier)                          |
|-------------------------------|-------------------------------------------|--------------------------------------------------|
| `N`                           | 7 B                                       | 70 B                                             |
| `D`                           | 1 T                                       | 15 T                                             |
| `D / N`                       | 143 (above Chinchilla, cost-optimised)     | 214 (above Chinchilla, deliberately)             |
| `C = 6ND`                     | 4.2 · 10^22 FLOPs                          | 6.3 · 10^24 FLOPs                                |
| Cluster                       | 440 A100                                  | 16 K H100                                        |
| Wall-clock                    | ~9.5 days                                  | months (see paper §3.3)                          |
| Cost anchor                   | ~$200 K (MosaicML blog)                   | Not publicly quoted; on the order of many $M+   |
| Parallelism                   | FSDP                                      | 4-D parallel (tensor + pipeline + data + seq)   |
| `μ_sus` (community anchor)     | ~0.40 (small-scale)                        | Substantially lower; reconcile from paper       |
| Best-fit lease tier            | Reserved / on-prem, mid-scale             | On-prem SuperPOD or DGX Cloud multi-year commit |
| Reproducibility artifacts     | llm-foundry repo, blog, model card         | Paper, model card, interruption chronicle       |

Every real feasibility study places its target in this table.

## Summary

- MPT-7B is the *cost-optimised* anchor: 7 B parameters, 1 T
  tokens, ~$200 K on a ~440-GPU A100 cluster, publicly
  documented recipe and cost. The best reference for a
  small-mid frontier-grade LLM cost estimate.
- Llama 3 is the *frontier* anchor: 8/70/405 B parameters,
  ~15 T tokens, 16 K-GPU H100 cluster, months of wall-clock,
  interruption chronicle and detailed cost taxonomy in the
  paper. The best reference for a large-scale run.
- The two anchors bracket the modern design space. Locate a
  new proposal on the ratio axes — `C` ratio, dollar ratio,
  `η(G)` ratio, `D / N` ratio — and any discrepancy needs a
  named reason.
- Sustained MFU at frontier scale is much lower than at small
  scale. Copying a small-cluster MFU into a frontier budget
  under-estimates by 3–5×.
- Chapter 7 uses these two anchors as the reference set the
  feasibility study defends its estimate against.
