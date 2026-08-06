# mod-109 — Cost, Capacity, and Cluster Economics for Training Runs

**Estimated effort:** 14 hours

The previous eight modules taught you to build a training run
that converges, survives, and stays close to peak. This module
teaches you to *price* it. A product team walks in with a
target — "a 34 B chat model, delivered by Q4, better than the
current baseline on our eval" — and needs to know what that
costs, on what hardware, by when, and with what risk. The
answer is a two-to-five-page **training-run feasibility study**,
and this module is the machinery that produces it.

The through-line: `(N, D) → C → GPU-hours → dollars → cluster
shape → wall-clock → risk register`. Chapter 1 turns the product
target into `(N, D)` via the scaling laws. Chapter 2 turns
`C = 6·N·D` into GPU-hours via MFU. Chapter 3 turns GPU-hours
into dollars and picks the cluster shape. Chapter 4 picks the
lease tier (reserved, spot, dedicated, on-prem). Chapter 5
translates every MFU point and every checkpoint decision into
dollars. Chapter 6 calibrates the whole pipeline against two
public anchors, MPT-7B (cost-optimised) and Llama 3
(frontier). Chapter 7 is the feasibility study itself.

## Learning objectives

- Apply Chinchilla / Kaplan scaling laws to translate a product
  target (parameter count × quality target) into a data
  budget.
- Convert data + parallelism strategy into GPU-hours and
  dollars for a target cluster shape.
- Reason about reserved-vs-spot / dedicated-vs-shared / on-
  prem-vs-cloud economics for training runs.
- Model the cost impact of MFU improvements and checkpointing
  / restart budgets.
- Author a training-run feasibility study (Chinchilla-scaled
  compute → cluster shape → dollar cost → wall-clock schedule).
- Reason about scaling-up (frontier) vs. scaling-down (cost-
  optimised) recipes with MPT-7B and Llama 3 as anchors.

## Chapters

1. [Scaling Laws and the Data Budget](01-scaling-laws-and-the-data-budget.md)
   — the `C = 6·N·D` compute ledger, Kaplan (2020) vs.
   Chinchilla (Hoffmann 2022), the "20 tokens per parameter"
   rule, the compute-optimal vs. inference-optimal distinction,
   and the decision list for picking `(N, D)`.
2. [From FLOPs to GPU-Hours](02-flops-to-gpu-hours.md)
   — the `GPU_hours = C / (P · μ_sus · 3600)` identity, the
   dense vendor peak, the split of MFU into `μ_nom · goodput`,
   the scaling-efficiency `η(G)`, the attention correction for
   long context, and the MFU / HFU convention.
3. [GPU-Hours to Dollars and the Cluster-Shape Choice](03-gpu-hours-to-dollars-and-cluster-shape.md)
   — vendor SKUs (`p5.48xlarge`, `a3-highgpu-8g`, `ND_H100_v5`,
   DGX H100), on-prem amortisation, the wall-clock vs. GPU-
   hours vs. lease-tier three-way trade, and the side-term
   budget (storage, staging, egress, checkpoints, dev, staff).
4. [Reserved vs. Spot, Dedicated vs. Shared, On-Prem vs.
   Cloud](04-reserved-vs-spot-dedicated-vs-shared.md) — the
   lease-tier ladder (on-demand → reserved → capacity block →
   spot → on-prem), dollars per *productive* GPU-hour, the
   four tier-decision patterns, and dedicated-vs-shared queue
   economics.
5. [MFU Uplift and Checkpointing Budgets to Dollars](05-mfu-uplift-and-checkpointing-budgets-to-dollars.md)
   — the linear dollars-per-MFU-point identity, the goodput-
   to-dollars identity, Young's checkpoint-cadence formula,
   straggler / SDC detection as insurance, the inference-
   lifetime dollar, and the MFU roadmap as a business proposal.
6. [Frontier vs. Cost-Optimised Recipes: MPT-7B and Llama
   3](06-frontier-vs-cost-optimised-recipes-mpt-7b-and-llama-3.md)
   — MPT-7B as the ~$200 K, 7 B, 1 T-token cost-optimised
   anchor; Llama 3 70B as the 39.3 M H100-hour, 16 K-GPU
   frontier anchor; the ratio sanity checks a feasibility
   study defends against both.
7. [The Feasibility Study](07-the-feasibility-study.md) — the
   seven-section artifact (product target, recipe, compute
   budget, cluster shape, dollar range, risk register, kill
   criteria) plus the one-page front matter for stakeholders;
   how the study lives during the run.

## Exercises

- [exercise-01 — Chinchilla scaled compute budget](exercises/exercise-01-chinchilla-scaled-compute-budget.md) (3 h)
- [exercise-02 — GPU-hours and dollars for a target cluster](exercises/exercise-02-gpu-hours-and-dollars-for-target-cluster.md) (3 h)
- [exercise-03 — Reserved vs. spot economics](exercises/exercise-03-reserved-vs-spot-economics.md) (3 h)
- [exercise-04 — MFU uplift to dollars](exercises/exercise-04-mfu-uplift-to-dollars.md) (2 h)
- [exercise-05 — Feasibility study capstone](exercises/exercise-05-feasibility-study-for-a-target-product.md) (3 h)

## Labs and quizzes

- `labs/` — an end-to-end feasibility-study lab (author two
  full studies, one at MPT-7B scale and one at frontier scale,
  and defend them in a mock review) lands here on the next
  autonomous cycle.
- `quizzes/` — one knowledge check lands here on the next
  autonomous cycle.

## Resources

- [resources.md](resources.md) — Chinchilla / Kaplan / Llama
  papers, the MPT-7B blog post, hyperscaler pricing pages,
  NVIDIA datasheets, DGX / capacity-block references, and a
  recommended reading order.

## How the module fits together

Chapter 1 is the *input* — the recipe. Chapter 2 is the
*compute-to-hours* leg. Chapter 3 is the *hours-to-dollars-
and-cluster-shape* leg. Chapter 4 is the *lease-tier* choice
that determines the per-hour multiplier. Chapter 5 is the
*sensitivity* — how much every MFU point, every goodput point,
every checkpoint decision costs in dollars, which is what
turns mod-107 and mod-106 optimisations into fundable
proposals. Chapter 6 is the *calibration* — every proposal
gets ratioed against MPT-7B and Llama 3 before being
published. Chapter 7 is the *artifact* — the seven-section
feasibility study that everything else composes into. The
exercises walk the same arc: 1 does the scaling-law
arithmetic; 2 combines chapters 2 and 3 into a GPU-hours-and-
dollars calculator; 3 does the tier arithmetic; 4 does the
MFU-to-dollars translation; 5 is the capstone.

## What this module deliberately does not cover

- **Distributed-training semantics** (DDP, FSDP2, TP, PP, 3D-
  parallel) — owned by mod-101. This module *consumes* their
  parallelism-efficiency numbers as inputs; it does not
  re-derive them.
- **Framework internals** (Megatron, DeepSpeed, torchtitan) —
  owned by mod-102.
- **Data pipeline design and shard formats** — owned by
  mod-103. The data-loader cost line is a mod-103 measurement.
- **Scheduler and topology** — owned by mod-104. The
  dedicated-vs-shared discussion in chapter 4 stops at "the
  scheduler policy is a cost input"; the scheduler
  implementation is mod-104.
- **Fabric configuration and NCCL tuning** — owned by mod-105.
  The `η(G)` term in chapter 2 is a mod-105 outcome; this
  module treats it as a measured input.
- **Checkpointing, elastic training, incident classification,
  and the goodput SLO** — owned by mod-106. Chapter 5 of this
  module quantifies the dollar impact of mod-106's mechanisms;
  the mechanisms themselves are mod-106.
- **MFU, HFU, kernel/comm efficiency** — owned by mod-107.
  Chapter 5 of this module quantifies the dollar impact of
  mod-107's optimisations; the optimisations themselves are
  mod-107.
- **Observability and reproducibility** — owned by mod-108.
  The weekly re-price cycle in chapter 7 depends on
  mod-108's dashboards and tracker; this module treats them as
  inputs.
- **Cross-team org contract, RFC process, sign-off authority,
  build-vs-buy strategy** — owned by mod-110. The feasibility
  study of chapter 7 is the artifact; the *workflow* around it
  is mod-110.
- **Post-training economics** (fine-tuning cost, evaluation
  cost, serving cost) — outside this track. Chapter 5's
  inference-lifetime dollar is the boundary; the details of
  serving cost belong to the serving platform track.

## The one number every stakeholder learns from this module

At the exit of this module, every training-pipeline engineer
should be able to state: "each additional percentage point of
sustained MFU on the current large run is worth
`$_run / (μ_sus · 100)` dollars, which at our scale is `$X`,
and across the next-generation run at our scale is `$Y`." That
one sentence — and the ability to defend it — is what makes
kernel and reliability work fundable in a business context.
