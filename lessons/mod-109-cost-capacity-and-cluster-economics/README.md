# mod-109 — Cost, Capacity, and Cluster Economics for Training Runs

**Estimated effort:** 14 hours

mod-101 through mod-108 gave you the engineering surface of the
training platform. This module gives you the *economics* — the
back-of-envelope arithmetic that turns "we want a 30B model competitive
with X" into a defensible dollar figure, wall-clock schedule, and
cluster shape. Every conversation with a VP or a CFO about a training
run eventually reduces to the numbers this module teaches you to
produce.

By the end of the module you should be able to (a) apply Kaplan (2020)
and Chinchilla (Hoffmann et al., 2022) scaling laws to turn a product
target (parameter count `N`, quality target) into a compute budget `C`
in FLOPs, (b) turn `C` into GPU-hours and dollars on a specific cluster
shape given peak FLOPs/s per GPU and a defensible MFU assumption, (c)
reason about reserved / spot / dedicated / on-prem trade-offs and pick
a buying mix, (d) monetise MFU uplift so mod-107 engineering work is
legible to the CFO, (e) author a training-run feasibility study — the
canonical deliverable that lands in the go / no-go conversation — and
(f) read MPT-7B and Llama 3 as anchor recipes for the scaling-down and
scaling-up ends of the cost-vs-quality curve.

## Learning objectives

- Apply Chinchilla / Kaplan scaling laws to translate a product target
  (parameter count × quality target) into a data budget.
- Convert data + parallelism strategy into GPU-hours and dollars for a
  target cluster shape.
- Reason about reserved-vs-spot / dedicated-vs-shared / on-prem-vs-cloud
  economics for training runs.
- Model the cost impact of MFU improvements and checkpointing /
  restart budgets.
- Author a training-run feasibility study (Chinchilla-scaled compute →
  cluster shape → dollar cost → wall-clock schedule).
- Reason about scaling-up (frontier) vs. scaling-down (cost-optimised)
  recipes with MPT-7B and Llama 3 as anchors.

## Chapters

1. [Scaling Laws for Compute-Optimal Budgeting](01-scaling-laws-for-compute-optimal-budgeting.md) —
   Kaplan (2020) power-law loss, Chinchilla (Hoffmann et al., 2022)
   `D ≈ 20 · N` compute-optimal recipe, the `6 · N · D` FLOP model, when
   it fails (MoE, sparse attention, long-context), and how to translate
   a product target into a compute budget in three lines of arithmetic.
2. [From FLOPs to GPU-Hours to Dollars](02-flops-to-gpu-hours-to-dollars.md) —
   peak FLOPs/s per GPU from the datasheet, effective FLOPs/s at a
   defensible MFU, aggregate cluster FLOPs/s, wall-clock, GPU-hours,
   and dollars. Two worked examples (7B / 200 B on 512 × H100 and 70B
   / 1.4 T on 2048 × H100) plus a sensitivity table for MFU, GPU
   generation, and precision.
3. [Reserved vs. Spot vs. On-Prem: The Buying Menu](03-reserved-vs-spot-vs-on-prem.md) —
   the three-axis space of term (on-demand / reserved / spot), sharing
   (dedicated / multi-tenant), and ownership (cloud / neocloud /
   on-prem). Break-even for on-prem vs. 3-year reserved cloud; when
   spot is viable and when it is malpractice; how the axes compose into
   a real procurement plan.
4. [MFU Uplift as Dollars](04-mfu-uplift-as-dollars.md) — the bridge to
   mod-107. Every point of MFU translated into a dollar figure per run;
   a three-shape reference table; the ROI arithmetic for FA3 + FP8;
   a one-page decision brief template you bring to prioritisation
   meetings.
5. [The Training-Run Feasibility Study](05-training-run-feasibility-study.md) —
   the deliverable this module trains you to produce. Eight-section
   template with a fully worked 13B / 300 B example, a numbered
   assumptions block, and a priced risk register. The scaffold for
   exercise 05 and lab 01.
6. [Anchor Recipes: MPT-7B and Llama 3](06-anchor-recipes-mpt7b-llama3.md) —
   two public runs read as feasibility studies. MPT-7B as the "scaling
   down" anchor (7B, ~1 T tokens, ~$200 k on A100s, optimised for
   dollars per point of eval quality). Llama 3 (Grattafiori et al.,
   2024) as the "scaling up" anchor (405B, ~15 T tokens, 4-D
   parallelism on ~16 k H100s, ~$50 M back-of-envelope). What each
   optimised for and how your feasibility study reads their choices.

## Exercises

- [exercise-01 — Chinchilla-scaled compute budget](exercises/exercise-01-chinchilla-scaled-compute-budget.md) (3 h)
- [exercise-02 — GPU-hours and dollars for a target cluster](exercises/exercise-02-gpu-hours-and-dollars-for-target-cluster.md) (3 h)
- [exercise-03 — Reserved-vs-spot economics](exercises/exercise-03-reserved-vs-spot-economics.md) (3 h)
- [exercise-04 — MFU uplift to dollars](exercises/exercise-04-mfu-uplift-to-dollars.md) (2 h)
- [exercise-05 — Feasibility study for a target product](exercises/exercise-05-feasibility-study-for-a-target-product.md) (3 h)

## Labs and quizzes

- `labs/` — `lab-01` (training-run feasibility doc) lands here on the
  next autonomous cycle. It builds on exercise 05: a full ~5-page
  feasibility study for a stated product target, complete with real
  vendor quotes, an MFU baseline from a small ladder run, and a
  priced risk register.
- `quizzes/` — one knowledge check lands here on the next autonomous
  cycle.

## Resources

- [resources.md](resources.md) — primary scaling-law papers (Kaplan
  2020, Chinchilla 2022), hardware datasheets (H100, A100, DGX H100
  whitepaper), cloud pricing pages (marked `needs-research` because
  they move), case studies (Llama 3 herd paper, MPT-7B blog, OPT-175B
  logbook), reserved / spot analyses, and a recommended reading order.

## How the module fits together

Chapter 1 fixes the budgeting instrument (`C = 6 · N · D`, Chinchilla
`D ≈ 20 · N`). Chapter 2 turns `C` into GPU-hours and dollars on a
specific cluster shape at a specific MFU. Chapter 3 makes the buying
mix explicit — reserved, spot, on-prem, dedicated vs. shared — so the
dollar figure has a real procurement plan behind it. Chapter 4 is the
ROI bridge to mod-107 engineering: every point of MFU is a dollar
figure, and this is where you defend the MFU work to your VP. Chapter 5
composes the previous four into the feasibility-study deliverable that
you actually ship to the go / no-go conversation. Chapter 6 calibrates
your feasibility study against two public anchors (MPT-7B and Llama 3)
so your MFU, cost, and cluster-shape assumptions can be defended
against real runs.

The exercises march in the same order. Do them in sequence; the
feasibility-study exercise (05) is downstream of every previous one and
is the deliverable that ties the module together.

## What this module deliberately does not cover

- **Parallelism strategy and NCCL collective cost** — owned by mod-101.
  This module *uses* mod-101's `6 · N · D` FLOP accounting and its
  comm-to-compute reasoning as inputs, but the strategy choice itself
  lives there.
- **Framework internals (FSDP2, DeepSpeed, Megatron, NeMo, JAX)** —
  owned by mod-102. This module *uses* the framework as a black-box
  MFU number.
- **Training-scale data pipelines** — owned by mod-103. This module
  *uses* loader throughput as a line item in the risk register.
- **Cluster orchestration, quota, fair-share, gang preemption** —
  owned by mod-104. This module *uses* mod-104's quota model as the
  interface between the buying mix and the run.
- **Fabric and storage architecture** — owned by mod-105. This module
  *uses* fabric shape as a constraint on cluster-shape choice.
- **Checkpointing, fault tolerance, elastic training** — owned by
  mod-106. This module *uses* mod-106's availability budget as an
  input to the feasibility study risk register; it does not teach how
  to build DCP or torchrun-elastic.
- **MFU engineering (FA3, FP8, activation checkpointing, `torch.compile`,
  communication overlap)** — owned by mod-107. This module *monetises*
  mod-107's outputs but does not teach how to engineer them.
- **Training-run observability, dashboards, and reproducibility** —
  owned by mod-108. This module *uses* mod-108's MFU dashboard as the
  alerting surface for regressions priced in chapter 4.
- **Platform-architecture altitude, RFC authoring, build-vs-buy at the
  org level, cross-team hand-off contracts** — owned by mod-110. This
  module is one input to that altitude, not a substitute for it.
