# Build-vs-Buy at Level-35 Altitude

Every 12–18 months, someone in leadership asks: "why are we
running our own training stack? Wouldn't it be cheaper to buy
this from Databricks, or Together, or NVIDIA?" The right
response is not a Slack thread; it is a 4–6 page decision memo
that names the four canonical options, applies a shared
decision matrix, and lands with a defensible recommendation.

This chapter is that memo's template. It is not an argument
that in-house is always right or that hosted is always right —
both are wrong. It is the machinery to make the call, defend it
in front of leadership, and re-open it annually as the market
moves. This is the level-35 decision that binds capital and
multiple teams for multiple quarters; it must be written and
signed off.

## The four canonical options

Four options recur across the industry. Every proposal in this
space maps to one of them (or a hybrid — a common answer).

### Option A — Build your own on Megatron-LM (or Megatron-Core)

Run NVIDIA/Megatron-LM (or the more modular Megatron-Core) in
your own cluster, on your own accelerators, with your own team
integrating parallelism strategies (TP / PP / EP / CP), FSDP2,
Transformer Engine, and NCCL tuning. You own everything from
the container image up.

- **Owned by you.** Everything. Recipe, kernels (via mod-107),
  scheduler, storage, observability, incident response.
- **Bought from vendors.** Only the compiler stack (CUDA, cuDNN,
  NCCL) and the kernel-level primitives (FlashAttention v2/v3,
  Transformer Engine).
- **Anchor references.** Megatron-LM (Shoeybi et al., 2019,
  https://arxiv.org/abs/1909.08053), the Megatron-LM GitHub
  repository, NVIDIA/Megatron-Core.
- **Anchor users.** NVIDIA's own reference training stack;
  cited throughout the Llama 3 paper (Grattafiori et al., 2024,
  §3) as the primary in-house tradition frontier labs converge
  toward.

### Option B — Integrate an open-source higher-level framework (NeMo, torchtitan)

Run NVIDIA NeMo or PyTorch/torchtitan on your own cluster.
Both wrap Megatron-style parallelism in a higher-level
recipe / experiment interface. NeMo is the more feature-
complete production framework; torchtitan is the leaner PyTorch-
native reference implementation.

- **Owned by you.** Cluster, scheduler, storage, observability,
  recipe adaptation, ops.
- **Bought from vendors (open source).** The framework's
  training loop, parallelism composition, and recipe library.
- **Anchor references.** NVIDIA/NeMo
  (https://github.com/NVIDIA/NeMo), NeMo documentation, and
  pytorch/torchtitan
  (https://github.com/pytorch/torchtitan).
- **Anchor users.** NVIDIA's own developer-relations recipes,
  the Nemotron model family, and torchtitan-based training in
  the PyTorch ecosystem.

### Option C — Buy hosted training from a hyperscaler-aligned platform (Databricks/Mosaic)

Databricks Mosaic AI Training (the successor to MosaicML's
Composer/LLM Foundry offering) provides managed training
infrastructure on Databricks-managed clusters. The customer
supplies recipes and data; the platform provides the
orchestration, the checkpoint durability, and the elastic
recovery.

- **Owned by you.** Recipe, data, evaluation, model IP.
- **Bought.** Cluster, scheduler, checkpoint durability,
  elastic recovery, monitoring surface, most of on-call.
- **Anchor references.** Databricks Mosaic AI Training
  documentation
  (https://docs.databricks.com/en/machine-learning/foundation-models/index.html),
  MosaicML Composer / LLM Foundry
  (https://github.com/mosaicml/composer,
  https://github.com/mosaicml/llm-foundry),
  the MPT-7B blog post (Mosaic, 2023,
  https://www.databricks.com/blog/mpt-7b) as the
  build-cost anchor.
- **Anchor users.** Enterprise customers pretraining and
  fine-tuning on Databricks; historically MosaicML customers
  before the 2023 acquisition.

### Option D — Buy training-as-a-service from a specialist provider (Together AI, others)

Together AI (and similar specialists — CoreWeave with managed
training layers, Modal, Anyscale for Ray-native training)
provides managed training on their capacity. Customer supplies
recipe and data; provider supplies everything else, including
the accelerators.

- **Owned by you.** Recipe, data, evaluation, model IP.
- **Bought.** Every other layer — accelerators, cluster,
  scheduler, storage, on-call, observability.
- **Anchor references.** Together AI documentation
  (https://docs.together.ai/), Together AI training endpoints.
- **Anchor users.** Companies that have chosen to outsource
  the entire infra layer, typically because they have a small
  team and a compressed schedule.

## The seven-axis decision matrix

Every option gets rated on the same seven axes. The output is a
qualitative table that goes in the memo. Numbers are for
illustration; a real memo cites its measurements.

| Axis                          | A — Megatron | B — NeMo/torchtitan | C — Databricks/Mosaic | D — Together etc. |
|-------------------------------|--------------|---------------------|-----------------------|-------------------|
| Total cost of ownership (2yr) | Highest capex; lowest per-hour opex if scale is high | Similar to A minus recipe eng | Middle; managed premium ~10–30% over raw compute | Highest per-hour; near-zero capex |
| Time-to-first-run             | 3–6 months   | 2–4 months          | Weeks                 | Days              |
| Team headcount required       | 8–15 platform ICs | 5–10               | 2–4 (integration)     | 1–2 (integration) |
| Capability ceiling            | Frontier (Llama 3-class) | Frontier (with effort) | Below frontier for very large runs (~today) | Below frontier |
| Lock-in and portability       | Low (own the stack)      | Low (open source)      | Medium (Databricks-native APIs) | High (provider APIs) |
| Roadmap control               | Full         | Full (upstream)     | Partial (feature request) | None |
| Auditability / provenance     | Full         | Full                | Partial (managed layer opaque) | Partial |

Read the axes top-to-bottom in this order — they are ranked by
what leadership actually asks about, not by what platform
engineers care about. Cost and time-to-first-run come first
because they are the two numbers in the recommendation
paragraph.

## Costing the two-year TCO

The dollar arithmetic is where most build-vs-buy memos are
lightest, because the components differ across options. Use
this cost decomposition for every option so the numbers are
comparable.

Two-year total = capex + opex + labor + risk-adjusted overhead

- **Capex (options A, B).** Hardware amortisation across 3–4
  years, per mod-109 chapter 3. If leasing cloud capacity
  (reserved / capacity blocks), amortise the reservation
  commitment instead. Options C and D: near zero.
- **Opex (all options).** Electricity, cooling, network
  transit for on-prem; per-hour rate for cloud; per-GPU-hour
  service fee for hosted. Cite mod-109 chapter 4's
  per-productive-GPU-hour lens — the naive per-hour rate is
  not the right unit.
- **Labor (all options).** Loaded cost per platform IC × FTE
  count × 2 years. Options A and B are labor-heavy; options C
  and D are labor-light. A 6-IC platform team fully loaded at
  $300 K/year is $3.6 M / 2 years — the largest single line in
  most in-house budgets.
- **Risk-adjusted overhead.** Cost of one Sev-1 incident × its
  probability, per option. In-house options own the risk;
  hosted options transfer it (partially) to the provider's
  SLA. Do not zero this line for hosted — read the SLA credits
  and understand they are pennies against a Sev-1's cost.

For a fair comparison, hold three inputs constant across
options: the recipe (`N`, `D`, precision), the target
`μ_sus`, and the wall-clock target. Then vary the platform.
The output is four dollar numbers with three-year amortisation,
side by side.

A typical result at *mid-scale* (e.g., a 34 B pretraining plus
routine fine-tuning workload consuming 512 H100 equivalents
year-round) — this is *illustrative*, not a lookup:

- Option A crosses over option C somewhere around 200–400
  H100-year-equivalents of sustained use, historically.
  Below that scale, C is cheaper.
- Option D is almost always more expensive per productive
  GPU-hour than A or B at scale, and almost always cheaper on
  the time-to-first-run axis.
- Option B has the same crossover as A but with lower
  time-to-first-run.

The crossover point is not stable — hardware prices, managed-
service pricing, and headcount costs all move. Re-run the
arithmetic annually.

## Capability ceiling and the frontier question

The capability ceiling is what determines whether the platform
can support the roadmap two years out. Two questions:

- **Can this option run our largest planned model?**
  The largest planned pretraining shape sets the floor for
  parallelism support, checkpoint durability, and elastic
  recovery. Options A and B run at frontier scale; options C
  and D historically have not (their operating envelopes are
  wide but do not always reach the largest publicly-reported
  runs).
- **Can this option adopt the next kernel and format
  improvements on our timeline?** FP8 with Transformer Engine,
  FlashAttention v3, DCP-async — do they land inside our
  compatibility window? For options A and B you own that
  timeline; for C and D the vendor does.

If the answer to either is "no", the option is disqualified
regardless of TCO.

## Lock-in and roadmap control

Lock-in is not a boolean. Rank it on three sub-axes:

- **Data-plane lock-in.** Can our training data leave? Options
  C and D generally allow customer data to remain in customer-
  owned storage; verify contractually.
- **Format lock-in.** Are checkpoints portable? Options A and B
  produce PyTorch-native or safetensors checkpoints usable
  anywhere. Options C and D vary; some produce native formats
  that require the provider's tools to consume.
- **API lock-in.** How much recipe code is tied to the
  vendor's SDK? Read the provider's example recipes; count
  the vendor-specific import lines. Options A and B: zero.
  Options C and D: order-of-hundreds is typical.

The exit cost from a hosted option is the labor cost to
re-integrate on options A or B. Bound it explicitly in the
memo.

## Time-to-first-run and organisational maturity

The fastest path to a running training loop is usually option
D, followed by C, then B, then A. But that is only relevant to
teams whose blocker is time-to-first-run.

Teams that already have a working platform and a functioning
on-call rotation should not weight time-to-first-run heavily —
they have already paid that cost. Teams that are net-new should
weight it strongly; the alternative is spending six months
building a platform whose need is not yet proven.

The corollary is the *phased* recommendation, which is often
the right answer: start on D or C, learn what workloads look
like at your scale, then migrate to B or A when the workload
justifies the ownership.

## The recommendation paragraph

Every build-vs-buy memo ends with one paragraph, structured:

> Recommendation: [Option X, or a specific hybrid]. Rationale
> in one sentence per binding axis (typically cost, capability
> ceiling, lock-in). Explicit re-review trigger: [a date, a
> scale threshold, or a market event that re-opens the
> question]. Sign-off required from: [named individuals].

Two properties of this paragraph:

- **The re-review trigger is named.** Build-vs-buy is not a
  one-time decision. If pricing moves, if a new provider
  enters, if the workload shape changes materially, the memo
  is re-opened. State the trigger; do not just say "we should
  revisit annually".
- **The alternative that came closest is named.** "We chose A;
  option C came closest and would have won at [specific
  threshold]." This tells the next author of this memo where
  the sensitive axis is.

## Worked example — the memo skeleton

```
# Build-vs-Buy Memo: Foundation Training Platform

Author:       (name)
Sponsor:      Platform Lead
Date:         2026-08-15
Version:      v1
Re-review by: 2027-Q3, or on any of the trigger conditions
              in §7.

## 1. Executive summary (1 paragraph, ~150 words)
Recommend Option B (NeMo integration on owned cluster) for the
2026-Q4–2028-Q2 window. Continue Option D (Together) for
research spike capacity above 128 H100s during the same window.

## 2. The four options (1 paragraph each)
(A / B / C / D descriptions, per the framing above)

## 3. Decision matrix (the seven-axis table)
Filled in with our-cluster numbers.

## 4. Two-year TCO comparison (table)
Capex + opex + labor + risk-adjusted overhead, 24 months,
holding recipe / MFU / wall-clock constant across options.

## 5. Capability ceiling for the roadmap
Our 2027-Q2 planned largest run: 300 B parameters, 6 T tokens.
Options A and B support; options C and D not confirmed.

## 6. Lock-in analysis
Data-plane, format, API — sub-scored per option.

## 7. Recommendation and re-review triggers
(recommendation paragraph as above)

## 8. Alternative that came closest and why we did not pick it
Option A. Chosen against because our team is 4 platform ICs
and the fully-owned recipe layer is not staffable in the
window. If we grow to 8+ platform ICs, re-open.

## 9. Sign-off
(named individuals; dates)
```

Length: 4–6 pages. Longer is not richer; longer is not read.

## Common failure modes

- **The "build" case ignores labor.** In-house platform is
  cheap "if you don't count the platform team". Count them,
  loaded, over two years.
- **The "buy" case ignores exit cost.** The provider is
  cheaper this year; what does re-integration cost if next
  year's pricing changes? Bound it.
- **The "buy" case cites TCO without holding recipe constant.**
  A provider's per-GPU-hour rate is not comparable to
  in-house's if the provider's `μ_sus` on your recipe is
  unmeasured. Pilot both on the same recipe or the numbers
  are not comparable.
- **Recommendation without a re-review trigger.** The memo
  becomes stale; two years later someone re-writes it from
  scratch and reaches the opposite conclusion for the same
  reason.
- **Recommendation without a sign-off list.** Whose decision is
  this? Without names, it is aspirational. With names, it is
  committed.
- **Ignoring the phased hybrid.** The two-option answer (all-A
  or all-D) is rarely optimal. A common pattern is A/B for the
  baseline workload, D for spike or research; the memo names
  it explicitly.
- **Treating the provider's SLA as risk-free.** Read the SLA
  credit language. A Sev-1 outage of 8 hours during a
  frontier run costs orders of magnitude more than the credit.
  Do not treat SLAs as insurance.

## Summary

- Four canonical options: build on Megatron (A), integrate NeMo
  or torchtitan (B), Databricks/Mosaic-hosted (C), Together or
  peer specialist (D). Every proposal maps to one or a hybrid.
- Seven-axis decision matrix (TCO, time-to-first-run,
  headcount, capability ceiling, lock-in, roadmap control,
  auditability) is the invariant; every memo fills the same
  table.
- Two-year TCO uses a fixed decomposition (capex + opex +
  labor + risk-adjusted overhead) with recipe / `μ_sus` /
  wall-clock held constant across options. Otherwise the
  numbers are not comparable.
- The recommendation paragraph names the option, the binding
  axes, the re-review trigger (date, scale, market event), and
  the alternative that came closest. Sign-off is by named
  individuals, not "leadership".
- Phased hybrids are common and often optimal — for example,
  in-house baseline plus hosted spike. The memo names them
  explicitly rather than treating build-vs-buy as binary.
- Re-run annually; every year the numbers move. A five-year-old
  memo is not evidence, it is archaeology.
