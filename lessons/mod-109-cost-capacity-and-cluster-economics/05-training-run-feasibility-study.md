# The Training-Run Feasibility Study

This chapter is the deliverable this module trains you to produce: a
single document that walks from a product target ("we want a 13B model
at Llama-2-13B eval quality by end of Q3") to a defended dollar and
wall-clock plan on a specific cluster. It is the artefact you bring to
the go / no-go conversation with your VP, and — with the numbers in
place — it is what lets the fine-tuning team, the platform team, and the
capacity team agree on what is being built.

Every training-pipeline engineer at this altitude writes feasibility
studies constantly. Some are ten pages; some are two sentences in a
Slack thread. The template below is the ten-page version. Once you can
write it long, the short version is the executive summary.

Exercise 05 has you write one; lab 01 upgrades it to a full doc.

## The template, at a glance

Every feasibility study answers eight questions in order:

1. **Product target.** What are we building and to what quality bar?
2. **Compute budget.** How many FLOPs (`C`) does that imply?
3. **Cluster shapes.** What three cluster options are on the table?
4. **Framework + MFU assumption.** What is our `F_eff` per GPU?
5. **GPU-hours × $/hr.** What does each shape cost?
6. **Buying mix.** Reserved vs. spot vs. on-prem for each shape.
7. **Risk register.** What can go wrong, priced.
8. **Recommendation.** With a numbered list of the assumptions we
   depended on.

The rest of this chapter is that template with a filled-out example.

## Section 1 — Product target

State the product target in three lines: the model, the quality bar, and
the deadline. Everything else is downstream.

**Fill-out fields.**

- Model family and parameter count `N`.
- Quality target — a specific number on a specific eval, ideally
  benchmarked against a public model.
- Wall-clock deadline: date by which the run must be complete.
- Downstream constraints: inference budget, context length, licensing.

**Worked example.**

- Model: 13B decoder-only transformer (Llama-2 architecture).
- Quality target: `MMLU 5-shot ≥ 55` on the base model (Llama-2 13B is
  at 54.8 in Touvron et al., 2023; we want parity or better).
- Deadline: 8 weeks from run start.
- Downstream: post-training fine-tune under 32 GPUs; inference at
  8k-token context on a single 8×H100 node.

## Section 2 — Compute budget

Use chapter 1's arithmetic. Set `D` from a Chinchilla-optimal or
overtrained anchor; compute `C = 6 · N · D`; sanity-check against
published anchors.

**Fill-out fields.**

- Chosen tokens-per-parameter ratio (Chinchilla `~20` or overtrained
  `50–200+`) with justification.
- `D` and `C`.
- Assumptions on data availability (is 300 B tokens of quality data
  even available in your domain?).

**Worked example.** A 13B / 300 B-token fine-tune (light overtraining,
~23 tokens/param, close to Chinchilla).

- `N = 1.3 × 10^10`.
- `D = 3.0 × 10^11` (300 B tokens).
- `C = 6 · N · D = 2.34 × 10^22 FLOPs`.
- Anchor: Llama-2 13B saw 2 T tokens (~154 tokens/param). We are far
  below that on purpose because this is a fine-tune from a strong base,
  not a from-scratch pretraining. Justification: transfer-from-base
  makes marginal loss reduction per training token much smaller past
  ~300 B tokens on top of a well-pretrained base.

## Section 3 — Candidate cluster shapes

Sketch three concrete options — small, medium, large — with a clear
wall-clock and dollar estimate for each. Do not carry more than three
into the executive-summary section; if you have five candidates, kill
two before the meeting.

**Fill-out fields.**

- Cluster shape (GPU type, count `G`, fabric).
- Wall-clock at your MFU assumption.
- GPU-hours.
- Dollar estimate at a stated buying mix.
- Availability: can we actually book this shape in the deadline
  window?

**Worked example.**

Cluster options for the 13B / 300 B-token run:

| Shape          | GPU     | G   | Fabric   | MFU  | `T_wall` | GPU-hours | Dollars @ mix          |
|----------------|---------|-----|----------|------|----------|-----------|------------------------|
| Small (dev)    | H100 SXM | 32  | 1 SU IB  | 55%  | 15.4 d   | 11,850    | ~$29 k (reserved $2.50)|
| Medium         | H100 SXM | 128 | 1 SU IB  | 52%  | 4.1 d    | 12,540    | ~$31 k (reserved $2.50)|
| Large          | H100 SXM | 512 | 2 SU IB  | 48%  | 1.1 d    | 13,570    | ~$34 k (reserved $2.50)|

<!-- needs-research: verify current H100 reserved and on-demand hourly
across AWS EC2 P5, GCE A3, Azure ND H100 v5, and CoreWeave. $2.50 / hr
is a teaching order-of-magnitude, not a quote -->

The pattern: **GPU-hours grow slightly with `G` because MFU sags**;
wall-clock shrinks near-linearly. The dollar delta between shapes is
small compared to the wall-clock delta — the deadline drives the
decision, not the dollar figure.

## Section 4 — Framework, MFU, and reliability posture

State the framework and the MFU assumption *before* you defend the
dollar figures. Both are load-bearing.

**Fill-out fields.**

- Training framework and parallelism strategy (mod-101 chapter 4;
  mod-102 chapters 3–5).
- MFU baseline (measured on a previous run; or, if none, the
  conservative 40–45% for a new team).
- MFU stretch target (what mod-107 work would bring in).
- Reliability posture: checkpoint interval, expected failure rate,
  availability budget (mod-106 chapter 8).
- Data-pipeline posture: how the loader is fed, throughput budget
  (mod-103 chapter 4).

**Worked example.** For the 13B / 300B fine-tune:

- Framework: PyTorch FSDP2 with DeviceMesh, HSDP across nodes for the
  128 / 512 shapes. torchtitan-shaped code.
- MFU baseline: 52% (measured on our previous 7B run on the same
  fabric).
- MFU stretch: 55% with FA3 already integrated, +2 points if we
  land FP8 on QKV projections in time; treat FP8 as a stretch, not a
  dependency.
- Reliability posture: async DCP checkpoint every 30 minutes; target
  90% goodput; expected 1 host-level failure per 24 hours on the
  512-GPU shape (empirical from our previous quarter).
- Data pipeline: MosaicML Streaming from S3 through a WEKA staging
  tier (mod-103 chapter 4; mod-105 chapter 7). Loader throughput
  budget: 4 GB/s per node, well above the ~1.5 GB/s the training loop
  demands.

## Section 5 — Buying mix

For each cluster shape you shortlisted, spell out the reservation
split. Chapter 3 is the reference; here it is a line item.

**Fill-out fields.**

- Reserved fraction (baseload).
- On-demand fraction (burst).
- Spot fraction (opportunistic; requires mod-106 elasticity to be
  real).
- Dedicated vs. shared: are we carving reserved capacity into a
  dedicated gang, or running under mod-104 multi-tenant policy?
- Sensitivity: if reserved is not available in the deadline window,
  what is Plan B?

**Worked example.** Medium (128-GPU) recommended shape:

- Reserved: 128 GPUs of an existing 1-year reservation on the shared
  cluster (mod-104 quota carves ~64 as guarantee, ~128 as cap).
- On-demand: 0 (do not need burst; wall-clock deadline is comfortable).
- Spot: 0 (mod-106 elasticity is not fully proven; not worth the
  incident risk for a 4-day run).
- Dedicated vs. shared: dedicated for the duration via mod-104
  reservation window; reverts to shared quota after.
- Plan B: if the 128-GPU reservation cannot be honoured (e.g. a
  higher-priority pretraining pre-empts), fall back to the 64-GPU
  option at 8 days wall-clock. Still fits in the deadline.

## Section 6 — Risk register

The section that separates a napkin estimate from a defensible
feasibility study. List every risk, its likelihood, its impact in
GPU-hours or dollars, and the mitigation.

**Fill-out fields (categories to cover explicitly).**

- **Fabric health.** NCCL timeout, silent NIC drop, PXN misconfiguration
  (mod-105 chapter 8). Priced at `Δ wall-clock × $/GPU-hr`.
- **Host failure.** GPU OOM, ECC error, host reboot. Priced from a
  published failure rate (Llama 3 or OPT-175B logbook anchor).
- **Silent corruption.** Wrong gradient, NaN, divergent loss (mod-108
  chapter 5's run-time signature catalog). Priced as "restart from
  checkpoint N steps back".
- **Capacity risk.** Reserved capacity not honoured. Priced against
  Plan B.
- **MFU regression.** New framework version drops MFU. Priced as
  chapter 4's regression cost.
- **Data-pipeline failure.** Shard corruption, tokenizer drift
  (mod-103 chapter 6). Priced as restart cost.
- **Reproducibility failure.** Seed / config drift blocks re-run
  (mod-108 chapter 3). Priced as re-run cost.

**Worked example.** Risk register for the 128-GPU 4-day run:

| Risk                        | Likelihood | Impact (GPU-hrs) | Impact ($) | Mitigation                             |
|-----------------------------|------------|------------------|------------|----------------------------------------|
| 1× host failure             | 60%        | ~1 h × 128 = 128 | ~$320      | DCP resume; torchrun elastic (mod-106) |
| Fabric-level NCCL timeout   | 20%        | ~4 h × 128 = 512 | ~$1,280    | mod-105 runbook; restart from ckpt     |
| Silent MFU regression       | 15%        | ~10% × 12,540    | ~$3,140    | mod-108 alert; roll back framework     |
| Data shard corruption       | 10%        | ~4 h × 128 = 512 | ~$1,280    | mod-103 checksum on load; skip shard   |
| Reserved capacity denial    | 25%        | 0 (fall to 64-GPU) | ~$0 net  | Plan B: 64-GPU 8-day run               |
| Divergent loss / retrain    | 10%        | ~30% × 12,540    | ~$9,400    | Small-scale ladder; abort early        |

Total *expected* risk cost (probability-weighted): ~$3,500 over the
$31 k base — a ~11% risk load. Budget accordingly; do not treat the
base number as the final answer.

## Section 7 — Recommendation

The one-page bit. Assumes the reader read sections 1–6.

**Fill-out fields.**

- Recommended cluster shape, buying mix, wall-clock, and dollar
  estimate.
- The `k` critical assumptions the recommendation depends on.
- Explicit go / no-go criteria.

**Worked example.**

> **Recommendation.** Run the 13B / 300B fine-tune on the 128-GPU H100
> shape under the existing shared-cluster reservation for 4 days
> wall-clock, at an all-in expected cost of **~$31 k + ~$3.5 k risk
> reserve = ~$35 k**. Target completion: 6 weeks from start (2 weeks
> preparation + 4 days training + buffer for restart-from-checkpoint
> events).
>
> **Assumptions.**
>
> 1. Chinchilla-adjacent 300 B tokens is sufficient to reach `MMLU
>    5-shot ≥ 55` on top of the Llama-2 13B base. Justification: our
>    7B ablation on the same corpus tracked Llama-2 7B eval within 2
>    points; the 13B run should transfer.
> 2. MFU holds at 52% ± 3 points on the 128-GPU shape. mod-108
>    dashboard is monitoring; regression alerts on 3+ point drop.
> 3. Reserved capacity of 128 H100s is honoured within the deadline
>    window. Plan B (64-GPU / 8-day) is priced and viable.
> 4. mod-106 checkpointing story is real; async DCP interval at 30
>    minutes, resume tested against a synthetic node failure in the
>    last quarter's drill.
> 5. Data pipeline throughput is 4 GB/s per node with headroom.
>    Verified against mod-103 loader benchmark.
> 6. `$/GPU-hr` reserved holds at ~$2.50 for the H100 shape.
>    Sensitivity: at $3/hr the run is $37 k; at $2/hr the run is $25 k.
>
> **Go / no-go criteria.**
>
> - Small-scale (32-GPU, 20 B token) ladder run completes with
>   monotone loss and MFU ≥ 45%. If either fails, abort and
>   re-evaluate.
> - Reserved capacity confirmed at least 2 weeks before start.
> - Fine-tune corpus dedup + tokenizer hash frozen (mod-103) at
>   least 1 week before start.

Every one of those assumptions is a link in the chain. Make them
explicit so that when one breaks, the review can be pointed and
fast.

## Anti-patterns

Common failure modes in feasibility studies you should catch and
reject:

- **Single-shape presentation.** Only one cluster option offered.
  Always show three; the trade-off is the point.
- **No MFU assumption.** "It will run for 4 days at $30 k" without
  stating the MFU is unfalsifiable. Every number must be defensible
  against a specific assumption.
- **No risk register.** "It will cost $30 k" is not a defensible
  answer; "it will cost $30 k ± $10 k with these risks priced" is.
- **Rounded-to-zero risk.** A 20% chance of a $10 k restart is
  $2 k of expected cost; do not omit it.
- **No plan B.** Reserved capacity can be denied; frameworks can
  regress; a host can catch fire. Every plan has a Plan B.
- **Confusing wall-clock with dollars.** They move together only if
  `G` is fixed. A larger cluster is roughly the same GPU-hours (at
  constant MFU) but shorter wall-clock at higher $/hr per unit-time.
  Present both.
- **Optimism about MFU stretch targets.** Do not budget on the
  stretch. Budget on the baseline; treat the stretch as upside.
- **Fine print in the middle of the doc.** Buying mix and capacity
  assumptions belong in the executive summary, not in section 6.

## From feasibility to run

Once the feasibility study is approved, it turns into a set of
concrete artefacts:

- The compute budget in `C` feeds into the run's SLO tracking (mod-108).
- The cluster shape and buying mix feed into mod-104's quota
  allocation and mod-106's reliability posture.
- The MFU assumption is the target the mod-107 team owns.
- The data-pipeline throughput budget is the target the mod-103 team
  owns.
- The risk register is the incident-classification framework mod-106
  chapter 6 uses to classify events during the run.

The feasibility study is the contract between the platform, the
researcher, and the capacity planner. When the run actually goes,
every one of the numbers in the study becomes a check against reality
— that check is what the run report is.

## The short-form version

Not every conversation warrants the ten-page version. When someone asks
in Slack "how much will a 7B pretraining cost?", the short form is:

> "About 12 k GPU-hours at Chinchilla-optimal 200B tokens, ~$25–35 k
> reserved on H100, ~9 hours on 512 GPUs at 50% MFU. Doubles or triples
> if we overtrain past Chinchilla. Full study if we want to defend a
> plan; happy to write one."

That is the answer that ends the conversation without over-promising.
Every number in it comes from chapters 1 and 2. If you can produce that
one-paragraph answer for arbitrary model shapes without reaching for a
calculator, this module has done its job.

## Summary

- The feasibility study is the module's deliverable. It answers eight
  questions in order: product target, compute budget, cluster shapes,
  framework + MFU, GPU-hours × $/hr, buying mix, risk register,
  recommendation-with-assumptions.
- Always present at least three cluster shapes (small / medium /
  large). The trade-off between wall-clock and dollars is the point.
- Every number is defensible against a specific assumption; every
  assumption is numbered so that when one breaks, the review can
  point to it directly.
- The risk register is not optional. Price fabric health, host
  failure, silent corruption, capacity risk, MFU regression, and
  data-pipeline failure explicitly. Expect the probability-weighted
  risk load to be 5–20% of the base dollar figure.
- Exercise 05 has you write a feasibility study for a stated product
  target; lab 01 turns it into a full 5-page doc with real vendor
  quotes. This chapter's template is the scaffold for both.
