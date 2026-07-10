# Anchor Recipes: MPT-7B and Llama 3

The scaling laws in chapter 1, the arithmetic in chapter 2, and the
feasibility template in chapter 5 all need calibration against actual
runs. This chapter walks two anchors from public reports — one that
optimised aggressively for dollars per point of eval quality, one that
optimised for the frontier — and reads their choices as feasibility
studies you can imitate.

The two anchors:

- **MPT-7B (MosaicML, May 2023).** A 7B decoder-only LM trained on
  ~1 T tokens for ~$200 k in ~9.5 days on ~440 A100-40GBs. Read as the
  "scaling down" recipe.
- **Llama 3 (Meta, 2024; Grattafiori et al., "The Llama 3 Herd of
  Models").** An 8B / 70B / 405B family trained on ~15 T tokens on
  thousands to tens of thousands of H100s. Read as the "scaling up"
  recipe.

Public information on these runs comes from different-quality sources.
MPT-7B's cost figures come from a MosaicML blog post; the exact numbers
are marked with `<!-- needs-research -->` because the primary source
should be re-checked before citing in your own docs. Llama 3's figures
come from the peer-reviewed / arXiv-posted technical paper and are
better anchored.

## MPT-7B: the "scaling down" recipe

### Product target as MosaicML defined it

Read the choices as if MosaicML had filled in chapter 5's template:

- Model: 7B decoder-only transformer, ALiBi position embeddings (no
  RoPE), FlashAttention v1, no bias terms.
- Data: ~1 T tokens across a curated mix of CommonCrawl, C4, Books,
  arXiv, StackExchange, Wikipedia, and code.
- Quality target: parity with LLaMA-1 7B (Touvron et al., 2023a) on
  standard eval suites, at a commercial-friendly license.
- Wall-clock: ~9.5 days.
- Cost target: publicly reported as ~$200 k.

<!-- needs-research: verify exact MPT-7B token count, cost, wall-clock,
and hardware from MosaicML's original blog post ("Introducing MPT-7B",
May 2023). The 1 T tokens, $200 k, 9.5 days, 440 A100-40GB figures are
the numbers most widely repeated; the primary source should be
re-checked before citing in your own doc -->

### Compute budget

Applying chapter 1's arithmetic to the reported numbers:

- `N = 7 × 10^9`.
- `D ≈ 1 × 10^12` (1 T tokens; ~143 tokens/param, overtrained ~7×
  above Chinchilla).
- `C = 6 · N · D ≈ 4.2 × 10^22 FLOPs`.

The overtraining is deliberate. MosaicML was optimising for a base
model that would be cheap at inference; the extra training FLOPs pay
back over deployment lifetime. This is the same overtraining logic
that Llama 2 (Touvron et al., 2023b) later scaled up.

### Cluster shape and MFU

The reported shape: **~440 A100-40GBs**, on Oracle Cloud
Infrastructure's bare-metal H100/A100 offering. A100 SXM BF16 peak is
~312 TFLOPS (NVIDIA A100 datasheet).

<!-- needs-research: verify MPT-7B hardware (A100-40GB SXM vs. A100-80GB
SXM) and cloud (Oracle vs. others) from MosaicML's blog -->

Applying chapter 2's arithmetic backwards from the reported wall-clock:

    T_wall_seconds ≈ 9.5 days ≈ 8.2 × 10^5 s
    F_cluster_effective = C / T_wall = 4.2e22 / 8.2e5 ≈ 5.1 × 10^16 FLOPs/s
    F_eff_per_gpu = 5.1e16 / 440 ≈ 1.16 × 10^14 FLOPs/s
    MFU ≈ F_eff_per_gpu / 3.12e14 ≈ 37%

That is a *reasonable* MFU for A100 BF16 on a well-tuned FSDP1 stack.
MosaicML's blog reports slightly higher figures on their tuned Composer
stack (upper 30s to low 40s MFU on that generation of hardware), which
tracks.

### Framework and engineering choices

Reading the choices for a feasibility-study lens:

- **FSDP1 (flat-parameter sharding)** was the state of the art at the
  time. MPT-7B was small enough that TP was unnecessary and pure DP
  would have run out of memory on 40GB A100s; FSDP was the right pick.
- **ALiBi over RoPE** — a deliberate choice for long-context
  extrapolation without re-training. Cheap FLOP-wise; a bet on
  usability.
- **FlashAttention v1** — the Dao et al. (2022) kernel; not the FA2
  that came out later. This is what got the attention math off the
  MFU floor.
- **No bias terms, no dropout in the transformer blocks** — small
  FLOP savings, cleaner numerics.
- **Aggressive dedup** in the training corpus. Read as a data-quality
  investment: MosaicML believed 1 T deduped tokens was worth
  substantially more than 1 T undeduped tokens.

### What the recipe optimised for

**Dollars per point of eval quality.** MosaicML's public thesis: a
7B model trained on 1 T deduped tokens with a battle-tested framework
was the best cost-per-quality point they could ship as a commercial
base model. They were not chasing the frontier — they were chasing a
Pareto-optimal point for downstream fine-tuning customers.

The choices this drove:

- Small (7B) not large. Cheap at inference; wide adoption possible.
- Overtrained past Chinchilla. Trade extra training FLOPs (one-time)
  for inference savings (recurring).
- A100 not H100 (H100 was hard to book in early 2023 in volume; A100
  was available). A100 was economically better *for what was
  available at the time*.
- FSDP1 on a mature stack, not the latest research code. Reliability
  posture over frontier features.

For your own feasibility studies, MPT-7B is the anchor for "we want a
small base model at a defensible cost". Chapter 5's 13B / 300 B
worked example is a direct cousin of this recipe: use it when your
target is downstream cost-per-quality, not eval-leaderboard position.

## Llama 3: the "scaling up" recipe

### Product target as Meta defined it

Grattafiori et al., 2024, "The Llama 3 Herd of Models", is the primary
source. The published training scope:

- Family: 8B, 70B, and 405B decoder-only transformers, all trained on
  the same corpus and tokenizer.
- Data: ~15 T tokens (versus Chinchilla's ~20 T for 405B), so the 8B
  and 70B are heavily overtrained (~1,900 and ~215 tokens/param
  respectively) and the 405B is close to Chinchilla-optimal.
- Quality: frontier-competitive as of 2024. Detailed eval numbers in
  the paper.
- Reliability: the paper reports specific failure rates on the
  16k-H100 cluster.

### Compute budget

For the 405B, taking the paper's headline numbers:

- `N = 4.05 × 10^11`.
- `D ≈ 1.5 × 10^13` (15 T tokens).
- `C = 6 · N · D ≈ 3.6 × 10^25 FLOPs`.

For the 70B:

- `N = 7 × 10^10`.
- `D ≈ 1.5 × 10^13` (same corpus).
- `C ≈ 6.3 × 10^24 FLOPs`.

For the 8B:

- `N = 8 × 10^9`.
- `D ≈ 1.5 × 10^13`.
- `C ≈ 7.2 × 10^23 FLOPs`.

All three at Chinchilla-plus-overtraining scale; the 8B in particular
is trained at ~1,900 tokens/param, a factor of ~100 above Chinchilla.
The bet: massive inference cost savings from the small model over the
deployment lifetime.

### Cluster shape and parallelism

The paper describes a **16,000 H100 cluster** for the 405B training
and 4-D parallelism composed as `TP × CP × PP × DP`:

- **TP (tensor parallel)** across NVLink inside the DGX box.
- **CP (context parallel)** for long-context training — a newer axis
  that shards the sequence dimension.
- **PP (pipeline parallel)** across the InfiniBand fabric.
- **DP (data parallel)** including FSDP-style sharding across
  replicas.

Read for feasibility: at frontier scale, you cannot escape composing
all four axes. mod-101 chapter 4 is where the design space lives; this
is the production shape.

### Precision, MFU, and reliability

- **BF16 baseline with FP8 through Transformer Engine.** The paper
  reports FP8 on selective ops (mainly QKV projections and MLP
  matmuls). BF16 elsewhere.
- **MFU / HFU numbers** are reported in the paper; the effective
  hardware FLOPs utilisation for the 405B run is in the range you
  would expect for a well-tuned dense LM at 16 k GPUs (roughly
  ~38–41% on H100 with FP8 mixed in).

<!-- needs-research: pull exact BF16 vs. FP8 MFU numbers from the Llama
3 paper section on training. The paper reports both MFU and HFU
(hardware FLOPs utilisation) which are related but not identical -->

- **Failure rates.** The paper reports thousands of interruptions over
  the pretraining window, dominated by hardware failures (GPUs,
  NVLink, IB) with a smaller tail of software / config causes. This
  is the empirical anchor for mod-106's availability budget.

<!-- needs-research: pull the exact number of interruptions and the
distribution across failure categories from the Llama 3 paper -->

### Framework and engineering choices

- **Custom-plus-open stack.** Meta's internal fork of PyTorch plus
  their own scheduler stack; the paper details the parallelism and
  reliability engineering but not the exact framework.
- **Distributed checkpointing** with async save (mod-106 chapter 3
  covers the shape).
- **Straggler detection and node quarantine** are explicit engineering
  concerns; the paper describes the operational protocol.
- **Data pipeline** at 15 T tokens is a serious engineering artefact in
  its own right; the paper describes the deduplication, quality
  filtering, and mixture-tuning at length.

### Cost estimate

The paper does not publish a dollar figure. Applying chapter 2's
arithmetic to the reported cluster and MFU gives an estimate.

For the 405B run:

- `C ≈ 3.6 × 10^25 FLOPs`.
- Cluster: 16,000 H100 SXM.
- MFU: ~40% (mixed BF16/FP8; frontier scale).
- `F_eff_per_gpu ≈ 989 × 0.40 ≈ 396 TFLOPS`.
- `F_cluster ≈ 16,000 × 3.96 × 10^14 ≈ 6.34 × 10^18 FLOPs/s`.
- `T_wall = C / F_cluster ≈ 5.68 × 10^6 s ≈ 66 days`.
- `GPU_hours ≈ 16,000 × 66 × 24 ≈ 25.3 × 10^6 GPU-hours`.

At **very** rough cloud reserved pricing of ~$2/GPU-hr for H100, that
is on the order of **$50 M** for the 405B run alone.

<!-- needs-research: this cost estimate is not from the paper; it is
back-of-envelope. Cross-check against SemiAnalysis or CoreWeave
whitepaper estimates for frontier-scale training costs -->

The 8B and 70B are much cheaper: the 8B at 7.2 × 10^23 FLOPs and the
same cluster / MFU is ~500 k GPU-hours, ~$1 M reserved.

### What the recipe optimised for

**Frontier eval quality plus a deployable 8B/70B family.** Meta was
building the frontier for the 405B and simultaneously producing 8B
and 70B checkpoints trained on the same high-quality corpus for the
Llama-family ecosystem. The overtraining on the 8B and 70B pays back
across every downstream deployment.

The choices this drove:

- 15 T tokens across a joint corpus rather than a size-specific
  corpus for each model. Amortises the data engineering across three
  model sizes.
- 4-D parallelism as a first-class engineering artefact. There is no
  scaling to 405B without it.
- FP8 through Transformer Engine on the ops where it works, BF16
  elsewhere. Precision as an engineering knob, not a religion.
- Detailed reliability engineering. At 16 k GPUs, the availability
  budget is the schedule.

For your own feasibility studies, Llama 3 is the anchor for "we are
trying to hit the frontier at cost". Use it as the reference when the
executive summary needs to defend a nine- or ten-figure training
budget.

## The two recipes side by side

| Dimension            | MPT-7B (2023)                    | Llama 3 405B (2024)                    |
|----------------------|----------------------------------|----------------------------------------|
| `N`                  | 7 × 10^9                         | 4.05 × 10^11                           |
| `D`                  | ~1 × 10^12                       | ~1.5 × 10^13                           |
| Tokens/param         | ~143 (overtrained ~7×)           | ~37 (~2× overtrained)                  |
| `C`                  | ~4.2 × 10^22                     | ~3.6 × 10^25 (~850× more)              |
| Hardware             | ~440 A100-40GB                   | ~16,000 H100 SXM                       |
| Precision            | BF16                             | BF16 + FP8 (Transformer Engine)        |
| Framework            | Composer + FSDP1 + FlashAttn v1  | Meta stack + 4-D parallel + TE-FP8     |
| Wall-clock           | ~9.5 days                        | ~2 months                              |
| MFU                  | ~37% (back-derived)              | ~38–41% (reported)                     |
| Cost (rough)         | ~$200 k (reported)               | ~$50 M (back-of-envelope)              |
| Optimised for        | $ per point of eval quality      | Frontier eval + deployable family      |

<!-- needs-research: verify every number in this table against primary
sources (MosaicML blog and Grattafiori et al., 2024) -->

Two patterns worth naming:

- **MFU is not that different.** Both runs sit in the 35–45% MFU band
  despite two years and one GPU generation apart. This is the "well-tuned
  dense-LM baseline" you should expect on your own runs.
- **Wall-clock scales sub-linearly with `C`.** The Llama 3 405B is ~850×
  more compute than MPT-7B but only ~7× longer wall-clock, because the
  cluster is ~36× larger. This is the linear-scaling assumption in
  chapter 2 holding up at frontier scale — with the MFU sag priced in.

## Reading the recipes as feasibility studies

For your own feasibility work, both recipes give you *anchor points* on
the cost-vs-quality curve. When you present three cluster shapes in
section 3 of the feasibility template, at least one of them should be
justified by an anchor:

- "Our small option is MPT-7B-shaped: A100 or H100 in the hundreds,
  cost target in the low six figures, MFU target in the high 30s to
  low 40s. Anchor: MosaicML MPT-7B, ~$200 k."
- "Our medium option is Llama-2-13B-shaped: H100 in the low
  thousands, cost target ~$1 M, MFU target 45–50%. Anchor: Llama 2
  paper, or scaled-down Llama 3 8B."
- "Our large option is Llama-3-70B-shaped: H100 in the several
  thousands, cost target in the tens of millions, MFU target ~40%.
  Anchor: Llama 3 70B, ~$3–5 M back-of-envelope."

<!-- needs-research: verify the Llama 3 70B and 8B dollar estimates
against SemiAnalysis or CoreWeave published analyses -->

The anchors do two things. First, they calibrate your MFU and cost
assumptions against real runs — hard to argue with. Second, they force
you to name what you are optimising for: are you MPT-7B-shaped (dollars
per quality) or Llama-3-shaped (frontier eval)? A feasibility study
that cannot answer that question is not ready for the go / no-go
meeting.

## What is missing from the public record

Neither report is a complete feasibility study, so treat both as
partial anchors:

- **MPT-7B** does not disclose the full run-report (per-step MFU
  timeline, failure log, restart count). The $200 k is a single
  number, not a distribution.
- **Llama 3** does disclose reliability data but not a dollar figure.
  You can back-derive the cost from the reported cluster and MFU, as
  above, but Meta's actual cost is buried in internal accounting and
  is not the same as the cloud-reserved back-of-envelope.

For a defensible feasibility study of your own, you will always know
more about your own run than either public report tells you. Use the
anchors for the *shape* of the answer; do not treat their numbers as
canonical for anything but the shape.

## Summary

- MPT-7B (MosaicML, 2023) is the "scaling down" anchor: 7B, ~1 T
  tokens, ~$200 k, ~9.5 days on ~440 A100-40GBs, MFU in the high 30s.
  Optimised for dollars per point of eval quality; overtrained past
  Chinchilla to minimise downstream inference cost. Primary source:
  MosaicML blog (numbers to be re-verified against the original
  post).
- Llama 3 (Meta, 2024, Grattafiori et al.) is the "scaling up"
  anchor: 405B trained on ~15 T tokens on ~16 k H100s over ~2 months,
  ~$50 M back-of-envelope. Uses 4-D parallelism (TP × CP × PP × DP),
  BF16 + FP8 through Transformer Engine, custom reliability
  engineering with detailed published failure rates.
- MFU is remarkably consistent across the two recipes (35–45%) despite
  the gap in scale, GPU generation, and precision. That band is a
  reasonable prior for a well-tuned dense LM feasibility study.
- The two anchors force the "what are we optimising for?" question:
  dollars per quality (MPT-7B) or frontier eval (Llama 3). A feasibility
  study that cannot pick one is not ready.
- Use both as calibrations for your own cluster shapes and cost
  estimates. Neither is a complete feasibility study; both are
  partial anchors.
- Every published cost figure in this chapter carries a
  `<!-- needs-research -->` marker because the primary sources should
  be re-checked before quoting the numbers in a doc that will land in
  procurement.
