# MFU Uplift as Dollars

Every point of MFU you gain on a training run is a dollar figure. This
chapter makes that figure explicit and gives you the arithmetic to walk
into a prioritisation meeting with a number the CFO understands.

mod-107 owns the *engineering* of MFU — FlashAttention v3, FP8 through
Transformer Engine, activation checkpointing, communication overlap,
`torch.compile` integration. This chapter is the bridge: it takes each
of those engineering artefacts and translates it into a dollar amount
against a specific cluster shape, so you can rank them against every
other engineering priority.

The framing you need on your team is: **MFU work is a revenue line, not
an efficiency line.** A percentage point of MFU on a large cluster is
worth tens of thousands to millions of dollars per run. That is the
altitude you defend the work at.

## The base formula

From chapter 2:

    GPU_hours = C / F_eff = C / (F_peak · MFU)
    Dollars = GPU_hours · $/GPU-hr

The dollar cost is inversely proportional to MFU. Going from MFU `m_0`
to MFU `m_1` saves:

    ΔDollars = Dollars_0 · (1 - m_0 / m_1)

Equivalently, since `GPU_hours = C / (F_peak · MFU)`:

    ΔGPU_hours = C / F_peak · (1/m_0 - 1/m_1)
               = (C / F_peak) · (m_1 - m_0) / (m_0 · m_1)

Two facts to internalise:

- **The savings are non-linear in MFU.** Going from 50% to 55% saves
  more dollars than going from 55% to 60%, because you are dividing by
  a smaller `m_0 · m_1`.
- **The savings scale linearly in both `C` and `$/GPU-hr`.** Bigger run,
  bigger savings. Higher-cost cluster, bigger savings.

## Worked example: 512 × H100, 3-week pretraining

The scenario. You have a 7B model, ~2 T tokens (overtrained past
Chinchilla, Llama-2 style). `C = 6 · 7e9 · 2e12 = 8.4 × 10^22 FLOPs`.
Cluster: 512 × H100 SXM. Cost per H100-hour: **$2** (a reserved-ish
teaching number — verify against current vendor pricing at procurement
time).

<!-- needs-research: verify current H100 reserved and on-demand hourly
at AWS EC2 P5, GCE A3, Azure ND H100 v5, and CoreWeave -->

Base case at 55% MFU:

    F_eff = 989 TFLOPS × 0.55 ≈ 544 TFLOPS = 5.44 × 10^14 FLOPs/s
    F_cluster = 512 × 5.44e14 ≈ 2.79 × 10^17 FLOPs/s
    T_wall = 8.4e22 / 2.79e17 ≈ 3.01e5 s ≈ 3.49 days
    GPU_hours = 512 × 3.49 × 24 ≈ 42,900
    Dollars = 42,900 × $2 = $85,800

If MFU sags to 50% (say, a regression from a framework upgrade), the
same run becomes:

    F_eff = 989 × 0.50 ≈ 495 TFLOPS
    T_wall = 8.4e22 / (512 × 4.95e14) ≈ 3.32e5 s ≈ 3.84 days
    GPU_hours = 512 × 3.84 × 24 ≈ 47,180
    Dollars = 47,180 × $2 = $94,360

The 5-point regression cost you **$8,560** on one run. If you run this
every quarter, it is $34 k/year. If the platform runs 10 such projects
per year across teams, it is $340 k/year. That is one engineer for a
few months — a very concrete "MFU uplift work is worth funding" case.

Going the other direction, moving to 60% MFU (say, via FA3 + FP8 +
better overlap; mod-107 chapters 4–6):

    F_eff = 989 × 0.60 ≈ 593 TFLOPS
    T_wall = 8.4e22 / (512 × 5.93e14) ≈ 2.77e5 s ≈ 3.20 days
    GPU_hours = 512 × 3.20 × 24 ≈ 39,300
    Dollars = 39,300 × $2 = $78,700

Savings vs. 55% baseline: **$7,100 per run**. Same math scaled: at 10
runs/year, ~$70 k. That is not enormous compared to the run cost, but
the same 5-point uplift on a 4096-GPU frontier run is ~$570 k saved —
firmly in "hire an MFU specialist" territory.

## A three-shape reference table

Here is the same 5-point uplift priced against three cluster shapes and
a three-week wall-clock at $2 / GPU-hr:

| Cluster        | GPUs | Base MFU | GPU-hrs at 55% | GPU-hrs at 60% | Savings                   |
|----------------|------|----------|----------------|----------------|---------------------------|
| Dev / small    | 8    | 55 → 60% | 2,772          | 2,541          | ~$460 saved               |
| Pretraining    | 512  | 55 → 60% | 177,408        | 162,624        | ~$29,570 saved            |
| Frontier       | 8192 | 55 → 60% | 2,838,528      | 2,601,984      | ~$473,090 saved           |

(Numbers assume the same 3-week wall-clock at the base MFU, so the
comparison is per-fixed-wall-clock rather than per-fixed-`C`. Redo the
math per-fixed-`C` — as in the previous section — for the more common
"same run, lower cost" framing.)

Two things this table teaches:

- **On dev clusters, MFU work does not pay for itself.** A 5-point
  uplift on 8 GPUs is under $1 k per run. Do MFU work on dev only to
  the extent it is a stepping stone to production.
- **On frontier clusters, a single engineer-week of MFU work is often
  seven figures.** This is why frontier labs employ dedicated
  performance teams.

## The FA3-plus-FP8 example

A common mod-107 lever: FlashAttention v3 with FP8 on H100 (Hopper's
Transformer Engine) typically buys 1.3–1.5× throughput vs. BF16-only
on attention-heavy models. Concretely, for a Llama-shape 70B pretraining
with attention at ~35% of compute, expect ~4–6 MFU points from FA3
alone and another 5–10 points from FP8 through Transformer Engine on
compatible layers.

<!-- needs-research: verify FA3 and TE-FP8 uplift numbers against
published benchmarks in the FA3 paper (Shah et al., 2024) and NVIDIA
Transformer Engine documentation -->

Priced against a 2048-GPU 7-day pretraining at $2/hr:

- Base at 42% MFU: `2048 × 168 × $2 = $688 k`.
- +8 points to 50% MFU (FA3 alone): saves `~$110 k`.
- +12 points to 54% MFU (FA3 + FP8): saves `~$153 k`.

Every dollar of that is dollars that would have gone to the cloud
vendor. The framing to your VP: "We can spend two engineer-weeks
integrating FA3 + FP8 and pay for it 5× on this quarter's run alone."

## The regression-cost side of the ledger

MFU is not just a target — it is a *reliability* metric. mod-108
chapter 4 owns the observability of MFU regressions; this chapter owns
their pricing.

Every framework upgrade, kernel patch, or configuration change is a
candidate for a silent MFU regression. A 3-point drop that goes
un-noticed for a quarter on a 512-GPU cluster is:

    Δcost = C × (1/0.52 − 1/0.55) / F_peak × $/GPU-hr
          ≈ C × 0.105 / F_peak × $2
          # per 8.4e22 FLOPs, at H100 989 TFLOPS peak
          # ≈ 8.4e22 × 0.105 / 9.89e14 × $2
          # ≈ $17,850 per run

Per-run. Per quarter, across a shared cluster, that is a five-figure
budget line no one is tracking. This is why mod-108's MFU dashboard is
not a nice-to-have — it is what protects the dollar figure.

The rule you want to sell your platform team: **MFU is a service-level
objective for the platform.** Report it per-run, alert on regressions,
attribute regressions to their cause (framework version, kernel
version, config change, fabric state), and hold owners accountable in
dollar terms.

## MFU uplift vs. cluster expansion: the choice framing

A useful framing for org-level prioritisation. You have a training
deadline; you can hit it two ways:

- **Cluster expansion.** Book more GPUs (bigger cluster or longer
  reservation) to finish the same `C` in less wall-clock at the same
  MFU. Cost: proportional to the extra GPU-hours × `$/GPU-hr`.
- **MFU uplift.** Invest engineer-time in MFU work; finish the same `C`
  in less wall-clock at higher MFU on the *same* cluster. Cost:
  engineer-months × loaded rate.

The break-even. Suppose an MFU uplift of `Δm` saves `S` dollars per run
and you run `k` times per year. Then MFU work is worth funding up to:

    max_engineer_cost = k · S / annual_discount_factor

For a 5-point uplift on the 512-GPU shape above (`S ≈ $7 k / run`,
`k = 4` / year), that is about $28 k / year of ongoing dollar savings —
enough to fund a portion of an engineer's time on it perpetually, and
far more than that as a one-off investment.

For a 5-point uplift on a 4096-GPU shape (`S ≈ $250 k / run`,
`k = 4` / year), that is $1 M / year. This is the number that hires
mod-107-specialist teams.

## The frontier-scale amplification

At frontier scale (thousands of H100s), MFU work has a further amplifier
that budgeting-time analyses miss: **higher MFU shortens wall-clock,
which shortens the exposure to hardware failure.**

The Llama 3 paper (Grattafiori et al., 2024) reports meaningful failure
rates on their 16k-GPU cluster; every extra day of wall-clock is
another few failures to absorb. Higher MFU → shorter wall-clock →
fewer restarts → less GPU-hour tax from mod-106's availability budget.

For the 4096-GPU shape, going from 40% to 45% MFU saves ~11% of
wall-clock, which at the published failure rates saves roughly the
same fraction of restart cost on top of the direct MFU savings. This
is why the ROI on MFU work grows *super-linearly* with cluster size.

<!-- needs-research: pull specific failure-rate numbers from the Llama
3 paper section on training reliability, and cross-check against
BLOOM (Le Scao et al., 2022) and OPT-175B logbook -->

## When *not* to invest in MFU

Two failure modes for the MFU-as-dollars framing:

- **Compute is not the bottleneck.** If your run is data-loader-bound
  (mod-103) or fabric-bound (mod-105) or storage-bound, higher matmul
  MFU does nothing; you have a plumbing problem, not a kernel problem.
  Diagnose before you optimise. mod-108's step-time decomposition is
  where you start.
- **The MFU number is wrong.** If you are measuring MFU with the wrong
  peak (BF16 vs. FP8), the wrong FLOP model (dense vs. MoE-activated),
  or with attention long-context not accounted for, your headline number
  will move without any real engineering. mod-107 chapter 1 tells you
  how to compute MFU correctly. Fix that first; then optimise.

For the same reason: **do not optimise MFU past ~60% without a strong
prior that the remaining gap is real.** Above 60% on a well-tuned FSDP2
dense LM stack you are usually chasing measurement noise, not real
gains.

## Decision template you can hand to your director

The one-page artefact that closes the loop from mod-107 engineering to
mod-109 economics. Fill this in whenever an MFU-related trade-off has
to be defended.

```
MFU Trade-off Decision Brief
============================

Change proposed: <FA3 integration / FP8 rollout / kernel patch / …>
Owner: <team>
Cluster shape at stake: <G GPUs of X>
Cost per GPU-hour (current buying mix): $Y
Runs per year at this shape: k
Baseline MFU (measured, mod-108 dashboard): m_0
Projected MFU after change (mod-107 estimate): m_1

Per-run savings:
  ΔGPU-hours = C · (1/m_0 - 1/m_1) / F_peak
  ΔDollars = ΔGPU-hours × $Y
Annual savings: k × ΔDollars

Investment required:
  Engineer-months: E
  Loaded cost per engineer-month: $Z
  Total: $E · Z

Break-even runs: (E · Z) / ΔDollars
Break-even years: (E · Z) / (k · ΔDollars)

Reliability caveat:
  - Framework compatibility risk: <low / medium / high>
  - Rollback plan: <named>
  - MFU regression alert: <mod-108 dashboard link>

Recommendation: <fund / defer / reject>
Assumptions:
  - Cluster shape holds for the next N quarters.
  - $/GPU-hr holds within ±20%.
  - MFU projection carries ±3 points of measurement noise.
```

This template plus the arithmetic in chapter 2 is what you bring to the
go / no-go conversation. Every MFU investment gets pushed through this
funnel; if the numbers do not clear break-even, the work does not get
scheduled.

## Summary

- Every MFU point is a dollar figure: `Dollars ∝ 1 / MFU`, so going from
  `m_0` to `m_1` saves `Dollars_0 · (1 − m_0/m_1)`. The savings are
  non-linear in MFU and scale linearly with `C` and `$/GPU-hr`.
- On a 512-GPU H100 cluster at ~$2/GPU-hr, a 5-point MFU uplift on a
  3-week run is order ~$7 k saved. On a 4096-GPU frontier cluster, the
  same uplift is ~$250 k+ per run. This is why frontier labs employ
  dedicated performance teams.
- MFU regressions are as real as MFU uplift. A 3-point silent
  regression on a 512-GPU quarterly run is ~$18 k per run of hidden
  cost. Treat MFU as a platform SLO; alert on it in mod-108.
- At frontier scale, higher MFU compounds: shorter wall-clock also
  reduces the mod-106 restart tax from hardware failure.
- Use the decision brief template above whenever an MFU trade-off has
  to be defended. Every engineering investment gets a break-even in
  runs or years; if it does not clear, it does not get funded.
- The chapter's core reframing: **MFU work is a revenue line, not an
  efficiency line.** Sell it that way to your VP, because it is true.
