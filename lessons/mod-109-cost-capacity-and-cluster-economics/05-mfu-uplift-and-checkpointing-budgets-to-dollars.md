# MFU Uplift and Checkpointing Budgets to Dollars

Chapters 1–4 give you the machinery to price a training run. This
chapter is the *reverse* machinery: given a proposed optimisation
— a kernel swap that lifts MFU by 3 points, a checkpoint cadence
change that costs 2% of goodput, a straggler-detection policy
that averts a 6-hour restart per week — what is its dollar
value at the current run scale?

This is the chapter that closes the loop with mod-107 (throughput
and MFU) and mod-106 (checkpointing, fault tolerance, elastic
training). Every optimisation those modules propose has a dollar
number attached; this is how you compute it.

The framing to keep: the training-pipeline engineer's job is to
translate a technical proposal ("switch to FlashAttention v3")
into a business proposal ("save $42 K on this run, plus $180 K
on the next-generation model"). Without this translation, the
throughput and reliability work is invisible to the org above
the platform team.

## The dollars-per-MFU-point identity

Recall from chapter 2:

```
GPU_hours = C / (P · μ_sus · 3600)
```

`C`, `P`, and `3600` are constants of the run. Differentiate
GPU-hours with respect to `μ_sus`:

```
d(GPU_hours) / d(μ_sus) = -C / (P · μ_sus² · 3600)
                        = -GPU_hours / μ_sus
```

So a small change `Δμ_sus` in sustained MFU produces a
proportional change in GPU-hours:

```
ΔGPU_hours ≈ -GPU_hours · (Δμ_sus / μ_sus)
```

In dollars:

```
Δ$_run ≈ -$_run · (Δμ_sus / μ_sus)
```

**The rule of thumb**: at a starting MFU of 40%, each additional
percentage point of MFU is worth roughly 2.5% of the run's total
cost. At 20% starting MFU, each point is worth ~5% of the run's
cost. The cheaper you already are, the less each further point
is worth in absolute terms — but in relative terms, MFU points
compound.

### Numeric example

Continuing the 34 B Chinchilla-optimal example (`GPU_hours ≈
97 600`, reserved rate ≈ `$3/hr` → `$_run ≈ $293 K`), a 3-point
MFU uplift from 40% to 43% is worth:

```
ΔGPU_hours = 97 600 · (0.03 / 0.40) = 7 320 GPU-hours
Δ$_run    = 7 320 · $3.00 ≈ $22 000
```

The same uplift on the inference-optimal recipe (`GPU_hours ≈
683 000`, `$_run ≈ $2.05 M`) is worth `~$154 K`. Same three
points of MFU, seven times the dollar impact — because the run
is seven times larger.

At **frontier scale** (Llama 3 70B, 39.3 M H100-hours per Meta's
paper, at a similar per-hour rate), three points of MFU is worth
tens of millions of dollars. This is why upstream kernel work
(FlashAttention, Transformer Engine, torch.compile) has an ROI
that justifies a dedicated performance-engineering role at every
frontier lab.

## The goodput-to-dollars identity

Goodput lives inside `μ_sus = μ_nom · goodput`. So a change in
goodput has exactly the same dollar-multiplier form as a
change in MFU:

```
Δ$_run ≈ -$_run · (Δgoodput / goodput)
```

At 90% goodput, each additional goodput point is worth roughly
1.1% of the run's total cost. At 60% goodput (a preemption-
heavy spot run), each point is worth 1.7%. Recovering from
60% to 80% via mod-106's elastic-training work is worth ~25%
of the run's cost — nine figures in absolute terms at
frontier scale.

This is also how you set the **goodput SLO** (mod-106
chapter 8) in dollars. A team target of "0.90 goodput" is
worth publishing as "each point below the target costs ~$3 K
on this run and ~$300 K at the next-generation scale". The
dollar framing is what makes the SLO stick with product
leadership.

## The checkpointing cadence trade-off, in dollars

Checkpointing (mod-106 chapters 1–3) costs goodput in two ways:

- **Write cost**: every checkpoint burns some seconds of the
  step boundary. Async DCP (mod-106 chapter 3) can hide most of
  it; naïve synchronous save exposes all of it.
- **Retention cost**: every retained checkpoint occupies object
  storage until deleted.

Checkpointing also *saves* goodput by shortening the "wasted
work" window on a restart. The full accounting:

```
goodput = 1
         − (write_time · N_writes / T_run)          # write overhead
         − (restart_time_avg · N_restarts / T_run)  # restart cost
         − (…)                                     # other overhead
```

Where the restart cost decomposes as (mod-106 chapter 8):

```
restart_time_avg =
    time_to_detect
  + time_to_reallocate
  + time_to_reload_checkpoint
  + wasted_work_since_last_checkpoint
```

The last term is the one your cadence choice controls: shorter
cadence → less wasted work per restart → higher goodput.
Longer cadence → less write overhead → higher goodput. There
is an optimum.

### The optimum-cadence formula (rough)

Let:

- `t_write` = seconds per checkpoint write (on the critical
  path; use `0` if fully asynchronous and always hidden).
- `t_cadence` = interval between checkpoint writes (the knob).
- `t_restart_fixed` = detection + realloc + reload, per
  restart.
- `λ` = restart rate (restarts per second of wall-clock).

Write overhead per second of wall-clock: `t_write / t_cadence`.
Restart overhead per second of wall-clock:
`λ · (t_restart_fixed + t_cadence / 2)` (expected wasted work
is half the cadence, by uniform-arrival assumption).

Minimise the sum by differentiating with respect to
`t_cadence`:

```
d/dt (t_write / t_cadence + λ · t_cadence / 2) = 0
⇒ -t_write / t_cadence² + λ / 2 = 0
⇒ t_cadence_opt = sqrt(2 · t_write / λ)
```

This is Young's classical checkpoint-optimisation formula
(Young, 1974). It generalises: as `t_write` shrinks toward
zero (async DCP), the optimal cadence shrinks toward zero (keep
checkpointing continuously); as `λ` shrinks toward zero (a
stable cluster), the optimal cadence grows without bound (stop
checkpointing).

### Worked example

Continue the 34 B run on 512 H100s.

- `t_write = 30 s` (synchronous DCP write of a 60 GB shard set;
  chapter 2 of mod-106 for the sizing).
- `t_restart_fixed = 600 s` (10 minutes for detect + realloc +
  reload on a well-run cluster; mod-106 chapter 8 anchors).
- Cluster restart rate `λ`: assume 2 restarts per week average
  on a well-run reserved cluster. `λ ≈ 2 / (7 · 86400) ≈
  3.3 · 10⁻⁶ /s`.

Optimal cadence:

```
t_cadence_opt = sqrt(2 · 30 / 3.3e-6) ≈ sqrt(1.8e7) ≈ 4 250 s ≈ 71 minutes
```

Roughly every hour. At this cadence, the checkpoint-related
goodput loss is (write overhead + expected wasted work):

```
write_overhead = 30 / 4250 ≈ 0.7%
restart_overhead = 2 · (600 + 4250/2) / (7 · 86400) ≈ 0.09%
total ≈ 0.8%
```

Translated to dollars on the $293 K run: `0.8% · $293 K ≈
$2 300`. If cadence were pushed to 15 minutes (over-
checkpointing), write overhead alone becomes `30 / 900 = 3.3%`,
or ~$9 700 in cost. If cadence were pushed to 12 hours (under-
checkpointing), the average wasted work per restart is 6 hours
of a 512-GPU run — a single restart alone costs `600 · 512 · $3
/ 3600 ≈ $256` of restart wall-clock plus `21600 · 512 · $3 /
3600 · 0.90 = $8 300` of wasted work. Two of those a week over
a month-long run is roughly $67 K — 23% of the run cost, gone
to bad cadence.

The takeaway: **cadence is a $10-20 K knob on a $300 K run, or
a $M knob on a $30 M run.** Chapter 6 does the arithmetic at
frontier scale where the numbers are much larger.

## Async DCP as a goodput lever

If `t_write` drops from 30 s (sync) to effectively 0 (fully async,
overlapped with the next step), the optimal cadence drops toward
zero and the write-overhead term vanishes. The remaining
optimisation is just on the restart-cost side.

The concrete recommendation: **treat async DCP (mod-106
chapter 3) as a mandatory dependency of the cost model**. A
synchronous-write checkpoint budget wastes 1–5% of run cost on
its own; the async-write budget pushes that toward 0.

## Straggler / SDC detection as goodput

Mod-106 chapter 6 covers straggler and silent-data-corruption
detectors. Every detected straggler is a hang the cluster does
*not* spend 6 hours in; every detected SDC is a re-run the
team does *not* have to schedule.

The dollar value of a straggler-detection policy is the
`expected_restarts_averted · restart_cost_dollars`. On a
frontier run at 16K GPUs, restarting from a straggler-induced
hang costs (per mod-106 chapter 8 anchors) hours to days of
wall-clock. In dollars at $3/hr, a single averted 6-hour
straggler on a 16 K-GPU cluster is:

```
$_averted = 6 · 16 000 · $3 = $288 000
```

A single incident. The straggler-detection policy pays for
itself if it averts one such incident per run. Publish this
in the feasibility study.

## The inference-lifetime dollar

Chapter 1 introduced the compute-optimal vs. inference-optimal
distinction; this section is the dollar arithmetic that decides
between the two recipes.

For a model with `N` active parameters served for `T` total
user tokens over its lifetime, the inference compute is roughly
`2 · N · T` FLOPs. Convert to dollars via the inference cluster
shape's per-hour rate and MFU:

```
$_inference_lifetime =
    (2 · N · T) / (P_infer · μ_infer · 3600) · $_per_infer_GPU_hour
```

Note `P_infer` and `μ_infer` can differ from training —
inference runs on different hardware (or the same hardware in
different mode), often with different precision (FP8, INT4)
and lower MFU (batch sizes are smaller). The `2` factor
replaces training's `6` because inference is forward-only.

Compare `$_inference_lifetime` to `$_training_run` for both the
compute-optimal and inference-optimal recipes. The optimal
choice is the one that minimises the *sum*.

### The tipping point

Rearranging Chinchilla's "20 tokens per parameter" against the
inference-lifetime cost gives (approximately) a break-even for
`D / N`:

```
D / N ≈ (T / D) · (μ_train / μ_infer) · ($_train / $_infer) · (2/6)
```

The dominant factor is `T / D` — the ratio of inference tokens
to training tokens. If a model is going to be served for 10× its
training-token count over its lifetime, the inference-optimal
`D / N` ratio is well above Chinchilla.

For the 34 B chat model from chapter 1 (50 M user tokens per
day for 12 months → `T ≈ 1.8 · 10^10`), `T / D` is *tiny* — well
below 1. Chinchilla-optimal wins. For a widely-deployed model
that serves 10 B tokens per day for 3 years → `T ≈ 1.1 · 10^13`,
`T / D` is ~16, and inference-optimal wins.

The feasibility study should always compute both totals and
name the winner.

## Where mod-107 and mod-106 optimisations pay off

Bringing the module together:

- **mod-107 chapter 2 (FlashAttention v2/v3)**: primary lever
  on `μ_nom`. Typical uplift of 10–30 percentage points from a
  no-FA baseline; a few points from FA v2 to v3 on Hopper. On a
  $293 K run, worth $25–75 K; on a frontier run, worth tens of
  millions.
- **mod-107 chapter 3 (BF16 policy)**: correctness-preserving
  lever on `μ_nom`. Sets the baseline against which mod-107
  chapter 4 (FP8) can add another 20–30% on FP8-eligible ops.
- **mod-107 chapter 5 (activation checkpointing + packing)**:
  moves the memory–throughput frontier. Packing can lift MFU
  5–10 points at no correctness cost. Worth ~$15–30 K on a
  $293 K run.
- **mod-107 chapter 6 (`torch.compile`)**: kernel-fusion lever;
  2–8 points of MFU on top of a well-tuned baseline.
- **mod-107 chapter 7 (comm-compute overlap)**: primary lever on
  `η(G)` at large scale. Can be worth 15–25% of run cost at
  4K+ GPUs.
- **mod-106 chapter 3 (async DCP)**: removes the write overhead
  from the checkpoint budget. 1–5% of run cost freed.
- **mod-106 chapter 4 (elastic training)**: turns a spot-tier
  cost disadvantage into a cost advantage. 10–40% of run cost
  swing depending on tier.
- **mod-106 chapter 6 (straggler / SDC detection)**: catastrophic-
  event insurance. Value per averted incident is often larger
  than the whole per-optimization budget above.
- **mod-106 chapter 8 (goodput SLO)**: the umbrella metric that
  ties all of the above to a single business number.

None of these are theoretical. Every one has a public reference
recipe (torchtitan, DeepSpeed, Megatron) and a published anchor
for the uplift. The dollar arithmetic makes them fundable.

## The MFU roadmap as a business proposal

The pattern to internalise: an MFU roadmap is a *business
proposal*, not just a technical one. The format:

```
| Optimisation             | Expected Δμ_sus | Δ$_run | Δ$_next_gen | Cost to ship  |
|--------------------------|-----------------|--------|-------------|---------------|
| Switch to FA v3          | +4 pp           | $30K   | $2.8M       | 1 eng-week    |
| Async DCP                | +3 pp goodput   | $9K    | $850K       | 2 eng-weeks   |
| Enable torch.compile     | +3 pp           | $22K   | $2.1M       | 3 eng-weeks   |
| Elastic training on spot | +25 pp goodput  | $73K   | $6.5M       | 6 eng-weeks   |
| ...                      | ...             | ...    | ...         | ...           |
```

The `Δ$_next_gen` column is what makes the case for investment
that outlives the current run. Kernel work does not have a
2-week payback; it has a 2-year payback across every subsequent
run. Show both.

## Failure modes

- **Applying the linear approximation to large `Δμ`.** The `Δ$
  ≈ -$ · Δμ/μ` identity is accurate for small changes. For
  swings above ~20% (e.g., going from `μ = 0.20` to `μ = 0.50`),
  do the full arithmetic — the linear expansion undercounts.
- **Confusing MFU uplift with goodput uplift.** MFU is a nominal
  peak; goodput is the sustained fraction. The two multiply; a
  proposal that raises nominal MFU 4 points but costs 5 points
  of goodput is a net loss.
- **Not attaching a "next-run" number to the proposal.** A
  $30 K MFU proposal on a $300 K run is a 10% ROI; the same
  proposal on a $30 M run is a $3 M win. The next-run column is
  what turns a marginal proposal into a shipped one.
- **Ignoring the write cost of `t_write = 0` claims.** "Fully
  async" is often "async up to a saturation point" — a
  `10 GB/s` object store cannot swallow a `100 GB/s` async
  write firehose without back-pressure. Measure `t_write`
  before assuming it away.
- **Publishing the checkpoint-cadence formula's answer without
  the assumptions.** `λ` (restart rate) is workload- and
  cluster-specific. Cadence advice without a cited `λ` is
  advice from a different cluster.

## Summary

- Each additional MFU point is worth `$_run · (1/μ_sus)`
  percentage points of the run's total cost. At `μ = 0.40`,
  every point is worth ~2.5% of the run.
- Each additional goodput point is worth
  `$_run · (1/goodput)` percentage points. At goodput = 0.90,
  every point is ~1.1% of the run.
- Checkpoint cadence follows Young's formula:
  `t_cadence_opt = sqrt(2 · t_write / λ)`. Async DCP pushes
  `t_write` toward zero and the optimal cadence toward zero.
- Straggler / SDC detection is catastrophic-event insurance;
  the dollar value per averted incident often exceeds the
  whole per-optimisation budget.
- Compute-optimal (Chinchilla) vs. inference-optimal — the
  choice is set by `T / D` (inference tokens per training
  token). For heavy-serving products, inference-optimal wins
  and buys 3–10× the training compute.
- The MFU roadmap is a business proposal. Show `Δ$_this_run`
  *and* `Δ$_next_gen`; the second column is what gets kernel
  work funded.
- Chapter 6 turns this arithmetic into two anchor recipes —
  the MPT-7B cost-optimised one and the Llama 3 frontier one.
