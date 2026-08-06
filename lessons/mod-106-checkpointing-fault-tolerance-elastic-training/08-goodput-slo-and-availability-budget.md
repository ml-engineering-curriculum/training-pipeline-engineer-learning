# Goodput SLO and the Availability Budget

Everything in the previous seven chapters is a mechanism. This
chapter is the **objective** those mechanisms are optimized against:
a single number the platform team commits to and reports against
weekly, that captures how much of the cluster's compute actually
resulted in training progress.

The framing is the standard SRE service-level objective (SLO)
framing — a target on a service-level indicator (SLI), and an error
budget derived from it — adapted to training. If you are new to the
SRE framing, skim the primary reference before continuing:

- **Google SRE Book, chapter 4 ("Service Level Objectives").**
  https://sre.google/sre-book/service-level-objectives/ — the
  definitional chapter for SLI / SLO / SLA / error budget. Every
  term used below traces back to this chapter.

## Recap: what "goodput" is

Chapter 1 introduced goodput informally. Now, the definition we
report against:

> **Goodput** is the number of training tokens that (a) were
> consumed by a training step whose gradient was applied, and (b)
> made it into a persisted checkpoint that the platform would
> resume from, per unit wall-clock time.

Two subtleties in that definition worth calling out:

- **"Gradient was applied"** excludes tokens that were consumed by a
  step whose loss was NaN-filtered, whose gradient scale collapsed,
  or that was rolled back. If the step was rolled back, the tokens
  didn't count.
- **"Made it into a persisted checkpoint"** excludes work that was
  in-flight but not yet checkpointed at the time of the last
  failure. On average, `T_ckpt_interval / 2` of every failure's
  work is lost.

Both conditions have to hold; goodput is the intersection of
throughput and successful persistence.

The PaLM paper (Chowdhery et al. 2023, https://arxiv.org/abs/2204.02311,
§5) is the origin of the term for LLM training. The TPU v4 paper
(Jouppi et al. 2023) uses the same framing for the TPU-pod
substrate. Chapter 7 called out that the Llama 3 team reports a
similar effective-training-time metric of >90%.

## Deriving the SLO

The SLO shape you commit to is:

> "Over a rolling 28-day window, goodput will be at least G* useful
> training tokens per H100-hour of allocated capacity."

Two pieces to that shape:

- **Rolling window.** 28 days smooths out day-to-day incident
  noise; long enough that a bad week does not immediately blow the
  SLO but short enough that a chronic regression is caught within
  ~a month.
- **Per-H100-hour of allocated capacity.** The denominator has to
  be *allocated* capacity (what the scheduler granted the run),
  not *actual* running capacity. Otherwise, the metric self-heals:
  fewer running GPUs → higher per-GPU goodput → SLO passes even
  though the run is degraded.

The value `G*` you commit to has to come from measurement, not
guess. Two anchor points:

- **The nominal maximum**, `G_max`, is the tokens/hour a healthy
  step-time × world size delivers with zero downtime and no failed
  steps. Compute it: `G_max = (batch_tokens · world_size) /
  step_time_s · 3600`.
- **The historical actual**, `G_actual`, is what your last 28 days
  measured. On a well-run cluster this will be some fraction — 60%
  to 95% — of `G_max`. The Llama 3 team reported >90%; a new
  platform is likely to start much lower.

Set `G* = G_actual + margin` for the SLO. Setting it above
`G_actual` from day 1 guarantees you burn the budget immediately;
setting it too far below hides regressions. A common pattern is
"start at the P50 of the last 8 weeks; ratchet up quarterly as the
mechanisms in chapters 3–6 mature".

## The availability budget

Once you have `G*`, the availability budget follows directly. Define:

- **Ideal goodput** `G_ideal = G_max`.
- **SLO goodput** `G* = SLO_frac × G_max`, where `SLO_frac` is
  something like 0.90.
- **Downtime budget** `D_budget = (G_ideal − G*) / G_ideal · (window
  length)`. For `SLO_frac = 0.90` on a 28-day window, that's ~67
  hours of "acceptable loss" per 28 days across the whole cluster.

That's the total budget across **all** loss categories: crashes,
recovery, straggler slowdown, silent-corruption rollbacks, planned
maintenance. Splitting the budget:

- **Unplanned incidents** (fail-stop, fail-slow, SDC): typically the
  largest slice, 50%–70% of the budget on a new platform, shrinking
  as chapter 6's automation matures.
- **Planned maintenance and rolling restarts**: the smallest slice
  a mature platform can afford. 10%–20%.
- **Recovery overhead per incident** (chapter 2's detection +
  rendezvous + load): 20%–30%. Chapter 4's discussion of the
  NCCL-timeout knob and chapter 3's async-save cadence are what
  moves this.

Every incident post-mortem should categorize the incident's cost
against one of those slices. Over time, the slice-level breakdown
tells you which chapter's investment to double down on.

## Worked example

A 1 024-H100 cluster running Llama-3-scale pretraining. Assume:

- Step time: 3 s. World size: 1 024. Per-step tokens: ~2M. So:
  `G_max = (2 × 10⁶ × 1024) / 3 × 3600 ≈ 2.5 × 10¹²` tokens/hour
  under ideal conditions. (In practice you would compute this from
  your recipe's per-step tokens; the number here is illustrative.)
- Historical actual over the last 28 days: `G_actual ≈ 1.9 × 10¹²`
  tokens/hour. That is `G_actual / G_max ≈ 0.76` — the SLO ratio.
- Set `G* = 0.75 × G_max ≈ 1.87 × 10¹²`. Slightly below actual so
  the platform starts with a small positive budget.

Downtime budget:

- Wall-clock hours in 28 days: 672. At `SLO_frac = 0.75`, budget =
  `672 × 0.25 = 168 H100-hours-per-hour` × 1 024 GPUs = **168 × 1 024
  ≈ 172 000 GPU-hours** across the window.
- Divided by 1 024 GPUs = 168 hours of "any GPU is unavailable"
  budget over 28 days.
- Split roughly 65% / 15% / 20% across unplanned / planned /
  recovery: ~110 hours unplanned, ~25 hours planned, ~33 hours
  recovery overhead.

Compare against the Llama 3 numbers from chapter 7: 419 unexpected
interruptions over 54 days on a 16 384-GPU cluster. Scaled to
1 024 GPUs and 28 days that would be roughly `(419 × (28/54) × (1024/16384))
= ~13 interruptions` — for context; this is a very rough proportional
scaling, not a prediction. At an average recovery overhead of ~15
minutes per interruption (an aggressive number, achievable with
chapters 3 and 6 in place), that's ~3 hours of unplanned downtime
budget consumed, leaving substantial headroom.

The point of the exercise is not the specific numbers. It is that
**the SLO makes the trade-off explicit**: if the platform doubles the
checkpoint interval to save on I/O overhead, the extra work-loss per
failure eats into the budget. If the platform tightens the NCCL
timeout, false-positive quarantines eat into the budget. The
scoreboard is what forces those decisions to be quantified.

## Error-budget policy

The error budget is not just a number to report; it is a
**policy trigger**. The Google SRE Book's chapter 4 recommends
policy that is proportional to budget consumption:

- **Budget healthy (< 50% consumed):** proceed with normal
  development, roll out new PyTorch versions, take moderate risks.
- **Budget stressed (50%–90% consumed):** freeze non-essential
  changes to the training platform. Rollouts require an explicit
  budget-cost estimate.
- **Budget exhausted (> 90%):** platform-team focus flips to
  reliability work only. No new features until the budget recovers.

This is a **policy the team commits to in advance**, not a case-by-case
judgment. The value of a budget-triggered freeze is that it removes
"should we prioritize reliability" from the weekly negotiation and
turns it into a data question.

## What to measure and where

The concrete instrumentation for a goodput SLO:

- **Per-step tokens applied**, logged from the training loop. This
  is `batch_tokens · world_size` for a step whose gradient was
  applied; 0 for a NaN-filtered or rolled-back step.
- **Wall-clock hours of allocated capacity**, per job, from the
  scheduler. mod-104 chapter 4 covers the plumbing.
- **Checkpoint persistence events.** Each successful DCP
  `future.result()` completion is a persistence event. Tokens
  applied since the last persistence event are "at risk"; on a
  failure they are lost.
- **Incident stream.** Every quarantine event, every rendezvous
  reshape, every rollback, tagged with duration and cause. This is
  what fuels the incident-category breakdown of the budget.

The metric emitters go into your Prometheus / metrics store (mod-108
covers the collection). The aggregation and reporting is a scheduled
28-day rolling job that emits the current SLO ratio, the budget
consumed, and the breakdown by incident category.

## Common mistakes when authoring the SLO

- **Reporting uptime instead of goodput.** A run that is hung but
  not crashed has 100% uptime and 0 goodput. Reporting uptime hides
  exactly the failure mode chapters 5 and 6 exist to catch.
- **Using nominal capacity instead of allocated capacity.** A run
  that got 512 nodes when 1 024 were requested is at 50% goodput
  before it even starts. Denominator has to be what was granted.
- **Rolling window too short.** A 24-hour window makes every
  incident look like a budget crisis; you optimize for false
  alarms. 28 days is a defensible starting point.
- **Setting `SLO_frac` at the aspirational number from day 1.**
  Setting `G* = 0.95 × G_max` before you have the mechanisms to
  achieve it means every week is over-budget and the policy trigger
  is meaningless. Ratchet up as the platform matures.
- **Not budgeting for planned maintenance.** A firmware roll or a
  PyTorch upgrade takes cluster-hours. If planned maintenance is
  unbudgeted, it steals from the incident budget and the metric
  is meaningless.

## Where the SLO meets the other chapters

Every mechanism in this module rolls up into this SLO:

- **Chapter 2's checkpoint interval formula** is what determines
  the average work loss per failure.
- **Chapter 3's DCP async save** is what keeps `T_save` off the
  step-time budget.
- **Chapter 4's rendezvous timeout tuning and elastic reshape** is
  what makes `T_recover` small.
- **Chapter 5's incident playbooks** are what makes recovery
  consistent (not "how did we handle this last time").
- **Chapter 6's detection + quarantine automation** is what
  shortens the "silent-corruption dwell time" component of goodput
  loss.
- **Chapter 7's logbook and drill patterns** are what let the SLO
  be reported honestly rather than optimistically.

The SLO is the number the platform team is measured on; every
chapter above is a lever on it.

## Summary

- **Goodput** is tokens applied and persisted, per wall-clock hour,
  per unit of allocated cluster capacity. Chapter 1 defined it;
  this chapter operationalizes it.
- The SLO commits to a `G* = SLO_frac × G_max` target on a rolling
  28-day window. Set `SLO_frac` from measured baseline; ratchet up
  quarterly.
- The availability budget is `(G_max − G*) × window`. Split into
  unplanned incidents, planned maintenance, and recovery overhead
  slices; every post-mortem categorizes against them.
- Error-budget policy is a pre-committed rule: healthy → normal
  development; stressed → change freeze; exhausted → reliability
  focus only.
- Instrument per-step tokens applied, allocated capacity from the
  scheduler, checkpoint-persistence events, and an incident stream.
  Report the rolling SLO and the budget-consumed breakdown weekly.
- Common mistakes: reporting uptime instead of goodput, using
  nominal instead of allocated capacity, setting `SLO_frac` too
  aggressively, ignoring planned maintenance. Avoid all four
  before you ship the SLO.
