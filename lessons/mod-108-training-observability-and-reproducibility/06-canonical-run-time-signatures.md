# Canonical Run-Time Signatures: What Trouble Looks Like on the Dashboard

Chapter 2 built the dashboard. mod-106 chapter 5 classified the
five incident classes and gave the response runbook for each. This
chapter closes the loop: given the metrics on the dashboard, what
does each class actually *look like*, and how do you tell one from
the other quickly?

The signal-side skill is different from the response-side skill.
On-call at 3 AM needs to recognize a pattern from the shape of a
few curves in under a minute; the correct response is a lookup
against the runbook once the pattern is named. This chapter's job
is to build that pattern library.

Five canonical signatures, each with the same shape:

- **Curve shape** — the qualitative picture on the dashboard.
- **Correlated metrics** — the second and third panels that
  co-move (or don't) and disambiguate.
- **Fastest disambiguation query** — the single Prometheus / tracker
  query that confirms it in one call.
- **First-response action** — the pointer into mod-106 chapter 5's
  runbook.

The five: **divergence**, **loss spike**, **throughput cliff**,
**straggler**, **silent corruption**. Learn them cold; every incident
you will see for a year is one of them.

## Signature 1: divergence

### Curve shape

The training loss stops decreasing and either plateaus at a high
value or slowly climbs. The validation loss follows or leads. No
crash, no NaN — the loop is running.

Two sub-shapes:

- **Plateau-and-drift**: loss levels off at the current value and
  slowly walks up over hundreds of steps. Usually optimization-
  pathology.
- **Immediate re-ascent**: after a checkpoint restart or an LR
  change, loss shoots up and stays up. Usually a config or
  restart bug.

### Correlated metrics

- **Gradient norm**: often grows monotonically over the divergence
  window. If it also spikes, see signature 2 first.
- **Weight norm**: also grows, often faster than gradient norm.
- **Per-layer gradient norms** (chapter 4's tracker view): usually
  one or two layers dominate. LayerNorm scales going to zero or
  a specific attention layer's `q_proj.weight` going to infinity
  are common patterns.
- **MFU**: usually unchanged. The step is executing at the same
  rate; the model is just walking off the manifold.

### Fastest disambiguation query

Is the gradient norm rising or stable? A rising gradient norm on a
still-executing step is divergence, not any of the other four.

```
rate(train_gradient_norm_global{run_id="$run_id"}[10m])
```

A positive value sustained for tens of minutes is the signal.

### First-response action

Diagnose *which* layer is diverging (per-layer norms). If it is a
known-fragile op (softmax with large logits, layernorm at a
specific position), the fix is upstream in the recipe (temperature
clamp, dtype promotion for that op). If nothing is obviously wrong,
roll back to the last DCP checkpoint before the divergence started
and lower LR or increase gradient clipping (mod-106 chapter 5,
class 1 runbook).

Reference: Chowdhery et al. (2022, arXiv 2204.02311, appendix F)
document PaLM's loss-spike experience and the "restart from an
earlier checkpoint with a different data order" workaround. Zhang
et al. (2022, arXiv 2205.01068, OPT-175B chronicles) document the
same class extensively.

## Signature 2: loss spike

### Curve shape

The loss curve has a sharp discontinuity: it jumps by some multiple
(often `2×` to `10×`) at a single step, then either recovers within
a handful of steps or stays elevated. The distinguishing feature
from divergence is that a spike is *fast* — one to a few steps
wide — where divergence is slow.

### Correlated metrics

- **Gradient norm**: also spikes at the same step, usually by a
  larger factor. This is the causal signal — the spike in
  gradient norm is what the optimizer's step size then translates
  into a loss spike.
- **LR schedule position**: overlay the LR schedule. Warm-up
  boundaries, cosine minima, and phase transitions (e.g., a
  data-mix change) are frequent spike triggers. A spike at a
  schedule inflection is a config problem; a spike mid-plateau
  is a data or numerical problem.
- **Data sampler position**: a spike that recurs when the sampler
  hits a specific range of `(shard_id, offset)` is a data-side
  problem (corrupted shard, unusual token distribution).
- **Per-rank loss** (chapter 4's sharded emission): a spike that
  is on *one rank* but not the others is silent-corruption-adjacent
  (see signature 5) — the "spike everywhere at once" is a real
  training-side spike.

### Fastest disambiguation query

Is the loss discontinuity narrow (a few steps) or wide (many
steps)? Query:

```
train_loss{run_id="$run_id"} - train_loss{run_id="$run_id"} offset 5m
```

A large jump followed by a return-to-band within a few queries is
a spike; a sustained elevation is divergence.

### First-response action

If the spike self-recovered within a handful of steps, log it and
continue. If it stayed elevated, roll back to the pre-spike DCP
checkpoint, lower LR by 10–50%, resume (mod-106 chapter 5, class
1 decision box). If the spike correlates with a specific data
range, quarantine that range from the sampler (mod-103 owns the
sampler; the quarantine is the sampler-side runbook).

Every spike gets a `(step, event, cause)` entry in the run's
logbook. Six weeks later, when a similar spike appears, on-call
compares against the entry. This is the OPT-175B chronicles
pattern (mod-106 chapter 7).

## Signature 3: throughput cliff

### Curve shape

The per-step time (median across ranks) increases sharply — often by
`1.5×` to `3×` — and stays elevated. The loss curve looks completely
normal. From the researcher's perspective, nothing is wrong; from
the platform's perspective, goodput just collapsed.

Two sub-shapes:

- **Step function**: step time doubles at a specific step and stays
  there. Usually a scheduler-side event (a co-tenant landing on
  the node, a fabric partitioning change).
- **Ramp**: step time climbs monotonically over hundreds of steps.
  Usually a slow degradation (thermal throttling as ambient
  temperature drifts up, HBM row-remap cost).

### Correlated metrics

- **Per-step-time dispersion across ranks** (chapter 2, Row 2): if
  dispersion is normal but the *whole* fleet slowed, it is a
  cluster-wide event (fabric-wide congestion, framework upgrade,
  data-loader stall). If dispersion is elevated, it is a
  straggler (see signature 4).
- **Communication time as a fraction of step time**: rising during
  the cliff suggests a fabric-side problem (mod-105 chapter 8).
- **GPU utilization histogram**: if utilization dropped for most
  ranks together, the trainer is waiting on something outside the
  GPU — data loader, checkpoint, host-side code path.
- **MFU**: drops by the same fraction as the step time increased,
  which is what makes it visible on the dashboard.

### Fastest disambiguation query

Did dispersion move? A cluster-wide slowdown is one class; a
per-rank slowdown is another.

```
system_step_time_dispersion{run_id="$run_id"}
```

Values around `1.0` → cluster-wide slowdown. Values elevated →
straggler (jump to signature 4).

Then, if cluster-wide, check comm fraction:

```
comm_time_ms{run_id="$run_id"} / step_time_ms{run_id="$run_id"}
```

Rising → fabric. Stable → framework / loader / host.

### First-response action

Check the change log — a recent framework upgrade, kernel driver
update, or fabric change is often the cause. If cluster-wide and
fabric-attributable, engage mod-105 chapter 8's runbook. If
cluster-wide and driver/framework-attributable, roll back the
change. If not obvious, correlate against DCGM's throttle-reason
counts (chapter 3) — a fleet-wide `THERMAL` throttling event is a
facilities incident.

Throughput cliffs are the class most often initially misdiagnosed
as "the model is just slow today"; the disambiguation query above
prevents that.

## Signature 4: straggler

### Curve shape

Per-step-time dispersion (`max_rank_step_time / mean_rank_step_time`)
walks above `1.10–1.20` and *stays* there for many consecutive
steps. Median step time may or may not rise (a straggler slower
than the median by 20% is dragging the collective by 20% too);
usually it rises somewhat, because everyone waits at the
collective.

Distinguishing feature: **the same rank is the max-time rank for
tens or hundreds of consecutive steps**. A different rank being the
max each step is normal jitter and not a straggler.

### Correlated metrics

- **DCGM temperature, per node** (chapter 3): if the affected
  rank's GPU is hotter, thermal throttling is the cause.
- **DCGM throttle reason, per rank**: `THERMAL`, `HW_SLOWDOWN`, or
  `POWER` on the straggler rank confirms hardware-side throttling.
- **DCGM ECC counters**: an increase in row-remap or SBE counters
  on the straggler correlates with a slow-memory pattern.
- **NIC counters** (mod-105 chapter 8): a rank whose NIC is
  quietly retransmitting slows every collective it participates
  in. `mlnx_perf` per-node deltas expose this.
- **Data-loader wait time** (per-rank, chapter 4 sharded emission):
  if the straggler's loader is stalling, the issue is upstream in
  the storage tier (mod-103 chapter 4).

### Fastest disambiguation query

Which rank is the max, and for how long?

```
argmax_over_time(rank_step_time_ms{run_id="$run_id"}[1h])
```

The same rank appearing repeatedly is a straggler. Correlate that
rank's node against DCGM:

```
DCGM_FI_DEV_GPU_TEMP{Hostname="<rank_host>"}
DCGM_FI_DEV_CLOCKS_EVENT_REASONS{Hostname="<rank_host>"}
```

### First-response action

Mod-106 chapter 6's straggler runbook. In brief: verify signal (a
few outlier steps do not count); identify root cause via DCGM /
NIC / loader; quarantine the node; force an elastic rendezvous;
the job resumes at `WORLD_SIZE − nproc_per_node`. Never auto-clear
the quarantine on a timer (mod-106 chapter 6).

## Signature 5: silent corruption

### Curve shape

**Almost nothing on the dashboard.** This is what makes SDC the
hardest class to catch. All the primary panels look normal:

- Loss curve trends normally.
- Gradient norm is in the usual band.
- Step time is normal.
- GPU utilization is normal.
- No throttling, no obvious ECC counter jumps.

The signature is in the *comparative* signals, not the absolute
ones. What you look for:

- **Per-rank loss divergence**: chapter 4's sharded per-rank loss
  emission plotted as a heatmap or a `max_over_ranks(loss) - mean_over_ranks(loss)`
  time series. A specific rank whose loss walks off from the
  others' loss over many samples is the SDC signal.
- **Cross-rank weight-hash disagreement**: mod-106 chapter 6's
  periodic weight-hash all-reduce. A rank whose hash of the
  replicated parameters differs from the others is either a bug
  or SDC.
- **Downstream eval regression**: a checkpoint whose eval score is
  measurably worse than an adjacent checkpoint on the same eval,
  with no obvious training-side cause. This is the *lagging*
  version of the signal; you want to catch SDC before eval does.
- **Rising SBE ECC counter, well before any DBE**: SBE is
  correctable; a rising rate suggests the memory is degrading and
  the DBE (uncorrectable) event is coming. Chapter 3's
  ECC-delta panel is the on-call surface.

### Fastest disambiguation query

Two queries in sequence:

1. Is a specific rank consistently the loss outlier?
   ```
   argmax_over_time(rank_loss{run_id="$run_id"}[1h])
   ```
   The same rank being the max-loss over many samples is a red
   flag.

2. Does that rank's GPU have rising ECC?
   ```
   increase(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL{Hostname="<rank_host>"}[1d])
   increase(DCGM_FI_DEV_ROW_REMAP_PENDING{Hostname="<rank_host>"}[1d])
   ```
   (Confirm the exact field names against your DCGM release.)

### First-response action

Mod-106 chapter 6's silent-corruption runbook. In brief: run the
deterministic replay drill (load a pre-suspect-window checkpoint,
replay N steps on a small known-good cluster, compare loss curves).
If confirmed, roll back to the last known-good checkpoint,
quarantine the node, and *record the affected physical GPU UUID*
so hardware ops does not re-issue it. Silent-corruption dwell
time (mod-106 chapter 5, class 5 post-mortem) is the metric the
platform team tracks.

Primary references for the class: the Meta paper "Silent Data
Corruptions at Scale" (Dixit et al., 2021, arXiv 2102.11245) and
the Google paper "Cores that don't count" (Hochschild et al.,
2021, ACM HotOS 2021) are the community's ground truth for how
common and how expensive SDC is at scale.

## The two-page summary

Print and keep on-call:

| Signature | Curve shape (60-second version) | First correlate | Runbook |
|-----------|--------------------------------|------------------|---------|
| Divergence | Loss stops decreasing, slowly rises; no crash | Gradient norm rising | mod-106 ch. 5 class 1 |
| Loss spike | Loss jumps `2–10×` for 1–5 steps | Gradient norm same-step spike | mod-106 ch. 5 class 1 |
| Throughput cliff | Step time jumps or ramps; loss normal | Dispersion (cluster vs. straggler split) | mod-106 ch. 5 class 3 / mod-105 ch. 8 |
| Straggler | Dispersion `> 1.1` sustained; same rank is max | DCGM throttle / ECC on that rank | mod-106 ch. 6 |
| Silent corruption | Nothing on primary panels | Per-rank loss + ECC on suspect rank | mod-106 ch. 6 |

## What NOT to do

- **Do not read the mean and assume the fleet is healthy.** The
  mean hides everything. Read distributions and outliers.
- **Do not stack signatures.** If loss spike *and* throughput
  cliff hit at the same step, deal with the more urgent one
  (spike) first; treat the cliff as a follow-up. Multiple
  concurrent signatures usually share a root cause you'll surface
  by finishing the first response.
- **Do not skip the reproducibility bundle diff** (chapter 5)
  when a signature shows up after a recent config change. A
  `bundle-diff` between "run that was healthy yesterday" and
  "run that is diverging today" is the fastest attribution tool
  available.
- **Do not treat the runbook as static.** Every incident that
  does not match a signature above becomes a new entry — either
  by extending an existing signature or by adding a sixth.
  Signatures are a living catalog; a runbook that has not been
  updated in a year is stale.

## Summary

- Five canonical signatures cover the incident surface at the
  metric level: divergence, loss spike, throughput cliff,
  straggler, silent corruption. Learn them cold.
- Each has a distinctive curve shape, a distinctive first
  correlate, a specific disambiguation query, and a pointer into
  mod-106's response runbook. The chapter's tables are the
  minimum you carry on-call.
- Divergence is slow; spikes are fast. Cliffs affect throughput
  but not loss; stragglers show up as dispersion, not median.
- Silent corruption is the class that does not show up on the
  primary panels. It shows up in comparative signals — per-rank
  loss divergence, cross-rank weight-hash disagreement, ECC
  counter deltas, and downstream eval regression. Mod-106
  chapter 6's proactive detection is what closes the gap.
- The signature catalog is a living document. Every incident that
  does not match extends or adds a signature; the catalog is one
  of the artifacts chapter 7's metadata store references.
