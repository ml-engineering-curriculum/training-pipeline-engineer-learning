# Goodput SLO, Availability Budget, and Case Studies

The previous six chapters gave you the mechanics — DCP, elastic
reshape, incident playbooks, straggler and silent-corruption
detection, auto-quarantine. This chapter is the layer above:
given a business target ("this run must finish in X wall-clock
days"), how do you derive the checkpoint interval, the
allowed-downtime budget, and the expected recovery cost that make
that target achievable? Then we read two public runs — OPT-175B
and Llama 3 — as case studies at platform-engineer altitude.

Two references anchor the chapter:

- **OPT-175B logbook.** Zhang et al. (2022), arXiv:2205.01068.
  The paper links to the day-by-day operational log; treat it as
  the reference on-call diary for large-scale pretraining.
- **Llama 3 §6.3.** Grattafiori et al. (2024), "The Llama 3 Herd
  of Models". The failure taxonomy, MTTR, and effective goodput
  numbers for the 405B run on 16 384 H100s.

Both are worth a proper read after this chapter. Exercise 05
walks you through building a run SLO against them.

## The goodput identity, revisited

Chapter 1 introduced the identity you carry through this chapter:

```
goodput  =  throughput  ×  availability  ×  retention
```

- **`throughput`** — tokens/sec while a step is running. Owned by
  mod-107 (MFU engineering, kernels, mixed precision,
  comm-compute overlap).
- **`availability`** — fraction of wall-clock spent inside a
  running step. Everything in this module that reduces downtime
  (async checkpoints, elastic reshape, auto-quarantine) shows up
  here.
- **`retention`** — fraction of tokens that survived and were
  incorporated into the model. Rewinds after loss spikes,
  NaN-forced restarts, or silent-corruption incidents show up
  here.

You do not tune goodput directly. You tune the three factors
against a fixed compute budget, and the run's ship date falls out.

## Deriving an SLO from a business target

The exercise looks like this. You have:

- **A dataset budget.** How many tokens the run must consume.
  For a Chinchilla-optimal 70B model that is ~1.4T tokens
  (Hoffmann et al., 2022, arXiv:2203.15556).
- **A cluster shape.** How many GPUs, of which model. Say 512 ×
  H100 for our example.
- **A measured throughput.** From a well-tuned initial run: say
  4 000 tokens/sec/GPU across the cluster (this is a proxy
  number; measure yours).
- **A business ship-date target.** Say 21 wall-clock days from
  first token.

Do the arithmetic.

```
tokens_target      =  1.4e12
gpus               =  512
throughput_per_gpu =  4 000 tok/s
cluster_throughput =  512 × 4 000 = 2.048e6 tok/s

compute_seconds        =  tokens_target / cluster_throughput
                       =  1.4e12 / 2.048e6  ≈  683 594 s
                       ≈  7.9 days
```

If perfect availability and perfect retention were free, the run
finishes in 7.9 days of pure compute. You are targeting 21 days
of wall-clock, so:

```
required_wall_clock  =  21 days = 1 814 400 s
required_availability × retention =
                          compute_seconds / wall_clock
                       ≈  683 594 / 1 814 400
                       ≈  0.377
```

You need `availability × retention ≥ 0.377`. That is a lot of
budget on paper — nearly 63% of your wall clock can be spent on
downtime and rewinds. In practice you set a stretch:

```
target availability = 0.90
target retention    = 0.90
implied compute     = 0.90 × 0.90 = 0.81 of wall_clock
                    = 1 814 400 × 0.81 ≈ 1 469 664 s
run finishes in     = compute_seconds / 0.81 ≈ 9.75 days
```

So a 21-day target at `A × R = 0.81` finishes in ~9.75 days,
with ~11 days of budget to burn on unplanned incidents. That is
comfortable. A tighter target — say 14 days — with the same
`A × R` finishes in ~9.75 days *if the budget matches*, meaning
you have ~4 days of slack.

The point of this exercise is that the SLO is not "keep the run
up". It is a specific numeric availability and retention target
that leaves a specific unplanned-downtime budget on the table.
Chapter 6 of mod-108 will surface those numbers on the dashboard.

## Turning the SLO into a checkpoint interval

Now the operational question: how frequently should you
checkpoint? Two forces pull in opposite directions.

- **Frequent checkpoints minimize retention loss on rewind.** If
  you checkpoint every 5 minutes and rewind on incident, you
  lose at most 5 minutes of tokens.
- **Frequent checkpoints hurt availability.** Every save costs
  the staging time (chapter 3). Even with async save, staging
  costs seconds per save.

Formalize. Let:

- `I` = checkpoint interval (seconds).
- `S` = per-save staging cost (seconds; measured on your cluster).
- `M` = incident MTBF (seconds; measured from your run history).
- `R` = per-incident recovery cost (seconds; DCP load + reshape +
  ramp-up). Include the *retention* cost too: on average an
  incident rewinds `I / 2` seconds worth of tokens.

The **expected unavailability rate** per unit time is:

```
unavailability_rate =
    (staging cost fraction)       + (per-incident cost fraction)

  =  S / I                        +  (R + I / 2) / M
```

Two things to notice:

- Cranking `I` down (more frequent checkpoints) increases the
  first term linearly and decreases the second term linearly.
- The **optimal `I`** is where the derivatives cancel:

```
d/dI [ S/I + (R + I/2)/M ]  =  -S/I² + 1/(2M)  =  0

           I_opt  =  sqrt(2 · S · M)
```

Plug in reasonable numbers:

- `S` = 20 s (staging time for a 70B model, measured).
- `M` = 3 h = 10 800 s (Llama 3-scale incident rate).
- `I_opt` = `sqrt(2 × 20 × 10 800)` = `sqrt(432 000)` ≈ **657 s
  ≈ 11 minutes**.

Interpretation: on a 3-hour MTBF and 20-second staging, you
should be checkpointing roughly every 11 minutes. Any faster,
you burn availability on staging; any slower, you burn retention
on rewinds.

Adjust for your cluster's numbers. Larger `M` (better MTBF)
extends the interval; larger `S` (slower storage) also extends
it; but if `M` collapses (a bad run) the SLO tells you to shorten
`I`. This is the arithmetic that says "if we can't fix the
hardware, we can at least shorten the checkpoint interval."

## The four cost buckets that eat availability

Beyond staging and rewinds, four other buckets consume
availability. Track each one:

| Bucket | Typical share on a healthy run | Where to look |
|--------|-------------------------------|---------------|
| Checkpoint staging + write | 1–3% | Async save telemetry |
| Elastic reshape (rendezvous + load) | ~1% | torchrun events |
| Rewinds after incidents | 2–5% | Runbook ticket log |
| Node repair wait | 0–2% | Scheduler quarantine log |
| **Total unavailability** | **4–10%** | dashboards |

A run at 90% availability is spending 10% of wall clock on the
sum of these. If any one bucket balloons — say elastic reshape is
15% of wall clock because rendezvous is thrashing — the run's
goodput drops and the SLO is at risk.

mod-108 owns the dashboards; mod-106 owns the alert thresholds.
An SLO's `A × R = 0.81` target implicitly says "keep every
bucket at its typical share". Deviations page.

## Case study 1: OPT-175B (Meta AI, 2022)

Zhang et al. (2022) trained OPT-175B for ~90 days on ~992 A100s
(reported specifically as one of the first industry-scale open
replications). The paper is worth reading; the *logbook* linked
from the paper is the on-call artifact. Read at platform-engineer
altitude, three patterns stand out.

### 1. Incidents of all five chapter-5 classes appeared

The logbook records loss spikes, NaN incidents, NCCL timeouts,
hardware failures, and periods where the team suspected silent
corruption (with corresponding deterministic re-runs). Every
category chapter 5 defined has an entry.

### 2. The team rewound and skipped batches routinely

For loss spikes, the team learned to skip specific problematic
batch ranges rather than reduce LR every time. This required
knowing the batch index (which is why chapter 3's stateful
dataloader is critical) and a lightweight "skip N steps" hook
in the training loop.

### 3. The on-call cadence was continuous

The team ran shifts around the clock. The paper's discussion of
the training-time realism is one of the earliest public
acknowledgements that "just start it and check back" is not a
viable operations model at scale.

The lesson for your own runs: budget for a rotating on-call
during any run above ~2 weeks. Automate everything that can be
automated (chapter 6's auto-quarantine), page a human only for
class-E incidents and class-A divergent spikes.

## Case study 2: Llama 3 405B (Meta AI, 2024)

Grattafiori et al. (2024), "The Llama 3 Herd of Models",
tabulates the reliability data for the 405B run in section 6.3.
The headline numbers, as reported by the paper:

- **54-day pretraining window.**
- **16 384 H100 GPUs** (2048 hosts × 8 GPUs).
- **~419 unexpected interruptions** across the window.
- **~78% attributable to confirmed hardware issues**; the
  remainder split across software, network, and unresolved.
- **~30 seconds** of average recovery time when auto-recovery
  succeeded (i.e., no human paged).
- **Effective goodput ≥ 90%** across the full window.

Read the tables directly for the failure-class breakdown. The
categories the paper reports (GPU issues, host failures, network
issues, silent data corruption events, software issues,
maintenance, unknown) map cleanly onto chapter 5's five
classes. mod-106 was designed to be readable against those
tables.

### The Llama 3 recovery arithmetic

Plug the numbers into the formulas above:

- **MTBF from the observed rate:** 54 days / 419 incidents
  = ~3.1 hours per incident. This matches the 2048-node
  back-of-envelope from chapter 1 (~44 minutes if you naively
  divide, but the paper's "unexpected interruption" definition
  is stricter than "any node failure", so the effective MTBF is
  lower per-incident-that-matters).
- **Recovery cost per incident:** 30 s + typical checkpoint
  interval / 2 for the retention loss.
- **Total unavailability:** at ~30-second recovery per
  incident, 419 incidents × 30 s = 12 570 s ≈ 3.5 hours over 54
  days, or ~0.27% of wall clock. Retention loss is on top of
  that: `419 · I/2` seconds of rewound work.

If checkpoint interval was 30 minutes (`I = 1800`), retention
loss = `419 · 900 s = 377 100 s ≈ 4.4 days`. That is the
retention bill for a checkpoint interval that long. Reducing
`I` to 5 minutes (`I = 300`) would take that to `419 · 150 =
62 850 s ≈ 17 hours`, at the cost of more staging.

The point: even a well-tuned production run at H100 scale is
spending days of wall clock on retention loss. Every minute you
shave off `I` (down to the `I_opt` from the formula) shows up
directly in the ship date.

### The 90% goodput headline

The paper reports **≥90% effective goodput** across the 54-day
window. That is the metric the whole module is oriented around;
it is what a well-designed platform delivers to the modeling
team.

Ways to lose 10 points of goodput (i.e., to end up at 80%):

- Skip async save; do sync. Adds ~3–5 points to staging burden.
- Skip auto-quarantine; page every incident. Adds hours of
  paged-latency per incident × 400 incidents = a lot.
- Skip stateful loader; replay batches on every restart. Adds
  linear retention loss × N restarts.
- Overshoot checkpoint interval; rewind more per incident.

The point is that each mechanism in this module contributes a
few percent of goodput; skipping any of them costs proportionally.
The exercises tie each mechanism to its contribution.

## The SLO write-up (what a document looks like)

Exercise 05 asks you to author a goodput SLO for a hypothetical
30-day 400B pretraining run. The written artifact should be
roughly this shape:

```
## Run SLO: run-alpha-2026-08

### Target
Ship-date: 30 wall-clock days from first token.
Tokens target: 8T tokens on 1024 H100s.

### Measured baselines (from bring-up run)
Throughput per GPU: 3 600 tok/s
Cluster throughput: 3.69 × 10^6 tok/s
Compute duration (100% A×R): ~2 168 000 s ≈ 25.1 days
Slack: 30 - 25.1 = 4.9 days of unavailability budget

### Target availability × retention
0.90 × 0.90 = 0.81 → run finishes in ~31 days
Adjust target throughput +5% (mod-107 tune) or A×R = 0.85
(target both at 0.92) → run finishes in 29.5 days ✓

### Checkpoint policy
Staging cost (measured): 25 s
Assumed MTBF: 3.0 h
I_opt = sqrt(2 × 25 × 10 800) = ~735 s ≈ 12 min
Set checkpoint interval: 15 minutes
Async save + CPU staging + stateful loader per mod-106 ch. 3.

### Quarantine policy
DCGM XID {13, 43, 45, 48, 79, ...}: auto-drain, page.
Uncorrectable ECC: auto-drain, page.
Step-time straggler > 20% for 5 min: page (no auto-drain).
Checkpoint hash mismatch: halt run, page.

### Elastic reshape
MIN_NODES = 120 (of 128); MAX_NODES = 128.
Rendezvous: etcd (HA), 3-node etcd cluster.

### On-call
2 engineers, alternating weeks. Playbooks per mod-106 ch. 5.

### Success criteria
Goodput ≥ 0.85; run completes within 30 days.
```

That document is the SLO. It fits on one page. It is
review-able by any of your peers. It is the thing you commit to,
in the same sense that an SRE team commits to a 99.9% latency
target.

## Tie-back to the module

- **Chapters 2 and 3** gave you the checkpoint mechanism whose
  interval you tune here.
- **Chapter 4** gave you the elastic reshape whose cost shows up
  in the availability budget.
- **Chapter 5** gave you the incident playbooks that the rewind
  and quarantine terms are counting.
- **Chapter 6** gave you the detection that makes auto-quarantine
  possible, keeping recovery cheap.
- **This chapter** does the arithmetic. Everything below the
  goodput identity is what the previous six chapters were for.

Exercise 05 walks you through this SLO write-up end-to-end.
Exercise 03 walks you through the incident-playbook mapping
against the Llama 3 §6.3 tables. Lab 01 asks you to read the
OPT-175B logbook cover-to-cover and translate the top-five
incident classes into your own runbook entries.

## Summary

- Goodput = throughput × availability × retention. Chapter 7
  turns the identity into an SLO you can write on one page.
- Deriving the SLO: take a business ship-date target, back
  out compute-seconds, and the ratio to wall-clock gives you
  `A × R`. Set target availability and retention independently.
- Optimal checkpoint interval is `I_opt = sqrt(2 · S · M)` where
  `S` is per-save staging cost and `M` is incident MTBF.
  For typical H100-scale numbers this lands around 10–20 minutes.
- Four cost buckets eat availability: staging, elastic reshape,
  rewinds, node repair wait. A healthy run keeps their sum below
  ~10%.
- OPT-175B (2022) is the canonical logbook: every chapter-5
  incident class appears, the team ran continuous on-call, and
  batch-skipping was the pragmatic loss-spike recovery.
- Llama 3 §6.3 (2024) reports ~419 interruptions over 54 days
  on 16 384 H100s (~78% hardware), ~30 s auto-recovery, and
  ≥90% effective goodput. Read the failure-class table directly.
- The SLO write-up is a one-page contract with the modeling
  team. It commits you to specific availability and retention
  numbers and derives a checkpoint interval, quarantine policy,
  and on-call rotation from them.
