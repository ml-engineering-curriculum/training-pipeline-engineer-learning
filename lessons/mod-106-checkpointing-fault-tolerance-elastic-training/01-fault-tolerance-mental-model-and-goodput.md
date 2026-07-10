# The Fault-Tolerance Mental Model: A Training Run Is a Stream of Jobs

mod-101 through mod-105 taught you how to build one training step:
the collectives, the framework internals, the data loader, the
scheduler that places the gang, the fabric that carries the traffic.
Every one of those chapters implicitly assumed the same thing — that
the step you just wired runs, unchanged, for the next thirty days.
On a 1024-GPU pretraining run, that assumption is wrong by roughly
one incident per day. This module is about what happens the other
399 hours.

## Motivation: at scale, a training run is not a single job

The single most useful mental shift when you move from a 8-GPU
fine-tune to a 1024-GPU pretraining run is this: **the training run
is not a single job. It is a stream of jobs, stitched together by
checkpoints, that together approximate the trajectory of the single
job you would have run if hardware were perfect.** The stitching
work — checkpoint I/O, rendezvous, elastic reshape, incident
diagnosis, node quarantine — is not overhead you tolerate; it is the
substrate on which the training run actually exists.

Two pieces of public evidence that make this concrete:

- **The OPT-175B logbook.** Meta AI released the day-by-day
  operational log of the OPT-175B pretraining run alongside the
  paper (Zhang et al., 2022; arXiv:2205.01068). Over 90 days on
  1024 A100s, the on-call engineers logged loss spikes, hardware
  failures, NCCL timeouts, host reboots, and rewinds — dozens of
  distinct incidents that would each have killed the run without
  the checkpoint / rewind machinery.
- **The Llama 3 405B paper.** Grattafiori et al. (2024), "The Llama
  3 Herd of Models", reports in section 6.3 that during a
  54-day pretraining window on 16 384 H100s the team observed
  roughly **419 unexpected interruptions**, ~78% of them
  attributable to confirmed hardware issues (GPU failures, HBM
  faults, network fabric events). That is one interruption every
  ~three hours across the run.

Neither run "worked" in the sense that you press start and come
back later. Both ran because the platform was designed with the
assumption that individual jobs would die and that the *system* had
to keep making progress anyway. mod-106 is that system.

## Goodput: the metric that survives the shift in perspective

Once you accept that the run is a stream of jobs, wall-clock
throughput (tokens/sec while training) stops being the right
top-line metric. What matters is how many *useful* training tokens
per hour of *wall clock* land in the model that eventually ships.
That ratio has a name: **goodput**.

Google's paper on ML productivity — Mohan et al. (2024), "Characterizing
ML Training Workloads on Nvidia H100 GPUs" and Kokolis et al. (2025),
"Revisiting Reliability in Large-Scale Machine Learning Research
Clusters" — uses the term formally, but the working definition you
need is straightforward:

```
                        useful training tokens produced
    goodput  =  ------------------------------------------
                  wall-clock time from run-start to done
```

"Useful" excludes tokens that were computed but then discarded when
the run rewound to an earlier checkpoint. "Wall-clock" includes every
second the cluster was allocated to the run, whether the trainer
was up, down, checkpointing, rendezvous-ing, or waiting for a bad
node to be replaced.

Three secondary rates roll up into goodput. You will see them again
in chapter 7 as the top-level budget for the run:

| Rate | What it measures | Where it degrades |
|------|------------------|-------------------|
| `throughput` | tokens/sec **while a step is running** | Kernel efficiency, MFU, comm-compute overlap (owned by mod-107) |
| `availability` | fraction of wall-clock spent inside a running step | Rendezvous, checkpoint save/load, node repair, elastic reshape (this module) |
| `retention` | fraction of computed tokens that survive rewinds | Checkpoint interval vs. incident MTTR, whether an incident forces rewind or replay (this module) |

`goodput = throughput * availability * retention` is the identity to
carry through the rest of the module. If any of the three factors
tanks, the run's ship date slips proportionally.

## Mean-time-between-failures at scale

The reason you cannot ignore availability and retention on a large
run is that MTBF at scale is *worse than the per-node MTBF divided
by the number of nodes* — it is roughly that number multiplied by
the number of *dependent* components you have. A pretraining run
gang-scheduled onto 2048 nodes fails when **any** node fails, so
its cluster-level MTBF scales inversely with node count.

A back-of-envelope calculation you should be able to reproduce:

```
per-GPU MTBF (published H100 field data)  ~=  20 000 hours
GPUs per node                              =  8
per-node MTBF (GPU-only)                   ~=  2 500 hours
per-node MTBF (with NIC, PSU, DRAM, ...)   ~=  1 500 hours (field estimate)

cluster of 2048 nodes:
  expected time between any node failing   =  1500 / 2048 hours
                                           ~=  0.73 hours  =  ~44 minutes
```

That number is not "unusual for AI infrastructure". It matches the
Llama 3 405B measurement — 419 interruptions over ~1300 hours on
2048 nodes is one every ~3 hours, which is in the same order of
magnitude with a healthier per-node MTBF factored in. Grattafiori
et al.'s section 6.3 tables show the actual failure taxonomy behind
that number; we come back to it in chapter 7.

Two consequences for how you design the platform:

1. **The checkpoint interval has to be shorter than the MTBF, not
   the whole-run duration.** If your MTBF is 3 hours and your
   checkpoint interval is 6 hours, on average every incident costs
   you 3 hours of retention. If your interval is 30 minutes, the
   average cost drops to 15 minutes. Chapter 7 does the arithmetic.
2. **Recovery must be automatic and mostly-hands-off.** If every
   incident pages a human, on-call burns out inside a week. The
   incident playbooks in chapter 5 exist so that the common cases
   auto-recover; humans only touch the run when something is
   genuinely novel.

## Availability budget: the SLO framing

Availability budgets are a well-worn idea from SRE — Beyer et al.
(2016), *Site Reliability Engineering*, chapter 3 — and they carry
over almost unchanged to training runs. You pick a wall-clock
target ("we want the 30-day run to finish in 30 days, not 45") and
back-derive the availability the platform must deliver.

```
target run duration          =  T_target        (wall clock)
useful compute duration      =  T_compute       (throughput * tokens_target)
required availability        =  T_compute / T_target
allowed unavailability       =  T_target - T_compute
allowed unavailability rate  =  (T_target - T_compute) / T_target
```

If your throughput allows the run to finish in 25 days of pure
compute and you target 30 days of wall clock, your unavailability
budget is `(30 - 25) / 30 = 16.7%`. That budget then decomposes
into:

- **Planned unavailability** — DCP save/load stalls, scheduled node
  drain, upgrade windows.
- **Unplanned unavailability** — hardware failures, NCCL timeouts,
  loss spike rewinds.

The rest of this module is how you spend that budget. Chapter 7
turns the SLO into a checkpoint interval and a recovery-cost
budget explicitly.

## The recovery ladder

Every incident falls somewhere on a ladder of recovery cost,
ordered by how much wall-clock and how much retention it costs:

| Recovery action | Wall-clock cost | Retention cost | Owned by |
|-----------------|-----------------|----------------|----------|
| Resume in-place (same ranks) | seconds | zero | Chapter 4 |
| Elastic reshape (drop a node, keep going) | ~minutes | zero | Chapter 4 |
| Reload last DCP, resume | ~10 min | last interval | Chapter 2, 3 |
| Reload last DCP, skip N steps (bad data) | ~10 min | last interval + skipped batch | Chapter 5 |
| Rewind to last "known good" checkpoint (loss spike) | ~10 min–hours | large | Chapter 5 |
| Full restart from checkpoint zero | hours–days | catastrophic | Never; document why |

The point of an on-call runbook is to keep every incident as close
to the top of the ladder as possible. If loss-spike incidents force
a rewind by six hours every time, your effective goodput is a
fraction of your throughput.

## What this module owns vs. what other modules own

The fault-tolerance / recovery story sits in the middle of the
curriculum; several other modules touch it, and the boundaries
are worth stating up-front.

- **mod-101 (foundations)** owns the collectives whose semantics
  DCP inherits. mod-106 uses `DTensor` and `ShardedTensor` state
  dict layouts from mod-101 and does not re-derive them.
- **mod-104 (orchestration)** owns SLURM / Kueue / KubeRay. mod-106
  uses `torchrun` on top of these; the scheduler's requeue policy
  and gang semantics live in mod-104.
- **mod-105 (networking + storage)** owns the fabric your DCP shards
  actually travel on, the parallel filesystem the shards land in,
  and the on-call runbook for NCCL timeouts caused by the *fabric*.
  mod-106 owns the on-call runbook for NCCL timeouts caused by
  *the trainer*.
- **mod-107 (MFU)** owns the `throughput` factor in `goodput`;
  mod-106 owns `availability` and `retention`.
- **mod-108 (observability)** owns the dashboards you look at
  during an incident; mod-106 owns which signals belong on those
  dashboards and what alerts they raise.
- **mod-109 (cost)** owns the dollars the availability budget maps
  onto; mod-106 owns the availability budget.
- **mod-110 (platform)** owns how the recovery story is exposed to
  users and how it composes with multi-team scheduling.

If you find yourself writing about kernel throughput or MFU inside
mod-106, stop — that's mod-107. If you find yourself writing about
switch counters, stop — that's mod-105. mod-106 is about the *loop*
that turns a stream of jobs into one run.

## Map of the module

- **Chapter 2** introduces PyTorch Distributed Checkpoint (DCP): what
  it is, how the planner + storage-writer split works, and why the
  on-wire format is world-size-agnostic.
- **Chapter 3** adds `async_save` (checkpoint I/O hidden behind
  compute), resharding across topologies, and the stateful data
  loader that lets you resume without replaying batches.
- **Chapter 4** covers `torchrun` and `torch.distributed.elastic`:
  rendezvous, `min_nodes` / `max_nodes`, and the elastic-reshape
  flow that survives a mid-epoch node crash.
- **Chapter 5** is the on-call runbook: five incident classes with
  signatures, diagnostics, and recovery playbooks tied to specific
  OPT-175B logbook entries.
- **Chapter 6** covers straggler detection and silent-data-corruption
  detection — the two failure classes that do *not* announce
  themselves and therefore need active instrumentation.
- **Chapter 7** derives goodput SLOs and availability budgets from
  business targets and reads the OPT-175B logbook and Llama 3
  §6.3 as platform-engineer case studies.

Exercises 01–05 in the `exercises/` directory march in the same
order; do them in sequence when possible.

## Summary

- A large training run is not a single job; it is a stream of jobs
  stitched together by checkpoints. The stitching is the platform.
- Goodput = throughput × availability × retention. The rest of this
  module is about the last two factors; mod-107 owns the first.
- MTBF at scale is dominated by node count. On 2048 nodes, expect
  an incident every few hours — Llama 3 §6.3 measured 419 in 54
  days, and OPT-175B's logbook shows the same pattern qualitatively.
- Availability budgets are an SLO exercise. You derive a checkpoint
  interval and a recovery cost from a business-level ship-date
  target; chapter 7 does the arithmetic.
- The recovery ladder has cheap rungs (resume in place, elastic
  reshape) and expensive ones (rewind, full restart). Playbooks
  exist to keep incidents at the top of the ladder.
- mod-106 owns availability + retention; it explicitly does not own
  fabric (mod-105), MFU (mod-107), dashboards (mod-108), or cost
  (mod-109). Read the boundaries so you know which module a given
  question belongs to.
