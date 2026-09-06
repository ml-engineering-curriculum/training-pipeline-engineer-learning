# mod-106 — Checkpointing, Fault Tolerance, and Elastic Training

**Estimated effort:** 18 hours

mod-101 through mod-105 taught you how to make a training job go
*fast*. This module is about making it not stop. At the scale these
clusters run at, "runs to completion without operator intervention"
stops being the default and becomes a designed property of the
platform. If you skip that design work, your GPUs still spin, but
the tokens they produce do not accumulate — a failed step is a step
you paid for and got nothing back for.

By the end of the module you should be able to (a) implement a
PyTorch Distributed Checkpoint (DCP) save/load with async save, a
resharding load onto a different world size, and a stateful sampler;
(b) design a `torchrun` rendezvous + elastic reshape flow that
survives a node crash mid-epoch; (c) classify the five canonical
incident classes (loss spike, NaN, NCCL timeout, hardware fault,
silent corruption) and route each to the correct playbook; (d)
instrument straggler and silent-corruption detection and wire it
into a node-quarantine automation; (e) read the OPT-175B chronicles
and the Llama 3 §3.3.2 numbers and translate them into on-call
runbooks; and (f) author a goodput SLO with a derived availability
budget the platform team is measured on.

## Learning objectives

- Implement a PyTorch Distributed Checkpoint (DCP) save/load with
  async save, resharding across a different world size, and a
  stateful sampler.
- Design a torchrun rendezvous + elastic reshape flow that survives
  a node crash mid-epoch.
- Classify training-run incidents (loss spike, NaN, NCCL timeout,
  hardware fault, silent corruption) and codify recovery playbooks.
- Instrument straggler detection, silent-data-corruption detection,
  and automatic node quarantine.
- Read the OPT-175B logbook and Llama 3 failure statistics and
  translate them into on-call runbooks.
- Author an SLO for "goodput" (useful training tokens per
  wall-clock hour) and derive an availability budget from it.

## Chapters

1. [Fault Tolerance as a Scale Problem](01-why-fault-tolerance-is-a-scale-problem.md) —
   why individual reliability does not compose to cluster
   reliability, the two design numbers (per-job MTBF and recovery
   time), goodput vs. throughput vs. uptime, and the three failure
   classes (fail-stop, fail-slow, silent corruption) that the rest
   of the module treats separately.
2. [Checkpoint Anatomy and the Storage-Cost
   Model](02-checkpoint-anatomy-and-cost-model.md) — what has to be
   in a correct checkpoint (model, optimizer, LR scheduler, step,
   per-rank RNG, sampler, scaler, EMA), the save- and load-cost
   models, the failure-window formula for the optimal save
   interval, and the two anti-patterns to eliminate before scaling.
3. [PyTorch Distributed Checkpoint (DCP)](03-pytorch-distributed-checkpoint-dcp.md) —
   the DCP data model, the `Stateful` protocol and `AppState`
   pattern, `dcp.async_save` and the future-based completion
   contract, the resharding load algorithm, the atomic-rename
   pattern, and a stateful sampler wired through DCP.
4. [torchrun, Rendezvous, and Elastic
   Training](04-torchrun-rendezvous-and-elastic-training.md) — the
   three actors (rendezvous backend, agent, workers), static vs.
   elastic rendezvous, the `c10d` vs. `etcd-v2` trade-off, the
   `MIN:MAX` semantics, the re-entrant training-script contract,
   and the NCCL-timeout + watchdog knobs that shorten detection.
5. [Classifying Training Incidents and Codifying
   Playbooks](05-incident-classification-and-playbooks.md) — the
   five incident classes (loss spike, NaN, NCCL timeout, hardware
   fault, silent corruption), the "symptom → tools → likely causes
   → decision → post-mortem" template, and the per-class decision
   line that routes to the recovery mechanisms from chapters 3-4.
6. [Straggler Detection, Silent-Corruption Detection, and Node
   Quarantine](06-straggler-and-silent-corruption-detection.md) —
   per-step-time dispersion, cross-rank loss agreement, DCGM
   compute-health checks, the green/yellow/orange/red state
   machine, the quarantine action, and the deterministic-replay
   drill for silent corruption.
7. [Lessons from OPT-175B and Llama 3: Turning Logbooks Into
   Runbooks](07-lessons-from-opt-175b-and-llama-3.md) — the
   OPT-175B chronicles and Llama 3 §3.3.2 as primary sources, the
   published incident rates and taxonomies, and the mapping from
   each logbook lesson to a runbook entry. Introduces the daily
   logbook and scheduled recovery-path drill patterns.
8. [Goodput SLO and the Availability
   Budget](08-goodput-slo-and-availability-budget.md) — the SRE
   SLO framing applied to training, the goodput SLI definition,
   the derivation of `G_max` / `G*` / the availability budget,
   the three-slice split (unplanned / planned / recovery), and
   the error-budget policy bands.

## Exercises

- [exercise-01 — DCP async save, stateful sampler, and reshard](exercises/exercise-01-dcp-async-save-and-reshard.md) (4 h)
- [exercise-02 — torchrun elastic reshape drill](exercises/exercise-02-torchrun-elastic-reshape-drill.md) (3 h)
- [exercise-03 — incident classification playbooks](exercises/exercise-03-incident-classification-playbooks.md) (3 h)
- [exercise-04 — straggler and silent-corruption detection](exercises/exercise-04-straggler-and-silent-corruption-detection.md) (4 h)
- [exercise-05 — goodput SLO and availability budget](exercises/exercise-05-goodput-slo-and-availability-budget.md) (2 h)

## Labs and quizzes

- `labs/` — a long-form recovery-path drill lab lands here on a
  subsequent autonomous cycle.
- `quizzes/` — one knowledge check lands here on a subsequent
  autonomous cycle.

## Resources

- [resources.md](resources.md) — primary PyTorch DCP / torchrun /
  elastic docs, the OPT-175B chronicles and Llama 3 / PaLM / BLOOM
  papers, the DCGM + Xid references for chapter 6, the SDC papers,
  and the SRE Book chapter that chapter 8 is grounded in.

## How the module fits together

Chapter 1 fixes the vocabulary (goodput, three failure classes) and
the two design numbers (MTBF and recovery time) whose product sets
the ceiling everything else optimizes against. Chapter 2 is the
correctness-and-cost model the rest of the module lives inside.
Chapters 3 and 4 are the two mechanisms — DCP for state, elastic
`torchrun` for membership — that together give you a "the run
continues" path after a fail-stop event. Chapters 5 and 6 add the
non-fail-stop paths: chapter 5 is the incident taxonomy and the
decision framing, chapter 6 is the detection layer for the two
classes (fail-slow, silent corruption) where detection is the
dominant investment. Chapter 7 grounds all of the above in the two
published primary sources (OPT chronicles, Llama 3 §3.3.2) so the
design is anchored to empirical rates rather than intuition.
Chapter 8 is the objective: the goodput SLO the whole platform is
measured on, with the availability budget derived from it. The
exercises march in the same order — 1 builds on 2/3, 2 builds on
1/3/4, 3 codifies 5/7, 4 codifies 6, and 5 lands the SLO from 8.

## What this module deliberately does not cover

- **Distributed-training semantics** (DDP, FSDP2, 3D-parallel) —
  owned by mod-101. This module *uses* the process-group
  abstraction to place checkpoint I/O and rendezvous.
- **Framework internals** (Megatron, DeepSpeed, torchtitan) —
  owned by mod-102. torchtitan's checkpoint helpers wrap DCP;
  DeepSpeed has its own checkpoint format; this module treats
  them as consumers of the same underlying storage tier.
- **The fabric itself** — owned by mod-105. NCCL timeouts are
  covered here as a *symptom*; the diagnosis of *why* NCCL timed
  out (a NIC drop, a PXN misconfiguration) lives in mod-105
  chapter 8.
- **Cluster orchestration and scheduling** — owned by mod-104.
  This module runs on top of a scheduler that has already granted
  a gang; what happens when the gang shrinks mid-run is chapter
  4's problem.
- **Observability and metrics collection plumbing** — owned by
  mod-108. This module *emits* the counters (per-step time
  histograms, ECC delta counters, goodput ratio) that mod-108
  turns into dashboards.
- **Cost accounting on the recovered-vs-lost token axis** — owned
  by mod-109. Chapter 8's SLO is the input; mod-109 turns it into
  a dollar number.
