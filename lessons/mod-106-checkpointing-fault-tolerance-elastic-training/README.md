# mod-106 — Checkpointing, Fault Tolerance, and Elastic Training

**Estimated effort:** 18 hours

mod-101 through mod-105 taught you how to build one training
*step* — the collectives, the framework internals, the data
loader, the scheduler that places the gang, the fabric that
carries the traffic. This module is about what happens over the
*run*: the 30, 60, or 90 days of wall clock in which every one
of those steps has to survive rank churn, host reboots,
mid-epoch node crashes, loss spikes, NaN incidents, silent
hardware corruption, and everything else that goes wrong on a
1024-GPU pretraining job. mod-106 is the layer between "we
built a step" and "the model shipped".

By the end of the module you should be able to (a) implement a
PyTorch Distributed Checkpoint (DCP) save/load with async save,
resharding across a different world size, and a stateful
sampler; (b) design a `torchrun` rendezvous + elastic reshape
flow that survives a node crash mid-epoch; (c) classify the
five canonical training-run incident classes and codify
recovery playbooks against the OPT-175B logbook; (d)
instrument straggler and silent-data-corruption detection
tied to automatic node quarantine on SLURM and Kubernetes; and
(e) author a goodput SLO from a business ship-date target and
derive the checkpoint interval and availability budget that
make it achievable.

## Learning objectives

- Implement a PyTorch Distributed Checkpoint (DCP) save/load with
  async save, resharding across a different world size, and a
  stateful sampler.
- Design a torchrun rendezvous + elastic reshape flow that
  survives a node crash mid-epoch.
- Classify training-run incidents (loss spike, NaN, NCCL timeout,
  hardware fault, silent-corruption) and codify recovery playbooks.
- Instrument straggler detection, silent-data-corruption detection,
  and automatic node quarantine.
- Read the OPT-175B logbook and Llama 3 failure statistics and
  translate them into on-call runbooks.
- Author an SLO for "goodput" (useful training tokens per
  wall-clock hour) and derive an availability budget from it.

## Chapters

1. [The Fault-Tolerance Mental Model: A Training Run Is a Stream
   of Jobs](01-fault-tolerance-mental-model-and-goodput.md) —
   why large runs are streams of jobs stitched by checkpoints,
   goodput = throughput × availability × retention, MTBF at
   scale, and the recovery ladder that keeps incidents cheap.
   Anchored on OPT-175B (Zhang et al., 2022) and Llama 3 §6.3
   (Grattafiori et al., 2024).
2. [DCP Mechanics: Save, Load, Planners, and the Storage
   Split](02-dcp-mechanics-save-load-and-planners.md) —
   `torch.distributed.checkpoint`, the SavePlanner /
   LoadPlanner design, StorageWriter / StorageReader, the
   world-size-agnostic on-disk format, and why
   `torch.save(state_dict())` breaks at any scale.
3. [Async Save, Resharding, and the Stateful Data
   Loader](03-dcp-async-save-and-resharding.md) —
   `dcp.async_save`, CPU staging, the resharding contract
   across TP/PP/FSDP topologies, and `torchdata`'s
   `StatefulDataLoader` so a resume sees the same batch
   sequence as the original run. The training-loop skeleton the
   exercises assume.
4. [torchrun, Rendezvous, and Elastic
   Reshape](04-torchrun-rendezvous-and-elastic-reshape.md) —
   `torch.distributed.elastic`, c10d vs. etcd rendezvous,
   `min_nodes` / `max_nodes`, the `on_membership_change` hook,
   and how restart-in-place vs. restart-with-fewer-nodes flows
   survive a mid-epoch node crash.
5. [Incident Classification and Recovery
   Playbooks](05-incident-classification-and-recovery-playbooks.md) —
   five canonical incident classes (loss spike, NaN / Inf,
   NCCL timeout, hardware fault, silent data corruption) with
   signature → diagnostics → containment → recovery →
   post-mortem playbooks tied to the OPT-175B logbook.
6. [Straggler and Silent-Corruption Detection, and
   Auto-Quarantine](06-straggler-and-silent-corruption-detection.md) —
   per-rank step-time histograms, all-reduce timing, DCGM SM
   occupancy, checkpoint hashing, grad-norm outliers,
   deterministic-mode re-run diff, canary eval, and the SLURM
   `scontrol drain` + Kubernetes taint/cordon quarantine loop.
7. [Goodput SLO, Availability Budget, and Case
   Studies](07-goodput-slo-availability-budget-case-studies.md) —
   deriving the SLO from a business ship-date target,
   computing the optimal checkpoint interval
   `I_opt = sqrt(2 · S · M)`, and reading the OPT-175B logbook
   and Llama 3 §6.3 as platform-engineer case studies.

## Exercises

- [exercise-01 — DCP async save and resharding drill](exercises/exercise-01-dcp-async-save-and-reshard.md) (4 h)
- [exercise-02 — torchrun elastic reshape drill](exercises/exercise-02-torchrun-elastic-reshape-drill.md) (4 h)
- [exercise-03 — Incident classification playbooks](exercises/exercise-03-incident-classification-playbooks.md) (3 h)
- [exercise-04 — Straggler and silent-corruption detection](exercises/exercise-04-straggler-and-silent-corruption-detection.md) (3 h)
- [exercise-05 — Goodput SLO and availability budget](exercises/exercise-05-goodput-slo-and-availability-budget.md) (2 h)

## Labs and quizzes

- `labs/` — `lab-01-opt175b-logbook-to-runbook`, a 2-hour
  reading + translation lab that turns the OPT-175B operational
  logbook into your own runbook entries, lands here on the next
  autonomous cycle.
- `quizzes/` — one knowledge check lands here on the next
  autonomous cycle.

## Resources

- [resources.md](resources.md) — primary papers (OPT-175B,
  BLOOM, Llama 3), DCP / elastic docs, DCGM and hardware
  health tooling, SLURM / Kubernetes / Kueue quarantine
  references, and a recommended reading order.

## How the module fits together

Chapter 1 fixes the mental model: a run is a stream of jobs,
goodput is the metric, and the recovery ladder keeps incidents
cheap. Chapters 2 and 3 give you the checkpoint mechanism —
what DCP is, how the async + staging + resharding + stateful
sampler pieces compose, and the training-loop skeleton the rest
of the module assumes. Chapter 4 wraps the loop in the elastic
launcher so a mid-epoch node crash triggers automatic reshape
and DCP resume. Chapter 5 is the on-call runbook — five
incident classes, each with a playbook you should be able to
walk through under pressure. Chapter 6 is the detection layer
that makes chapter 5's incidents auto-quarantine before a
human is paged. Chapter 7 pulls the whole module up to the SLO
altitude: given a business ship-date target, derive the
checkpoint interval, availability budget, and quarantine policy
that turn ~90% goodput into a realistic target. The exercises
march in the same order.

## What this module deliberately does not cover

- **Distributed-training semantics (DDP, FSDP2, 3D-parallel)** —
  owned by mod-101. mod-106 uses DTensor and ShardedTensor
  state-dict layouts from mod-101 and does not re-derive them.
- **Data-pipeline design** — owned by mod-103. mod-106 requires
  a resumable, shard-aware loader (`StatefulDataLoader` over a
  sharded dataset); how that loader is *built* is mod-103's
  job.
- **Scheduler and launcher plumbing (SLURM, Kueue, KubeRay,
  MPI Operator)** — owned by mod-104. mod-106 uses `torchrun`
  on top of these schedulers; requeue policy, gang scheduling,
  and topology admission live in mod-104.
- **Fabric and storage (NCCL tuning, InfiniBand vs. RoCEv2,
  parallel filesystems)** — owned by mod-105. mod-106 owns
  trainer-side NCCL-timeout diagnosis; mod-105 owns
  fabric-side NCCL-timeout diagnosis (silent NIC drop, PXN
  misconfig, congestion tree). The two runbooks compose; do
  not conflate them.
- **MFU / throughput engineering (kernels, mixed precision,
  torch.compile, comm-compute overlap)** — owned by mod-107.
  mod-106 owns the availability and retention factors of
  goodput; mod-107 owns the throughput factor.
- **Observability (Prometheus / DCGM / dashboards)** — owned by
  mod-108. mod-106 owns the alert *rules* (which signal
  triggers which response); mod-108 owns the scrapers,
  storage, and rendering.
- **Cost and capacity planning** — owned by mod-109. mod-106
  produces the availability budget; mod-109 converts it into
  dollars.
- **Platform architecture and cross-team leadership** — owned
  by mod-110. mod-106 produces the SLO; mod-110 owns how the
  SLO is exposed to model teams and negotiated across an org.
