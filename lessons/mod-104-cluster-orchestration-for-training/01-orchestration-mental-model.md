# The Training-Job Orchestration Mental Model

mod-101 taught you what a distributed-training job *is* — ranks, world size,
process groups, collectives. This module teaches you how such a job actually
gets onto a shared cluster, survives contention with hundreds of other jobs,
and comes up in the right topology so NCCL will run at line rate. Every
subsequent chapter is a variation on the same three concerns, so it is
worth naming them up front.

## What "orchestrating a training job" actually means

Compared with a web service, a distributed-training job has four defining
properties that shape every scheduler decision:

1. **It is a gang.** All N workers must be co-scheduled and running before
   any of them makes progress. A rank-0 that starts 30 seconds ahead of the
   other ranks is not "warming up" — it is holding NCCL open on
   `init_process_group` and burning quota.
2. **It is topology-sensitive.** Two ranks on the same NVLink island cost
   ~10× less to talk to each other than two ranks on different racks. The
   scheduler that hands you nodes decides your effective bandwidth before you
   ever call `all_reduce`.
3. **It is long-running and elastic-ish.** A run is minutes to weeks, not
   milliseconds. Losing a worker mid-step is a checkpoint-restart event,
   not an auto-heal. (mod-106 covers the elastic story; this module owns
   the scheduling side.)
4. **It shares the cluster.** Very few shops give a single team a private
   supercomputer. Fair-share, quota, and preemption policies decide who
   gets GPUs when — and are the single most common source of "the training
   job did not start" incidents.

Whether you are on SLURM, Kubernetes + Kueue, Kubernetes + Volcano, KubeRay,
or a scheduler-agnostic launcher, you are solving these four problems.

## The three-layer stack

Every training launcher in this module stacks the same three responsibilities.
Naming them makes the differences between SLURM, Kueue, Volcano, KubeRay, and
TorchX easier to compare:

- **Placement / gang admission** — the scheduler decides *which* nodes the
  job runs on, in what topology, and only starts the job when all workers
  can run at once. Examples: SLURM's backfill scheduler + `--exclusive`;
  Volcano's PodGroup gang plugin; Kueue's `Workload` admission with a
  `PodSet` per role; KubeRay's placement group.
- **Process launch and rendezvous** — once the nodes are held, something has
  to start N Python processes on them, teach each one its rank / world
  size, and stand up a rendezvous point (torchrun's c10d store, MPI's
  `mpirun`, Ray's head node). Examples: `srun`, `mpirun`, `torchrun`,
  `ray start`.
- **Lifecycle and observability** — the platform tracks the job through
  submission → queued → admitted → running → completed/failed, exposes
  logs, propagates cancellations, and re-queues on preemption.

Skim the primary docs — the SLURM Quick Start User Guide
(https://slurm.schedmd.com/quickstart.html), the Kueue concepts docs
(https://kueue.sigs.k8s.io/docs/concepts/), the Volcano architecture guide
(https://volcano.sh/en/docs/), and the KubeRay overview
(https://docs.ray.io/en/latest/cluster/kubernetes/index.html) — and you
will notice each of them names those three layers explicitly, just with
slightly different vocabulary.

## Two families of scheduler: batch-first and pod-first

The schedulers this module covers cluster into two families with meaningfully
different DNAs.

**Batch-first (SLURM):** designed for HPC. Nodes are the unit of allocation;
gang scheduling and topology-aware placement are the default; the shell
process is the workload. Everything else (containers, quotas, accounting)
was added later on top of a mature scheduler. The strong points are gang
scheduling, backfill, topology plugins, and Reservations. The weak points
are container ergonomics and integration with the rest of the platform
(image registries, secrets, service discovery).

**Pod-first (Kubernetes + Kueue / Volcano / KubeRay):** designed for
long-lived services, with batch bolted on. Pods are the unit; gang
scheduling and quotas are add-ons that you *must* install (Kueue or Volcano
or both) or the default scheduler will happily start half your workers.
The strong points are container ergonomics, RBAC, secrets, network
policies, autoscaling, and CI/CD alignment. The weak points are that you
must configure gang scheduling yourself and topology-aware placement is
still maturing (TopologyAwareScheduling in Kueue is a recent addition;
see https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/).

Neither family is strictly better. Most shops that run at ≥ 1000 GPUs run
both — SLURM on the research supercomputer, Kubernetes on the shared
training platform. The rest of this module is the tour of the pieces you
need in each.

## What the researcher sees vs. what you build

A researcher wants exactly one command:

```bash
$ mytrain launch configs/llama-3-8b.yaml --gpus 128
```

Under the hood that command has to:

- Pick a scheduler (SLURM? K8s? which cluster?).
- Turn the config into an sbatch script or a Kubernetes manifest.
- Reserve N GPUs of the right SKU with the right topology.
- Wait for gang admission.
- Launch `torchrun` (or `ray start`, or `mpirun`) on each worker with the
  right rank / world-size / rendezvous config.
- Stream logs somewhere the researcher can watch them.
- Re-queue on preemption, up to some retry budget.
- Clean up on cancel.

Chapters 2–7 unpack each mechanism you need to build that command. Chapter
8 is the multi-tenant policy those launches live inside. Chapter 9 is
the SDK / abstraction layer that hides all of it. If you can build the
version of that CLI that works on both SLURM and Kubernetes and picks the
right scheduler per job, you have the core artifact this module trains.

## Summary

- Training jobs differ from services in four ways — gang, topology,
  duration, and multi-tenant sharing — and every scheduler decision in this
  module addresses one of those.
- Every launcher, no matter how different its surface, decomposes into
  placement + launch + lifecycle.
- Batch-first (SLURM) and pod-first (Kubernetes + Kueue / Volcano /
  KubeRay) schedulers have different strengths; production platforms
  usually run both.
- The learning target is the launcher CLI that hides all of this and
  makes the researcher's day-one experience one command.
