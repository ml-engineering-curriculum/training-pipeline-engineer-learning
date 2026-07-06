# mod-104 — Cluster Orchestration for Training Jobs: SLURM, Kubernetes (Kueue / Volcano / KubeRay / MPI Operator), and Multi-Scheduler Launchers

**Estimated effort:** 20 hours

This module teaches you to actually run distributed training jobs on
shared clusters — both the batch-first (SLURM) and pod-first
(Kubernetes with Kueue, Volcano, MPI Operator / Kubeflow Training
Operator, and KubeRay) stacks. The through-line is the same in every
chapter: gang admission, gang scheduling, topology-aware placement,
and a launcher SDK that hides the scheduler diversity behind one API.

By the end you should be able to (a) submit a hardened multi-node
`torchrun` job on SLURM with the right sbatch topology and
prolog/epilog hooks, (b) design a Kubernetes training queue with
Kueue + Volcano + PyTorchJob (or MPIJob or RayJob), (c) reason about
when to reach for KubeRay vs. PyTorchJob, (d) use TorchX to submit
the same job to more than one scheduler, (e) design a multi-tenant
quota / fair-share / preemption policy, and (f) build the internal
launcher SDK that makes all of that one command for a researcher.

## Learning objectives

- Launch a multi-node training job on SLURM with a scheduled
  Reservation, correct sbatch topology, and prolog/epilog hardening.
- Design a Kubernetes training queue with Kueue (workload queue) or
  Volcano (gang scheduler) + MPI Operator for tightly-coupled jobs.
- Run Ray Train on KubeRay with autoscaling worker groups and
  gang-scheduled workers.
- Use TorchX / torchrun for scheduler-agnostic launch and reason
  about when to pick which launcher.
- Design quotas, fair-share, gang preemption, and priority classes
  for a shared training cluster.
- Author a launcher SDK that hides scheduler differences behind a
  single API for researchers.

## Chapters

1. [The Training-Job Orchestration Mental Model](01-orchestration-mental-model.md) —
   what "orchestrating a training job" actually means (gang, topology,
   duration, sharing), the three-layer stack every launcher stacks
   (placement / launch / lifecycle), and the batch-first vs. pod-first
   scheduler families.
2. [SLURM for Multi-Node Training: sbatch, srun, Reservations,
   Prolog/Epilog](02-slurm-for-training.md) — the sbatch pattern for
   `torchrun`, topology (`--switches`, `--distribution`,
   `--cpu-bind`, `--gpu-bind`), Reservations, and the prolog/epilog
   hooks that keep a bad node out of a run.
3. [Kueue: Workload Queueing on Kubernetes](03-kueue-workload-queueing.md) —
   ResourceFlavor / ClusterQueue / LocalQueue / Workload, admission
   as gang gate, cohort borrowing, topology-aware scheduling.
4. [Volcano: Gang Scheduling on Kubernetes](04-volcano-gang-scheduling.md) —
   PodGroups, `schedulerName: volcano`, the difference between Kueue's
   admission gang and Volcano's pod-scheduling gang, and how they
   compose.
5. [The MPI Operator and the Kubeflow Training Operator](05-mpi-operator-and-training-operator.md) —
   PyTorchJob vs. MPIJob, when to pick each, and how they wire into
   Kueue + Volcano.
6. [KubeRay and Ray Train](06-kuberay-and-ray-train.md) — RayCluster /
   RayJob, autoscaling worker groups, why the training gang should
   *not* autoscale, and composition with Kueue and Volcano.
7. [TorchX and torchrun: Scheduler-Agnostic Launch](07-torchx-and-scheduler-agnostic-launch.md) —
   `torchrun` as a process launcher, TorchX as a job submitter with
   one `AppDef` and many backends, and how to decide when to adopt
   TorchX vs. build your own launcher.
8. [Quotas, Fair-Share, Priority, Gang Preemption](08-quotas-fairshare-preemption-policy.md) —
   the four levers (quota, fair-share, priority, preemption) and how
   to compose them into a real training-cluster policy on both SLURM
   and Kubernetes.
9. [Designing the Launcher SDK](09-launcher-sdk.md) — the capstone.
   `Job` / `SchedulerBackend` / `Handle` / `ClusterRouter`, the
   pure-until-`submit()` submission pipeline, the ledger for
   reproducibility, and the failure-mode contract the SDK owns
   once so no researcher has to.

## Exercises

- [exercise-01 — SLURM multi-node sbatch hardening](exercises/exercise-01-slurm-multi-node-sbatch-hardening.md) (4 h)
- [exercise-02 — Kueue vs. Volcano queue design](exercises/exercise-02-kueue-vs-volcano-queue-design.md) (4 h)
- [exercise-03 — KubeRay autoscaling training cluster](exercises/exercise-03-kuberay-autoscaling-training-cluster.md) (4 h)
- [exercise-04 — TorchX multi-scheduler launch](exercises/exercise-04-torchx-multi-scheduler-launch.md) (3 h)
- [exercise-05 — Quota, preemption, fair-share policy](exercises/exercise-05-quota-preemption-fair-share-policy.md) (3 h)

## Labs and quizzes

- `labs/` — a `lab-01` end-to-end multi-scheduler launcher build
  lands here on the next autonomous cycle.
- `quizzes/` — one knowledge check lands here on the next autonomous
  cycle.

## Resources

- [resources.md](resources.md) — primary docs (SLURM, Kueue, Volcano,
  Kubeflow Training Operator, KubeRay, TorchX), the two fair-share
  papers (DRF and SLURM multi-factor), and a recommended reading order.

## How the module fits together

Chapter 1 fixes the vocabulary. Chapter 2 gives you the SLURM side
end-to-end. Chapters 3–6 give you the Kubernetes side end-to-end,
from workload queueing (Kueue) through pod-scheduling gangs
(Volcano) to the actual job CRDs (PyTorchJob, MPIJob, RayJob).
Chapter 7 makes the launch layer scheduler-agnostic. Chapter 8 makes
the multi-tenant policy concrete. Chapter 9 pulls everything into
one SDK. The exercises march roughly in the same order — do them in
sequence if you can; each one leaves you with an artifact the next
one builds on.

## What this module deliberately does not cover

- **Distributed-training semantics** (DDP, FSDP2, 3D-parallel) —
  owned by mod-101.
- **Training framework internals** (Megatron, DeepSpeed, torchtitan
  configs) — owned by mod-102.
- **Data pipelines and staging fabrics** — owned by mod-103.
- **Network and storage fabric depth** (NCCL fabric tuning, Lustre /
  WEKA, GPUDirect Storage) — owned by mod-105.
- **Elastic training, distributed checkpointing, fault tolerance** —
  owned by mod-106. This module *composes with* the checkpointing
  story (see chapter 8's preemption discussion) but does not own it.
- **Throughput / MFU engineering** — owned by mod-107.
- **Observability and reproducibility of training runs** — owned by
  mod-108. This module names the ledger and the artifacts a launcher
  SDK should record (chapter 9); mod-108 goes deep on the
  observability side.
- **Cluster economics and capacity planning** — owned by mod-109.
- **Platform architecture and leadership** — owned by mod-110.
