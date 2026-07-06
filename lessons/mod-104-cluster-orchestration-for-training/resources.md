# Resources for mod-104 — Cluster Orchestration for Training Jobs

Primary documentation first, papers second. This module is heavier on
"read the docs" than mod-101 was, because the systems involved (SLURM,
Kueue, Volcano, KubeRay, TorchX, Kubeflow) are documented projects with
active release cycles. Version-pin against your cluster's installed
release when you go to production; API surfaces move.

## SLURM

- **SLURM documentation index.** https://slurm.schedmd.com/documentation.html
  — the top-level. Bookmark and skim the sidebar.
- **SLURM Quick Start User Guide.** https://slurm.schedmd.com/quickstart.html
  — the first read for chapter 2.
- **SLURM Quick Start Administrator Guide.**
  https://slurm.schedmd.com/quickstart_admin.html — required if you
  also run a cluster.
- **`sbatch` man page.** https://slurm.schedmd.com/sbatch.html — the
  flag reference. Every flag in chapter 2's skeleton comes from here.
- **`srun` man page.** https://slurm.schedmd.com/srun.html — job-step
  launch, `--cpu-bind`, `--gpu-bind`, `--distribution`.
- **SLURM Topology Guide.** https://slurm.schedmd.com/topology.html —
  `topology/tree` and `topology/block`, `topology.conf`, `--switches`.
- **SLURM Reservations.** https://slurm.schedmd.com/reservations.html
  — the reference for chapter 2's Reservations section.
- **SLURM Prolog and Epilog Guide.** https://slurm.schedmd.com/prolog_epilog.html
  — the taxonomy of prolog / epilog / prolog_slurmctld hooks and their
  semantics.
- **SLURM cgroup plugin.** https://slurm.schedmd.com/cgroups.html —
  how SLURM enforces limits per job.
- **SLURM Preemption.** https://slurm.schedmd.com/preempt.html — the
  reference for chapter 8's SLURM preemption discussion.
- **SLURM Multi-Factor Priority.** https://slurm.schedmd.com/priority_multifactor.html
  — fair-share, QoS, age, job size, and the decay window.

## Kubernetes batch: Kueue, Volcano, Kubeflow

- **Kueue.** https://kueue.sigs.k8s.io/ — the docs site. Chapters 3
  and 8 lean on these.
  - Concepts: https://kueue.sigs.k8s.io/docs/concepts/
  - PyTorchJob integration:
    https://kueue.sigs.k8s.io/docs/tasks/run/kubeflow/pytorchjob/
  - RayJob integration: https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/
  - Topology Aware Scheduling:
    https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/
  - Preemption: https://kueue.sigs.k8s.io/docs/concepts/preemption/
- **Volcano.** https://volcano.sh/ — CNCF batch scheduler.
  - Docs: https://volcano.sh/en/docs/
  - PodGroup: https://volcano.sh/en/docs/podgroup/
  - Scheduler plugins (proportion, drf, gang, preempt):
    https://volcano.sh/en/docs/plugins/
  - Kubeflow integration: https://volcano.sh/en/docs/pytorch_on_volcano/
- **Kubeflow Training Operator (v1).**
  https://www.kubeflow.org/docs/components/training/ — PyTorchJob,
  TFJob, MPIJob, MXNetJob, XGBoostJob, PaddleJob.
  - PyTorchJob:
    https://www.kubeflow.org/docs/components/training/user-guides/pytorch/
- **Kubeflow Training Operator v2.**
  https://github.com/kubeflow/training-operator — the rewrite that
  consolidates the CRDs into `TrainJob` + `TrainingRuntime`.
- **MPI Operator.** https://github.com/kubeflow/mpi-operator —
  independent lifecycle of the MPI-specific operator.
- **Kubernetes Scheduler Plugins — Coscheduling.**
  https://github.com/kubernetes-sigs/scheduler-plugins/tree/master/pkg/coscheduling
  — the SIG-Scheduling gang plugin as an alternative to Volcano.

## KubeRay and Ray Train

- **KubeRay.** https://ray-project.github.io/kuberay/ — the operator
  home.
  - Overview: https://docs.ray.io/en/latest/cluster/kubernetes/index.html
  - Configuring autoscaling:
    https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/configuring-autoscaling.html
- **Ray docs index.** https://docs.ray.io/en/latest/
  - Ray Train: https://docs.ray.io/en/latest/train/train.html
  - Ray Train `TorchTrainer` API:
    https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html
  - Ray Tune: https://docs.ray.io/en/latest/tune/index.html
  - Ray placement groups (gang semantics inside Ray):
    https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html

## PyTorch launch: torchrun and TorchX

- **`torchrun` and elastic runtime.**
  https://pytorch.org/docs/stable/elastic/run.html
- **`torch.distributed` overview.**
  https://pytorch.org/docs/stable/distributed.html
- **TorchX.** https://pytorch.org/torchx/latest/
  - Quickstart: https://pytorch.org/torchx/latest/quickstart.html
  - Schedulers: https://pytorch.org/torchx/latest/schedulers/
  - Distributed component:
    https://pytorch.org/torchx/latest/components/distributed.html
- **`torch.distributed.elastic.rendezvous.c10d`.**
  https://pytorch.org/docs/stable/elastic/rendezvous.html — the
  reference for `--rdzv-backend=c10d` and its knobs.

## NCCL and topology (referenced from chapter 2 and 6)

- **NVIDIA NCCL documentation.** https://docs.nvidia.com/deeplearning/nccl/
- **`nccl-tests`.** https://github.com/NVIDIA/nccl-tests

## Papers — scheduling and fairness

- **Ghodsi, A., Zaharia, M., Hindman, B., Konwinski, A., Shenker, S.,
  & Stoica, I. (2011). "Dominant Resource Fairness: Fair Allocation of
  Multiple Resource Types."** *NSDI '11.*
  https://www.usenix.org/legacy/event/nsdi11/tech/full_papers/Ghodsi.pdf
  — the DRF paper. The theoretical basis for Volcano's `drf` plugin
  and for reasoning about multi-resource fair-share more generally.
- **Yoo, A. B., Jette, M. A., & Grondona, M. (2003). "SLURM: Simple
  Linux Utility for Resource Management."** *JSSPP '03.* The original
  SLURM paper. Historical context, not required for day-to-day work.

## Recommended reading order

1. Chapter 1 (this module) + the SLURM Quick Start User Guide (1 h).
2. Chapter 2 + `sbatch(1)` + `srun(1)` + SLURM Topology (2 h).
3. Chapter 3 + Kueue Concepts (1.5 h).
4. Chapter 4 + Volcano PodGroup + Volcano scheduler plugins (1.5 h).
5. Chapter 5 + Kueue's PyTorchJob integration + Kubeflow PyTorchJob
   user guide (1 h).
6. Chapter 6 + KubeRay Quickstart + Ray Train intro (1.5 h).
7. Chapter 7 + `torchrun` docs + TorchX Quickstart (1 h).
8. Chapter 8 + SLURM Preemption + SLURM Multi-Factor Priority +
   Kueue Preemption + DRF paper (2 h).
9. Chapter 9 (30 min).

The exercises assume you have read at least the chapter they map to
and skimmed the linked primary docs for that chapter.
