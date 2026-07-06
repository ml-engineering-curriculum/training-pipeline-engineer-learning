# Volcano: Gang Scheduling on Kubernetes

Volcano (https://volcano.sh/) is a CNCF batch-first scheduler that
replaces (or coexists with) the default Kubernetes scheduler for
workloads that need real gang semantics at the pod-scheduling layer.
Where Kueue is a queue *in front of* jobs, Volcano is a scheduler
*underneath* pods. It is the piece most large training platforms use
when "all N workers must land together or none do" is a correctness
requirement, not a nicety.

This chapter is the working knowledge you need to reason about when to
reach for Volcano, what a PodGroup guarantees, and how it composes with
Kueue and the Kubeflow Training Operator.

## Motivation: what "gang scheduling" actually buys you

Suppose you submit a 16-pod tightly-coupled `MPIJob`. The default
Kubernetes scheduler processes pods one at a time and does not know
they belong together. It will happily schedule 15 pods, run out of
GPUs on the 16th, and leave your job wedged: 15 pods running (and
holding NCCL open on `init_process_group`), 1 pod pending, no
progress, GPU-hours burning.

A **gang scheduler** treats the whole set as a scheduling unit. If it
cannot admit all N pods, it admits *none* of them, keeps the set
queued, and re-evaluates when capacity frees up. That is the
correctness property Volcano gives you.

Two related properties come free:

- **Gang preemption.** A higher-priority gang can preempt a
  lower-priority gang; the low-priority pods are all evicted together,
  which means the low-priority job is cleanly re-queueable (mod-106
  makes the checkpointing side concrete).
- **Fair-share and DRF within a queue.** Volcano ships with
  `proportion`, `drf`, `priority`, and `binpack` scheduler plugins you
  can compose in `volcano-scheduler.conf`.

## Volcano's data model: Job, PodGroup, Queue

- **`vcjob` (Volcano Job)** — Volcano's native batch-job CRD. Has
  `tasks[]`, each with a replica count, a pod template, and an optional
  `minAvailable`. You do not have to use it — many people use Kubeflow's
  PyTorchJob / MPIJob and let Volcano schedule the pods — but the
  vocabulary lives here.
- **`PodGroup`** — the gang-scheduling primitive. A PodGroup carries a
  `minMember` (the size of the smallest set that counts as scheduled),
  a `queue`, a `priorityClassName`, and a `minResources` block. Every
  pod that belongs to the gang has to reference the PodGroup via the
  `scheduling.k8s.io/group-name` annotation.
- **`Queue`** — Volcano's queue object. Holds capability quotas, a
  weight for fair-share, and reclaim policy. Different from Kueue's
  `LocalQueue` — Volcano's Queue is cluster-scoped and is where DRF /
  proportion plugins do their math.

The important claim: PodGroup semantics are *strict*. Volcano's
scheduler will not bind any pod in a PodGroup until it has confirmed
that `minMember` pods can all bind. If it cannot, it re-queues and
retries.

## A worked example: gang-scheduled MPIJob

The idiomatic pattern for tightly-coupled training on Kubernetes today is
Kubeflow's Training Operator (`MPIJob` or `PyTorchJob`) with Volcano as
the pod scheduler:

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  namespace: alice-research
  name: llama-3-8b-pg
spec:
  minMember: 8                              # 8 workers, all-or-nothing
  queue: training
  priorityClassName: training-normal
  minResources:
    nvidia.com/gpu: 64
    cpu: 128
    memory: 1600Gi
```

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  namespace: alice-research
  name: llama-3-8b
spec:
  slotsPerWorker: 8
  runPolicy:
    schedulerName: volcano                  # <-- ask Volcano to schedule
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        metadata:
          annotations:
            scheduling.k8s.io/group-name: llama-3-8b-pg
        spec:
          schedulerName: volcano
          containers:
            - name: mpi-launcher
              image: registry/train-mpi:v2026-07
              command: ["/entrypoint.sh"]
    Worker:
      replicas: 8
      template:
        metadata:
          annotations:
            scheduling.k8s.io/group-name: llama-3-8b-pg
        spec:
          schedulerName: volcano
          containers:
            - name: mpi-worker
              image: registry/train-mpi:v2026-07
              resources:
                limits:
                  nvidia.com/gpu: 8
                  cpu: 16
                  memory: 200Gi
```

Notes:

- Every pod carries `scheduling.k8s.io/group-name` **and** sets
  `schedulerName: volcano`. Both are required; Volcano ignores pods that
  are still using `default-scheduler`.
- `minMember: 8` names how many workers must land together. If your
  launcher counts, set it to launcher + workers.
- `minResources` is Volcano's admission bookkeeping — the aggregate
  request across the gang. Get it right or `proportion` / `drf`
  plugins will double-count.

## Interaction with Kueue: queue vs. schedule, and who owns what

Kueue and Volcano solve *different* problems and are commonly deployed
together on the same cluster.

- Kueue owns the **queue layer**: whose quota, which flavor, when to
  admit. It flips the source job from suspended to running.
- Volcano owns the **pod-scheduling layer**: once the job is running,
  which nodes each pod binds to and the gang guarantee that all pods
  bind together.

There is one integration wrinkle worth knowing: Kueue's admission
already reserves quota, but if you then also set `Queue` on the
PodGroup, Volcano is doing a *second* round of quota accounting on the
same pods. You have to pick one owner for the quota:

- **Kueue-primary layout:** Kueue holds the quota; Volcano's Queue has
  no capability (or a very large one) and Volcano is used only for its
  gang scheduler.
- **Volcano-primary layout:** Volcano holds the quota and DRF; Kueue is
  not installed.
- **Split layout:** Kueue holds tenant quota; Volcano's Queue is used
  for cluster-wide preemption between priority tiers.

The first layout is what most Kueue + Volcano installations do in
practice. Read Kueue's Volcano integration docs
(https://kueue.sigs.k8s.io/docs/tasks/) and the Volcano PodGroup docs
(https://volcano.sh/en/docs/podgroup/) before you commit to a topology.

## When you want Volcano vs. only Kueue

Reach for Volcano when:

- You run tightly-coupled MPI / NCCL / RDMA jobs where a single
  un-scheduled pod stalls the run (i.e., every training job in this
  module).
- You need preemption at pod granularity — a high-priority training
  gang should be able to evict a lower-priority one, cleanly, without
  the default scheduler racing to backfill.
- You care about DRF (Dominant Resource Fairness, Ghodsi et al., 2011,
  https://www.usenix.org/conference/nsdi11/technical-sessions/presentation/ghodsi)
  across mixed CPU/GPU/memory workloads.

Skip Volcano if:

- You are running "workload-y" batch that tolerates a slow-start pod
  (e.g., a hyperparameter search where each trial is one pod). Kueue
  alone is enough.
- You cannot install a second scheduler on the cluster and the default
  scheduler + Coscheduling plugin already gives you what you need.

## Common gotchas

- **Forgetting `schedulerName: volcano` on the pod template.** Pods
  fall through to the default scheduler and the gang guarantee is
  lost. This is by far the most common misconfiguration.
- **`minMember` set to `replicas - 1`.** Someone had an idea about
  tolerating "one missing" and forgot that NCCL will not come up with
  a missing rank. `minMember == replicas` for training gangs.
- **`minResources` drift.** If the pod template's resource requests
  drift and no one updates `minResources`, DRF's fair-share math goes
  wrong. Consider generating the manifests together.
- **Priority class not defined.** Volcano's preemption plugin honors
  `priorityClassName` on the PodGroup; if the class is missing, gangs
  are all equal-priority and preemption becomes FIFO.

## Summary

- Volcano is a pod-scheduler-level gang scheduler for Kubernetes. Its
  guarantee: no pod in a PodGroup binds until `minMember` pods can all
  bind together.
- The `PodGroup` object plus `schedulerName: volcano` on every pod is
  what turns any batch job (MPIJob, PyTorchJob, `vcjob`) into a gang-
  scheduled workload.
- Kueue + Volcano is a widely used pattern: Kueue owns quota and
  admission, Volcano owns gang scheduling and preemption. Do not run
  quota accounting on both sides at once.
- Reach for Volcano whenever an un-scheduled worker will stall the run
  (i.e., real training). Skip it for embarrassingly parallel batch.
