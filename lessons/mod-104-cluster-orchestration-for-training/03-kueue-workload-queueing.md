# Kueue: Workload Queueing on Kubernetes

The default Kubernetes scheduler is a fit-and-forget bin-packer. It has
no concept of a "job" that owns multiple pods, no notion of gang
admission, no notion of a training queue. Point a pod-first scheduler at
a `PyTorchJob` with 64 workers and, absent extra machinery, you will get
back 40 running pods, 24 pending pods, and a stuck `init_process_group`
call. Kueue is the SIG-Scheduling project that fixes this at the
*queueing* layer without replacing the default scheduler.

This chapter is the working knowledge you need to design a training
queue on Kueue: what the objects mean, how admission works, and where
gang scheduling comes from.

## Motivation: why "just add a Job" is not enough

Batch on Kubernetes has been through several eras. The `batch/v1.Job`
API handles retries and completion counts; the Kubeflow Training
Operator (formerly `tf-operator`, `pytorch-operator`) adds
`PyTorchJob` / `MPIJob` / `TFJob` CRDs that model *roles* (master /
worker) and per-role replica counts. Neither answers the harder
questions:

- Whose quota is this job spending?
- If two teams both want the same 128 GPUs, who goes first?
- What happens when a burst of jobs arrives at the same time?
- How do I preempt a lower-priority job to run a higher-priority one
  without losing the low-priority job entirely?

Kueue (https://kueue.sigs.k8s.io/) answers those questions by putting a
**Workload** admission layer in front of the batch job. Jobs are
*suspended* on creation and only unsuspended once Kueue admits their
Workload against a quota. The workload objects are the ones this chapter
teaches you to design.

## The four objects: ResourceFlavor, ClusterQueue, LocalQueue, Workload

Kueue's data model, in order of the axis it names:

- **`ResourceFlavor`** — a labelled variety of a resource. In practice
  this is your GPU SKU tier: `h100`, `a100-40g`, `spot-a10g`. A
  ResourceFlavor carries node selectors, tolerations, and taints so that
  a Workload admitted against it lands on the right hardware.
- **`ClusterQueue`** — cluster-scoped quota bucket, keyed by
  ResourceFlavor. This is where you say "team `foundations` gets 128
  H100s, team `finetune` gets 32 H100s and 128 A100-40g". A ClusterQueue
  has `resourceGroups`, each of which lists resources
  (`cpu`, `memory`, `nvidia.com/gpu`) and per-flavor quotas.
- **`LocalQueue`** — namespaced pointer to a ClusterQueue. Users submit
  Workloads to a LocalQueue in their namespace; the LocalQueue routes
  admission to the ClusterQueue.
- **`Workload`** — the queueable unit. Kueue creates one automatically
  per PyTorchJob / MPIJob / RayJob / plain `batch/v1.Job` (via the
  matching integration). Each Workload has one or more **`PodSets`**,
  one per role.

Skim the Concepts page
(https://kueue.sigs.k8s.io/docs/concepts/) once; the model is small and
the docs are the source of truth for exact field names.

## A worked example: a single-tenant training queue

Say your platform has 64 H100s and 128 A10G nodes and you want a
training-only queue that admits H100 jobs at guaranteed 32-GPU
minimum and lets an A10G job burst up to 128 GPUs opportunistically.

```yaml
# ResourceFlavors — one per hardware tier.
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: h100
spec:
  nodeLabels:
    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
  tolerations:
    - key: dedicated
      operator: Equal
      value: training-h100
      effect: NoSchedule
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: a10g-spot
spec:
  nodeLabels:
    nvidia.com/gpu.product: NVIDIA-A10G
    node.k8s.aws/lifecycle: spot
```

```yaml
# One ClusterQueue for the training team, with two flavors in one resource group.
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: training-cq
spec:
  namespaceSelector: {}                # any namespace with a LocalQueue may use it
  cohort: research                     # borrow/lend inside the `research` cohort
  queueingStrategy: BestEffortFIFO
  preemption:
    withinClusterQueue: LowerPriority
    reclaimWithinCohort: Any
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: h100
          resources:
            - name: nvidia.com/gpu
              nominalQuota: 64
              borrowingLimit: 0        # no borrowing H100s from other CQs
            - name: cpu
              nominalQuota: 128
            - name: memory
              nominalQuota: 2Ti
        - name: a10g-spot
          resources:
            - name: nvidia.com/gpu
              nominalQuota: 32
              borrowingLimit: 96       # can borrow up to 96 more from cohort
            - name: cpu
              nominalQuota: 256
            - name: memory
              nominalQuota: 4Ti
```

```yaml
# LocalQueue in the researcher's namespace.
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  namespace: alice-research
  name: default
spec:
  clusterQueue: training-cq
```

```yaml
# The researcher submits a suspended PyTorchJob pointing at the LocalQueue.
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  namespace: alice-research
  name: llama-3-8b
  labels:
    kueue.x-k8s.io/queue-name: default
spec:
  runPolicy:
    suspend: true                      # required — Kueue unsuspends on admission
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
            - name: pytorch
              image: registry/train:v2026-07
              resources:
                limits:
                  nvidia.com/gpu: 8
                  cpu: 16
                  memory: 200Gi
    Worker:
      replicas: 7
      template:
        spec:
          containers:
            - name: pytorch
              image: registry/train:v2026-07
              resources:
                limits:
                  nvidia.com/gpu: 8
                  cpu: 16
                  memory: 200Gi
```

Kueue watches PyTorchJobs (via the integration installed on the
controller-manager, see
https://kueue.sigs.k8s.io/docs/tasks/run/kubeflow/pytorchjob/), notices
that this one is suspended and labelled, creates a Workload with two
PodSets (Master ×1 and Worker ×7), and admits it against
`training-cq` when 64 H100s are free. Only then does Kueue flip
`spec.runPolicy.suspend` to `false` and let the Training Operator
create pods.

## Gang admission in Kueue

Kueue's admission is **PodSet-aware and all-or-nothing at the Workload
level**. It will not admit a Workload unless every PodSet has enough
capacity available at the same time. That is gang admission at the
queueing layer.

That is *not* the same as gang scheduling at the pod-scheduling layer.
Once Kueue admits the Workload and unsuspends the Job, individual pods
still go through the default scheduler. On a healthy cluster with the
quota reserved by admission this works — but if a node dies mid-boot,
you can end up with 63 running pods and 1 pending pod, and you are
holding NCCL open again.

Two ways to close that gap:

- Pair Kueue with the `PodGroup` gang plugin (Coscheduling KEP;
  https://github.com/kubernetes-sigs/scheduler-plugins/tree/master/pkg/coscheduling)
  so the default scheduler also insists on all-or-nothing.
- Pair Kueue with **Volcano** (next chapter) and let Volcano own the
  pod-scheduling side while Kueue owns the workload queue. Volcano's
  PodGroup is a stricter gang than Coscheduling and is widely used with
  the Kubeflow Training Operator.

Whichever you pick, "Kueue alone" is not a full gang scheduler at the
pod layer; it is a *quota-managed workload queue* that plays very well
with a pod-layer gang.

## Topology-aware placement

Kueue 0.9+ ships **Topology Aware Scheduling** (TAS), which
lets a Workload declare that the pods in a PodSet must be co-located
inside a topology domain (node, rack, block). You annotate the PodSet
with a `kueue.x-k8s.io/podset-required-topology` (or `preferred`) key
and Kueue's admission will honour the topology when it selects nodes.
See https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/
for the exact fields. TAS closes some of the topology gap SLURM's
`topology/tree` had traditionally owned.

## Common gotchas

- **Forgetting `spec.suspend: true` on the source Job.** Without it,
  the Job creates pods immediately and Kueue never gets to admit; the
  Workload sits in a permanently-not-admitted state.
- **`nominalQuota` mis-sized to the actual node capacity.** Kueue does
  not check nodes; it just tracks the abstract number. If you promise
  64 H100 GPUs and only have 56 healthy, you will admit jobs that pend
  forever.
- **Borrowing from a cohort without setting `borrowingLimit`.** By
  default Kueue is generous; set explicit limits per resource so a
  runaway job cannot drain the cohort.
- **Preemption misconfiguration.** `reclaimWithinCohort: Any` will
  preempt lower-priority jobs from *other* ClusterQueues to reclaim
  borrowed quota. This is often what you want — but understand it
  before you set it, or you will preempt a sibling team by accident.

## Summary

- Kueue is a workload-level queue in front of Kubernetes batch jobs. It
  turns "run this Job" into "admit this Workload against a quota".
- The four objects — ResourceFlavor, ClusterQueue, LocalQueue,
  Workload — cover per-SKU quota, per-team quota, per-namespace queue
  entry, and the queueable unit.
- Kueue is a *gang admitter* (all-or-nothing at admission) but not by
  itself a *gang scheduler* at the pod layer. Pair it with Volcano or
  the Coscheduling plugin for pod-layer gang guarantees.
- Topology-aware scheduling is new but real; use it when your fabric
  cares.
