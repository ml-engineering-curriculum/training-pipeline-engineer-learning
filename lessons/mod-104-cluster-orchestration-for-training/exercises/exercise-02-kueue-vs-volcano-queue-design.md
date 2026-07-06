# exercise-02: Kueue vs. Volcano Queue Design

**Estimated effort:** 4 hours

## Objective

Design and stand up a training-only queue on Kubernetes that supports
three tenants with per-tenant quotas, gang admission, and gang
scheduling. Produce (a) the Kueue + Volcano manifest set, (b) a
demonstration that gang admission is doing what you claim, and
(c) a short RFC-style write-up justifying the split of quota-owner vs.
gang-scheduler-owner between Kueue and Volcano.

This exercise proves you can compose the workload-queue layer and the
pod-scheduling layer correctly on the same cluster.

## Prerequisites

- Chapters 3 and 4.
- A Kubernetes cluster you can install operators on. If you do not
  have one, `kind` (https://kind.sigs.k8s.io/) with 3 GPU-simulated
  nodes is enough — replace `nvidia.com/gpu` with a fake extended
  resource (`example.com/gpu`) via a
  DaemonSet-mocked device plugin, and note the substitution in the
  RFC.
- Install:
  - Kueue (https://kueue.sigs.k8s.io/docs/installation/) with the
    PyTorchJob integration enabled.
  - Volcano (https://volcano.sh/en/docs/installation/).
  - Kubeflow Training Operator
    (https://www.kubeflow.org/docs/components/training/installation/).

## Problem statement

Three teams share a cluster with 24 (simulated) H100s:

- `foundations` — guaranteed 12, cap 20, priority tier
  `training-high` for their main run.
- `finetune` — guaranteed 8, cap 16, priority tier
  `training-normal`.
- `research` — guaranteed 4, cap 12, priority tier `research`.

Design the queue policy so:

1. Each team's guarantee is admittable within seconds when idle.
2. A team can burst above its guarantee up to its cap by borrowing
   from the cohort, and the borrow is reclaimable when the lender
   wants its guarantee back.
3. A gang-scheduled PyTorchJob (via Volcano) never leaves partial pods
   pending; either all `Worker` pods start or none do.
4. Quota accounting is owned in exactly one place, not double-counted
   between Kueue and Volcano.

## Requirements

1. **Manifest set (in `manifests/`).**
   - Three `ResourceFlavor`s (or one and document why).
   - Three `ClusterQueue`s, one per tenant, all in cohort
     `training`, with `nominalQuota`, `borrowingLimit`, and
     `lendingLimit` set to the numbers above.
   - Three `LocalQueue`s, one per tenant namespace.
   - Volcano `Queue`s with no capability (or capability large enough
     to be a no-op) — quota belongs to Kueue.
   - Four `PriorityClass`es matching the tiers in chapter 8.
   - One example `PyTorchJob` per tenant that admits into the
     LocalQueue, annotates a Volcano PodGroup, and sets
     `schedulerName: volcano`.
2. **Gang admission demonstration.**
   - Submit a job that needs 20 GPUs when only 16 are free. Show that
     the PyTorchJob's `runPolicy.suspend` stays `true` and no pods are
     created.
   - Free up capacity (delete another job or resize a mock quota).
     Show the PyTorchJob is unsuspended and all Worker pods are
     created simultaneously, not one at a time.
   - Capture the transition with `kubectl get workload,pytorchjob,pod
     -n <ns> -w` output and paste the relevant excerpt into the RFC.
3. **Gang scheduling demonstration.**
   - Cause a scheduling contention: submit a gang whose size exceeds
     free node capacity by one pod, and demonstrate that Volcano does
     not partially bind — no pod is bound until all can bind.
   - Contrast: temporarily set `schedulerName: default-scheduler` on
     the same job and show the partial-bind failure mode. Capture
     `kubectl describe podgroup` output for the difference.
4. **RFC (`RFC.md`).**
   - 500–900 words, RFC-style.
   - Explain the split: Kueue owns tenant quota, Volcano owns gang
     scheduling and preemption. Cite Kueue Concepts
     (https://kueue.sigs.k8s.io/docs/concepts/) and Volcano PodGroup
     docs (https://volcano.sh/en/docs/podgroup/).
   - Explain what would go wrong if you also gave Volcano's `Queue`
     capabilities (double-accounting).
   - Explain what would go wrong if you skipped Volcano and left
     `schedulerName` at default (partial-bind hangs).
   - Include a "Known limitations" section — at minimum,
     Kueue-vs-Volcano preemption interactions, and Kueue-TAS's
     current maturity for topology-aware placement.

## Starter guidance

- Read Kueue's Kubeflow PyTorchJob integration
  (https://kueue.sigs.k8s.io/docs/tasks/run/kubeflow/pytorchjob/) end
  to end before you start. It answers most of the "why is my job
  never admitting?" questions.
- Read Volcano's Quick Start (https://volcano.sh/en/docs/quick-start/)
  for the scheduler-config knobs (`proportion`, `gang`, `preempt`,
  `drf` plugins).
- Use a mock GPU device plugin (e.g., the sample device plugin from
  https://github.com/kubernetes/kubernetes/tree/master/pkg/kubelet/cm/deviceplugin)
  if you do not have real GPUs. Note the substitution up top in the
  RFC.
- Start with `nominalQuota` values that let *one* small demo job run
  quickly, then scale up. Debugging admission failures with 8 pending
  jobs is painful.
- `kubectl describe workload <name> -n <ns>` is the fastest source of
  truth for "why isn't Kueue admitting?".

## Acceptance criteria

- Each tenant's example PyTorchJob admits successfully to their
  LocalQueue.
- The `runPolicy.suspend: true` → `false` transition is observable
  via `kubectl get pytorchjob -n <ns>` output pasted in the RFC.
- The gang admission demo clearly shows *no pods* until quota is
  free, then *all pods* at once.
- The gang scheduling demo clearly shows Volcano refusing to partial-
  bind, and a paired experiment showing the default scheduler *does*
  partial-bind under the same conditions.
- The RFC cites at least one Kueue doc page and one Volcano doc page
  for its claims. Any claim that cannot be cited is marked
  `<!-- needs-research: ... -->` rather than asserted.

## Stretch goals

- Add preemption: give the `foundations` team a job that preempts a
  `research` job when quota is contested. Show the preempted job re-
  queues and eventually re-admits when capacity frees.
- Turn on Kueue's TopologyAwareScheduling
  (https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/)
  for one flavor and demonstrate that a PodSet with a required
  topology annotation admits only when all pods can be co-located.
- Compare wall-clock time-to-admission for a burst of 10 submissions
  under `queueingStrategy: BestEffortFIFO` vs.
  `StrictFIFO`. Explain the difference.
