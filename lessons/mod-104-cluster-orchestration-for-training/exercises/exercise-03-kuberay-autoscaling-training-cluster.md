# exercise-03: KubeRay Autoscaling Training Cluster

**Estimated effort:** 4 hours

## Objective

Run a Ray Train job on KubeRay with two worker groups — a fixed-size
gang-scheduled *trainers* group and an autoscaling *rollouts* group —
and demonstrate that the autoscaler responds to load on the rollouts
group without touching the trainers gang. Produce the RayJob manifest,
a short driver script, and a write-up that explains why the trainers
group is not autoscaled.

## Prerequisites

- Chapter 6.
- Chapters 3 and 4 (Kueue + Volcano) if you want the Kueue admission
  path; you can also do this exercise on a bare cluster with no queue
  layer.
- A Kubernetes cluster (real GPUs or `kind`-based mock; note which in
  the write-up). Install:
  - KubeRay operator
    (https://ray-project.github.io/kuberay/deploy/installation/).
  - Optional: Kueue with the RayJob integration
    (https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/) and Volcano
    for gang scheduling the trainers.

## Problem statement

Your team wants to run a small RLHF-style loop: N trainers doing
gradient updates, plus a bursty pool of rollout actors that produce
trajectories. Trainers must all come up together (gang). Rollouts
should scale from 0 to many based on Ray demand and back down.

Model this as a `RayJob` with two worker groups. Prove the two
different behaviours.

## Requirements

1. **`rayjob.yaml`.**
   - `spec.entrypoint` runs a short Python driver (`ray_driver.py`)
     that starts a Ray Train `TorchTrainer` with `num_workers = 8`
     and, in a background loop, submits Ray tasks that require the
     rollouts group.
   - `spec.rayClusterSpec.enableInTreeAutoscaling: true`.
   - Head group with `num-cpus: "0"`, non-GPU node selector, small
     memory.
   - `trainers` worker group: `minReplicas: 8`, `maxReplicas: 8`,
     `replicas: 8`, annotated with a Volcano PodGroup, and
     `schedulerName: volcano`.
   - `rollouts` worker group: `minReplicas: 0`, `maxReplicas: 8`,
     `replicas: 0`, no gang annotation.
2. **`ray_driver.py`.**
   - Uses `ray.init(address="auto")`.
   - Starts a `TorchTrainer` (or `TorchTrainer`-style scaffold; the
     model does not have to converge). The `train_func` can be a
     dummy loop that reports metrics.
   - Concurrently submits Ray tasks (`@ray.remote(num_gpus=1)`
     dummy tasks that sleep) to force the rollouts group to scale
     up.
   - Prints, on a loop, the current number of active Ray nodes and
     the pending task count so the autoscaler behaviour is visible
     in the logs.
3. **Demonstration and observations (`OBSERVATIONS.md`).**
   - Include `kubectl get pods -n <ns>` output at three checkpoints:
     (a) job just admitted, before rollouts are triggered; (b)
     during peak rollouts load; (c) after rollouts idle for the
     autoscaler timeout.
   - Show the trainers group stays at 8 through the entire run.
   - Include `kubectl logs <head pod>` excerpts that show the
     autoscaler decisions (log line prefix `[autoscaler]`).
   - 300–500 words explaining: (i) why the trainers group must not
     autoscale mid-run and (ii) what would happen if `minReplicas`
     on trainers were `1`.

## Starter guidance

- The KubeRay quickstart with autoscaling
  (https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/configuring-autoscaling.html)
  is the reference for `enableInTreeAutoscaling` and
  `autoscalerOptions`.
- Ray Train's `TorchTrainer` scaffold is in the Ray docs
  (https://docs.ray.io/en/latest/train/train.html). You do not need a
  real model — the point is Ray Train's placement group and rank
  assignment behaviour.
- If you cannot get real GPUs, replace `num_gpus=1` on the rollouts
  tasks with `num_cpus=2` and note the substitution in
  `OBSERVATIONS.md`. Everything scales the same way.
- The autoscaler is *reactive*, not *predictive*. Give it 30–60 s
  after a rollout burst to see scale-up; give it
  `idleTimeoutSeconds` to see scale-down.
- Ray's `ray status` command from inside the head pod is the fastest
  way to confirm the autoscaler's view of demand.

## Acceptance criteria

- The `rayjob.yaml` starts a Ray cluster with a head and 8 trainer
  pods (visible via `kubectl get pods`). If Kueue is in the mix, the
  `runPolicy.suspend: true` → `false` transition is captured.
- The trainers group's pod count stays at 8 for the entire run.
  You demonstrate this by `kubectl get pods -n <ns> -l ray.io/group=trainers`
  at three or more time points.
- The rollouts group scales from 0 → N (where N ≤ `maxReplicas`) when
  the driver submits tasks, and scales back down after
  `idleTimeoutSeconds`.
- `OBSERVATIONS.md` correctly explains that autoscaling the trainers
  group would trigger a Ray Train re-rendezvous / re-init, which is
  functionally a preemption event, and cites either the Ray Train
  docs or the KubeRay autoscaling docs for the claim.

## Stretch goals

- Wire Kueue in front of the RayJob and demonstrate quota-gated
  admission. Show a queued RayJob that only unsuspends when a
  competing job releases capacity.
- Replace the `sleep` rollout tasks with actual `torch` CPU/GPU
  workloads and observe how the autoscaler responds when demand is
  non-trivially expensive.
- Chaos test: kill a trainer pod mid-run. Observe what Ray Train
  does. Write 100 words on how you would want the SDK (chapter 9) to
  handle this failure mode.
- Switch the trainers group's gang scheduler from Volcano to the
  Coscheduling plugin
  (https://github.com/kubernetes-sigs/scheduler-plugins/tree/master/pkg/coscheduling)
  and note any behavioural difference.
