# KubeRay and Ray Train: Autoscaling, Gang Scheduling, and Ray on Kubernetes

Ray (https://docs.ray.io/) is a general-purpose distributed compute
framework that has become the default engine for a chunk of the modern
training and post-training stack — RLHF loops (rllib, rl-hf), online
learning, hyperparameter sweeps (Ray Tune), and increasingly full
pretraining via Ray Train
(https://docs.ray.io/en/latest/train/train.html). KubeRay
(https://ray-project.github.io/kuberay/) is the operator that runs
Ray clusters on Kubernetes.

This chapter is the working knowledge you need to run a training job on
KubeRay so it (a) autoscales worker groups when the workload changes,
(b) gang-schedules the training workers so a partial cluster does not
poison the run, and (c) plays cleanly with Kueue and Volcano when it
must share the cluster with other batch workloads.

## Motivation: why Ray Train sits at a different altitude

`torchrun` and `mpirun` are process launchers — they start N Python
processes and tell each one its rank. Ray sits one level up:

- Ray is a **long-lived cluster** of head + worker processes with an
  object store and a scheduler that dispatches Ray tasks and actors.
- Ray Train wraps a distributed-training loop in Ray actors. From the
  researcher's perspective they write a `train_func` and hand it to a
  `TorchTrainer` (or `TFTrainer`, `XGBoostTrainer`) — Ray handles the
  worker startup, the rank assignment, the failure handling, and the
  checkpoint reporting.

That "long-lived cluster" is what KubeRay needs to model. It is *not*
the "one Job, N pods, all together" pattern of MPIJob / PyTorchJob.
It is a **head pod + heterogeneous worker groups** that can grow and
shrink under load.

## The core object: RayCluster (and RayJob, RayService)

- **`RayCluster`** — the head + workers cluster. `spec.headGroupSpec` is
  the head pod template; `spec.workerGroupSpecs[]` is an array of
  worker groups, each with a pod template, `minReplicas`,
  `maxReplicas`, and its own labels / node selectors.
- **`RayJob`** — one-shot submission wrapper. Creates a RayCluster,
  submits a Ray job (`ray job submit` semantics), waits for it, and
  optionally tears down the cluster. This is the object you point
  training at.
- **`RayService`** — long-lived Ray Serve deployment. Not relevant to
  training; mentioned so you do not confuse it with RayJob.

The autoscaling story is: turn on the KubeRay Autoscaler sidecar and it
watches Ray's demand signal — pending tasks, actor demand, and the
`num_gpus` request per actor — and scales worker groups up to
`maxReplicas` (or down to `minReplicas`).

## A worked example: Ray Train with an autoscaling worker group

Ray Train's `TorchTrainer` wants N training workers, each with 1 GPU
(the typical setup — one Ray actor per GPU). We size the head pod tiny
(no GPUs; it is coordination only) and let a single worker group hold
the trainers:

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  namespace: alice-research
  name: rayjob-llama-3-8b
  labels:
    kueue.x-k8s.io/queue-name: default
spec:
  suspend: true                                # Kueue admission gate
  entrypoint: python train_ray.py --config configs/llama-3-8b.yaml
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 600
  rayClusterSpec:
    rayVersion: "2.35.0"
    enableInTreeAutoscaling: true              # KubeRay autoscaler
    autoscalerOptions:
      upscalingMode: Aggressive
      idleTimeoutSeconds: 120
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
        num-cpus: "0"                          # head does no work
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.35.0-py311-cu121
              resources:
                requests: {cpu: "4", memory: "16Gi"}
                limits:   {cpu: "4", memory: "16Gi"}
    workerGroupSpecs:
      - groupName: trainers
        replicas: 8                            # initial size
        minReplicas: 8                         # never below one gang
        maxReplicas: 64                        # cap
        rayStartParams:
          num-gpus: "8"
        template:
          metadata:
            labels:
              ray.io/group: trainers
            annotations:
              scheduling.k8s.io/group-name: rayjob-llama-3-8b-pg
          spec:
            schedulerName: volcano             # gang-schedule the workers
            priorityClassName: training-normal
            containers:
              - name: ray-worker
                image: rayproject/ray:2.35.0-py311-cu121
                resources:
                  limits:
                    nvidia.com/gpu: 8
                    cpu: 16
                    memory: 200Gi
```

Two things to notice:

- The RayJob is `suspend: true` so **Kueue** can gate it, exactly like
  a PyTorchJob. The Kueue RayJob integration
  (https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/) knows how to
  admit and unsuspend.
- The **initial trainers group** is annotated with a PodGroup and set
  to `schedulerName: volcano` so the *training gang* comes up
  all-or-nothing. The autoscaler's incremental adds land as
  additional worker pods, which is fine — autoscaling elasticity is
  *inside* Ray, not for the initial gang.

## Autoscaling worker groups: when it helps, when it hurts

Autoscaling is the point of KubeRay for a lot of workloads. For
training specifically, there are three flavours of "autoscaling"
worth naming and each behaves differently:

1. **Autoscaling for RLHF / online / eval workers.** A Ray Train job
   often needs *other* worker groups — a rollout group, an eval group,
   a reward-model group — that are bursty. These groups should
   autoscale; you almost never want to reserve peak capacity for them
   full-time.
2. **Autoscaling for hyperparameter sweeps (Ray Tune).** Each trial is
   a Ray actor. The `trainers` group can scale from a few trials to
   many; use `minReplicas: 1` and `maxReplicas: <lots>`.
3. **Autoscaling for the actual training gang.** Almost always
   **do not** autoscale this. The training gang is a fixed size for the
   run (you sized the parallelism strategy in mod-101 around N GPUs);
   giving the autoscaler permission to add or remove trainers will
   trigger re-rendezvous and step-time variance you do not want.

The takeaway: model the *training* workers as a rigid worker group
(`minReplicas == replicas == some_fixed_N`) and reserve elasticity for
the surrounding workers. If you genuinely want elastic training,
Ray Train supports it via `TorchTrainer(scaling_config=ScalingConfig(...
num_workers=<current>))` and coordinated checkpoint restarts, but that
is the mod-106 story.

## Gang scheduling with KubeRay

Two layers to think about:

- **Gang admission at the queue layer** — Kueue with the RayJob
  integration, as above. Kueue admits the RayJob's Workload only when
  the initial head + trainers group quota is available.
- **Gang scheduling at the pod layer** — Volcano PodGroup + `schedulerName:
  volcano` on the trainers group, as above. KubeRay also supports the
  Coscheduling plugin (https://github.com/kubernetes-sigs/scheduler-plugins)
  as an alternative gang plugin.

The reason you want both: Kueue guarantees the *quota* was reserved
against your team's account; Volcano guarantees the *pods* land
together. Without Volcano (or Coscheduling), you can end up with a
head pod running and 6 of 8 trainer pods running while 2 are pending
because a node died — and Ray Train will happily start with a smaller
worker set unless the training code enforces the count. Prefer to
crash cleanly than to run with a silent-wrong world size.

## Composition sanity: Ray on top of the trio

If your platform already has Kueue + Volcano + the Kubeflow Training
Operator installed for PyTorchJobs, adding KubeRay is additive:

- Install the KubeRay operator (Helm chart or manifest).
- Install the Kueue RayJob integration (Kueue's CRD watcher list is
  configurable per install; add `rayjob`).
- Author RayJobs with the `suspend: true` + PodGroup + volcano
  scheduler pattern.

Where Ray earns its keep in this stack is *around* the training loop —
the RLHF rollout workers, the reward model actors, the eval workers,
the data prep tasks. If your training job has none of that and is
plain `torchrun train.py`, PyTorchJob is a simpler mental model. Do
not adopt Ray to run one `torchrun`.

## Common gotchas

- **`num-gpus` mismatch between `rayStartParams` and the pod
  resource limit.** Ray uses `num-gpus` to advertise capacity to its
  scheduler; if it disagrees with `nvidia.com/gpu`, Ray will
  double-book (or ignore) GPUs. Keep them in sync in a template.
- **Head pod scheduled on a GPU node.** Wasteful and blocks a training
  slot. Use a nodeSelector for the head; give it a non-GPU node.
- **Autoscaler flapping.** `idleTimeoutSeconds` too low + bursty demand
  produces "add worker, kill worker, add worker" churn. Start at 120 s
  or higher.
- **Ray version drift.** Ray's cross-version API is narrower than
  Kubernetes'; keep the head, workers, and the Ray client in the
  submitter all at the same Ray version.

## Summary

- KubeRay runs Ray clusters on Kubernetes. `RayJob` is the training
  submission object: head pod + one or more worker groups, one of
  which is your training gang.
- Autoscaling belongs on *surrounding* worker groups (rollouts, eval,
  sweeps), not on the training gang itself. Model the training gang
  as fixed-size.
- Compose with Kueue (RayJob suspend gate) and Volcano (PodGroup +
  `schedulerName`) exactly like a PyTorchJob. Both layers matter;
  quota reservation is not the same as pod-level gang.
- Ray shines when the training loop needs siblings — RLHF rollouts,
  eval workers, Ray Tune sweeps. If you have plain `torchrun`,
  PyTorchJob is a simpler fit.
