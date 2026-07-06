# The MPI Operator and the Kubeflow Training Operator

Kueue queues Workloads. Volcano schedules gangs. Neither *starts your
training processes*. On Kubernetes, that responsibility falls to a
**training operator** — a controller that watches a job-shaped CRD
(`MPIJob`, `PyTorchJob`, `TFJob`, `JAXJob`) and materialises the pods,
Services, ConfigMaps, and (for MPI) SSH plumbing that a distributed
training launcher expects.

This chapter is the working knowledge you need to run tightly-coupled
multi-node training on Kubernetes: what the MPI Operator vs. the
broader Kubeflow Training Operator do, when to reach for each, and how
they wire into the gang / queue layers.

## Two operators, one lineage

- **MPI Operator** (https://github.com/kubeflow/mpi-operator) — the
  original tightly-coupled operator. Owns `MPIJob` and the SSH-based
  `mpirun` launch pattern. Independent lifecycle, still actively
  maintained; the canonical choice when your training script *requires*
  MPI (e.g., Horovod, some NCCL-with-mpirun stacks) or when the
  simplicity of a single Launcher pod is what you want.
- **Kubeflow Training Operator (v1)** (https://www.kubeflow.org/docs/components/training/) —
  the multi-framework superset. Owns `PyTorchJob`, `TFJob`, `MXNetJob`,
  `XGBoostJob`, `PaddleJob`. Uses a role-and-replica model (Master +
  Worker), stands up a headless Service for peer discovery, and lets
  each pod compute its rank from `$RANK` / `$WORLD_SIZE` environment
  variables the operator sets. **Training Operator v2** (a rewrite; see
  https://github.com/kubeflow/training-operator) is the direction of
  travel; it consolidates the CRDs into a single `TrainJob` with a
  `TrainingRuntime` template and is the recommended target for new
  platforms.

For LLM-scale PyTorch training the practical picks in 2026 are:

- **PyTorchJob (v1) or TrainJob + PyTorch runtime (v2)** — for
  `torchrun`-driven jobs. This is the dominant pattern.
- **MPIJob (v2beta1)** — for anything that genuinely needs `mpirun`
  (Horovod, some NCCL-with-mpirun-launched runs, or when you already
  have an MPI-based codebase).

## What the operator actually does for you

Both operators share the same skeleton of responsibilities. Naming them
makes the difference between MPI Operator and PyTorchJob easy to see.

- **Pod template materialisation.** Turns each `replicaSpec` (Launcher,
  Master, Worker) into concrete pods. Injects the operator-specific
  environment (`OMPI_COMM_WORLD_RANK`, `RANK`, `WORLD_SIZE`, etc.).
- **Service and DNS.** Creates a headless Service so each pod can
  resolve its peers by DNS. PyTorchJob names workers
  `<jobname>-worker-<index>.<jobname>.<ns>.svc.cluster.local`.
- **Launch synchronization.** MPI Operator generates the SSH known-hosts
  file and the `hostfile` and launches `mpirun` from the Launcher.
  PyTorchJob sets `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE`
  and lets `torchrun` do the rendezvous.
- **Retry and completion.** Both honour `runPolicy.backoffLimit`,
  `activeDeadlineSeconds`, and `cleanPodPolicy`. Set `cleanPodPolicy:
  Running` in production so a crashed pod does not linger.
- **Integration hooks for Kueue and Volcano.** Both know how to leave
  the job suspended (Kueue) and both accept `schedulerName: volcano`
  and a PodGroup annotation.

## PyTorchJob: the `torchrun`-friendly path

The clean case — a PyTorchJob you can drop into a Kueue-gated queue and
have Volcano gang-schedule:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  namespace: alice-research
  name: llama-3-8b
  labels:
    kueue.x-k8s.io/queue-name: default
spec:
  runPolicy:
    suspend: true                            # Kueue admission gate
    cleanPodPolicy: Running
    backoffLimit: 3
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 8
    maxReplicas: 8
    nProcPerNode: 8
  pytorchReplicaSpecs:
    Worker:
      replicas: 8
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            scheduling.k8s.io/group-name: llama-3-8b-pg
        spec:
          schedulerName: volcano
          priorityClassName: training-normal
          containers:
            - name: pytorch
              image: registry/train:v2026-07
              env:
                - name: NCCL_ASYNC_ERROR_HANDLING
                  value: "1"
                - name: NCCL_DEBUG
                  value: "INFO"
              command:
                - torchrun
                - --nnodes=$(WORLD_SIZE)
                - --nproc-per-node=8
                - --rdzv-backend=c10d
                - --rdzv-endpoint=$(MASTER_ADDR):$(MASTER_PORT)
                - train.py
                - --config
                - configs/llama-3-8b.yaml
              resources:
                limits:
                  nvidia.com/gpu: 8
                  cpu: 16
                  memory: 200Gi
                  rdma/hca: 1                # if you use RDMA
```

The load-bearing details:

- `suspend: true` — required so Kueue can admit; Kueue flips it.
- `annotations: scheduling.k8s.io/group-name` + `schedulerName:
  volcano` — required so Volcano gang-schedules the workers.
- `elasticPolicy` — with `minReplicas == maxReplicas`, this is *not*
  elastic in the mod-106 sense; setting it just standardises rendezvous
  config. `min < max` unlocks elastic-style re-rendezvous on worker
  loss.
- `nvidia.com/gpu: 8` per pod is one pod per node; `torchrun`'s
  `--nproc-per-node=8` fans out inside the pod.
- `rdma/hca` (or similar) is how you request an RDMA HCA when the
  device plugin (SR-IOV, RDMA, or NVIDIA GPU Operator) exposes them.
  Naming varies by device plugin; check your cluster's plugin.
- `NCCL_ASYNC_ERROR_HANDLING=1` turns a hung collective into a
  Python exception you can trap instead of a wedged process. See
  https://pytorch.org/docs/stable/distributed.html.

## MPIJob: the mpirun launcher pattern

MPIJob's shape is Launcher + Workers, one Launcher pod that shells into
the Workers over SSH and starts `mpirun -np N`. It is the right choice
for Horovod (https://github.com/horovod/horovod), for legacy MPI
codebases, and occasionally for NCCL runs that prefer `mpirun` to
`torchrun`.

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  namespace: alice-research
  name: horovod-resnet
spec:
  slotsPerWorker: 8                          # GPUs per worker pod
  runLauncherAsWorker: false
  sshAuthMountPath: /root/.ssh
  runPolicy:
    cleanPodPolicy: Running
    schedulerName: volcano
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        metadata:
          annotations:
            scheduling.k8s.io/group-name: horovod-resnet-pg
        spec:
          schedulerName: volcano
          containers:
            - name: mpi-launcher
              image: registry/horovod:v2026-07
              command:
                - mpirun
                - --allow-run-as-root
                - -np
                - "64"
                - -bind-to
                - none
                - -map-by
                - slot
                - -mca
                - pml
                - ob1
                - -mca
                - btl
                - ^openib
                - python
                - train.py
    Worker:
      replicas: 8
      template:
        metadata:
          annotations:
            scheduling.k8s.io/group-name: horovod-resnet-pg
        spec:
          schedulerName: volcano
          containers:
            - name: mpi-worker
              image: registry/horovod:v2026-07
              resources:
                limits:
                  nvidia.com/gpu: 8
                  cpu: 16
                  memory: 200Gi
```

The MPI Operator injects the SSH secret and the `hostfile` and starts
the Launcher only after all Worker pods are Ready. When paired with
Volcano, that gang-of-workers guarantee lines up with MPI's expectation
that all peers are reachable at `mpirun` time.

## PyTorchJob vs. MPIJob: how to pick

The default answer for new `torchrun`-based training in 2026 is
**PyTorchJob** (or TrainJob with the PyTorch runtime under Training
Operator v2). Reach for **MPIJob** when:

- You are running Horovod.
- You have an MPI-based codebase you cannot easily port.
- You need `mpirun` to lay down process affinities or NUMA bindings the
  container runtime does not give you.

Do not use MPIJob because "MPI feels more HPC". PyTorchJob + `torchrun`
+ `c10d` rendezvous is the actively-developed path.

## Common gotchas

- **Headless Service DNS not resolving.** Almost always the Kubernetes
  DNS pod is starved for CPU or the Service was not created (Operator
  did not reconcile because the CRD is a version mismatch). Confirm
  with `kubectl get svc -n <ns>`.
- **`init_process_group` timeout of 30 minutes.** The default;
  training operators often override to 60+ minutes. If your rendezvous
  is racy under load, bump this via
  `torch.distributed.init_process_group(timeout=timedelta(minutes=60))`.
- **`cleanPodPolicy: None`.** Crashed pods stay around forever. Set to
  `Running` in production.
- **SSH secret drift on MPIJob.** If you rebuild the base image and
  forget to regenerate the SSH host keys, the Launcher's known_hosts
  goes stale and `mpirun` hangs waiting for keys.

## Summary

- Training operators (MPI Operator, Kubeflow Training Operator) turn a
  training-shaped CRD into pods, DNS, and launch plumbing. They are the
  "actually run the processes" layer under Kueue + Volcano.
- **PyTorchJob** is the default for `torchrun`-driven training in 2026;
  **MPIJob** is the pick for Horovod / MPI-native codebases; **TrainJob**
  (Training Operator v2) is the consolidated forward direction.
- Composition looks the same in both: `runPolicy.suspend: true` for
  Kueue, `schedulerName: volcano` + PodGroup annotation for gang
  scheduling, per-pod resource requests for the device plugin.
