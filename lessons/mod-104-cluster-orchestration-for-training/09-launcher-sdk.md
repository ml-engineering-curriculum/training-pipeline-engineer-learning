# Designing the Launcher SDK

Every previous chapter has been about a specific mechanism: sbatch,
Kueue, Volcano, PyTorchJob, KubeRay, TorchX, quotas. This chapter is
about the last mile — the internal SDK that hides all of it behind
one Python API so a researcher writes:

```python
from mytrain import launch, Job

launch(Job(
    config="configs/llama-3-8b.yaml",
    gpus=128,
    image="registry/train:v2026-07",
    priority="training-high",
))
```

...and the platform decides which cluster, which scheduler, how many
nodes, what queue, and what checkpoint policy. Building a good version
of that SDK is the capstone artifact of this module.

## Design goals

Before you write any code, name the goals. These are the ones this
chapter optimises for:

1. **One API, many schedulers.** The researcher never picks between
   SLURM and Kubernetes; the SDK does.
2. **Config-first, code-second.** Most jobs are a YAML config away
   from an existing recipe. Reach for Python for the exceptional
   cases.
3. **Reproducibility by construction.** Every submission is logged
   with the exact image, config, code SHA, and scheduler args, so
   "rerun what Alice ran on Tuesday" is one command.
4. **Fast local iteration.** The same `Job` runs on a laptop with
   `--scheduler local` before you ever consume cluster quota.
5. **Failure-mode aware.** Preemption, node failure, and rendezvous
   timeouts have first-class stories, not opaque stack traces.
6. **Cheap to extend.** Adding a scheduler is a plugin. Adding a
   priority tier is a config change.

Goals 3 and 6 quietly do most of the work; the others are the visible
API.

## The core abstractions

A workable SDK ends up with roughly four objects. Names are yours;
these are the shapes.

- **`Job`** — the researcher-facing spec. Immutable dataclass:
  `config`, `image`, `gpus`, `nodes`, `priority`, `wall_time`,
  `command`, `env`, `resources`, `checkpoint_path`.
- **`SchedulerBackend`** (interface) — one implementation per
  scheduler: `SlurmBackend`, `KueueVolcanoPyTorchJobBackend`,
  `KubeRayBackend`, `LocalBackend`, `TorchXBackend` (if you use it
  under the hood). Each implements `submit(Job) -> Handle`,
  `status(Handle)`, `cancel(Handle)`, `logs(Handle)`.
- **`Handle`** — an opaque, serializable id (`slurm://<jobid>`,
  `k8s://<ns>/<name>`) that lets you re-attach across processes.
- **`ClusterRouter`** — the piece that picks a backend for a `Job`.
  Reads a config file that maps `(gpus, priority, cluster
  preferences)` to a backend and a queue.

The submission flow, drawn as a pipeline:

```
Job (immutable)
  → ClusterRouter.route(Job) -> (SchedulerBackend, backend_args)
  → SchedulerBackend.render(Job, backend_args) -> Manifest
  → SchedulerBackend.submit(Manifest) -> Handle
  → JobLedger.record(Handle, Job, Manifest, code_sha, image_digest)
  → return Handle
```

Every arrow is testable in isolation. The router does not touch the
cluster. The backend rendering is pure. Only `submit()` and
`JobLedger.record()` do I/O.

## A concrete `Job` and a rendered PyTorchJob

The Python side is small enough to fit on a page:

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class Job:
    name: str
    config: str
    image: str
    gpus: int
    gpus_per_node: int = 8
    priority: Literal["critical", "training-high",
                      "training-normal", "research"] = "training-normal"
    wall_time_h: int = 24
    command: tuple[str, ...] = ("torchrun",)
    env: dict[str, str] = field(default_factory=dict)
    checkpoint_path: str = ""
    max_restarts: int = 3

    @property
    def nodes(self) -> int:
        return self.gpus // self.gpus_per_node
```

The Kubernetes backend turns that into a PyTorchJob:

```python
class KueueVolcanoPyTorchJobBackend(SchedulerBackend):
    def __init__(self, namespace: str, queue: str, cluster: str):
        self.namespace = namespace
        self.queue = queue
        self.cluster = cluster

    def render(self, job: Job) -> dict:
        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {
                "namespace": self.namespace,
                "name": job.name,
                "labels": {"kueue.x-k8s.io/queue-name": self.queue},
            },
            "spec": {
                "runPolicy": {
                    "suspend": True,
                    "cleanPodPolicy": "Running",
                    "backoffLimit": job.max_restarts,
                    "activeDeadlineSeconds": job.wall_time_h * 3600,
                },
                "pytorchReplicaSpecs": {
                    "Worker": {
                        "replicas": job.nodes,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "metadata": {
                                "annotations": {
                                    "scheduling.k8s.io/group-name":
                                        f"{job.name}-pg",
                                },
                            },
                            "spec": {
                                "schedulerName": "volcano",
                                "priorityClassName": job.priority,
                                "containers": [self._container(job)],
                            },
                        },
                    },
                },
            },
        }

    def _container(self, job: Job) -> dict:
        return {
            "name": "pytorch",
            "image": job.image,
            "env": [{"name": k, "value": v} for k, v in job.env.items()],
            "command": list(job.command) + [
                f"--nnodes={job.nodes}",
                f"--nproc-per-node={job.gpus_per_node}",
                "--rdzv-backend=c10d",
                "--rdzv-endpoint=$(MASTER_ADDR):$(MASTER_PORT)",
                f"--max-restarts={job.max_restarts}",
                "train.py",
                f"--config={job.config}",
            ],
            "resources": {
                "limits": {
                    "nvidia.com/gpu": job.gpus_per_node,
                    "cpu": 16,
                    "memory": "200Gi",
                },
            },
        }
```

The SLURM backend renders an sbatch script from the same `Job`. The
Ray backend renders a RayJob. The local backend runs `torchrun`
in-process. All four accept the same input.

## The router: how "which scheduler?" is decided

A production router is one function:

```python
def route(job: Job, config: RouterConfig) -> tuple[SchedulerBackend, dict]:
    for rule in config.rules:
        if rule.matches(job):
            return rule.backend, rule.backend_args
    raise NoRouteError(f"no backend accepts {job.name}")
```

`RouterConfig.rules` is a YAML file the platform team owns:

```yaml
rules:
  - name: local-smoke-test
    match:
      env: [LOCAL_DEV, CI]
    backend: local

  - name: slurm-large-h100
    match:
      gpus: {gte: 256}
      priority: [training-high, critical]
    backend: slurm
    backend_args:
      partition: training
      constraint: h100

  - name: k8s-default
    match: {}
    backend: kueue_volcano_pytorchjob
    backend_args:
      namespace: "{{ tenant }}"
      queue: default
      cluster: prod-us-east
```

Two things this router earns you:

- **Migration is a config change.** Move a rule and jobs shift
  scheduler on next submission. No code change; no researcher change.
- **Multi-cluster is native.** A rule can select a *cluster* as well
  as a backend; the SDK can hold connections to N kube configs and N
  SLURM controllers and route across them.

## The ledger: reproducibility by construction

Every `submit()` writes one row to a ledger:

```json
{
  "submitted_at": "2026-07-05T18:03:11Z",
  "job_name": "llama-3-8b-2026-07-05",
  "submitter": "alice@corp",
  "code_sha": "1a2b3c...",
  "image_digest": "sha256:...",
  "config_digest": "sha256:...",
  "backend": "kueue_volcano_pytorchjob",
  "backend_args": {"namespace": "...", "queue": "default"},
  "manifest_digest": "sha256:...",
  "handle": "k8s://foundations/llama-3-8b-2026-07-05"
}
```

Store it in an append-only place — object store, a database, a
Postgres — and mount it on a `mytrain history` command. The value is
not the individual row; it is that you can answer, six months later,
"what exactly ran when?" and "what code produced this checkpoint?"
without archeology.

## Failure-mode surface

The launcher SDK is where the failure-mode contract lives.

- **Rendezvous timeout.** Backends set
  `--rdzv-conf join_timeout=<long enough for Kueue admission +
  fabric warmup>`.
- **Preemption.** Backends set the source Job's retry / requeue
  policy so a preempted job re-lands on next admission. On SLURM,
  `--requeue`; on Kubernetes, `runPolicy.backoffLimit`.
- **Checkpoint discovery.** The SDK writes `job.checkpoint_path`
  into the environment (`MYTRAIN_CHECKPOINT_PATH`) and researchers'
  training loops resume from there. mod-106 owns the checkpointing
  side.
- **Wall-clock warning.** Backends set a signal 120 s before wall
  clock (`--signal=B:USR1@120` on SLURM, a preStop hook on
  Kubernetes) so the training loop can flush a final checkpoint.
- **Cancellations.** `Handle` supports `cancel()`. Backends translate
  to `scancel` or `kubectl delete pytorchjob`. Cancellation is
  idempotent.

## What NOT to build

- **Do not build a container image builder.** Use the standard image
  pipeline (CI-built image, digest-pinned reference). The SDK
  consumes an image URL; it does not build.
- **Do not build a data pipeline.** mod-103 owns that. The SDK
  passes `config` through; the training script reads it.
- **Do not build a Kubernetes / SLURM abstraction one level deeper.**
  If a researcher needs an unusual scheduler feature, the escape
  hatch is: pass `raw_manifest_overrides` on the `Job`. Do not try
  to model every feature; model the common path well and let power
  users patch.
- **Do not build a UI first.** A CLI + a Python API is enough for the
  first year. A UI is a downstream project once the API is stable.

## When to use TorchX under the hood

TorchX (chapter 7) is a legitimate implementation detail for some
backends. If you are on the reference PyTorch stack and your
scheduler is one TorchX supports well, you can implement
`SchedulerBackend` as a thin adapter over `torchx.runner`. Two
constraints will decide:

- If your platform's opinions map cleanly to TorchX's model, use
  TorchX.
- If your platform has a bespoke queueing / policy surface that
  TorchX does not model, generate the sbatch script / manifest
  directly and skip TorchX.

Both choices are respectable; the second is what most large shops
end up with because the SDK is where all the platform's opinions
converge.

## Summary

- The launcher SDK is the API researchers use. Its whole point is to
  hide the scheduler diversity of the last seven chapters behind one
  `Job` type.
- Four abstractions do the work: `Job`, `SchedulerBackend`, `Handle`,
  `ClusterRouter`. The pipeline is pure until `submit()`.
- The router is a config file. Migration between clusters and
  schedulers is a config change, not a code change.
- The ledger is what makes runs reproducible six months later.
- The failure-mode contract (rendezvous timeout, preemption,
  checkpoint discovery, wall-clock warning) is enforced at the SDK
  layer, once, not in every researcher's training script.
- TorchX is a fine implementation detail for the common case; do not
  build over it if platform opinions do not fit.
