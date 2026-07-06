# TorchX and torchrun: Scheduler-Agnostic Launch

The story so far has been "here is how you launch on SLURM", "here is
how you launch on Kubernetes + Kueue + Volcano + PyTorchJob", "here is
how you launch on KubeRay". A production training platform serves
multiple clusters — often at least one SLURM cluster and one Kubernetes
cluster — and researchers do not want to learn three launch flows.

This chapter is the working knowledge you need to reason about
scheduler-agnostic launchers: **`torchrun`**, PyTorch's process
launcher; **TorchX**, PyTorch's higher-level job-submission API; and
the trade-offs that decide when to pick which.

## What each layer actually does

- **`torchrun`** (formerly `torch.distributed.launch`;
  https://pytorch.org/docs/stable/elastic/run.html) — a process
  launcher. Given `--nnodes`, `--nproc-per-node`, and a rendezvous
  endpoint, it spawns N Python subprocesses per node, sets `RANK`,
  `WORLD_SIZE`, `LOCAL_RANK`, and stands up a c10d rendezvous store.
  It does *not* allocate nodes or submit jobs; something else — SLURM
  `srun`, PyTorchJob, MPIJob's `mpirun`, or a manual SSH loop — has to
  put N `torchrun` processes on N nodes.
- **`torch.distributed.elastic`** — the elastic runtime that lives
  under `torchrun`. It is what makes rendezvous re-negotiable when a
  worker leaves. mod-106 covers this in depth; here it is enough to
  know that "torchrun already knows how to re-rendezvous".
- **TorchX** (https://pytorch.org/torchx/) — a Python API that
  packages a training job as an `AppDef` (containers, args, resource
  requests) and *submits* it to a scheduler. TorchX ships adapters
  for `local`, `slurm`, `kubernetes` (via Kubeflow / Volcano),
  `kubernetes_mcad`, `ray`, `aws_batch`, `docker`, and more. Same
  Python code, different `--scheduler`.

Think of the layers as: `torchrun` is a process launcher, TorchX is a
job submitter. They are not competitors; they are stacked. TorchX
generates the sbatch script (or PyTorchJob manifest, or RayJob spec)
that runs `torchrun`.

## `torchrun` in one paragraph

Every training script in this module ultimately runs behind
`torchrun`:

```bash
torchrun \
  --nnodes=$WORLD_NODES \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$RDZV_HOST:29500 \
  --rdzv-id=$JOB_ID \
  --max-restarts=3 \
  train.py --config configs/llama-3-8b.yaml
```

The load-bearing knobs are the rendezvous ones (`--rdzv-backend`,
`--rdzv-endpoint`, `--rdzv-id`) and the elastic ones (`--min-nnodes`,
`--max-nnodes`, `--max-restarts`). Every scheduler in this module
either sets these for you (PyTorchJob) or gives you variables to plug
in (`$SLURM_JOB_MASTER_NODE`, `$MASTER_ADDR`). If you understand this
command you understand the process-launch layer everywhere.

## TorchX by example: one Python file, many schedulers

TorchX's `AppDef` is a container-agnostic description of the job. The
same `AppDef` runs on SLURM or Kubernetes just by swapping
`--scheduler`:

```python
# apps/llama_pretrain.py
from torchx import specs

def llama_pretrain(config: str, nnodes: int = 8, nproc: int = 8) -> specs.AppDef:
    role = specs.Role(
        name="trainer",
        image="registry/train:v2026-07",
        entrypoint="torchrun",
        args=[
            "--nnodes", str(nnodes),
            "--nproc-per-node", str(nproc),
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=$${MASTER_ADDR}:29500",
            "train.py", "--config", config,
        ],
        num_replicas=nnodes,
        resource=specs.Resource(cpu=16, memMB=200_000, gpu=nproc),
    )
    return specs.AppDef(name="llama-pretrain", roles=[role])
```

```bash
# Same AppDef, different backend.

# 1. Local smoke test
torchx run --scheduler local_cwd apps/llama_pretrain.py:llama_pretrain \
  --config configs/llama-3-8b.yaml --nnodes 1 --nproc 2

# 2. SLURM production
torchx run --scheduler slurm \
  --scheduler_args partition=training,time=24:00:00,constraint=h100 \
  apps/llama_pretrain.py:llama_pretrain \
  --config configs/llama-3-8b.yaml --nnodes 8 --nproc 8

# 3. Kubernetes production (via Volcano + Kubeflow Training Operator)
torchx run --scheduler kubernetes \
  --scheduler_args namespace=alice-research,queue=default \
  apps/llama_pretrain.py:llama_pretrain \
  --config configs/llama-3-8b.yaml --nnodes 8 --nproc 8
```

Two things TorchX buys you:

1. **A single-source-of-truth job definition.** The same `AppDef` is
   the CI smoke-test invocation and the production submission, so you
   cannot ship an sbatch script that has drifted from what CI ran.
2. **Structured scheduler args instead of shell strings.** SLURM
   `--time` and Kubernetes `time` are the same field surfaced
   consistently.

Skim the TorchX Schedulers docs
(https://pytorch.org/torchx/latest/schedulers/) and the built-in
components list before adopting it — it is opinionated and, like any
abstraction layer, has some scheduler features it does not fully cover.

## When to pick which launcher

There are three questions worth answering explicitly.

### Q1: Do I need TorchX at all?

Reach for TorchX when:

- You run on more than one scheduler and want one submission
  interface.
- You want the same job definition in CI and prod (regression testing).
- You are building a *platform* (which is exactly this module's goal)
  and want a stable Python API to build on top of.

Skip TorchX when:

- You are on a single scheduler and the researcher already knows how
  to write sbatch scripts / PyTorchJob manifests.
- Your platform has a proprietary submission API and TorchX would be
  the third layer on top; two is enough. (This is a real
  consideration; chapter 9 discusses.)

### Q2: torchrun or mpirun as the process launcher?

Prefer `torchrun` when:

- You are on modern PyTorch (2.x+) and want elastic rendezvous.
- Your training script uses `torch.distributed` directly.
- You want `--rdzv-backend=c10d` and its integrated store.

Prefer `mpirun` when:

- Horovod is in the loop.
- Your codebase already uses MPI collectives (some HPC-native training
  stacks).
- You need MPI's process-binding controls the container runtime does
  not give you.

### Q3: Should I write my own launcher instead of TorchX?

Sometimes yes. TorchX is *not* the right choice when your platform
has strong opinions the AppDef abstraction cannot carry — e.g., you
need per-scheduler artifacts (custom SLURM prolog, custom Kueue
resource flavor mapping) that TorchX would flatten. Chapter 9 is
about building your own SDK, and the honest answer is that many
mature platforms have their own thin launcher on top of the SDKs
here rather than depending on TorchX end-to-end.

## The two failure modes to plan for

- **Rendezvous timeouts.** The default `c10d` rendezvous will wait 30
  minutes for peers. On a Kueue-gated queue with a long backlog, that
  is not enough — the first pod comes up while the last is still
  admitting. Set `--rdzv-conf join_timeout=3600` or increase the
  Python-side `init_process_group(timeout=...)`.
- **Rank mismatch.** If your launcher computes rank from
  `$SLURM_PROCID` on one side and `$RANK` on the other, and something
  wires them wrong, you get duplicate ranks or missing ranks and NCCL
  hangs at `init_process_group`. Print `hostname`, `local_rank`,
  `global_rank` from every process at t=0. It is boring; it saves
  incidents.

## Summary

- `torchrun` is a process launcher; TorchX is a job submitter. They
  stack: TorchX generates the launcher config, `torchrun` starts the
  processes.
- One TorchX `AppDef` can submit to `local`, `slurm`,
  `kubernetes`, `ray`, and more; the pitch is "same job def, many
  clusters".
- Pick TorchX when you have multiple schedulers and want one Python
  API. Roll your own thin launcher (chapter 9) when platform opinions
  need surfaces TorchX flattens.
- Always instrument rendezvous — timeouts and rank correctness are
  the two failure modes that show up on day one.
