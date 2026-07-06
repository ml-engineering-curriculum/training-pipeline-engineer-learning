# exercise-04: TorchX Multi-Scheduler Launch

**Estimated effort:** 3 hours

## Objective

Author a single TorchX `AppDef` for a small `torchrun`-based training
job, and submit the same `AppDef` unchanged to three schedulers:
`local_cwd`, `slurm`, and `kubernetes` (via the Kubeflow Training
Operator + Volcano). Produce the `AppDef`, the three submission
transcripts, and a short comparison table.

## Prerequisites

- Chapter 7.
- Working access to at least two of the three targets. If you have
  only `local_cwd` + one cluster, do the third target with a mock
  scheduler backend and note it in the write-up. Do not skip either
  real cluster target entirely.
- Install TorchX and its scheduler extras
  (https://pytorch.org/torchx/latest/quickstart.html). The
  `torchx[kubernetes]` and `torchx[slurm]` extras add the schedulers
  you need.

## Problem statement

Your platform team is deciding whether to standardise on TorchX as the
implementation layer for the launcher SDK (chapter 9). Prove or
disprove the "one AppDef, many schedulers" pitch by actually running
it.

## Requirements

1. **`apps/train.py`.**
   - Define one function `train_app(config: str, nnodes: int = 2,
     nproc: int = 2) -> torchx.specs.AppDef` that returns an
     `AppDef` with a single `Role`.
   - The Role's entrypoint is `torchrun` with `--rdzv-backend=c10d`,
     `--rdzv-endpoint=$${MASTER_ADDR}:29500`, `--nnodes`,
     `--nproc-per-node`, and a small training script (`train.py`) as
     the payload.
   - Resource: cpu=2, memMB=4000, gpu=0 for local mode. Parametrise
     this so it is `gpu=1` on the real clusters (or just
     document the change per target if you make it manually).
2. **Submissions.** Submit the same `AppDef` on:
   - `torchx run --scheduler local_cwd apps/train.py:train_app ...`
   - `torchx run --scheduler slurm ...` (with the appropriate
     `--scheduler_args partition=...,time=...`).
   - `torchx run --scheduler kubernetes ...` (with
     `--scheduler_args namespace=...,queue=...`), against Kubeflow
     Training Operator + Volcano from exercise 2.
   - Capture each submission's stdout and the resulting handle (job
     id / pod name).
3. **Comparison (`COMPARISON.md`).**
   - A table of: submission command, target scheduler, TorchX
     backend, wall-clock time-to-first-log, whether gang scheduling
     was engaged, and any manual patches you had to make to the
     `AppDef` for that target.
   - 200–400 words on: what TorchX abstracted well vs. where the
     abstraction leaked (typical leaks: SLURM `topology`,
     Kubernetes PodGroup annotations, Kueue `runPolicy.suspend`
     flag).
   - A recommendation for or against adopting TorchX as the
     implementation layer for the SDK. Ground it in the specific
     leaks you observed, not general aesthetics.

## Starter guidance

- Start with the TorchX
  distributed example
  (https://pytorch.org/torchx/latest/components/distributed.html) and
  factor its Role into your own `AppDef` function.
- The `train.py` payload can be as small as three lines: import
  `torch.distributed`, `init_process_group("gloo")`, print
  `rank / world_size`, exit. The point is the launcher, not the
  model.
- On the Kubernetes backend, TorchX will generate an MCAD `AppWrapper`
  or a `Job` — check which flavour your version emits and, if it is
  not what your cluster ingests, either upgrade or use the
  `--scheduler kubernetes_mcad` variant. Note in the comparison.
- On SLURM, TorchX generates an sbatch script; you can print it with
  `--dryrun` and diff against the exercise 1 template. Do so and note
  the differences.
- Keep the `AppDef` minimal — every field you add is a field TorchX
  has to translate for each scheduler. Start small and grow.

## Acceptance criteria

- One `AppDef` function; three submissions from the same file.
- Each submission runs to completion (or produces the expected
  scheduler-side artifact — for local it prints; for SLURM the job
  reaches `COMPLETED`; for Kubernetes the PyTorchJob / Job reaches
  `Succeeded`).
- The comparison table lists at least three concrete rows and cites
  specific TorchX doc pages
  (https://pytorch.org/torchx/latest/schedulers/) for each
  scheduler.
- The write-up's recommendation is grounded in observed leaks and
  cites at least one TorchX GitHub issue or doc caveat rather than
  vibes.

## Stretch goals

- Add a `TorchXBackend` implementation of the `SchedulerBackend`
  interface from chapter 9. Route between it and a hand-rolled
  backend using the router pattern.
- Stress-test rendezvous: submit two of the same TorchX job to the
  same cluster at the same time. Observe whether TorchX gives them
  different `--rdzv-id` values automatically or whether you have to
  set it. Report.
- Replace the payload with a real 100M-parameter model port from
  mod-101 exercise 4 and confirm the busbw with `NCCL_DEBUG=INFO`.
- Submit the same `AppDef` via `torchx run --scheduler ray` for a
  fourth target. Compare against `kubernetes` submission mechanics.
