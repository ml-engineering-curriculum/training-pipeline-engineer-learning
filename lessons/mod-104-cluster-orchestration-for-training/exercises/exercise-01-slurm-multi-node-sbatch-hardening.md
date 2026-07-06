# exercise-01: SLURM Multi-Node sbatch Hardening

**Estimated effort:** 4 hours

## Objective

Author a hardened, production-shaped SLURM submission for a multi-node
`torchrun` training job. The deliverable is (a) an sbatch script that
would survive an unfriendly cluster and a paper-deadline reservation
window, (b) a prolog snippet that keeps a bad node out of the run, and
(c) a short write-up explaining the two knobs you tuned and why.

This is chapter 2 in your fingers. If you can defend every line of the
script to a colleague, you understand what SLURM is doing on your
behalf.

## Prerequisites

- Chapter 2 of this module.
- Access to a SLURM cluster with at least 2 GPU nodes. A test cluster
  is fine — no real training needs to happen — but you must be able to
  submit and observe a job. If you have no SLURM handy, spin up a
  two-node SLURM-in-Docker via
  https://github.com/giovtorres/slurm-docker-cluster (CPU-only is OK;
  substitute a CPU rendezvous demo for the GPU torchrun call and note
  the substitution in the write-up).
- A minimal training script (see mod-101 exercise 4) or the reference
  `torch.distributed` smoke test from
  https://pytorch.org/docs/stable/distributed.html.

## Problem statement

Your platform team wants a canonical sbatch template all new training
jobs start from. Existing scripts on the cluster are inconsistent —
some do not pin CPUs, some do not request GPUs correctly, some hang at
wall clock instead of checkpointing, and one team's prolog let a
degraded GPU into a run last month.

Author the canonical template and the prolog snippet that would have
prevented last month's incident.

## Requirements

1. **`sbatch/train.sbatch`.** Submit it with `sbatch train.sbatch`. It
   must:
   - Reserve at least 2 nodes with one task per GPU and 8 GPUs per
     node (or your cluster's actual GPUs-per-node — document what you
     used).
   - Set `--exclusive`.
   - Use `--cpu-bind=cores` and `--gpu-bind=closest` on the `srun`.
   - Compute `--nnodes` and `--nproc-per-node` from `$SLURM_*`
     variables, not hard-coded.
   - Set a rendezvous endpoint using `$SLURM_JOB_MASTER_NODE` (or
     `SLURM_JOB_NODELIST`-derived), not a hard-coded hostname.
   - Set a `--time` limit and use `--signal=B:USR1@120` to give the
     job 120 s of grace before wall clock.
   - Write logs to `logs/%x-%j.out` and `logs/%x-%j.err`.
   - Trap `SIGUSR1` in the script and log "checkpoint window" so you
     can see the plumbing works even if the training loop does not
     handle it.
2. **`prolog.d/00-gpu-health.sh`.** A prolog fragment that:
   - Drains the node if any GPU reports an *uncorrected* ECC error
     since boot (`nvidia-smi -q -d ECC`).
   - Drains the node if any InfiniBand port is not `Active`
     (`ibstat`).
   - Cleans `/dev/shm/nccl-*` and `/tmp/torch-*` from prior tenants.
   - Exits non-zero on failure so SLURM re-queues the job.
   - Logs every action with a stable `[prolog]` prefix.
3. **`WRITEUP.md`.** 300–600 words, covering:
   - Which two knobs you tuned for your cluster's topology and why
     (`--switches`, `--distribution`, `--gres`, or similar).
   - What happens to a queued job when your prolog drains a node.
   - How you would validate that `--gpu-bind=closest` is doing what
     you claim (hint: `nvidia-smi topo -m` and a note on PCIe
     locality).

## Starter guidance

- Start from the sbatch skeleton in chapter 2 §"The two commands you
  need first". Do not copy blindly — every flag you keep should have a
  one-sentence justification.
- If you cannot run a real `torchrun` (no GPUs), substitute
  `python -c "import torch.distributed as dist; dist.init_process_group('gloo'); print(dist.get_rank())"`
  and confirm you see each rank print. The point is the sbatch and
  prolog, not the model.
- Read the SLURM Quick Start User Guide
  (https://slurm.schedmd.com/quickstart.html) and the `srun` man page
  once before you write; the flag surface is large.
- Set `NCCL_DEBUG=INFO` in the job's env so your first run's log has
  the collectives NCCL picked. It is the fastest way to know the
  topology is what you think.
- If your test cluster has no `topology.conf`, note that in the
  write-up rather than pretending you tuned `--switches`.

## Acceptance criteria

- The sbatch script submits and runs `torchrun` (or the gloo
  smoke test) to completion on ≥ 2 nodes without hand-edits between
  runs.
- Every `$SLURM_*` variable used in the script comes from the SLURM
  environment (grep the man page section on "OUTPUT ENVIRONMENT
  VARIABLES"). No hard-coded hostnames, node counts, or GPU counts.
- The prolog exits non-zero for at least one synthetic failure case
  you inject (e.g., a mocked `nvidia-smi` that reports an ECC error;
  document how you injected). The job is re-queued on a different
  node, or logs show SLURM tried to.
- `logs/*.out` includes a `[prolog]` line at the start of the job
  and a `SIGUSR1 received — checkpoint window` line shortly before
  wall clock when you set a short `--time` for testing.
- WRITEUP.md is grounded in either the SLURM documentation
  (https://slurm.schedmd.com/) or the NVIDIA NCCL multi-node tuning
  notes; cite the specific page for each claim.

## Stretch goals

- Add an epilog that (a) collects `dmesg` since job start into an
  artifact under the job's output directory, (b) writes an sacct
  summary line to a per-team ledger, and (c) cleans the tenant's
  temp state.
- Create a Reservation for a scheduled run and submit into it. Show
  the `scontrol show reservation` output before and during the job.
- Compare wall-clock time-to-first-step between `--distribution=block`
  and `--distribution=cyclic` on the same allocation and explain the
  gap.
- Break `--gpu-bind=closest` deliberately (e.g., `--gpu-bind=none`)
  and measure the NCCL all-reduce busbw delta with `nccl-tests`. Note
  the number.
