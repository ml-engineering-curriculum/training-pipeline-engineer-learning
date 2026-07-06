# SLURM for Multi-Node Training: sbatch, srun, Reservations, and Prolog/Epilog

SLURM (the Simple Linux Utility for Resource Management,
https://slurm.schedmd.com/) is the batch scheduler that most research
supercomputers and a good fraction of production training clusters still
run on. If you are reading a large-model paper — Llama 3, BLOOM, OPT-175B
— the training run happened on SLURM. This chapter is the working
knowledge you need to launch, harden, and reserve capacity for a
multi-node training job on SLURM, using the primitives SLURM was actually
designed for.

## Motivation: what SLURM does that Kubernetes does not (yet) do out of the box

SLURM has a 20-year head start on gang scheduling and topology-aware
placement. Out of the box you get:

- **Gang allocation.** A job either has all its nodes or it is queued.
  No half-scheduled runs.
- **Topology plugin.** SLURM's `topology.conf` (see the topology guide at
  https://slurm.schedmd.com/topology.html) understands switches / racks and
  will keep a job's nodes on as few switches as possible.
- **Reservations.** A named block of nodes held for a user, group, or
  timespan. This is how you park capacity for a scheduled training run
  without racing the queue.
- **Prolog / epilog hooks.** Root-level scripts that run before / after
  every job step on every node — where you do GPU health checks, mount
  fabrics, drain a bad host, and clean up temp state.
- **Accounting.** `sacct` gives you the CPU-hour and GPU-hour ledger you
  need for fair-share and cost allocation. mod-109 revisits this.

## The two commands you need first: `sbatch` and `srun`

`sbatch` submits a job — a *node allocation* plus a shell script. `srun`
runs a *job step* — one or more processes across some or all of the
allocated nodes. The idiomatic training pattern is one `sbatch` per run
and one `srun` per step inside it:

```bash
#!/bin/bash
#SBATCH --job-name=llama-3-8b-pretrain
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8            # one task per GPU
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --exclusive                    # the whole node is ours
#SBATCH --time=24:00:00
#SBATCH --partition=training
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=B:USR1@120            # SIGUSR1 120 s before the wall clock

set -euo pipefail
srun --cpu-bind=cores \
     --gpu-bind=closest \
     bash -c 'python -m torch.distributed.run \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=$SLURM_NTASKS_PER_NODE \
        --node-rank=$SLURM_NODEID \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$SLURM_JOB_MASTER_NODE:29500 \
        train.py --config configs/llama-3-8b.yaml'
```

Notes on that skeleton:

- `--ntasks-per-node=8` + `--gres=gpu:8` gives you one task per GPU. Each
  task will be one rank; `torchrun` reads that from the environment.
- `--cpu-bind=cores` and `--gpu-bind=closest` pin ranks to the CPU cores
  and GPUs on the same PCIe root complex. On DGX-class hardware this
  matters a lot for NCCL performance; see the SLURM `srun` man page and
  NVIDIA's Multi-Node NCCL tuning notes.
- `--exclusive` prevents SLURM from packing another job on the same node.
  For a training job you almost always want this — noisy neighbours ruin
  step-time variance.
- `--signal=B:USR1@120` asks SLURM to send `SIGUSR1` to the batch script
  120 seconds before the wall clock expires. That is your window to
  checkpoint gracefully (mod-106 makes this concrete).
- `$SLURM_JOB_MASTER_NODE` is the resolved hostname of the first node in
  the allocation. Use it as your rendezvous endpoint; do not hard-code
  hostnames.

## Topology: getting your nodes onto the same switches

SLURM's topology plugin (`topology/tree` or `topology/block`, configured in
`topology.conf`) knows the switch tree. When you submit a job, SLURM's
selector will try to fit it on the smallest subtree that can hold it. You
have two knobs worth knowing about:

- `--switches=N[@wait]` — request that the allocation live on at most `N`
  switches, optionally waiting up to `wait` for topology to satisfy. For a
  large-model run this is the difference between "all NVLink islands share
  one leaf switch" and "half your all-reduce goes over a spine".
- `--distribution=block:block` — packs ranks by node before spreading. This
  gives you sensible defaults for both `--cpu-bind` and NCCL topology
  discovery.

There is no substitute for testing: run `nccl-tests`
(https://github.com/NVIDIA/nccl-tests) on your allocation and compare
`busbw` against the fabric's line rate. If they diverge by more than ~20%,
your topology plugin, `--switches`, or `--gpu-bind` is not doing what you
think.

## Reservations: parking capacity for a scheduled run

A **Reservation** in SLURM is a named block of nodes held out of the
regular scheduler for a user, group, or account over a specific window.
The two mechanics you will use:

```bash
# Ops-level: create the reservation.
scontrol create reservation \
    ReservationName=llama-pretrain-2026-07-08 \
    StartTime=2026-07-08T09:00:00 \
    Duration=48:00:00 \
    Nodes=gpu-[001-064] \
    Users=alice,bob \
    Flags=WEEKDAY
```

```bash
# Researcher-level: submit into it.
sbatch --reservation=llama-pretrain-2026-07-08 launch.sbatch
```

Reservations are the right tool when (a) you are cutting a training run
against a paper deadline, (b) you have a known failure-domain drain
coming, or (c) you are validating hardware after a fabric change and you
want a clean, uncontended cluster. Reservations count against the site
policy; get sign-off before you park hundreds of GPUs for a day. See the
Reservation guide at https://slurm.schedmd.com/reservations.html for the
full flag surface (`Flags=IGNORE_JOBS`, `Flags=FLEX`, partition-scoped
reservations, etc.).

## Prolog / Epilog: the "make the node safe" hooks

Every SLURM install worth using has these. They are shell scripts declared
in `slurm.conf` (`Prolog=`, `Epilog=`, `PrologSlurmctld=`, `EpilogSlurmctld=`)
that run as root before the first task and after the last task on each
node. For training clusters, the load-bearing content is:

- **GPU health.** Run `nvidia-smi -q -d ECC` and check for uncorrected
  errors; drain the node (`scontrol update NodeName=... State=DRAIN
  Reason="ECC"`) instead of running the user's job on broken hardware.
- **Fabric health.** Check IB link state (`ibstat`), RDMA counters, and
  the state of GPUDirect. A single dropped link on a 400-node job kills
  the whole run.
- **Fabric mount.** Make sure Lustre / WEKA / your parallel FS is
  actually mounted with the expected mount options; retry the mount if it
  is not.
- **Clean tmp.** `/tmp`, `/dev/shm`, HuggingFace / NCCL / torch caches
  from prior tenants. If you don't, tenant N+1 gets bit by tenant N's
  half-written checkpoint.
- **Cgroup / OOM sanity.** Make sure the cgroup limits SLURM set are what
  you expect. See the cgroup plugin docs at
  https://slurm.schedmd.com/cgroups.html.

The epilog is the mirror image plus one extra: capture NCCL debug output
and any Python coredumps you care about *before* the tmp cleanup runs.

A minimal prolog skeleton for a GPU node:

```bash
#!/bin/bash
# /etc/slurm/prolog.d/00-gpu-health.sh
set -euo pipefail

# Drain the node if any GPU has an uncorrected ECC error since boot.
if nvidia-smi -q -d ECC | grep -q 'Uncorrected.*: [1-9]'; then
  scontrol update NodeName=$(hostname -s) State=DRAIN Reason="ECC uncorrected"
  exit 1
fi

# Drain if any IB link is not Active/LinkUp.
if ibstat | awk '/State:/ {print $2}' | grep -qv Active; then
  scontrol update NodeName=$(hostname -s) State=DRAIN Reason="IB not Active"
  exit 1
fi

# Fresh caches for the tenant.
rm -rf /dev/shm/nccl-* /tmp/torch-*
mkdir -p /tmp/torch-$SLURM_JOB_USER
chown $SLURM_JOB_USER /tmp/torch-$SLURM_JOB_USER

exit 0
```

Two operational realities:

1. **Non-zero exit codes are load-bearing.** If the prolog exits non-zero,
   the node is drained and the job is requeued on a different node. That
   is exactly what you want: a bad node never poisons a run.
2. **Prolog logs are the incident record.** Whatever you log here is your
   audit trail during a post-mortem. Log to stdout with a stable prefix
   (`[prolog]`) and let SLURM's logging pipe it out.

## The "how it hangs together" picture for one training run

Putting the pieces of this chapter together, a well-behaved SLURM
training launch looks like this end-to-end:

1. Ops provisions a Reservation for the run.
2. The launcher (chapter 9) generates the sbatch script, injecting
   `--reservation=`, `--nodes=`, wall clock, and the environment.
3. `sbatch` submits; SLURM waits for `--nodes` to be free inside the
   Reservation and topology bound.
4. The prolog runs on each allocated node; bad nodes drain and requeue.
5. `srun` fans `torchrun` across all nodes; ranks call
   `init_process_group`; NCCL comes up.
6. Training proceeds. At `--time` minus 120 s, the batch script gets
   `SIGUSR1`; the trap in your script forwards it to Python; the
   training loop checkpoints and exits cleanly.
7. The epilog runs on each node; artifacts are pushed, tmp is wiped.
8. `sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,AllocGRES,State`
   is the accounting record. mod-109 uses this for cost.

Everything in chapters 3–7 either replicates that flow on Kubernetes or
abstracts it away.

## Summary

- `sbatch` reserves the allocation and defines the job; `srun` runs the
  step. Ask for GPUs with `--gres=gpu:N`, one task per GPU, and pin
  ranks with `--cpu-bind=cores --gpu-bind=closest`.
- Topology is not free — set `--switches` and confirm with `nccl-tests`.
- Reservations park capacity for scheduled runs; use them for deadlines
  and drains, not as default policy.
- Prolog / epilog are the platform's chance to keep a bad node out of a
  training run. Drain on health failure, clean caches on entry, capture
  debug artifacts on exit.
