# torchrun, Rendezvous, and Elastic Reshape

Chapter 3 gave you a training loop that can *resume* from a
checkpoint on any world size. This chapter is about the machinery
that decides *what world size the run gets to resume on*. When a
node crashes mid-epoch, something has to notice, remove the dead
rank from the process group, either wait for a replacement or
proceed with fewer ranks, and re-invoke your training script with
the new topology. In PyTorch that machinery is
`torch.distributed.elastic`, exposed to users through the
`torchrun` launcher.

Two references live open on this chapter:

- **`torchrun` docs.**
  https://pytorch.org/docs/stable/elastic/run.html
- **`torch.distributed.elastic`.**
  https://pytorch.org/docs/stable/elastic/ — rendezvous, agent
  design, membership changes, monitored barriers.

## Why `torchrun` and not `python -m torch.distributed.launch`

The older `torch.distributed.launch` launcher does one thing: it
`fork`s N worker processes with the right environment variables
(`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) and then
waits. If any worker dies, `launch` cannot rebuild the process
group — the whole job dies.

`torchrun` (introduced in PyTorch 1.9 as the successor to
`torch.distributed.launch`) is built on the elastic agent. It
does everything `launch` does *and*:

- Discovers peers through a **rendezvous** (c10d, static, or etcd
  backend) rather than a fixed master address.
- Watches its local worker group and either restarts locally
  (`max_restarts > 0`) or triggers a re-rendezvous when the
  group changes.
- Supports **`min_nodes`** and **`max_nodes`** as separate
  parameters, so the run can proceed with fewer nodes than it
  started with.
- Coordinates a **membership-change event** across all agents so
  every rank re-enters `init_process_group` with a consistent
  view of the new topology.

On any run large enough to hit the failure rates chapter 1
described, `torchrun` is the launcher of record. Do not roll your
own.

## Anatomy of a `torchrun` launch

A production launch line looks roughly like this — the exact form
varies by cluster; consult the docs for the current syntax and the
mod-104 launcher patterns you already know:

```bash
torchrun \
  --nnodes=64:96 \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$HEAD_NODE_ADDR:29500 \
  --rdzv-id=run-2026-07-09 \
  --max-restarts=3 \
  --monitor-interval=5 \
  train.py --config /etc/train/config.yaml
```

The important flags, in the order of what they buy you:

- **`--nnodes=MIN:MAX`.** Elastic mode. The job can run at any
  node count between MIN and MAX. torchrun waits at rendezvous
  until at least MIN nodes join, then proceeds.
- **`--rdzv-backend`.** How ranks find each other. `c10d` is the
  default (uses PyTorch's own store); `etcd` uses an external
  etcd cluster and is what you want for high-availability
  rendezvous. Static rendezvous exists but is not elastic.
- **`--rdzv-endpoint`.** The rendezvous coordination endpoint. For
  `c10d`, this is one node in the run (often the "head" node).
  For `etcd`, it is the etcd cluster.
- **`--rdzv-id`.** A run-level identifier. All agents that share
  a run must share this ID; agents from other runs must have a
  different one. Common pitfall: reusing the same ID across two
  parallel training runs and having them clobber each other's
  rendezvous state.
- **`--max-restarts`.** How many times an agent will re-rendezvous
  after a failure before giving up on the whole run. Zero is
  fail-fast (test / bring-up); production usually runs at 3–5.
- **`--monitor-interval`.** How often the agent polls the
  rendezvous store for membership changes.

On a SLURM cluster this launch line usually ends up wrapped by a
`sbatch` script that also sets `--rdzv-endpoint` to
`$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)`;
on Kubernetes (via TorchX or the MPI Operator) the endpoint is
usually the launcher pod's DNS name. Those wiring choices live in
mod-104.

## The rendezvous protocol, at the level you need it

Rendezvous is a distributed consensus problem: given N agents
starting at unpredictable times, decide which subset forms the
current worker group and give each of them a consistent global
rank. PyTorch's docs describe the protocol; the mental model you
need for on-call is:

1. **Every agent registers with the rendezvous backend** at
   startup with `(rdzv-id, node-id)`.
2. **The backend forms a "round"** — a snapshot of the currently
   registered agents. Rounds are numbered.
3. **When the round hits `min_nodes`**, the backend closes it and
   assigns global ranks to agents deterministically (e.g., by
   node-id).
4. **Agents call `init_process_group`** with the assigned ranks
   and world size. The training script proceeds.
5. **On any agent failure or new agent joining**, the backend
   opens a new round. Every agent must exit its process group
   and re-rendezvous.

Two properties of the protocol that matter operationally:

- **Rounds are monotonically increasing.** An agent that stayed up
  through a membership change knows a new round has started
  because the round number went up. This is how the on-membership
  handler (below) triggers.
- **The rendezvous backend is a single point of failure.** For
  `c10d`, that is the head node. If the head crashes, no new
  round can be formed even if all other nodes are healthy. On
  runs where the head is on commodity hardware, use `etcd`
  instead.

## `on_membership_change`: the training-script contract

`torch.distributed.elastic` exposes a hook you must implement to
survive rank changes: your training script has to detect that
membership has changed, tear down the process group, re-enter
rendezvous, and rebuild the training state at the new world size.

In practice, the shape looks like this. See the elastic docs for
the exact current API:

```python
import torch.distributed as dist
import torch.distributed.elastic as elastic

def train(cfg):
    while True:
        # (Re-)enter rendezvous.
        rdzv = elastic.rendezvous_handler(cfg.rdzv_endpoint,
                                          cfg.rdzv_id)
        world_info = rdzv.next_rendezvous()
        rank, world_size = world_info.rank, world_info.world_size

        dist.init_process_group(backend="nccl",
                                rank=rank, world_size=world_size)

        try:
            # Rebuild model, optimizer, loader at the new world size.
            model = build_and_shard_model(world_size)
            optim = torch.optim.AdamW(...)
            loader = build_stateful_loader(...)

            # DCP resume — DCP does not care about world size.
            maybe_resume(cfg.ckpt_root, model, optim, loader,
                         train_state)

            for batch in loader:
                # ... training step, save-on-interval ...

                if elastic.membership_changed():
                    # Someone joined or left. Break to re-rendezvous.
                    break
        finally:
            dist.destroy_process_group()
```

Two properties this shape gives you:

- **The training script's outer loop is world-size-agnostic.**
  Every time through, the script asks the rendezvous handler
  "what's the current world size?" and rebuilds the model to
  match.
- **DCP does the actual state restore.** The training script
  never has to know how to redistribute optimizer state between
  ranks — DCP's load planner does that.

In torchtitan the equivalent shape is embedded in the
`train.py` script; the exercise 02 asks you to trace it.

## Restart-in-place vs. restart-with-fewer-nodes

Two elastic-reshape variants matter operationally. They are
different both in what the scheduler does and in how the run
recovers.

### Restart-in-place

**What triggers it.** A worker process crashes (`SIGSEGV`, unhandled
Python exception, OOM) but the node itself is fine. torchrun's
agent restarts the worker locally, up to `max_restarts` times.

**What the training script sees.** The process group is torn down
and re-formed with the same world size. DCP reloads the most
recent checkpoint. Total downtime is on the order of tens of
seconds (rendezvous + DCP load).

**When it does not work.** If the crash is deterministic — a bad
model config, a bad batch that will always OOM at that shape —
each restart hits the same bug and you burn all your `max_restarts`
in a loop. The runbook in chapter 5 covers "crash loop" detection.

### Restart-with-fewer-nodes (elastic reshape)

**What triggers it.** An entire node goes away (host reboot,
network partition, scheduler evicted the pod). Agents on other
nodes detect the missing peer and re-rendezvous.

**What the training script sees.** Rendezvous forms a new round
at a smaller world size (`current_nodes` is now `MIN <= x <
original`). DCP reloads at the new world size. The training
script rebuilds model and optimizer at the new mesh shape
(fewer DP replicas, or the same TP/PP with fewer DP).

**When it does not work.** If the new world size drops below
`MIN`, torchrun stops. Also, if the parallelism topology is not
compatible with the new world size — e.g., you saved at `TP=8`
but a single node is now missing four of its GPUs — the reshape
fails because the TP grid does not tile. The typical fix is to
constrain the run's `MIN` to a whole-node granularity.

### The two elastic reshape idioms

For pure-DP or FSDP runs, elastic reshape is almost free: DP
replicas are exchangeable, so dropping one is a straightforward
reshape.

For 3D-parallel runs (mod-101 chapter 4), it is much less free
because TP and PP groups have topological constraints (TP inside a
node, PP along a specific ring). The pragmatic pattern most
production runs use:

- **Elastic in the DP dimension only.** Configure the launcher so
  `MIN` and `MAX` differ only by whole DP replicas (usually
  whole nodes). TP and PP degrees stay fixed. The DCP load
  planner redistributes DP shards across whatever DP degree the
  new round has.
- **Non-elastic TP/PP.** If a TP group loses a node, the round
  fails. The scheduler is expected to bring the run back on a
  full replacement or reduce the DP degree cleanly.

This restriction is the price of admission for 3D-parallelism at
scale. mod-101 chapter 4 discussed why TP groups have to fit
inside one node's NVLink domain; that same constraint is what
prevents them from reshaping without more thought.

## The elastic-reshape flow, end to end

Put the pieces together. Here is the wall-clock sequence when a
node crashes on a 64-node FSDP run configured with
`--nnodes=60:64` and `--max-restarts=3`:

1. **T+0s:** Node 37 host-panic. Its 8 workers die.
2. **T+~5s:** Other agents' watchdog detects that node 37's
   agent is not heart-beating. Rendezvous backend opens a new
   round.
3. **T+~10s:** All remaining agents call `next_rendezvous()`. The
   round completes at 63 nodes (still above `MIN=60`).
4. **T+~10s:** Every rank exits its current `init_process_group`,
   creates a new one at world size 504 (63 · 8).
5. **T+~15s:** Every rank rebuilds its model / optimizer /
   loader at the new mesh shape.
6. **T+~30s:** DCP load pulls the latest checkpoint (say step
   12 500) and redistributes shards. The stateful loader
   resumes at its saved iterator position.
7. **T+~35s:** First forward pass at the new world size begins.

Total downtime: ~35 seconds. Retention loss: from step 12 500 to
whenever the crash happened (~250 steps if you check every 250).
Availability cost: the 35 seconds plus, over the rest of the run,
the throughput hit from having ~1.5% fewer GPUs.

That is the win. Without elastic reshape, the alternatives are:

- **Wait for a replacement node.** SLURM / Kueue re-queue can
  take tens of minutes to hours depending on cluster load.
  During that wait, the whole gang sits idle.
- **Kill the run and requeue at 64 nodes.** Same wait, plus a
  full DCP load.

Elastic reshape trades a slice of throughput for a huge slice of
availability. The chapter 7 SLO exercise walks through when the
trade is worth taking.

## What the scheduler owns

`torchrun` handles the *process* side of elastic reshape. It does
not handle:

- **Draining a node.** SLURM's `scontrol update NodeName=<n>
  State=drain` and Kubernetes' `kubectl cordon` + `kubectl
  drain` still live at the scheduler level. Chapter 6 covers
  auto-quarantine.
- **Provisioning a replacement.** If the run wants to grow back
  from 63 nodes to 64 nodes after a repair, the scheduler has to
  produce the new node and the launcher has to notice it join
  the rendezvous round.
- **Job-level requeue on total failure.** If the whole gang
  cratered (rendezvous couldn't form at `MIN`), the scheduler's
  requeue policy is what decides whether the job restarts from
  scratch.

The joint contract with mod-104: the scheduler owns gang
placement and requeue; the launcher owns the process group and
membership changes. You need both.

## What `torchrun` does not fix on its own

Two failure modes elastic reshape does *not* handle, and where
the runbook takes over:

- **Deterministic crash loop.** Every restart hits the same bug.
  `max_restarts` decrements to zero and the whole run dies. The
  runbook wants a Prometheus alert on "restart count in the last
  10 minutes > K" (mod-108).
- **Rendezvous store failure.** If the head node running the
  `c10d` rendezvous crashes, the whole run cannot re-rendezvous.
  Move to `etcd` for HA rendezvous on any run large enough to
  care.

Both belong in the runbook. Chapter 5 owns those cases.

## Summary

- `torchrun` is the launcher of record. It uses
  `torch.distributed.elastic` under the hood and adds
  membership-change handling that `torch.distributed.launch`
  lacks.
- The three rendezvous backends are `c10d` (default, head-node
  based), `etcd` (highly available, recommended at scale), and
  `static` (not elastic).
- Elastic reshape variants: **restart-in-place** (worker crashes,
  node is fine) and **restart-with-fewer-nodes** (whole node
  gone). Both funnel through re-rendezvous; DCP handles the
  state redistribution.
- Practical rule: make elastic reshape only reshape the DP
  dimension. TP and PP topology is too constrained to reshape
  cheaply.
- End-to-end elastic-reshape downtime is on the order of tens of
  seconds for a run that checkpoints frequently. Compared to
  waiting for a replacement node, it is the primary way
  availability stays high enough to hit a goodput SLO.
- `torchrun` does not own node drain, replacement provisioning,
  or job-level requeue. The joint contract with mod-104 is:
  scheduler owns gang placement + requeue; launcher owns the
  process group.
- Failure modes torchrun does not fix — deterministic crash loop
  and rendezvous store failure — belong to the on-call runbook
  in chapter 5.
