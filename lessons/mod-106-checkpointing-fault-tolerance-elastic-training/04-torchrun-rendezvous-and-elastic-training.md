# torchrun, Rendezvous, and Elastic Training

Chapter 3 gave you a checkpoint that can be re-loaded onto a
different world size. This chapter gives you the launcher plumbing
that actually delivers a different world size after a failure — the
rendezvous protocol underneath `torchrun`, the two membership models
you can choose from, and the state machine that a node crash puts the
job through.

Keep these primary references open:

- **PyTorch Elastic (`torch.distributed.elastic`) documentation.**
  https://docs.pytorch.org/docs/stable/distributed.elastic.html —
  agent, rendezvous, run, and `torchrun` reference.
- **PyTorch `torchrun` documentation.**
  https://docs.pytorch.org/docs/stable/elastic/run.html — the CLI
  and the semantics of `--nnodes`, `--rdzv-backend`,
  `--max-restarts`.
- **Distributed elastic design doc.**
  https://docs.pytorch.org/docs/stable/elastic/train_script.html —
  the semantics your training script has to satisfy so that a
  restart is safe.

## The three actors in an elastic run

Every elastic PyTorch job has three components running on each node:

1. **The rendezvous backend.** A distributed key-value store the
   agents talk to. It is the source of truth for "who is in this
   job right now" and the coordination point for membership changes.
   The two supported backends in modern PyTorch are `c10d`
   (in-tree, uses the built-in TCPStore) and `etcd-v2` (external).
2. **The agent (`torch.distributed.elastic.agent`).** One agent per
   node. It talks to the rendezvous backend, decides when to join a
   rendezvous, launches the worker processes on that node, watches
   them, and reports back on their success or failure.
3. **The workers.** The actual training processes, one per GPU on
   the node. They receive their `RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
   and `MASTER_ADDR` / `MASTER_PORT` from the agent and use those
   to `init_process_group`.

`torchrun` is the CLI wrapper that sets up the agent and, indirectly,
the rendezvous backend. Everything below is what `torchrun` is doing
behind the scenes.

## Rendezvous: static vs. elastic

Two modes with very different failure semantics.

### Static rendezvous

The default for the classical `torchrun --nnodes=N` invocation with
`N` a single integer. Every node contributes a fixed number of
workers, the world size is exactly `N * nproc_per_node`, and if any
worker process dies, the whole job dies (or restarts up to
`--max-restarts` times, always at the same world size).

Use static rendezvous for jobs that assume a fixed shape and want
"kill everything on any failure" semantics — the default for
non-elastic production runs, and the correct baseline choice for
runs where a partial recovery would be worse than a fresh start.

### Elastic rendezvous

Enabled by specifying a range: `torchrun --nnodes=MIN:MAX`. The
job launches when at least `MIN` nodes have joined the rendezvous,
and it accepts up to `MAX`. If a node fails, the survivors can
re-rendezvous at the new (smaller) node count, provided it is still
`>= MIN`. The `RANK` and `WORLD_SIZE` change on every
re-rendezvous; the workers restart with new environment variables
and re-`init_process_group`.

Elastic is what enables the "keep training after a node crash"
flow. But — and this is the key contract — the training script
itself must be re-entrant: on every rendezvous change, the workers
are killed and restarted from the entry point of the training
script. If the script does not read the last checkpoint from disk
on startup, elastic buys you nothing.

## The rendezvous handshake

The rendezvous is a barrier-with-membership protocol. Detailed step
by step, when `torchrun --rdzv-backend=c10d --nnodes=MIN:MAX` starts
on a node:

1. The agent connects to the rendezvous backend at
   `--rdzv-endpoint`. This is the c10d TCPStore, or an etcd
   endpoint, or a static address; for c10d, one node is designated
   as the store server via `--rdzv-endpoint` and the rest connect
   to it.
2. The agent registers itself in the `--rdzv-id`'s pending pool.
   Each agent contributes `nproc_per_node`.
3. Once the pending pool has at least `MIN * nproc_per_node`
   members and a **rendezvous timeout window** has elapsed since
   the last join, the barrier closes. The rendezvous transitions
   from "gathering" to "closed for this generation".
4. The backend assigns each agent a **generation number** and a
   contiguous range of `RANK` values for its workers. The agent
   launches its workers with those env vars and
   `MASTER_ADDR=<generation leader>`.
5. Each worker calls `torch.distributed.init_process_group(...)`
   using the assigned rank / world size. NCCL communicators come
   up. Training begins.

On any subsequent membership change (a node dies, a node arrives
and `MAX` was not reached), the whole cycle repeats with a new
generation number. Workers from the previous generation are
terminated by their agents.

Two knobs on this handshake you will care about at scale:

- **`--rdzv-conf join_timeout=<seconds>`.** How long the rendezvous
  waits for late joiners after the first `MIN` nodes have arrived.
  Set larger than you think you need on cold starts (nodes come up
  at different rates); set smaller than your incident-recovery
  budget so you do not sit at the rendezvous during a real
  incident.
- **`--rdzv-conf timeout=<seconds>`.** Overall rendezvous timeout.
  If nodes have not joined in this window, the rendezvous fails
  and the job errors out. Distinct from `--max-restarts`, which
  gates worker-level restarts.

Exact key names and defaults change between PyTorch versions.
Confirm against the version you are running.

## What happens on a node crash mid-epoch

The full state-machine walk-through, because this is the flow that
the elastic contract exists for. Setup: a 16-node job launched with
`torchrun --nnodes=8:16` (min 8, max 16). Node 5's worker OOMs at
step 12 345. Assume DCP is checkpointing every 500 steps and the
last successful checkpoint is at step 12 000.

1. **T=0 s.** Worker on node 5 raises. The agent on node 5 sees the
   worker exit with a non-zero code.
2. **T~1 s.** The other 15 nodes are still in an NCCL collective —
   an all-reduce or all-gather in the training step. They are
   *waiting on rank(s) from node 5*.
3. **T~few s.** The agent on node 5, having noticed a dead worker,
   marks itself as having failed the current generation and either
   (a) tries to relaunch its workers up to `--max-restarts`, or (b)
   drops out of the rendezvous.
4. **T=T_nccl_timeout.** NCCL's watchdog fires on the other 15
   nodes. The `init_process_group`'s timeout kicks in (default 10
   minutes for NCCL, tunable via the `timeout=` kwarg). Each
   surviving worker raises a `DistBackendError` or equivalent.
   Their agents catch it.
5. **T=T_nccl_timeout + ε.** The 15 agents transition to
   "re-rendezvous". If the failed node comes back, it may rejoin;
   if not, the 15 agents wait for the join window and then form a
   new generation of 15 nodes.
6. **T=T_nccl_timeout + T_rendezvous.** New generation assigned.
   Each agent kills its old workers (they are already dead but the
   kill is idempotent) and launches new workers with the new
   `WORLD_SIZE = 15 * nproc_per_node`.
7. **T=T_nccl_timeout + T_rendezvous + T_script_startup.** New
   workers hit the training script's entry point. They read the
   last DCP checkpoint at `/ckpt/step-12000/`. DCP reshards from
   the 16-node layout onto the 15-node layout automatically
   (chapter 3). Optimizer, LR scheduler, RNGs, sampler position
   restore.
8. **T=T_nccl_timeout + T_rendezvous + T_script_startup +
   T_ckpt_load.** Training resumes at step 12 001. The 345 steps
   between the last checkpoint and the crash are lost work — the
   `Δt / 2` term from chapter 2's cost model, times two because we
   were near the top of the interval.

Two implications:

- The single biggest win is **shortening T_nccl_timeout**. The
  default 10-minute NCCL timeout is far too long for an elastic
  workflow — you burn 10 minutes of 15 nodes for every failure. The
  usual fix is `init_process_group(..., timeout=timedelta(minutes=1))`
  or similar, plus a watchdog that fails faster than NCCL on
  known-fatal signals.
- The second biggest win is **making T_script_startup + T_ckpt_load
  small**. Chapter 3's DCP `async_save` frequency and chapter 2's
  storage-side load bandwidth are what determine this. The
  no-work-lost number is `T_ckpt_interval / 2`; the added recovery
  overhead is the four other terms above.

## The training script's re-entrant contract

The elastic model requires your training script to be
**re-entrant**: it must work correctly when started fresh (initial
launch) and when restarted after a rendezvous change (recovery).
Concretely:

1. **Idempotent setup.** `init_process_group`, DCP setup, dataset
   iterator construction — all must work on both fresh start and
   restart.
2. **Read-checkpoint-if-exists on startup.** The script's first job
   after `init_process_group` is to check for the most recent DCP
   checkpoint and load it. Chapter 3's atomic-rename pattern is
   what makes "most recent" safe.
3. **Reproducible state from `(step, seed)`.** After load, the
   sampler / RNG / LR scheduler are all functions of the loaded
   step. No side channels — no wall-clock-time-based randomness,
   no reading from a file that changes between generations.
4. **No worker-local mutable global state that has to persist.**
   Anything you need across restarts goes into `AppState` and thus
   into the DCP checkpoint. Anything you do not need across
   restarts is fine to lose.

The `torchrun` docs page linked at the top of this chapter has a
worked "elastic-safe train script" example; the shape matches
mod-102 chapter 4's torchtitan trainer skeleton.

## Rendezvous backend choice

Two production backends in current PyTorch:

- **c10d (in-tree).** Uses PyTorch's own `TCPStore` as the store.
  One node is the "endpoint" and hosts the store; every other node
  connects to it. No external dependency. Correct default for
  single-cluster jobs where a control-plane node can be designated.
- **etcd-v2 (external).** Uses an existing etcd cluster as the
  store. Better when you have multiple concurrent jobs and want
  a durable store that survives the death of any one node,
  including the "endpoint". Common on Kubernetes deployments where
  a cluster-wide etcd is already in place.

Older Kubernetes-native etcd rendezvous variants exist and have
been through several renames; consult the elastic docs page for the
current recommendation on your PyTorch version.

## Interaction with the scheduler

Elastic training only makes sense if the scheduler will actually
*let* the job shrink and grow. Two scheduler patterns:

- **Fixed-gang scheduling.** SLURM's default and Kubernetes'
  default with a fixed replica count. The scheduler grants you N
  nodes for the whole run; if a node dies, you might get a
  replacement, but you might not. Elastic can help by letting the
  job keep running at N-1 while you wait for a replacement.
- **Elastic-friendly scheduling.** Kueue with
  `PodGroupTopologyAwareScheduling`, Volcano's elastic plugin,
  KubeRay's elastic training operator, or the Karpenter-style
  "add and remove nodes as needed" cloud auto-scalers. These
  cooperate with the rendezvous by adding new nodes to the
  `pending` pool automatically.

The details are mod-104's problem; from this chapter's perspective,
the important contract is that the scheduler and the rendezvous
agree on **what the current gang is** at each rendezvous
transition. If the scheduler adds a node while the rendezvous has
already closed for the current generation, the new node waits at
the pending pool until the next rendezvous — usually triggered by
another failure.

## Failure modes to plan for

Three specific ones that the flow above does not cover cleanly, and
what to do:

- **Split-brain rendezvous.** Two subsets of nodes lose network
  connectivity between them. Each forms its own rendezvous. Both
  believe they are the surviving majority. The c10d backend is
  susceptible; etcd-based backends are less so, because etcd's
  Raft quorum will only elect one leader. Mitigation: run the
  rendezvous backend on a node whose network partition is *less*
  likely than the compute fabric (e.g., a management-network
  address).
- **Rendezvous store dies.** The endpoint node itself crashes.
  With c10d, the whole rendezvous is gone; the job cannot recover
  without an operator relaunching from an entirely new endpoint.
  With etcd, the store keeps quorum through the crash. This is
  the main reason production platforms with strict availability
  targets tend to run etcd rendezvous.
- **The failed node comes back mid-rendezvous.** A node crashed
  and its scheduler-side replacement was already coming up when
  the original recovered. Both try to join the same rendezvous.
  You end up with an over-provisioned generation. Usually
  harmless if `MAX` is high enough; occasionally results in a
  duplicate-rank error if the agent implementation does not
  handle the race well. Set `MAX` deliberately and read your
  agent version's release notes.

Chapter 5's incident classification includes a runbook entry for
each of these.

## Summary

- Every torchrun-launched job has three actors: rendezvous backend,
  agent per node, and workers per GPU. Elastic training is a
  cooperation between the agent and the rendezvous, mediated by
  the training script's re-entrant contract.
- Static vs. elastic rendezvous is a choice you make at launch:
  static (`--nnodes=N`) means "kill on any failure", elastic
  (`--nnodes=MIN:MAX`) means "reshape and continue".
- The mid-epoch crash flow is: worker dies → NCCL times out on
  survivors → agents re-rendezvous at new world size → workers
  relaunch → training script reads last DCP checkpoint → resume.
  Optimize `T_nccl_timeout`, checkpoint interval, and load time
  in that order.
- The training script must be re-entrant: idempotent setup,
  read-checkpoint-if-exists on start, deterministic post-load
  state, no mutable globals outside the checkpoint. Chapter 3's
  `AppState` is what carries the "outside the checkpoint" list
  down to nothing.
- Backend choice (c10d vs. etcd) trades external dependency for
  robustness of the rendezvous itself. Production platforms with
  strict availability targets usually run etcd.
- Split-brain, store-death, and racy-recovery are the three
  failure modes to design for on top of the happy path. Chapter 5
  is the runbook layer.
