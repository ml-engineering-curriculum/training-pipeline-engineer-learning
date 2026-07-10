# Multi-Tenant Training-Cluster Architecture

Every previous module in this track has taught you a mechanism —
collectives, frameworks, loaders, schedulers, fabrics, checkpointing,
MFU, observability, cost. This module is where you stop being the
engineer who *implements* those mechanisms and start being the
engineer who *owns the platform surface* they compose into. The
altitude is different. You are no longer the person writing the
FSDP2 config; you are the person deciding that FSDP2 is the
supported way to shard on this cluster, and that DeepSpeed is
allowed only under an exception clause, and that anyone who wants
to bring Megatron-LM 3D-parallel has to file an RFC first.

This chapter is the architecture piece. It answers a question every
platform team eventually has to answer in writing: "what, physically
and organisationally, *is* the training platform?" The answer has
three layers: a physical cluster, a scheduler stack, and a set of
platform services with a launcher SDK on top. It also has an
organisational layer: who is on-call, who escalates to whom, and
what a "war room" looks like when a run diverges at 3 a.m.

The concrete target for this chapter — the shape the exercise-01
architecture doc is graded against — is a platform that supports
5–15 downstream teams doing pretraining, large fine-tuning, eval,
and interactive research on the same hardware without stepping on
each other.

## The three-layer picture

At the whiteboard, the platform is three horizontal layers with the
launcher SDK spanning the top:

```
                    +---------------------------+
researchers  -->    |     launcher SDK (mod-104) |
                    +-----------+---------------+
                                v
+-----------------+  +---------------------------+  +----------------+
| metadata svc    |  |   scheduler / admission    |  | checkpoint svc |
| (mod-108)       |  | Kueue + Volcano + PyTorchJob  | (mod-106)     |
|                 |  |   or SLURM + QoS + partitions |                |
+-----------------+  +-------------+-------------+  +----------------+
                                   v
+-----------------+  +---------------------------+  +----------------+
| dataset svc     |  |   physical cluster         |  | metrics /      |
| (mod-103)       |  |  DGX SUs + IB spine        |  |  alerting      |
|                 |  |  Lustre / WEKA + S3        |  | (mod-108)      |
+-----------------+  +---------------------------+  +----------------+
```

Read that top-down. A researcher submits a `Job` through the
launcher SDK. The SDK writes a manifest to the metadata service,
routes to a scheduler backend, and returns a handle. The scheduler
places the job on physical nodes. The training script pulls data
from the dataset service, writes checkpoints through the checkpoint
service, and emits metrics into the metrics/alerting stack. Every
one of those boxes is a service *your team* owns and versions.

The rest of this chapter goes layer by layer, then covers the
human layer on top.

## Layer 1: the physical cluster

You built the mental model in mod-105. Compose it here into a
production topology for 5–15 teams. The shape most product-company
platforms end up with:

- **Compute** — DGX or DGX-equivalent H100/H200 nodes, grouped into
  Scalable Units (SUs) of 32 nodes with rail-aligned IB HDR/NDR. A
  medium-size platform sits at 4–16 SUs (128–2048 GPUs).
- **Storage** — a parallel filesystem (Lustre or WEKA) at 200–400
  GB/s aggregate throughput, backed by an S3 tier for cold data.
  Dataset shards live on the parallel FS; raw datasets and
  checkpoints archive to S3. Chapter 7 of mod-105 owns the details.
- **Control plane** — a small management cluster (3–5 nodes) for
  Kubernetes control plane, SLURM head, Prometheus, Grafana, the
  metadata DB, and the launcher SDK backend. Do not run control
  plane on training nodes.
- **Bastion / login** — a handful of CPU nodes with shells,
  `kubectl`, `sinfo`, and the launcher CLI. These are where
  researchers actually type.

The bare number of GPUs is misleading. The important numbers are
(a) how many contiguous rail-aligned GPUs a single job can get
(this bounds the biggest run), (b) how many independent gang-scheduled
partitions the cluster can hold at once (this bounds concurrency),
and (c) storage headroom above the sum of active-loader demand.

## Layer 2: the scheduler stack

You have two workable scheduler-stack choices from mod-104, and the
choice is roughly the same conversation as "is this shop
Kubernetes-native or HPC-native?"

**Kubernetes-native stack:**

```
Kueue                       (queues, quotas, fair-share admission)
  Volcano                   (gang scheduling, priority preemption)
    training-operator       (PyTorchJob, MPIJob CRDs)
    KubeRay                 (RayJob, Ray autoscaling worker groups)
```

Kueue owns *admission*: which workloads are allowed to enter the
scheduler right now, given per-team quotas and cluster-wide
fair-share. Volcano owns *placement*: gang-scheduling all pods of
a training job atomically, and preempting lower-priority gangs
when a higher-priority gang arrives. The training-operator and
KubeRay expose the framework-specific CRDs your launcher SDK
renders into.

**SLURM-native stack:**

```
SLURM controller
  partitions (per-team hardware pool)
  QoS classes (priority, wall time, max nodes)
  fair-share (multifactor account tree)
```

SLURM does all three jobs — admission, placement, and preemption —
in one binary. Partitions carve the cluster into per-team or
per-priority pools. QoS classes attach to partitions and encode
priority ordering, wall-time caps, and max-node limits. The
multifactor fair-share account tree encodes long-term equity
between teams.

Which stack you pick is a build-vs-buy call (chapter 4) and a
prior-art call. If the surrounding infra is already Kubernetes,
you almost always take the Kubernetes-native stack; the alternative
is running SLURM on VMs inside a Kubernetes cluster and doing both
jobs. If the surrounding infra is HPC (national lab, big-science
consortium), SLURM is already there and you cooperate with it.

## Layer 3: quotas, priority classes, and gang preemption

Now for the policy surface. This is the layer where you actually
encode "who gets the cluster right now?" and it is the layer that
gets renegotiated every time a team's headcount changes.

### Quotas

Three quota knobs per team, and one cluster-wide knob:

| Quota                | Per-team? | Enforced by                     | What it means                                    |
|----------------------|-----------|---------------------------------|--------------------------------------------------|
| Minimum guarantee    | Yes       | Kueue ClusterQueue / SLURM QoS  | Team X always gets at least N GPUs on demand     |
| Maximum burst        | Yes       | Kueue ClusterQueue / SLURM QoS  | Team X never consumes more than M GPUs           |
| Priority ceiling     | Yes       | Kueue priority class / SLURM QoS| Team X may not submit above priority `training-high` |
| Cluster-wide fair-share | No     | Kueue cohort / SLURM multifactor| Long-term equity: teams converge to their share  |

The important invariant: `sum(min guarantees) <= cluster capacity`.
If you oversubscribe minimums, you have promised the same GPU to
two teams and someone will page you.

The maximum-burst knob is what lets an idle cluster be productive:
a single team can push above its minimum during a quiet weekend and
still be preempted back when other teams show up.

### Priority classes

Four priority classes cover the great majority of production cases.
This is the ordering, high to low:

| Priority                | Use case                                          | Preemptable? | Max wall time |
|-------------------------|---------------------------------------------------|--------------|---------------|
| `critical`              | On-call incident response, scheduled maintenance  | No           | 2 h           |
| `training-high`         | Sanctioned pretraining runs                       | Only by `critical` | 7 d       |
| `training-normal`       | Large fine-tunes, batched evals, capacity backfill | Yes         | 2 d           |
| `research` / `interactive` | Notebooks, small debug jobs                    | Yes, aggressively | 8 h        |

Two rules make this workable in practice: `critical` is capped at
a small fraction of cluster capacity (say 5 %), because otherwise
an on-caller with a sharp tool can accidentally drain the fleet;
and `research` is the *backfill* class, meaning it runs on whatever
is not currently occupied by `training-*` and is expected to be
preempted regularly. The interactive-notebook UX absorbs this
because interactive users check back in short cycles anyway.

### Gang preemption

Preemption of a training gang is not free. When a `training-high`
job preempts a `training-normal` job, the platform pays:

- The wall-clock time between the preempted job's last checkpoint
  and its preemption (lost work).
- The time to save any pending in-memory state, if the job supports
  a signal-based flush (mod-106 chapter 3).
- The rendezvous + warmup cost when the preempted job re-lands.

The platform-level rule that keeps this sane: **preemption is only
allowed against jobs whose checkpoint interval is smaller than the
expected preemption cost.** If a team ships a job that checkpoints
every 8 hours and asks to run at `training-normal`, they are
consenting to lose up to 8 hours of work per preemption. Encode
this in your launcher SDK: `training-normal` requires
`checkpoint_interval <= 30 min`; `training-high` requires
`checkpoint_interval <= 2 h`. Reject jobs that violate.

## Layer 4: the platform services

The scheduler and the fabric are the load-bearing parts, but the
platform is more than that. "The platform" is a set of services
your team owns, each with a versioned API and an SLO:

- **Launcher SDK** (mod-104 chapter 9) — the researcher-facing
  Python + CLI surface. Owns `Job` submission, routing, ledger
  writes, and the failure-mode contract (rendezvous timeout,
  preemption signal, checkpoint discovery, wall-clock warning).
  SLO: p99 submission latency ≤ 10 s.
- **Metadata service** (mod-108) — an append-only ledger of every
  run: submitter, code SHA, image digest, config digest,
  scheduler args, dataset hash, tokenizer hash, container digest,
  hardware manifest, final loss, MFU. Backed by Postgres or an
  equivalent. SLO: 99.9 % writes durable, p99 read latency ≤ 200 ms.
- **Checkpoint service** (mod-106) — the write path to the parallel
  filesystem for DCP shards + the archival path to S3. Owns
  retention policy, resharding on restore, and the "latest good"
  pointer. SLO: p99 checkpoint save time within X % of the raw FS
  throughput budget; loss-of-latest-checkpoint = P0.
- **Dataset service** (mod-103) — the read path for training shards.
  Owns shard-manifest publication, sampler determinism guarantees,
  and cache warmup on job admission. SLO: loader stall time ≤ 5 %
  of step time (goodput floor).
- **Metrics + alerting stack** (mod-108) — Prometheus + Grafana for
  cluster-wide roll-ups, DCGM per-node, W&B / TensorBoard for
  per-run researcher-facing dashboards. Owns the paging rules.
  SLO: DCGM lag ≤ 30 s; alerts fire within 60 s of the underlying
  telemetry crossing threshold.
- **Incident-runbook wiki** — the *human* service. Owns the loss
  spike / NCCL timeout / silent-corruption runbooks (mod-106 chapter
  6) plus platform-side runbooks (Kueue admission stuck, IB fabric
  degradation, storage tier full). Anything the on-caller is
  expected to know how to handle lives here.

Every service has an owner name, a Slack channel, a paging rotation,
and an SLO the platform team publishes and lives against. The SLOs
are what let you say "no, that's a P0" or "no, that's a P3" without
argument.

## Layer 5: on-call and escalation

The organisational layer sits on top and is the layer new architects
most consistently under-invest in.

### Rotation

A workable on-call rotation for a 6–10 person platform team:

- **Primary on-call** — one week at a time, carries the pager.
  Handles P0/P1 within 15 min, P2 within 4 h, P3 next business day.
- **Secondary on-call** — one week at a time, backup if primary is
  unreachable or stuck. Also on the "war-room bridge" when a P0
  escalates.
- **Architect on-call** — a one-month rotation of your senior
  engineers. Not paged directly, but consulted for
  cross-team-visible decisions (e.g. "should we pause pretraining
  runs for a fabric maintenance window?").

Rotations are published a quarter in advance. Follow-the-sun works
above ~8 people if you have geo distribution; below that,
US-daytime + US-weeknight is more honest.

### Escalation ladder

Every incident has an escalation ladder, and the ladder is
*written down* so the primary is not making it up at 2 a.m.:

```
P0 (cluster-wide outage / data loss risk)
   -> page primary + secondary immediately
   -> if not acked in 5 min, page architect on-call
   -> if scope > single team, open war-room bridge and post to #train-incident
   -> notify affected teams within 15 min
   -> notify leadership within 30 min if ETA > 2 h

P1 (single-team run failing repeatedly, storage tier degraded)
   -> page primary
   -> if not resolved in 1 h, page secondary
   -> notify affected team within 30 min

P2 (single-run failure with a known runbook, straggler node,
    scheduled quota renegotiation)
   -> primary handles in business hours
   -> file a ticket, close on runbook

P3 (cosmetic dashboard bug, single missed metric, doc gap)
   -> queue for next planning cycle
```

Two subtleties. First, the definition of P0 vs. P1 vs. P2 is
platform-team-defined; publish it and enforce it. Second, the
notification-to-affected-teams timeline is a *contract*. If a
pretraining run is going to lose 8 hours because you're doing an
emergency fabric restart, the pretraining team needs to know
inside the SLA window, not when they notice their loss curve is
flat.

### War-room protocol

For a real P0, spin up a war room. Concrete pieces:

- **Bridge** — a persistent Slack / Zoom channel with an owner
  (usually the secondary on-call, acting as scribe).
- **Incident commander** — one named human. Anyone else who needs
  to talk to leadership talks *through* the IC. The IC does not
  fix bugs; the IC coordinates.
- **Roles named** — subject-matter leads (fabric, scheduler,
  storage), scribe, comms. Roles are named at the start of the
  war room; if there are two people on a role, they explicitly
  split responsibilities.
- **Ticker** — the scribe posts a timestamped ticker of every
  hypothesis, every action, every observation. This is the raw
  material of the post-mortem (chapter 5).
- **Blast-radius cadence** — every 30 min the IC posts the current
  scope estimate ("still contained to SU-3") and the current ETA
  ("expect first pretraining run to resume by 06:00").

The most common failure mode of an inexperienced war room is
everyone typing into the channel with no scribe and no IC. In four
hours you have 800 messages and no timeline. The scribe + IC
convention is what prevents that.

## Reference topologies for three sizes

Three concrete platform shapes. These are starting points for the
exercise-01 doc.

**Small-lab platform (2 teams, 128 GPUs, 16 nodes)**

- One IB SU, one Lustre tier, no S3 hot tier.
- SLURM on 3 head nodes, two partitions (`train`, `interactive`),
  three QoS classes.
- Launcher SDK with a single backend (SLURM).
- Metadata in Postgres on the head node.
- One primary + one architect. War room = a Zoom link in the
  runbook.

**Product-company platform (8 teams, 1024 GPUs, 128 nodes)**

- Four IB SUs, WEKA tier, S3 archive.
- Kubernetes + Kueue + Volcano + PyTorchJob + KubeRay.
- Launcher SDK with three backends (SLURM for legacy HPC-style
  runs, Kubernetes for the default, local for smoke tests).
- Metadata + checkpoint + dataset services each as their own
  Deployment with its own SLO.
- Primary + secondary weekly rotation, architect monthly.

**Frontier-lab platform (15 teams, 20 000 GPUs, 2500 nodes)**

- Multiple pods, each 1–2 SUs, cross-pod IB spine at 400 Gb/s per
  rail. WEKA + Lustre bi-modal for different loader profiles.
- Kubernetes federation across pods; Kueue cohorts route across
  pods.
- Launcher SDK with multi-cluster routing (mod-104 chapter 9).
- Dedicated storage SRE, dedicated fabric SRE, dedicated
  scheduler SRE — the platform team decomposes into
  sub-services. Cross-service war rooms need an IC layer above.

The frontier-lab shape is not what you build first. It is the
shape a mature platform grows into over 2–4 years, and it is the
shape the OPT-175B logbook (chapter 5) describes when the
BigScience and Meta AI teams talk about "the platform." That
shape is not architecturally different from the product-company
platform; it is the product-company platform with more of each
service, more careful federation, and a bigger org chart around
it.

## Summary

- The training platform is three layers — physical cluster,
  scheduler stack, platform services — with a launcher SDK on
  top and an organisational layer around them.
- The scheduler stack is either Kubernetes-native (Kueue +
  Volcano + training-operator + KubeRay) or SLURM-native
  (partitions + QoS + fair-share). Pick based on the surrounding
  infra, not on which is "better."
- Quotas have three per-team knobs (min guarantee, max burst,
  priority ceiling) and one cluster-wide knob (fair-share). The
  sum of minimum guarantees must not exceed cluster capacity.
- Four priority classes (`critical`, `training-high`,
  `training-normal`, `research`) cover almost every real workload.
  Encode the "you may only preempt if your checkpoint interval is
  smaller than the preemption cost" rule in the launcher SDK.
- The platform is not just the scheduler. It is a set of services
  (launcher SDK, metadata, checkpoint, dataset, metrics,
  runbooks) each with an owner and an SLO.
- The organisational layer is load-bearing: rotation, escalation
  ladder, war-room protocol. Write it down before you need it.
- The frontier-lab shape is not architecturally different from the
  product-company shape; it is the same design at higher
  cardinality with a bigger org chart.
