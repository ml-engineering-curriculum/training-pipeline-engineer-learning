# Quotas, Fair-Share, Priority, and Gang Preemption

A shared training cluster is a multi-tenant system whether or not you
call it that. Two teams will overlap. Someone will submit 40 jobs at
2 AM. A paper deadline will push a team to want 512 GPUs for a week.
Absent a policy, the first symptom is "my job has been queued for six
hours" and the second symptom is a meeting.

This chapter is the working knowledge you need to design the
multi-tenant policy layer for a training cluster: quotas, fair-share,
priority classes, and gang preemption on both SLURM and Kubernetes.
The goal is the smallest set of policies that (a) makes utilisation
high, (b) gives every team a floor they can plan against, and
(c) resolves contention deterministically without a human in the loop.

## The four levers, named

Whatever scheduler you are on, the same four levers appear:

- **Quota.** A hard cap on a resource, keyed by team / project /
  cluster queue. Two flavours worth naming: **guaranteed quota**
  (this team will always get at least this much) and **capacity
  quota** (this team can never exceed this much).
- **Fair-share.** When multiple teams are demanding more than their
  guaranteed quota, who gets the *next* free unit? The classic answer
  is DRF (Dominant Resource Fairness, Ghodsi et al., NSDI 2011,
  https://www.usenix.org/legacy/event/nsdi11/tech/full_papers/Ghodsi.pdf).
  SLURM has its own priority multi-factor scheme; Volcano has DRF as
  a plugin; Kueue has cohort borrowing.
- **Priority.** A per-job (or per-workload) integer that says "this
  job should run before that one". Combined with preemption, it
  becomes "this job should displace that one".
- **Preemption.** Evict a lower-priority job to make room for a
  higher-priority one. For training jobs this only makes sense at
  **gang granularity** — you evict the whole gang together, otherwise
  NCCL wedges.

Every policy in this chapter is a specific composition of those four
levers.

## Priority classes: the smallest useful policy

Before you touch quota or fair-share, define three or four priority
classes. Every job on the cluster picks one; the class carries a
default preemption policy.

A typical training-cluster set:

| Class          | Priority | Preempts? | Preemptable? | Typical use                        |
|----------------|----------|-----------|--------------|------------------------------------|
| `critical`     | 10000    | yes       | no           | production runs against a deadline |
| `training-high`| 5000     | yes       | by critical  | large multi-week pretraining       |
| `training-normal` | 1000  | no        | by higher    | day-to-day training                |
| `research`     | 500      | no        | by higher    | experiments, hyperparam sweeps     |
| `preemptible`  | 100      | no        | yes          | interruptible jobs on spot / burst |

On Kubernetes, that maps to a `PriorityClass` per row:

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: training-normal
value: 1000
globalDefault: false
description: "Standard training jobs. Preemptable by higher-priority classes."
```

On SLURM, `sacctmgr` and QoS carry the same idea:

```bash
sacctmgr add qos name=training-normal Priority=1000
sacctmgr add qos name=training-high   Priority=5000 Flags=PreemptCancel \
    Preempt=training-normal,research,preemptible
```

The rest of the policy — quota, fair-share — is layered on top of
these classes. Do not start with quota; start with classes, because
they encode the "what breaks first" contract.

## Quota design: guarantee vs. cap, and the cohort

Two ideas anchor a workable quota model.

- **Guarantee** is the floor. If a team has a guarantee of 64 H100s,
  they can always get 64 H100s within some bounded queue time (usually
  seconds if idle capacity is available, minutes if a preemption is
  needed). Guarantees are the number a team can plan against.
- **Cap** is the ceiling. If a team has a cap of 128 H100s, they will
  never exceed 128 H100s. Caps prevent one team's runaway sweep from
  eating the whole cluster.

Between the guarantee and the cap sits **borrowable capacity**: a team
can grow above its guarantee if idle capacity exists, up to its cap,
and its usage above the guarantee is *reclaimable* by another team
that wants its guarantee back. This is what Kueue calls a **cohort**:

```yaml
# Cohort membership — teams that can borrow from each other.
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: foundations
spec:
  cohort: research
  resourceGroups:
    - coveredResources: ["nvidia.com/gpu"]
      flavors:
        - name: h100
          resources:
            - name: nvidia.com/gpu
              nominalQuota: 64          # guarantee
              borrowingLimit: 64        # can borrow 64 more; cap = 128
              lendingLimit: 32          # up to 32 of my guarantee is lendable
  preemption:
    reclaimWithinCohort: Any            # let siblings reclaim borrowed quota
    withinClusterQueue: LowerPriority   # preempt my own low-priority jobs
```

The important sentence: **guarantee is scheduled first, borrowed is
scheduled if idle, and borrowed is evicted when the lender wants it
back**. Make that contract explicit in your platform docs.

On SLURM the equivalent constructs are:

- **Accounts and associations** (`sacctmgr add account ...` and
  `sacctmgr add user ... Account=... GrpTRES=gres/gpu=64`). `GrpTRES`
  is the cap; there is no "guarantee" primitive per se — you emulate
  a guarantee by carving a partition or by fair-share targets (below).
- **Fair-share targets** — the multi-factor plugin's `FairShare`
  component. `PriorityWeightFairshare=` in `slurm.conf` and per-account
  `Shares=` set the share each account "should" receive over the
  historical decay window. This is how SLURM answers the "who goes
  next?" question when multiple accounts are over their share.
- **QoS** — bundles priority, preemption, per-user job limits, and
  time limits. QoS is often how you enforce the cap-per-class as well
  as the priority.

Both stacks converge to the same shape: a team has a **guaranteed
share** it can plan against, a **cap** that bounds runaway, and a
**cohort/account** that determines who inherits its idle capacity.

## Fair-share in one paragraph

Fair-share does not mean "equal share". It means: given each team's
target share, minus their historical usage over a decay window, sort
the queue so that the team furthest under target gets scheduled next.
SLURM's `PriorityWeightFairshare` + `PriorityDecayHalfLife` config
this directly. Volcano's `proportion` plugin is the equivalent for
Kubernetes. DRF is the multi-resource generalisation: instead of
"who is furthest under target on GPUs?" it is "who is furthest under
target on their *dominant* resource?". Kueue does not do DRF today;
Volcano's `drf` plugin does.

## Gang preemption: the training-specific correctness rule

Preemption for a service is boring: kill a pod, another one comes up.
Preemption for a training gang is not: killing 1 of 8 pods leaves 7
pods burning GPU-hours waiting on NCCL. **Gang preemption** is the
correctness rule: when preempting a gang, evict *all* of its pods
together.

On Kubernetes, this is what Volcano's `preempt` plugin does. Kueue's
`reclaimWithinCohort` also evicts at the Workload level, which is
gang-aware if the Job integration is (PyTorchJob, MPIJob, RayJob all
are).

On SLURM, gang preemption is straightforward: a job is one allocation,
so `PreemptCancel` or `PreemptRequeue` acts on the whole job. The
subtlety is what happens *after*:

- **`PreemptCancel`** — the low-priority job is killed. It has to be
  re-queued by the user (or an auto-requeue policy).
- **`PreemptRequeue`** — SLURM re-queues the job for you, and it will
  resume with the same JobID next time nodes free up.
- **`PreemptSuspend`** — the job is stopped in memory and resumed
  later; only works on non-GPU workloads for practical purposes,
  because you cannot generally suspend GPU state on a shared node.

**For training, use `PreemptRequeue` and pair it with checkpointing**
(mod-106). The mental model: the platform's contract with the
researcher is that a preempted job will resume from its last
checkpoint on its next admission — silently, without a human. Absent
checkpointing, `PreemptCancel` is a nasty surprise; with
checkpointing, `PreemptRequeue` is invisible.

## Composing the policy: a starter template

For a shared training cluster with three teams (foundations, finetune,
research), 128 H100s guaranteed each with 384 total, priorities as
above, this is roughly what the policy looks like at both scheduler
layers.

**Kubernetes (Kueue + Volcano):**

- Three `ClusterQueue`s, one per team, all in cohort `training`.
- Each ClusterQueue: `nominalQuota=128`, `borrowingLimit=128`,
  `lendingLimit=64`.
- `reclaimWithinCohort: Any`, `withinClusterQueue: LowerPriority`.
- Five `PriorityClass`es (critical / training-high / training-normal /
  research / preemptible).
- Volcano: `proportion` + `preempt` + `gang` + `drf` plugins enabled
  in the scheduler config; one `Queue` per ClusterQueue with weight ≈
  1.0.

**SLURM:**

- One partition `training`; one account per team with `Shares=1.0`.
- Four QoS: `critical`, `training-high`, `training-normal`, `research`,
  with the priority and preemption relations above.
- `PreemptType=preempt/qos`, `PreemptMode=REQUEUE`.
- `PriorityType=priority/multifactor`,
  `PriorityWeightFairshare=100000`,
  `PriorityWeightQOS=200000`, `PriorityDecayHalfLife=7-0`.
- Reservations for scheduled runs; auto-requeue on preemption.

Neither template is doctrine — you will tune the weights and the
decay window against your workload's actual queue times. The point
is that both stacks let you express the same policy, and both are
built out of the same four levers.

## Common gotchas

- **No explicit `critical` class.** When the CEO's demo run is
  contending with a research sweep, someone will hand-edit priorities
  in prod. Have the class defined in advance.
- **`reclaimWithinCohort: LowerPriority`** with equally-classed jobs
  will not reclaim — Kueue only reclaims from jobs whose priority is
  strictly lower. Use `Any` for reclaim-on-idle-borrow.
- **Fair-share decay too long.** A 30-day decay window means a team
  that spent October's quota starves in November. Start at 7 days
  and adjust.
- **Preemption without checkpointing.** Do not turn on preemption
  before mod-106 checkpointing is in place. You will lose work and
  incidents will follow.
- **Namespace-scoped priority classes on Kubernetes.** `PriorityClass`
  is cluster-scoped. Do not confuse it with `LimitRange` or
  `ResourceQuota`, which are namespaced.

## Summary

- Every multi-tenant scheduler exposes the same four levers: quota,
  fair-share, priority, preemption. Design the policy in those terms
  and translate to SLURM or Kueue/Volcano.
- Start with priority classes; they encode the "what breaks first"
  contract. Then add quota (guarantee + cap), then cohort borrowing,
  then fair-share weights.
- Gang preemption is a correctness requirement for training. Volcano's
  gang preemption on Kubernetes and SLURM's `PreemptRequeue` on
  batch-first clusters are the right primitives.
- Preemption without checkpointing is a bug. Do not enable it until
  the checkpointing story from mod-106 is real.
