# Straggler and Silent-Corruption Detection, and Auto-Quarantine

Chapter 5 gave you five incident playbooks. Four of them announce
themselves — a NaN, a NCCL timeout, an XID event. The two most
expensive ones do not: **stragglers** drag every collective's tail
without an explicit failure, and **silent data corruption** produces
metrics that look normal while the model quietly regresses.

This chapter is about how to detect those two, and how to plumb
detection into automatic node quarantine so the run heals itself
before an on-call human is paged. It sits at the intersection of
mod-106 (recovery semantics), mod-108 (observability) and mod-104
(scheduler-side quarantine). Where the boundary matters, this
chapter says so.

Two references:

- **NVIDIA DCGM (Data Center GPU Manager).**
  https://docs.nvidia.com/datacenter/dcgm/ — the current
  monitoring + diagnostics stack for NVIDIA GPUs in a cluster.
- **Llama 3 §6.3 (Grattafiori et al., 2024).** The tabulated
  failure taxonomy and per-class MTTR that this chapter's
  detection strategy is calibrated against.

## Why detection is the hard part

Every incident-response playbook in chapter 5 begins with "notice
that the incident has happened". For loud incidents that is easy —
`init_process_group(timeout=...)` fires, or the trainer's exception
handler catches a NaN. For quiet incidents:

- A rank that runs 15% slower than its peers on every step drops
  aggregate throughput by ~13%; the loss curve, the GPU utilization
  dashboard, and even the per-step wall-clock look "fine" — the
  step is just a little slower than yesterday.
- A single bit-flip in HBM that lands in a hot activation produces
  a wrong gradient once. The optimizer step incorporates it. The
  loss curve does not spike; the model is now slightly wrong.
  Nothing pages.

Detection for these classes is *active*: you have to instrument
for them, or you will not see them. That is what this chapter is
about.

## Straggler detection

### What "straggler" means precisely

A straggler is a rank whose step time is persistently higher than
the median across the world. Since a collective can only complete
when the slowest participant does, one straggler dominates the
whole run's throughput. On a well-behaved job, per-rank step time
should sit in a tight distribution: median ± 5% or so. A rank
that sits at median + 30% for hours is a straggler.

Two subtypes matter operationally:

- **Persistent straggler.** Same rank, every step. Almost always
  hardware — thermal throttle, bad HBM stack driving down memory
  bandwidth, a bad NIC.
- **Intermittent straggler.** Different ranks slow at different
  times. Often the fabric (mod-105) or the storage tier — one
  rank hits an unlucky read-latency spike on a particular
  batch.

### First-line instrumentation: per-rank step-time histograms

The minimal detection surface. Every rank records its own
per-step wall time; a periodic Prometheus scrape (mod-108) pulls
those out and computes:

- Per-rank median step time over the last 5 minutes.
- Global median step time over the last 5 minutes.
- Per-rank deviation `(rank_median - global_median) / global_median`.

A rank whose deviation stays above +20% for 5 consecutive
minutes is flagged. mod-108's dashboards should surface this as
a top-line panel and page on sustained flags.

Two implementation notes:

- Do not scrape per-step; the metric explodes. Aggregate to
  ~1 s resolution.
- Log the collective identifier alongside step time. A rank that
  is slow on `reduce-scatter` but fast on `forward` is a
  different problem from a rank that is slow on everything.

### Second-line: all-reduce timing

Step time confounds compute and communication. To localize the
straggler, time the collective itself. NCCL exposes per-collective
timing when you set `NCCL_DEBUG_SUBSYS=COLL` and can be sampled at
the framework level with:

```python
# Reference sketch — see torch.cuda.Event and
# torch.distributed.reduce_scatter/all_gather for the exact API.
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
end.record()
torch.cuda.synchronize()
per_rank_ms = start.elapsed_time(end)
```

Log per-rank `per_rank_ms` per collective, take the tail, and
graph. The mental model: on a healthy fabric, per-rank
all-reduce time should equal the *slowest* rank's time — the
collective is a barrier. A rank that consistently reports the
slowest time is either the actual straggler (its data path is
slower) or is downstream of a fabric drop (mod-105 chapter 8's
territory). Compare against the fabric-side `nccl-tests`
baseline; if the fabric is fine, the rank is the problem.

### Third-line: DCGM SM occupancy divergence

DCGM (`nvidia-smi` and `dcgmi dmon`) exports per-GPU counters at
sub-second granularity. The counter that best distinguishes
"straggler" from "load-imbalanced" is **SM occupancy**
(`DCGM_FI_PROF_SM_OCCUPANCY` in the DCGM field list).

Concretely:

- A rank with **low SM occupancy but healthy step time** is
  compute-bound but memory-bound; not a straggler.
- A rank with **low SM occupancy and high step time** is
  memory-bound or throttled — heat, ECC scrubbing, or a bad HBM.
- A rank with **high SM occupancy and high step time** is CPU
  bound or dataloader bound.

DCGM's field list also exposes `DCGM_FI_DEV_GPU_TEMP`,
`DCGM_FI_DEV_POWER_USAGE`, and the ECC counters — a straggler
that is thermally throttled shows a temperature ceiling before
it shows a step-time regression, which is your leading indicator.

mod-108 owns the DCGM → Prometheus pipeline; mod-106 owns the
alert rules that trigger quarantine.

## Silent-data-corruption detection

Where straggler detection is a signal-to-noise problem, silent
corruption is a signal-*existence* problem — the run's normal
metrics do not have the signal. You have to actively probe.

Four instrumentation strategies, in increasing order of cost /
increasing order of confidence:

### 1. Checkpoint hash verification

The cheapest defense. Every rank hashes its checkpoint shard
(e.g., BLAKE3 over the raw bytes) on save and again on load; a
mismatch means the storage tier corrupted the write.

```python
# On save
shard_bytes = serialize_shard(rank_shard)
shard_hash = blake3(shard_bytes).hexdigest()
write_shard_and_hash(shard_bytes, shard_hash)

# On load
loaded_bytes, expected_hash = read_shard_and_hash()
actual_hash = blake3(loaded_bytes).hexdigest()
assert actual_hash == expected_hash, "silent corruption on shard"
```

DCP does not do this out of the box; you plumb it as a
post-write / pre-read hook. mod-105 chapter 8 is where storage
corruption should be caught upstream (checksums on the FS), but
end-to-end hash verification catches the class of corruption
that happens between memory and disk (bad HBM row, bad DMA).

### 2. Gradient-norm outlier detection

Per-rank gradient norms are already computed for gradient
clipping. Log them, compute a running median and MAD (median
absolute deviation) across ranks, and flag any rank whose
per-step norm sits more than K MADs off the median for several
steps in a row.

The physical intuition: on a well-shuffled dataset, per-rank
grad norms should be *very* close after the first ~hundred
steps. A rank whose grad norm is consistently 3× the median is
either seeing a systematically different data distribution
(a mod-103 bug — check dataset shard-to-rank mapping) or its
compute is silently wrong.

### 3. Deterministic-mode re-run diff

The most powerful, and most expensive. Periodically (say once
per 10 000 steps), on a small canary process:

- Load a checkpoint from N steps ago.
- Set `torch.use_deterministic_algorithms(True)` and configure
  cuDNN / CUBLAS deterministic mode.
- Step the canary for K iterations (K small — 100 is often
  enough).
- Compare a fingerprint of the resulting parameters against the
  fingerprint the original run produced at the equivalent point.

If the fingerprints diverge at bit level, something during the
original run was non-deterministic in a way you did not expect.
The two candidates:

- **Silent hardware corruption.** A bit flip you did not catch.
- **Non-determinism you did not know about.** Some kernel path
  is nondeterministic for reasons other than seeds. mod-108
  covers the deterministic-mode setup.

The cost of deterministic mode is real (~2–5× slowdown on the
canary; some kernels do not have deterministic implementations
at all), which is why it runs on a canary, not on the whole run.

### 4. Held-out canary eval

Independently of everything above: reserve a small,
representative held-out set (say 1–10 k prompts) and run it
against every checkpoint. Log the resulting loss or accuracy.

A silent-corruption regression will show up here even when the
training-loss curve looks fine. It is the strongest signal;
it is also the highest-latency one (you find out at the next
canary eval, not the next step).

mod-108 covers what to put on the dashboard; mod-106 owns
"we do this at all". The exercise 04 designs the four-layer
defense.

### The hierarchy in practice

You do not need all four layers on every run. Pick the level of
protection appropriate to the run's stakes:

| Run stakes | Minimum detection layers |
|------------|-------------------------|
| Experiment (<1 day, cheap) | Checkpoint hash |
| Small production (<7 days) | Checkpoint hash + grad-norm outlier |
| Large production (weeks) | All four layers |
| Frontier / flagship | All four + shorter canary intervals + higher canary weight |

## Auto-quarantine: the recovery half

Detection without action is expensive telemetry. The other half
of the story is: when detection fires, drain the offending node
so the run can elastic-reshape without hitting it again.

Two host schedulers, two idioms.

### SLURM: `scontrol update NodeName=... State=drain`

SLURM's node states include `drain` (no new jobs; existing jobs
complete) and `drained` (fully offline). To quarantine a node:

```bash
scontrol update NodeName=gpu-node-037 \
  State=drain \
  Reason="dcgm-hw-flag: xid 79 at 2026-07-09T14:22Z"
```

The `Reason` field is critical — it is the audit trail for when
the ops team asks why the node came out. The node will not accept
new jobs; when the current job releases it (via elastic reshape
or job end), it enters `drained`.

A production quarantine loop looks like:

```
DCGM alert → Prometheus → alertmanager rule matches →
webhook to quarantine service → 
  ssh headnode "scontrol update NodeName=... State=drain Reason=..."
```

mod-104 owns the SLURM plumbing; mod-106 owns the alert-to-action
mapping (which alert triggers a drain, which alert triggers only
a page).

### Kubernetes: taint and cordon

The Kubernetes idioms:

- `kubectl cordon <node>` marks it unschedulable.
- `kubectl taint nodes <node> hardware=faulty:NoSchedule`
  additionally repels pods that do not tolerate the taint.
- `kubectl drain <node> --ignore-daemonsets` evicts the running
  pods.

For a training job, `drain` triggers pod eviction, which — if the
trainer runs under a supervising operator (TorchElastic Kubernetes
operator, Kueue, KubeRay) — triggers the elastic reshape.

Kueue's admission checks (see the Kueue docs on admission
checks) let you go further: an admission-check controller can
refuse to admit a workload onto a node with a specific taint, or
onto a node with a health metric in a bad range. This is how
you keep bad nodes out of the *next* job, not just the current
one.

The plumbing is:

```
DCGM alert → Prometheus → alertmanager rule matches →
webhook to quarantine controller →
  kubectl taint nodes ... hardware=faulty:NoSchedule
  kubectl drain <node> --ignore-daemonsets --disable-eviction=false
```

For both schedulers, the quarantine loop should be **idempotent**
(re-firing the alert should not re-drain a node that is already
draining) and **auditable** (every quarantine event goes into a
persistent log). mod-108 owns the log; mod-106 owns making sure
the events go there.

### What to alert on

Not every DCGM warning is a quarantine trigger. A rough hierarchy:

| Signal | Response |
|--------|----------|
| Transient temperature spike | Log, no action |
| Sustained thermal throttling >5 min | Page, no auto-quarantine |
| Correctable ECC rate above baseline | Page, no auto-quarantine |
| Single uncorrectable ECC | Auto-quarantine |
| XID indicating GPU fell off bus | Auto-quarantine + page |
| Step-time straggler >20% for 5 min | Page (may or may not auto-quarantine) |
| Checkpoint hash mismatch | Auto-halt run + page |
| Canary eval regression | Halt run + page |

The philosophy: **auto-quarantine anything with a clear physical
cause**; page on anything that requires human judgment. A false
positive on hardware quarantine costs you one node for a few
hours; a false negative can burn a run.

## The full detection-to-recovery loop

Putting the whole chapter together, the loop that keeps a run
alive without a paged human looks like:

1. **Steady state.** Detection layers are running: per-rank
   step-time histograms, per-rank grad-norm outliers, DCGM
   health, checkpoint hashes, periodic canary eval.
2. **Signal fires.** A DCGM XID event on node 37. The alert
   rule matches "uncorrectable ECC" and fires.
3. **Auto-quarantine.** The quarantine controller drains node
   37 (SLURM: `scontrol update ... State=drain`; Kubernetes:
   `taint` + `drain`).
4. **Elastic reshape.** The workers on node 37 die. torchrun's
   agents on other nodes detect the missing peer and re-rendezvous
   at the smaller world size (chapter 4).
5. **DCP resume.** New round loads the most recent checkpoint at
   the new world size. Sampler resumes. Training continues.
6. **Repair path.** Meanwhile, DGX ops replaces the faulty
   hardware. When the node comes back healthy, the taint is
   removed; on the next round (or the next job) the node is
   available.
7. **Post-mortem.** The incident is logged: which signal fired,
   which alert rule matched, which node was quarantined, how
   long the reshape took, whether the run recovered without
   human intervention.

Total wall clock: ~1–2 minutes if everything is wired correctly.
Compare that to "wake a human at 3 am, they SSH in, they drain
the node by hand, they wait 20 minutes for the run to notice" —
you have just moved a 20-minute incident into a 1-minute one.

## Boundaries with other modules

- **mod-108** owns the observability stack: Prometheus, DCGM
  scrapers, dashboards, and the alerting engine. mod-106 owns
  the alert *rules* — which signal maps to which response.
- **mod-105** owns fabric-side signals (silent NIC drop, PXN
  misconfiguration). This chapter's straggler detection may
  point at mod-105's runbook; that hand-off is normal.
- **mod-104** owns SLURM and Kueue as scheduler primitives. This
  chapter uses them as tools; the plumbing patterns live in
  mod-104.
- **mod-103** owns the dataloader shard-to-rank mapping. A
  grad-norm outlier tied to a specific rank's data distribution
  is a mod-103 bug; this chapter detects it, mod-103 fixes it.

## Summary

- Two classes of "quiet" incidents dominate the risk on a
  long-running training job: stragglers and silent data
  corruption. Neither announces itself in normal metrics.
- Straggler detection is a three-layer stack: per-rank
  step-time histograms, per-collective all-reduce timing, and
  DCGM SM occupancy divergence. Persistent stragglers are
  almost always hardware.
- Silent-corruption detection is a four-layer stack: checkpoint
  hash verification, gradient-norm outlier detection,
  deterministic-mode canary re-run diff, held-out canary eval.
  Pick layers to match the stakes of the run.
- Auto-quarantine converts detection into recovery. SLURM uses
  `scontrol update NodeName=... State=drain`; Kubernetes uses
  `taint` + `drain`; Kueue admission checks keep faulty nodes
  out of future jobs.
- Auto-quarantine anything with a clear physical cause; page on
  anything that requires judgment. False-positive quarantine
  costs one node for a few hours; false-negative can burn a
  run.
- The end-to-end loop is detection → quarantine → elastic
  reshape → DCP resume, and it should run to completion in ~1–2
  minutes without a human. Chapter 7 quantifies the cost when
  it does not.
