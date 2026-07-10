# Incident Classification and Recovery Playbooks

Chapter 4 wired up automatic re-rendezvous for benign failures.
This chapter is about the incidents that require a human on-call —
the ones a naive restart either does not fix or actively makes
worse. If you internalize one thing from mod-106, make it this
chapter's five-way classification. It is the mental scaffolding
that turns a paged wake-up into an ordered set of moves.

Two references live open on any incident:

- **The OPT-175B logbook.** Zhang et al. (2022), arXiv:2205.01068
  — the paper links to the day-by-day operational log. The
  logbook is the canonical public record of what a large
  pretraining run's on-call actually looks like.
- **Llama 3 paper §6.** Grattafiori et al. (2024), "The Llama 3
  Herd of Models" — section 6.3 tabulates the failure taxonomy
  and MTTR for the 405B run at scale.

Both are worth reading front-to-back; the case studies in chapter
7 walk through them.

## The five canonical incident classes

Every training-run incident falls into one of five classes. Each
has a distinct signature, first-line diagnostic, and recovery
playbook. Learn to classify the paged incident in the first 60
seconds — the rest of the response is class-dependent.

| Class | Signature | First-line diagnostic | Owned in this chapter |
|-------|-----------|-----------------------|-----------------------|
| A. Loss spike | Loss jumps by >5σ, then either recovers or diverges | Check loss curve, gradient norm, LR schedule | §Loss spike |
| B. NaN / Inf | Loss goes to `nan` or `inf`; training stops | Check for `torch.isnan` in loss and grads, mixed precision config | §NaN / Inf |
| C. NCCL timeout / hang | Job hangs, watchdog fires after `NCCL_TIMEOUT` | py-spy on all ranks; look for rank not inside the collective | §NCCL timeout |
| D. Hardware fault | GPU dropped, ECC error, XID event, HBM failure | `dmesg`, `nvidia-smi`, DCGM, XID log | §Hardware fault |
| E. Silent data corruption | Metrics look fine; downstream evals regress | Deterministic re-run diff, checkpoint hash verification | §Silent corruption |

The first four announce themselves. The fifth does not, which is
what makes it dangerous. Chapter 6 goes deeper on straggler and
silent-corruption *detection*; this chapter covers what to do
once one of the five has been called.

Each playbook below has the same shape: **signature → diagnostics
→ containment → recovery → post-mortem.** That shape mirrors the
on-call ticket template you should be filling in as you go.

## Playbook A: Loss spike

### Signature

Training loss jumps abruptly — often 5–10× the trailing rolling
average — over one or a few steps. Two sub-cases matter:

- **Transient spike:** loss returns to trend within tens of steps.
- **Divergent spike:** loss stays high or continues to climb; the
  run has effectively broken.

The OPT-175B logbook is a museum of transient spikes with the
occasional divergent one; the on-call team responded with a mix
of "let it recover" and "rewind and skip the offending batch".

### Diagnostics

- **Loss curve at fine granularity.** Zoom to per-step, not
  per-epoch, on the run dashboard (mod-108).
- **Gradient norm.** A spike in `||grad||` almost always precedes
  a loss spike. If grad norm exploded, the LR + gradient clipping
  combination was inadequate for that batch.
- **Recent config changes.** LR increase, batch-size change, warmup
  discontinuity, dropout change — anything altered in the last
  hour?
- **Batch composition.** If your dataset is heterogeneous, a spike
  correlates with certain document types (very long documents,
  low-quality shards). Log the batch index range.

### Containment

- **Do not stop the run for transient spikes.** With gradient
  clipping active, the model usually absorbs a 5σ spike within
  ~50 steps. Killing the run and rewinding is more expensive than
  letting it converge.
- **Do stop for divergent spikes.** If loss has not returned to
  trend after ~200 steps, you are watching the run diverge in
  slow motion. Kill it before the next checkpoint overwrites the
  last known-good state.

### Recovery

- **Transient:** none. Let the model absorb.
- **Divergent:** rewind to the checkpoint immediately *before* the
  spike (not the most recent — the most recent may already be
  poisoned). Options:
  - Skip N steps of data (advance the sampler past the offending
    batch range) and resume.
  - Reduce LR by ~2× and resume with the same data.
  - Enable / tighten gradient clipping.
  The OPT-175B team used all three, at various points, on the
  same run. See the logbook.

### Post-mortem

Record: which step the spike started at, which checkpoint you
rewound to, which batches you skipped, and what config you
changed. Loss-spike incidents accumulate — a run that has been
rewound eight times has a different training-trajectory story
than one that has been rewound zero times. mod-108's
experiment-metadata store is where this record lives.

## Playbook B: NaN / Inf

### Signature

Loss is `nan` or `inf` for one or more ranks. Training either
stops on its own (`loss.backward()` propagates the NaN into every
gradient and the optimizer step turns every parameter into NaN)
or, worse, keeps going with silently-broken weights.

### Diagnostics

- **`torch.isnan(loss)` / `torch.isinf(loss)` per rank.** Insert
  a check *before* backward. If one rank has NaN and others do
  not, the NaN is data-dependent or arithmetic-underflow, not
  systemic.
- **Mixed precision status.** BF16 does not underflow easily; FP16
  and FP8 do. If you are running FP8 (Transformer Engine on H100)
  and getting NaN, check the amax history and the scale factor
  update — mod-107 owns FP8 mechanics but the symptom lands here.
- **Reduce-scatter dtype.** FSDP2 reduces gradients in FP32 by
  default (`MixedPrecisionPolicy(reduce_dtype=torch.float32)`).
  If someone changed that to BF16 to save bandwidth, gradient
  underflow may explain the NaN.
- **`nvidia-smi` per rank.** Was there an ECC error at the same
  moment? A single-bit flip can produce a NaN in a hot tensor.

### Containment

- **Halt the run before saving another checkpoint.** A save that
  captures NaN weights poisons the last-good state.
- **Kill the trainer.** Do not let it save.

### Recovery

- **Rewind to the checkpoint immediately before the NaN appeared.**
  If the NaN happened at step 12 500 and the previous save was
  step 12 250, you resume from 12 250.
- **Determine root cause before restart.** The pathologies:
  - **Data-dependent underflow.** A specific batch produces
    gradients that underflow. Fix: skip the batch, or move to
    higher-precision reduction. Log the batch index.
  - **FP8 amax stale.** Transformer Engine's amax history did
    not update in time; fix with a shorter recipe interval
    (mod-107).
  - **ECC error.** Single-event upset in HBM. Fix: quarantine the
    node (chapter 6), rewind, resume. If the same node produces
    NaNs on multiple restarts, escalate to hardware ops.
  - **Bad LR / clipping combination.** Same as loss-spike
    recovery: reduce LR or tighten clipping.

### Post-mortem

The single most important post-mortem note on NaN incidents is
**"was the NaN transient or persistent?"** A transient NaN
(single batch, single rank) is a data or arithmetic issue. A
persistent NaN (every batch, every rank) is a config bug you
introduced. The two require different fixes.

## Playbook C: NCCL timeout / hang

### Signature

The job hangs; no rank makes forward progress; eventually a rank's
watchdog fires with a message like:

```
Watchdog caught collective operation timeout: WorkNCCL(SeqNum=...,
  OpType=ALLREDUCE) ran for X milliseconds before timing out
```

The default watchdog timeout is 30 minutes (`torch.distributed`
gives `init_process_group(timeout=timedelta(minutes=30))`). Lower
it for tighter feedback loops.

### Diagnostics

- **`py-spy dump --pid <PID>` on every rank.** Find the odd one
  out: the rank *not* waiting inside NCCL is the rank that stopped
  moving. That's your fault. All the ranks waiting inside NCCL
  are victims.
- **Recent NCCL logs.** `NCCL_DEBUG=INFO` and
  `NCCL_DEBUG_SUBSYS=INIT,COLL,NET` show which collective a rank
  was inside when it stopped.
- **Per-rank step time histogram.** Is the odd rank slower on
  average? A slow rank that drifts far enough eventually times out.

### Containment

- **Snapshot before you restart.** py-spy dumps, NCCL logs, and
  per-rank step times are the evidence. A restart loses them.

### Recovery

The recovery depends on *why* the rank stopped. Refer to the
NCCL timeout runbook in mod-105 chapter 8 for fabric-side causes
(silent NIC drop, PXN misconfiguration). This chapter owns the
trainer-side causes:

- **A rank crashed but its watchdog fires late.** Fix the crash
  (usually shows up as a Python exception in the crashed rank's
  log). torchrun's elastic reshape (chapter 4) then handles the
  restart automatically.
- **Uneven work.** One rank has a much larger microbatch or a
  much slower kernel path than the others. Fix the load
  imbalance (data-side: shuffle better; model-side: check for
  ragged sequence lengths).
- **Checkpoint save on rank 0 exceeded the watchdog.** A
  synchronous save on a slow storage tier can stall past the
  timeout. Move to async save (chapter 3) or raise the timeout
  for the save window.
- **CUDA kernel deadlock.** Rare, but happens with `torch.compile`
  regressions or custom kernels. Escalate to mod-107.

### Post-mortem

The wrong post-mortem note here is "NCCL was slow". NCCL is a
symptom; the collective was waiting on a rank that stopped
moving. Name the rank and what it was doing. This is the same
principle mod-105 chapter 8 makes for the fabric side of the
same incident class.

## Playbook D: Hardware fault

### Signature

One or more ranks report a CUDA error, an ECC event, or the node
falls off the network. Concrete tells:

- **`nvidia-smi` shows `Xid` in the error log.** The XID number
  matters — different XIDs mean different things (see the
  NVIDIA driver docs XID reference).
- **`dmesg` shows a GPU-related kernel message.** ECC errors,
  PCIe link-down, thermal shutdown, HBM row remapping events.
- **DCGM reports a health downgrade.** `dcgmi diag` and DCGM
  health checks (mod-108 wires them into Prometheus).
- **Node stops answering IB / RoCE traffic.** NCCL timeout on
  every collective involving the node.

### Diagnostics

- **XID lookup.** NVIDIA maintains an XID number → cause table
  in the driver documentation. A rough hierarchy of severity:
  - Recoverable XIDs (e.g., correctable ECC, some memory errors)
    — log, continue.
  - Fatal-per-process XIDs (e.g., MMU faults from a bad kernel)
    — kill the process, restart the trainer.
  - Fatal-per-GPU XIDs (e.g., uncorrectable ECC, GPU fell off the
    bus) — quarantine the GPU, remove the node from the run.
  - Fatal-per-node XIDs (e.g., power failure, PCIe root complex
    reset) — quarantine the node.
- **`dcgmi diag -r 3`.** Runs the full diagnostic. Report the
  passing / failing subsystems.
- **`nvidia-smi -q` for detailed status.** ECC counters,
  temperature, power state.

### Containment

- **Do not restart the run on the same node.** If the hardware is
  faulty, restart will hit the same fault. Quarantine first
  (chapter 6 has the automation).
- **Preserve the node in whatever state it is in.** Fabric ops
  and DGX ops need the raw evidence; a reboot clears the logs.

### Recovery

- **Elastic reshape**, using the machinery from chapter 4.
  Assumption: `MIN_NODES` is low enough that the run continues
  after dropping the faulty node.
- **DCP load** at the smaller world size. Resume.
- **Meanwhile, fabric / DGX ops replace or repair the node.**
  When it comes back, the run either grows back (if launcher is
  still watching, and `MAX_NODES` allows) or the node re-enters
  the pool for the next job.

### Post-mortem

The interesting question is: **should this node have been
quarantined earlier?** Almost every hardware fault has a
predictive precursor — rising ECC counter, thermal warning,
DCGM health degradation. Chapter 6 covers automating that
quarantine. The post-mortem's job is to say whether the signal
was there and missed.

Llama 3 §6.3 tabulates the observed frequency of each fault
class over 54 days on 16 384 H100s; the exercise 03 asks you to
build a Sankey diagram of that distribution and design your
runbook's coverage against it.

## Playbook E: Silent data corruption

### Signature

The training run has no failure signal. Loss looks normal.
Throughput looks normal. Gradient norms look normal. But
downstream evaluations regress — the model at step N performs
worse on eval than the model at step N-5000 did — or a
deterministic re-run from an earlier checkpoint produces
different numerics than the original run did.

This is the failure mode you should be most afraid of. It burns
compute silently and you find out weeks later.

### Diagnostics

Silent corruption cannot be diagnosed from the running run's
metrics alone — that is what makes it silent. It is diagnosed by:

- **Deterministic re-run diff.** Take a checkpoint, restart from
  it in a deterministic-mode config (`torch.use_deterministic_algorithms(True)`),
  step for K iterations, and compare the loss curve or a small
  parameter fingerprint to the original run's log. Divergence at
  bit level → hardware or nondeterminism you did not expect.
- **Checkpoint hash verification.** Every rank publishes a hash
  of its shard on save. On resume, hashes must match; mismatch
  means the storage tier corrupted the write. mod-105 owns the
  storage tier; mod-106 owns wiring the hash in.
- **Gradient-norm outlier detection.** Per-rank grad norms
  should be tightly clustered; a rank whose grad norm sits
  outside the cluster for many steps is suspect. Chapter 6
  covers the instrumentation.
- **Held-out eval divergence.** A small held-out canary set,
  evaluated every K steps, is the strongest silent-corruption
  signal you have. If the canary regresses, something is
  wrong even if the training-loss curve looks fine.

### Containment

- **Halt the run.** Every additional step propagates the
  corruption.
- **Preserve the current checkpoint and the last known-good
  checkpoint.** Do not delete either.

### Recovery

- **Bisect between known-good and now.** Load intermediate
  checkpoints, run each for a fixed number of steps in
  deterministic mode, compare to the original run. Find the
  earliest checkpoint that shows the divergence.
- **Rewind to the last known-good.** This is the expensive
  option — you may have to walk back tens of thousands of steps.
- **Isolate the cause** before restarting:
  - **Silent memory error** on a specific GPU — quarantine the
    GPU permanently.
  - **Framework version bump** that changed determinism — pin
    versions and rerun.
  - **Data pipeline change** that introduced a shard-level
    reorder — mod-103 owns the fix; mod-106 owns the detection.

### Post-mortem

The single most important operational takeaway from silent
corruption is that you can only *detect* it. You cannot detect
"the run is silently corrupted" from inside the run itself;
you need out-of-band checks (deterministic reproduction, canary
eval, hash verification). Every run's SLO should include a
budget for those out-of-band checks. Chapter 6 goes deeper.

## Cross-cutting on-call practices

Independent of which class the incident belongs to, the first
three moves are always the same. Practice these until they are
muscle memory.

1. **Snapshot before you restart.** Grab logs, per-rank stack
   traces, checkpoint filenames, storage state. A restart
   destroys the evidence.
2. **Classify.** Which of the five classes is it? If you cannot
   decide within a couple of minutes, it is probably class E
   masquerading as class A or C.
3. **Do not skip the checkpoint step.** Recovery starts from a
   checkpoint, not from scratch. If you cannot find the right
   checkpoint to resume from, you have not classified yet.

Two habits to build:

- **Every incident gets a ticket even if the run auto-recovered.**
  Silent corruption starts as "a NaN happened once but it
  recovered". If you do not log the incident, you cannot correlate
  when the pattern shows up months later.
- **Write the runbook entry as you diagnose.** The one thing every
  senior on-call regrets is not documenting the walk-through
  while it was fresh. Chapter 7's case studies exist because
  Meta AI documented OPT-175B's incidents in real time.

## What this chapter does *not* cover

- **Detection tooling** — chapter 6 owns straggler detection,
  silent-corruption detection, and auto-quarantine.
- **SLO / budget arithmetic** — chapter 7 owns turning incident
  rates into checkpoint intervals and availability budgets.
- **Fabric-caused NCCL timeouts** — mod-105 chapter 8 owns them.
  This chapter owns the trainer-caused half of the same class.
- **MFU / throughput regressions** — mod-107 owns them. A
  throughput regression is not an incident in the mod-106 sense
  unless it is degrading availability.

## Summary

- Five canonical incident classes: loss spike, NaN / Inf, NCCL
  timeout, hardware fault, silent data corruption. Classify
  within 60 seconds.
- Each class has a fixed playbook shape: signature → diagnostics
  → containment → recovery → post-mortem. Fill in the ticket in
  that shape as you go.
- Loss spikes: let transient ones recover; rewind divergent
  ones. Skip batches or reduce LR on rewind.
- NaN / Inf: halt before saving; rewind to *before* the NaN;
  root-cause before restart.
- NCCL timeout: py-spy every rank to find the one that stopped
  moving. mod-105 chapter 8 owns fabric causes.
- Hardware faults: `nvidia-smi`, XID, `dcgmi diag`. Quarantine
  first, elastic-reshape second. Chapter 6 automates the
  quarantine.
- Silent data corruption: cannot be detected from the running
  run's normal metrics. Use out-of-band deterministic re-runs,
  canary evals, and checkpoint hashes. Chapter 6 details the
  instrumentation.
- Snapshot before you restart. Every incident gets a ticket.
  Chapter 7's case studies are what happens when you don't.
