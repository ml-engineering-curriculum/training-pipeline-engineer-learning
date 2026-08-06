# Classifying Training Incidents and Codifying Playbooks

Chapters 3 and 4 gave you the recovery mechanism. This chapter is
about deciding **when to use which mechanism**. Two on-call incidents
that both surface as "training is unhealthy" can have opposite
correct responses — one wants an immediate rollback and rendezvous,
the other wants you to keep the job running while a hardware team
drains a node in the background. A team without an explicit incident
taxonomy will handle both the same way, and one of the two responses
will be wrong.

The frame is the same as mod-105 chapter 8's fabric runbook:
**symptom → tools → likely causes → decision → post-mortem note**. The
five incident classes below cover the vast majority of what a
training platform on-call rotation sees. If you can walk through each
without a reference, you have this chapter.

## The five incident classes

Named by what the on-call sees, not by root cause. Root cause is what
you diagnose during the incident.

| Class | On-call sees | Danger | Time to first response |
|-------|--------------|--------|------------------------|
| Loss spike / divergence | Loss curve jumps up or plateaus at a high value; training is running | Ruined checkpoint distribution | Minutes |
| NaN / Inf in loss or gradients | Training loop errors out; loss is NaN in TB; the run crashes hard | Ruined weights; possible hardware indicator | Immediate |
| NCCL timeout / hang | Job stops making progress; NCCL watchdog log line | Wasted GPU time until timeout expires | Immediate |
| Hardware fault | GPU / NIC / memory ECC counter spikes; DCGM alerts; node kernel logs | Silent data corruption if not caught | Minutes to hours |
| Silent data corruption | Nothing obvious; downstream evals or later step losses diverge from a reference run | Ruined training run; hardest to detect | Days to weeks |

The interesting property: incidents at the top of the table are
**loud and cheap**; incidents at the bottom are **quiet and
expensive**. A platform team's investment should be inverted from the
incident rate — you spend more on detecting silent-corruption than
you do on NaNs, because a NaN pages you instantly and an SDC does
not.

## Class 1: Loss spike / divergence

### Symptom

The loss curve, plotted by step, has a discontinuity: it jumps by
some multiple, or plateaus at a value much higher than the current
trajectory. The training loop does not crash; gradients are finite;
metrics are being written; from `nvidia-smi`, the GPUs look
healthy.

### Tools

- **The loss curve itself**, at high resolution. Downsampled dashboards
  (1-in-1000 steps) can hide a two-step spike. Read the raw
  per-step log.
- **Per-parameter gradient norms.** A spike that lives in one layer
  is different from a spike everywhere. Layer-wise gradient norms in
  TensorBoard / W&B are a cheap always-on signal — mod-108 covers
  the plumbing.
- **The most recent LR schedule change.** Warm-up boundaries and
  cosine minima are the two step-index-locked places a spike is
  most likely to appear.
- **Recent code / config changes.** If a spike appears after a hot
  reload of the recipe, it is a recipe change, not a training
  event.

### Likely causes

- **LR too high after a warm-up.** Most common. The optimizer takes
  a step that pushes weights off the optimization manifold; the
  next step's loss reflects that.
- **Numerical instability in BF16 / FP8 accumulation.** Some
  operators (softmax, layernorm) can produce values that overflow
  BF16 or destabilize FP8 accumulation. Rare in well-tested
  recipes; more common when tuning custom kernels.
- **Data bug.** A corrupt shard, a truncated file, a
  language-detection failure that starts serving garbage. The spike
  will correlate with a specific range of `sampler.position`.
- **Loss-scale race.** With dynamic loss scaling, a scale drop and
  a large-magnitude batch can compose into a spike. Usually
  self-resolving.

### Decision

The runbook question: **do we roll back or continue?**

- If the spike is a single-step blip and gradients look sane by
  step +5, keep going.
- If the loss stays elevated for more than a few tens of steps,
  roll back to the last DCP checkpoint before the spike, decrease
  LR by 10%–50% (or reset the loss scale), and resume.
- If the spike correlates with a specific data range, roll back and
  quarantine the offending shard range from the sampler.

### Post-mortem note

Every loss spike gets a `(step, event, cause)` entry in the run's
logbook (chapter 7's OPT-175B chronicles are the reference format).
The value of the logbook is not the entry — it is that six weeks
later, when a similar spike appears, the on-call has a diff to
compare against.

## Class 2: NaN / Inf in loss or gradients

### Symptom

Training loop hits an assertion or errors out on a `not-finite` loss
value. The TB/W&B loss series shows NaN. Sometimes preceded by a
one-step gradient-norm spike; sometimes appears without warning.

### Tools

- **`torch.isnan` / `torch.isinf` on gradients per parameter**,
  logged from the training loop. Requires plumbing that must be on
  by default in a production trainer.
- **The forward-pass activations of the offending step.** If you
  cannot get them from the crashed job, replay the step from the
  DCP checkpoint at step-1 with activation dumps enabled.
- **ECC error counters via DCGM** on the affected GPUs (see chapter
  6). A NaN on one specific GPU at a specific step is a *strong*
  hardware indicator.

### Likely causes

- **Numerical overflow in a specific op.** A softmax with large
  logits, an exponential in a positional encoding, an attention
  variant without a clamp. Fixed with `float32` accumulation for
  that op or a numerically-stable variant.
- **Undetected ECC error on HBM.** A GPU produced wrong numbers
  which propagated through the graph. Chapter 6's DCGM integration
  is what catches this.
- **Gradient explosion during a rare data batch.** A batch with a
  long-tail token distribution triggers an unusually large gradient
  on a rarely-updated parameter.
- **Mixed-precision loss-scale collapse.** Autoloss-scale hits its
  minimum and stays there; the next few steps see either NaN or
  zero updates.

### Decision

- **Immediate.** Kill the job. A NaN that got into the model weights
  poisons every subsequent checkpoint.
- **Roll back** to the last DCP checkpoint *before* the NaN. Verify
  the loaded weights are finite. If not, roll back further.
- **Diagnose before restarting.** If the NaN is on a specific rank,
  that rank's GPU is a suspect — check DCGM output. Do not simply
  restart on the same hardware; the next NaN may be minutes away.
- **If a rank's hardware is suspect**, drain the node and let
  elastic reshape rescue the remaining ranks (chapter 4).

### Post-mortem note

A NaN is either a code bug or a hardware bug. It is never
"training's fault" in a way that is worth debugging by "just try
again". If you cannot name the layer / op / rank the NaN originated
in, do not resume training on the same weights and hardware.

## Class 3: NCCL timeout / hang

### Symptom

Training loop stops making progress. Wall clock advances; step
counter does not. Eventually a `WorkNCCL(...timeout...)` line
appears in the log. Same symptom mod-105 chapter 8 opens with;
this section is about the *decision* side, not the diagnosis
side — read that chapter for the fabric-side tools.

### Tools

The tool set from mod-105 chapter 8 applies verbatim: NCCL debug
log, per-rank stack traces via `py-spy dump`, per-NIC counters,
switch counters. Add to it:

- **The rendezvous backend's log.** If the timeout is followed by
  agent-level restarts, the rendezvous log tells you whether the
  reshape happened.
- **DCGM per-node process count.** If N-1 nodes show a training
  process and the Nth does not, the missing node's worker died
  and NCCL is waiting on it.

### Likely causes

The mod-105 list applies (a rank died, a rank fell behind, a
fabric-level stall, a PXN misconfiguration). Add to it the
training-loop-specific causes:

- **Uneven data.** Chapter 6 covers the straggler-detection
  drills; a rank whose step took 3× the median because of an
  unusually large batch is a self-inflicted timeout.
- **Long checkpoint save on the critical path.** A synchronous
  DCP save that took longer than the timeout will trigger a
  hang on the *next* collective. Chapter 3's `async_save`
  eliminates this.
- **Sampler ran out of data on one rank.** A rank that finishes
  its epoch shard earlier than the others has no batch for the
  next step's forward pass; it hangs there while the others
  continue.

### Decision

- If the timeout is fabric — mod-105 chapter 8's runbook. Diagnose
  before restart.
- If the timeout is a dead rank — the elastic flow from chapter 4
  is the recovery. Let the rendezvous re-form; DCP resharding
  handles the world-size change.
- If the timeout is a slow-but-alive rank — chapter 6's straggler
  runbook. Quarantine the node, force a rendezvous, resume with
  N-1.
- If the timeout is a self-inflicted uneven-workload issue — fix
  the workload; do not just raise the NCCL timeout to hide it.

The NCCL timeout kwarg on `init_process_group` is a *load-bearing*
knob. Set it deliberately: long enough that a normal checkpoint or
LR-warmup batch does not trip it, short enough that a real
incident is detected fast. mod-105 chapter 8 discusses the
trade-off; from the training-side, prefer 1–3 minutes for elastic
runs and rely on chapter 6's per-step-time monitor for the finer
signal.

### Post-mortem note

"Raise the NCCL timeout" is a band-aid. Every time you raise it,
you also raise the window during which a *real* incident goes
undetected. If you find yourself raising it repeatedly, one of the
following is true: the checkpoint is too slow, the workload is
uneven, or the fabric is genuinely misbehaving. Fix the underlying
cause.

## Class 4: Hardware fault

### Symptom

The job is running (or is not — hardware faults sometimes fall out
as class 2 or 3), but a hardware-side signal has fired:

- DCGM's `dcgmi diag` reports a failed test or an `Xid` error on a
  GPU.
- Node kernel log (`dmesg`) shows a `NVRM: Xid` line, a bank of
  correctable-ECC errors, or a NIC link flap.
- The scheduler's node-health probe marks the node as unhealthy.

### Tools

- **DCGM (`dcgmi`, `dcgm-exporter`).** Field diagnostics, ECC
  counters, thermal state. Chapter 6 covers the specific counters
  to watch.
- **`nvidia-smi` and `nvidia-smi topo -m`.** For a live GPU
  inspection.
- **`ibdiagnet` / `mlnx_perf` / `ethtool -S`.** For NIC-level
  signals — same tools as mod-105 chapter 8.
- **Kernel log (`dmesg -T`).** Look for `NVRM: Xid`, `mlx5_core`,
  `EDAC`, `mce` messages.

### Likely causes

- **GPU-side Xid error.** NVIDIA's numbered error codes.
  Uncorrectable ECC is Xid 48 / 63 / 64 range; MMU faults are Xid
  31; PCIe timeouts are 79. The NVIDIA Xid documentation
  (https://docs.nvidia.com/deploy/xid-errors/) is the canonical
  reference; keep it bookmarked.
- **HBM row remapping exhausted.** Modern GPUs can transparently
  remap failed HBM rows; when the spare-row pool is exhausted,
  further errors are user-visible.
- **NIC or optic-transceiver failure.** Same failure mode class as
  mod-105 chapter 8's silent NIC drop, but sometimes surfaces here
  first via the switch's port-error counters.
- **Thermal throttling.** GPU or NVSwitch temperature past a
  vendor-set threshold; clocks drop; performance collapses. Almost
  always a data-center facilities issue (an HVAC problem, a stuck
  fan tray) rather than a per-GPU issue.

### Decision

- **Never re-use a suspect GPU without a clean DCGM diagnostic.**
  If Xid says the GPU is compromised, the node is out until
  hardware ops clears it.
- **Elastic reshape around the drained node.** Chapter 4's flow.
- If the diagnostic points at a rack or SU (multiple nodes with the
  same signal), escalate to the data-center team and consider
  suspending the run — a rack-level issue that affects half your
  gang will keep re-firing.
- **Do not conflate "the run recovered" with "the hardware is
  fine".** A run that resumed via elastic reshape *without* the
  bad node is a run that will fail again the next time the
  scheduler re-includes that node. Node quarantine (chapter 6) is
  the fix.

### Post-mortem note

Hardware faults are the class where the platform team's job is
mostly "detect, quarantine, escalate". The remediation is a
different team's job. Your job is to make sure the training-side
recovery (rendezvous + reshape) works smoothly and that the bad
node does not creep back in until it is cleared.

## Class 5: Silent data corruption

### Symptom

**Nothing.** No error message, no crash, no NCCL timeout. The
training loop keeps running at rate. The loss curve looks normal at
a glance. Days later, one of the following surfaces:

- A downstream evaluation shows the model has degraded relative to a
  reference run.
- A specific rank's per-step loss quietly diverges from the others
  after having been aligned.
- A restore from a checkpoint written in the silently-corrupted
  window fails to reproduce the loss curve when re-run on healthy
  hardware.

### Tools

- **Cross-rank loss agreement checks.** Every N steps, all-reduce
  the training loss and log both the mean and the variance across
  ranks. In a healthy setup with a well-shuffled sampler, the
  cross-rank variance is small; a persistent rank whose loss drifts
  is a red flag.
- **Deterministic replay.** Take a DCP checkpoint at step S, spin up
  a small sanity job on known-good hardware, run it forward 100
  steps with the same data. Compare the loss curve to what the
  production run produced. Divergence points at silent corruption
  in the production hardware.
- **DCGM's SDC / RowRemap counters.** GPUs surface some
  soft-corruption signals via DCGM's field values. Not exhaustive.
- **Vendor Blackwell-generation SDC diagnostics.** For B200-class
  hardware, NVIDIA has added silent-data-corruption checks; see the
  DCGM release notes for the current field IDs.

### Likely causes

- **HBM bit flip that got past ECC.** ECC catches most, but not all,
  bit flips. When one gets through, the affected weight or
  activation is subtly wrong.
- **A GPU with a defective computational unit.** Certain tensor
  cores or SMs can be silently wrong — see the Meta / Google SDC
  papers in `resources.md`. Chapter 6 covers the periodic
  compute-check drill for this.
- **Data-side corruption.** A shard on parallel FS whose contents
  drifted (silent bit flip in storage) reads back non-canonical
  values; the loss on batches from that shard is subtly off.
- **Cross-rank arithmetic mismatch.** Two ranks with slightly
  different NCCL versions or slightly different NCCL algorithm
  choices can produce different collective results in floating
  point. Legal (float-add is not associative), but a red flag when
  it manifests as loss drift.

### Decision

- **This is the class that requires proactive detection**, not
  reactive response. If your platform waits for the on-call to
  notice, you have already lost days of work.
- **Chapter 6's straggler + SDC detection loops** are the
  investment. Automatic node quarantine on the first cross-rank
  divergence beyond threshold, with the specific rank recorded for
  hardware follow-up.
- **Rollback + hardware quarantine.** If SDC is confirmed, roll
  back to a checkpoint before the confirmed-good window ended, and
  do *not* let the affected hardware back into the pool.

### Post-mortem note

The reason SDC gets its own class is that the on-call flow above is
different from every other class: the SLA is not "how fast do we
respond" but "how fast do we notice". The metric that matters is
**silent-corruption dwell time**: from the first corrupted
gradient to the first alert. Every mature training platform tracks
it directly.

## The unified playbook shape

Every entry above follows the same **symptom → tools → causes →
decision → post-mortem** shape. Exercise 3 asks you to codify this
shape for your own platform — because a runbook that lives only in
one on-call's head is not a runbook.

Two rules for playbook authoring:

1. **The decision must be prescriptive.** "Investigate further" is
   not a decision. "Roll back to the last DCP checkpoint before the
   spike, decrease LR by 30%, resume, page the ML researcher" is a
   decision. The decision line is what the on-call reads at 3 AM.
2. **The tools list must be runnable commands.** Not "check ECC
   counters" but `dcgmi dmon -e 322,323`. If the on-call has to
   figure out the flag, they will not.

## Summary

- Training-run incidents fall into five classes: loss spike, NaN,
  NCCL timeout, hardware fault, silent corruption. Each has a
  different decision surface even when the surface symptoms look
  similar.
- Investment inverts incident rate. Loud incidents (NaN, timeout)
  are cheap to catch. Silent incidents (SDC, slow straggler) are
  expensive; they deserve disproportionate detection engineering.
- Every playbook follows symptom → tools → likely causes →
  decision → post-mortem note. The decision line is prescriptive;
  the tools line is runnable.
- "Raise the NCCL timeout" and "just restart on the same hardware"
  are the two most common wrong reflexes. Both hide the real
  incident and let it re-fire.
- Exercise 3 turns this chapter into a runbook for your own
  platform. Chapter 6 gives you the detection primitives; chapter
  7 grounds the taxonomy in real logbooks.
