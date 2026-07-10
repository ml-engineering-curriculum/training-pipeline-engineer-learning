# The Run-Time Signature Catalog

Chapters 1–4 built the observability stack. This chapter is the
on-call training that goes with it. A **run-time signature** is a
distinctive pattern across the panels from chapter 1 that maps to
a specific class of incident. Fluency with signatures is what
separates a fresh training on-call from an experienced one — the
experienced on-call sees the panel shape and knows within thirty
seconds which of five things it is, which panel to open next, and
which team to escalate to.

The signatures below cover the vast majority of training-time
incidents in production. They complement, but do not replace, the
runbooks in adjacent modules:

- **Fabric failure modes** live in mod-105 chapter 8 (NCCL
  timeout, silent NIC drop, congestion tree, PXN mis-config).
- **Checkpoint / rescue playbooks** live in mod-106 chapter 5
  (loss spike, NaN, hardware fault, silent corruption from the
  *rescue* side).
- **MFU regressions** live in mod-107 (the causes; this chapter
  owns the *detection*).

This chapter's job is signature → next action. mod-106 chapter 5
owns the go-rewind-to-checkpoint decision; this module owns the
signature that fires the decision. That split is what keeps both
runbooks short.

## The signature template

Every signature entry uses the same shape:

- **Signature.** What the panels look like.
- **Which chapter-1 panels move.** Panel numbers 1–8.
- **Underlying telemetry.** What DCGM / NCCL / tracker says.
- **Likely causes.** Ordered by probability in practice.
- **First mitigation.** What the on-call does in the first
  60 seconds.
- **Escalation.** Which adjacent runbook (mod-105 chapter 8,
  mod-106 chapter 5) picks up next.

Snapshot before you act. As with mod-105 chapter 8, restart is
the enemy of diagnosis. Snapshot dashboards, tracker page,
`NCCL_DEBUG=INFO` log, per-rank stack traces first; then act.

## Signature 1: divergence

### Signature

The training loss climbs monotonically over dozens of steps and
does not recover on its own. Gradient norm p95 climbs alongside,
often exponentially. Eventually you see `NaN` or `Inf` in the
loss and the run either wedges (`grad_norm = inf` blocks the
optimizer) or crashes (`torch.isnan(loss)` assert).

### Panels

- **Panel 1 (loss).** Sharply upward. This is what pages you.
- **Panel 2 (gradient norm).** p50 climbing, p95 climbing
  faster.
- **Panels 3, 4 (throughput, MFU).** Flat until the crash.
- **Panels 5, 6, 7, 8 (cluster view).** Quiet. This is a
  math / config problem, not a fabric problem.

### Telemetry

Tracker: `loss` monotone up, `grad_norm` growing. Prometheus /
DCGM: nothing anomalous. NCCL: silent. The absence of cluster-view
noise is diagnostic — if only the model view is moving, the
model / config is where you look.

### Likely causes

- **Learning rate too high for the current tokens-seen budget.**
  Bad LR schedule, forgotten warmup, or a resume that reset the
  scheduler to peak LR. This is the most common cause.
- **Loss-scale collapse in mixed precision.** In BF16 you rarely
  see this; in FP8 (Transformer Engine) an amax miscalibration
  or delayed-scaling bug can push activations out of range.
- **Data catastrophe.** A shard of the corpus is corrupted or
  malformed, tokens are out-of-vocab, embedding table blows up.
- **Silent code regression.** Someone changed a norm layer, an
  activation, or the residual scale; the model just does not
  train under the new geometry.

### First mitigation

Do not restart. Freeze the run in place if you can (SIGSTOP if
the scheduler supports it) and diff:

1. Config against the last-good run's `run_manifest.json`.
2. Container digest against the last-good run's digest.
3. Code commit against the last-good run's git state.

If everything matches, the issue is data or LR schedule; check
the first bad step's batch composition in the tracker (or the
loader's `sample_ids` history). If the config diverges from last
good, the regression is your first suspect.

### Escalation

Roll back to the last checkpoint whose eval-loss curve was still
descending — mod-106 chapter 5 owns the decision framework.

## Signature 2: loss spike

### Signature

A single sharp upward jump in loss over one or a few steps,
followed by gradual recovery over the next dozens to hundreds
of steps. Gradient norm shows a matching sharp spike on the same
step. The run does not diverge — it recovers to the pre-spike
loss trend within some N steps.

This one is documented explicitly in the OPT-175B logbook and
Meta's Llama 3 report §6.3.
<!-- needs-research: exact Llama 3 report section number for
run-time signatures; anchor URL for the report. -->
Both note that some spikes recover on their own and some
don't; the operational question is which kind you are looking
at.

### Panels

- **Panel 1 (loss).** Single step up, then curve continues
  descending.
- **Panel 2 (gradient norm).** Matching spike, then back to
  baseline.
- **Panel 5 (per-rank step-time histogram).** A vertical
  stripe on the spike step — every rank hit it, so this is
  data-driven or FP8-scale-driven, not a straggler.
- **Panels 6–8.** Quiet.

### Telemetry

Tracker: one bad point on loss, one bad point on grad norm,
otherwise nominal. Cluster view: quiet. Nothing in DCGM.

### Likely causes

- **A "bad batch" of tokens.** A shard boundary produced a
  window of highly-repetitive or highly-noisy tokens, and the
  model's step on that batch produced a large gradient.
- **FP8 amax miscalibration in Transformer Engine.** The
  delayed-scaling recipe re-samples amax on a fixed cadence; a
  single step's amax was stale and activations clipped.
- **Optimizer step transient.** Adam's first moment for a
  freshly-initialized parameter or a warmup transition
  occasionally spikes.
- **Weight-decay coupling with an LR change.** After a warmup
  transition, weight decay's effective magnitude changes
  suddenly.

### First mitigation

Do not react. Loss spikes that recover are normal noise at large
scale; over-reacting (rolling back, changing the LR schedule
mid-run) does more harm than good. Instead:

1. Mark the spike step in the tracker with a `spike` tag and
   the pre/post loss values.
2. Verify recovery over the next N steps (N ~ 100–1000
   depending on your batch size).
3. If recovery does not happen, this is no longer a spike —
   it is divergence, and signature 1 applies.

### Escalation

If spikes recur at high frequency (multiple per epoch), the
underlying cause is real; that is a data-quality escalation to
mod-103, or a mixed-precision escalation to mod-107 chapter 5
(FP8 recipe).

## Signature 3: throughput cliff

### Signature

Tokens/s/GPU drops abruptly by 20% or more, then stabilizes at the
new lower level. Loss keeps descending. Gradient norms are
normal. MFU on panel 4 drops in step. The step-time histogram
(panel 5) either widens (a straggler appeared) or shifts to the
right uniformly (a fabric event).

### Panels

- **Panel 3 (tokens/s/GPU).** Cliff. This is what pages you.
- **Panel 4 (MFU).** Matching cliff.
- **Panel 5 (per-rank step-time histogram).** Either widens
  (straggler; signature 4) or shifts uniformly (fabric).
- **Panel 7 (NCCL comm overlay).** Comm fraction of step time
  goes up. This is the sub-panel that discriminates fabric
  from compute.
- **Panels 6, 8.** Look for HBM temp climbing, XID events,
  ECC errors.

### Telemetry

DCGM: sometimes an XID event coincident with the cliff, or a
throttle-reason bit set (thermal, power). NCCL: `nccl-tests`
run right now shows `busbw` down vs. baseline. Fabric counters:
`port_xmit_discards`, `symbol_errors`, PFC pause frames all
candidates.

### Likely causes

- **Silent NIC drop.** A single node's NIC is silently
  retransmitting; NCCL step time increases. mod-105 chapter 8
  failure mode 2.
- **PXN misconfiguration after a cluster change.** mod-105
  chapter 8 failure mode 4.
- **Congestion tree on RoCEv2.** mod-105 chapter 8 failure
  mode 3.
- **Thermal throttling on one rack.** Airflow degraded,
  HBM temp climbs, GPU clocks down-clock. DCGM's
  `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` names it.
- **Loader-side regression.** A staging cache miss started
  hitting the object store; storage lane saturated. See mod-105
  chapter 6.

### First mitigation

Compare panel 5 shapes:

- **Uniform shift right → fabric.** Escalate to mod-105
  chapter 8; run `nccl-tests` against baseline.
- **Widened distribution / one row darker → straggler.**
  Signature 4 below.
- **Comm fraction on panel 7 flat, compute fraction up →
  compute regression.** Escalate to mod-107.

Do not restart the job. The state contains the evidence.

### Escalation

mod-105 chapter 8 for fabric root-cause; mod-106 chapter 4 for the
straggler-quarantine decision; mod-107 for compute regressions.

## Signature 4: straggler

### Signature

Cluster-wide step time creeps up. Tokens/s/GPU degrades gradually
or steps down. Per-rank step-time histogram (panel 5) widens
substantially — one rank (or a small set) has step times 20-50%
longer than the median. Every collective is bottlenecked on
that rank's slow step.

### Panels

- **Panel 5.** *This* is the signature panel. One row is
  darker than the rest; the row is the straggler rank.
- **Panels 3, 4.** Throughput / MFU drop by 15-30%.
- **Panel 6 (DCGM SM/HBM heatmap).** The straggler node may
  show elevated HBM temp, throttle reasons set, or SBE ECC
  activity.
- **Panel 7 (NCCL comm overlay).** Comm fraction of step time
  goes up because every rank waits for the slow one on each
  collective.

### Telemetry

DCGM on the straggler node: `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`
set (thermal / power), `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL`
incrementing, `DCGM_FI_DEV_MEMORY_TEMP` above baseline. NCCL: no
timeout, just slow collectives.

### Likely causes

- **Thermal / airflow issue on one node.** Fan degraded, dust,
  or a hot spot in the rack.
- **Failing HBM stack.** SBE ECC increment rate rising; the
  card is about to lose a die.
- **Bad NIC on one node.** Same node as signature 3's silent
  NIC drop but caught on the per-rank view rather than the
  aggregate.
- **Uneven data.** Sequence packing produced a rank with much
  longer sequences than others; this is a code-side issue, not
  hardware.

### First mitigation

Identify the straggler rank from panel 5, map to node via the
scheduler's placement. Then:

1. Cordon the node from new work (scheduler side).
2. Decide whether to *drain immediately* (kills the current
   run; requires mod-106 rescue) or *finish current step and
   requeue* (safer, small MFU loss).
3. Log the incident against the node ID; if the node has been
   the straggler N times in the last M runs, quarantine it and
   send it to hardware review.

### Escalation

mod-106 chapter 4 owns the node-quarantine automation; mod-105
chapter 8 owns the fabric side if NIC drop is suspected.

## Signature 5: silent corruption

### Signature

The most dangerous. All the panels look nominal: loss is
descending, gradient norm is stable, throughput is on target,
MFU is at expected. But evaluation loss stalls or degrades over
multiple eval checkpoints. Or: a run finishes, weights are
uploaded, and the downstream fine-tune / eval shows the model
has learned less than it should have given the training tokens.

Nothing in the cluster view or the model view *at step
resolution* fires. The signature only becomes visible when you
compare eval loss over many evals or against a sibling run.

### Panels

- **Panels 1-8.** All nominal.
- **Panel 6 (DCGM heatmap).** May show cumulative
  `ECC_DBE_VOL_TOTAL` increment on one card.
- **Panel 8 (XID / ECC).** Cumulative counter drift on the
  affected node.
- **Eval loss panel** (not on the top screen; on the
  researcher's tracker view). Stalled or wandering.

### Telemetry

DCGM: `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL > 0` on a specific GPU is
the smoking gun when it happens. NCCL: silent. Tracker: eval
metrics diverge from expected trend.

### Likely causes

- **Uncorrectable ECC on HBM.** A double-bit error corrupted a
  parameter or a gradient. NCCL then all-reduced it into every
  rank's copy of the parameter. Weights are now corrupted;
  training continues but the model is silently degraded.
- **NCCL data race / driver bug.** Extremely rare, but has
  happened. A collective wrote to a buffer while another
  op was reading from it. Manifests as small numerical
  discrepancies that compound over training.
- **PCIe / NVLink bit flip.** Extremely rare with modern ECC;
  even rarer to escape all ECC layers.
- **Storage-side corruption on load.** A checkpoint's shard was
  read incorrectly on resume; the shard-manifest hash check
  in chapter 4 catches this if you wired it up.

### First mitigation

The most conservative response is the right one:

1. Stop the run.
2. Roll back to the last checkpoint whose eval metrics were
   on trend.
3. Quarantine any node that showed a DBE ECC event since that
   checkpoint.
4. Restart on a clean fleet.

The GPU-hours lost are much smaller than the GPU-hours you have
already spent training a silently-corrupted model.

### Escalation

mod-106 chapter 5 owns the rollback decision framework. mod-105
chapter 8 owns the hardware quarantine. This signature triggers
both; you route to whichever picks up first.

## The cross-reference table

Quick-reference for the on-call:

| # | Signature | First panel | Discriminating panel | Mitigation | Escalation |
|---|-----------|-------------|----------------------|-----------|------------|
| 1 | Divergence | 1 (loss up) | 8 (fabric quiet) | Diff config / container / code | mod-106 ch 5 |
| 2 | Loss spike | 1 (one bad step) | 5 (vertical stripe) | Wait for recovery | mod-103 if recurring |
| 3 | Throughput cliff | 3 (tokens/s down) | 7 (comm fraction) | Compare to baseline | mod-105 ch 8 |
| 4 | Straggler | 5 (one row darker) | 6 (throttle reasons) | Cordon node | mod-106 ch 4 |
| 5 | Silent corruption | eval loss (drill-down) | 8 (cumulative ECC) | Rollback | mod-106 ch 5 + mod-105 |

Print it. Tape it to your monitor. In an incident you should
know which row you are on by the time you have finished reading
the page.

## Summary

- Five canonical training-time run-time signatures: divergence,
  loss spike, throughput cliff, straggler, silent corruption.
  Fluency with these is the on-call skill this module trains.
- Signatures compose across chapter 1's panels — no signature
  fires on a single panel alone. Read the whole screen.
- Divergence and loss spike are model-view signatures (the
  cluster view is quiet). Throughput cliff and straggler
  are cluster-view signatures. Silent corruption is diagnosable
  only by comparing eval trends over time.
- The first move on any signature is snapshot, not restart.
  Runbook state is diagnostic state.
- Every signature escalates to an adjacent runbook: mod-105
  chapter 8 for fabric, mod-106 chapter 4 for straggler
  quarantine, mod-106 chapter 5 for rollback, mod-107 for
  MFU regressions, mod-103 for corpus issues.
- The cross-reference table is the on-call's first-thirty-second
  triage tool. Keep it visible.
