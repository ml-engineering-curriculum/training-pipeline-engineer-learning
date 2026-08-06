# Straggler Detection, Silent-Corruption Detection, and Node Quarantine

Chapter 5 named silent corruption and slow-but-alive stragglers as
the two incident classes where **detection**, not response, is the
dominant investment. This chapter is that detection layer: the
signals to instrument, the thresholds to page on, and the automation
that turns a signal into a node-out-of-service action without waking
anyone up.

The primary references for the tools in this chapter:

- **NVIDIA Data Center GPU Manager (DCGM) documentation.**
  https://docs.nvidia.com/datacenter/dcgm/latest/ — field IDs,
  policies, integrations. `dcgm-exporter` is the Prometheus bridge.
- **NVIDIA Xid error reference.**
  https://docs.nvidia.com/deploy/xid-errors/ — every kernel Xid
  code, its meaning, and its severity.
- **Meta / Google silent-data-corruption papers**, listed in
  `resources.md` — the empirical grounding for the "quiet" nature of
  hardware compute faults at scale.
- **NVIDIA-Resiliency-Ext for large-scale training.**
  https://github.com/NVIDIA/nvidia-resiliency-ext — the open-source
  detection utilities NVIDIA ships for straggler and health checks
  on top of DCGM.

## Straggler detection: the per-step time histogram

The single most useful signal for fail-slow is **per-rank per-step
time**. It is trivially instrumented (a `time.monotonic()` around the
step body) and it surfaces every fail-slow class chapter 5 named.

The instrumentation:

```python
step_start = time.monotonic()
train_step()
torch.cuda.synchronize()
step_end = time.monotonic()
step_time_ms = (step_end - step_start) * 1000

# Cheap all-reduce: get max, mean, and min step time across ranks.
step_tensor = torch.tensor([step_time_ms], device="cuda")
max_t = step_tensor.clone(); torch.distributed.all_reduce(max_t, op=torch.distributed.ReduceOp.MAX)
min_t = step_tensor.clone(); torch.distributed.all_reduce(min_t, op=torch.distributed.ReduceOp.MIN)
mean_t = step_tensor.clone(); torch.distributed.all_reduce(mean_t, op=torch.distributed.ReduceOp.SUM); mean_t /= world_size

if rank == 0:
    log("step_time", step=step,
        max=max_t.item(), mean=mean_t.item(), min=min_t.item(),
        dispersion=(max_t.item() / mean_t.item()))
```

Read the **dispersion ratio** `max / mean`. On a healthy homogeneous
cluster it should be very close to 1.0 (typically within a few
percent), because every rank is running the same code on the same
data and is bound by the same collectives. A dispersion that walks
above ~1.10–1.20 is a strong straggler signal.

Two additional signals worth logging in parallel:

- **Which rank was the max, and for how many consecutive steps.**
  A one-off outlier is fine; the same rank being the max for 50
  consecutive steps is a straggler.
- **Per-rank cumulative wait time.** How long did each rank spend
  waiting in the last collective, according to NCCL's profiler or a
  `torch.profiler` capture? The rank whose wait time is *lowest*
  is often the actual straggler (everyone else is waiting on it).

## Straggler causes: mapping the signal to the root cause

Once you have the "rank X is dispersion-N" signal, the diagnosis
tree:

- **GPU thermal throttle.** DCGM field ID 155
  (`DCGM_FI_DEV_GPU_TEMP`) plus 100–105 for the sm and memory
  clocks. If the affected rank's clock is below nominal, it is
  throttling; if the surrounding neighborhood is also throttling, it
  is an airflow / facilities issue.
- **HBM ECC row-remap.** DCGM field IDs 310–321 track ECC errors
  and remap actions. A GPU actively remapping rows will show slow
  memory access on the affected pages and produce dispersion.
- **NCCL retransmission / NIC error.** Chapter 8 of mod-105 covers
  the tools: `ethtool -S`, `mlnx_perf`. A NIC quietly retransmitting
  slows every collective it participates in.
- **Data-loader stall.** The rank ran out of prefetched batches and
  the loader is blocked on the storage tier. mod-103 chapter 4
  covers the loader instrumentation; a stalled loader shows up as a
  step-time spike with GPU-utilization drop right before the collective.
- **Software-side jitter.** GC pause, Python-side allocation, or a
  logging call that flushes to disk mid-step. Rare on a
  well-instrumented trainer; check it last.

Every diagnosis path starts with the dispersion signal and drops
into one of DCGM / mod-105 / mod-103.

## Silent-corruption detection

Silent-corruption detection is harder than straggler detection
because there is no "wall-clock" signal that flags it. Three
complementary detection loops that cover the failure surface:

### 1. Per-rank loss agreement

Sample every N steps (`N = 100` is a reasonable starting point):

```python
if step % 100 == 0:
    loss_tensor = loss.detach().unsqueeze(0)
    gathered = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, loss_tensor)
    if rank == 0:
        arr = torch.stack(gathered).cpu()
        mean = arr.mean().item()
        stddev = arr.std().item()
        rel_dev = ((arr - mean).abs() / mean).max().item()
        log("loss_agreement", step=step, mean=mean, stddev=stddev,
            worst_rel_dev=rel_dev, worst_rank=int(arr.sub(mean).abs().argmax()))
```

Interpretation:

- On a **well-shuffled sampler with i.i.d. batches**, per-rank losses
  should have a small stddev relative to the mean (order of a few
  percent). Persistent outliers with large `rel_dev` are red flags.
- On a **poorly-shuffled sampler or a shard-locked data pipeline**,
  cross-rank variance can be legitimately larger. Baseline your
  cluster and only page on deviations from the baseline, not on the
  raw number.
- A single spike is not the signal. The signal is **the same rank
  being the worst-rank for many consecutive samples**.

### 2. Deterministic replay for a suspect window

When you suspect SDC on rank X over some window, replay it out of
band:

1. Load the DCP checkpoint from just before the suspect window
   onto a small known-good sanity cluster.
2. Run 100 steps forward using the same data batches (deterministic
   sampler makes this reproducible).
3. Compare the loss curve to what production produced in the same
   window.
4. If the curves diverge beyond a threshold, SDC is confirmed for
   that window on that hardware.

This is a heavier drill; it is the runbook for confirmed suspicion,
not for continuous detection. Exercise 4 walks through the
implementation.

### 3. Periodic compute health checks

Every N steps, or every checkpoint boundary, run a **fixed
deterministic kernel** on every GPU and compare its output to a
known-good reference. NVIDIA's `dcgmi diag -r 3` (or the current
level flag) runs a comprehensive built-in compute check; there are
also community "canary" implementations (a fixed matmul, a fixed
softmax) you can run in a few tens of milliseconds.

Compare each rank's output to a bitwise-identical reference. Any
divergence is either an SDC or a driver/version mismatch — both of
which want a node quarantine until diagnosed.

## DCGM as the always-on background signal

Everything above is training-side instrumentation. The complementary
signal is the **always-on GPU-side telemetry** — DCGM.

Run `dcgm-exporter` on every training node; scrape into your
Prometheus / metrics store; alert on:

- **Any `Xid` event.** DCGM surfaces these as field ID events.
  Chapter 5's class 4 (hardware fault) is what pages on this.
- **ECC error deltas.** Field IDs in the 310s. A jump in
  `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` (double-bit ECC — uncorrectable)
  is severity-1; a rising `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL`
  (single-bit corrected) is severity-2 that indicates a GPU on its
  way out.
- **Row-remap pending / failure.** Field IDs 373–385 (consult the
  DCGM release for the current IDs). Remap failures indicate the
  spare-row pool is exhausted; the GPU is at end-of-life.
- **Thermal violations.** Field IDs for `SLOWDOWN_TEMP_VIOLATION`
  and `THERMAL_VIOLATION`. Persistent violations correlate with
  stragglers.
- **Clock throttling reasons.** Field ID
  `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` (older DCGM: `THROTTLE_REASONS`)
  — the bitmask that tells you why a GPU dropped its clock.
- **NVLink error counters.** For DGX systems; a rising NVLink error
  counter is a chapter 5 class 4 or class 5 signal depending on
  severity.

The two `dcgm-exporter` gotchas:

1. **Field IDs and metric names change between DCGM releases.** Pin
   the exporter version and re-verify field IDs on upgrades. The
   NVIDIA docs page linked above is authoritative for the
   installed version.
2. **DCGM policies do not fire on training-loop faults.** They fire
   on GPU-hardware faults. The other detectors in this chapter
   cover the training-loop side; DCGM is the hardware-side
   complement.

## The node-quarantine automation

Detection is only half the goal. The other half is that a detected
bad node is **out of service** before it gets scheduled again. The
automation pattern:

1. **Signal source.** Any detector above emits an event tagged with
   `node_id`, `severity`, `reason`.
2. **Quarantine action.** For a severity-1 event:
   - Label the node `training-quarantine=true` in the scheduler
     (Kubernetes label, SLURM node drain, cloud instance tag). See
     mod-104 chapter 5 for the scheduler-side plumbing.
   - Fail the currently-scheduled worker on that node with a
     specific exit code the elastic agent can distinguish from a
     software crash.
   - Emit a page or ticket to the hardware team with a link to
     the raw DCGM / detector output.
3. **Elastic reshape.** With the bad node drained, the surviving
   ranks re-rendezvous per chapter 4's flow. The job keeps running
   at `WORLD_SIZE - nproc_per_node`.
4. **Clear-to-return.** The quarantine label is only cleared by
   hardware ops after a passing diagnostic. Never auto-clear on a
   timer.

The single most common bug: **quarantine that clears itself.** A
five-minute timer removes the label, the scheduler re-adds the node,
the same fault re-fires. Every quarantine action has to be paired
with an explicit human-in-the-loop clearance.

## Cross-rank sanity checks as a safety net

A cheap defensive check that catches a class of "the collective
completed but the values were wrong" bugs:

- **Periodic weight-hash all-reduce.** Every M steps, compute a
  cheap deterministic hash of a slice of the model weights on every
  rank. All-gather. Any rank whose hash differs is either (a) a
  legitimate FSDP-sharded case (only certain params live on certain
  ranks) or (b) genuinely different weights. Correctly implemented,
  this is a bitwise-exact check on the *replicated* parameters — LR
  scheduler state, layer-norm scales, etc.
- **Gradient-norm all-reduce agreement.** Similar shape: log per-rank
  gradient L2 norms; a rank whose norm walks off from the mean is
  either learning something the others are not (bug) or has an
  incorrectly-sharded parameter (bug).

These are the equivalent of unit tests that run in production; they
should be cheap enough to leave on and loud enough that a real
divergence pages.

## Composing the detectors: a health-check state machine

Each detector is a source; the aggregator is a small state machine
per node:

- **Green.** No active signals. Node is in service.
- **Yellow.** One low-severity signal in the last window (e.g., a
  single 1.15 dispersion sample; one SBE ECC delta). Log and
  continue.
- **Orange.** A pattern of yellow signals (e.g., dispersion > 1.10
  for 100+ consecutive steps; SBE ECC deltas rising steadily).
  Fire a warning; consider proactive drain at the next checkpoint.
- **Red.** A severity-1 signal (Xid, uncorrectable ECC, exhausted
  remaps, confirmed SDC). Automatic quarantine; page.

The transitions matter more than the levels; you want to catch the
yellow-to-orange drift, not just the red-alert. Exercise 4 asks you
to author this state machine for one detector and defend the
thresholds.

## What NOT to do

- **Do not average away the straggler signal.** A dashboard that
  plots mean step time obscures the worst rank, which is the entire
  point of the metric. Log max, min, and dispersion — not just the
  mean.
- **Do not treat DCGM `dcgmi diag` as sufficient.** It catches
  hardware faults; it does not catch training-loop-level signals
  (uneven data, sampler stall). Both layers are necessary.
- **Do not skip the deterministic replay drill.** It is the only
  way to confirm suspected SDC. Without it, you are guessing.
- **Do not treat quarantine as reversible on a timer.** Every
  auto-clear-quarantine bug becomes a repeated-incident report six
  weeks later.

## Summary

- Straggler detection is per-step time with a dispersion ratio.
  Dispersion above ~1.1 sustained for many steps points at a slow
  rank; diagnose via DCGM (thermal, ECC), mod-105 (fabric), or
  mod-103 (loader).
- Silent-corruption detection needs three loops: per-rank loss
  agreement, deterministic replay for suspects, periodic compute
  health checks. DCGM is the always-on hardware complement.
- DCGM's Xid, ECC, remap, thermal, throttling, and NVLink field
  IDs are the always-on background signal. Pin the exporter
  version and alert on deltas, not absolutes.
- Node quarantine automation: detector event → scheduler label +
  worker fail + page → elastic reshape → human-in-the-loop clear.
  Never auto-clear on a timer.
- The health-check state machine (green / yellow / orange / red) is
  where the individual detectors compose. The transitions are
  where the interesting signals live.
