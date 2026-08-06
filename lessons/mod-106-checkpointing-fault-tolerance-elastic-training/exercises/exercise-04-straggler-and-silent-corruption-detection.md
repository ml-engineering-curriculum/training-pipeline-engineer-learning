# exercise-04: Straggler and Silent-Corruption Detection

**Estimated effort:** 4 hours

## Objective

Implement, and defend the thresholds for, the three-loop detection
stack from chapter 6: per-step-time dispersion, cross-rank loss
agreement, and a compute health check backed by DCGM. Wire the
detectors into the health-check state machine (green / yellow /
orange / red) and demonstrate — via an injected fault — that a red
transition triggers the node-quarantine automation without a human
in the loop. The deliverable is code plus a threshold-defense
report.

## Prerequisites

- Chapter 6 of this module (required) and chapter 5 (required for
  the class-4 / class-5 mapping).
- Exercise 1's DCP trainer as the base for the training-side
  instrumentation.
- DCGM installed on the training nodes (`dcgmi` on the command
  line, `dcgm-exporter` if you already have Prometheus scraping).
  If DCGM is not available, the exercise is still runnable — you
  synthesize the hardware-side signal and note the limitation in
  the report.
- A way to inject faults for the demo: a script that can throttle
  a GPU (`nvidia-smi --lock-gpu-clocks=<low>`), simulate an ECC
  event (via DCGM's `dcgmi test --inject`), or artificially delay
  one rank (a `time.sleep(0.5)` guarded by a rank-N check).
- Access to the scheduler-side node-labeling primitive from mod-104
  (Kubernetes label / SLURM `scontrol update NodeName=... State=DRAIN`
  / cloud instance tag). If you do not have production scheduler
  access, a mock quarantine action that just writes to a file is
  acceptable — say so in the report.

## Problem statement

Your platform currently detects stragglers and silent corruption
manually: someone notices the loss curve on a dashboard, files a
ticket, a hardware ops person eventually finds an ECC-remap event
in `dmesg` from two days earlier, and the affected node was already
back in service. Chapter 6 argues this is exactly the wrong
investment shape — silent incidents are the ones that need
proactive detection, and detection has to be automated to the point
of quarantine.

Your job is to build the three-loop detection stack, calibrate the
thresholds against a healthy baseline, inject three synthetic
faults, and confirm that the automation catches each. Along the
way, you produce the threshold-defense report that goes into your
platform's runbook so future changes to the detectors are audited
against the same defense.

## Requirements

Deliver a directory containing:

- `detectors/` — the three detectors as Python modules importable
  by exercise 1's trainer.
- `automation/` — the quarantine action script.
- `bench.sh` — the injection driver.
- `report.md` — the threshold-defense and injection report.

### 1. Detector: per-step time dispersion

Instrumented as chapter 6 sketches:

- Timing captured with `time.monotonic()` around each step; sync
  before the end timestamp.
- Every K steps (`K = 50` is a reasonable starting point),
  all-reduce max, min, mean across ranks.
- Compute the **dispersion ratio** `max / mean` and the
  **worst-rank identity**.
- Emit a per-step-time event to a structured log (JSONL) with
  fields `step`, `max_ms`, `mean_ms`, `min_ms`, `dispersion`,
  `worst_rank`.

Also emit a **consecutive-worst-rank counter** — how many
consecutive windows the same rank has been the worst. This is what
chapter 6's state machine keys off of.

### 2. Detector: cross-rank loss agreement

- Every N steps (`N = 100` is a reasonable start), all-gather the
  per-rank scalar loss.
- Compute mean, stddev, and the per-rank relative deviation from
  mean.
- Log the `worst_rank`, `worst_rel_dev`, and the number of samples
  in a rolling window where the same rank has been the worst.

Two subtleties to handle in the code:

- Skip the check for the first W steps of a fresh training run
  where losses are dominated by init noise.
- Handle the "one rank OOMed and its loss is nan" case gracefully
  — do not let a nan poison the mean.

### 3. Detector: DCGM + compute health check

DCGM-side, running as a sidecar:

- Poll DCGM every 10 seconds. Capture the field IDs from chapter 6
  (Xid, ECC SBE / DBE volumes, remap-pending, thermal violations,
  clock-throttle reasons). Emit a per-node event on any delta.

Compute-check-side, running from the trainer:

- Every M steps (`M = 500` is fine), each rank runs a
  deterministic fixed kernel (e.g., a matmul on a fixed input) and
  hashes the output. All-gather the hashes.
- Any rank whose hash differs is either a driver / version
  mismatch or an SDC. Log the divergent rank as an `orange`
  signal; if you can reproduce it on a re-run at the same step,
  promote to `red`.

Chapter 6 discusses NVIDIA's `dcgmi diag` as a higher-cost
periodic check; use it as an on-suspicion drill, not a per-step
loop.

### 4. Aggregator: the health-check state machine

Per-node state machine (as chapter 6 defines):

- **Green**: no active signals.
- **Yellow**: one low-severity signal in the last window.
- **Orange**: pattern of yellow signals; specific thresholds below.
- **Red**: severity-1 signal (Xid, DBE ECC, exhausted remaps,
  confirmed compute-check divergence, sustained cross-rank loss
  deviation past a very high threshold).

Publish the exact thresholds for **your** platform, defended in
the report. Suggested (defensible) starting values, to be
calibrated against the baseline:

- **Yellow → Orange (dispersion):** dispersion > 1.10 for ≥ 100
  consecutive windows.
- **Yellow → Orange (loss agreement):** same worst_rank with
  rel_dev > 3 × baseline stddev for ≥ 20 consecutive samples.
- **Yellow → Orange (ECC SBE):** rate > 3× the cluster-wide
  baseline over a 24 h window.
- **Orange → Red:** any Xid event, any DBE, any remap-pending
  failure, any confirmed compute-check divergence, or persistence
  in the orange state past a defined dwell time.

Transitions must be logged with timestamps and the specific
detector signal that caused them.

### 5. Automation: the quarantine action

On a red transition, the automation must:

1. Label the affected node in the scheduler (or write to a mock
   file if you have no scheduler access).
2. Send a SIGKILL to the worker on that node with a distinctive
   exit code the elastic agent can recognize.
3. Post a structured ticket / page (a stdout line is fine for the
   exercise) with the full detector context: which detector
   fired, which fields, which recent history.
4. **Never** clear the quarantine automatically. The clear action
   is deliberately manual and requires a passing DCGM diagnostic
   as a precondition — implement the check even if you never run
   it in the exercise.

### 6. The threshold-defense report

`report.md` covers:

- **Baseline calibration.** Ten to twenty minutes of healthy
  training, per-detector summary statistics (mean, stddev, P99).
  This is where your thresholds come from; publish the raw numbers.
- **Injected fault 1: throttle a GPU.**
  `nvidia-smi -i <gpu> --lock-gpu-clocks=1200` on one rank's GPU
  during training. Record: how long from the injection to the
  first yellow signal, from the first yellow signal to orange,
  from orange to red. Confirm the quarantine action fired.
- **Injected fault 2: simulated ECC event.** Use
  `dcgmi test --inject` (or the fallback: a stub event injected
  into your detector code) to fire an SBE / DBE. Same timeline.
- **Injected fault 3: cross-rank loss divergence.** Rig one rank
  to add a small numerical bias into its loss computation
  (deliberately, in code, guarded by a `--inject-sdc` flag). Same
  timeline.
- **False-positive check.** Run 20 minutes of healthy training
  with all detectors on. Report the count of yellow, orange, red
  transitions. Ideally zero orange and zero red. Any false
  positive is a threshold that needs to be tightened.
- **Discussion.** For each threshold, defend the value with a
  reference to either baseline data or chapter 6's guidance. If a
  threshold "just felt right", change it to one that has a
  numeric defense.

## Starter guidance

- **Baseline first.** Do not calibrate thresholds against your
  intuition; calibrate against 15 minutes of healthy training data.
  Every threshold in the report should be justifiable as "N σ
  above baseline" or "specifically because chapter 6 says so".
- **Instrument the aggregator, not each detector, with the state
  machine.** Detectors emit signals; the aggregator maps signals
  to states. This separation is what lets you evolve detector
  thresholds without touching the machine.
- **Use structured logs everywhere.** JSONL per detector, one
  file per node. Post-hoc analysis is much easier and the report
  numbers become reproducible.
- **The injection scripts should be idempotent and reversible.**
  You will run each injection several times while tuning. A GPU
  clock lock that you forget to reset will haunt the next
  drill.
- **Do not skip the false-positive check.** A detector that fires
  on healthy training is worse than no detector.
- **If you do not have DCGM.** Fake the DCGM side of the stack
  with a script that emits synthetic events. Note in the report
  that the DCGM detector was not evaluated on real data, and
  scope your claims accordingly.

## Acceptance criteria

- All three detectors are present, each emitting structured
  events with the fields specified above.
- The state machine has explicit numeric thresholds published in
  the report with a baseline-derived justification for each.
- All three synthetic faults are injected, and for each the
  report shows a timeline from injection to quarantine.
- The false-positive check shows zero orange / red transitions
  during a healthy window, or any positives are analyzed and the
  thresholds adjusted.
- The automation actually fires the quarantine action (real or
  mocked) on a red transition. A worker on the affected node
  exits; the surviving ranks continue if elastic (chapter 4).
- No auto-clear on the quarantine. The clear-action code path
  exists and requires a passing DCGM diagnostic; the report
  discusses why this design.
- No invented numbers. If a specific detector could not be
  demonstrated on real hardware (no DCGM, no way to inject an
  ECC event), the report says so and scopes the claim
  accordingly.

## Stretch goals

- **Nightly regression drill.** Wrap the injection driver into a
  scheduled job that fires nightly against a canary training run.
  Confirm the detectors + automation continue to work after
  PyTorch / NCCL / DCGM upgrades. Chapter 7's recovery-testing
  pattern applies verbatim.
- **Silent-corruption replay drill.** Implement chapter 6's
  deterministic-replay drill end-to-end: take a DCP checkpoint
  from a suspect window, spin it up on known-good hardware, run
  100 steps, compare loss trajectories. Automate the trigger from
  a cross-rank-loss-agreement orange signal.
- **NVIDIA-Resiliency-Ext integration.** Adopt one of the
  detectors from the open-source
  https://github.com/NVIDIA/nvidia-resiliency-ext package and
  compare its output to your hand-rolled equivalent. Where they
  agree, choose based on maintenance cost; where they disagree,
  investigate the disagreement.
- **Composite yellow-signals aggregator.** Extend the state
  machine to a real cross-detector model — e.g., a rank that has
  both a rising SBE ECC rate and a persistent dispersion signal
  is orange even if neither alone would qualify. Discuss the
  false-positive vs. false-negative trade-off you are choosing.
