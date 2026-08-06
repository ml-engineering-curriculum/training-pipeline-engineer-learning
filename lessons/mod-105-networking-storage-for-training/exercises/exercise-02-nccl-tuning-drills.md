# exercise-02: NCCL Tuning Drills

**Estimated effort:** 4 hours

## Objective

Turn chapter 5's tuning philosophy into a set of measured drills against
your own fabric (or a cloud fabric you rent for the day). By the end you
should have a runbook-quality reference of what your cluster does for
each of the four NCCL knob families, and — the point of the exercise —
you should have proven at least one change with `nccl-tests` numbers
rather than intuition. The deliverable is a short "tuning report" a
teammate could compare their fabric against.

The exercise assumes you have some form of RDMA-capable multi-node
setup. If you do not, do the drills against a two-node public-cloud
instance pair (AWS EC2 P5 with EFA, Azure ND-series, GCP A3) and note in
the report that "single-node NVLink drills only" against a single DGX or
equivalent workstation. The `nccl-tests` reading exercise is worth doing
even at that scale.

## Prerequisites

- Chapters 1, 2, 3, and (especially) 5 of this module.
- `nccl-tests` (https://github.com/NVIDIA/nccl-tests) built against the
  same NCCL version you plan to test.
- The NCCL user guide open in a browser
  (https://docs.nvidia.com/deeplearning/nccl/).
- A working `mpirun` or `torchrun` launcher on your cluster with at
  least 2 nodes × 8 GPUs, or 1 node × ≥ 4 GPUs for the NVLink-only
  fallback.
- Ability to write to a runbook / wiki / Markdown file where the
  baselines land.

## Problem statement

Your team has agreed to publish a "fabric tuning report" for every
cluster it runs training on, so that later regressions have a healthy
baseline to diff against and later tuning changes have to defend
themselves in numbers. Your job is to produce the first one for the
cluster in front of you.

The report answers, at minimum, these questions:

- **Discovery.** What topology did NCCL discover on this cluster? Does
  it match the physical wiring described in chapter 2's mental model?
- **Baseline.** What is the `busbw` curve for `all_reduce`,
  `all_gather`, `reduce_scatter`, and `alltoall` across payload sizes
  on this fabric? At what payload does the fabric approach line rate?
- **Algorithm & protocol.** Where does the auto-tuner pick `Tree` vs.
  `Ring`, `LL` vs. `LL128` vs. `Simple`? Do those choices match your
  reading of chapter 5?
- **Rail alignment / PXN.** Is PXN actually helping on your fabric?
  Quantify.
- **Documented deltas.** At least one intentional tuning change (any
  of the knobs in the taxonomy) that you verified either wins or
  loses. Publishing a *loss* is worth as much as publishing a win —
  it prevents someone else from re-running the same experiment.

## Requirements

Deliver a single Markdown report (`nccl-tuning-report.md`) in your
team's runbook. It must contain the following sections, in order.

### 1. Cluster fingerprint

State — in one screenful — what cluster you tested against:

- Node count and GPUs per node used for the tests.
- GPU model (H100 SXM, H200, B200, A100, etc.).
- Interconnect (NVLink4 / NVSwitch3 for intra-node, NDR IB / HDR IB /
  RoCEv2 / EFA-SRD for inter-node).
- NCCL version, CUDA version, driver version, OFED / EFA-installer
  version.
- Launcher (`mpirun` version or `torchrun` version).

If you are on a cloud, name the instance type and region.

### 2. Discovery drill

Run a trivial 2-node × 8-GPU `all_reduce_perf` with `NCCL_DEBUG=INFO`
and `NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET`. Capture the log and pull out:

- The HCA names and rail indices NCCL identified per rank.
- The rings and trees NCCL constructed for the collective.
- The transport (IB / NVLink / P2P / SHM) chosen per rank pair.
- Whether PXN is on or off, and for cross-rail pairs, the intermediate
  rank NCCL picked.

Quote 15–30 lines of the log inline (with sensitive hostnames redacted)
and, in your own words, explain what NCCL saw. If the discovery does
not match the physical wiring you documented in exercise-01, that is
its own finding — do not paper over it.

### 3. Baseline sweep

Run each of the four collectives — `all_reduce_perf`,
`all_gather_perf`, `reduce_scatter_perf`, `alltoall_perf` — with the
canonical payload sweep:

```
mpirun -np <N> \
  -x NCCL_DEBUG=WARN \
  ./<collective>_perf -b 1K -e 1G -f 2 -g 1
```

For each, publish a table of `size / algbw / busbw`. Include at least
one data point at ~4 KiB (the α regime), one at ~1 MiB (the crossover
regime), and one at ≥ 256 MiB (the β regime).

For each collective, state in one line the peak `busbw` and what
fraction of nominal per-link bandwidth that is. Chapter 1's "effective
vs. nominal" framing is what you are quantifying.

### 4. Algorithm & protocol drill

Pick two payload sizes that bracket the tree/ring crossover on your
fabric (say ~64 KiB and ~64 MiB). For each, run `all_reduce_perf` with
`NCCL_DEBUG=INFO` and record which `(algo, proto)` pair the auto-tuner
picked.

Then force each of the following and record `busbw`:

- `NCCL_ALGO=Ring NCCL_PROTO=Simple`
- `NCCL_ALGO=Tree NCCL_PROTO=Simple`
- `NCCL_ALGO=Ring NCCL_PROTO=LL`
- `NCCL_ALGO=Ring NCCL_PROTO=LL128`

Publish the six numbers (auto + four overrides) at each payload as a
small table and answer, in one paragraph:

- Did the auto-tuner pick the winning combination at each payload?
- If not, what is the delta, and does the delta actually matter for
  your training step (order of ms vs. µs)?
- Does forcing `Simple` on very small messages regress as chapter 5
  predicts?

### 5. Rail alignment / PXN drill

On the same 2-node × 8-GPU baseline, run `all_reduce_perf` twice at
your ~256 MiB payload:

- Once with defaults (PXN on).
- Once with `NCCL_PXN_DISABLE=1`.

Publish the two `busbw` numbers. In one paragraph:

- Is PXN a win, a wash, or a loss on your fabric at that payload?
- On rail-optimized fat-tree IB, it should be a modest-to-large win.
  On a single-rail or non-rail-aligned fabric it may be neutral. If
  you see a loss, that is a finding — it means PXN's topology
  discovery is fighting your wiring.
- Cross-reference chapter 8's failure mode 4 for the diagnosis path
  if you found a loss.

If your fabric supports SHARP (IB with the SHARP Aggregation Manager
running), also run:

- `NCCL_ALGO=CollnetDirect all_reduce_perf` at ≥ 256 MiB.

Publish the number and compare against the Ring baseline. On a healthy
SHARP deployment at higher world size (≥ 32 GPUs), Collnet should
begin to pull ahead — small pods will not see the win.

### 6. Documented tuning change

Choose one production-relevant knob change and run a controlled A/B on
`nccl-tests`. Examples of "production-relevant":

- `NCCL_CROSS_NIC=0` vs. `NCCL_CROSS_NIC=1` on your fabric.
- `NCCL_IB_QPS_PER_CONNECTION=1` vs. `4` at the large-payload end.
- `NCCL_NET_GDR_LEVEL` sweep on a cloud fabric to check GDR is
  actually kicking in.
- `NCCL_IB_HCA` restricted to exclude the storage HCA on a shared-NIC
  design.
- A `NCCL_TOPO_FILE` you dumped with `NCCL_TOPO_DUMP_FILE` and
  hand-edited to correct a discovery bug.

For the chosen change, publish:

- The knob you touched and why (one paragraph, referencing chapter 5).
- Baseline `busbw` at your reference payload.
- After-change `busbw` at the same payload.
- Whether the change wins, loses, or is a wash.
- Whether you would ship the change to production. (A "no" here is a
  perfectly valid finding.)

### 7. Baselines saved

The last section of the report is a link (or embedded copy) of the raw
`nccl-tests` output files from sections 3–5. These are the reference
future incidents will compare against. Do not skip this; the baseline
that lives in someone's terminal history is a baseline that stops
existing.

## Starter guidance

- Run every drill on the same idle allocation, back-to-back, so that
  cross-run variance is minimized. Grabbing fresh nodes between drills
  will introduce placement noise.
- Set `NCCL_DEBUG_FILE=/path/to/nccl-%h-%p.log` on every run so the
  debug output does not spam stdout and so you can grep it after.
- The `-g 1` flag on `nccl-tests` means one GPU per MPI rank. With
  `torchrun`, you get one rank per process by default — either works;
  pick one and be consistent.
- On a public cloud, use the vendor-published `aws-ofi-nccl` (EFA) or
  the equivalent GCP / Azure NCCL plugin, not the vanilla IB
  transport. `NCCL_DEBUG_SUBSYS=NET` will tell you which plugin loaded.
- Do not try to tune every knob at once. The A/B in section 6 is one
  change against one baseline. Change one knob, measure, revert,
  repeat.
- The tuning report is a runbook artifact, not a research paper —
  aim for numbers and one-paragraph interpretations, not prose.

## Acceptance criteria

- The report opens with a cluster fingerprint (section 1) that a
  reader could reproduce the environment from.
- Section 2 quotes a real NCCL debug log and interprets it. A reader
  who cannot see your cluster can still tell what NCCL discovered.
- Section 3 has all four collectives, all with a sweep, all with the
  `busbw` numbers. No missing data points.
- Section 4 has six data points per payload and a written
  interpretation. The auto-tuner's choice is called out explicitly.
- Section 5 has a numeric PXN win/loss/wash statement. If you have
  SHARP, it is measured; if you do not, the report says so.
- Section 6 has one A/B with numbers and a ship/no-ship decision. A
  loss is acceptable; a change without measurement is not.
- Every number in the report is reproducible from the saved
  `nccl-tests` output files linked in section 7.
- No invented numbers. If a drill did not run on your cluster (e.g.,
  SHARP unavailable), the report says so and does not fabricate a
  data point.

## Stretch goals

- Add a "world-size scaling" section: repeat the section-3 sweep at
  1, 2, 4, and 8 nodes and plot `busbw(N)` at your reference payload.
  Chapter 5's α + β prediction should approximately hold; the deviation
  from prediction is your fabric's real behavior.
- Write the same report against a *second* fabric — a different
  cluster, a different cloud provider, or the same cluster after a
  known change (firmware, NCCL version). Diff the two reports; the
  interesting findings are in the diff, not in either report alone.
- Instrument a real training step (a DDP or FSDP run against your
  model) with `NCCL_DEBUG=INFO` and confirm the collective sizes and
  transports match what your report predicts. If they do not, you
  have found either a report bug or a real workload behavior worth
  investigating.
- Turn the tuning report into a scheduled job: run the baseline sweep
  nightly against an idle allocation and alert on regressions from
  the saved reference. Chapter 8's post-mortem note ("keep
  baselines") becomes a monitored property rather than a hope.
