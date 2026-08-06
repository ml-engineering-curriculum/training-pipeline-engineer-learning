# exercise-03: InfiniBand vs. RoCEv2 Measurement

**Estimated effort:** 3 hours

## Objective

Produce a defensible, numbers-backed recommendation for whether an
InfiniBand or RoCEv2 (or EFA/SRD) fabric better fits a specific
training workload, and document the measurement protocol that got you
there. The point is to turn the "when RoCEv2 wins" table in chapter 4
into an operationally executable A/B — the kind of document that
survives a design review.

If you have access to both fabric types, run the drills on both. If
you have access to one, run the drills on the one you have and write
the second-fabric section as a hypothetical grounded in vendor docs
and published measurements — clearly labelled as such. Do not
fabricate numbers for a fabric you cannot test.

## Prerequisites

- Chapters 3 and 4 of this module. Chapter 5 for the NCCL-side
  measurement vocabulary.
- `perftest` installed on the test nodes
  (https://github.com/linux-rdma/perftest) — `ib_write_bw`,
  `ib_send_bw`, `ib_send_lat` are the tools you will use.
- `nccl-tests` from exercise 2 (or your team's already-built copy).
- Access to at least one node pair on each fabric you claim to
  compare. On a hyperscaler cloud, this typically means a P5 pair
  with EFA (AWS) or an ND-series pair with IB (Azure) or an A3-series
  pair (GCP) — verify the current SKUs.
- Ability to read your NIC counters: `ethtool -S`, `mlnx_perf`,
  `perfquery` (IB), or the cloud-provider's equivalent metrics.

## Problem statement

Your team is being asked to sign off on the fabric choice for the next
training pod. The stakes: the choice is roughly irreversible for the
life of the pod, and getting it wrong costs single-digit-percent step
time forever. The design review needs a specific, quantitative answer
for a *specific workload*, not a generic "IB is better" or "RoCEv2 is
cheaper".

The workload the pod will run is one of the following — pick whichever
is closest to what your team actually does, or negotiate a plausible
one with a training team you talk to:

- **Workload A: dense LLM pretraining.** 70B-parameter dense decoder,
  hybrid FSDP with shard groups of size 16, long-sequence (~8K–32K
  token) packed batches, distributed checkpoint every ~1000 steps.
  Dominated by ~large-message all-gather / reduce-scatter and periodic
  checkpoint bursts.
- **Workload B: dense LLM post-training / RLHF.** 70B model, plain
  DDP (fully replicated on each DP shard), smaller per-step batches,
  frequent (every ~50 steps) small-model updates over policy /
  value / reward nets. Dominated by mixed-size all-reduces at
  moderate world size.
- **Workload C: MoE / expert-parallel training.** 200B sparse model
  with 64 experts, all-to-all routing every layer. Dominated by
  all-to-all traffic that is small per-source but many-to-many.

Name the workload in the first line of your report; every subsequent
recommendation refers back to its dominant collective pattern.

## Requirements

Deliver one report (`fabric-choice-<workload>.md`) with the following
sections.

### 1. Workload characterization

State, in numbers:

- Model size (parameters), precision, and the framework used.
- The dominant collective(s) per step and their approximate payload
  size. For workload A, the FSDP shard-group all-gather on the
  parameters is the one to size; for B, the DDP all-reduce on
  gradients; for C, the routing all-to-all.
- The world size the fabric choice is being sized for. State it
  explicitly (e.g., 512 GPUs, 64 nodes on 8-GPU DGX).

Chapter 5 of mod-101 gives you the α + β cost model you use to
estimate the collective cost per step. Publish the arithmetic; the
number does not have to be perfect, but the reader has to see it.

### 2. Fabric fingerprint(s)

For each fabric you test, capture the same fingerprint you did in
exercise-02 section 1: node count, GPUs per node, NIC / HCA generation
and port speed, OFED / EFA version, NCCL version. Also capture:

- **On IB:** the SM implementation and version (OpenSM vs. UFM),
  whether SHARP is enabled, and the MTU.
- **On RoCEv2:** the switch model, whether PFC and ECN are enabled on
  the RoCEv2 priority (`show_gids` and switch-side config), and the
  DSCP-to-priority mapping.
- **On EFA:** the `aws-ofi-nccl` plugin version, the placement group
  configuration, and whether the instance type supports SRD 100 Gb/s
  or 400 Gb/s per NIC.

If you can only test one fabric, still write this section for the
other from vendor documentation, clearly labelled `(reference-only,
not measured)`.

### 3. Verbs-layer baseline (`perftest`)

Run `perftest` between two nodes on each fabric:

- **Bandwidth.** `ib_write_bw` and `ib_send_bw` at multiple message
  sizes (say 4 KiB, 1 MiB, 64 MiB) with `RDMA_WRITE` for the
  bandwidth run.
- **Latency.** `ib_send_lat` at the smallest message size the tool
  supports.
- **Cross-rail (if multi-rail).** Repeat the bandwidth run once
  same-rail, once cross-rail. On IB this shows the underlying
  rail-fat-tree wiring; on RoCEv2 it shows what PFC-aware routing is
  doing.

Publish a small table per fabric: `{operation, size, achieved_bw,
%_of_nominal, achieved_lat}`. The `%_of_nominal` column is what makes
the comparison honest — you are comparing efficiency, not just
absolute speed of different port generations.

### 4. NCCL-layer sweep (`nccl-tests`)

Run `nccl-tests` on each fabric at the workload's world size (or the
largest world size you can allocate, noting the reduction):

- The specific collective from section 1 (all-reduce for B,
  all-gather / reduce-scatter for A, alltoall for C). Sweep from
  ~4 KiB to ~1 GiB.
- Publish `busbw` at three payload sizes matching α-regime,
  crossover, β-regime.
- On IB, also run once with `NCCL_ALGO=CollnetDirect` if SHARP is
  available.

For each fabric, state:

- Peak `busbw` and payload where achieved.
- `busbw` at the *workload's* dominant payload from section 1.
- Latency at the α-regime payload.

### 5. Congestion / health snapshot

Capture, on each fabric during the section-4 sweep:

- **IB.** `perfquery` at the port level for `RcvErrors`,
  `LinkDowned`, `SymbolErrors`. `ibdiagnet` summary for the fabric.
- **RoCEv2.** Per-priority PFC RX / TX PAUSE counters (NIC and
  switch) at the beginning and end of the sweep. ECN mark counts
  and per-QP CNP counters if available. Chapter 4's "what to
  measure" section lists the specific counters.
- **EFA.** SRD retransmission counters if the CloudWatch metric or
  `efa-info` exposes them.

State whether counters are quiet at the start, whether they grew
during the sweep, and whether the growth rate matches what "healthy
under load" looks like on your baseline. On a well-configured RoCEv2
fabric you expect some ECN marking under load but no PFC pauses;
if PFC pauses fire, note it.

### 6. End-to-end step-time A/B

The number that actually matters for the design review. Run the same
short training script (a scripted 100–200 steps of a small dense
transformer under DDP or FSDP is fine) against each fabric with all
non-fabric variables held constant (same NCCL version, same batch,
same seed, same optimizer state).

Publish:

- `predicted_step_time` from the α + β model of section 1.
- `observed_step_time` on fabric 1.
- `observed_step_time` on fabric 2.
- The delta between the two.
- The delta between prediction and observation on each fabric.

If the model in section 1 predicts fabric 1 wins by 6% and you
observe 20%, something else is dominating and the report should say
so (a metadata-heavy loader, a checkpoint burst, a straggler rank).

### 7. Recommendation and forcing argument

The final section is a short (300–600 words) written recommendation:

- **Which fabric wins for this workload, at this world size, on this
  cluster generation.** State the choice.
- **The forcing argument.** Which measurement made the case? Section
  3 shows the verbs layer is comparable; section 4 shows NCCL sees
  the same at large payloads; section 6 shows end-to-end delta of X%
  in fabric Y's favor. Walk the reader through the chain.
- **The operational cost.** Chapter 4 makes the point that RoCEv2 is
  operationally harder because of PFC + ECN + DCQCN. State what your
  team's operational readiness is for the recommended choice, and
  what the section-5 counters imply about ongoing tuning cost.
- **The alternative you rejected.** Explicit statement: "we also
  considered X and rejected it because Y." This is what turns a
  benchmark into a decision.
- **The measurement protocol going forward.** Which of the drills
  above becomes the recurring baseline, and how often.

## Starter guidance

- Do not confuse "fabric is faster" with "collective is faster".
  Sections 3 → 4 → 6 walk the abstraction ladder: verbs, NCCL,
  application. A fabric can win at verbs and lose end-to-end because
  NCCL is not exploiting it, or vice versa.
- On EFA / SRD specifically: it does not use PFC, so its congestion
  story is different from vanilla RoCEv2. The `aws-ofi-nccl` plugin
  is what makes NCCL work on it; if `NCCL_DEBUG_SUBSYS=NET` does not
  show that plugin loading, your NCCL is falling back to a slower
  transport.
- On single-cloud-provider comparisons (Azure IB vs. Azure Ethernet,
  say), treat them as two fabrics even if the switch fabric is the
  same underneath — the driver stack and transport are what you
  actually measure.
- When you must extrapolate from vendor docs (because you cannot
  test one of the fabrics), cite the doc URL and the specific figure
  you are using. Do not paraphrase.
- Keep the report focused on the *chosen* workload. A general
  "RoCEv2 vs. IB shootout" report is a worse deliverable than a
  narrow "RoCEv2 vs. IB for workload B" report.

## Acceptance criteria

- The workload is characterized in numbers (section 1) and every
  subsequent section refers back to those numbers.
- Both fabrics have a fingerprint (section 2), even if one is
  reference-only. Reference-only sections are labelled.
- Section 3 has `perftest` results with `%_of_nominal` computed.
- Section 4 has NCCL sweep results at the workload payload.
- Section 5 has counter data with an interpretation. On RoCEv2, the
  PFC / ECN state is explicitly named; on IB, error counters and
  fabric health are stated.
- Section 6 has a predicted-vs-observed comparison on both fabrics.
- Section 7 has an explicit fabric choice, a forcing argument tied
  to the numbers, an operational-cost paragraph, and a named
  rejected alternative.
- Every number is traceable to a command or a cited doc. No invented
  benchmarks.

## Stretch goals

- Run the same measurement protocol at a smaller world size (say 16
  GPUs) and at a larger one (say 128 GPUs, if you can). Does the
  fabric recommendation change with scale? Chapter 4 predicts SHARP
  and DCQCN both matter more at higher N; the numbers should confirm.
- If your section-6 A/B shows fabric X losing by 8% end-to-end,
  spend 30 minutes with the losing fabric's NCCL tuning (exercise 2
  drills) and see how much of the gap you can close. Chapter 5 makes
  the point that fabric choice and NCCL tuning compose; this shows
  by how much.
- Extend section 5 to a *stress* run: run three tenants' worth of
  traffic through the fabric simultaneously (multiple `nccl-tests`
  jobs at once) and re-take the counters. Congestion behavior under
  contention is the single best proxy for how the fabric will feel
  in production; chapter 8's failure-mode 3 (PFC storm) is what you
  are stress-testing against.
- Turn the report into a template your team can rerun each time a
  fabric is refreshed (firmware, driver, NCCL version). The template
  is the artifact; individual reports are instances.
