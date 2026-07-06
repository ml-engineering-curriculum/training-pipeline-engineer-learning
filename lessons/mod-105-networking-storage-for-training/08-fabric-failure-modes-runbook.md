# Diagnosing Fabric Failure Modes: A Runbook

The previous seven chapters describe the fabric working correctly.
This chapter is about what it looks like when it does not, and how to
find the cause fast enough that you do not burn through the
GPU-hours budgeted for the run. The four failure modes below cover
the vast majority of training-time fabric incidents in production; if
you can walk through each one on-call without looking it up, you have
the module.

Every runbook entry follows the same shape: **symptom → tools to
inspect → likely causes → fix → post-mortem note**. That shape is the
shape of the on-call ticket the incident review process expects.

Two references you should have open on any fabric incident:

- **NCCL troubleshooting guide.**
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html
- Your fabric vendor's error-counter reference: the Mellanox
  `mlxreg` / OFED counter list for InfiniBand and RoCEv2, or the
  hyperscaler's networking metrics dashboard for cloud RDMA.

## Before any incident: the baseline

Every failure-mode diagnosis compares "what the fabric is doing now"
against "what the fabric does when it is healthy". So the very first
thing every training platform owns is a *baseline*:

- `nccl-tests` outputs from chapter 5's bring-up checklist.
- `ib_write_bw` per-node-pair numbers from chapter 4.
- Per-priority PFC counters at steady-state (RoCEv2).
- SM sweep interval and last resweep time (IB).
- Per-client parallel FS read/write bandwidth (chapter 7).

If you do not have baselines, every incident is a green-field
investigation. If you do, most incidents are "compare current to
baseline, find the delta".

## Failure mode 1: NCCL timeout

### Symptom

The job hangs. Eventually you see something like:

```
Watchdog caught collective operation timeout: WorkNCCL(SeqNum=..., OpType=ALLREDUCE)
  ran for X milliseconds before timing out
```

or the job exits with an NCCL `unhandled cuda error` after the
watchdog fires. `NCCL_ASYNC_ERROR_HANDLING=1` (or its modern
equivalent) is what surfaces this cleanly.

### Tools

- **NCCL debug log.** Set `NCCL_DEBUG=INFO` and
  `NCCL_DEBUG_FILE=/path/to/nccl-rank%h-%p.log` on the run. If the
  hang has already happened, re-launch a minimal repro with these on.
- **Per-rank stack traces.** For a live-hung job, `py-spy dump --pid
  <trainer PID>` on every rank. The rank that is *not* waiting on
  the collective is the odd one out.
- **`torchrun` rendezvous logs.** Confirm every rank actually joined
  the group. Missing ranks show as "waiting for N joiners".

### Likely causes

- **A rank died.** The most common. One rank OOMed, hit an
  `AssertionError`, or was SIGTERM'd by the scheduler; the rest are
  waiting forever on the collective. The stack trace on N-1 ranks
  is inside NCCL; the trace on the dead rank is your bug.
- **A rank fell behind by more than the timeout.** Uneven data,
  irregular batch composition, or a slow rank (thermal throttle,
  bad ECC on an HBM stack) makes one rank take much longer on
  a step. Look at per-step time histograms across ranks.
- **A fabric-level stall.** A NIC or switch dropped the QP; NCCL
  waited for a completion that never came. Check per-NIC and
  per-QP counters — `ethtool -S`, `mlnx_perf`, and vendor tooling.
- **The gang was scheduled across an SU boundary and PXN
  misrouting compounds latency past the timeout.** See failure
  mode 4.

### Fix

- If a rank died: fix the rank, not NCCL. Raise the timeout as a
  band-aid (`NCCL_TIMEOUT` env or the PyTorch `timeout` on
  `init_process_group`) *only* if the underlying cause is expected
  and rare (e.g., checkpoint saves that occasionally take longer).
- If a rank fell behind: fix the workload (data shuffling,
  gradient accumulation alignment). Sometimes the answer is
  synchronizing at every step (`torch.cuda.synchronize()` and a
  `torch.distributed.barrier()`) so per-rank drift is bounded.
- If it was fabric: escalate to the fabric team with the specific
  QP / NIC identifiers from NCCL's log and the vendor counters. Do
  not restart the job until the root cause is scoped.

### Post-mortem note

The single most common wrong post-mortem note on NCCL-timeout
incidents is "NCCL was slow". NCCL is a symptom; the collective it
was inside is waiting on a rank that stopped moving. Name the rank
that stopped and what it was doing when it stopped.

## Failure mode 2: silent NIC drop

### Symptom

The job does not hang. Step time gets worse — say from 500 ms to
1200 ms — but stays roughly stable at the new higher level. `nccl-tests`
run right after shows `busbw` down by 30–50%. There is no error in
the training logs; the only signal is throughput.

### Tools

- **Vendor NIC counters.** On Mellanox / NVIDIA HCAs, `ethtool -S
  <iface>` and `mlnx_perf` expose per-port RX/TX counters. Look for:
  - `port_xmit_discards`, `port_rcv_errors` (IB / RoCEv2).
  - `symbol_errors` — physical-layer errors.
  - `link_downed` — cumulative link flap count.
  - PFC counters (RoCEv2).
- **Switch counters.** The switches at the leaf and spine expose
  the same counters via their management interface. Fabric ops
  should have a dashboard.
- **`ibdiagnet` (IB).** Diagnoses the whole fabric, calls out bad
  cables and unhealthy links.
- **`NCCL_DEBUG=INFO` on a `nccl-tests` run right now.** Which HCA is
  NCCL using per rank? Compare with the healthy baseline.

### Likely causes

- **Failed HCA on one node.** The card is up on the wire but
  silently dropping packets, so the transport retransmits. NCCL
  keeps running but slowly.
- **Bad cable or bad transceiver.** Same symptom, one hop into the
  fabric.
- **Firmware regression.** A recent firmware update flipped a
  default. Compare firmware version to the last-known-good.
- **Congestion in the storage fabric bleeding onto the compute
  fabric.** Rare, but happens on shared-cable deployments. Chapter
  2 covered fabric isolation; if it is not enforced, this happens.

### Fix

- If a specific HCA is dropping: drain the node, replace the card,
  bring it back in. This is a hardware ticket, not a code ticket.
- If a specific cable / transceiver: same. Fabric ops replaces it.
- If firmware: pin the version and roll back if the regression is
  fabric-wide.
- If congestion is bleeding: enforce the fabric isolation from
  chapter 2. Move storage traffic off the compute rails.

### Post-mortem note

Silent NIC drops are why you keep baselines. The delta of
`busbw` from baseline is what makes this diagnosable in minutes
instead of days. Every training-cluster runbook should page on a
sustained regression from the nccl-tests baseline, not just on hard
failures.

## Failure mode 3: congestion tree (PFC storm on RoCEv2)

### Symptom

RoCEv2-specific. Step time collapses under load, then partially
recovers, then collapses again. Multiple jobs' step times worsen
simultaneously across a shared switch's traffic. Per-priority PFC
counters explode across the switch fabric — pause frames propagating
back toward the source.

Sometimes called a **congestion tree** because the pauses ripple
back through the fabric like the branches of a tree.

### Tools

- **Switch PFC counters** on the RoCEv2 priority. Baseline is near
  zero; incident-state is millions of pause frames per second.
- **ECN counters.** Cross-check that ECN is marking and DCQCN is
  reacting. Runaway PFC with no matching ECN indicates DCQCN is not
  keeping up — often a config mismatch between switches and NICs.
- **`show_gids` and PFC config on the NICs.** Confirm the DSCP → PFC
  priority mapping matches on both switches and NICs.
- **Per-QP CNP counters** on the NIC. Confirms whether the transport
  layer is even receiving congestion feedback.

### Likely causes

- **DSCP-to-priority mismatch.** Switches and NICs disagree on which
  DSCP value maps to the RoCEv2 priority. PFC pauses land on the
  wrong queue.
- **PFC without matching ECN / DCQCN.** PFC is on, DCQCN is off (or
  misconfigured on the NIC). PFC alone will oscillate — pause,
  release, pause, release — because there is no rate-based feedback
  slowing the sender down.
- **Buffer overrun on a switch under multi-tenant traffic.** Another
  workload's burst is pushing your priority's queues past the PFC
  threshold. Solution: better traffic classification and dedicated
  buffer allocations on the switch.
- **A "victim flow" pattern.** A flow that itself is not causing the
  congestion but gets throttled anyway because it shares a queue
  with a congesting flow. The victim's step time is what pages you;
  the congesting flow is what needs the fix.

### Fix

- Confirm the DSCP mapping is coherent across the whole fabric.
- Confirm DCQCN is enabled on the NICs *and* ECN marking is on at
  the switches with the right threshold.
- If a specific tenant is congesting: talk to them, or move them to
  a different priority.
- On chronic multi-tenant congestion: revisit the fabric buffer
  allocation and consider isolating queues per tenant.

### Post-mortem note

InfiniBand does not have this failure mode in the same shape because
its credit-based link-layer flow control avoids the pause-frame
mechanism. When teams say "RoCEv2 is harder to operate than IB",
this is what they mean. If your team does not have deep networking
expertise, budget for that on RoCEv2.

## Failure mode 4: PXN misconfiguration

### Symptom

Cross-node collectives are much slower than they should be. Small
`nccl-tests` message sizes look normal; large message sizes at high
N look badly worse than the baseline. Or: a specific rank pair's
collective is disproportionately slow. Chapter 5 introduced PXN and
what it should look like on a healthy fabric.

### Tools

- **`NCCL_DEBUG=INFO` output.** Look for the PXN state per rank
  pair. Healthy: cross-rail traffic uses an intermediate rank on the
  source node (an NVLink hop) before going onto the fabric. Broken:
  cross-rail traffic is being routed directly through a leaf switch
  that also carries same-rail traffic.
- **`NCCL_DEBUG_SUBSYS=NET`.** Detailed transport logging.
- **`nccl-tests` comparison with `NCCL_PXN_DISABLE=1`.** If disabling
  PXN improves things, PXN is misconfigured; if disabling PXN makes
  things worse (the expected case on a healthy fabric), PXN is fine
  and something else is wrong.

### Likely causes

- **NCCL cannot see the topology.** `NCCL_TOPO_FILE` is missing or
  wrong. NCCL falls back to a topology it thinks is correct and
  picks wrong routes.
- **HCA-to-GPU affinity is wrong.** The kernel is exposing
  affinities that do not match the physical wiring; NCCL picks the
  wrong HCA for a given GPU.
- **Storage HCA is included as a compute HCA.** NCCL is using the
  storage NIC for a compute collective; PXN sends cross-rail
  traffic through it and creates cross-fabric interference.
  Fix by setting `NCCL_IB_HCA` to only the compute HCAs.
- **QP-count limits at very high N.** At many-thousand-GPU jobs the
  per-HCA QP count runs out; NCCL falls back to a routing that does
  not respect rails. Increase `NCCL_IB_QPS_PER_CONNECTION` sparingly
  or restructure subcommunicators.

### Fix

- Regenerate `NCCL_TOPO_FILE` from `NCCL_TOPO_DUMP_FILE=/tmp/topo.xml`
  and confirm every HCA and every GPU is present with the correct
  NUMA / PCIe affinity.
- Restrict `NCCL_IB_HCA` to compute HCAs only.
- Confirm PXN is enabled (`NCCL_PXN_DISABLE=0`, the default) and that
  `NCCL_CROSS_NIC` allows cross-NIC operation on your topology.
- Re-baseline `nccl-tests` after the fix.

### Post-mortem note

PXN misconfigurations tend to show up after cluster changes — a new
node type joining the pool, a firmware update, a driver bump. If
"the fabric changed and now the runs are slower" is the pattern,
this is one of the first two things to check (silent NIC drop is
the other).

## Failure mode 5 (bonus): scheduler placed the gang across SUs

### Symptom

A job that used to run at rate X is now running at rate 0.6X. Nothing
in the code, model, or NCCL config changed. The scheduler placed the
gang across three SUs instead of one.

### Tools

- The scheduler's placement output. On SLURM, `squeue -o "%N"` and
  the topology plugin's logs. On Kubernetes, the Kueue
  Topology-Aware Scheduling annotations and the actual node labels
  the pods landed on.
- `NCCL_DEBUG=INFO`. NCCL will show cross-SU rings.
- The DGX SuperPOD reference architecture SU map.

### Fix

- Fix the placement constraint. Add topology-aware admission
  requirements (chapter 3 of mod-104 covers this).
- If the cluster is fragmented, defragment (or requeue with a
  reservation).

### Post-mortem note

This one is a mod-104 issue as much as a mod-105 issue. The runbook
entry lives here because the symptom looks like a fabric
degradation even though the fabric is fine.

## The general shape of a fabric on-call

When paged on any fabric incident, the first three moves in order:

1. **Snapshot.** Grab the `NCCL_DEBUG=INFO` log from the affected
   run, the per-NIC counters, and the switch counters. Do not
   restart anything yet — you lose diagnostic state on a restart.
2. **Compare to baseline.** Which metric is off, and by how much,
   relative to the last-known-good baseline (`nccl-tests`, PFC
   counters, `ib_write_bw`)?
3. **Localize.** Is it one host, one rack, one SU, or fabric-wide?
   That localization narrows the failure mode from the list above
   to one or two candidates.

Then and only then do you start applying fixes. The temptation to
restart the job first and diagnose second is real and expensive; the
job's state contains most of the evidence.

## Summary

- Four canonical training-time fabric failure modes: NCCL timeout,
  silent NIC drop, congestion tree, PXN misconfiguration. A fifth
  (scheduler placed across SUs) looks like a fabric issue and lives
  here for that reason.
- Every diagnosis compares "current" to a saved baseline. Own the
  baselines from chapter 5 and chapter 7's bring-up checklists.
- NCCL timeouts are almost always about a rank that stopped moving,
  not about NCCL being slow. Diagnose the rank, not the collective.
- Silent NIC drops are what per-NIC error counters exist for. Page
  on regressions from the nccl-tests baseline, not just on hard
  errors.
- Congestion trees are a RoCEv2-specific failure mode of PFC / ECN /
  DCQCN interaction. Get the DSCP mapping right or you will spend
  quarters here.
- PXN misconfiguration shows up after cluster changes. Regenerate
  `NCCL_TOPO_FILE` and re-baseline before chasing framework code.
- Do not restart before you snapshot. The job state is the evidence.
