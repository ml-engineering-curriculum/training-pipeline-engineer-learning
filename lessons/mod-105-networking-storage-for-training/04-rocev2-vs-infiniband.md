# RoCEv2 vs. InfiniBand: When Each Wins and What to Measure

InfiniBand is the reference. RoCEv2 (RDMA over Converged Ethernet
version 2) is the alternative, and it is the fabric of choice for the
majority of hyperscaler training deployments — including much of
AWS EC2 P5/P5e (via EFA / SRD, which is a related but distinct
transport), Azure ND-series, and GCP A3. Any training platform engineer
who cannot argue when to pick RoCEv2 over IB, and what to measure to
prove it, will lose that argument in a design review.

## What RoCEv2 actually is

**RoCEv2** is the InfiniBand transport running over UDP/IP over
Ethernet. The IBTA-published spec (Annex A17 of the InfiniBand
Architecture Specification) defines it; the wire format is:

- Ethernet header (with a specific EtherType for RoCEv2)
- IPv4 or IPv6 header
- UDP header (with the well-known RoCEv2 destination port 4791)
- InfiniBand Base Transport Header (BTH)
- IB payload (RDMA WRITE, READ, SEND, etc.)

The important consequence: **the verbs API is identical**. `ibv_post_send`
of an `RDMA_WRITE` works the same way, from the application's
perspective, on RoCEv2 as on IB. NCCL's IB transport code is largely
shared between the two — same QPs, same completion queues, same
verb types. What differs is the underlying transport reliability
mechanics.

RoCEv1 was RDMA directly on Ethernet without an IP layer; it never
achieved wide adoption because it could not route across subnets.
RoCEv2 is the version deployed everywhere today. When someone says
"RoCE" without a version, they mean RoCEv2.

## The lossless-Ethernet contract

InfiniBand is lossless by construction: link-layer credit-based flow
control. Ethernet is *not* lossless by construction — it drops
packets on congestion. RoCEv2 as originally specified assumed a
lossless underlay, so getting Ethernet to behave losslessly is where
most of the operational complexity lives.

The three mechanisms:

- **PFC — Priority-based Flow Control (IEEE 802.1Qbb).** A per-priority
  pause frame. When a switch's egress queue for priority `p` fills
  past a threshold, it sends a PAUSE frame upstream telling the
  upstream port to stop sending traffic on priority `p` until told
  otherwise. Configured correctly, PFC prevents drops on the RoCEv2
  priority.
- **ECN — Explicit Congestion Notification (RFC 3168).** A two-bit
  field in the IP header (or IPv6 flow label) that switches set when
  their queues start filling. The receiver echoes ECN back to the
  sender via a CNP (Congestion Notification Packet), which triggers
  rate reduction.
- **DCQCN — Data Center QCN.** The congestion-control algorithm on
  the RoCEv2 NIC. When a CNP arrives, the NIC reduces its transmit
  rate; when no CNPs arrive for a period, it slowly ramps back up.
  DCQCN is what actually keeps a RoCEv2 fabric alive under load;
  PFC without DCQCN causes "PFC storms" — cascading pause frames that
  freeze the fabric.

The single most common RoCEv2 misconfiguration is a mismatch between
PFC's priority mapping and the actual traffic class RoCEv2 packets
carry. Get the DSCP-to-priority mapping wrong and pauses land on the
wrong queue; the fabric silently drops RoCEv2 packets. Chapter 8
discusses this failure mode.

Read the papers before you tune anything:

- **DCQCN paper.** Zhu, Y., et al. (2015). "Congestion Control for
  Large-Scale RDMA Deployments." *SIGCOMM '15.*
- **HPCC paper.** Li, Y., et al. (2019). "HPCC: High Precision
  Congestion Control." *SIGCOMM '19.* Successor congestion-control
  algorithm designed for RDMA at data-center scale.
- **RDMA over commodity Ethernet paper.** Guo, C., et al. (2016).
  "RDMA over Commodity Ethernet at Scale." *SIGCOMM '16.* Microsoft's
  production-scale write-up of running RoCEv2 in Azure.

## Non-standard RDMA over Ethernet: EFA and SRD

Hyperscalers do not always use RoCEv2. AWS's Elastic Fabric Adapter
(EFA) uses a proprietary transport called Scalable Reliable Datagram
(SRD) that is *not* wire-compatible with RoCEv2 but exposes a
libfabric API that NCCL's `aws-ofi-nccl` plugin talks to. The important
property of SRD is that it does not require a lossless underlay; it
handles reordering and retransmission at the transport layer, so
SRD does not need PFC.

Similarly, some GCP deployments use gRPC-based collectives on IRDMA
or a variant of RoCEv2 with a hyperscaler-tuned congestion algorithm.
Read the vendor's specific documentation before assuming your
"RoCEv2" cluster follows the vanilla IBTA specification.

Where possible in the exercises we call out both "standard RoCEv2"
and "EFA + SRD" as separate cases.

## When RoCEv2 wins over InfiniBand

The honest tradeoff:

| Criterion | InfiniBand | RoCEv2 |
|-----------|-----------|--------|
| Line-rate collective bandwidth on well-tuned fabric | Slight edge (lower per-hop latency, in-network reduction with SHARP) | Comparable at NDR / 400 Gb/s on modern switches |
| Latency floor per hop | Lower (µs range) | Slightly higher (extra headers, ECN/PFC state machines) |
| Complexity to bring up | Lower — the SM does most of it | Higher — PFC, ECN, DCQCN all need tuning |
| Operational familiarity | HPC-team-native | Network-team-native (looks like a data-center Ethernet fabric) |
| Vendor concentration | High (NVIDIA / Mellanox effectively single vendor) | Broader (Broadcom, Arista, Cisco, Juniper, NVIDIA) |
| Integration with existing DC network | Separate fabric | Fits into existing L2/L3 |
| In-network reduction | SHARP | Not standardized; proprietary or absent |
| Multi-tenancy / VXLAN / overlays | Awkward | Native |
| Public-cloud availability | Bare-metal only | Available on hyperscaler VMs (with EFA / equivalents) |

RoCEv2 wins when:

- You are running on a hyperscaler cloud (AWS / Azure / GCP) — you
  do not get a choice; the hyperscaler exposes RoCEv2 or an
  RoCEv2-adjacent RDMA over Ethernet.
- You are integrating into an existing Ethernet-only data center and
  the incremental cost of a second fabric is prohibitive.
- Your team has strong network engineers who are comfortable operating
  lossless Ethernet.
- You need multi-tenancy features (VXLAN, EVPN) that Ethernet-native
  fabrics handle naturally.

InfiniBand wins when:

- You are building a dedicated on-prem SuperPOD and validated
  reference architectures matter more than integration with the
  existing DC network.
- You want SHARP-based in-network reduction — RoCEv2 does not have a
  standardized equivalent.
- Your team has HPC operational experience and prefers the SM /
  fabric-manager model over configuring PFC per switch.

Do not choose based on nominal spec-sheet bandwidth alone. At NDR
per port both fabrics deliver essentially the same peak collective
bandwidth; the interesting differences are second-order (SHARP
availability, PFC operational complexity, integration cost).

## What to measure to prove the choice

A design-review claim like "RoCEv2 costs us 8% on all-reduce" is only
defensible if backed by numbers you can reproduce. The measurement
protocol:

### 1. `ib_write_bw` and `ib_send_bw` from `perftest`

The `perftest` suite (https://github.com/linux-rdma/perftest) has
`ib_write_bw`, `ib_read_bw`, `ib_send_bw`, and `ib_send_lat`. These
run the raw verbs and report per-QP bandwidth and latency without
NCCL's algorithmic overhead. Run them:

- Between two nodes on the same rail (same-rail cross-node bandwidth).
- Between two nodes on different rails (cross-rail — this is where
  PXN vs. non-PXN shows up on IB).
- With `RDMA_WRITE` (matches NCCL's data-plane operation) and with
  `SEND` (matches control-plane).

The `ib_write_bw` command should reach within a few percent of nominal
per-port bandwidth on a healthy fabric. If it does not, no NCCL tuning
will fix it.

### 2. `nccl-tests` sweeps

`all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf`, and
`alltoall_perf` at message sizes from ~4 KiB to ~1 GiB. Chapter 5
discusses how to read the output; for the IB-vs-RoCEv2 comparison the
number you care about is:

- **Peak `busbw` at large message sizes** (say ≥256 MiB). This is the
  bandwidth term of the α+β model, and it is a direct proxy for
  fabric quality.
- **Latency at small message sizes** (say ≤4 KiB). This is the α term
  and is where IB tends to edge RoCEv2 out.

### 3. End-to-end training step time

The number that actually matters. Configure DDP or FSDP on a fixed
model and dataset and measure step time on both fabrics with all
other variables held constant. Publish `predicted_step_time` from the
chapter-5-of-mod-101 cost model alongside `observed_step_time` on
both fabrics. The delta between the two is your real comparison.

### 4. Congestion-control health

For RoCEv2 specifically:

- **PFC counters.** Every switch and every NIC exposes per-priority
  RX/TX PAUSE counters. Non-zero counts at steady-state on the
  RoCEv2 priority mean the fabric is congested and DCQCN is working
  overtime.
- **ECN marks.** Count of ECN-marked packets received per unit time.
  Steady-state marking is expected under load; a runaway climb
  usually means a hot spine port.
- **DCQCN metrics.** `mlxreg` and the vendor's fabric-management tools
  expose per-QP CNP counts and rate-limit state. Chapter 8 discusses
  what "normal" looks like here.

If PFC counters are climbing without corresponding ECN marks, PFC is
firing too aggressively and DCQCN is not slowing traffic down. That
is a fabric-config bug, not a training-code bug.

## The composite recommendation for a design review

If you are producing an RFC for a new training pod, the honest
default recommendation:

- **Dedicated on-prem pod, ≥256 GPUs, HPC-native team → InfiniBand.**
  Follow the DGX SuperPOD Reference Architecture. Reach for RoCEv2
  only if there is a specific integration constraint that forces it.
- **Public-cloud training → whatever the hyperscaler offers.** EFA
  on AWS, InfiniBand on Azure ND-series (as of most current
  generations, but verify against the current SKU), IRDMA / RoCEv2
  on GCP. You do not get a real choice, but you *do* get to tune
  what you have (chapter 5).
- **Existing Ethernet-only DC, small pod (< 128 GPUs) → RoCEv2.** The
  incremental cost of a second fabric is not worth the SHARP win at
  that scale, and your team likely has more Ethernet expertise than
  IB expertise.

Whatever the choice, make the measurement protocol above part of the
RFC. A design that ships without an `ib_write_bw` and `nccl-tests`
baseline is a design that will surprise you.

## Summary

- RoCEv2 is InfiniBand's transport running on UDP/IP/Ethernet. The
  verbs API is identical; NCCL's plumbing is largely shared.
- Ethernet is not natively lossless. RoCEv2 requires PFC + ECN + DCQCN
  to behave; misconfiguration is the #1 source of RoCEv2 pain.
- InfiniBand's advantages: lower per-hop latency, SHARP, simpler
  operational model. RoCEv2's advantages: integration with the DC
  network, hyperscaler availability, broader vendor set.
- Choose based on the deployment context, not nominal bandwidth. At
  NDR both fabrics deliver comparable peak collective bandwidth.
- The measurement protocol (perftest, nccl-tests, end-to-end step
  time, congestion-control counters) is what turns a preference into
  a defensible design choice. Exercise 3 walks it end-to-end.
