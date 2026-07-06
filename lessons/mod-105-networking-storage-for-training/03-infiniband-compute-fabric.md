# The InfiniBand Compute Fabric: Rails, Fat-Tree, and Adaptive Routing

Chapter 2 stopped at the SU boundary and said "the inter-node compute
fabric is a rail-optimized fat-tree on InfiniBand". This chapter opens
that black box. If you cannot explain, at the level of hops and per-hop
latency, how a byte gets from a GPU on host A to a GPU on host B on a
DGX SuperPOD, you cannot debug the "why is our all-reduce slow"
incidents that show up on-call at scale.

The primary reference for everything in this chapter is the InfiniBand
Architecture Specification, maintained by the InfiniBand Trade
Association (IBTA, https://www.infinibandta.org/). Every knob NCCL or
`ibv_*` exposes is defined by that spec. Read at minimum the Volume 1
overview (transport, addressing, congestion control) before working
the exercises.

## Why InfiniBand for training

Ethernet plus RoCEv2 is the alternative and chapter 4 is where we
compare them fairly. First we cover InfiniBand because the vast
majority of NVIDIA SuperPOD reference architectures and academic HPC
clusters ship it as the default, and because most of the RDMA
vocabulary — queue pairs, verbs, MTU, HCAs — was defined for
InfiniBand and inherited by RoCEv2.

The three properties that make InfiniBand the reference choice for
training fabrics:

- **RDMA-native.** The transport is designed around one-sided reads and
  writes into remote memory, not around the TCP/IP socket abstraction.
  Zero-copy from GPU HBM to remote HBM (via GPUDirect RDMA) is a
  first-class operation, not an add-on.
- **Lossless by construction.** InfiniBand uses credit-based flow
  control at the link layer: a sender only transmits when it holds a
  credit from the receiver. Packets are essentially never dropped in
  the fabric, which lets the transport-layer congestion machinery
  ignore drop-detection entirely.
- **Centralized subnet management.** Every InfiniBand fabric has
  exactly one active *subnet manager* (SM) that discovers the topology,
  assigns Local Identifiers (LIDs), and computes routing tables. That
  gives you a single source of truth for "what does the fabric look
  like right now".

The trade-off is cost, vendor concentration (NVIDIA / Mellanox), and
integration with commodity Ethernet networking. Chapter 4 opens that
comparison; here we take InfiniBand as given and explain what a
training platform engineer has to know about it.

## Port speeds: EDR / HDR / NDR / XDR

InfiniBand port speeds have doubled roughly every generation and are
named alphabetically. The generations you meet in training:

| Generation | Signaling per lane | 4× lanes per port | Approximate byte-rate per port |
|------------|--------------------|-------------------|-------------------------------|
| EDR | 25 Gb/s | 100 Gb/s | ~12.5 GB/s |
| HDR | 50 Gb/s | 200 Gb/s | ~25 GB/s |
| NDR | 100 Gb/s | 400 Gb/s | ~50 GB/s |
| XDR | 200 Gb/s | 800 Gb/s | ~100 GB/s |

Cross-check the exact rates against the current IBTA roadmap; the
"4×" is the standard aggregation and matches the port on every DGX
HCA. When your monitoring says "the fabric is at 92% of NDR", that
means ~46 GB/s per port on the wire.

Two things are worth flagging up front:

1. **The published rate is per direction.** A ConnectX-7 NDR port does
   ~50 GB/s each way — not 100 GB/s total. Full-duplex is real, but
   most collective algorithms only use one direction at a time for a
   given phase.
2. **The effective rate is lower than the signaling rate** because of
   64/66b encoding overhead and per-packet headers. NCCL's `nccl-tests`
   `busbw` at large messages is the number you should compare against
   "line rate" — chapter 1 introduced that vocabulary.

## Queue pairs, verbs, and the RDMA transport

The InfiniBand programming model is exposed as **verbs**
(`ibv_post_send`, `ibv_post_recv`, `ibv_poll_cq`, etc.). NCCL uses
these directly on the compute fabric; you rarely write verbs code
yourself, but you have to be able to read `NCCL_DEBUG=INFO` output
that names them.

The pieces you should be able to name:

- **HCA (Host Channel Adapter).** The InfiniBand NIC. On DGX H100 you
  have 8 compute-fabric HCAs, one per GPU.
- **Port.** A physical port on the HCA. Each DGX compute HCA is one
  port on the compute fabric.
- **GID (Global Identifier) and LID (Local Identifier).** Two levels
  of address. LID is a subnet-scoped 16-bit id assigned by the SM;
  GID is a 128-bit address with a global scope. Training-cluster
  fabrics are usually a single subnet, so LIDs are what you see in
  routing tables.
- **Queue Pair (QP).** A pair of send/receive queues on one HCA that
  is bound to a QP on another HCA. A connection, essentially. NCCL
  creates one QP per rank pair that will exchange traffic.
- **Completion Queue (CQ).** Where the HCA posts completion events
  after a work request finishes.
- **Verb types.** `SEND` / `RECV` (two-sided messaging), `RDMA_WRITE`
  (one-sided write into remote memory), `RDMA_READ` (one-sided read).
  NCCL uses `RDMA_WRITE` for most of its data movement.
- **Transport service.** RC (Reliable Connection — ordered, delivered,
  what most collectives use), UC (Unreliable Connection), UD
  (Unreliable Datagram — used for the SM and out-of-band chatter).
  Training collectives are RC.

Two useful facts to keep in mind:

- **Every QP is a resource on the HCA.** Very large jobs create very
  many QPs, and there is a per-HCA cap. On multi-thousand-GPU jobs you
  configure NCCL to use PXN or shared receive queues (SRQ) to keep the
  QP count sub-quadratic. Chapter 5 discusses PXN.
- **The MTU matters.** Bigger InfiniBand MTU (up to 4096 bytes on most
  hardware) means fewer packets and headers per message, which matters
  at the 50–100 GB/s NDR line rate.

## The subnet manager

An InfiniBand subnet has exactly one active subnet manager. Its job:

- **Discover topology.** Walk the subnet, learn HCAs, switches, and
  their connections.
- **Assign LIDs.** Each port gets a subnet-local 16-bit identifier the
  fabric routes on.
- **Compute routing tables.** Fill in the forwarding tables on every
  switch. On a fat-tree, this is where UPDN, DFSSSP, or similar
  algorithms pick "which output port for which destination LID".
- **Detect topology changes.** A link flap or a hot-swap triggers a
  resweep and a re-route.

The two SM implementations you meet in production are OpenSM (from the
Rdma-Core project) and NVIDIA's UFM-integrated SM (Unified Fabric
Manager). Consult the OpenSM man page and the UFM documentation on
NVIDIA's site — every scheduled maintenance and every "the fabric went
noisy at 03:00" incident touches these.

Two operational facts that catch teams:

- **The SM restart is a fabric-wide event.** A resweep during a
  training run can pause every collective for the duration of the
  sweep — seconds at scale. Schedule SM upgrades during maintenance
  windows.
- **Only one SM should be active.** A misconfigured redundant SM on a
  different partition assigns conflicting LIDs and hard-breaks routing.

## The fat-tree topology

The compute fabric on a SuperPOD is a *fat-tree* (also called a
Clos network). Fat-trees have three defining properties that make them
the reference choice for training clusters:

1. **Non-blocking bisection.** The aggregate uplink bandwidth from a
   leaf switch equals its downlink bandwidth. Any half of the ports can
   simultaneously reach the other half at full line rate.
2. **Predictable per-pair bandwidth.** In an idealized fat-tree with
   uniform load, every pair of endpoints can reach every other pair at
   line rate simultaneously.
3. **Recursive.** You can build a bigger fat-tree by stacking another
   spine tier on top.

The picture:

```
      spine     spine     spine     spine
        |         |         |         |
     +--+--+   +--+--+   +--+--+   +--+--+
     | leaf|   | leaf|   | leaf|   | leaf|
     +-----+   +-----+   +-----+   +-----+
       ||        ||        ||        ||
     (nodes)   (nodes)   (nodes)   (nodes)
```

Every leaf switch has half its ports facing downward (to hosts) and
half facing upward (to spine switches). Every spine switch has all
its ports facing downward (to leaves). The number of spine switches is
chosen so the bisection is preserved.

For very large pods you add a third tier (aggregation switches) above
the spines and get a **3-tier fat-tree**. The DGX SuperPOD reference
architecture uses a 3-tier fat-tree above ~1000 GPUs; the SU is 2-tier
inside itself.

## Rails: what "rail-optimized" actually buys you

The SuperPOD compute fabric is not one fat-tree; it is *eight parallel
fat-trees, one per rail*. That is what "rail-optimized" means: HCA
index `k` on every DGX only ever talks through the fabric of rail `k`.

The reason this matters: on a large all-reduce, a naive fat-tree
would send traffic from all 8 rails through the same switches, and
adjacent-rank collectives would collide inside the spine. Rail
separation eliminates that class of contention entirely — same-rank
traffic (e.g., a data-parallel all-reduce where every rank contributes
the same tensor) stays inside its own rail's fat-tree.

Two consequences for training platform engineers:

- **A single per-rank NIC failure only degrades one rail.** The rest
  of the fabric is unaffected. Chapter 8 covers how to detect this
  before NCCL times out.
- **A subcommunicator that shards along a "same-rank-across-nodes"
  axis lives entirely on one rail.** This is why HSDP shard groups
  work well on rail-optimized fabrics: the shard group is a rail-local
  collective; the replicate group is a same-node all-reduce that
  never touches the fabric. Chapter 5 goes into how NCCL and the
  application together produce this.

Cross-rail traffic (a rank talking to a different-rank on a different
node) has to hop through a leaf switch to change rails. On DGX H100
with PXN, NCCL can route this via an intermediate NIC on the *source*
node so the cross-rail hop happens on NVLink, not on IB. Chapter 5 is
where we make this concrete.

## Adaptive routing

Even with rail separation, real training workloads can create
same-rail hotspots — for example, all GPU-0s on all nodes talking to
GPU-0 on rank-0-node simultaneously. If the fabric routes those on a
single shortest path, you get congestion.

**Adaptive routing** is the switch-level mechanism that reroutes
packets across less-loaded uplinks. On modern NVIDIA / Mellanox
switches this is exposed as a per-VL (virtual lane) feature and is
controlled by the SM. See the NVIDIA Adaptive Routing documentation
and the OpenSM adaptive routing routing engine documentation for the
current knobs.

Two things to know:

- **Adaptive routing is on by default on modern SuperPOD builds.** If
  you turn it off, cross-node collectives on hot patterns get slower
  in a way you cannot fix in the framework.
- **Out-of-order delivery.** Adaptive routing can reorder packets. The
  transport layer (RC) reassembles in order, but very old firmware
  can have subtle interactions here. Verify against the current NVIDIA
  networking firmware release notes.

## SHARP: in-network reduction

NVIDIA's **Scalable Hierarchical Aggregation and Reduction Protocol
(SHARP)** offloads the reduction step of an all-reduce into the
switches themselves. Instead of every rank sending gradients around a
ring, ranks send their contribution *up* the tree and receive the
reduced result *down*.

The relevant properties:

- **Latency reduction.** A tree-of-switch reduction is `~log N` hops;
  a ring is `~N` hops.
- **Bandwidth relief.** The reduced result is one payload's worth of
  bytes; the ring pushes ~`2n · β` per rank.
- **Restrictions.** SHARP is supported by specific switch generations
  (Quantum-2 for NDR, and earlier Quantum-1 for HDR), needs the
  SHARP Aggregation Manager (an out-of-band service), and only some
  reduction operations and datatypes are supported. NCCL selects
  SHARP via `NCCL_ALGO=CollnetChain` / `NCCL_ALGO=CollnetDirect` when
  the topology permits. See the NVIDIA SHARP documentation and NCCL's
  algorithm selection docs.

SHARP is one of the single biggest fabric wins on very large jobs.
Confirm that it is enabled on your SuperPOD before optimizing NCCL
knobs — if the aggregation manager is down, NCCL silently falls back
to ring and you leave 20–40% of the theoretical all-reduce bandwidth
on the floor.

## The picture you should be able to draw

At the end of this chapter you should be able to draw, from memory:

```
  [DGX A]                                        [DGX B]
   GPU0 ─── HCA0 ── leaf0-rail0 ── spine0-rail0 ── leaf0-rail0 ── HCA0 ─── GPU0
   GPU1 ─── HCA1 ── leaf0-rail1 ── spine0-rail1 ── leaf0-rail1 ── HCA1 ─── GPU1
   ...
   GPU7 ─── HCA7 ── leaf0-rail7 ── spine0-rail7 ── leaf0-rail7 ── HCA7 ─── GPU7
```

With the annotations:
- Same-rank cross-node traffic stays on its rail's fat-tree.
- Cross-rank cross-node traffic either hops through a leaf switch or
  (with PXN) hops via NVLink on the source node first.
- The subnet manager knows every LID and every switch's forwarding
  table.
- Adaptive routing spreads hotspots across parallel spine paths.
- SHARP, when enabled, replaces the ring reduction with a switch tree.

Exercise 1 has you produce that diagram for a specific DGX SuperPOD
generation.

## Summary

- InfiniBand is the reference training-compute fabric: RDMA-native,
  lossless at the link layer, centrally managed by a subnet manager.
- Port speeds double per generation (EDR / HDR / NDR / XDR). NDR is the
  Hopper-generation default at ~50 GB/s per port per direction.
- The programming model is verbs on queue pairs. NCCL uses `RDMA_WRITE`
  over RC transport for training collectives.
- The topology is a rail-optimized fat-tree with 8 rails, one per
  HCA per DGX. Same-rank cross-node collectives stay inside one rail's
  fat-tree.
- Adaptive routing and SHARP are the two switch-level features that
  matter most for training at scale. Verify both are enabled and
  healthy before optimizing NCCL algorithm knobs.
- Chapter 4 does the same tour for RoCEv2 and tells you when the
  Ethernet answer is the right one.
