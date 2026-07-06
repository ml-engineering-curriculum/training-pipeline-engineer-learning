# DGX SuperPOD Reference Architecture: NVLink, NVSwitch, and the Scalable Unit

If you cannot draw the NVLink and NVSwitch layout of a DGX H100 box
from memory, and cannot stitch 32 of them together into an SU with
the InfiniBand fabric in the right places, you cannot reason about
step time on a real training cluster. This chapter fixes that mental
model against NVIDIA's published reference architecture. Every number
below should be cross-checked against the current NVIDIA DGX SuperPOD
Reference Architecture PDF for your generation of hardware — the pod
composition rules and switch counts move between Hopper (H100) and
Blackwell (B200) generations.

## Why a reference architecture at all

A **reference architecture** is a vendor-published blueprint that
specifies component counts, cabling, switch placement, and firmware
versions for a validated cluster shape. The point of reading it is
not to buy hardware; it is to internalize what the *canonical* layout
of a training pod looks like so that when your on-call incident says
"NCCL is running at 60% of nccl-tests", you can compare "what your
cluster is" to "what the reference architecture says it should be"
and locate the delta.

The three NVIDIA reference architectures worth reading, in decreasing
order of currency: DGX SuperPOD with DGX H100 / H200 systems, DGX
SuperPOD with DGX B200 systems, and (for historical context) DGX
SuperPOD with DGX A100 systems. The Hopper-generation document is the
one this chapter is grounded in. See the NVIDIA DGX SuperPOD
documentation index (https://docs.nvidia.com/dgx-superpod/) for the
current versions.

## The DGX H100 box (domain 1)

The DGX H100 is the compute chassis the rest of the pod is built out
of. The pieces that matter to a training platform engineer:

- **8 × H100 SXM GPUs.** 80 GB HBM3 per GPU (or 141 GB HBM3e per H200);
  peak matmul throughput per GPU is on the order of ~1000 TFLOPS BF16
  dense. Confirm against the current H100/H200 datasheet.
- **4 × NVSwitch3 chips.** Every H100 has 18 fourth-generation NVLink
  ports, each providing 25 GB/s per direction (so 50 GB/s bidirectional
  per port, 900 GB/s bidirectional aggregate per GPU). The 4 NVSwitches
  crossbar-connect all 8 GPUs so any pair has full non-blocking
  bandwidth.
- **8 × ConnectX-7 (or BlueField-3) 400 Gb/s NDR InfiniBand HCAs for
  the compute fabric.** One HCA per GPU, wired 1:1 to give each GPU a
  private NIC on the compute fabric. This is what makes GPUDirect RDMA
  possible without the traffic transiting the host memory bus.
- **2 × 400 Gb/s NICs for the storage / management fabric.** Separate
  fabric so storage traffic never contends with collectives on the
  compute rails.
- **CPU + DRAM + NVMe.** Two host CPUs, ~2 TB of DRAM, and local NVMe
  for scratch. Not on the training critical path once GDS is in play,
  but you still need it for the loader, the launcher, and OS-level
  telemetry.

The most important consequence of this layout: **every GPU has both a
dedicated NVLink domain to every other GPU in the box AND a dedicated
NIC on the compute fabric.** That symmetry is what makes rail-optimized
collectives (chapter 5) work.

### The intra-box mental picture

```
          NVSwitch3   NVSwitch3   NVSwitch3   NVSwitch3
             |            |           |           |
   +---+---+---+---+---+---+---+---+---+---+---+---+
   | GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7          |
   +---+---+---+---+---+---+---+---+---+---+---+---+
     |     |     |     |     |     |     |     |
    HCA0  HCA1  HCA2  HCA3  HCA4  HCA5  HCA6  HCA7   ← 8 rails
     |     |     |     |     |     |     |     |
    (to compute-fabric leaf switches)
```

Two invariants worth naming:

1. **Every GPU pair inside the box has full-bisection NVLink
   bandwidth** — the 4 NVSwitches implement a non-blocking crossbar.
2. **Every GPU has exactly one HCA on the compute fabric.** The rail
   index is stable; NCCL's PXN feature uses that stability to keep
   traffic on the right rail (chapter 5).

## NVLink4 and NVSwitch3

NVLink is a proprietary point-to-point serial link. The relevant facts
for training:

- **NVLink4 (Hopper).** 900 GB/s bidirectional per GPU aggregate across
  its 18 ports. Latency is measured in ~µs for a small message — well
  under the ~10 µs floor of any RDMA fabric.
- **NVSwitch3.** The switch chip that lets NVLink extend past a single
  GPU pair. On DGX H100 there are 4 NVSwitch3 chips per box and every
  GPU-to-GPU pair sees the full aggregate bandwidth. Cross-check
  against the DGX H100 specification and the NVLink / NVSwitch
  whitepapers (NVIDIA H100 GPU whitepaper; DGX H100 datasheet).
- **NVLink Sharp / NVLS.** NVSwitch3 supports in-switch SHARP
  reductions (all-reduce accelerated by the switch itself) for
  supported message sizes. NCCL will pick this via `NCCL_ALGO=NVLS`
  when the topology and payload permit — chapter 5 discusses when it
  fires.

Compared with tier 3 (InfiniBand, chapter 3), NVLink is roughly an
order of magnitude higher bandwidth and roughly an order of magnitude
lower latency per hop. That is why tensor-parallel groups stay inside
the box (`TP ≤ 8` on DGX H100) and why FSDP shard groups often stay
inside a small number of nodes.

## The Scalable Unit (SU)

The **Scalable Unit** is the composition primitive of the SuperPOD.
For the Hopper-generation reference architecture, an SU is nominally
32 DGX H100 nodes — 256 GPUs — with a dedicated leaf/spine tier of the
InfiniBand compute fabric. Confirm the exact count against the current
DGX SuperPOD Reference Architecture PDF; NVIDIA has revised the SU
composition across generations.

The SU is the unit that matters because it is:

- **The scheduler's placement target.** SLURM's topology plugin, Kueue's
  Topology Aware Scheduling, and Volcano's node-topology plugin all try
  to keep a job inside a single SU when possible. A 128-GPU job that
  lands entirely inside one SU has qualitatively lower cross-node
  latency than one spread across three SUs.
- **The natural HSDP shard-group size.** In HSDP (Hybrid FSDP), the
  shard group is the set of ranks that does the all-gather /
  reduce-scatter on model parameters, while the replicate group does a
  cheaper all-reduce on gradients. Confining the shard group to an SU
  keeps the expensive collective on the fastest inter-node domain.
- **The failure-blast-radius unit.** One noisy leaf switch degrades
  every job that touches its SU. When you page on-call, you scope the
  incident to an SU first.

### The compute fabric inside an SU

The inter-node compute fabric inside an SU is a rail-optimized fat-tree
on InfiniBand. Chapter 3 covers this in depth; for the reference
architecture picture you need three facts:

1. **Rails.** The 8 HCAs per DGX are indexed by rail: HCA 0 on every
   node connects to the same leaf switch (leaf 0 for rail 0), HCA 1 to
   leaf 1, and so on. That is the "rail-optimized" part — same-rail
   traffic stays inside its own set of switches and does not compete
   with other rails.
2. **Leaf / spine.** Each rail has its own set of leaf switches (one per
   ~half-SU worth of nodes on Hopper) that fan into spine switches
   inside the SU.
3. **Full bisection at the SU boundary.** The reference architecture
   provisions the fat-tree so that the aggregate NDR bandwidth between
   any two halves of the SU is preserved — the SU is non-blocking at
   this scale.

Draw this from memory before continuing.

## The SuperPOD (domains 3 and above)

A SuperPOD is multiple SUs stitched together by a higher spine tier of
the compute fabric plus a separate storage fabric plus a management
fabric.

- **Compute spine (aggregation).** The rail-optimized fat-tree extends
  upward: every SU's rail-0 spine switch connects to an aggregation
  switch for rail 0 in the SuperPOD, and so on for the other rails.
  This preserves the "rail-aligned" property across the pod.
- **Storage fabric.** A separate InfiniBand (or in some designs
  Ethernet) fabric connecting storage nodes (Lustre / WEKA / DDN) to
  every DGX's dedicated storage NICs. This isolation is critical:
  storage bursts (checkpoint writes, dataset stream-ins) do not steal
  bandwidth from collectives.
- **In-band / out-of-band management.** Ethernet fabrics for BMC,
  cluster management, and login nodes. Not on the training critical
  path.

The SuperPOD as a whole hits between 4 and 32 SUs depending on the
generation; for Hopper the reference range is 4 to 32 SUs (~1024 to
~8192 H100 GPUs). Cross-check the current PDF for exact numbers.

## The four fabrics on a DGX SuperPOD

To pin the picture down before chapter 3 opens up the InfiniBand side:
a SuperPOD has **four distinct fabrics** that a training platform
engineer must be able to name and reason about independently. Getting
these mixed up in a design review is the source of most fabric-related
mistakes.

1. **Compute fabric.** Rail-optimized InfiniBand (NDR on Hopper) with
   one HCA per GPU. Owns every training-time collective. This is the
   fabric that dominates chapter 3 and NCCL tuning in chapter 5.
2. **Storage fabric.** A second high-speed fabric (usually IB, sometimes
   Ethernet) that carries loader traffic, checkpoint I/O, and GDS
   reads. Isolated so it does not contend with collectives.
3. **In-band management.** Ethernet for the cluster orchestrator,
   monitoring, and login nodes.
4. **Out-of-band management (OOB / BMC).** Separate low-speed Ethernet
   for lights-out management, IPMI, and Redfish. Not on any hot path
   but essential to bring nodes back after a failure.

Every subsequent chapter in this module operates on fabric 1 (chapters
3–5), fabric 2 (chapters 6–7), or names the boundary between them
(chapter 8).

## Reading a reference-architecture diagram

When you open the DGX SuperPOD Reference Architecture PDF, work
top-down:

1. **What generation of DGX?** H100/H200 (Hopper) is what this chapter
   assumes. B200 (Blackwell) uses NVLink5 and a different SU shape.
2. **How many GPUs per node, and how are they connected?** Confirm the
   8-GPU + 4-NVSwitch3 layout.
3. **How many nodes per SU, and how are the rails wired?** Confirm SU
   node count, leaf-switch count per rail, and that every same-index
   HCA on every node connects to the same leaf switch.
4. **What's the spine tier between SUs?** Note the switch model and the
   port count so you can estimate bisection bandwidth.
5. **Where does the storage fabric attach?** Confirm it uses different
   NICs from the compute fabric (so it is fabric 2, not fabric 1).

At the end of that read you should be able to answer three questions
about the specific pod in front of you:

- What is the largest gang that fits inside one SU on this pod, at
  8 GPUs / node?
- What is the nominal aggregate all-reduce bandwidth per GPU at the SU
  boundary?
- Which rail of which SU does a given host's HCA 0 land on?

If you cannot, re-read the reference architecture for that page.

## Summary

- The DGX H100 box is 8 GPUs + 4 NVSwitch3 + 8 compute-fabric HCAs.
  Every GPU has full-bisection NVLink to every other GPU in the box
  and a dedicated compute-fabric NIC.
- The SU is 32 nodes / 256 GPUs (Hopper) with a rail-optimized
  fat-tree of InfiniBand leaf and spine switches. It is the natural
  scheduler placement target and the natural HSDP shard group.
- The SuperPOD is 4–32 SUs stitched together by a spine tier plus a
  separate storage fabric plus a management fabric. Four distinct
  fabrics in total; do not confuse them.
- The reference architecture PDF is the ground-truth document. Read
  it top-down and answer the three questions above before you touch
  any tuning knob.
- Every generation change (Hopper → Blackwell) revises the numbers;
  always re-verify against the current PDF for your hardware.
