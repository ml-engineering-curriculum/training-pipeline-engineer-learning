# The Training-Fabric Mental Model

mod-101 taught you that every parallelism strategy is a mapping from
*collectives* onto a *hierarchy of links*. This module is about the
links. Before we look at any specific hardware — NVLink, NVSwitch,
InfiniBand, RoCEv2, Lustre, WEKA — we need a shared way to talk about
what a fabric is, what layers it has, and where "the number on the spec
sheet" and "the number you observe in your training run" diverge.

## Motivation: at scale, the fabric is the bottleneck

Two facts you should internalize before the rest of this module:

1. **A single H100 does ~1000 TFLOPS of BF16 dense matmul at peak.** A
   400 Gb/s NDR InfiniBand port does ~50 GB/s at line rate.
   That is ~20 000× more compute per second than wire per second on a
   *single* GPU. Any strategy that lets the fabric become the critical
   path has already lost the game.
2. **The comm-to-compute ratio scales the wrong way.** As you shard
   more (larger world size), per-GPU compute stays roughly constant but
   per-GPU communication grows — linearly for pipeline stage boundaries,
   as `2(N-1)/N · S · β` for a ring all-reduce, and worse for tensor
   parallel across the node boundary. That is why every large-scale
   training paper (Llama 3, BLOOM, OPT-175B, Megatron-LM) devotes
   sections to fabric topology — because on a 10 000-GPU run, a 5%
   fabric misconfiguration is more expensive than any tuning knob in
   the framework.

The consequence: as a training platform engineer you spend an
uncomfortable fraction of your time debugging fabric, not code. This
module is training for that.

## The four-tier link hierarchy

Every training cluster has the same four link tiers, in decreasing
order of bandwidth per GPU. You should be able to sketch this from
memory before moving on:

| Tier | Link | Order-of-magnitude bandwidth per GPU | Owner in a training step |
|------|------|--------------------------------------|--------------------------|
| 1 | Intra-GPU HBM | ~3 TB/s (HBM3 on H100) | Every kernel |
| 2 | Intra-node NVLink / NVSwitch | ~900 GB/s bidirectional per H100 | Tensor-parallel activations, intra-node all-reduce |
| 3 | Inter-node RDMA (InfiniBand or RoCEv2) | ~25 GB/s (HDR) or ~50 GB/s (NDR) per port | Data-parallel gradients, pipeline P2P, cross-node FSDP all-gather |
| 4 | Storage fabric (parallel FS / object store) | ~1–10 GB/s per node, aggregate | Data loader, checkpoint I/O |

Verify the exact numbers against your hardware's spec sheet — the NVIDIA
DGX H100/H200 datasheet, your fabric vendor's port speed, and your
storage vendor's per-client throughput number. The columns above are
the *shape* of the answer, not the answer.

Two rules of thumb that fall out of the table:

- **A tier-2 collective is roughly 20× cheaper than the same collective
  on tier 3.** That is why intra-node tensor-parallel (TP = 8) works
  and cross-node TP does not.
- **A tier-3 collective is roughly 20× cheaper than a tier-4 read** for
  a similarly-sized payload, but tier 4 is *much* higher latency per
  request. That mismatch is why the data loader lives behind a prefetch
  queue and not on the critical path — see mod-103.

## Nominal vs. effective bandwidth

Every bandwidth number on a spec sheet is nominal — the number you get
when the link is running under ideal conditions with a perfectly
matched workload. The number you observe is effective, and it is
almost always lower. The reasons:

- **Protocol overhead.** InfiniBand and RoCEv2 both have per-packet
  headers, MTU limits, and per-message setup. For small messages,
  effective bandwidth can be less than 10% of nominal.
- **Topology penalties.** If your GPU-to-NIC pinning is wrong, a
  collective can transit the host memory controller instead of going
  straight through GPUDirect RDMA. Effective bandwidth drops
  multiplicatively — often to the PCIe root-complex bandwidth, not the
  NIC's.
- **Congestion.** Two rails using the same spine switch port drop each
  other's effective bandwidth. On lossless Ethernet (RoCEv2 with PFC)
  this manifests as pause frames; on InfiniBand as credit stalls.
- **Algorithm overhead.** Ring all-reduce achieves at best
  `1 - 1/N` of the link bandwidth per rank; tree all-reduce is worse
  for large messages.

Chapter 5 of mod-101 introduced `algbw` (algorithmic bandwidth, `n/T`)
and `busbw` (bus bandwidth, `n/T · f(N)`) as the two ways `nccl-tests`
reports this. Use `busbw` to compare against nominal link speed; use
`algbw` to compare across world sizes. When someone says "our fabric
runs at 92% of line rate", they almost always mean `busbw / nominal`
on a large-message all-reduce.

## The comm domain hierarchy

The mental picture that carries you through the rest of this module:
a training cluster is a **hierarchy of communication domains**.

- **Domain 0: one GPU.** All kernels and their HBM reads.
- **Domain 1: one node.** The 8 (usually) GPUs on one host, connected by
  NVLink through NVSwitch. Bisection bandwidth is measured in ~TB/s.
- **Domain 2: one Scalable Unit (SU).** ~32 nodes = ~256 GPUs on the
  NVIDIA reference design, connected by a leaf/spine of the RDMA
  fabric. Bisection bandwidth is measured in ~TB/s but with much
  higher per-hop latency than NVLink.
- **Domain 3: one SuperPOD.** Multiple SUs stitched together by a
  higher spine tier. Bisection bandwidth is measured in ~TB/s but
  further degraded per rank.
- **Domain 4: the storage fabric.** Parallel filesystem clients and
  object storage. Latency is measured in ms, not µs.

Every parallelism strategy places its collectives on one of those
domains. TP on domain 1. FSDP intra-SU on domain 2. HSDP replicates
across SUs on domain 3. Data loader on domain 4. If you can name the
domain a collective runs on, you can predict its cost.

## What "topology-aware" actually means

Every scheduler and NCCL knob you meet in this module claims to be
topology-aware. What they actually do:

- **The scheduler chooses your nodes.** SLURM's `topology.conf`,
  Kueue's Topology Aware Scheduling, and Volcano's node-topology plugin
  all try to place a job's ranks *inside* the smallest possible comm
  domain that fits. A 128-GPU job that lands entirely in one SU
  performs qualitatively better than one spread across three.
- **NCCL discovers the intra-domain layout.** `NCCL_DEBUG=INFO` prints
  the topology NCCL inferred; `NCCL_TOPO_FILE` lets you override it.
  Ring construction, rail assignment, and PXN routing all happen here.
- **The application chooses process groups.** A rail-aware FSDP shard
  group or a tensor-parallel group is a subcommunicator: you deliberately
  keep its collectives inside a fast domain.

Making these three layers agree — scheduler picks the right nodes,
NCCL discovers the right topology, application defines the right
groups — is the single most impactful set of choices this module
teaches. Chapter 5 is where they compose.

## The "training step" as a fabric budget

The rest of this module keeps coming back to the same accounting: one
training step has a fixed budget of GPU time. That budget is spent on:

- **Compute** — the forward + backward matmuls, elementwise ops, and
  attention.
- **Intra-node collectives** — NVLink traffic for tensor parallel,
  intra-node all-reduce, sequence-parallel activations.
- **Inter-node collectives** — RDMA traffic for data parallel,
  cross-node FSDP, pipeline P2P.
- **Storage I/O** — the loader has to have the next batch ready and
  the checkpoint has to land somewhere.

If any of these grows past the compute budget, the step slows down;
if it grows past *twice* the compute budget, no overlap in the world
saves you. The rest of this module is a tour of the fabric-side of
those four line items: chapters 2–5 for tiers 2 and 3, chapters 6–7
for tier 4. Chapter 8 is what happens when the accounting stops
adding up on-call.

## Summary

- At training scale, the fabric — not the model — is the dominant
  cost of a step. Any fabric misconfiguration shows up multiplicatively
  in step time.
- Every training cluster has the same four link tiers: HBM, NVLink
  (intra-node), RDMA (inter-node), storage. Their bandwidths differ
  by ~1000×.
- Nominal bandwidth is a spec-sheet number; effective bandwidth is
  what you measure. The delta is where every fabric-tuning story in
  this module lives.
- A cluster is a hierarchy of comm domains: GPU → node → SU →
  SuperPOD → storage. Every strategy places its collectives on one
  domain; naming the domain names the cost.
- The three-layer topology stack (scheduler / NCCL / application)
  must agree for the strategy to run at line rate. Chapter 5 is where
  we make them agree.
