# mod-105 — Networking, Storage, and RDMA for Training Clusters

**Estimated effort:** 18 hours

mod-101 taught you the collectives; mod-104 taught you how a job gets
scheduled. This module is about the *fabric* those collectives actually
run on: the NVLink island inside a DGX, the InfiniBand or RoCEv2 spine
between DGXs, and the storage tier that keeps the loader ahead of the
GPUs. Every parallelism strategy in mod-101 lands, in the end, as bytes
moving over one of these links; if the fabric is misconfigured, no
amount of tensor-parallel cleverness will save the step time.

By the end of the module you should be able to (a) read a DGX SuperPOD
reference architecture and reproduce the NVLink / NVSwitch / IB fabric
mental model on a whiteboard, (b) tune NCCL for the topology you
actually have — algorithm, protocol, rail alignment, PXN, subcommunicators
— (c) argue when InfiniBand vs. RoCEv2 wins and prove it with a
benchmark, (d) configure GPUDirect Storage and reason about the
storage-to-GPU throughput budget of a training step, (e) pick between
Lustre, WEKA, and FSx for Lustre for a target workload and design an
S3-staging pattern that hides object-store latency, and (f) diagnose
the canonical training-time fabric failure modes — NCCL timeout, silent
NIC drop, congestion tree, PXN misconfiguration — against a runbook.

## Learning objectives

- Read a DGX SuperPOD reference architecture and reproduce its NVLink /
  NVSwitch / IB fabric mental model.
- Tune NCCL for a given topology (`NCCL_ALGO`, `NCCL_PROTO`, tree vs.
  ring, subcommunicators, PXN, rail alignment).
- Explain when InfiniBand vs. RoCEv2 wins and what to measure to prove it.
- Configure GPUDirect Storage and reason about the storage-to-GPU
  throughput budget for a training step.
- Choose between Lustre, WEKA, and FSx for Lustre for a target workload,
  and design an S3-staging pattern that hides object-store latency.
- Diagnose the canonical training-time network failure modes: NCCL
  timeout, silent NIC drop, congestion tree, PXN misconfiguration.

## Chapters

1. [The Training-Fabric Mental Model](01-training-fabric-mental-model.md) —
   why the fabric dominates step time at scale, the four-tier hierarchy
   (HBM → NVLink → RDMA → storage), and how to think about "effective"
   vs. "nominal" bandwidth on every tier.
2. [DGX SuperPOD Reference Architecture: NVLink, NVSwitch, and the
   Scalable Unit](02-dgx-superpod-reference-architecture.md) — the
   H100/H200 DGX box, NVLink4/NVSwitch3, the Scalable Unit as the
   composition primitive, and the SuperPOD as a hierarchy of SUs.
3. [The InfiniBand Compute Fabric: Rails, Fat-Tree, and Adaptive
   Routing](03-infiniband-compute-fabric.md) — HDR / NDR, rail-optimized
   fat-tree, the subnet manager, adaptive routing, and SHARP in-network
   reduction. Ends with the picture you should be able to draw of a
   SuperPOD compute fabric.
4. [RoCEv2 vs. InfiniBand: When Each Wins and What to
   Measure](04-rocev2-vs-infiniband.md) — Ethernet + RDMA, PFC and ECN,
   DCQCN, the "lossless data-center Ethernet" contract, and the
   benchmark protocol that lets you defend the choice.
5. [Tuning NCCL for the Fabric](05-nccl-tuning.md) — `NCCL_ALGO`,
   `NCCL_PROTO`, `NCCL_IB_HCA`, `NCCL_TOPO_FILE`, PXN, subcommunicators
   and rail-aware collectives, and using `NCCL_DEBUG=INFO` + `nccl-tests`
   as the two feedback loops that keep tuning honest.
6. [GPUDirect RDMA, GPUDirect Storage, and the Storage-to-GPU
   Path](06-gpudirect-storage.md) — the GDR path, `nvidia-fs`, the cuFile
   API, the per-step storage throughput budget, and when the loader
   actually needs GDS.
7. [Parallel Filesystems for Training: Lustre, WEKA, FSx for Lustre, and
   S3 Staging](07-parallel-filesystems-and-s3-staging.md) — MDS/OSS
   Lustre, WEKA's NVMe-tier architecture, FSx for Lustre with S3
   linking, and the tiered staging pattern that hides object-store
   latency behind training.
8. [Diagnosing Fabric Failure Modes: A Runbook](08-fabric-failure-modes-runbook.md) —
   NCCL timeout, silent NIC drop, congestion tree, PXN misconfiguration:
   the symptom, the tools, the fix, and how to write the on-call
   post-mortem.

## Exercises

- [exercise-01 — DGX SuperPOD fabric teardown](exercises/exercise-01-dgx-superpod-fabric-teardown.md) (3 h)
- [exercise-02 — NCCL tuning drills](exercises/exercise-02-nccl-tuning-drills.md) (4 h)
- [exercise-03 — InfiniBand vs. RoCEv2 measurement](exercises/exercise-03-infiniband-vs-rocev2-measurement.md) (3 h)
- [exercise-04 — GPUDirect Storage throughput budget](exercises/exercise-04-gpudirect-storage-throughput-budget.md) (3 h)
- [exercise-05 — Parallel filesystem selection and S3 staging](exercises/exercise-05-parallel-filesystem-selection-and-s3-staging.md) (3 h)

## Labs and quizzes

- `labs/` — a `lab-01` end-to-end fabric characterization (nccl-tests
  sweep + fabric-diagram writeup) lands here on the next autonomous
  cycle.
- `quizzes/` — one knowledge check lands here on the next autonomous
  cycle.

## Resources

- [resources.md](resources.md) — primary docs (NVIDIA DGX SuperPOD
  reference architecture, NCCL, GPUDirect Storage, Lustre, WEKA, FSx
  for Lustre), IETF / IBTA specifications, and a recommended reading
  order.

## How the module fits together

Chapter 1 fixes the mental model and vocabulary. Chapter 2 gives you
the intra-node picture (the DGX box) and the SU composition primitive.
Chapter 3 gives you the inter-node picture on InfiniBand. Chapter 4
does the same for RoCEv2 and gives you a measurement protocol to
choose between them. Chapter 5 is the NCCL tuning surface that maps
your strategy onto the fabric. Chapters 6 and 7 do the same job for
the storage tier: chapter 6 is the GPU-side path (GPUDirect), chapter
7 is the filesystem side (Lustre / WEKA / FSx) and the staging
pattern that ties them together. Chapter 8 is the failure-mode
runbook you will actually reach for on-call. The exercises march in
roughly the same order; do them in sequence when possible.

## What this module deliberately does not cover

- **Distributed-training semantics** (DDP, FSDP2, 3D-parallel) — owned
  by mod-101. This module *uses* the collective cost model from
  chapter 5 of mod-101 as its α + β starting point.
- **Framework internals** (Megatron, DeepSpeed, torchtitan) — owned by
  mod-102.
- **Data pipeline design and shard formats** (WebDataset, MDS, Ray
  Data) — owned by mod-103. This module *uses* the loader as the
  demand side of the storage budget; mod-103 owns how the loader is
  built.
- **Scheduler and launcher plumbing** (SLURM, Kueue, Volcano, KubeRay,
  TorchX) — owned by mod-104. This module *uses* the topology hints
  those schedulers pass through.
- **Checkpointing and elastic training** (fault-tolerant collectives,
  distributed checkpoint I/O) — owned by mod-106. The storage-side
  path is shared; this module owns the fabric, mod-106 owns the
  correctness and recovery semantics.
- **MFU engineering, `torch.compile`, activation checkpointing** —
  owned by mod-107.
- **Observability of training runs** — owned by mod-108.
- **Cost and capacity planning** — owned by mod-109.
