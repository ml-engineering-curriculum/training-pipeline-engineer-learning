# Resources for mod-105-networking-storage-for-training

Primary standards and vendor docs first, papers second, tooling and
runbook references third. Every chapter and exercise is grounded in
this list; skim the vendor docs while working the drills, and read the
papers in full for the "why".

## NVIDIA reference architectures and DGX documentation

- **NVIDIA DGX SuperPOD documentation index.**
  https://docs.nvidia.com/dgx-superpod/ — the top-level jump page for
  the current SuperPOD reference architectures (H100 / H200 Hopper,
  B200 Blackwell) and the associated design guides. The primary source
  for chapters 2 and 3 and for exercise 1.
- **NVIDIA DGX H100 System Documentation.**
  https://docs.nvidia.com/dgx/dgxh100-user-guide/ — user guide,
  hardware overview, and firmware notes for the DGX H100 box.
- **NVIDIA H100 Tensor Core GPU Whitepaper** (from NVIDIA's H100
  product pages, https://www.nvidia.com/en-us/data-center/h100/) —
  the GPU-side whitepaper that documents NVLink4, HBM3, and the
  fourth-generation NVLink Switch. Read alongside chapter 2.
- **NVIDIA Blackwell architecture technical brief** (from
  https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
  — for readers on B200-class hardware; the Hopper-focused chapters
  need to be cross-referenced against this when tuning for Blackwell.

## InfiniBand, RoCE, and Ethernet-RDMA specs

- **InfiniBand Trade Association (IBTA) specifications.**
  https://www.infinibandta.org/ibta-specifications-download/ — the
  Volume 1 and Volume 2 InfiniBand Architecture Specifications,
  including Annex A16 (RoCE) and Annex A17 (RoCEv2). Chapters 3 and
  4 refer back to these; you do not need to read cover-to-cover, but
  own the verbs and transport chapters.
- **IEEE 802.1Qbb — Priority-based Flow Control (PFC).**
  https://standards.ieee.org/ieee/802.1Qbb/4083/ — the PFC standard
  underlying RoCEv2's lossless-Ethernet contract.
- **RFC 3168 — The Addition of Explicit Congestion Notification (ECN)
  to IP.** https://www.rfc-editor.org/rfc/rfc3168 — the ECN standard
  that RoCEv2's DCQCN reacts to.
- **RFC 7567 — IETF Recommendations Regarding Active Queue
  Management.** https://www.rfc-editor.org/rfc/rfc7567 — background
  for the AQM behavior modern DC switches implement.

## RoCEv2 / RDMA-at-scale papers

- **Zhu, Y., et al. (2015). "Congestion Control for Large-Scale RDMA
  Deployments."** *SIGCOMM '15.* The DCQCN paper. Read alongside
  chapter 4's PFC + ECN + DCQCN discussion.
- **Guo, C., et al. (2016). "RDMA over Commodity Ethernet at Scale."**
  *SIGCOMM '16.* Microsoft's production account of running RoCEv2
  across an Azure-scale fleet. This is the paper that made "RoCEv2
  can actually be operated at scale" a defensible position.
- **Li, Y., et al. (2019). "HPCC: High Precision Congestion Control."**
  *SIGCOMM '19.* The RDMA-native successor congestion-control
  algorithm; useful background for chapter 4's operational discussion.
- **Bai, W., et al. (2023). "Empowering Azure Storage with RDMA."**
  *NSDI '23.* Storage-side RDMA at hyperscaler scale; complements
  chapter 6 / 7 on the storage-fabric side.

## Cloud RDMA transports (EFA, Azure IB, GCP)

- **AWS Elastic Fabric Adapter (EFA) documentation.**
  https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html — the
  reference for EFA, SRD, and the associated placement group /
  instance-type constraints. Chapter 4 refers to this for the
  cloud-fabric case.
- **`aws-ofi-nccl` plugin.**
  https://github.com/aws/aws-ofi-nccl — the NCCL network plugin that
  lets NCCL run on EFA / SRD.
- **Azure InfiniBand-enabled VM sizes.**
  https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-hpc
  — for the ND-series and adjacent H/HB series that expose
  InfiniBand as the compute fabric.
- **GCP A3 instance documentation.**
  https://cloud.google.com/compute/docs/accelerator-optimized-machines
  — for the A3 GPU instances and the accompanying networking
  configuration. Verify the current RDMA transport story per
  generation.

## NCCL

- **NCCL user guide and API reference.**
  https://docs.nvidia.com/deeplearning/nccl/ — collective operations,
  environment variables, algorithm and protocol selection, topology
  discovery, PXN. Primary reference for chapter 5 and exercise 2.
- **`nccl-tests` source and benchmarks.**
  https://github.com/NVIDIA/nccl-tests — the reference benchmark
  suite. `all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf`,
  `alltoall_perf`. Used in exercises 2 and 3.
- **NCCL troubleshooting guide.**
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html
  — first stop for chapter 8's runbook entries.

## Verbs-level tooling

- **`perftest` — InfiniBand / RoCE micro-benchmarks.**
  https://github.com/linux-rdma/perftest — `ib_write_bw`,
  `ib_read_bw`, `ib_send_bw`, `ib_send_lat`. Used in chapter 4 and
  exercise 3 for the verbs-layer baseline.
- **Linux RDMA / rdma-core.**
  https://github.com/linux-rdma/rdma-core — the userspace RDMA
  library, verbs headers, and diagnostic tools (`ibv_devinfo`,
  `ibv_rc_pingpong`).
- **Mellanox / NVIDIA OFED documentation.**
  https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
  — the driver-stack docs, including `mlxreg`, `mlnx_perf`, and
  `ibdiagnet` — the tools chapter 8's runbook reaches for on
  incidents.

## GPUDirect and storage-side RDMA

- **NVIDIA GPUDirect Storage documentation.**
  https://docs.nvidia.com/gpudirect-storage/ — the top-level GDS
  documentation. Primary reference for chapter 6 and exercise 4.
- **NVIDIA `cuFile` API reference.**
  https://docs.nvidia.com/cuda/cufile-api/index.html — the userspace
  API loader code integrates against for GDS.
- **NVIDIA GPUDirect RDMA documentation.**
  https://docs.nvidia.com/cuda/gpudirect-rdma/index.html — the
  GPU↔NIC side of the GPUDirect family.
- **`gdscheck.py` and `gdsio`.** Ship with the `nvidia-fs` package;
  see the GPUDirect Storage docs for current invocation flags.

## Parallel filesystems and object storage

- **Lustre wiki and documentation.** https://wiki.lustre.org/ — the
  reference community documentation for Lustre configuration,
  striping, and troubleshooting. Chapter 7's Lustre section is
  grounded here.
- **Lustre operations manual.**
  https://doc.lustre.org/lustre_manual.xhtml — the canonical operator
  manual for open-source Lustre.
- **DDN documentation portal.** https://www.ddn.com/resources/ — DDN
  EXAScaler and AI400X reference material for the DDN-hardware
  case.
- **WEKA documentation.** https://docs.weka.io/ — WEKA architecture,
  operations, S3 integration, and `nvidia-fs` compatibility.
- **Amazon FSx for Lustre user guide.**
  https://docs.aws.amazon.com/fsx/latest/LustreGuide/ — the reference
  for FSx architecture, Data Repository Associations to S3, and the
  throughput-tier configuration.
- **AWS Mountpoint for Amazon S3.**
  https://github.com/awslabs/mountpoint-s3 — the FUSE client for
  reading S3 objects with training-friendly defaults.
- **Alluxio documentation.** https://docs.alluxio.io/ — for the
  caching-layer alternative discussed in chapter 7's staging section.

## I/O benchmarks used in exercise 4 / 5

- **`fio` — flexible I/O tester.** https://fio.readthedocs.io/ — local
  NVMe and generic block-device benchmark. Used in exercise 4.
- **`ior`.** https://github.com/hpc/ior — parallel-filesystem
  benchmark, canonical for Lustre / DDN cluster acceptance.
- **`mdtest`.** Distributed with `ior`, https://github.com/hpc/ior —
  metadata benchmark for parallel filesystems.
- **`warp` — S3 benchmark tool.** https://github.com/minio/warp —
  standalone S3 benchmark used for the object-store side of the
  staging pattern.
- **s3-benchmark.** https://github.com/wasabi-tech/s3-benchmark —
  simpler S3 benchmark; useful for a quick sanity check.

## Adjacent papers (background)

- **Patarasuk, P., & Yuan, X. (2009). "Bandwidth Optimal All-reduce
  Algorithms for Clusters of Workstations."** *JPDC.* The ring
  all-reduce derivation NCCL's ring algorithm implements.
- **Sanders, P., Speck, J., & Träff, J. L. (2009). "Two-tree Algorithms
  for Full Bandwidth Broadcast, Reduction and Scan."** *Parallel
  Computing.* The double-binary-tree algorithm behind NCCL's tree
  all-reduce.
- **NVIDIA SHARP overview** (from
  https://developer.nvidia.com/networking/sharp) — the whitepaper
  and product page for Scalable Hierarchical Aggregation and
  Reduction Protocol, the in-network reduction primitive chapters 3
  and 5 refer to.

## Recommended reading order for a first pass

1. Chapters 1 and 2 of this module + the current DGX SuperPOD
   Reference Architecture PDF (3 h). Answer the three onboarding
   questions from exercise 1.
2. Chapter 3 + the IBTA Volume 1 chapters on transport and verbs
   (2 h). Skim, do not memorize.
3. Chapter 4 + Zhu et al., 2015 (DCQCN) + Guo et al., 2016 (Azure
   RoCEv2 at scale) (2 h). Enough to run exercise 3.
4. Chapter 5 + the NCCL user guide's collective operations and
   environment-variables chapters (2 h). Enough to run exercise 2.
5. Chapter 6 + GDS docs + `cuFile` API reference (1.5 h).
6. Chapter 7 + Lustre wiki's "Optimizing Lustre" section + your
   chosen vendor's docs (2 h).
7. Chapter 8 + the NCCL troubleshooting guide (1 h). Then read the
   chapter again after your first on-call incident and mark it up.

The exercises assume you have done items 1–5 before starting; items
6–7 come in for exercises 4 and 5.
