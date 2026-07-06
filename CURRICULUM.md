# Training Pipeline Engineer Curriculum

**Role level:** 35 (deep-specialist architect — owns the large-scale distributed-training platform)
**Status:** planned — modules and projects below are the planned scope authored from [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json). Lessons and projects will be drafted by subsequent autonomous content cycles.

## Overview

This track teaches the large-scale distributed-training platform end-to-end for a deep-specialist Training Pipeline Engineer: distributed-training foundations → training-framework deep dive (FSDP2 / DeepSpeed / Megatron-LM / NeMo / JAX) → training-scale data pipelines → cluster orchestration for training (SLURM / Kueue / Volcano / KubeRay) → networking and storage for training (NCCL / InfiniBand / RoCE / GPUDirect / Lustre / WEKA) → checkpointing / fault tolerance / elastic training → throughput and MFU engineering → training observability and reproducibility at scale → cost, capacity, and cluster economics → training-platform architecture and cross-team leadership.

It is a **specialist-architect track at level 35** — it assumes PyTorch / packaging fundamentals from [`ml-engineer-learning`](https://github.com/ml-engineering-curriculum/ml-engineer-learning) (level 20) and Docker / Kubernetes / cloud fundamentals from [`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) (level 20). It is peer to [`ai-infra-ml-platform-learning`](https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning) (which owns the *serving* / registry surface — complementary, not overlapping) and peer to [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) (which authors kernels this track integrates).

Total planned commitment: **180 hours** across 10 modules + **135 hours** across 3 projects = **~315 hours**.

## Ownership rule

Following the project-wide ownership rule, this curriculum:

- **Owns** the large-scale distributed-training platform end-to-end at depth — distributed-training frameworks, cluster orchestration for training, training-scale data pipelines, networking / storage for training, checkpointing / fault-tolerance / elastic training, throughput / MFU engineering, training-run observability, cluster economics, and training-platform architecture and cross-team leadership at level-35 altitude.
- **Defers down** to [`ml-engineer-learning`](https://github.com/ml-engineering-curriculum/ml-engineer-learning) (level 20) for PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals, and to [`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) (level 20) for Docker / Kubernetes / cloud fundamentals, and to [`ai-infra-junior-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning) for engineering-craft prerequisites.
- **Defers sideways** to [`ai-infra-ml-platform-learning`](https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning) for the *serving / registry / model-lifecycle* platform surface; to [`ai-infra-mlops-learning`](https://github.com/ai-infra-curriculum/ai-infra-mlops-learning) for CI/CD-for-ML and model monitoring after training; to [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) for kernel-level performance (CUDA authoring, FlashAttention internals, KV-cache micro-optimisation); to [`fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning) for the post-training workflow that *consumes* this platform; to [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) / [`ai-eval-engineer-learning`](https://github.com/ml-engineering-curriculum/ai-eval-engineer-learning) for eval platform engineering depth.
- **Defers up** to [`ai-infra-senior-architect-learning`](https://github.com/ai-infra-curriculum/ai-infra-senior-architect-learning), [`ai-infra-principal-architect-learning`](https://github.com/ai-infra-curriculum/ai-infra-principal-architect-learning), and [`staff-ml-engineer-learning`](https://github.com/ml-engineering-curriculum/staff-ml-engineer-learning) / [`principal-ml-engineer-learning`](https://github.com/ml-engineering-curriculum/principal-ml-engineer-learning) for cross-org and multi-team leadership scope.
- **Links out** to [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/security-learning) (peer at level 35) for deep ML/AI security (training-data provenance, checkpoint supply chain, cluster boundary); and to [`ai-governance-analyst-learning`](https://github.com/ml-engineering-curriculum/ai-governance-analyst-learning) for pretraining-data licensing and model-card review depth.

See [`JOB_REQUIREMENTS.md`](JOB_REQUIREMENTS.md) for the requirements-to-coverage map and the cited public references the catalog is grounded in.

## Module Plan

| Module | Title | Hours | Status |
|---|---|---|---|
| mod-101-distributed-training-foundations | Distributed-Training Foundations: DDP, Parallelism Strategies, and the Communication-Cost Model | 18 | planned |
| mod-102-training-frameworks-deep-dive | Training Frameworks Deep Dive: PyTorch FSDP2 / DeepSpeed / Megatron-LM / NeMo / JAX | 22 | planned |
| mod-103-training-scale-data-pipelines | Training-Scale Data Pipelines: WebDataset, MosaicML Streaming, Ray Data, and Tokenization at Scale | 20 | planned |
| mod-104-cluster-orchestration-for-training | Cluster Orchestration for Training Jobs: SLURM, Kubernetes (Kueue / Volcano / KubeRay / MPI Operator), and Multi-Scheduler Launchers | 20 | planned |
| mod-105-networking-storage-for-training | Networking, Storage, and RDMA for Training Clusters | 18 | planned |
| mod-106-checkpointing-fault-tolerance-elastic-training | Checkpointing, Fault Tolerance, and Elastic Training | 18 | planned |
| mod-107-throughput-and-mfu-engineering | Throughput and MFU Engineering: Kernels, Mixed Precision, and Communication Overlap | 20 | planned |
| mod-108-training-observability-and-reproducibility | Training Observability and Reproducibility at Scale | 15 | planned |
| mod-109-cost-capacity-and-cluster-economics | Cost, Capacity, and Cluster Economics for Training Runs | 14 | planned |
| mod-110-training-platform-architecture-and-leadership | Training-Platform Architecture and Cross-Team Leadership | 15 | planned |

## Project Plan

| Project | Title | Hours | Status |
|---|---|---|---|
| project-101-small-scale-pretraining-run | Small-Scale Pretraining Run: 1–3B model on a 2-node GPU cluster with reproducibility bundle | 40 | planned |
| project-102-elastic-fault-tolerant-training | Elastic, Fault-Tolerant Training: 7B–13B fine-tune with DCP + torchrun elastic reshape | 45 | planned |
| project-103-training-platform-capstone | Training Platform Capstone: Multi-Tenant Training Cluster Architecture + Roll-out Plan + Hand-off Contracts | 50 | planned |

## Module summaries

### mod-101 — Distributed-Training Foundations
DDP (all-reduce), FSDP (reduce-scatter + all-gather), ZeRO-3 partitioned step, and the full parallelism-strategy design space (DP / TP / PP / SP / EP / HSDP / 3D-parallel). NCCL collective cost model (bandwidth × latency × topology). Ends with a Llama 3 / BLOOM / OPT-175B parallelism teardown.

### mod-102 — Training Frameworks Deep Dive
Same 3B model, three stacks: PyTorch FSDP2 (torchtitan-style), DeepSpeed ZeRO-3, and Megatron-LM 3D-parallel. FSDP2 per-parameter sharding + DTensor vs. FSDP1 flat-parameter. ZeRO-Infinity + ZeRO-Offload vs. FSDP2 + CPU offload. Megatron-LM → NeMo (Megatron-Core + Lightning) port. JAX/Flax + pjit / shard_map / GSPMD on 8 GPUs. Ends with a decision doc.

### mod-103 — Training-Scale Data Pipelines
WebDataset tar-shard pipelines that saturate compute; MosaicML StreamingDataset for resumable / deterministic / shard-aware sampling; distributed tokenization + dedup with Ray Data over Parquet; staging layer from S3 / GCS to Lustre / WEKA with I/O budget matched to per-step compute; the four canonical failure modes (shuffle collapse, epoch drift, tokenizer drift, silent shard corruption). Ends with a data-pipeline runbook.

### mod-104 — Cluster Orchestration for Training Jobs
SLURM sbatch topology and prolog/epilog hardening; Kubernetes with Kueue (workload queues) or Volcano (gang scheduler) + MPI Operator; Ray Train on KubeRay with autoscaling worker groups; TorchX / torchrun for scheduler-agnostic launch; quotas, fair-share, gang preemption, priority classes. Ends with a researcher-facing launcher SDK.

### mod-105 — Networking, Storage, and RDMA for Training Clusters
DGX SuperPOD reference architecture; NCCL tuning (algo / proto / tree vs. ring / subcommunicators); InfiniBand vs. RoCEv2 measurement; GPUDirect Storage and the storage-to-GPU throughput budget; Lustre vs. WEKA vs. FSx for Lustre with an S3-staging pattern; NCCL timeout, silent NIC drop, congestion tree, and PXN mis-config diagnostics.

### mod-106 — Checkpointing, Fault Tolerance, and Elastic Training
PyTorch Distributed Checkpoint (DCP) with async save + resharding across a different world size, stateful sampler; torchrun rendezvous + elastic reshape flow that survives a node crash mid-epoch; incident-classification playbooks (loss spike, NaN, NCCL timeout, hardware fault, silent-corruption); straggler detection and node quarantine; goodput SLO + availability budget. OPT-175B logbook and Llama 3 failure statistics as anchor cases.

### mod-107 — Throughput and MFU Engineering
MFU calculated correctly and gap-to-peak analysis; FlashAttention v2 (Ampere) and v3 (Hopper + FP8) integration; BF16 across FSDP2 layered with NVIDIA Transformer Engine FP8 on H100; activation + gradient checkpointing + sequence packing to hit a memory target; `torch.compile` (PT2) + Triton-authored kernels with FSDP2 and DeepSpeed; communication-compute overlap design and measurement. Boundary with `ai-infra-performance-engineer` framed explicitly.

### mod-108 — Training Observability and Reproducibility at Scale
Training-run dashboards (loss / gradient / throughput / MFU / GPU utilization / thermals / NCCL health); DCGM into per-node telemetry + Prometheus / Grafana cluster-wide roll-ups; W&B / TensorBoard / MLflow at pretraining scale (sharded logging, sub-sampled metrics, offline sync); reproducibility bundle spec (seed + config + dataset hash + tokenizer hash + framework versions + container digest + hardware manifest); run-time signature catalog (divergence / loss spike / throughput cliff / straggler / silent-corruption); hand-off metadata contract with fine-tuning / model-evaluation peer tracks.

### mod-109 — Cost, Capacity, and Cluster Economics
Chinchilla / Kaplan scaling laws for compute-optimal budgeting; parameter count × data budget → GPU-hours × dollars for a target cluster shape; reserved-vs-spot / dedicated-vs-shared / on-prem-vs-cloud economics; MFU-uplift-to-dollars; training-run feasibility study (product target → cluster shape → dollar cost → wall-clock schedule); MPT-7B and Llama 3 as anchor cost-vs-quality recipes.

### mod-110 — Training-Platform Architecture and Cross-Team Leadership
Multi-tenant training-cluster architecture for 5–15 downstream teams; RFC authoring for surface changes (framework upgrade, hardware refresh, storage migration); explicit hand-off contracts with fine-tuning-engineer (consumer), ai-infra-ml-platform-engineer (serving/registry), ai-infra-mlops-engineer (CI/CD), ai-infra-performance-engineer (kernels), ai-infra-security-engineer (data provenance + cluster boundary); build-vs-buy at level-35 altitude (in-house Megatron vs. NeMo integration vs. Databricks/Mosaic-hosted vs. Together-hosted); incident review anchored in OPT-175B / BLOOM; migration plan with rollback + compatibility windows.

## Assessment

Each module ships **1 quiz** plus the exercises and lab listed in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json). Each project ships a portfolio-grade README, a reproducibility bundle (config + dataset hashes + seeds + container digest + hardware manifest), and an explicit rubric covering distributed-training correctness, data-pipeline throughput, cluster orchestration, checkpointing / fault-tolerance behavior, MFU / goodput evidence, observability contract, and platform documentation.

## Where to go after this curriculum

- **`ai-infra-senior-architect-learning`** — next step upstream on the architect ladder for cross-domain AI infrastructure architecture.
- **`ai-infra-principal-architect-learning` / `ai-infra-principal-engineer-learning`** — org-level strategy and multi-team architecture scope.
- **`staff-ml-engineer-learning` / `principal-ml-engineer-learning`** — ML engineering ladder for architectural / leadership scope.
- **`ai-infra-ml-platform-learning`** — peer specialist for the *serving / registry / model-lifecycle* platform surface, complementary to this track.
- **`ai-infra-performance-learning`** — peer specialist for kernel-level performance authoring (deeper than this track's integration altitude).
- **`ai-infra-security-learning`** — peer specialist for deep ML/AI security (data poisoning, training-data exfiltration, adapter supply chain, secure model registry).
- **`fine-tuning-engineer-learning`** — the post-training track that *consumes* this platform.

<!-- needs-research: backfill industry-frequency evidence into JOB_REQUIREMENTS.md once the autonomous research loop runs with WebSearch / WebFetch exercised against direct Training-Pipeline-Engineer-titled postings; demote any module or exercise whose underlying requirement does not show up in ≥3 in-window direct postings. -->
