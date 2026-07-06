# ML Engineering · Training Pipeline Engineer — Learning Repository

<!-- aicg:site-banner -->
> 🎓 Part of the free, open-source **AI Career Curriculum** ecosystem — [Infrastructure](https://github.com/ai-infra-curriculum) · [ML Engineering](https://github.com/ml-engineering-curriculum) · [AI Engineering](https://github.com/ai-engineering-curriculum) · [Governance](https://github.com/ai-governance-curriculum). Live cohorts &amp; team programs: **[ai-infra-curriculum.github.io](https://ai-infra-curriculum.github.io/)**.
<!-- /aicg:site-banner -->

<!-- aicg:sponsor -->
> 💜 **[Sponsor this curriculum](https://github.com/sponsors/ml-engineering-curriculum)** — sponsorships keep the whole open-source AI Career Curriculum free and moving.
<!-- /aicg:sponsor -->

Build the platform underneath frontier training: high-throughput, fault-tolerant distributed-training systems and data loading at trillion-token scale.

> **Status**: curriculum plan authored (2026-07-05). Lessons and project READMEs will be drafted by subsequent autonomous content cycles. See [`CURRICULUM.md`](CURRICULUM.md) for the planned scope and [`JOB_REQUIREMENTS.md`](JOB_REQUIREMENTS.md) for the requirements-to-coverage map (postings list uses the peer-track reuse-with-attribution pattern — see `JOB_REQUIREMENTS.md` Status section).

**Level:** 35 (deep-specialist architect — owns the large-scale distributed-training platform).
**Planned scope:** 10 modules (180h) + 3 projects (135h) = ~315 hours.

## What this track owns

The large-scale distributed-training platform end-to-end at depth:

- Distributed-training foundations — DDP, FSDP2, ZeRO, tensor / pipeline / sequence / expert parallel, HSDP, 3D parallel, communication-cost model
- Training frameworks in depth — PyTorch FSDP2 + torchtitan, DeepSpeed ZeRO-Infinity + Offload, Megatron-LM 3D-parallel, NVIDIA NeMo, ColossalAI, JAX/Flax with pjit / shard_map / GSPMD
- Training-scale data pipelines — WebDataset, MosaicML StreamingDataset, Ray Data, distributed tokenization, dedup at trillion-token scale, S3/GCS-to-Lustre staging
- Cluster orchestration for training — SLURM, Kubernetes + Kueue / Volcano / KubeRay / MPI Operator, TorchX / torchrun launchers, gang scheduling, quota / preemption / fair-share policy
- Networking & storage for training — NCCL tuning, NVLink / NVSwitch, InfiniBand / RoCEv2, GPUDirect Storage, Lustre / WEKA / FSx for Lustre
- Checkpointing, fault tolerance, elastic training — PyTorch DCP with async save + resharding, torchrun elastic reshape, straggler / silent-corruption detection, goodput SLOs
- Throughput and MFU engineering — FlashAttention v2/v3, BF16 / FP8 mixed precision, activation / gradient checkpointing, `torch.compile` + Triton integration, communication-compute overlap
- Training-run observability & reproducibility at scale — DCGM + Prometheus / Grafana, W&B / TensorBoard / MLflow at pretraining scale, reproducibility bundles, run-time signature catalog
- Cost, capacity, and cluster economics — Chinchilla-scale compute budgeting, GPU-hours × dollars, reserved-vs-spot economics, MFU-uplift-to-dollars
- Training-platform architecture and cross-team leadership at Level 35 — multi-tenant cluster design, RFCs, hand-off contracts with peer platform tracks, build-vs-buy decisions, migration plans

## What this track defers

- PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals → [`ml-engineer-learning`](https://github.com/ml-engineering-curriculum/ml-engineer-learning) (level 20).
- Docker / Kubernetes / cloud fundamentals → [`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) (level 20).
- The serving / registry / model-lifecycle platform surface → [`ai-infra-ml-platform-learning`](https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning) (peer at level 30).
- CI/CD-for-ML, model monitoring, run-to-production pipelines *after* training → [`ai-infra-mlops-learning`](https://github.com/ai-infra-curriculum/ai-infra-mlops-learning) (peer at level 30).
- Kernel-level performance authoring (CUDA kernels, FlashAttention internals, KV-cache micro-optimisation) → [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) (peer at level 35).
- Post-training workflow (SFT, PEFT, DPO/RLHF, adapter release) — the *customer* of this platform → [`fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning).
- Eval platform engineering → [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) / [`ai-eval-engineer-learning`](https://github.com/ml-engineering-curriculum/ai-eval-engineer-learning).
- Deep ML/AI security → [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/security-learning) (peer at level 35).
- Governance / compliance / pretraining-data licensing → [`ai-governance-analyst-learning`](https://github.com/ml-engineering-curriculum/ai-governance-analyst-learning).

## Layout

```
training-pipeline-engineer-learning/
├── .aicg/                    curriculum-plan.json and job-requirements.json (machine-readable catalog)
├── lessons/mod-XXX-*/        modules with lectures, exercises, labs, quizzes (to be drafted)
├── projects/project-XXX-*/   multi-module capstones (to be drafted)
├── CURRICULUM.md             role-level coverage map
├── JOB_REQUIREMENTS.md       requirements catalog with citations and ownership map
├── PREREQUISITES.md          assumed entry skills
├── VERSIONS.md               release history
└── README.md                 this file
```

## Paired Solutions Repo

[`training-pipeline-engineer-solutions`](https://github.com/ml-engineering-curriculum/training-pipeline-engineer-solutions) carries the reference implementations.

---

<!-- aicg:maintained-by -->
Maintained by [VeriSwarm.ai](https://veriswarm.ai)
