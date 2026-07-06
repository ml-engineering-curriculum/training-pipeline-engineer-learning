# Job Requirements — Training Pipeline Engineer

**Role level:** 35 (deep-specialist architect — owns the large-scale distributed-training platform)
**Track:** `training-pipeline-engineer-learning`
**Research window:** 2026-03-07 → 2026-07-05 (last ~120 days)
**Today:** 2026-07-05

This file documents the requirements catalog used to seed the Training Pipeline Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the planned curriculum lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json).

## Status — bootstrap session, direct posting sample deferred

<!-- needs-research: gather ≥25 in-window postings whose titles include "Training Pipeline Engineer", "Distributed Training Engineer", "Software Engineer, Distributed Training", "ML Infrastructure Engineer, Training", "Pretraining Infrastructure Engineer", "Foundation Model Training Infrastructure Engineer", "LLM Training Infrastructure Engineer", "Model Training Platform Engineer" (platform-specifically-for-training), or "GPU/HPC Training Systems Engineer" at LLM scale. Filter OUT generic MLOps / ML Engineer / Inference-Serving / Data-Engineer / Research-Scientist titles (owned by peer tracks). Re-validate every requirement against the direct evidence and demote any whose evidence_post_ids stays empty. -->

This packet was authored in a bootstrap session **without an exercised WebSearch / WebFetch pass against live job boards** — both tools returned `Claude requested permissions to use WebSearch, but you haven't granted it yet` on every attempt, and the sub-agent tasked with the research fan-out reported the same block. Per the project rules (*"Do not invent facts, incidents, or salary figures. Cite sources."*), the `postings` array in `.aicg/job-requirements.json` does **not** claim direct Training-Pipeline-Engineer-titled evidence. Instead, this bootstrap follows the [`fine-tuning-engineer-learning`](../fine-tuning-engineer-learning/JOB_REQUIREMENTS.md) precedent and adds a **reuse-with-attribution** layer:

- The 15 postings in `postings[]` are **reused** from the sibling `ai-infra-ml-platform-learning` and `ai-infra-mlops-learning` research cycles, filtered to those whose verbatim requirements demonstrably touch training-infrastructure work (Reddit "ML Training Platform"; Anthropic "Research Data Platform"; Pinterest "Distributed training with GPUs"; Multiverse "NVIDIA NeMo for distributed training"; Salesforce/Slack "Architect distributed training and data processing systems using platforms such as Ray, Airflow, and Spark"; JPMorgan "distributed model training on large compute clusters"; etc.). Each carries a `reused_from` field pointing to its origin record and a `key_quotes` block with verbatim language.
- The **authoritative references** list grounds each requirement in the framework and paper canon the role is hired against — PyTorch FSDP / FSDP2 / DCP / torchtitan, DeepSpeed, Megatron-LM, NVIDIA NeMo, ColossalAI, JAX/Flax, NCCL, SLURM / Kueue / Volcano / KubeRay / TorchX, WebDataset, MosaicML Streaming, Ray Data, FlashAttention (v2/v3), Chinchilla / Kaplan scaling laws, PaLM MFU, DGX SuperPOD reference architecture, and the OPT-175B / BLOOM / Llama 3 / MPT-7B open training reports.

Every requirement below cites at least one such reference. Every requirement is shaped so that direct posting-frequency evidence can be added underneath it without restructure on the next research cycle.

## Methodology

1. Inventoried authoritative public references for the role's domain — see `authoritative_references` in `.aicg/job-requirements.json`:
   - Distributed-training framework docs: PyTorch FSDP / FSDP2, PyTorch DCP, torchtitan, DeepSpeed, Megatron-LM, NVIDIA NeMo, ColossalAI, JAX/Flax
   - Cluster / launcher docs: SLURM, Kueue, Volcano, KubeRay, TorchX, torch.distributed.elastic, Accelerate
   - Networking / storage docs: NCCL, NVLink/NVSwitch, InfiniBand, RoCEv2, Lustre, WEKA, FSx for Lustre, GPUDirect Storage, DGX SuperPOD reference architecture
   - Data-pipeline docs: WebDataset, MosaicML StreamingDataset, Ray Data, Arrow/Parquet, HF datasets
   - Performance references: FlashAttention (v1/v2/v3), Mixed Precision (Micikevicius 2017), NVIDIA Transformer Engine (FP8), `torch.compile`, OpenAI Triton, PaLM MFU
   - Systems / economics references: Chinchilla (Hoffmann 2022), Kaplan scaling laws (2020), MPT-7B, BLOOM, OPT-175B logbook, Llama 3 Herd of Models
2. Mapped each requirement to (a) the role on our level ladder that should own it primarily and (b) the curriculum module that covers it.
3. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill, with higher levels linking back rather than duplicating fundamentals.
4. **Reused with attribution** peer-role posting evidence (from `ai-infra-ml-platform-learning` and `ai-infra-mlops-learning`) where verbatim requirements demonstrably touch training-infrastructure work.
5. Flagged every requirement whose evidence stays empty after the reuse pass with `<!-- needs-research -->` so the next cycle can demote it if direct evidence never materialises.

## Requirement themes → curriculum ownership

| # | Theme | Direct evidence | Reused peer evidence | Owner role | Coverage |
|---|---|---|---|---|---|
| 1 | Distributed-training frameworks (PyTorch FSDP/FSDP2, DeepSpeed ZeRO 1/2/3 + Infinity + Offload, Megatron-LM tensor/pipeline/sequence/expert parallel, NeMo, ColossalAI, JAX/Flax) | <!-- needs-research --> | 6 postings (`p-reuse-01, 07, 08, 09, 10, 11`) | `training-pipeline-engineer` (this) | [`mod-101-distributed-training-foundations`](lessons/mod-101-distributed-training-foundations), [`mod-102-training-frameworks-deep-dive`](lessons/mod-102-training-frameworks-deep-dive) |
| 2 | Training-scale data pipelines (WebDataset, MosaicML StreamingDataset, Ray Data, distributed tokenization, dedup at trillion-token scale) | <!-- needs-research --> | 4 postings (`p-reuse-03, 10, 11, 14`) | `training-pipeline-engineer` | [`mod-103-training-scale-data-pipelines`](lessons/mod-103-training-scale-data-pipelines) |
| 3 | Cluster orchestration for training (SLURM, Kubernetes + Kueue / Volcano / KubeRay / MPI Operator, TorchX / torchrun / Ray Train launchers, gang scheduling) | <!-- needs-research --> | 5 postings (`p-reuse-01, 05, 06, 08, 12`) | `training-pipeline-engineer` | [`mod-104-cluster-orchestration-for-training`](lessons/mod-104-cluster-orchestration-for-training) |
| 4 | Networking & storage for training (NCCL, NVLink / NVSwitch, InfiniBand / RoCEv2, GPUDirect Storage, Lustre / WEKA / FSx, S3 staging) | <!-- needs-research --> | 2 postings (`p-reuse-05, 08`) | `training-pipeline-engineer` | [`mod-105-networking-storage-for-training`](lessons/mod-105-networking-storage-for-training) |
| 5 | Checkpointing, fault tolerance, elastic training (PyTorch DCP async / resharding, torchrun rendezvous, restart-from-crash, straggler / silent-corruption detection) | <!-- needs-research --> | 0 postings — anchor on framework docs + OPT-175B / BLOOM / Llama 3 open reports | `training-pipeline-engineer` | [`mod-106-checkpointing-fault-tolerance-elastic-training`](lessons/mod-106-checkpointing-fault-tolerance-elastic-training) |
| 6 | Throughput & MFU engineering (FlashAttention v2/v3 integration, BF16 / FP8 mixed precision, activation / gradient checkpointing, `torch.compile` + Triton kernel integration, communication overlap, MFU as the throughput metric) | <!-- needs-research --> | 1 posting (`p-reuse-08`) | `training-pipeline-engineer` | [`mod-107-throughput-and-mfu-engineering`](lessons/mod-107-throughput-and-mfu-engineering) |
| 7 | Training observability & reproducibility at scale (loss / gradient / throughput dashboards, DCGM GPU-health telemetry, W&B / TensorBoard / MLflow at pretraining scale, reproducibility bundles) | <!-- needs-research --> | 2 postings (`p-reuse-03, 06`) | `training-pipeline-engineer` | [`mod-108-training-observability-and-reproducibility`](lessons/mod-108-training-observability-and-reproducibility) |
| 8 | Training economics (Chinchilla / Kaplan-scale budgeting, tokens-per-dollar, GPU-hours-per-run, reserved-vs-spot, feasibility-to-cluster-shape) | <!-- needs-research --> | 0 postings — anchor on scaling-laws / MPT-7B / Llama 3 open economics | `training-pipeline-engineer` | [`mod-109-cost-capacity-and-cluster-economics`](lessons/mod-109-cost-capacity-and-cluster-economics) |
| 9 | Training-platform DX (launcher SDKs, run-templating, config-first ergonomics, researcher-as-customer thinking) | <!-- needs-research --> | 2 postings (`p-reuse-01, 04`) | `training-pipeline-engineer` | [`mod-110-training-platform-architecture-and-leadership`](lessons/mod-110-training-platform-architecture-and-leadership) |
| 10 | Training-platform architecture & cross-team leadership at level 35 (multi-tenant cluster design, quota / preemption / fair-share, on-call for training runs, incident reviews, RFC authoring, hand-off contracts with peer platform tracks, migration strategy) | <!-- needs-research --> | 3 postings (`p-reuse-02, 07, 15`) | `training-pipeline-engineer` | [`mod-110-training-platform-architecture-and-leadership`](lessons/mod-110-training-platform-architecture-and-leadership) + [`project-103-training-platform-capstone`](projects/project-103-training-platform-capstone) |
| 11 | PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals | n/a — prerequisite | n/a | `ml-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 12 | Docker / Kubernetes / cloud fundamentals | n/a — prerequisite | n/a | `ai-infra-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught. Training-specific K8s (Kueue / Volcano / KubeRay / MPI Operator) IS owned here in mod-104 |
| 13 | SFT / PEFT / preference-optimization methods and the release workflow around them | n/a — peer track | n/a | `fine-tuning-engineer` (level 30) | Out of scope. Fine-tuning-engineer is the CUSTOMER of this platform |
| 14 | Model registry, inference serving (vLLM / TGI / TensorRT-LLM), LLM gateway | n/a — peer track | n/a | `ai-infra-ml-platform` (level 30) | Out of scope. ai-infra-ml-platform-engineer owns the serving surface |
| 15 | Kernel-level CUDA / FlashAttention-internal / KV-cache micro-optimization | n/a — peer track | n/a | `ai-infra-performance` (level 35) | Peer at same level. This track *uses* published kernels; ai-infra-performance *authors* them |
| 16 | Deep ML/AI security (data poisoning, training-data exfiltration, adapter supply chain, secure model registry) | n/a — peer track | n/a | `ai-infra-security` (level 35) | Peer at same level. Surfaced as awareness in mod-105 and mod-110 |
| 17 | Pretraining-data licensing due-diligence, model-card authoring for release | n/a — peer track | n/a | `ai-governance-analyst` | Surfaced as awareness in mod-103 and mod-110 |

## Posting evidence

<!-- needs-research: replace the reused-peer sample below with ≥25 direct Training-Pipeline-Engineer-titled postings on the next autonomous cycle. Use the same JSON shape as ai-infra-ml-platform-learning/.aicg/job-requirements.json postings[]. -->

The 15 postings included in `.aicg/job-requirements.json` are **reused with attribution** from `ai-infra-ml-platform-learning/.aicg/job-requirements.json` and `ai-infra-mlops-learning/.aicg/job-requirements.json`. Each carries a `reused_from` field and a `training_relevance` note. Reuse is limited to postings whose verbatim requirements demonstrably touch training-infrastructure work:

| # | Employer | Title | Reused from | Training relevance |
|---|---|---|---|---|
| 1 | Reddit | Senior Machine Learning Engineer, ML Training Platform | ml-platform p007 | Direct — title is "ML Training Platform" |
| 2 | Samsara | Staff ML Engineer - ML Infrastructure | ml-platform p011 | Partial — "training, experimentation, batch/online inference, edge" |
| 3 | Anthropic | Software Engineer, Research Data Platform | ml-platform p014 | Direct — "Pipelines from research training runs into storage" |
| 4 | Anthropic | Research Engineer, Reward Models Platform | ml-platform p017 | Partial — RL / fine-tuning platform work |
| 5 | Fireworks AI | Member of Technical Staff, Cloud Infrastructure | ml-platform p019 | Partial — "Job schedulers, resource managers, autoscalers" |
| 6 | Fireworks AI | Member of Technical Staff, Cluster Management | ml-platform p020 | Direct — "GPU cluster management" |
| 7 | Pinterest | Sr. Staff Software Engineer, ML Platform | ml-platform p028 | Direct — "Distributed training with GPUs and incremental training", "Large-scale model training for Ads, Home Feed, Recs" |
| 8 | Multiverse Computing | Senior MLOps Engineer (Training & Inference Optimization) | ml-platform p033 | Direct — "NVIDIA NeMo for distributed training", "KubeRay, MPI Operator, Run:ai", "CUDA, NCCL, Triton" |
| 9 | DoorDash | Senior Software Engineer, Machine Learning Infrastructure - Generative AI | mlops entry | Partial — LLM fine-tuning at production scale |
| 10 | JPMorgan Chase | Lead Machine Learning Engineer - MLOps | mlops entry | Direct — "distributed model training on large compute clusters" |
| 11 | Salesforce (Slack) | Software Engineer (Multiple Levels), Machine Learning Infrastructure - Slack | mlops entry | Direct — "Architect distributed training and data processing systems using platforms such as Ray, Airflow, and Spark" |
| 12 | NVIDIA | Senior MLOps Engineer, GenAI Framework | mlops entry | Direct — NVIDIA GenAI framework work covers training-framework contributions |
| 13 | Together AI | Machine Learning, Platform Engineer | mlops entry | Partial — training-as-a-service shop |
| 14 | Anthropic | ML Infrastructure Engineer, Safeguards | ml-platform p016 | Partial — "Distributed systems for high-throughput, low-latency workloads"; PyTorch/JAX |
| 15 | NVIDIA | Senior ML Platform Engineer | ml-platform p034 | Direct — IaC-driven ML platform engineering at NVIDIA (Megatron/NeMo shop) |

## Frontier-scale case-study anchor

Because posting evidence is limited to peer-role reuse, the curriculum leans on four fully-disclosed frontier-scale training reports to anchor the depth expectations:

- **Llama 3 Herd of Models** ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)) — most fully-disclosed frontier pretraining run of 2024: data pipeline, parallelism strategy, cluster shape, failure statistics, and MFU. Case study across mod-101, mod-105, mod-106, mod-109.
- **OPT-175B Logbook** ([facebookresearch/metaseq](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)) — canonical open reference for the operational reality of a frontier pretraining run (hardware failures, restarts, loss spikes). Anchor case study for mod-106 and mod-108.
- **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model** ([arXiv:2211.05100](https://arxiv.org/abs/2211.05100)) — Jean Zay cluster topology, data pipeline, and training-time incidents. Cited across mod-101, mod-105, mod-106.
- **MosaicML MPT-7B** ([mosaicml.com/mpt](https://www.mosaicml.com/mpt)) — cost-optimised sub-frontier pretraining recipe with StreamingDataset + FSDP. Anchor for mod-103 and mod-109.

The next autonomous research cycle should fan out across:

- `job-boards.greenhouse.io`, `jobs.lever.co`, `jobs.ashbyhq.com`, `boards.greenhouse.io`
- Frontier-lab careers pages: Anthropic, OpenAI, Google DeepMind, Meta AI (FAIR / GenAI), Mistral, xAI, Cohere, Reka, AI21, Character.AI, Databricks/Mosaic, NVIDIA, Together AI, Hugging Face, Snowflake, Salesforce Einstein / CodeGen, Amazon AGI/Bedrock, Microsoft AI Frontiers, Apple Foundation Models
- Scale-up training shops: Perplexity, Contextual, Writer, Cursor/Anysphere, Sakana, Poolside, Magic, Runway, Suno, Pika, Luma, Ideogram, Black Forest Labs
- GPU/infra providers: Lambda, CoreWeave, Crusoe, Nebius, Fireworks, RunPod, Modal, Anyscale, Foundry, Voltage Park, Vultr
- HPC-adjacent shops: NVIDIA NeMo, Intel Gaudi, AMD ROCm, Groq
- Product cos with training-infra teams: Reddit, Netflix, Uber, Airbnb, Pinterest, ByteDance/TikTok, Bloomberg AI

For each posting, capture employer, exact title, URL, `date_observed`, `posted_date` (or `estimated:2026-MM`), location, salary_range (only if verbatim; else null), 5–10 verbatim requirement bullets, 2–6 preferred-qualification bullets, and 1–3 short representative quotes. Filter OUT generic MLOps / ML Engineer / Inference-Serving / Data-Engineer / Research-Scientist titles (owned by peer tracks).

## Ownership map — quick reference for next cycle

When backfilling postings, use this ownership decision to keep the curriculum from drifting into peer territory:

- **Training Pipeline Engineer (this track, level 35)** owns the large-scale distributed-training platform end-to-end: distributed-training frameworks (FSDP/DeepSpeed/Megatron/NeMo/JAX), cluster orchestration for training (SLURM/Kueue/Volcano/KubeRay), training-scale data pipelines (WebDataset/StreamingDataset/Ray Data), networking & storage for training (NCCL/IB/RoCE/GPUDirect/Lustre/WEKA), checkpointing / fault tolerance / elastic training, throughput & MFU engineering, training-run observability, cluster economics, and training-platform architecture & cross-team leadership.
- **ML Engineer** (level 20) owns build-altitude PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals. Assumed prerequisite.
- **AI Infrastructure Engineer** (level 20) owns Docker / Kubernetes / cloud fundamentals. Assumed prerequisite.
- **Fine-Tuning Engineer** (level 30, peer) is the *customer* of this platform — SFT / PEFT / preference-optimization workflow.
- **AI Infrastructure ML Platform Engineer** (level 30, peer) owns the *serving / registry / model-lifecycle* platform surface — complementary, not overlapping.
- **AI Infrastructure MLOps Engineer** (level 30, peer) owns CI/CD-for-ML, model monitoring, and the run-to-production pipeline that sits *after* training completes.
- **AI Infrastructure Performance Engineer** (level 35, peer) owns kernel-level performance depth — CUDA kernels, FlashAttention internals, KV-cache micro-optimization. This track uses those primitives without authoring them.
- **AI Infrastructure Security Engineer** (level 35, peer) owns deep ML/AI security. This track surfaces awareness for training-data provenance, cluster boundaries, and adapter/checkpoint supply chain.
- **AI Governance Analyst** (level 25) / **AI Risk Engineer** (level 30) own compliance / dataset-licensing / model-card depth. This track surfaces awareness for pretraining-data governance and release review.
- **Staff / Principal ML Engineer**, **AI Infrastructure Senior/Principal Architect** — org-level scope above this specialist architect.

## Conclusion

<!-- needs-research: re-run on the next autonomous cycle with web tools exercised against live job boards, populate `postings` in `.aicg/job-requirements.json` with ≥25 direct postings, backfill the "Direct evidence" column in the requirement-themes table, and demote any requirement whose evidence_post_ids stays empty. -->

The curriculum plan in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json) is structured so that requirement frequencies can be added underneath each module without restructure. The themes themselves are grounded in the framework/paper canon the role is hired against — not invented — plus reused-with-attribution peer-role posting evidence for every requirement where such evidence exists.
