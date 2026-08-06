# Job Requirements — Training Pipeline Engineer

**Role level:** 35 (deep-specialist architect — owns the large-scale distributed-training platform)
**Track:** `training-pipeline-engineer-learning`
**Research window:** 2026-05-08 → 2026-08-06 (last ~90 days, with two Anthropic/xAI postings verified live on 2026-08-06 that were originally posted 2026-04-07 and 2026-03-21 respectively)
**Today:** 2026-08-06

This file documents the requirements catalog used to seed the Training Pipeline Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the planned curriculum lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json).

## Status — 2026-08 cycle, direct posting evidence now in place

The 2026-08 research cycle exercised WebSearch/WebFetch against live job boards and verified **23 direct Training-Pipeline-Engineer-titled (or equivalent) postings**, up from **0 in the 2026-07 bootstrap cycle**. Two shy of the 25-minimum target: additional matching postings existed at Netflix (Training Platform L4/L5), ByteDance (Seed Infra AI Training Systems), AMD (four training-focused Principal/Senior roles), Airbnb, and Fireworks AI — but were either recently removed, behind JS-only Ashby/LinkedIn/Workday walls, or returned 403 to WebFetch. Rather than pad with weakly-verified evidence (per the packet's *"PREFER ZERO ADDITIONS over weakly-justified ones"* rule), the sample stops at 23 and the residual gap is documented in `research_status.residual_needs_research` in [`.aicg/job-requirements.json`](.aicg/job-requirements.json).

The 15 reused-with-attribution peer-track postings from the bootstrap cycle are **retained** for continuity in the `postings` array; direct evidence lives alongside them in the new `direct_postings` array. Per-requirement `evidence_post_ids` now cite both `p-live-XX` (direct) and `p-reuse-XX` (peer).

## Methodology

1. Inventoried authoritative public references for the role's domain — see `authoritative_references` in `.aicg/job-requirements.json`:
   - Distributed-training framework docs: PyTorch FSDP / FSDP2, PyTorch DCP, torchtitan, DeepSpeed, Megatron-LM, NVIDIA NeMo, ColossalAI, JAX/Flax
   - Cluster / launcher docs: SLURM, Kueue, Volcano, KubeRay, TorchX, torch.distributed.elastic, Accelerate
   - Networking / storage docs: NCCL, NVLink/NVSwitch, InfiniBand, RoCEv2, Lustre, WEKA, FSx for Lustre, GPUDirect Storage, DGX SuperPOD reference architecture
   - Data-pipeline docs: WebDataset, MosaicML StreamingDataset, Ray Data, Arrow/Parquet, HF datasets
   - Performance references: FlashAttention (v1/v2/v3), Mixed Precision (Micikevicius 2017), NVIDIA Transformer Engine (FP8), `torch.compile`, OpenAI Triton, PaLM MFU
   - Systems / economics references: Chinchilla (Hoffmann 2022), Kaplan scaling laws (2020), MPT-7B, BLOOM, OPT-175B logbook, Llama 3 Herd of Models
2. Fanned out via WebSearch and WebFetch across two employer segments in parallel: (a) frontier labs + scale-up training shops + GPU/infra providers, and (b) product cos with training-infra teams + HPC-adjacent hardware + enterprise ML platforms.
3. Verified each posting by directly fetching the source page and capturing verbatim `required_skills`, `preferred_skills`, and 1–3 `key_quotes`. Excluded postings behind auth walls or returning HTTP errors to avoid fabricating content.
4. Mapped each requirement to (a) the role on our level ladder that should own it primarily and (b) the curriculum module that covers it.
5. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill, with higher levels linking back rather than duplicating fundamentals.
6. Recomputed direct-posting frequency per requirement and updated the coverage table below. Kept the reused peer-role postings in place for continuity, but direct evidence now takes primacy.
7. Applied the **continuity-bias rule** — default to no curriculum change. Reviewed emerging themes (post-training/RLHF infra, compiler-level performance, agent-driven cluster ops, Rust for cluster services) against the ≥3-posting-and-≥30%-frequency-and-no-existing-module-covers-it triple gate. **No theme cleared all three gates.**

## Requirement themes → curriculum ownership

| # | Theme | Direct frequency (N=23) | Reused peer evidence | Owner role | Coverage |
|---|---|---|---|---|---|
| 1 | Distributed-training frameworks (PyTorch FSDP/FSDP2, DeepSpeed ZeRO 1/2/3 + Infinity + Offload, Megatron-LM tensor/pipeline/sequence/expert parallel, NeMo, ColossalAI, JAX/Flax) | **13/23 (57%)** — Anthropic (×3), xAI, Databricks (×2), Character.AI, Magic, Anyscale, NVIDIA (×2), Uber, Reddit, Cerebras, Snap | 6 postings (`p-reuse-01, 07, 08, 09, 10, 11`) | `training-pipeline-engineer` (this) | [`mod-101-distributed-training-foundations`](lessons/mod-101-distributed-training-foundations), [`mod-102-training-frameworks-deep-dive`](lessons/mod-102-training-frameworks-deep-dive) |
| 2 | Training-scale data pipelines (WebDataset, MosaicML StreamingDataset, Ray Data, distributed tokenization, dedup at trillion-token scale) | **3/23 (13%)** — Anthropic Pretraining (Spark, web-scale, tokenization), Anthropic RE Pretraining (ETL preferred), Poolside (trillion-scale curation + dedup + curriculum) | 4 postings (`p-reuse-03, 10, 11, 14`) | `training-pipeline-engineer` | [`mod-103-training-scale-data-pipelines`](lessons/mod-103-training-scale-data-pipelines) |
| 3 | Cluster orchestration for training (SLURM, Kubernetes + Kueue / Volcano / KubeRay / MPI Operator, TorchX / torchrun / Ray Train launchers, gang scheduling; increasingly paired with Terraform/IaC) | **14/23 (61%)** — Anthropic Pretraining, Anthropic RE Pretraining, xAI Post-training, xAI Supercompute, Databricks, Character.AI, Magic Supercompute, Anyscale, NVIDIA Sr FM (SLURM+K8s explicit), Anthropic Cluster, Anthropic Research Infra, Reddit, Tenstorrent, Bloomberg, Snap | 5 postings (`p-reuse-01, 05, 06, 08, 12`) | `training-pipeline-engineer` | [`mod-104-cluster-orchestration-for-training`](lessons/mod-104-cluster-orchestration-for-training) |
| 4 | Networking & storage for training (NCCL, NVLink / NVSwitch, InfiniBand / RoCEv2, GPUDirect Storage, Lustre / WEKA / FSx, S3 staging) | **4/23 (17%)** — xAI Network Eng (100k+ GPU interconnects, PAM4, optics), Databricks (NVLink/IB/RoCE + collective comm), Character.AI (high-perf networking preferred), Magic Supercompute (networking + storage) | 2 postings (`p-reuse-05, 08`) | `training-pipeline-engineer` | [`mod-105-networking-storage-for-training`](lessons/mod-105-networking-storage-for-training) |
| 5 | Checkpointing, fault tolerance, elastic training (PyTorch DCP async / resharding, torchrun rendezvous, restart-from-crash, straggler / silent-corruption detection) | **6/23 (26%)** — Anthropic Pretraining (fault-tolerant systems), Databricks (checkpointing + failure detection + auto recovery), Character.AI (diagnose ML infra failures), Magic Pretraining (debug cross-layer + reproducibility), Magic Supercompute (long-running distributed jobs), Anyscale (fault-tolerant distributed), Anthropic Cluster (fault tolerance) | 0 postings (framework-doc + Llama 3 / OPT-175B / BLOOM anchor) | `training-pipeline-engineer` | [`mod-106-checkpointing-fault-tolerance-elastic-training`](lessons/mod-106-checkpointing-fault-tolerance-elastic-training) |
| 6 | Throughput & MFU engineering (FlashAttention v2/v3 integration, BF16 / FP8 mixed precision, activation / gradient checkpointing, `torch.compile` + Triton kernel integration, communication overlap, MFU as the throughput metric) | **5/23 (22%)** — Anthropic Pretraining Scaling ("performance optimization"), Databricks ("bottlenecks that govern training throughput and utilization"), NVIDIA Sr FM (GPU acceleration, CUDA), NVIDIA New Grad (analyze/profile/optimize training workloads, CUDA), Uber preferred (GPU/TPU training perf, Triton), Reddit (memory + GPU profiling), Cerebras (perf profiling, LLVM/MLIR compiler) | 1 posting (`p-reuse-08`) | `training-pipeline-engineer` | [`mod-107-throughput-and-mfu-engineering`](lessons/mod-107-throughput-and-mfu-engineering) |
| 7 | Training observability & reproducibility at scale (loss / gradient / throughput dashboards, DCGM GPU-health telemetry, W&B / TensorBoard / MLflow at pretraining scale, reproducibility bundles) | **5/23 (22%)** — Anthropic Pretraining (monitoring + observability preferred), Anthropic Pretraining Scaling (observability tools preferred), Character.AI (diagnose ML infra), Magic Pretraining (reproducibility under extreme scale), Magic Supercompute (observable + resilient), Poolside (evals tracking), Reddit (MLflow / Wandb explicit) | 2 postings (`p-reuse-03, 06`) | `training-pipeline-engineer` | [`mod-108-training-observability-and-reproducibility`](lessons/mod-108-training-observability-and-reproducibility) |
| 8 | Training economics (Chinchilla / Kaplan-scale budgeting, tokens-per-dollar, GPU-hours-per-run, reserved-vs-spot, feasibility-to-cluster-shape) | **3/23 (13%)** — Anthropic Pretraining Scaling (scaling laws preferred), Databricks (SLAs/SLOs for multi-tenant training), Poolside (scaling laws + data ablations) | 0 postings (scaling-laws / MPT-7B / Llama 3 economics anchor) | `training-pipeline-engineer` | [`mod-109-cost-capacity-and-cluster-economics`](lessons/mod-109-cost-capacity-and-cluster-economics) |
| 9 | Training-platform DX (launcher SDKs, run-templating, config-first ergonomics, researcher-as-customer thinking) | **5/23 (22%)** — xAI Post-training (user-friendly training + eval frameworks), Magic Pretraining (own the systems), Anyscale (ML training platforms), Uber Michelangelo (empower production teams), Anthropic Research Infra (partner directly with researchers), Reddit (advocate for platform users) | 2 postings (`p-reuse-01, 04`) | `training-pipeline-engineer` | [`mod-110-training-platform-architecture-and-leadership`](lessons/mod-110-training-platform-architecture-and-leadership) |
| 10 | Training-platform architecture & cross-team leadership at level 35 (multi-tenant cluster design, quota / preemption / fair-share, on-call for training runs, incident reviews, RFC authoring, hand-off contracts with peer platform tracks, migration strategy) | **9/23 (39%)** — Anthropic Pretraining (design + implement high-perf training infra), Anthropic RE Pretraining (scale training infra), xAI Supercompute (operating world's largest GPU supercomputing clusters), Databricks (drive architecture + evolution + mentor), Magic Pretraining + Magic Supercompute (own the systems + supercomputing platform architecture), NVIDIA (Tech Lead coordination), Uber (design/build scalable end-to-end training systems), Anthropic Cluster (leading complex multi-quarter initiatives across teams), Anthropic Research Infra (architectural decisions others build on), Reddit (undying advocate for platform users), Tenstorrent (large-scale AI infrastructure), Bloomberg (architect multi-tenant AI platform systems) | 3 postings (`p-reuse-02, 07, 15`) | `training-pipeline-engineer` | [`mod-110-training-platform-architecture-and-leadership`](lessons/mod-110-training-platform-architecture-and-leadership) + [`project-103-training-platform-capstone`](projects/project-103-training-platform-capstone) |
| 11 | PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals | n/a — prerequisite | n/a | `ml-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 12 | Docker / Kubernetes / cloud fundamentals + Terraform/IaC | n/a — prerequisite | n/a | `ai-infra-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught. Training-specific K8s (Kueue / Volcano / KubeRay / MPI Operator) IS owned here in mod-104 |
| 13 | SFT / PEFT / preference-optimization methods and the release workflow around them | n/a — peer track | n/a | `fine-tuning-engineer` (level 30) | Out of scope. Fine-tuning-engineer is the CUSTOMER of this platform. **Note:** RLHF/post-training INFRA (as seen in xAI's Post-training Infrastructure Engineer role) is a growing sub-specialization at frontier labs; watched next cycle but currently below the 3-posting threshold. |
| 14 | Model registry, inference serving (vLLM / TGI / TensorRT-LLM), LLM gateway | n/a — peer track | n/a | `ai-infra-ml-platform` (level 30) | Out of scope. ai-infra-ml-platform-engineer owns the serving surface |
| 15 | Kernel-level CUDA / FlashAttention-internal / KV-cache micro-optimization + compiler-level (MLIR / LLVM / TorchInductor) internals | n/a — peer track | n/a | `ai-infra-performance` (level 35) | Peer at same level. This track *uses* published kernels; ai-infra-performance *authors* them and owns the compiler stack. Cerebras Senior ML Systems Engineer (`p-live-20`) is the boundary-illustrative posting. |
| 16 | Deep ML/AI security (data poisoning, training-data exfiltration, adapter supply chain, secure model registry) | n/a — peer track | n/a | `ai-infra-security` (level 35) | Peer at same level. Surfaced as awareness in mod-105 and mod-110 |
| 17 | Pretraining-data licensing due-diligence, model-card authoring for release | n/a — peer track | n/a | `ai-governance-analyst` | Surfaced as awareness in mod-103 and mod-110 |

## Direct posting evidence — 2026-08 cycle

23 direct postings verified via WebFetch, spanning frontier labs, scale-up training shops, hardware vendors, and product cos with training-infra teams. Full verbatim records live in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) under `direct_postings[]`.

| # | Employer | Title | ID | Location | Comp band |
|---|---|---|---|---|---|
| 1 | Anthropic | Infrastructure Engineer, Pre-training | `p-live-01` | San Francisco, CA | $350K–$850K |
| 2 | Anthropic | Research Engineer, Pretraining Scaling — London | `p-live-02` | London, UK | £260K–£630K |
| 3 | Anthropic | Research Engineer / Research Scientist, Pre-training | `p-live-03` | SF / Seattle / NYC | $350K–$850K |
| 4 | xAI | Post-training Infrastructure Engineer | `p-live-04` | SF Bay Area | $180K–$440K |
| 5 | xAI | Member of Technical Staff — Infrastructure Supercompute | `p-live-05` | London, UK | — |
| 6 | xAI | Network Engineer — ML Infrastructure (High-Speed Interconnects) | `p-live-06` | Palo Alto, CA | $180K–$440K |
| 7 | Databricks | Staff Software Engineer, AI Runtime | `p-live-07` | Mountain View / SF, CA | $190K–$265K |
| 8 | Databricks (Mosaic) | ML Engineer, Foundation Models | `p-live-08` | SF / Remote | $230K–$340K |
| 9 | Character.AI | Machine Learning Infrastructure Engineer | `p-live-09` | Redwood City, CA | $150K–$350K |
| 10 | Magic | Member of Technical Staff, Pre-training Systems | `p-live-10` | San Francisco | $225K–$550K |
| 11 | Magic | Member of Technical Staff, Supercomputing Platform & Infrastructure | `p-live-11` | SF or Remote | $200K–$550K |
| 12 | Poolside | Member of Engineering (Pre-training / Data Research) | `p-live-12` | Remote (EMEA/East Coast) | — |
| 13 | Anyscale | Software Engineer — Model Training Infrastructure | `p-live-13` | Palo Alto / SF, CA | $170K–$237K + Equity |
| 14 | NVIDIA | Senior Research Engineer, Foundation Model Training Infrastructure | `p-live-14` | Santa Clara, CA | $224K–$357K |
| 15 | NVIDIA | High-Performance LLM Training Engineer, New Grad 2026 | `p-live-15` | Santa Clara, CA | $124K–$196K + equity |
| 16 | Uber | Staff Software Engineer — ML Michelangelo | `p-live-16` | Sunnyvale, CA | $232K–$258K |
| 17 | Anthropic | Staff+ Infrastructure Engineer, Cluster Infrastructure | `p-live-17` | London, UK | £325K–£485K |
| 18 | Anthropic | Software Engineer, Research Infrastructure | `p-live-18` | SF / NYC | $405K–$625K |
| 19 | Reddit | Staff Machine Learning Systems Engineer | `p-live-19` | Remote — US | $230K–$322K |
| 20 | Cerebras Systems | Senior ML Systems Engineer | `p-live-20` | Sunnyvale, CA | — |
| 21 | Tenstorrent | Staff Infrastructure Engineer — Models | `p-live-21` | Belgrade, Serbia | — |
| 22 | Bloomberg | Senior ML Platform Engineer — Artificial Intelligence | `p-live-22` | New York | $160K–$240K |
| 23 | Snap Inc. | Software Engineer, ML Infrastructure, Level 4 | `p-live-23` | Palo Alto / SF, CA (Hybrid) | $133K–$235K |

Notable observations from the direct sample:

- **Ownership language is remarkably uniform.** Anthropic Pretraining Scaling calls the role "highly operational"; Magic uses "own the systems that make large-scale pre-training stable and fast"; Databricks references SLAs/SLOs for multi-tenant training. These are production-critical infra ownership roles, not research-adjacent — consistent with the level-35 framing.
- **The training-vs-serving split is real but blurry at product companies.** Uber Michelangelo (`p-live-16`) and Reddit ML Systems (`p-live-19`) are training-first but touch serving; Snap L4 (`p-live-23`) and Bloomberg (`p-live-22`) blend training + inference. Frontier labs and hardware vendors give purer training-infra specialization.
- **Cluster orchestration + IaC is the highest-frequency requirement in the sample** (14/23, 61%) — driven partly by the rise of Terraform-based cluster provisioning (Magic Supercompute, Anthropic Cluster, xAI Supercompute, Reddit all cite it). IaC fluency itself is prerequisite (ai-infra-engineer); training-specific K8s (Kueue / Volcano / KubeRay / MPI Operator) is what mod-104 owns.
- **Comp bands are elite.** Verified bases span $124K (NVIDIA new grad) → $625K (Anthropic Research Infra top). Frontier-lab and hyperscaler top-of-band pay ~1.5–2× product-company equivalents at the same level (Anthropic $405–625K vs. Reddit $230–322K vs. Uber $232–258K vs. Bloomberg $160–240K).

## Emerging themes below threshold — watched, not added

Per the continuity-bias rule, an addition requires ≥3 distinct postings AND ≥30% frequency AND no existing module can be incrementally extended to cover it. These emerging themes surfaced in the direct sample but do NOT meet the triple gate:

| Theme | Direct postings | Assessment |
|---|---|---|
| Post-training / RLHF-training infra as a dedicated sub-specialization | 1/23 (xAI Post-training Infrastructure Engineer, `p-live-04`) | Emerging at frontier labs. RLHF algorithms belong to `fine-tuning-engineer` (peer track); post-training INFRA is at the boundary. Below the 3-posting floor — do NOT add a module. Watch next cycle. |
| Compiler-level performance (MLIR / LLVM / TorchInductor internals) | 2/23 (Cerebras `p-live-20`, Uber Triton preferred `p-live-16`) | Peer-track territory (`ai-infra-performance` level 35). Below floor, and this track uses (not authors) kernels. |
| Agent-driven / autonomous cluster lifecycle management | 1/23 (Anthropic Cluster `p-live-17`) | Very early signal — one Anthropic posting frames "agent-driven cluster lifecycle management" as strategic direction. Watch next cycle. |
| Rust for cluster services / IaC tooling | 3/23 (Anthropic Pretraining `p-live-01` required, xAI Supercompute `p-live-05` required, Anthropic Cluster `p-live-17` accepted alongside Go/Python) | Clears the 3-posting floor but falls to prerequisite tracks (ai-infra-engineer for systems languages). Not training-platform-specific competency. |
| Trillion-scale corpus curation + dedup + curriculum design | 2/23 (Anthropic Pretraining `p-live-01`, Poolside `p-live-12`) | Already covered in mod-103 (WebDataset, StreamingDataset, dedup with Ray Data). Below floor and no gap. |
| Multi-modal / long-context / MoE training specialization | 2/23 (Magic Pretraining long-context `p-live-10`, NVIDIA multi-modal FM `p-live-14`) | Already implicit in mod-101 (expert parallelism for MoE) and mod-107 (sequence packing). Below floor. |

## Frontier-scale case-study anchor

The curriculum continues to lean on four fully-disclosed frontier-scale training reports to anchor depth expectations. Direct posting evidence now backs the requirement themes; these open reports still supply the operational reality that no job posting fully articulates:

- **Llama 3 Herd of Models** ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)) — most fully-disclosed frontier pretraining run of 2024: data pipeline, parallelism strategy, cluster shape, failure statistics, and MFU. Case study across mod-101, mod-105, mod-106, mod-109.
- **OPT-175B Logbook** ([facebookresearch/metaseq](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)) — canonical open reference for the operational reality of a frontier pretraining run (hardware failures, restarts, loss spikes). Anchor case study for mod-106 and mod-108.
- **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model** ([arXiv:2211.05100](https://arxiv.org/abs/2211.05100)) — Jean Zay cluster topology, data pipeline, and training-time incidents. Cited across mod-101, mod-105, mod-106.
- **MosaicML MPT-7B** ([mosaicml.com/mpt](https://www.mosaicml.com/mpt)) — cost-optimised sub-frontier pretraining recipe with StreamingDataset + FSDP. Anchor for mod-103 and mod-109.

## Residual research

Next cycle should:

- Pick up 2+ additional direct postings to clear the 25-minimum target — good candidates are **Netflix Training Platform** L4/L5, **ByteDance Seed Infra AI Training Systems**, and **AMD Principal ML / Training Performance Optimization** (all confirmed to exist during this cycle but not verbatim-fetchable due to page-rendering constraints).
- Attempt an authenticated LinkedIn/Ashby pass to unlock the JS-walled roles at OpenAI, Together AI, Fireworks, and Runway.
- Re-verify the 90-day window for the two Anthropic/xAI postings that sit just outside a strict 90-day window (`p-live-01` posted 2026-04-07, `p-live-05` posted 2026-03-21) — they were verifiably live on 2026-08-06 but are ≈120–140 days old.
- Watch the post-training / RLHF-infra sub-specialization: xAI's dedicated role today (1/23) could well be 3+/25 next cycle if OpenAI, Anthropic, and Databricks all publish equivalents.

## Ownership map — quick reference

- **Training Pipeline Engineer (this track, level 35)** owns the large-scale distributed-training platform end-to-end: distributed-training frameworks (FSDP/DeepSpeed/Megatron/NeMo/JAX), cluster orchestration for training (SLURM/Kueue/Volcano/KubeRay), training-scale data pipelines (WebDataset/StreamingDataset/Ray Data), networking & storage for training (NCCL/IB/RoCE/GPUDirect/Lustre/WEKA), checkpointing / fault tolerance / elastic training, throughput & MFU engineering, training-run observability, cluster economics, and training-platform architecture & cross-team leadership.
- **ML Engineer** (level 20) owns build-altitude PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals. Assumed prerequisite.
- **AI Infrastructure Engineer** (level 20) owns Docker / Kubernetes / cloud fundamentals + Terraform / IaC. Assumed prerequisite.
- **Fine-Tuning Engineer** (level 30, peer) is the *customer* of this platform — SFT / PEFT / preference-optimization workflow. RLHF-training infra is a shared frontier; watched for future ownership shifts.
- **AI Infrastructure ML Platform Engineer** (level 30, peer) owns the *serving / registry / model-lifecycle* platform surface — complementary, not overlapping.
- **AI Infrastructure MLOps Engineer** (level 30, peer) owns CI/CD-for-ML, model monitoring, and the run-to-production pipeline that sits *after* training completes.
- **AI Infrastructure Performance Engineer** (level 35, peer) owns kernel-level performance depth — CUDA kernels, FlashAttention internals, KV-cache micro-optimization, MLIR/LLVM compiler internals. This track uses those primitives without authoring them.
- **AI Infrastructure Security Engineer** (level 35, peer) owns deep ML/AI security. This track surfaces awareness for training-data provenance, cluster boundaries, and adapter/checkpoint supply chain.
- **AI Governance Analyst** (level 25) / **AI Risk Engineer** (level 30) own compliance / dataset-licensing / model-card depth. This track surfaces awareness for pretraining-data governance and release review.
- **Staff / Principal ML Engineer**, **AI Infrastructure Senior/Principal Architect** — org-level scope above this specialist architect.

## Conclusion

Direct posting evidence now backs every existing requirement. Three requirements clear the ≥30% direct-frequency bar (cluster orchestration 61%, distributed-training frameworks 57%, platform architecture/leadership 39%); the other seven fall between 13–26% but are still cited by multiple direct postings, still owned by this track (no peer track owns them), and already baked into the mature 10-module curriculum. Under the continuity-bias rule, no requirement gap warrants a net-new module, exercise, or project this cycle — see [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json) for the empty-additions delta with rationale.
