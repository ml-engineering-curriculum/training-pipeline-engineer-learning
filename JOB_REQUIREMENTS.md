# Job Requirements — Training Pipeline Engineer

**Role level:** 35 (deep-specialist architect — owns the large-scale distributed-training platform)
**Track:** `training-pipeline-engineer-learning`
**Research window:** 2026-06-08 → 2026-09-06 (last 90 days). Three postings (`p-live-01` posted 2026-04-07, `p-live-05` posted 2026-03-21, `p-live-08` posted 2026-03-31) sit just outside the strict window but were re-verified live on 2026-09-06.
**Today:** 2026-09-06

This file documents the requirements catalog used to seed the Training Pipeline Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the planned curriculum lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json).

## Status — 2026-09 cycle, direct evidence refreshed, no delta

The 2026-09 refresh added **11 net-new direct postings** on top of the 23 verified in 2026-08 (total: **34 direct Training-Pipeline-Engineer-titled or equivalent postings**), clearing the 25-minimum target. The refresh explicitly closed the four employer gaps the prior cycle flagged — **Netflix Training Platform, Airbnb Post Training, ByteDance Seed RL Systems & Infrastructure, and AMD Principal ML Training Performance Optimization** — plus picked up new evidence at Perplexity, Together AI (Bangalore), Lightning AI, Cursor (two roles), Poolside (Experiment Platform), and Reka.

**One theme moved: post-training / RLHF-training infrastructure as its own specialization cleared the 3-posting floor for the first time (6/34, up from 1/23 last cycle).** It still does NOT clear the 30% frequency gate AND existing mod-102/mod-110 can be incrementally extended if it grows. Under the continuity-bias rule, this does not warrant new content this cycle — see the "Emerging themes below threshold — watched, not added" section below. The curriculum remains stable at 10 modules + 3 projects = 315 hours.

All 15 reused-with-attribution peer-track postings from prior cycles are retained in `postings` for continuity; direct posting evidence lives alongside them in `direct_postings`. Per-requirement `evidence_post_ids` cite both `p-live-XX` (direct) and `p-reuse-XX` (peer).

## Methodology

1. Inventoried authoritative public references for the role's domain — see `authoritative_references` in `.aicg/job-requirements.json` (unchanged this cycle):
   - Distributed-training framework docs: PyTorch FSDP / FSDP2, PyTorch DCP, torchtitan, DeepSpeed, Megatron-LM, NVIDIA NeMo, ColossalAI, JAX/Flax
   - Cluster / launcher docs: SLURM, Kueue, Volcano, KubeRay, TorchX, torch.distributed.elastic, Accelerate
   - Networking / storage docs: NCCL, NVLink/NVSwitch, InfiniBand, RoCEv2, Lustre, WEKA, FSx for Lustre, GPUDirect Storage, DGX SuperPOD reference architecture
   - Data-pipeline docs: WebDataset, MosaicML StreamingDataset, Ray Data, Arrow/Parquet, HF datasets
   - Performance references: FlashAttention (v1/v2/v3), Mixed Precision (Micikevicius 2017), NVIDIA Transformer Engine (FP8), `torch.compile`, OpenAI Triton, PaLM MFU
   - Systems / economics references: Chinchilla (Hoffmann 2022), Kaplan scaling laws (2020), MPT-7B, BLOOM, OPT-175B logbook, Llama 3 Herd of Models
2. Fanned out via WebSearch and WebFetch across the employer gap list from the 2026-08 cycle (Netflix, Airbnb, ByteDance Seed, AMD, Perplexity) plus the broader frontier-lab + scale-up-training-shop + hardware-vendor + product-co segments.
3. Verified each new posting by directly fetching the source page and capturing verbatim `required_skills`, `preferred_skills`, and 1–3 `key_quotes`. Excluded postings behind auth walls or returning HTTP errors to avoid fabricating content (see the residual-research section).
4. Re-verified two prior-cycle postings (`p-live-01` Anthropic Pre-training and `p-live-12` Poolside Pre-training) as still-live on 2026-09-06; recorded them as `p-live-25` and `p-live-35` with `duplicate_of` links so they don't double-count.
5. Mapped each requirement to (a) the role on our level ladder that should own it primarily and (b) the curriculum module that covers it.
6. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill, with higher levels linking back rather than duplicating fundamentals.
7. Recomputed direct-posting frequency per requirement against the new N=34 denominator and updated the coverage table below.
8. Applied the **continuity-bias rule** — default to no curriculum change. Reviewed every emerging theme (post-training/RLHF-infra, agent-driven cluster ops, Blackwell-in-JD, bare-metal fleet ops, Rust for cluster services, trillion-scale curation, multi-modal/MoE) against the ≥3-posting-and-≥30%-frequency-and-no-existing-module-covers-it triple gate. **No theme cleared all three gates.**

## Requirement themes → curriculum ownership

| # | Theme | Direct frequency (N=34) | Δ vs 2026-08 (N=23) | Reused peer evidence | Owner role | Coverage |
|---|---|---|---|---|---|---|
| 1 | Distributed-training frameworks (PyTorch FSDP/FSDP2, DeepSpeed ZeRO 1/2/3 + Infinity + Offload, Megatron-LM tensor/pipeline/sequence/expert parallel, NeMo, ColossalAI, JAX/Flax) | **18/34 (53%)** | -4pp (was 57%) | 6 postings (`p-reuse-01, 07, 08, 09, 10, 11`) | `training-pipeline-engineer` (this) | [`mod-101-distributed-training-foundations`](lessons/mod-101-distributed-training-foundations), [`mod-102-training-frameworks-deep-dive`](lessons/mod-102-training-frameworks-deep-dive) |
| 2 | Training-scale data pipelines (WebDataset, MosaicML StreamingDataset, Ray Data, distributed tokenization, dedup at trillion-token scale) | **3/34 (9%)** | -4pp (was 13%; no new postings this cycle hit this theme, so denominator growth pulled the ratio down) | 4 postings (`p-reuse-03, 10, 11, 14`) | `training-pipeline-engineer` | [`mod-103-training-scale-data-pipelines`](lessons/mod-103-training-scale-data-pipelines) |
| 3 | Cluster orchestration for training (SLURM, Kubernetes + Kueue / Volcano / KubeRay / MPI Operator, TorchX / torchrun / Ray Train launchers, gang scheduling; increasingly paired with Terraform/IaC) | **22/34 (65%)** | +4pp (was 61%) | 5 postings (`p-reuse-01, 05, 06, 08, 12`) | `training-pipeline-engineer` | [`mod-104-cluster-orchestration-for-training`](lessons/mod-104-cluster-orchestration-for-training) |
| 4 | Networking & storage for training (NCCL, NVLink / NVSwitch, InfiniBand / RoCEv2, GPUDirect Storage, Lustre / WEKA / FSx, S3 staging) | **7/34 (21%)** | +4pp (was 17%) | 2 postings (`p-reuse-05, 08`) | `training-pipeline-engineer` | [`mod-105-networking-storage-for-training`](lessons/mod-105-networking-storage-for-training) |
| 5 | Checkpointing, fault tolerance, elastic training (PyTorch DCP async / resharding, torchrun rendezvous, restart-from-crash, straggler / silent-corruption detection) | **10/34 (29%)** | +3pp (was 26%) — now just below the 30% line | 0 postings (framework-doc + Llama 3 / OPT-175B / BLOOM anchor) | `training-pipeline-engineer` | [`mod-106-checkpointing-fault-tolerance-elastic-training`](lessons/mod-106-checkpointing-fault-tolerance-elastic-training) |
| 6 | Throughput & MFU engineering (FlashAttention v2/v3 integration, BF16 / FP8 mixed precision, activation / gradient checkpointing, `torch.compile` + Triton kernel integration, communication overlap, MFU as the throughput metric) | **10/34 (29%)** | +7pp (was 22%) — now just below the 30% line | 1 posting (`p-reuse-08`) | `training-pipeline-engineer` | [`mod-107-throughput-and-mfu-engineering`](lessons/mod-107-throughput-and-mfu-engineering) |
| 7 | Training observability & reproducibility at scale (loss / gradient / throughput dashboards, DCGM GPU-health telemetry, W&B / TensorBoard / MLflow at pretraining scale, reproducibility bundles) | **8/34 (24%)** | +2pp (was 22%) | 2 postings (`p-reuse-03, 06`) | `training-pipeline-engineer` | [`mod-108-training-observability-and-reproducibility`](lessons/mod-108-training-observability-and-reproducibility) |
| 8 | Training economics (Chinchilla / Kaplan-scale budgeting, tokens-per-dollar, GPU-hours-per-run, reserved-vs-spot, feasibility-to-cluster-shape) | **4/34 (12%)** | -1pp (was 13%) | 0 postings (scaling-laws / MPT-7B / Llama 3 economics anchor) | `training-pipeline-engineer` | [`mod-109-cost-capacity-and-cluster-economics`](lessons/mod-109-cost-capacity-and-cluster-economics) |
| 9 | Training-platform DX (launcher SDKs, run-templating, config-first ergonomics, researcher-as-customer thinking) | **9/34 (26%)** | +4pp (was 22%) | 2 postings (`p-reuse-01, 04`) | `training-pipeline-engineer` | [`mod-110-training-platform-architecture-and-leadership`](lessons/mod-110-training-platform-architecture-and-leadership) |
| 10 | Training-platform architecture & cross-team leadership at level 35 (multi-tenant cluster design, quota / preemption / fair-share, on-call for training runs, incident reviews, RFC authoring, hand-off contracts with peer platform tracks, migration strategy) | **15/34 (44%)** | +5pp (was 39%) | 3 postings (`p-reuse-02, 07, 15`) | `training-pipeline-engineer` | [`mod-110-training-platform-architecture-and-leadership`](lessons/mod-110-training-platform-architecture-and-leadership) + [`project-103-training-platform-capstone`](projects/project-103-training-platform-capstone) |
| 11 | PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals | n/a — prerequisite | — | n/a | `ml-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 12 | Docker / Kubernetes / cloud fundamentals + Terraform/IaC | n/a — prerequisite | — | n/a | `ai-infra-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught. Training-specific K8s (Kueue / Volcano / KubeRay / MPI Operator) IS owned here in mod-104 |
| 13 | SFT / PEFT / preference-optimization methods and the release workflow around them | n/a — peer track | — | n/a | `fine-tuning-engineer` (level 30) | Out of scope. Fine-tuning-engineer is the CUSTOMER of this platform. **Note:** the post-training / RLHF-training INFRA sub-specialization (as seen at xAI, Airbnb, ByteDance Seed, Cursor, Reka this cycle) crossed the 3-posting floor for the first time; still below 30% and can be absorbed by mod-102/mod-110 if it grows. See "Emerging themes" below. |
| 14 | Model registry, inference serving (vLLM / TGI / TensorRT-LLM), LLM gateway | n/a — peer track | — | n/a | `ai-infra-ml-platform` (level 30) | Out of scope. ai-infra-ml-platform-engineer owns the serving surface |
| 15 | Kernel-level CUDA / FlashAttention-internal / KV-cache micro-optimization + compiler-level (MLIR / LLVM / TorchInductor) internals | n/a — peer track | — | n/a | `ai-infra-performance` (level 35) | Peer at same level. This track *uses* published kernels; ai-infra-performance *authors* them and owns the compiler stack. Cerebras (`p-live-20`), Uber Triton (`p-live-16`), and AMD Principal ML Training Perf (`p-live-28`) are the boundary-illustrative postings. |
| 16 | Deep ML/AI security (data poisoning, training-data exfiltration, adapter supply chain, secure model registry) | n/a — peer track | — | n/a | `ai-infra-security` (level 35) | Peer at same level. Surfaced as awareness in mod-105 and mod-110 |
| 17 | Pretraining-data licensing due-diligence, model-card authoring for release | n/a — peer track | — | n/a | `ai-governance-analyst` | Surfaced as awareness in mod-103 and mod-110 |

## Direct posting evidence — 2026-09 cycle refresh

34 direct postings verified via WebFetch. Full verbatim records live in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) under `direct_postings[]`. Records `p-live-25` and `p-live-35` are re-verifications of `p-live-01` and `p-live-12` and are NOT counted separately.

### Postings retained from 2026-08 (23)

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

### Net-new postings this cycle (11)

| # | Employer | Title | ID | Location | Comp band |
|---|---|---|---|---|---|
| 24 | Netflix | Software Engineer L4/L5 — Training Platform | `p-live-24` | Remote, USA | $466K–$750K |
| 25 | Airbnb | Senior Staff Machine Learning Engineer, Post Training | `p-live-26` | Remote, USA | $248K–$310K |
| 26 | ByteDance (Seed Infra) | Research Engineer — Reinforcement Learning (RL) Systems & Infrastructure | `p-live-27` | San Jose, CA | $254K–$480K |
| 27 | AMD | Principal ML Engineer — Large Scale Training Performance Optimization | `p-live-28` | San Jose, CA | $148K–$216K (est.) |
| 28 | Perplexity | Member of Technical Staff (AI Infrastructure Engineer) | `p-live-29` | San Francisco, CA | — |
| 29 | Together AI | AI Infrastructure Systems Engineer (Bangalore) | `p-live-30` | Bangalore, India | — |
| 30 | Lightning AI | Senior Infrastructure Software Engineer | `p-live-31` | NYC / SF / Seattle / London | $180K–$220K |
| 31 | Cursor (Anysphere) | Software Engineer, ML Infrastructure | `p-live-32` | SF / NYC | — |
| 32 | Cursor (Anysphere) | Software Engineer, ML Research (Research Engineer) | `p-live-33` | SF / NYC | — |
| 33 | Poolside | Member of Engineering (Experiment Platform) | `p-live-34` | Remote (EMEA/East Coast) | — |
| 34 | Reka AI | Member of Technical Staff (GPU Performance Engineer) | `p-live-36` | Singapore | — |

Notable observations from this cycle's refresh:

- **Post-training is now a named team, not just a task bullet.** Airbnb (`p-live-26`) titles its role "Senior Staff Machine Learning Engineer, **Post Training**"; ByteDance Seed (`p-live-27`) publishes "Research Engineer — Reinforcement Learning (RL) Systems & Infrastructure" as a discrete title. Combined with xAI's Post-training Infrastructure Engineer (`p-live-04`), this is now a 6-posting theme (17.6% of sample). Not enough for a new module, but a real trend.
- **Blackwell first appears in JD required-skills text.** Cursor ML Infrastructure (`p-live-32`) lists "Nvidia GPUs with Infiniband or RoCE, particularly with **Blackwell and Hopper-class hardware**" as preferred. First appearance in the sample. mod-105 and mod-107 will absorb this as it grows.
- **"AI Infrastructure Agents" as a day-job deliverable.** Together AI (`p-live-30`) explicitly requires building "AI Infrastructure Agents that automate deployment, root-cause failures, incident triage, and autonomous remediation." First time this appears as a JD requirement (not a strategic direction as in Anthropic Cluster Infra's `p-live-17` last cycle). Still only 2 postings; below floor.
- **Bare-metal / OOB fleet-ops language surfacing at neo-cloud training providers.** Lightning AI (`p-live-31`) and Together AI (`p-live-30`) both demand BMC / Redfish / IPMI / PXE / VAST-storage fluency. Prerequisite territory (ai-infra-engineer for hardware lifecycle) — do NOT amplify here.
- **Comp bands widening.** Netflix opens a $466K–$750K Training Platform band (widest in the sample); Airbnb's Senior Staff Post Training is $248K–$310K; Cursor and Perplexity list no bands. Frontier-lab and hyperscaler top-of-band pay continues to run ~1.5–2× product-company equivalents at the same level.
- **Torchtitan / async DCP / FP8-with-Transformer-Engine still absent from JD required-skills text.** These appear in industry blogs, PyTorch conference talks, and NVIDIA docs, but not yet in the hiring bar. mod-102, mod-106, and mod-107 continue to cover them at framework-doc altitude.

## Emerging themes below threshold — watched, not added

Per the continuity-bias rule, an addition requires ≥3 distinct postings AND ≥30% frequency AND no existing module can be incrementally extended to cover it. These emerging themes surfaced but do NOT meet the triple gate:

| Theme | Direct postings | Assessment |
|---|---|---|
| **Post-training / RLHF-training infra as a dedicated sub-specialization** | **6/34 (17.6%)** — xAI Post-training Infrastructure Engineer (`p-live-04`), Airbnb Post Training named team (`p-live-26`), ByteDance Seed RL Systems & Infrastructure (`p-live-27`), Cursor ML Infra RL workloads (`p-live-32`), Cursor ML Research RL infra (`p-live-33`), Reka MTS post-training + RL (`p-live-36`) | **Cleared the 3-posting floor for the first time this cycle (was 1/23 in 2026-08).** Frequency 17.6% is well below the 30% gate, AND existing mod-102 (training frameworks — includes RLHF-relevant frameworks) plus mod-110 (platform architecture with hand-off contracts to fine-tuning-engineer as consumer) can be incrementally extended if it grows. **Watch next cycle.** If it crosses 30% (would require ~4 more distinct postings), the smallest incremental response is one exercise addition to mod-110 (post-training platform hand-off contract) — NOT a new module. |
| Agent-driven / autonomous cluster lifecycle management | 2/34 — Anthropic Cluster Infrastructure (`p-live-17`), Together AI Bangalore (`p-live-30` with explicit "AI Infrastructure Agents") | Trending up (1→2 postings) but still below the 3-posting floor. Together AI is the first to require it as a day-job deliverable rather than a strategic direction. Watch next cycle. |
| Compiler-level performance (MLIR / LLVM / TorchInductor internals) | 2/34 — Cerebras (`p-live-20`), Uber Triton preferred (`p-live-16`) | Unchanged from prior cycle — peer-track territory (`ai-infra-performance` level 35). Below floor, and this track uses (not authors) kernels. AMD Principal ML Training Perf (`p-live-28`) is kernel/GPU-programming, not compiler-internals, so does not add to this count. |
| Blackwell (B100 / B200 / GB200) in JD required-skills text | 1/34 — Cursor ML Infrastructure (`p-live-32`) | First appearance in the sample. Well below floor but a marker for the hardware-generation transition. mod-105 (networking/storage) and mod-107 (throughput/MFU) will absorb this as it grows without needing a new module. Watch next cycle. |
| Bare-metal / OOB fleet ops (BMC / Redfish / IPMI / PXE) in training-infra JDs | 2/34 — Together AI (`p-live-30`), Lightning AI (`p-live-31`) | New signal this cycle at neo-cloud training providers. Below the 3-posting floor and prerequisite territory (`ai-infra-engineer` owns hardware-lifecycle fluency). Do NOT add. |
| Rust for cluster services / IaC tooling | 4/34 — Anthropic Pretraining (`p-live-01`), xAI Supercompute (`p-live-05`), Anthropic Cluster (`p-live-17`), Cursor (`p-live-32`) | Clears the 3-posting floor as a language requirement but falls to prerequisite tracks (`ai-infra-engineer` for systems languages). Not training-platform-specific competency. |
| Trillion-scale corpus curation + dedup + curriculum design | 2/34 — Anthropic Pretraining (`p-live-01`), Poolside Pre-training (`p-live-12`) | Same 2 postings re-verified this cycle. Already covered in mod-103 (WebDataset, StreamingDataset, dedup with Ray Data). Below floor and no gap. |
| Multi-modal / long-context / MoE training specialization | 4/34 — Magic long-context (`p-live-10`), NVIDIA multi-modal FM (`p-live-14`), Airbnb multimodal (`p-live-26`), Reka multimodal (`p-live-36`) | Cleared the 3-posting floor this cycle (2→4) but at 12% is well below 30% AND already implicit in mod-101 (EP for MoE) and mod-107 (sequence packing). Do NOT add. |

## Frontier-scale case-study anchor

The curriculum continues to lean on four fully-disclosed frontier-scale training reports to anchor depth expectations. Direct posting evidence backs the requirement themes; these open reports still supply the operational reality that no job posting fully articulates:

- **Llama 3 Herd of Models** ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)) — most fully-disclosed frontier pretraining run of 2024: data pipeline, parallelism strategy, cluster shape, failure statistics, and MFU. Case study across mod-101, mod-105, mod-106, mod-109.
- **OPT-175B Logbook** ([facebookresearch/metaseq](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)) — canonical open reference for the operational reality of a frontier pretraining run (hardware failures, restarts, loss spikes). Anchor case study for mod-106 and mod-108.
- **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model** ([arXiv:2211.05100](https://arxiv.org/abs/2211.05100)) — Jean Zay cluster topology, data pipeline, and training-time incidents. Cited across mod-101, mod-105, mod-106.
- **MosaicML MPT-7B** ([mosaicml.com/mpt](https://www.mosaicml.com/mpt)) — cost-optimised sub-frontier pretraining recipe with StreamingDataset + FSDP. Anchor for mod-103 and mod-109.

## Residual research

Next cycle should:

- Track the post-training / RLHF-training-infra theme (now 6/34, 17.6%). If it crosses 30% frequency by the 2026-10 cycle, the smallest incremental response is a mod-110 exercise on the post-training platform hand-off contract — NOT a new module. Emphatically do NOT preemptively add.
- Re-verify the three prior-cycle postings whose `posted_date` sits outside a strict 90-day window (`p-live-01` 2026-04-07, `p-live-05` 2026-03-21, `p-live-08` 2026-03-31); if they've dropped, cite the newer equivalents that will have appeared by then.
- Attempt an authenticated LinkedIn / Ashby / Workday pass to unlock the JS-walled roles at OpenAI (Research Infrastructure, Compute Infrastructure), Runway, Character.AI (Ashby JS), Apple (expired IDs; check for renewed listings), and the remaining AMD Principal ML positions.
- Watch Blackwell adoption in JD required-skills text — surfaced in 1 posting this cycle (Cursor); likely to grow as B200/GB200 systems ship.
- Watch "AI Infrastructure Agents" as a day-job deliverable (Together AI is the first) — if this becomes a common requirement, mod-108 (observability) and mod-110 (platform architecture) should surface awareness of the pattern.

## Ownership map — quick reference

- **Training Pipeline Engineer (this track, level 35)** owns the large-scale distributed-training platform end-to-end: distributed-training frameworks (FSDP/DeepSpeed/Megatron/NeMo/JAX), cluster orchestration for training (SLURM/Kueue/Volcano/KubeRay), training-scale data pipelines (WebDataset/StreamingDataset/Ray Data), networking & storage for training (NCCL/IB/RoCE/GPUDirect/Lustre/WEKA), checkpointing / fault tolerance / elastic training, throughput & MFU engineering, training-run observability, cluster economics, and training-platform architecture & cross-team leadership.
- **ML Engineer** (level 20) owns build-altitude PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals. Assumed prerequisite.
- **AI Infrastructure Engineer** (level 20) owns Docker / Kubernetes / cloud fundamentals + Terraform / IaC + hardware lifecycle (BMC / Redfish / IPMI / PXE). Assumed prerequisite.
- **Fine-Tuning Engineer** (level 30, peer) is the *customer* of this platform — SFT / PEFT / preference-optimization workflow. RLHF-training infra is a shared frontier; watched for future ownership shifts.
- **AI Infrastructure ML Platform Engineer** (level 30, peer) owns the *serving / registry / model-lifecycle* platform surface — complementary, not overlapping.
- **AI Infrastructure MLOps Engineer** (level 30, peer) owns CI/CD-for-ML, model monitoring, and the run-to-production pipeline that sits *after* training completes.
- **AI Infrastructure Performance Engineer** (level 35, peer) owns kernel-level performance depth — CUDA kernels, FlashAttention internals, KV-cache micro-optimization, MLIR/LLVM compiler internals. This track uses those primitives without authoring them.
- **AI Infrastructure Security Engineer** (level 35, peer) owns deep ML/AI security. This track surfaces awareness for training-data provenance, cluster boundaries, and adapter/checkpoint supply chain.
- **AI Governance Analyst** (level 25) / **AI Risk Engineer** (level 30) own compliance / dataset-licensing / model-card depth. This track surfaces awareness for pretraining-data governance and release review.
- **Staff / Principal ML Engineer**, **AI Infrastructure Senior/Principal Architect** — org-level scope above this specialist architect.

## Conclusion

Direct posting evidence backs every existing requirement. Three requirements clear the ≥30% direct-frequency bar (cluster orchestration 65%, distributed-training frameworks 53%, platform architecture/leadership 44%); two more (checkpointing / fault tolerance and throughput / MFU) crossed to 29% this cycle — just below the line. The remaining five fall between 9–26% but are still cited by multiple direct postings, still owned by this track (no peer track owns them), and already baked into the mature 10-module curriculum. **The post-training / RLHF-training-infra theme cleared the 3-posting floor for the first time (6/34, 17.6%) but does not clear the 30% frequency gate and can be incrementally addressed by extending mod-102 or mod-110 if it grows further.** Under the continuity-bias rule, no requirement gap warrants a net-new module, exercise, or project this cycle — see [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json) for the empty-additions delta with rationale.
