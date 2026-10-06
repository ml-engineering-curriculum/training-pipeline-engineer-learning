# Job Requirements — Training Pipeline Engineer

**Role level:** 35 (deep-specialist architect — owns the large-scale distributed-training platform)
**Track:** `training-pipeline-engineer-learning`
**Research window:** 2026-07-08 → 2026-10-06 (last 90 days). Boundary postings outside the strict window are retained with explicit `window_note` fields: `p-live-01` (2026-04-07), `p-live-05` (2026-03-21), `p-live-08` (2026-03-31) carried over from 2026-09 observation; `p-live-38` (OpenAI Training Systems, 2026-06-11) and `p-live-40` (Anthropic RL Velocity, 2026-04-23) reconfirmed live this cycle via mirror sites.
**Today:** 2026-10-06

This file documents the requirements catalog used to seed the Training Pipeline Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the planned curriculum lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json).

## Status — 2026-10 cycle, direct evidence refreshed, no delta

The 2026-10 refresh added **7 net-new direct postings** on top of the 34 verified in 2026-09 (total: **41 direct Training-Pipeline-Engineer-titled or equivalent postings**), well above the 25-minimum target. The refresh closed the two residual employer gaps the prior cycle flagged — **Apple Pre-training Infrastructure** and **OpenAI** (two distinct roles: Research Infrastructure Engineer Training Systems and Training Performance Engineer) — and expanded evidence into the frontier-RL-infrastructure sub-cluster via **Anthropic Research Engineer, Machine Learning (RL Velocity)**, **Prime Intellect** (both RL Infrastructure and Distributed Training roles), and **Lightning AI** (Senior Research Engineer, LLM Training & Post-Training).

**Two more requirements crossed the ≥30% direct-frequency gate this cycle:** throughput/MFU engineering moved from 29% to 39% (+10pp), and checkpointing/fault-tolerance moved from 29% to 32% (+3pp). Both are already fully owned end-to-end by existing modules (mod-107 and mod-106 respectively) — no gap. Five requirements now clear the ≥30% bar in total (cluster orchestration 61%, distributed-training frameworks 61%, platform architecture/leadership 54%, throughput/MFU 39%, checkpointing 32%).

**One theme kept moving: post-training / RLHF-training infrastructure grew from 6/34 (17.6%) to 9/41 (22.0%)**, up 4pp and 3 absolute postings. The trajectory across the three sampled cycles is 1/23 (4.3%) → 6/34 (17.6%) → 9/41 (22.0%). It still does NOT clear the 30% frequency gate (would need ~4 more distinct direct postings against the current denominator) AND existing mod-102 (training frameworks, which already includes RLHF-relevant framework stack) plus mod-110 (platform architecture with hand-off contracts to fine-tuning-engineer as consumer) can be incrementally extended if the trend continues. Under the continuity-bias rule, this still does not warrant new content this cycle — see "Emerging themes" below. The curriculum remains stable at 10 modules + 3 projects = 315 hours.

Thinking Machines Lab briefly published three on-point infrastructure roles (Training Systems, RL Systems, Numerics) that were removed on 2026-09-07 before this cycle's observation window closed; they are captured in `dropped_or_unfetchable_this_cycle` as a market indicator but not counted toward the 41-posting total.

All 15 reused-with-attribution peer-track postings from prior cycles are retained in `postings` for continuity; direct posting evidence lives alongside them in `direct_postings`. Per-requirement `evidence_post_ids` cite both `p-live-XX` (direct) and `p-reuse-XX` (peer).

## Methodology

1. Inventoried authoritative public references for the role's domain — see `authoritative_references` in `.aicg/job-requirements.json` (unchanged this cycle):
   - Distributed-training framework docs: PyTorch FSDP / FSDP2, PyTorch DCP, torchtitan, DeepSpeed, Megatron-LM, NVIDIA NeMo, ColossalAI, JAX/Flax
   - Cluster / launcher docs: SLURM, Kueue, Volcano, KubeRay, TorchX, torch.distributed.elastic, Accelerate
   - Networking / storage docs: NCCL, NVLink/NVSwitch, InfiniBand, RoCEv2, Lustre, WEKA, FSx for Lustre, GPUDirect Storage, DGX SuperPOD reference architecture
   - Data-pipeline docs: WebDataset, MosaicML StreamingDataset, Ray Data, Arrow/Parquet, HF datasets
   - Performance references: FlashAttention (v1/v2/v3), Mixed Precision (Micikevicius 2017), NVIDIA Transformer Engine (FP8), `torch.compile`, OpenAI Triton, PaLM MFU
   - Systems / economics references: Chinchilla (Hoffmann 2022), Kaplan scaling laws (2020), MPT-7B, BLOOM, OPT-175B logbook, Llama 3 Herd of Models
2. Fanned out via WebSearch and WebFetch across the two employer gaps the 2026-09 cycle flagged (Apple, OpenAI) plus the frontier-RL-training-infrastructure sub-cluster (Anthropic RL Velocity, Prime Intellect, Thinking Machines Lab, Lightning AI, Cognition, Baseten).
3. Verified each new posting by directly fetching the source page (where accessible) or a documented mirror (freehire, builtin.com, builtinsf.com, dreamworkhq, General Catalyst job board) and capturing verbatim `required_skills`, `preferred_skills`, and 2–5 `key_quotes`. Excluded postings behind JS shells or returning HTTP errors unless a mirror carried the full content.
4. Captured three Thinking Machines Lab roles (Training Systems, RL Systems, Numerics) and Cognition / Baseten direct Ashby links that were either removed mid-window or returned JS-only shells. These sit in `dropped_or_unfetchable_this_cycle` and are NOT counted toward the 41-posting total — but their existence is a market indicator noted in the emerging-themes section.
5. Mapped each requirement to (a) the role on our level ladder that should own it primarily and (b) the curriculum module that covers it.
6. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill, with higher levels linking back rather than duplicating fundamentals.
7. Recomputed direct-posting frequency per requirement against the new N=41 denominator and updated the coverage table below.
8. Applied the **continuity-bias rule** — default to no curriculum change. Reviewed every emerging theme (post-training/RLHF-infra, agent-driven cluster ops, Blackwell-in-JD, bare-metal fleet ops, Rust for cluster services, trillion-scale curation, multi-modal/MoE, TPU-first training stacks) against the ≥3-posting-and-≥30%-frequency-and-no-existing-module-covers-it triple gate. **No theme cleared all three gates.**

## Requirement themes → curriculum ownership

| # | Theme | Direct frequency (N=41) | Δ vs 2026-09 (N=34) | Reused peer evidence | Owner role | Coverage |
|---|---|---|---|---|---|---|
| 1 | Distributed-training frameworks (PyTorch FSDP/FSDP2, DeepSpeed ZeRO 1/2/3 + Infinity + Offload, Megatron-LM tensor/pipeline/sequence/expert parallel, NeMo, ColossalAI, JAX/Flax) | **25/41 (61%)** | +8pp (was 53%) | 6 postings (`p-reuse-01, 07, 08, 09, 10, 11`) | `training-pipeline-engineer` (this) | [`mod-101-distributed-training-foundations`](lessons/mod-101-distributed-training-foundations), [`mod-102-training-frameworks-deep-dive`](lessons/mod-102-training-frameworks-deep-dive) |
| 2 | Training-scale data pipelines (WebDataset, MosaicML StreamingDataset, Ray Data, distributed tokenization, dedup at trillion-token scale) | **3/41 (7%)** | -2pp (was 9%; no new postings this cycle hit this theme) | 4 postings (`p-reuse-03, 10, 11, 14`) | `training-pipeline-engineer` | [`mod-103-training-scale-data-pipelines`](lessons/mod-103-training-scale-data-pipelines) |
| 3 | Cluster orchestration for training (SLURM, Kubernetes + Kueue / Volcano / KubeRay / MPI Operator, TorchX / torchrun / Ray Train launchers, gang scheduling; increasingly paired with Terraform/IaC) | **25/41 (61%)** | -4pp (was 65%) | 5 postings (`p-reuse-01, 05, 06, 08, 12`) | `training-pipeline-engineer` | [`mod-104-cluster-orchestration-for-training`](lessons/mod-104-cluster-orchestration-for-training) |
| 4 | Networking & storage for training (NCCL, NVLink / NVSwitch, InfiniBand / RoCEv2, GPUDirect Storage, Lustre / WEKA / FSx, S3 staging) | **10/41 (24%)** | +3pp (was 21%) | 2 postings (`p-reuse-05, 08`) | `training-pipeline-engineer` | [`mod-105-networking-storage-for-training`](lessons/mod-105-networking-storage-for-training) |
| 5 | Checkpointing, fault tolerance, elastic training (PyTorch DCP async / resharding, torchrun rendezvous, restart-from-crash, straggler / silent-corruption detection) | **13/41 (32%)** | +3pp (was 29%) — **NEW this cycle crosses the 30% gate** | 0 postings (framework-doc + Llama 3 / OPT-175B / BLOOM anchor) | `training-pipeline-engineer` | [`mod-106-checkpointing-fault-tolerance-elastic-training`](lessons/mod-106-checkpointing-fault-tolerance-elastic-training) |
| 6 | Throughput & MFU engineering (FlashAttention v2/v3 integration, BF16 / FP8 mixed precision, activation / gradient checkpointing, `torch.compile` + Triton kernel integration, communication overlap, MFU as the throughput metric) | **16/41 (39%)** | +10pp (was 29%) — **NEW this cycle crosses the 30% gate** | 1 posting (`p-reuse-08`) | `training-pipeline-engineer` | [`mod-107-throughput-and-mfu-engineering`](lessons/mod-107-throughput-and-mfu-engineering) |
| 7 | Training observability & reproducibility at scale (loss / gradient / throughput dashboards, DCGM GPU-health telemetry, W&B / TensorBoard / MLflow at pretraining scale, reproducibility bundles) | **11/41 (27%)** | +3pp (was 24%) | 2 postings (`p-reuse-03, 06`) | `training-pipeline-engineer` | [`mod-108-training-observability-and-reproducibility`](lessons/mod-108-training-observability-and-reproducibility) |
| 8 | Training economics (Chinchilla / Kaplan-scale budgeting, tokens-per-dollar, GPU-hours-per-run, reserved-vs-spot, feasibility-to-cluster-shape) | **4/41 (10%)** | -2pp (was 12%) | 0 postings (scaling-laws / MPT-7B / Llama 3 economics anchor) | `training-pipeline-engineer` | [`mod-109-cost-capacity-and-cluster-economics`](lessons/mod-109-cost-capacity-and-cluster-economics) |
| 9 | Training-platform DX (launcher SDKs, run-templating, config-first ergonomics, researcher-as-customer thinking) | **12/41 (29%)** | +3pp (was 26%) — just below the 30% line | 2 postings (`p-reuse-01, 04`) | `training-pipeline-engineer` | [`mod-110-training-platform-architecture-and-leadership`](lessons/mod-110-training-platform-architecture-and-leadership) |
| 10 | Training-platform architecture & cross-team leadership at level 35 (multi-tenant cluster design, quota / preemption / fair-share, on-call for training runs, incident reviews, RFC authoring, hand-off contracts with peer platform tracks, migration strategy) | **22/41 (54%)** | +10pp (was 44%) | 3 postings (`p-reuse-02, 07, 15`) | `training-pipeline-engineer` | [`mod-110-training-platform-architecture-and-leadership`](lessons/mod-110-training-platform-architecture-and-leadership) + [`project-103-training-platform-capstone`](projects/project-103-training-platform-capstone) |
| 11 | PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals | n/a — prerequisite | — | n/a | `ml-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 12 | Docker / Kubernetes / cloud fundamentals + Terraform/IaC | n/a — prerequisite | — | n/a | `ai-infra-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught. Training-specific K8s (Kueue / Volcano / KubeRay / MPI Operator) IS owned here in mod-104 |
| 13 | SFT / PEFT / preference-optimization methods and the release workflow around them | n/a — peer track | — | n/a | `fine-tuning-engineer` (level 30) | Out of scope. Fine-tuning-engineer is the CUSTOMER of this platform. **Note:** the post-training / RLHF-training INFRA sub-specialization (now 9/41 = 22.0%, up from 17.6% last cycle) is approaching the 30% gate — if it crosses next cycle, the smallest incremental response is an exercise addition to mod-110 or an mod-102 case study, NOT a new module. See "Emerging themes" below. |
| 14 | Model registry, inference serving (vLLM / TGI / TensorRT-LLM), LLM gateway | n/a — peer track | — | n/a | `ai-infra-ml-platform` (level 30) | Out of scope. ai-infra-ml-platform-engineer owns the serving surface |
| 15 | Kernel-level CUDA / FlashAttention-internal / KV-cache micro-optimization + compiler-level (MLIR / LLVM / TorchInductor) internals | n/a — peer track | — | n/a | `ai-infra-performance` (level 35) | Peer at same level. This track *uses* published kernels; ai-infra-performance *authors* them and owns the compiler stack. Cerebras (`p-live-20`), Uber Triton (`p-live-16`), AMD Principal ML Training Perf (`p-live-28`), and Apple's Pallas/XLA kernel-authoring preferred-skills (`p-live-37`) are the boundary-illustrative postings. |
| 16 | Deep ML/AI security (data poisoning, training-data exfiltration, adapter supply chain, secure model registry) | n/a — peer track | — | n/a | `ai-infra-security` (level 35) | Peer at same level. Surfaced as awareness in mod-105 and mod-110 |
| 17 | Pretraining-data licensing due-diligence, model-card authoring for release | n/a — peer track | — | n/a | `ai-governance-analyst` | Surfaced as awareness in mod-103 and mod-110 |

## Direct posting evidence — 2026-10 cycle refresh

41 direct postings verified. Full verbatim records live in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) under `direct_postings[]`. Records `p-live-25` and `p-live-35` are re-verifications of `p-live-01` and `p-live-12` and are NOT counted separately.

### Postings retained from 2026-09 (34)

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

### Net-new postings this cycle (7)

| # | Employer | Title | ID | Location | Comp band |
|---|---|---|---|---|---|
| 35 | Apple | AIML Staff ML Infrastructure Engineer, ML Platform & Technology — Pre-training Infrastructure | `p-live-37` | San Francisco Bay Area | — |
| 36 | OpenAI | Research Infrastructure Engineer, Training Systems | `p-live-38` | San Francisco, CA | $295K–$380K |
| 37 | OpenAI | Training Performance Engineer | `p-live-39` | San Francisco, CA (Hybrid) | $295K–$500K |
| 38 | Anthropic | Research Engineer, Machine Learning (RL Velocity) | `p-live-40` | SF / NYC / London | $500K–$850K |
| 39 | Prime Intellect | Research Engineer — RL Infrastructure | `p-live-41` | SF / Remote | $150K–$350K + equity |
| 40 | Prime Intellect | Research Engineer — Distributed Training | `p-live-42` | SF / Remote | $150K–$350K + equity |
| 41 | Lightning AI | Senior Research Engineer, LLM Training & Post-Training | `p-live-43` | NYC / Remote / SF / Seattle | $165K–$310K |

Notable observations from this cycle's refresh:

- **Two more requirements crossed the ≥30% direct-frequency gate.** Throughput/MFU engineering moved from 29% → 39% (the biggest frequency gain of the cycle), driven by Apple's JAX/XLA + MoE-kernel language, both OpenAI roles ('push the boundaries of throughput and uptime'), both Prime Intellect roles ('optimize training performance across kernels, memory movement, communication overhead'), and Lightning AI ('GPU performance optimization, mixed precision, memory optimization'). Checkpointing/fault-tolerance moved from 29% → 32%, driven by OpenAI's 'large-scale data loading and checkpointing systems' + 'reliability, debuggability' language and Anthropic RL Velocity's 'own the reliability and performance of research runs end-to-end'. **Both are already fully owned by mod-107 and mod-106 — no gap.**
- **Post-training / RLHF-training-infrastructure keeps growing.** Up from 6/34 (17.6%) in 2026-09 to 9/41 (22.0%) in 2026-10. The three new postings in the theme — Anthropic 'RL Velocity' as its own named platform-engineering surface, Prime Intellect RL Infrastructure as a distinct title from Prime Intellect Distributed Training, and Lightning AI's explicit 'LLM Training & Post-Training' title — represent a maturation from "post-training is a task bullet on a generic ML-infra role" to "post-training/RLHF-infrastructure is its own job title at frontier labs". Still 8pp below the 30% gate.
- **First OpenAI direct postings captured.** `p-live-38` Research Infrastructure Engineer, Training Systems and `p-live-39` Training Performance Engineer. Both close the 2026-09 OpenAI gap. The Training Systems role hits nine of the ten requirement themes owned by this track — parallel to Databricks `p-live-07` as a density-maximum anchor posting.
- **First Apple direct posting captured since 2026-07 bootstrap.** `p-live-37` AIML Staff ML Infrastructure Engineer, Pre-training Infrastructure. First direct posting in the sample to make TPU the primary accelerator target (vs an alternative mention), with JAX/XLA + Pallas kernel language. mod-102 (JAX branch) and mod-107 (MoE sequence packing) already cover the training-pipeline-engineer-level altitude; Pallas kernel authoring stays peer with ai-infra-performance.
- **Thinking Machines Lab briefly published three on-point infrastructure roles.** Research Engineer, Infrastructure, Training Systems (greenhouse); Research Engineer, Infrastructure, RL Systems (ashby); Research Engineer, Infrastructure, Numerics (ashby). All removed on 2026-09-07 before this cycle's observation window. Not counted toward the 41-posting total but captured in `dropped_or_unfetchable_this_cycle` as a market indicator of frontier-lab training-infrastructure team-building. Expected to re-post.
- **Comp bands continue to widen at frontier labs.** Anthropic RL Velocity opens a $500K–$850K band (US) and £370K–£630K (London). OpenAI Training Performance's $295K–$500K matches OpenAI Research Infrastructure's $295K–$380K ceiling-plus. Netflix's $466K–$750K (`p-live-24`) remains the widest product-company band.
- **No new Blackwell / agent-driven-ops / bare-metal-fleet-ops signals captured.** Cursor (`p-live-32`) remains the only Blackwell JD mention; Anthropic Cluster Infra + Together AI remain the only agent-driven-ops signals; Together AI + Lightning AI remain the only bare-metal-OOB signals. All below floor and unchanged.

## Emerging themes below threshold — watched, not added

Per the continuity-bias rule, an addition requires ≥3 distinct postings AND ≥30% frequency AND no existing module can be incrementally extended to cover it. These emerging themes surfaced but do NOT meet the triple gate:

| Theme | Direct postings | Assessment |
|---|---|---|
| **Post-training / RLHF-training infra as a dedicated sub-specialization** | **9/41 (22.0%)** — xAI Post-training Infrastructure Engineer (`p-live-04`), Airbnb Post Training named team (`p-live-26`), ByteDance Seed RL Systems & Infrastructure (`p-live-27`), Cursor ML Infra RL workloads (`p-live-32`), Cursor ML Research RL infra (`p-live-33`), Reka MTS post-training + RL (`p-live-36`), **Anthropic Research Engineer, Machine Learning (RL Velocity)** (`p-live-40`, new), **Prime Intellect Research Engineer, RL Infrastructure** (`p-live-41`, new), **Lightning AI Senior Research Engineer, LLM Training & Post-Training** (`p-live-43`, new) | **Grew from 6/34 (17.6%) → 9/41 (22.0%)**, up 4pp and 3 absolute postings. Trajectory across sampled cycles: 1/23 (4.3%) → 6/34 (17.6%) → 9/41 (22.0%). The newly captured Anthropic 'RL Velocity' role is the first frontier-lab role explicitly branded as RL-training-infrastructure-as-platform, parallel to xAI's Post-training Infrastructure Engineer. Frequency is still 8pp below the 30% gate AND existing mod-102 (training frameworks — includes RLHF-relevant frameworks) plus mod-110 (platform architecture with hand-off contracts to fine-tuning-engineer as consumer) can be incrementally extended if it keeps growing. **Watch next cycle (2026-11).** If it crosses 30% (would require ~4 more distinct direct postings against a likely 44–48 denominator), the smallest incremental response is one exercise addition to mod-110 (post-training platform hand-off contract) and/or an mod-102 case study on RL-specific framework patterns (rollout systems, asynchronous training pipelines) — NOT a new module. |
| Agent-driven / autonomous cluster lifecycle management | 2/41 (4.9%) — Anthropic Cluster Infrastructure (`p-live-17`), Together AI Bangalore (`p-live-30` with explicit "AI Infrastructure Agents") | Unchanged from prior cycle. No new postings require 'AI Infrastructure Agents' as a day-job deliverable this cycle. Still below the 3-posting floor. Watch next cycle. |
| Compiler-level performance (MLIR / LLVM / TorchInductor internals) as primary scope | 2/41 (4.9%) — Cerebras (`p-live-20`), Uber Triton preferred (`p-live-16`) | Apple pre-training (`p-live-37`) mentions JAX/XLA + Pallas/Triton kernel optimization in preferred-skills; Prime Intellect preferred mentions 'compiler or runtime optimization for ML systems'; OpenAI Training Perf preferred mentions 'ML compiler optimization'. All three are prerequisite-grade mentions, not primary scope — do not add to count. Peer-track territory (`ai-infra-performance` level 35). Below floor, and this track uses (not authors) kernels. |
| Blackwell (B100 / B200 / GB200) in JD required-skills text | 1/41 (2.4%) — Cursor ML Infrastructure (`p-live-32`) | No new Blackwell-specific JD language captured this cycle. Apple is TPU-focused; OpenAI/Anthropic/Prime Intellect roles are hardware-agnostic. mod-105 and mod-107 will absorb this as it grows without needing a new module. Watch next cycle. |
| Bare-metal / OOB fleet ops (BMC / Redfish / IPMI / PXE) in training-infra JDs | 2/41 (4.9%) — Together AI (`p-live-30`), Lightning AI (`p-live-31`) | Unchanged from prior cycle. Below the 3-posting floor and prerequisite territory (`ai-infra-engineer` owns hardware-lifecycle fluency). Do NOT add. |
| Rust for cluster services / IaC tooling | 5/41 (12%) — Anthropic Pretraining (`p-live-01`), xAI Supercompute (`p-live-05`), Anthropic Cluster (`p-live-17`), Cursor (`p-live-32`), **OpenAI Training Performance ('Rust or CUDA programming' preferred, `p-live-39`)** | +1 posting this cycle. Clears the 3-posting floor as a language requirement but falls to prerequisite tracks (`ai-infra-engineer` for systems languages). Not training-platform-specific competency. |
| Trillion-scale corpus curation + dedup + curriculum design | 2/41 (4.9%) — Anthropic Pretraining (`p-live-01`), Poolside (`p-live-12`) | Same 2 postings as last cycle. Already covered in mod-103 (WebDataset, StreamingDataset, dedup with Ray Data). Below floor and no gap. |
| Multi-modal / long-context / MoE training specialization | 5/41 (12%) — Magic long-context (`p-live-10`), NVIDIA multi-modal FM (`p-live-14`), Airbnb multimodal (`p-live-26`), Reka multimodal (`p-live-36`), **Apple ('attention and Mixture-of-Experts', `p-live-37`)** | +1 posting this cycle. 12% is well below 30% AND already implicit in mod-101 (EP for MoE) and mod-107 (sequence packing). Do NOT add. |
| TPU-first training stack (JAX / XLA / Pallas) as primary accelerator target | 1/41 (2.4%) — Apple AIML Staff ML Infra, Pre-training Infrastructure (`p-live-37`) | First time a direct Training-Pipeline-Engineer-titled posting in the sample has made TPU the primary accelerator target (vs an alternative mention). mod-102 already covers JAX/pjit/shard_map on 8 GPUs at training-pipeline-engineer altitude; TPU-specific kernels (Pallas) stay peer with ai-infra-performance. Watch next cycle. |

## Frontier-scale case-study anchor

The curriculum continues to lean on four fully-disclosed frontier-scale training reports to anchor depth expectations. Direct posting evidence backs the requirement themes; these open reports still supply the operational reality that no job posting fully articulates:

- **Llama 3 Herd of Models** ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)) — most fully-disclosed frontier pretraining run of 2024: data pipeline, parallelism strategy, cluster shape, failure statistics, and MFU. Case study across mod-101, mod-105, mod-106, mod-109.
- **OPT-175B Logbook** ([facebookresearch/metaseq](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)) — canonical open reference for the operational reality of a frontier pretraining run (hardware failures, restarts, loss spikes). Anchor case study for mod-106 and mod-108.
- **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model** ([arXiv:2211.05100](https://arxiv.org/abs/2211.05100)) — Jean Zay cluster topology, data pipeline, and training-time incidents. Cited across mod-101, mod-105, mod-106.
- **MosaicML MPT-7B** ([mosaicml.com/mpt](https://www.mosaicml.com/mpt)) — cost-optimised sub-frontier pretraining recipe with StreamingDataset + FSDP. Anchor for mod-103 and mod-109.

## Residual research

Next cycle should:

- Re-verify the three oldest retained postings (`p-live-01` 2026-04-07, `p-live-05` 2026-03-21, `p-live-08` 2026-03-31) that are now 180+ days past `posted_date`. If any have dropped, cite the newer equivalents (Anthropic Pretraining has posted several times through 2025–2026; Databricks/Mosaic frequently re-opens the Foundation Models role).
- Track the post-training / RLHF-training-infra theme (now 9/41, 22.0%). If it crosses 30% frequency by the 2026-11 cycle, the smallest incremental response is a mod-110 exercise on the post-training platform hand-off contract and/or a mod-102 case study on RL-specific framework patterns (rollout systems, asynchronous training pipelines) — NOT a new module.
- Re-attempt the Thinking Machines Lab training-infrastructure roles when they republish (Training Systems, RL Systems, Numerics). If they return and remain live for the full observation window, they would push the post-training theme ~2 postings higher.
- Attempt an authenticated LinkedIn / Ashby / Workday pass to unlock Cognition's direct Ashby links (Post-Training, ML Infrastructure), Baseten's Training Infrastructure, Runway, Character.AI (Ashby JS), and any additional AMD Principal roles behind Workday.
- Watch Blackwell adoption in JD required-skills text — still 1 posting (Cursor) this cycle. Likely to grow as B200/GB200 systems ship at scale.
- Watch "AI Infrastructure Agents" as a day-job deliverable — unchanged at 2 postings this cycle. If it crosses the 3-posting floor, mod-108 (observability) and mod-110 (platform architecture) should surface awareness of the pattern.

## Ownership map — quick reference

- **Training Pipeline Engineer (this track, level 35)** owns the large-scale distributed-training platform end-to-end: distributed-training frameworks (FSDP/DeepSpeed/Megatron/NeMo/JAX), cluster orchestration for training (SLURM/Kueue/Volcano/KubeRay), training-scale data pipelines (WebDataset/StreamingDataset/Ray Data), networking & storage for training (NCCL/IB/RoCE/GPUDirect/Lustre/WEKA), checkpointing / fault tolerance / elastic training, throughput & MFU engineering, training-run observability, cluster economics, and training-platform architecture & cross-team leadership.
- **ML Engineer** (level 20) owns build-altitude PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals. Assumed prerequisite.
- **AI Infrastructure Engineer** (level 20) owns Docker / Kubernetes / cloud fundamentals + Terraform / IaC + hardware lifecycle (BMC / Redfish / IPMI / PXE). Assumed prerequisite.
- **Fine-Tuning Engineer** (level 30, peer) is the *customer* of this platform — SFT / PEFT / preference-optimization workflow. RLHF-training infra is a shared frontier; watched for future ownership shifts.
- **AI Infrastructure ML Platform Engineer** (level 30, peer) owns the *serving / registry / model-lifecycle* platform surface — complementary, not overlapping.
- **AI Infrastructure MLOps Engineer** (level 30, peer) owns CI/CD-for-ML, model monitoring, and the run-to-production pipeline that sits *after* training completes.
- **AI Infrastructure Performance Engineer** (level 35, peer) owns kernel-level performance depth — CUDA kernels, FlashAttention internals, KV-cache micro-optimization, MLIR/LLVM compiler internals, Pallas/XLA kernel authoring. This track uses those primitives without authoring them.
- **AI Infrastructure Security Engineer** (level 35, peer) owns deep ML/AI security. This track surfaces awareness for training-data provenance, cluster boundaries, and adapter/checkpoint supply chain.
- **AI Governance Analyst** (level 25) / **AI Risk Engineer** (level 30) own compliance / dataset-licensing / model-card depth. This track surfaces awareness for pretraining-data governance and release review.
- **Staff / Principal ML Engineer**, **AI Infrastructure Senior/Principal Architect** — org-level scope above this specialist architect.

## Conclusion

Direct posting evidence backs every existing requirement. **Five requirements now clear the ≥30% direct-frequency bar** (cluster orchestration 61%, distributed-training frameworks 61%, platform architecture/leadership 54%, throughput/MFU engineering 39% NEW, and checkpointing/fault-tolerance 32% NEW); platform-DX sits one posting below the line at 29%. The remaining four fall between 7–27% but are still cited by multiple direct postings, still owned by this track (no peer track owns them), and already baked into the mature 10-module curriculum. **The post-training / RLHF-training-infra theme grew to 9/41 (22.0%), up from 17.6% last cycle and 4.3% two cycles ago. It still does not clear the 30% frequency gate AND can be incrementally addressed by extending mod-102 or mod-110 if it keeps growing.** Under the continuity-bias rule, no requirement gap warrants a net-new module, exercise, or project this cycle — see [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json) for the empty-additions delta with rationale.
