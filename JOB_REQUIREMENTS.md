# Job Requirements — Training Pipeline Engineer

**Role level:** 35 (deep-specialist architect — owns the large-scale distributed-training platform)
**Track:** `training-pipeline-engineer-learning`
**Research window:** 2026-08-05 → 2026-11-03 (last 90 days). Boundary postings outside the strict window are retained with explicit `window_note` fields: `p-live-01` (2026-04-07), `p-live-05` (2026-03-21), `p-live-08` (2026-03-31) carried from the 2026-09 observation; `p-live-38` (OpenAI Training Systems, 2026-06-11) and `p-live-40` (Anthropic RL Velocity, 2026-04-23) retained from 2026-10; new boundary retentions this cycle: `p-live-47` (Reflection AI Pre-Training Infra, 2026-03-24 posted, 2026-10-28 updated) and `p-live-49` (GM Staff ML Training Infra, 2026-06-05 posted but still live via Workday).
**Today:** 2026-11-03

This file documents the requirements catalog used to seed the Training Pipeline Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the planned curriculum lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json).

## Status — 2026-11 cycle, direct evidence refreshed, no delta

The 2026-11 refresh added **10 net-new direct postings** on top of the 41 verified in 2026-10 (total: **51 direct Training-Pipeline-Engineer-titled or equivalent postings**), well above the 25-minimum target. The most consequential movement is **Thinking Machines Lab's full infrastructure team republishing** — Training Systems (`p-live-44`), Infrastructure RL Systems (`p-live-45`), and Software Engineer Supercomputing (`p-live-46`) are all back live after the 2026-09-07 removal flagged in last cycle's `dropped_or_unfetchable_this_cycle`. The refresh also captured first direct evidence from **Reflection AI** (`p-live-47` Pre-Training Infra), **Microsoft AI** (`p-live-48` Member of Technical Staff, Pre-Training Infrastructure), **General Motors** (`p-live-49` Staff + `p-live-50` Senior ML Training Infrastructure), **FAR AI** (`p-live-51` Senior Software Engineer, GPU Cluster Infrastructure), **NVIDIA Senior DL SW Infrastructure Engineer** (`p-live-52`, fresh 2026-10-31 posting), and a second Anthropic pre-training-platform role (`p-live-53` **Pre-training Data Infrastructure Engineer**, distinct from `p-live-01`).

**No new requirements crossed the ≥30% gate this cycle, but one requirement is on the brink:** req-training-networking-storage climbed from 24% → 29% (+5pp — biggest frequency gain of the cycle), now just 1pp below the gate. Driven by TML Supercomputing (`p-live-46`), Reflection AI's NCCL required-skill, Microsoft AI's InfiniBand/NVLink preference, FAR AI's VAST/Weka/Lustre/Ceph enumeration, and NVIDIA Senior DL SW Infra's parallel-filesystems required-skill. mod-105 fully owns this scope — no gap even if it crosses next cycle. The five requirements that cleared the gate in prior cycles stay above it: cluster orchestration 65% (+4pp), distributed-training frameworks 65% (+4pp), platform architecture/leadership 63% (+9pp — **largest absolute gain this cycle**; all 10 net-new postings cite it), throughput/MFU engineering 39% (flat), and checkpointing/fault-tolerance 33% (+1pp).

**Post-training / RLHF-training-infrastructure DECELERATED this cycle.** Grew from 9/41 (22.0%) in 2026-10 to 10/51 (19.6%) in 2026-11 — up only 1 absolute posting (TML Infrastructure RL Systems returned from the dropped list) while the denominator grew by 10. Trajectory across four sampled cycles: 1/23 (4.3%) → 6/34 (17.6%) → 9/41 (22.0%) → 10/51 (19.6%). The 10 net-new postings this cycle were overwhelmingly pre-training-focused rather than post-training-focused, which temporarily reversed the trend. Still above the 3-posting floor but 10pp below the 30% frequency gate AND existing mod-102 + mod-110 can be incrementally extended if it re-accelerates. Under the continuity-bias rule, this still does not warrant new content this cycle — see "Emerging themes" below. The curriculum remains stable at **10 modules + 3 projects = 315 hours**.

**req-training-data-pipelines just crossed the 3-posting direct-floor** via the new Anthropic Pre-training Data Infrastructure role (`p-live-53` adds explicit tokenization, deduplication, and web-scale corpus pipeline language at the $500K-$850K band). Now 6/51 (12%), up from 3/41 (7%) — the biggest proportional gain in the catalog. Signals that pretraining-data-engineering is becoming its own staffing line at frontier labs. Still well below the 30% gate; mod-103 fully owns the scope.

AMD briefly posted a Post-Training Platform Infrastructure Engineer (closed 2026-10-16) that is scope-wise inference / P-D-disaggregation territory (peer with `ai-infra-ml-platform`), so NOT counted — logged in `dropped_or_unfetchable_this_cycle` for signal only. Prime Intellect `p-live-41` RL Infrastructure was removed 2026-10-16 but reposted with a near-identical scope ~2026-10-19 at the same company — retained with a status note.

All 15 reused-with-attribution peer-track postings from prior cycles are retained in `postings` for continuity; direct posting evidence lives alongside them in `direct_postings`. Per-requirement `evidence_post_ids` cite both `p-live-XX` (direct) and `p-reuse-XX` (peer).

## Methodology

1. Inventoried authoritative public references for the role's domain — see `authoritative_references` in `.aicg/job-requirements.json` (unchanged this cycle):
   - Distributed-training framework docs: PyTorch FSDP / FSDP2, PyTorch DCP, torchtitan, DeepSpeed, Megatron-LM, NVIDIA NeMo, ColossalAI, JAX/Flax
   - Cluster / launcher docs: SLURM, Kueue, Volcano, KubeRay, TorchX, torch.distributed.elastic, Accelerate
   - Networking / storage docs: NCCL, NVLink/NVSwitch, InfiniBand, RoCEv2, Lustre, WEKA, FSx for Lustre, GPUDirect Storage, DGX SuperPOD reference architecture
   - Data-pipeline docs: WebDataset, MosaicML StreamingDataset, Ray Data, Arrow/Parquet, HF datasets
   - Performance references: FlashAttention (v1/v2/v3), Mixed Precision (Micikevicius 2017), NVIDIA Transformer Engine (FP8), `torch.compile`, OpenAI Triton, PaLM MFU
   - Systems / economics references: Chinchilla (Hoffmann 2022), Kaplan scaling laws (2020), MPT-7B, BLOOM, OPT-175B logbook, Llama 3 Herd of Models
2. Fanned out via WebSearch and WebFetch across (a) the Thinking Machines Lab republish signal flagged in last cycle, (b) new frontier-lab employers absent from the 2026-10 sample (Reflection AI, Microsoft AI, FAR AI, Radical Numerics, Trajectory, Preference Model), (c) direct equivalents outside the frontier-lab + hyperscaler clusters (General Motors — new automotive-vertical direct Training-Pipeline-Engineer-titled postings), and (d) fresh NVIDIA/Anthropic re-postings observable in-window.
3. Verified each new posting by directly fetching the source page (where accessible) or a documented mirror (freehire, builtin.com, builtinsf.com, dreamworkhq, Welcome to the Jungle, simplify.jobs) and capturing verbatim `required_skills`, `preferred_skills`, and 2–5 `key_quotes`. Excluded postings behind JS shells or returning HTTP errors unless a mirror carried the full content.
4. Captured smaller-shop postings (Trajectory MTS ML Infra, Radical Numerics MTS Infrastructure and Training Systems, Hark MTS Pretraining, Nuance Labs MTS Pretraining, Preference Model MTS Research Engineer Post-training, xAI MTS Pre-training Data Infrastructure) that returned JS-only shells or mirror-only snippets this cycle. These sit in `dropped_or_unfetchable_this_cycle` and are NOT counted toward the 51-posting total — but their existence is a market indicator noted in the emerging-themes section.
5. Marked **AMD Post-Training Platform Infrastructure Engineer** (closed 2026-10-16) as NOT counted because its scope is inference / P-D-disaggregation (peer-track territory for `ai-infra-ml-platform`), despite the role title.
6. Marked **Prime Intellect `p-live-41` RL Infrastructure** as removed 2026-10-16 but retained its slot because the shop reposted a near-identical scope at a different Ashby ID ~2026-10-19.
7. Mapped each new posting to (a) the role on our level ladder that should own it primarily and (b) the curriculum module that covers it.
8. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill, with higher levels linking back rather than duplicating fundamentals.
9. Recomputed direct-posting frequency per requirement against the new N=51 denominator and updated the coverage table below.
10. Applied the **continuity-bias rule** — default to no curriculum change. Reviewed every emerging theme against the ≥3-posting-and-≥30%-frequency-and-no-existing-module-covers-it triple gate. **No theme cleared all three gates.**

## Requirement themes → curriculum ownership

| # | Theme | Direct frequency (N=51) | Δ vs 2026-10 (N=41) | Reused peer evidence | Owner role | Coverage |
|---|---|---|---|---|---|---|
| 1 | Distributed-training frameworks (PyTorch FSDP/FSDP2, DeepSpeed ZeRO 1/2/3 + Infinity + Offload, Megatron-LM tensor/pipeline/sequence/expert parallel, NeMo, ColossalAI, JAX/Flax) | **33/51 (65%)** | +4pp (was 61%) | 6 postings (`p-reuse-01, 07, 08, 09, 10, 11`) | `training-pipeline-engineer` (this) | [`mod-101-distributed-training-foundations`](lessons/mod-101-distributed-training-foundations), [`mod-102-training-frameworks-deep-dive`](lessons/mod-102-training-frameworks-deep-dive) |
| 2 | Training-scale data pipelines (WebDataset, MosaicML StreamingDataset, Ray Data, distributed tokenization, dedup at trillion-token scale) | **6/51 (12%)** | +5pp (was 7%) — **biggest proportional gain in catalogue**; crossed 3-posting direct-floor | 4 postings (`p-reuse-03, 10, 11, 14`) | `training-pipeline-engineer` | [`mod-103-training-scale-data-pipelines`](lessons/mod-103-training-scale-data-pipelines) |
| 3 | Cluster orchestration for training (SLURM, Kubernetes + Kueue / Volcano / KubeRay / MPI Operator, TorchX / torchrun / Ray Train launchers, gang scheduling; increasingly paired with Terraform/IaC) | **33/51 (65%)** | +4pp (was 61%) | 5 postings (`p-reuse-01, 05, 06, 08, 12`) | `training-pipeline-engineer` | [`mod-104-cluster-orchestration-for-training`](lessons/mod-104-cluster-orchestration-for-training) |
| 4 | Networking & storage for training (NCCL, NVLink / NVSwitch, InfiniBand / RoCEv2, GPUDirect Storage, Lustre / WEKA / FSx, S3 staging) | **15/51 (29%)** | +5pp (was 24%) — **1pp below the 30% gate** | 2 postings (`p-reuse-05, 08`) | `training-pipeline-engineer` | [`mod-105-networking-storage-for-training`](lessons/mod-105-networking-storage-for-training) |
| 5 | Checkpointing, fault tolerance, elastic training (PyTorch DCP async / resharding, torchrun rendezvous, restart-from-crash, straggler / silent-corruption detection) | **17/51 (33%)** | +1pp (was 32%) — stays above the 30% gate | 0 postings (framework-doc + Llama 3 / OPT-175B / BLOOM anchor) | `training-pipeline-engineer` | [`mod-106-checkpointing-fault-tolerance-elastic-training`](lessons/mod-106-checkpointing-fault-tolerance-elastic-training) |
| 6 | Throughput & MFU engineering (FlashAttention v2/v3 integration, BF16 / FP8 mixed precision, activation / gradient checkpointing, `torch.compile` + Triton kernel integration, communication overlap, MFU as the throughput metric) | **20/51 (39%)** | +0pp (was 39%) — stays above the 30% gate | 1 posting (`p-reuse-08`) | `training-pipeline-engineer` | [`mod-107-throughput-and-mfu-engineering`](lessons/mod-107-throughput-and-mfu-engineering) |
| 7 | Training observability & reproducibility at scale (loss / gradient / throughput dashboards, DCGM GPU-health telemetry, W&B / TensorBoard / MLflow at pretraining scale, reproducibility bundles) | **14/51 (27%)** | +0pp (was 27%) | 2 postings (`p-reuse-03, 06`) | `training-pipeline-engineer` | [`mod-108-training-observability-and-reproducibility`](lessons/mod-108-training-observability-and-reproducibility) |
| 8 | Training economics (Chinchilla / Kaplan-scale budgeting, tokens-per-dollar, GPU-hours-per-run, reserved-vs-spot, feasibility-to-cluster-shape) | **4/51 (8%)** | -2pp (was 10%; same 4 postings, growing denominator) | 0 postings (scaling-laws / MPT-7B / Llama 3 economics anchor) | `training-pipeline-engineer` | [`mod-109-cost-capacity-and-cluster-economics`](lessons/mod-109-cost-capacity-and-cluster-economics) |
| 9 | Training-platform DX (launcher SDKs, run-templating, config-first ergonomics, researcher-as-customer thinking) | **14/51 (27%)** | -2pp (was 29%) — new cycle's additions were platform-architecture-heavy rather than researcher-DX-heavy | 2 postings (`p-reuse-01, 04`) | `training-pipeline-engineer` | [`mod-110-training-platform-architecture-and-leadership`](lessons/mod-110-training-platform-architecture-and-leadership) |
| 10 | Training-platform architecture & cross-team leadership at level 35 (multi-tenant cluster design, quota / preemption / fair-share, on-call for training runs, incident reviews, RFC authoring, hand-off contracts with peer platform tracks, migration strategy) | **32/51 (63%)** | +9pp (was 54%) — **largest absolute gain**; ALL 10 net-new postings cite platform-architecture | 3 postings (`p-reuse-02, 07, 15`) | `training-pipeline-engineer` | [`mod-110-training-platform-architecture-and-leadership`](lessons/mod-110-training-platform-architecture-and-leadership) + [`project-103-training-platform-capstone`](projects/project-103-training-platform-capstone) |
| 11 | PyTorch / classical ML / FastAPI / Docker / MLflow fundamentals | n/a — prerequisite | — | n/a | `ml-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 12 | Docker / Kubernetes / cloud fundamentals + Terraform/IaC | n/a — prerequisite | — | n/a | `ai-infra-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught. Training-specific K8s (Kueue / Volcano / KubeRay / MPI Operator) IS owned here in mod-104 |
| 13 | SFT / PEFT / preference-optimization methods and the release workflow around them | n/a — peer track | — | n/a | `fine-tuning-engineer` (level 30) | Out of scope. Fine-tuning-engineer is the CUSTOMER of this platform. **Note:** the post-training / RLHF-training INFRA sub-specialization **DECELERATED** this cycle to 10/51 (19.6%), DOWN from 22% at N=41 last cycle. See "Emerging themes" below. |
| 14 | Model registry, inference serving (vLLM / TGI / TensorRT-LLM), LLM gateway | n/a — peer track | — | n/a | `ai-infra-ml-platform` (level 30) | Out of scope. ai-infra-ml-platform-engineer owns the serving surface. **Note:** AMD's Post-Training Platform Infrastructure Engineer posting (closed 2026-10-16) sat in this peer-track scope despite its title. |
| 15 | Kernel-level CUDA / FlashAttention-internal / KV-cache micro-optimization + compiler-level (MLIR / LLVM / TorchInductor) internals | n/a — peer track | — | n/a | `ai-infra-performance` (level 35) | Peer at same level. This track *uses* published kernels; ai-infra-performance *authors* them and owns the compiler stack. The newly-republished Thinking Machines Lab Infrastructure Kernels + Infrastructure Numerics roles are peer-track territory even though they are at a frontier-lab shop. |
| 16 | Deep ML/AI security (data poisoning, training-data exfiltration, adapter supply chain, secure model registry) | n/a — peer track | — | n/a | `ai-infra-security` (level 35) | Peer at same level. Surfaced as awareness in mod-105 and mod-110. FAR AI's `p-live-51` 'cluster security (admission control, RBAC, sandboxed runtimes, autonomous agent isolation)' is boundary-illustrative — mod-104 surfaces awareness; depth stays with ai-infra-security. |
| 17 | Pretraining-data licensing due-diligence, model-card authoring for release | n/a — peer track | — | n/a | `ai-governance-analyst` | Surfaced as awareness in mod-103 and mod-110 |

## Direct posting evidence — 2026-11 cycle refresh

51 direct postings verified. Full verbatim records live in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) under `direct_postings[]`. Records `p-live-25` and `p-live-35` are re-verifications of `p-live-01` and `p-live-12` and are NOT counted separately.

### Postings retained from 2026-09 and 2026-10 (34)

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

### Postings retained from 2026-10 cycle refresh (7)

| # | Employer | Title | ID | Location | Comp band |
|---|---|---|---|---|---|
| 35 | Apple | AIML Staff ML Infrastructure Engineer, ML Platform & Technology — Pre-training Infrastructure | `p-live-37` | San Francisco Bay Area | — |
| 36 | OpenAI | Research Infrastructure Engineer, Training Systems | `p-live-38` | San Francisco, CA | $295K–$380K |
| 37 | OpenAI | Training Performance Engineer | `p-live-39` | San Francisco, CA (Hybrid) | $295K–$500K |
| 38 | Anthropic | Research Engineer, Machine Learning (RL Velocity) | `p-live-40` | SF / NYC / London | $500K–$850K |
| 39 | Prime Intellect | Research Engineer — RL Infrastructure | `p-live-41` | SF / Remote | $150K–$350K + equity (removed 2026-10-16, reposted ~2026-10-19 at new Ashby ID with same scope) |
| 40 | Prime Intellect | Research Engineer — Distributed Training | `p-live-42` | SF / Remote | $150K–$350K + equity |
| 41 | Lightning AI | Senior Research Engineer, LLM Training & Post-Training | `p-live-43` | NYC / Remote / SF / Seattle | $165K–$310K |

### Net-new postings this cycle (10)

| # | Employer | Title | ID | Location | Comp band |
|---|---|---|---|---|---|
| 42 | Thinking Machines Lab | Research Engineer, Infrastructure, Training Systems | `p-live-44` | San Francisco, CA | $350K–$475K |
| 43 | Thinking Machines Lab | Research Engineer, Infrastructure, RL Systems | `p-live-45` | San Francisco, CA | $350K–$475K |
| 44 | Thinking Machines Lab | Software Engineer, Supercomputing | `p-live-46` | San Francisco, CA | $350K–$475K |
| 45 | Reflection AI | Member of Technical Staff — Pre-Training Infra | `p-live-47` | San Francisco, CA (On-site) | — |
| 46 | Microsoft AI | Member of Technical Staff, Pre-Training Infrastructure | `p-live-48` | Mountain View, CA | $188K–$304.2K (SF/NYC) / $142.8K–$274.8K (standard) |
| 47 | General Motors | Staff Machine Learning Engineer — ML Training Infrastructure | `p-live-49` | Austin, TX / Mountain View, CA / Sunnyvale, CA | — |
| 48 | General Motors | Senior Machine Learning Engineer — ML Training Infrastructure | `p-live-50` | Sunnyvale, CA | — |
| 49 | FAR AI | Senior Software Engineer — GPU Cluster Infrastructure | `p-live-51` | Remote (USA preferred) / Berkeley, CA / Singapore | $150K–$275K |
| 50 | NVIDIA | Senior Deep Learning Software Infrastructure Engineer | `p-live-52` | Santa Clara, CA (Remote-friendly) | $224K–$431K (L5-L6) |
| 51 | Anthropic | Pre-training Data Infrastructure Engineer | `p-live-53` | San Francisco, CA | $500K–$850K |

Notable observations from this cycle's refresh:

- **Thinking Machines Lab is back.** Last cycle's `dropped_or_unfetchable_this_cycle` flagged three TML infrastructure roles removed 2026-09-07. All three are now republished — Training Systems and RL Systems on Ashby, plus a new Software Engineer Supercomputing role. Combined with the Kernels + Numerics + Inference roles that stay peer-track (ai-infra-performance / ai-infra-ml-platform), TML now has one of the most clearly-split training-infrastructure team structures in the sample. All three new TML roles sit at the $350K–$475K level, confirming the frontier-startup band narrative.
- **Platform architecture / cross-team leadership is the dominant growth vector.** +9pp (54% → 63%) this cycle, with ALL 10 net-new postings citing it. This is now the SECOND-densest requirement in the catalog, closing on cluster-orchestration and distributed-training-frameworks at 65% each.
- **Networking & storage climbed to within 1pp of the gate.** 24% → 29%, driven by TML Supercomputing (`p-live-46`), Reflection AI's NCCL requirement, Microsoft AI's InfiniBand/NVLink preference, FAR AI's exhaustive storage-fabric enumeration (VAST/Weka/Lustre/Ceph), and NVIDIA Senior DL SW Infra's parallel-filesystems requirement. mod-105 fully owns this scope.
- **New employers to the catalogue — Reflection AI, Microsoft AI, General Motors, FAR AI.** Reflection AI's $6.3B October 2026 funding round positions it as the next frontier-lab tier; Microsoft AI (the Suleyman org, not MS Research) now visibly staffs pre-training-infra as its own line; GM is the first automotive-vertical direct Training-Pipeline-Engineer-titled posting at both Staff and Senior levels; FAR AI's non-profit AI-safety-research posting is unusually explicit on multi-tenancy + fault-tolerance language.
- **req-training-data-pipelines just crossed the 3-posting direct-floor.** The new Anthropic Pre-training Data Infrastructure Engineer (`p-live-53`, $500K-$850K band) is a dedicated staffing line distinct from `p-live-01` Infrastructure Engineer Pre-training — same company, same compensation band, different scope. Signals pretraining-data-engineering becoming its own staffing surface.
- **Post-training/RLHF-training-infra DECELERATED.** 9/41 (22%) → 10/51 (19.6%). Net-new postings this cycle were overwhelmingly pre-training-focused, which temporarily reversed the growth trend. Trajectory across sampled cycles: 1/23 (4.3%) → 6/34 (17.6%) → 9/41 (22.0%) → 10/51 (19.6%). See "Emerging themes" below.
- **AMD Post-Training Platform Infrastructure Engineer** (posted 2026-09-15, closed 2026-10-16) was NOT counted despite its title — its scope is LLM-inference / P-D-disaggregation / KV-cache-lifecycle, which is peer-track territory (ai-infra-ml-platform-engineer owns the serving surface). Logged in `dropped_or_unfetchable_this_cycle` for signal only.
- **No new Blackwell / agent-driven-ops / bare-metal-fleet-ops signals captured.** Despite MLPerf 6.0 Blackwell sweeps and Meta Prometheus / CoreWeave GB200 announcements through October 2026, no new direct postings mention Blackwell as a required skill. Cursor (`p-live-32`) remains the only Blackwell JD mention; Anthropic Cluster Infra + Together AI remain the only agent-driven-ops signals; Together AI + Lightning AI remain the only bare-metal-OOB signals. All below floor and unchanged against a growing denominator (a decelerating signal).

## Emerging themes below threshold — watched, not added

Per the continuity-bias rule, an addition requires ≥3 distinct postings AND ≥30% frequency AND no existing module can be incrementally extended to cover it. These emerging themes surfaced but do NOT meet the triple gate:

| Theme | Direct postings | Assessment |
|---|---|---|
| **Post-training / RLHF-training infra as a dedicated sub-specialization** | **10/51 (19.6%)** — xAI Post-training Infrastructure Engineer (`p-live-04`), Airbnb Post Training (`p-live-26`), ByteDance Seed RL Systems & Infrastructure (`p-live-27`), Cursor ML Infra RL workloads (`p-live-32`), Cursor ML Research RL infra (`p-live-33`), Reka post-training + RL (`p-live-36`), Anthropic RL Velocity (`p-live-40`), Prime Intellect RL Infrastructure (`p-live-41`), Lightning AI post-training (`p-live-43`), **Thinking Machines Lab Research Engineer, Infrastructure, RL Systems** (`p-live-45`, new this cycle) | **DECELERATED this cycle.** Grew from 9/41 (22.0%) in 2026-10 to 10/51 (19.6%) in 2026-11 — up only 1 absolute posting while the denominator grew by 10. Trajectory across sampled cycles: 1/23 (4.3%) → 6/34 (17.6%) → 9/41 (22.0%) → 10/51 (19.6%). The theme's momentum has visibly cooled this cycle — the 10 net-new postings this cycle were overwhelmingly PRE-training-focused (TML Training Systems, Reflection AI Pre-Training Infra, Microsoft AI Pre-Training Infra, Anthropic Pre-training Data Infra) rather than post-training-focused. Still 10pp below the 30% frequency gate AND existing mod-102 + mod-110 can be incrementally extended if it re-accelerates. **Multiple additional candidates in `dropped_or_unfetchable_this_cycle` (Trajectory MTS ML Infra, Preference Model MTS Research Engineer Post-Training) would push this theme higher if confirmed next cycle.** Under continuity bias, no action. |
| Agent-driven / autonomous cluster lifecycle management | 2/51 (3.9%) — Anthropic Cluster Infrastructure (`p-live-17`), Together AI Bangalore (`p-live-30`) | Unchanged in absolute count; denominator grew. Still below the 3-posting floor. A decelerating signal — if this were going to catch on, the Prometheus / Blackwell cluster-build cycle would have surfaced new postings this quarter. Watch next cycle but no action. |
| Compiler-level performance (MLIR / LLVM / TorchInductor internals) as primary scope | 2/51 (3.9%) — Cerebras (`p-live-20`), Uber Triton preferred (`p-live-16`) | Unchanged. Prime Intellect preferred ('compiler or runtime optimization for ML systems'), OpenAI Training Perf preferred ('ML compiler optimization'), Apple preferred (Pallas/XLA kernel optimization), and the newly-republished TML Kernels + Numerics roles are all 'use the compiler' / peer-track territory (`ai-infra-performance` level 35). Below floor and this track uses (not authors) kernels. |
| Blackwell (B100 / B200 / GB200) in JD required-skills text | 1/51 (2.0%) — Cursor ML Infrastructure (`p-live-32`) | **Still 1 posting despite the public MLPerf 6.0 Blackwell sweeps, Meta's Prometheus 1 GW supercluster announcement, and CoreWeave's 2,496-GPU GB200 cluster** — most new direct postings are deliberately hardware-agnostic. mod-105 and mod-107 will absorb this as it grows. Continue watching. |
| Bare-metal / OOB fleet ops (BMC / Redfish / IPMI / PXE) in training-infra JDs | 2/51 (3.9%) — Together AI (`p-live-30`), Lightning AI (`p-live-31`) | Unchanged. Below the 3-posting floor and prerequisite territory (`ai-infra-engineer` owns hardware-lifecycle fluency). Do NOT add. |
| Rust for cluster services / IaC tooling | 6/51 (12%) — Anthropic Pretraining (`p-live-01`), xAI Supercompute (`p-live-05`), Anthropic Cluster (`p-live-17`), Cursor (`p-live-32`), OpenAI Training Performance (`p-live-39`), **Anthropic Pre-training Data Infra ('Python and Rust expertise' required, `p-live-53`, new this cycle)** | +1 posting this cycle (Anthropic Pre-training Data Infra). Clears the 3-posting floor as a language requirement but falls to prerequisite tracks (`ai-infra-engineer` for systems languages). Not training-platform-specific competency; the fact that it's REQUIRED (not preferred) at the top-band $500K-$850K Anthropic data-infra role is a notable band marker. |
| Trillion-scale corpus curation + dedup + curriculum design | 3/51 (6%) — Anthropic Pretraining (`p-live-01`), Poolside (`p-live-12`), **Anthropic Pre-training Data Infra ('build and operate end-to-end data pipelines that turn raw web-scale corpora into training-ready datasets' + explicit tokenization + dedup, `p-live-53`, new this cycle)** | **Just crossed the 3-posting direct-floor this cycle.** Still well below the 30% gate (6%). Already fully covered by mod-103 (WebDataset, StreamingDataset, dedup with Ray Data). Floor-crossing is a signal that pretraining-data-engineering is becoming its own staffing line — monitor for Databricks Mosaic, Reflection, and xAI expanding to multiple dedicated data-infra roles next cycle. |
| Multi-modal / long-context / MoE training specialization | 5/51 (10%) — Magic long-context (`p-live-10`), NVIDIA multi-modal FM (`p-live-14`), Airbnb multimodal (`p-live-26`), Reka multimodal (`p-live-36`), Apple MoE (`p-live-37`) | Unchanged in absolute count; denominator grew. 10% is well below 30% AND already implicit in mod-101 (EP for MoE) and mod-107 (sequence packing). Do NOT add. |
| TPU-first training stack (JAX / XLA / Pallas) as primary accelerator target | 1/51 (2.0%) — Apple AIML Staff ML Infra, Pre-training Infrastructure (`p-live-37`) | Unchanged. mod-102 already covers JAX/pjit/shard_map on 8 GPUs at training-pipeline-engineer altitude; TPU-specific kernels (Pallas) stay peer with ai-infra-performance. Continue watching. |

## Frontier-scale case-study anchor

The curriculum continues to lean on four fully-disclosed frontier-scale training reports to anchor depth expectations. Direct posting evidence backs the requirement themes; these open reports still supply the operational reality that no job posting fully articulates:

- **Llama 3 Herd of Models** ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)) — most fully-disclosed frontier pretraining run of 2024: data pipeline, parallelism strategy, cluster shape, failure statistics, and MFU. Case study across mod-101, mod-105, mod-106, mod-109.
- **OPT-175B Logbook** ([facebookresearch/metaseq](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)) — canonical open reference for the operational reality of a frontier pretraining run (hardware failures, restarts, loss spikes). Anchor case study for mod-106 and mod-108.
- **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model** ([arXiv:2211.05100](https://arxiv.org/abs/2211.05100)) — Jean Zay cluster topology, data pipeline, and training-time incidents. Cited across mod-101, mod-105, mod-106.
- **MosaicML MPT-7B** ([mosaicml.com/mpt](https://www.mosaicml.com/mpt)) — cost-optimised sub-frontier pretraining recipe with StreamingDataset + FSDP. Anchor for mod-103 and mod-109.

## Residual research

Next cycle should:

- Re-verify the oldest retained postings (`p-live-01` 2026-04-07, `p-live-05` 2026-03-21, `p-live-08` 2026-03-31) that are now 200+ days past `posted_date`. If any have dropped, cite the newer equivalents (Anthropic Pretraining has posted several times through 2026; Databricks/Mosaic frequently re-opens the Foundation Models role).
- Track the post-training / RLHF-training-infra theme. It DECELERATED this cycle (9/41 → 10/51, 22% → 19.6%) because the pre-training-focused net-new postings outpaced new post-training-infra postings. If 2026-12 shows re-acceleration (specifically if 3+ of the `dropped_or_unfetchable_this_cycle` RL-infra candidates — Trajectory, Preference Model, Thinking Machines Lab Research RL Scaling — are confirmed live and in-scope), re-evaluate. The smallest incremental response remains a mod-110 exercise on the post-training platform hand-off contract and/or a mod-102 case study on RL-specific framework patterns — NOT a new module.
- Track req-training-networking-storage closely. Jumped +5pp to 29% this cycle — just 1pp below the gate. If it crosses in 2026-12, mod-105 fully owns the scope so there is no gap — but the frequency crossing should be documented.
- Attempt an authenticated LinkedIn / Ashby / Workday pass to unlock the `dropped_or_unfetchable_this_cycle` smaller-shop MTS infrastructure roles: Trajectory, Radical Numerics, Hark, Nuance Labs, Preference Model, xAI MTS Pre-training Data Infrastructure. Several of these would shift the post-training/RLHF-infra theme higher if confirmed.
- Watch Blackwell adoption in JD required-skills text — still 1/51 (2.0%) this cycle despite the MLPerf 6.0 sweeps and Meta Prometheus / CoreWeave GB200 production announcements. Decelerating signal; most new postings are deliberately hardware-agnostic. If this theme is still at ≤2 postings in 2026-12, the hypothesis that it would catch on in JD text as hardware shipped should be revised.
- Watch "AI Infrastructure Agents" as a day-job deliverable — unchanged at 2 postings against a growing denominator (now 2/51 = 3.9%). Decelerating signal. If it crosses the 3-posting floor, mod-108 (observability) and mod-110 (platform architecture) should surface awareness of the pattern.
- Watch for the trillion-scale-corpus theme (just crossed the 3-posting floor this cycle via Anthropic Pre-training Data Infra). If Databricks Mosaic / Reflection / xAI post multiple dedicated data-infra roles in 2026-12, pretraining-data-engineering is maturing into its own staffing line and mod-103 should surface awareness that this is a distinct job title at some shops.

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

Direct posting evidence backs every existing requirement. **Five requirements continue to clear the ≥30% direct-frequency bar** (cluster orchestration 65%, distributed-training frameworks 65%, platform architecture/leadership 63%, throughput/MFU engineering 39%, and checkpointing/fault-tolerance 33%); networking & storage climbed to 29% — 1pp below the line. The remaining four fall between 8–27% but are still cited by multiple direct postings, still owned by this track (no peer track owns them), and already baked into the mature 10-module curriculum. **The post-training / RLHF-training-infra theme DECELERATED to 10/51 (19.6%), down 2pp from 22% last cycle, because the pre-training-focused net-new postings outpaced new post-training-infra postings. Trajectory: 1/23 (4.3%) → 6/34 (17.6%) → 9/41 (22.0%) → 10/51 (19.6%). It still does not clear the 30% frequency gate AND can be incrementally addressed by extending mod-102 or mod-110 if it re-accelerates.** Under the continuity-bias rule, no requirement gap warrants a net-new module, exercise, or project this cycle — see [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json) for the empty-additions delta with rationale.
