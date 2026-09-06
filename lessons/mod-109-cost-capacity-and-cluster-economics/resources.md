# Resources for mod-109-cost-capacity-and-cluster-economics

Primary papers first, then vendor pricing pages and hardware
datasheets, then framework and anchor artifacts, then pointers
into other modules of this track. Every chapter and exercise is
grounded in this list.

Cloud pricing pages move; anchor blog posts get re-hosted.
Every URL in this file should be re-checked before quoting a
number from it. Chapter 3's "cite the date you checked" rule
applies to *this file* as well.

## Scaling laws and the compute ledger (chapter 1, exercise 1)

- **Kaplan, J., McCandlish, S., et al. (2020). "Scaling Laws for
  Neural Language Models."** arXiv 2001.08361 —
  https://arxiv.org/abs/2001.08361. The original power-law fit
  for `L(N)`, `L(D)`, `L(C)`; the paper the "20 tokens per
  parameter" folklore had to be re-derived away from. Chapter
  1's Kaplan discussion and exercise 1's `kaplan_optimal_*`
  functions cite this.
- **Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022).
  "Training Compute-Optimal Large Language Models" (Chinchilla).**
  arXiv 2203.15556 — https://arxiv.org/abs/2203.15556. The
  compute-optimal `D / N ≈ 20` result, the three-approach fit
  in §3, and equation 3 that chapter 1 references directly.
  Primary reference for chapter 1, exercise 1, and every
  feasibility study in exercise 5.
- **Muennighoff, N., Rush, A. M., Barak, B., et al. (2023).
  "Scaling Data-Constrained Language Models."** arXiv
  2305.16264 — https://arxiv.org/abs/2305.16264. Fits the
  repeat-data correction; chapter 1's data-constrained
  discussion and exercise 1's optional MoE / data-constrained
  stretch goal.
- **DeepMind Chinchilla replication / re-analysis literature.**
  Multiple public re-derivations exist (e.g., the Epoch AI
  replication of Chinchilla's approach 3); consult
  https://epochai.org/blog for the current best pointers. Use
  these to sanity-check exercise 1's `chinchilla_frontier`
  fits.

## Anchor recipes: MPT-7B and Llama 3 (chapter 6, all exercises)

- **MosaicML / Databricks MPT-7B announcement.**
  https://www.databricks.com/blog/mpt-7b — the cost-optimised
  anchor from chapter 6. Contains the recipe, the training
  hardware (~440× A100-40GB), the ~9.5-day wall-clock, and the
  ~$200 K figure. Preserved on the Databricks domain post-
  acquisition.
- **MosaicML MPT-30B announcement.**
  https://www.databricks.com/blog/mpt-30b — the 30 B extension
  of the recipe with a comparable cost breakdown; useful when
  scaling MPT-style reasoning above 7 B.
- **`llm-foundry` (MosaicML's training repo).**
  https://github.com/mosaicml/llm-foundry — the code that
  produced MPT-7B. Contains the exact recipe, config files, and
  data-preparation scripts. The reference implementation for
  the cost-optimised anchor.
- **Grattafiori, A., et al. (2024). "The Llama 3 Herd of
  Models."** arXiv 2407.21783 — https://arxiv.org/abs/2407.21783.
  The frontier anchor. Section 3 documents the training recipe,
  §3.2 documents the 4-D parallel decomposition on the 16 K-GPU
  cluster, §3.3 reports the 39.3 M H100-hours figure for the
  70 B model, and §3.3.2 documents the interruption taxonomy
  every frontier feasibility study is graded against. Primary
  reference for chapter 6, and mandatory reading before
  exercise 5's frontier study.
- **Touvron, H., et al. (2023). "LLaMA: Open and Efficient
  Foundation Language Models" (Llama 1).** arXiv 2302.13971 —
  https://arxiv.org/abs/2302.13971. Chapter 1's Llama-1-style
  inference-optimal ratio (`D / N ≈ 140`) is grounded here.
- **Touvron, H., et al. (2023). "Llama 2: Open Foundation and
  Fine-Tuned Chat Models."** arXiv 2307.09288 —
  https://arxiv.org/abs/2307.09288. Reports observed MFU on the
  A100 pretraining and calibrates chapter 2's MFU discussion.
- **OPT-175B: Zhang, S., et al. (2022). "OPT: Open Pre-trained
  Transformer Language Models."** arXiv 2205.01068 —
  https://arxiv.org/abs/2205.01068. Together with the
  chronicles logbook at
  https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf
  provides the canonical incident chronicle a chapter-7 risk
  register cites. Cross-referenced from mod-108 and mod-106.
- **Chowdhery, A., et al. (2022). "PaLM: Scaling Language
  Modeling with Pathways."** arXiv 2204.02311 —
  https://arxiv.org/abs/2204.02311. §5 introduces the "goodput"
  term chapter 5 uses; appendix F documents the loss-spike
  experience.

## MFU / HFU and parallelism accounting (chapter 2)

- **Chowdhery et al. (2022) — PaLM paper.** See above. §5 is
  the canonical reference for MFU as a metric.
- **Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-
  Billion Parameter Language Models Using Model Parallelism."**
  arXiv 1909.08053 — https://arxiv.org/abs/1909.08053. The
  origin of the tensor-parallel accounting chapter 2 uses in its
  `η(G)` discussion.
- **Narayanan, D., et al. (2021). "Efficient Large-Scale
  Language Model Training on GPU Clusters Using Megatron-LM."**
  arXiv 2104.04473 — https://arxiv.org/abs/2104.04473. Reports
  Megatron-LM MFU numbers at multiple cluster sizes; useful for
  calibrating chapter 2's small-scale-vs-frontier discussion.

## NVIDIA GPU datasheets and product pages (chapters 2, 3)

- **NVIDIA H100 Tensor Core GPU datasheet.**
  https://resources.nvidia.com/en-us-tensor-core — the H100
  SXM5 dense BF16 peak (989 TFLOP/s) and FP8 peak numbers
  chapter 2 and exercise 2 anchor on. The sparsity-doubled
  numbers advertised on marketing pages must not be used
  as the dense peak; chapter 2 calls this out as a failure
  mode.
- **NVIDIA A100 Tensor Core GPU datasheet.**
  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
  — the A100 SXM4 BF16 peak (312 TFLOP/s) used in chapter 6's
  MPT-7B reconciliation.
- **NVIDIA H200 product page.**
  https://www.nvidia.com/en-us/data-center/h200/ — the HBM3e
  bandwidth uplift over H100 with the same tensor-core throughput
  ceiling; relevant to chapter 3's cluster-shape discussion.
- **NVIDIA B200 / Blackwell product page.**
  https://www.nvidia.com/en-us/data-center/hgx-b200/ — the next-
  generation reference for chapter 3's amortisation discussion
  and for exercise 5's frontier-scale next-gen scaling.
- **NVIDIA DGX H100 reference architecture.**
  https://www.nvidia.com/en-us/data-center/dgx-h100/ — the 8-GPU
  node and NVLink/NVSwitch topology the "on-prem shape"
  arithmetic in chapter 3 uses.
- **NVIDIA SuperPOD reference architecture.**
  https://www.nvidia.com/en-us/data-center/dgx-superpod/ — the
  multi-rack extension of the DGX unit; the reference for
  frontier on-prem shapes in exercise 5.

## Hyperscaler pricing and instance pages (chapters 3, 4, exercises 2, 3)

Cite the URL and date every time; these pages move.

### AWS

- **AWS EC2 P5 instance page (H100).**
  https://aws.amazon.com/ec2/instance-types/p5/
- **AWS EC2 on-demand pricing.**
  https://aws.amazon.com/ec2/pricing/on-demand/
- **AWS EC2 reserved-instance pricing.**
  https://aws.amazon.com/ec2/pricing/reserved-instances/ and
  https://aws.amazon.com/ec2/pricing/reserved-instances/pricing/
- **AWS EC2 Capacity Blocks for ML.**
  https://aws.amazon.com/ec2/capacityblocks/ — the fixed-window
  reservation product chapter 4 discusses.
- **AWS EC2 Spot Instances.**
  https://aws.amazon.com/ec2/spot/
- **AWS HPC / UltraCluster overview.**
  https://aws.amazon.com/hpc/ultraclusters/ — the fabric and
  cluster-shape reference for P5 UltraClusters.

### Google Cloud

- **GCP GPU compute overview.**
  https://cloud.google.com/compute/docs/gpus and pricing at
  https://cloud.google.com/compute/gpus-pricing.
- **GCP A3 (H100) VM family.**
  https://cloud.google.com/blog/products/compute/introducing-a3-supercomputers-with-nvidia-h100-gpus
  — announcement and specs for `a3-highgpu-8g`.
- **GCP Cloud TPU (v4/v5p/v5e).** https://cloud.google.com/tpu
  — the alternative accelerator lineage; chapter 3 lists it as
  a variant with its own peak-FLOPs sheet.
- **GCP Spot / Preemptible VMs.**
  https://cloud.google.com/spot-vms
- **GCP compute reservations.**
  https://cloud.google.com/compute/docs/instances/reservations-overview

### Azure

- **Azure ND H100 v5 VM series.**
  https://learn.microsoft.com/azure/virtual-machines/nd-h100-v5-series
- **Azure Linux VM pricing.**
  https://azure.microsoft.com/pricing/details/virtual-machines/linux/
- **Azure Reserved VM Instances.**
  https://azure.microsoft.com/pricing/reserved-vm-instances/
- **Azure Spot VMs.**
  https://azure.microsoft.com/pricing/spot-vms/

### Specialty GPU clouds

Chapter 4's "specialty GPU cloud" tier is populated by providers
that price and provision differently from the hyperscalers. The
list moves; verify a provider's current SLAs before quoting.
Common references at time of writing:

- **CoreWeave.** https://www.coreweave.com/pricing
- **Lambda.** https://lambdalabs.com/service/gpu-cloud
- **Crusoe.** https://crusoe.ai/
- **Together AI.** https://www.together.ai/pricing

### NVIDIA DGX Cloud

- **NVIDIA DGX Cloud overview.**
  https://www.nvidia.com/en-us/data-center/dgx-cloud/ — the
  NVIDIA-run offering the "capacity block" and "multi-year
  frontier commit" patterns in chapter 4 sometimes route into.

## Checkpoint-cadence theory (chapter 5)

- **Young, J. W. (1974). "A First Order Approximation to the
  Optimum Checkpoint Interval."** Comm. ACM, 17(9), 530–531 —
  https://dl.acm.org/doi/10.1145/361147.361115. The
  `t_cadence_opt = sqrt(2 · t_write / λ)` formula chapter 5
  and exercise 4 use. Foundational HPC-checkpointing reference.
- **Daly, J. T. (2006). "A higher order estimate of the optimum
  checkpoint interval for restart dumps."** Future Generation
  Computer Systems, 22, 303–312 —
  https://www.sciencedirect.com/science/article/pii/S0167739X04001694
  — Young's formula extended past the first-order regime. Cite
  when `t_write / MTBF` grows past ~10%.
- **Cross-reference: mod-106 chapters 1–3.** The mechanism side
  (async DCP, retention policy, storage layer) is owned by
  mod-106; chapter 5 of this module quantifies the dollar
  impact of those mechanisms.

## Frameworks and training-loop references (chapters 2, 5)

- **PyTorch FSDP2 documentation.**
  https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
  — the parallelism baseline chapter 2 assumes for `η(G)`
  reasoning at small-to-medium scale.
- **torchtitan.** https://github.com/pytorch/torchtitan — the
  reference PyTorch large-model training template with FSDP2 +
  tensor / sequence parallelism; a public codebase to calibrate
  MFU / `η(G)` numbers against.
- **NVIDIA Megatron-LM.** https://github.com/NVIDIA/Megatron-LM
  — the 3-D / 4-D parallel implementation Llama 3 §3.2's
  decomposition descends from.
- **Microsoft DeepSpeed.**
  https://github.com/microsoft/DeepSpeed — ZeRO-style
  parallelism; useful for chapter 2's `η(G)` alternatives and
  chapter 5's `μ_nom` uplift catalog.
- **NVIDIA Transformer Engine.**
  https://github.com/NVIDIA/TransformerEngine — the FP8 policy
  library exercise 4's roadmap prices; cross-referenced from
  mod-107 chapter 4.

## Reference-quality economic analyses and commentary

Use these to sanity-check exercise 4's next-gen numbers and
exercise 5's dollar ranges. They are analyses, not primary
data — cite them as commentary, not as authoritative cost
figures.

- **Epoch AI compute and cost trend articles.**
  https://epochai.org/blog — recurring publications on frontier
  training-compute trends, cost estimates, and dataset scaling.
  Useful for calibrating a frontier feasibility study's dollar
  range.
- **SemiAnalysis frontier-training analyses.**
  https://www.semianalysis.com/ — third-party analyses of
  hyperscaler training-cost economics. Paywalled; treat
  quantitative claims as inputs to reconcile, not as ground
  truth.
- **Stanford HAI AI Index (annual).**
  https://aiindex.stanford.edu/report/ — the annual public
  survey; the "training compute and cost" chapter each year
  aggregates frontier-training cost estimates with primary-
  source citations. Chapter 6's "the paper does not commit to a
  dollar figure" note is why third-party estimates like the AI
  Index's exist.

## Related pointers into other modules of this track

- **mod-101 chapters (distributed-training semantics).** This
  module treats parallelism-efficiency numbers as inputs;
  mod-101 derives them. Read mod-101 chapter on FSDP2 /
  tensor-parallelism before doing exercise 2's `η(G)`
  calibration.
- **mod-102 chapters (training frameworks).** The framework
  layer (Megatron, DeepSpeed, torchtitan) that produces the
  MFU numbers. Chapter 2's `μ_nom` is a mod-102 outcome.
- **mod-103 chapters (data pipeline).** The data-loader cost
  line in chapter 3's side-term budget is a mod-103
  measurement. Exercise 5's risk register cites mod-103 for
  the data-stall guardrail.
- **mod-104 chapters (orchestration).** Chapter 4's dedicated-
  vs-shared discussion stops at "the scheduler policy is an
  input"; the scheduler implementation is mod-104.
- **mod-105 chapters (networking / storage).** The `η(G)` term
  in chapter 2 is a mod-105 outcome; the fabric assumptions in
  chapter 3's cluster shape descend from mod-105.
- **mod-106 chapters 1–8 (checkpointing, elastic training,
  incidents).** Chapter 5 of this module quantifies the dollar
  impact of mod-106's mechanisms. Exercise 4's roadmap cites
  mod-106 chapters 3, 4, 6, 8 for the reliability uplifts.
  Exercise 5's risk register mirrors mod-106 chapter 5's
  incident classes.
- **mod-107 chapters 1–7 (MFU / kernel engineering).** Chapter
  5 of this module quantifies the dollar impact of mod-107's
  optimisations. Exercise 4's roadmap cites mod-107 chapters
  2, 3, 4, 5, 6, 7 for the MFU-uplift lineup.
- **mod-108 chapters 2, 4 (observability, experiment
  tracking).** Chapter 7's "living study" pattern (weekly re-
  price) depends on mod-108's operational dashboards and
  tracker for the actuals-to-date; this module treats them as
  inputs.
- **mod-110 chapters (platform architecture and leadership).**
  Chapter 7 stops at the study schema; mod-110 owns the RFC
  process, review cadence, sign-off authority, and build-vs-buy
  workflow around it.

## Recommended reading order for a first pass

1. Chapter 1 + Hoffmann et al. (2022) §§1–3 + skim Kaplan et
   al. (2020) (2 h). Enough to run exercise 1.
2. Chapter 2 + skim Narayanan et al. (2021) + the H100
   datasheet's peak-FLOPs table (1.5 h). Enough to start
   exercise 2's `hardware.py`.
3. Chapter 3 + the AWS P5, GCP A3, and Azure ND H100 v5
   pricing pages (1.5 h). Enough to finish exercise 2's
   `dollars.py`.
4. Chapter 4 + the AWS reserved / capacity-blocks / spot
   pages (1.5 h). Enough to run exercise 3.
5. Chapter 5 + Young (1974) + skim mod-106 chapters 3 and 6
   (1.5 h). Enough to run exercise 4.
6. Chapter 6 + read Grattafiori et al. (2024) §3 twice + the
   MPT-7B blog (2.5 h). Enough to anchor exercise 5's frontier
   study.
7. Chapter 7 + skim the OPT-175B chronicles (1.5 h). Enough to
   author exercise 5's mid-scale study.

The exercises assume the reading in the same order; do them in
sequence. The exercise-5 capstone assumes every prior exercise
is complete.
