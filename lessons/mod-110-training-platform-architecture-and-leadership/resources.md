# Resources for mod-110-training-platform-architecture-and-leadership

RFC-process foundations first, then scheduler and framework
documentation, then the hosted-training vendor pages, then the
two anchor logbooks for chapter 6, then SRE / incident-review
and platform-engineering references, then pointers into other
modules of this track. Every chapter and exercise is grounded
in this list.

Vendor pages (Databricks, Together AI, NVIDIA, hyperscalers) and
PyTorch blog posts move; anchor repository URLs occasionally get
re-homed. Every URL in this file should be re-checked before
quoting a number, API, or SLA from it. Chapter 3's
"cite-the-date-you-checked" rule applies to *this file* as well.

## RFC-process foundations (chapter 3, exercise 2)

Chapter 3's nine-section template is adapted from the four
mature public RFC traditions below. Read at least one in full
for the *shape* before authoring your first training-platform
RFC.

- **IETF RFC 2119 — "Key words for use in RFCs to Indicate
  Requirement Levels."** Bradner, S. (1997) —
  https://www.rfc-editor.org/rfc/rfc2119. The MUST / SHOULD /
  MAY vocabulary chapter 3 inherits for its "normative language"
  convention. Pair with **BCP 14 / RFC 8174** —
  https://www.rfc-editor.org/rfc/rfc8174 — which clarifies the
  all-caps convention. Short reads; read both.
- **IETF RFC 7322 — "RFC Style Guide."** Flanagan, H. &
  Ginoza, S. (2014) —
  https://www.rfc-editor.org/rfc/rfc7322. The structural style
  guide for IETF RFCs. The source of chapter 3's "metadata
  first, summary second, motivation third" ordering.
- **PEP 1 — "PEP Purpose and Guidelines."** Warsaw, B., Hylton,
  J., Goodger, D., Coghlan, N. (2000, ongoing) —
  https://peps.python.org/pep-0001/. The Python-community RFC
  process. Chapter 3's status-ladder (Draft → Discussion →
  Accepted → Implemented → Superseded) is a direct adaptation.
  Exercise 2 cites PEP 1 as a shape reference.
- **rust-lang RFC process and the RFCs repository.**
  https://github.com/rust-lang/rfcs and the submission guide at
  https://github.com/rust-lang/rfcs/blob/master/README.md.
  Chapter 3's "sponsor-signs-not-author-signs" rule descends
  from the rust-lang model. Read any accepted RFC under
  `text/` for the "unresolved questions" and "alternatives"
  sections.
- **Kubernetes Enhancement Proposals (KEPs).**
  https://github.com/kubernetes/enhancements/tree/master/keps
  and the KEP template at
  https://github.com/kubernetes/enhancements/blob/master/keps/NNNN-kep-template/README.md.
  The KEP template is the closest public analogue to a
  training-platform RFC — graduation stages (alpha / beta /
  stable), rollout + rollback plans, and a dedicated test-plan
  section all appear here. Exercise 2 cites KEP-2000 and any
  recent KEP as a shape reference.
- **"Design Docs at Google"** (Industrial Empathy blog) —
  https://www.industrialempathy.com/posts/design-docs-at-google/.
  A frequently-cited external description of Google's internal
  design-doc culture. Useful as a point of comparison for the
  nine-section template.

## Multi-tenant scheduling and the capacity contract (chapter 2, exercise 1)

The scheduler primitives are owned by mod-104; this module
treats them as inputs. The references below are the ones chapter
2 and exercise 1 touch directly.

### Kubernetes-native stack

- **Kueue — Kubernetes-native job queueing.** Project page:
  https://kueue.sigs.k8s.io/ and
  https://github.com/kubernetes-sigs/kueue. Concepts:
  https://kueue.sigs.k8s.io/docs/concepts/. The source of
  `ClusterQueue`, `LocalQueue`, `cohort`, `nominalQuota`,
  `borrowingLimit`, `lendingLimit`, and `reclaimWithinCohort`
  referenced throughout chapter 2 and exercise 1's Kubernetes
  config bundle.
- **Volcano — batch scheduler for Kubernetes.** Project page:
  https://volcano.sh/ and https://github.com/volcano-sh/volcano.
  Plugin reference:
  https://github.com/volcano-sh/volcano/blob/master/docs/design/plugins.md.
  Source of the `proportion`, `preempt`, `gang`, and `drf`
  plugins chapter 2 and exercise 1 enable, and of
  `preemptGracePeriodSeconds`.
- **Kubeflow Training Operator.**
  https://www.kubeflow.org/docs/components/training/ and
  https://github.com/kubeflow/training-operator. The
  `PyTorchJob` / `TFJob` custom resource types chapter 2's
  "unit of gang" discussion assumes.
- **Kubernetes PriorityClasses.**
  https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/.
  The core-API primitive the Kueue + Volcano stack composes
  with; chapter 2's priority-class ladder maps to one
  `PriorityClass` per class.
- **ValidatingAdmissionPolicy (CEL).**
  https://kubernetes.io/docs/reference/access-authn-authz/validating-admission-policy/.
  The declarative admission-webhook mechanism exercise 1's
  "admission-time cluster invariants" bundle uses for the
  Kubernetes variant.

### Slurm stack

- **Slurm Workload Manager — documentation index.**
  https://slurm.schedmd.com/documentation.html. Primary
  entry point. Specifically relevant to chapter 2 and
  exercise 1:
  - `slurm.conf` reference —
    https://slurm.schedmd.com/slurm.conf.html. Source of
    `PriorityWeightFairshare`, `PriorityWeightQOS`,
    `PriorityDecayHalfLife`, `PreemptType`, `PreemptMode`,
    `KillWait`.
  - Multifactor priority plugin —
    https://slurm.schedmd.com/priority_multifactor.html.
  - Resource limits and QoS —
    https://slurm.schedmd.com/resource_limits.html and
    https://slurm.schedmd.com/qos.html.
  - Preemption —
    https://slurm.schedmd.com/preempt.html.
  - `sacctmgr` —
    https://slurm.schedmd.com/sacctmgr.html. Exercise 1's
    `sacctmgr-init.sh` is scripted against this.
  - `job_submit` plugin interface —
    https://slurm.schedmd.com/job_submit_plugins.html.
    Exercise 1's `job_submit.lua` is the Lua variant.

### Fair-share and gang-scheduling background

- **DRF (Dominant Resource Fairness).** Ghodsi, A., Zaharia,
  M., Hindman, B., Konwinski, A., Shenker, S., Stoica, I.
  (2011). NSDI'11 —
  https://www.usenix.org/legacy/event/nsdi11/tech/full_papers/Ghodsi.pdf.
  The theoretical basis for Volcano's `drf` plugin and the
  "weighted multi-resource fair-share" line chapter 2 relies
  on.
- **"The Google File System" / Borg / Omega lineage.** Verma,
  A., Pedrosa, L., Korupolu, M., Oppenheimer, D., Tune, E.,
  Wilkes, J. (2015). "Large-scale cluster management at Google
  with Borg." EuroSys'15 —
  https://research.google/pubs/large-scale-cluster-management-at-google-with-borg/.
  The reference ancestor for Kueue / Volcano's quota-and-cohort
  model; worth reading once for the vocabulary chapter 2 uses
  ("gang", "priority band", "quota").

## Frameworks for build-vs-buy option A/B (chapter 5, exercise 4)

- **NVIDIA Megatron-LM.** https://github.com/NVIDIA/Megatron-LM.
  Primary reference for build-vs-buy option A. Includes the
  canonical Megatron-Core modular variant at
  https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core.
- **Shoeybi, M., Patwary, M., et al. (2019). "Megatron-LM:
  Training Multi-Billion Parameter Language Models Using Model
  Parallelism."** arXiv 1909.08053 —
  https://arxiv.org/abs/1909.08053. The paper. Chapter 5 cites
  it as the anchor paper for option A.
- **Narayanan, D., Shoeybi, M., et al. (2021). "Efficient
  Large-Scale Language Model Training on GPU Clusters Using
  Megatron-LM."** arXiv 2104.04473 —
  https://arxiv.org/abs/2104.04473. The 3-D parallel paper.
  Useful for calibrating option A's "capability ceiling"
  claim.
- **NVIDIA NeMo Framework.** https://github.com/NVIDIA/NeMo and
  user guide at
  https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html.
  Primary reference for build-vs-buy option B (the NeMo
  variant). The build-vs-buy memo in exercise 4 skims the user
  guide before rating.
- **pytorch/torchtitan.** https://github.com/pytorch/torchtitan.
  Primary reference for build-vs-buy option B (the torchtitan
  variant) and the "whole-framework cutover" target in chapter
  7's second worked example. Includes the recipe templates
  chapter 5 cites as the lean PyTorch-native reference
  implementation.
- **Microsoft DeepSpeed.** https://github.com/microsoft/DeepSpeed.
  The ZeRO-style alternative chapter 3's RFC-042 example
  rejects. Chapter 5's "DeepSpeed as the third major framework"
  entry references this.
- **PyTorch FSDP2.** API reference:
  https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html.
  Design and migration background:
  https://pytorch.org/blog/fsdp2/ and the PyTorch 2.6 release
  notes. Primary reference for chapter 7's FSDP1 → FSDP2 worked
  example and exercise 2 option A.
- **PyTorch Distributed Checkpoint (DCP).**
  https://docs.pytorch.org/docs/stable/distributed.checkpoint.html.
  The checkpoint format FSDP2 integrates with and chapter 7's
  FSDP1 → FSDP2 converter is scripted against.
- **NVIDIA Transformer Engine.**
  https://github.com/NVIDIA/TransformerEngine and docs at
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html.
  The FP8 policy library chapter 3's hardware-refresh RFC
  example calls out as "new Transformer Engine version";
  cross-referenced from mod-107.

## Hosted and managed training for build-vs-buy option C/D (chapter 5, exercise 4)

Vendor pages move; verify every quantitative claim before
quoting. Each entry below is the primary reference exercise 4's
memo cites.

### Databricks Mosaic AI (option C)

- **Databricks Mosaic AI Model Training / foundation-model
  training.**
  https://docs.databricks.com/en/machine-learning/foundation-models/index.html
  and
  https://docs.databricks.com/en/machine-learning/foundation-model-training/index.html.
  Primary reference for option C.
- **MosaicML MPT-7B blog post.**
  https://www.databricks.com/blog/mpt-7b. Preserved on the
  Databricks domain post-2023-acquisition. Chapter 5's
  build-cost anchor for option C and the "~$200 K / ~9.5 day"
  reference point. (Primary MPT-7B reference also lives in
  mod-109's resources.md.)
- **MosaicML MPT-30B blog post.**
  https://www.databricks.com/blog/mpt-30b. Secondary anchor;
  the 30 B extension of the MPT-7B recipe.
- **MosaicML Composer.** https://github.com/mosaicml/composer.
  The training-loop library the MPT recipes are built on.
- **MosaicML LLM Foundry.**
  https://github.com/mosaicml/llm-foundry. The MPT training
  repo (code and configs).

### Together AI (option D)

- **Together AI documentation.** https://docs.together.ai/.
  Primary reference for option D.
- **Together AI fine-tuning / training overview.**
  https://docs.together.ai/docs/fine-tuning-overview and
  https://www.together.ai/pricing. The training-as-a-service
  surface area exercise 4's option D is rated against.

### Other option-D / option-C adjacent providers

- **NVIDIA DGX Cloud.**
  https://www.nvidia.com/en-us/data-center/dgx-cloud/. The
  NVIDIA-run offering chapter 5 lists as an option-C/D hybrid.
- **CoreWeave managed services.** https://www.coreweave.com/.
  Named in chapter 5 as a managed-training option.
- **Modal.** https://modal.com/. Named in chapter 5 as an
  option-D specialist.
- **Anyscale (Ray Train).** https://www.anyscale.com/ and
  https://docs.ray.io/en/latest/train/train.html. Named in
  chapter 5 as a Ray-native training option.

### Underlying accelerator pricing (option A/B TCO inputs)

Reuse the hyperscaler and specialty-cloud pages catalogued in
`mod-109/resources.md` (AWS P5, GCP A3, Azure ND H100 v5,
CoreWeave, Lambda, Crusoe, Together, DGX Cloud). Exercise 4's
TCO decomposition reads them from mod-109; this module does not
re-maintain the list.

## Incident reviews — the two open anchors (chapter 6, exercise 5)

Both of the open logbooks below should be read at least once
before authoring a training-run incident review. Chapter 6's
worked examples and exercise 5's canonical options are anchored
to these.

- **OPT-175B — Zhang, S., et al. (2022). "OPT: Open Pre-trained
  Transformer Language Models."** arXiv 2205.01068 —
  https://arxiv.org/abs/2205.01068. The paper describing
  Meta's open 175 B pretraining.
- **OPT-175B chronicles (training logbook).**
  https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf
  and the surrounding `chronicles/` directory at
  https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles.
  The day-by-day narrative of the loss spikes, LR adjustments,
  skipped batches, and host failures. Chapter 6's "loss-spike"
  worked example and exercise 5's option A anchor.
- **BLOOM — BigScience, 2022. "BLOOM: A 176B-Parameter
  Open-Access Multilingual Language Model."** arXiv 2211.05100 —
  https://arxiv.org/abs/2211.05100. The BLOOM paper.
- **BLOOM training logbook.**
  https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml
  (the `chronicles-prequel.md`, `chronicles.md`, and
  surrounding `README.md`). Chapter 6's "hardware failure"
  worked example and exercise 5's option B anchor.
- **The Llama 3 Herd of Models.** Grattafiori, A., et al.
  (2024). arXiv 2407.21783 —
  https://arxiv.org/abs/2407.21783. §3.3.2 reports the
  interruption taxonomy, the 16 K-GPU goodput, and the
  per-category failure count across the 54-day pretraining.
  Chapter 2's "goodput target 0.85–0.95" and chapter 6's
  "frontier runs are 20–50% goodput" claims cite this
  section directly.
- **NVIDIA Xid error reference.**
  https://docs.nvidia.com/deploy/xid-errors/index.html. The
  canonical taxonomy for GPU-driver-level errors. Exercise 5's
  hardware worked example cites Xid codes by number.

## SRE and incident-review discipline (chapter 6, exercise 5)

- **Beyer, B., Jones, C., Petoff, J., Murphy, N. R. (eds.)
  (2016). "Site Reliability Engineering: How Google Runs
  Production Systems."** O'Reilly / Google — full text free at
  https://sre.google/sre-book/table-of-contents/. The reference
  work.
  - **Chapter 15 — "Postmortem Culture: Learning from Failure"** —
    https://sre.google/sre-book/postmortem-culture/. The source
    for chapter 6's blameless framing. Mandatory reading before
    exercise 5.
  - **Chapter 16 — "Tracking Outages"** —
    https://sre.google/sre-book/tracking-outages/. Cited for
    the "monthly incident-trends review" cadence.
  - **Chapter 11 — "Being On-Call"** —
    https://sre.google/sre-book/being-on-call/. Grounds chapter
    2's on-call structures A and B.
- **Beyer, B., Murphy, N. R., Rensin, D. K., Kawahara, K.,
  Thorne, S. (eds.) (2018). "The Site Reliability Workbook:
  Practical Ways to Implement SRE."** O'Reilly / Google —
  https://sre.google/workbook/table-of-contents/. The
  companion volume. Chapter 10 ("Postmortem Culture: Learning
  from Failure") is the applied version of the SRE book
  Chapter 15.
- **Google — "Postmortem Template."**
  https://sre.google/sre-book/example-postmortem/ and
  https://sre.google/workbook/postmortem-culture/. The reference
  template chapter 6's eight-section template adapts.
- **Allspaw, J. (2012). "Blameless PostMortems and a Just
  Culture."** Etsy / Code as Craft —
  https://www.etsy.com/codeascraft/blameless-postmortems/. The
  short, widely-cited external statement of blameless
  discipline. Useful to send to leadership if blameless framing
  is pushed back on.
- **ACM SIGOPS SOSP / NSDI incident-case-study literature.**
  No single URL; search ACM DL and USENIX for "post-mortem"
  and "incident analysis" case studies in SOSP, OSDI, and
  NSDI. Useful as grounding for the "five-class root-cause
  taxonomy" chapter 6 uses.

## Platform-engineering and cross-team contracts (chapters 1, 4)

- **Skelton, M. & Pais, M. (2019). "Team Topologies:
  Organizing Business and Technology Teams for Fast Flow."**
  IT Revolution — https://teamtopologies.com/book. The
  reference work for "platform team" as a distinct type and
  for the "team interaction mode" vocabulary chapter 4 uses
  implicitly. Chapter 1's framing of the training platform as
  "a product consumed by other engineering teams" is a direct
  descendant.
- **Thoughtworks — "Platform as a product."**
  https://www.thoughtworks.com/insights/articles/platform-as-a-product
  (and the related entries on
  https://www.thoughtworks.com/radar/techniques). Short public
  articulation of the platform-as-product framing chapter 1
  opens with.
- **Fowler, M. "Internal Platform."**
  https://martinfowler.com/articles/talk-about-platforms.html.
  Useful commentary on the "platform team as product team"
  pattern.
- **CNCF Platforms White Paper.**
  https://tag-app-delivery.cncf.io/whitepapers/platforms/. The
  CNCF articulation of the internal-developer-platform model;
  the four-contracts framing in chapter 1 is adjacent to this
  document's "platform capabilities" model.

## Peer-track references (chapter 4, exercise 3)

Chapter 4's five peer interfaces reference the following AICG
sibling tracks (and their role / curriculum documents). Each
track's own `role-description.md` and `CURRICULUM.md` are the
authoritative sources for what the peer team owns.

- **`fine-tuning-engineer` track** — the primary platform
  consumer. Interface 1.
- **`ai-infra-ml-platform-engineer` track** — serving and the
  model registry. Interface 2.
- **`ai-infra-mlops-engineer` track** — CI/CD and
  post-training. Interface 3.
- **`ai-infra-performance-engineer` track** — kernel and
  low-level optimisation. Interface 4.
- **`ai-infra-security-engineer` track** — training-data
  provenance and cluster boundary. Interface 5.

External standards referenced by the interface templates:

- **JSON Schema (draft 2020-12).**
  https://json-schema.org/draft/2020-12/release-notes.html.
  Exercise 3 uses JSON Schema (or an equivalent) for the formal
  interface-artifact spec.
- **Protocol Buffers v3 language guide.**
  https://protobuf.dev/programming-guides/proto3/. Alternative
  formal-spec format for exercise 3.
- **OCI image specification.**
  https://github.com/opencontainers/image-spec. Grounds the
  "image digest" admission-controller invariant in chapters 2
  and 4.
- **Sigstore / in-toto for provenance.**
  https://www.sigstore.dev/ and https://in-toto.io/. The
  provenance-signing stack chapter 4 interface 5 (security)
  cites as the publication format for the data-provenance
  manifest.
- **SLSA — Supply-chain Levels for Software Artifacts.**
  https://slsa.dev/. The vocabulary interface 5's
  provenance-manifest contract borrows.

## Related pointers into other modules of this track

This module *consumes* every mechanism owned by mod-101 through
mod-109; the pointers below are the specific chapters each
chapter of mod-110 cross-references.

- **mod-101 (distributed-training semantics).** Chapter 3's
  RFC-042 worked example (FSDP1 → FSDP2) assumes the FSDP2
  semantics owned by mod-101. Chapter 7's first worked example
  cites the same chapters.
- **mod-102 (training frameworks).** Chapter 5's build-vs-buy
  options A / B are *about* which framework (Megatron, NeMo,
  torchtitan, DeepSpeed) to run; mod-102 is *how* those
  frameworks work internally.
- **mod-103 (data pipeline).** Chapter 4's interface 5
  (security) consumes the provenance manifest mod-103 emits.
  The manifest schema is mod-103's.
- **mod-104 (orchestration).** Chapter 8 of mod-104 is the
  scheduler-plumbing side of chapter 2 of this module. Read
  mod-104 chapter 8 before authoring the exercise-1
  capacity-contract bundle.
- **mod-105 (networking / storage).** Chapter 3 of this
  module templates storage-migration RFCs; mod-105 chapters 3
  and 5 are the storage-tier reference.
- **mod-106 (checkpointing / elastic / incidents).** Chapter
  6 of this module is the *review process*; mod-106 chapter 5
  is the *runbook* the reviews reference. Chapter 7's rollback
  stage links back into mod-106's elastic-recovery runbook.
- **mod-107 (MFU and kernel engineering).** Chapter 3 of this
  module templates kernel-upgrade RFCs; mod-107 is the
  kernel-integration mechanics.
- **mod-108 (observability and reproducibility).** Chapter 2
  of this module cites mod-108 chapter 7's metadata store as
  the source for the "historical demand" guarantee input.
  Chapter 4's interface 3 (MLOps) consumes mod-108's metadata
  schema. Chapter 6 cites mod-108 chapter 3 (metrics) and
  chapter 6 (loss-spike signatures) as the detection surface.
- **mod-109 (cost, capacity, cluster economics).** Chapter 2
  of this module cites mod-109 chapter 7's feasibility study
  as the "forward demand" guarantee input. Chapter 5
  (build-vs-buy) is the strategic re-test around mod-109's
  per-run arithmetic; exercise 4's TCO decomposition is scripted
  against mod-109 chapters 3 and 4.

## Recommended reading order for a first pass

1. Chapter 1 + SRE Book chapter 1 ("Introduction") + skim
   Team Topologies' "Platform team" chapter (1.5 h). Enough
   vocabulary to open any other chapter.
2. Chapter 2 + Kueue "Concepts" + Volcano plugin reference
   (or Slurm `priority_multifactor` + `preempt` pages) + skim
   mod-104 chapter 8 (2 h). Enough to run exercise 1.
3. Chapter 3 + PEP 1 + one accepted rust-lang RFC + one recent
   KEP (1.5 h). Enough to run exercise 2.
4. Chapter 4 + skim the peer-track `role-description.md` for
   whichever interface you plan to author in exercise 3
   (1.5 h). Enough to run exercise 3.
5. Chapter 5 + the Megatron-LM, NeMo, torchtitan, Databricks
   Mosaic, and Together AI landing pages (2 h). Enough to run
   exercise 4.
6. Chapter 6 + SRE Book chapter 15 + the OPT-175B chronicles
   PDF cover-to-cover + skim the BLOOM chronicles (3 h).
   Enough to run exercise 5.
7. Chapter 7 + the PyTorch FSDP2 blog + skim the torchtitan
   README (1 h). Grounds the migration-plan template every
   chapter 3 RFC's §9 links to.

Exercises assume this reading order; do them in sequence. The
exercise-5 incident review assumes exercise 1's capacity
contract and the chapter 6 reading above.
