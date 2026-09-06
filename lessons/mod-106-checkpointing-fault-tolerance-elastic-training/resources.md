# Resources for mod-106 — Checkpointing, Fault Tolerance, and Elastic Training

Primary docs first, empirical logbooks second, detection / hardware
references third, SRE / SLO background fourth. Every chapter and
exercise is grounded in this list; the OPT chronicles, the Llama 3
report, and the DCP / elastic docs are the four references you
should be able to reach for by memory by the end of the module.

## PyTorch Distributed Checkpoint (DCP)

- **PyTorch Distributed Checkpoint documentation.**
  https://docs.pytorch.org/docs/stable/distributed.checkpoint.html —
  API reference for `save`, `load`, `async_save`, the `Stateful`
  protocol, planners, and the storage backends. Primary reference
  for chapter 3 and exercise 1.
- **PyTorch DCP recipe / tutorial.**
  https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
  — the canonical worked example. Get it running end-to-end before
  starting exercise 1.
- **`torch.distributed.checkpoint.state_dict` helper API.** Refer to
  the `state_dict` / `get_state_dict` / `set_state_dict` section of
  the DCP documentation above — the recommended bridge for FSDP2
  model + optimizer state extraction, referenced in exercise 1's
  starter guidance.
- **torchtitan repository.**
  https://github.com/pytorch/torchtitan — a reference training stack
  that wraps DCP into an `AppState`-style loop; useful to read as a
  concrete example while working chapter 3.

## torchrun, rendezvous, and elastic training

- **PyTorch Elastic (`torch.distributed.elastic`) documentation.**
  https://docs.pytorch.org/docs/stable/distributed.elastic.html —
  the agent, rendezvous, run, and `torchrun` reference. Primary
  reference for chapter 4 and exercise 2.
- **PyTorch `torchrun` CLI reference.**
  https://docs.pytorch.org/docs/stable/elastic/run.html — the CLI
  and the semantics of `--nnodes`, `--rdzv-backend`,
  `--rdzv-endpoint`, `--max-restarts`.
- **Elastic training-script contract.**
  https://docs.pytorch.org/docs/stable/elastic/train_script.html —
  the semantics your training script has to satisfy so a restart
  is safe. Read alongside chapter 4's re-entrancy discussion.
- **etcd project.** https://etcd.io/ — the external rendezvous
  backend option; chapter 4's comparison between `c10d` and
  `etcd-v2` refers to etcd's Raft quorum semantics.

## Empirical logbooks and frontier-scale training reports

- **OPT-175B chronicles (Meta AI, 2022).**
  https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf
  — the dated logbook for the 992-A100 OPT-175B pretraining run.
  Primary source for chapter 7. Read front-to-back at least once.
- **OPT paper (Zhang et al., 2022).**
  https://arxiv.org/abs/2205.01068 — "OPT: Open Pre-trained
  Transformer Language Models". The paper that accompanies the
  chronicles.
- **Llama 3 tech report (Grattafiori et al., 2024).**
  https://arxiv.org/abs/2407.21783 — "The Llama 3 Herd of Models".
  Section 3.3.2 ("Training infrastructure, at scale") is the
  modern-frontier reference for interruption rates and the
  incident-category distribution. Primary source for chapter 7 and
  the anchor for exercise 5's availability-budget derivation.
- **BLOOM paper (Le Scao et al., 2022).**
  https://arxiv.org/abs/2211.05100 — "BLOOM: A 176B-Parameter
  Open-Access Multilingual Language Model". A distinct-team
  perspective on 176B-scale infrastructure; complements the OPT
  chronicles.
- **PaLM paper (Chowdhery et al., 2023).**
  https://arxiv.org/abs/2204.02311 — "PaLM: Scaling Language
  Modeling with Pathways". Section 5 is the origin of the
  "goodput" framing for LLM training; primary reference for
  chapter 8's SLI definition.
- **TPU v4 paper (Jouppi et al., 2023).**
  https://dl.acm.org/doi/10.1145/3579371.3589350 — "TPU v4: An
  Optically Reconfigurable Supercomputer for Machine Learning",
  ISCA 2023. The goodput framing at TPU-pod scale that transfers
  to GPU pods; supporting reference for chapter 8.

## Silent-data-corruption (SDC) papers

- **Hochschild, P., et al. (2021). "Cores that don't count."**
  HotOS '21. https://dl.acm.org/doi/10.1145/3458336.3465297 —
  Google's account of CPU cores that produce incorrect results at
  low rate under production workloads. The paper that made SDC a
  first-class concern for hyperscalers. Referenced in chapter 6.
- **Dixit, H. D., et al. (2021). "Silent Data Corruptions at Scale."**
  https://arxiv.org/abs/2102.11245 — Meta / Facebook's account of
  SDC in production, the sibling paper to Google's "cores that
  don't count". Together they anchor chapter 6's SDC discussion.

<!-- needs-research: additional SDC-in-DL-training citations if
     the reader wants an ML-training-specific empirical study
     beyond the two hyperscaler CPU-focused papers above. -->


## NVIDIA hardware-health tooling

- **NVIDIA Data Center GPU Manager (DCGM) documentation.**
  https://docs.nvidia.com/datacenter/dcgm/latest/ — DCGM field
  IDs, policies, `dcgmi diag`, and the health-monitoring APIs.
  Primary reference for chapter 6's DCGM-side detector.
- **`dcgm-exporter` (Prometheus bridge).**
  https://github.com/NVIDIA/dcgm-exporter — the Prometheus /
  OpenMetrics exporter for DCGM, referenced by exercise 4.
- **NVIDIA Xid error reference.**
  https://docs.nvidia.com/deploy/xid-errors/ — every kernel-level
  Xid code, its meaning, and its severity. Chapter 6's Xid
  discussion refers back to this.
- **NVIDIA Resiliency Extension for PyTorch.**
  https://github.com/NVIDIA/nvidia-resiliency-ext — open-source
  detectors and quarantine helpers. Referenced as a comparison
  point in exercise 4's stretch goals.
- **NVIDIA `nvidia-smi` documentation.**
  https://docs.nvidia.com/deploy/nvidia-smi/ — the front-line tool
  for GPU clock-lock injection (`--lock-gpu-clocks`) used by
  exercise 4's fault-injection driver.

## NCCL (recovery and detection interface)

- **NCCL user guide.**
  https://docs.nvidia.com/deeplearning/nccl/ — timeouts,
  `NCCL_DEBUG`, `NCCL_ASYNC_ERROR_HANDLING`, and the environment
  variables chapter 4 tunes for aggressive detection.
- **NCCL troubleshooting guide.**
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html
  — first stop for classifying an NCCL timeout as fabric-side vs.
  training-side. Cross-referenced from mod-105 chapter 8.

## Debugging aids for training incidents

- **`py-spy`.** https://github.com/benfred/py-spy — live-process
  Python stack sampler used in exercise 2 to catch a hung rank
  before the NCCL timeout fires.
- **`gdb` + `py-bt`.** https://sourceware.org/gdb/ +
  https://wiki.python.org/moin/DebuggingWithGdb — attach to a
  hung C-level rank when `py-spy` cannot get through.
- **`strace`.** https://strace.io/ — syscall-level tracer, useful
  for stuck-on-I/O ranks in chapter 5's class-3 playbook.

## SRE / SLO background (chapter 8 and exercise 5)

- **Google SRE Book, chapter 4: "Service Level Objectives".**
  https://sre.google/sre-book/service-level-objectives/ — the
  definitional chapter for SLI / SLO / SLA / error budget. Every
  term chapter 8 uses traces back here. Read before writing
  exercise 5.
- **Google SRE Book, chapter 3: "Embracing Risk".**
  https://sre.google/sre-book/embracing-risk/ — the error-budget
  policy framing chapter 8's three-band template comes from.
- **Google SRE Workbook, chapter 2: "Implementing SLOs".**
  https://sre.google/workbook/implementing-slos/ — the more
  operational companion to the SRE Book chapter; useful for the
  measurement wiring in exercise 5's appendix A.

## Adjacent modules and cross-references

- **mod-101** — DDP / FSDP2 collectives underneath the checkpointed
  state; chapter 2's `AppState` inventory assumes you know what
  each collective's state actually is.
- **mod-102** — framework internals (Megatron, DeepSpeed,
  torchtitan). torchtitan's checkpoint helpers wrap DCP; DeepSpeed
  has its own checkpoint format. Chapter 3 refers to torchtitan as
  a reference impl.
- **mod-103** — data pipeline design. Chapter 2's dataloader / stateful-
  sampler discussion assumes mod-103 chapter 4's stateful-sampler
  pattern is familiar.
- **mod-104** — scheduler and gang scheduling. Chapter 4's elastic
  reshape sits on top of a scheduler that has already granted a
  gang; chapter 6's node quarantine writes to the scheduler's node
  labels.
- **mod-105** — the fabric. Chapter 5's class-3 (NCCL timeout)
  playbook routes into mod-105 chapter 8's fabric runbook for the
  root-cause step. Chapter 2's storage-cost model is grounded in
  mod-105 chapter 7's tier characteristics.
- **mod-108** — observability. This module *emits* the counters
  (step-time histograms, ECC delta counters, goodput ratio) that
  mod-108 turns into dashboards.
- **mod-109** — cost accounting. Chapter 8's SLO is the input;
  mod-109 turns burnt-budget hours into dollar figures.

## Recommended reading order for a first pass

1. Chapter 1 + Llama 3 §3.3.2 + PaLM §5 (2 h). Fixes the vocabulary
   and the two design numbers.
2. Chapter 2 + skim the DCP docs' top page (1.5 h). Enough to reason
   about the cost model before touching code.
3. Chapter 3 + the DCP recipe end-to-end (2.5 h). Do the tutorial;
   start exercise 1.
4. Chapter 4 + the PyTorch Elastic docs (2 h). Enough to run
   exercise 2's baseline.
5. Chapter 5 + the OPT chronicles' first 20 pages (2 h). Enough
   context for exercise 3.
6. Chapter 6 + the DCGM docs' field-ID reference + the two SDC
   papers, at least the abstracts (2 h). Enough to start
   exercise 4.
7. Chapter 7 + finish the OPT chronicles + Llama 3 §3.3.2 in full
   (2 h). Do the exercise 3 appendix A translation while it is fresh.
8. Chapter 8 + Google SRE Book chapter 4 (1.5 h). Then write
   exercise 5.
9. The DCGM + Xid references, kept open on-call. Not read cover-to-
   cover; treated as a lookup.

The exercises assume you have done items 1–4 before starting
exercise 1, items 5–6 before starting exercises 3 and 4, and items
7–8 before starting exercise 5. Some rearrangement is fine if your
team's schedule pushes the SLO conversation earlier.
