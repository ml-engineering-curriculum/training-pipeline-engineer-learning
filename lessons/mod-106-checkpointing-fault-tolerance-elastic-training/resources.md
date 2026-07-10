# Resources for mod-106-checkpointing-fault-tolerance-elastic-training

Primary sources first, tooling and vendor docs second. These are
the references the chapters and exercises are grounded in. Skim
the framework docs (PyTorch DCP, `torch.distributed.elastic`) and
the NVIDIA DCGM docs as-needed while working the exercises; the
papers and logbooks reward a full read.

## Primary papers — training reliability at scale

- **Zhang, S., et al. (2022). "OPT: Open Pre-trained Transformer
  Language Models."** arXiv:2205.01068. Read alongside the
  **OPT-175B logbook** the paper links to — the definitive
  public record of large-scale training operational realism
  (loss spikes, NCCL timeouts, rewinds, batch skips, host
  reboots). Chapters 1, 5, and 7 draw on it directly. Lab 01
  translates its top-five incident classes into your own
  runbook entries.
- **Le Scao, T., et al. (2022). "BLOOM: A 176B-Parameter
  Open-Access Multilingual Language Model."** arXiv:2211.05100.
  The BigScience workshop's operational write-up of a 176B run
  on the Jean Zay supercomputer; useful for a second
  data-point on incident distribution alongside OPT-175B.
- **Grattafiori, A., et al. (2024). "The Llama 3 Herd of
  Models."** Meta AI technical paper. Section 6.3 tabulates the
  failure taxonomy, MTTR, and effective goodput for the 405B
  pretraining window (16 384 H100s, 54 days, ~419 unexpected
  interruptions, ~78% attributable to confirmed hardware
  issues, ≥90% effective goodput). Chapters 1, 5, 6, and 7
  cite it. Exercise 03 asks you to build your incident
  playbooks against its failure-class table.

## Related reliability / goodput references

- **Kokolis, A., et al. (2025). "Revisiting Reliability in
  Large-Scale Machine Learning Research Clusters."** arXiv
  preprint. A production study of failure modes and goodput
  across Meta's research clusters; useful for the definition
  of goodput and its decomposition. <!-- needs-research: confirm
  the exact arXiv ID and publication venue if citing in a
  formal document -->
- **Mohan, J., et al. (2024). "Characterizing ML Training
  Workloads on Nvidia H100 GPUs."** Referenced in chapter 1 for
  the formal definition of goodput and its instrumentation.
  <!-- needs-research: confirm the exact venue / arXiv ID
  before quoting figures directly. -->
- **Beyer, B., et al. (eds.) (2016). "Site Reliability
  Engineering: How Google Runs Production Systems."** O'Reilly.
  Chapter 3 on SLOs and error budgets is the source of the
  availability-budget framing used in chapter 7.

## Distributed checkpoint (DCP) references

- **PyTorch `torch.distributed.checkpoint` documentation.**
  https://pytorch.org/docs/stable/distributed.checkpoint.html —
  the `save` / `load` / `async_save` APIs, the
  `DefaultSavePlanner` / `DefaultLoadPlanner`, the
  `FileSystemWriter` / `FileSystemReader`, and the
  `Stateful` protocol.
- **PyTorch `torch.distributed.checkpoint.state_dict` helpers.**
  https://pytorch.org/docs/stable/distributed.checkpoint.html —
  `get_state_dict`, `set_state_dict`, and
  `StateDictOptions(cpu_offload=...)`. What FSDP2 users actually
  call.
- **torchtitan `Checkpointer`.**
  https://github.com/pytorch/torchtitan — see
  `torchtitan/components/checkpoint.py`. PyTorch's reference
  large-scale-training implementation of DCP + async save +
  stateful loader integration. The exercises assume this shape.
- **PyTorch `torchdata` StatefulDataLoader.**
  https://github.com/pytorch/data — the resumable dataloader
  used in chapter 3.

## Elastic training and rendezvous

- **PyTorch `torch.distributed.elastic` documentation.**
  https://pytorch.org/docs/stable/elastic/ — rendezvous
  backends, agent design, membership-change handling.
- **`torchrun` documentation.**
  https://pytorch.org/docs/stable/elastic/run.html — the
  launcher's CLI flags (`--nnodes=MIN:MAX`, `--rdzv-backend`,
  `--rdzv-endpoint`, `--rdzv-id`, `--max-restarts`,
  `--monitor-interval`).
- **PyTorch `torch.distributed` overview.**
  https://pytorch.org/docs/stable/distributed.html — the
  `init_process_group` timeout parameter used throughout
  chapters 4 and 5.

## Hardware and node health

- **NVIDIA Data Center GPU Manager (DCGM) documentation.**
  https://docs.nvidia.com/datacenter/dcgm/ — DCGM field IDs
  (`DCGM_FI_PROF_SM_OCCUPANCY`, `DCGM_FI_DEV_GPU_TEMP`,
  `DCGM_FI_DEV_POWER_USAGE`, ECC counters), the `dcgmi dmon`
  and `dcgmi diag` command-line tools, and the health-check
  subsystem. Chapter 6 depends on DCGM as the telemetry
  substrate.
- **`nvidia-smi` documentation.**
  https://docs.nvidia.com/deploy/nvidia-smi/ — the command-line
  reference. `nvidia-smi -q`, `nvidia-smi -q -d ECC`, and the
  XID event log.
- **NVIDIA GPU XID error reference (in the driver / user
  documentation).**
  https://docs.nvidia.com/deploy/xid-errors/ — the canonical
  XID number → cause table used in chapter 5's hardware-fault
  playbook.
- **NVIDIA `dcgmi diag` reference.**
  https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
  — the diagnostic subsystem chapter 5 recommends running
  post-incident.

## Scheduler-side quarantine

- **SLURM `scontrol` documentation.**
  https://slurm.schedmd.com/scontrol.html — the `update
  NodeName=... State=drain` command used in chapter 6's
  auto-quarantine loop, and the `Reason=` field for the audit
  trail.
- **Kubernetes taints and tolerations.**
  https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/
  — the taint / cordon / drain idiom used to quarantine
  Kubernetes nodes.
- **Kubernetes `kubectl drain` documentation.**
  https://kubernetes.io/docs/reference/kubectl/generated/kubectl_drain/
  — the semantics of node drain with respect to running pods
  and eviction.
- **Kueue admission checks.**
  https://kueue.sigs.k8s.io/docs/concepts/admission_check/ —
  the mechanism chapter 6 uses to keep faulty nodes out of
  future workloads (not just the current one).

## Case-study reading order

For a first pass through the module, work the chapters in order
and drop into the case-study material at chapter 7:

1. Chapter 1 of this module. (30 min)
2. PyTorch `torch.distributed.checkpoint` overview + chapter 2
   of this module. (2 h)
3. torchtitan's `checkpoint.py` + chapter 3 of this module. (2 h)
4. PyTorch `torch.distributed.elastic` overview +
   `torchrun` docs + chapter 4 of this module. (2 h)
5. Chapter 5 of this module + skim the OPT-175B logbook to see
   the five incident classes in the wild. (2 h)
6. NVIDIA DCGM overview + chapter 6 of this module. (2 h)
7. Chapter 7 of this module + Llama 3 §6.3 (in full). (2 h)
8. OPT-175B logbook read cover-to-cover (as lab 01). (2 h)

The exercises assume you have done at least items 1–5 before
starting exercise 01, and items 1–7 before exercise 05.

## Cross-module references

- **mod-101** for DTensor / ShardedTensor semantics used by DCP.
- **mod-103** for the sharded, resumable dataset formats
  (WebDataset, MDS) that `StatefulDataLoader` sits on top of.
- **mod-104** for SLURM / Kueue / MPI Operator plumbing that
  chapters 4 and 6 use as tools.
- **mod-105** chapter 8 for the fabric-side NCCL-timeout runbook
  that composes with mod-106 chapter 5's trainer-side runbook.
- **mod-107** for the throughput factor in goodput.
- **mod-108** for the Prometheus / DCGM observability pipeline
  that chapter 6 uses as its detection substrate.
- **mod-109** for turning the availability budget into dollars.
- **mod-110** for how the SLO is exposed to model teams and
  negotiated across teams at platform-org altitude.
