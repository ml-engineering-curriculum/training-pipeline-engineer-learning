# Prerequisites for the Training Pipeline Engineer track

The Training Pipeline Engineer track sits at **Level 35** — a deep-specialist architect for large-scale distributed-training platforms. The curriculum assumes the following prerequisite depth. Learners without it should complete the linked lower-level tracks first.

## Assumed skills (do NOT re-teach in this track)

### PyTorch and ML practitioner fundamentals — [`ml-engineer-learning`](https://github.com/ml-engineering-curriculum/ml-engineer-learning) (Level 20)

- Fluent PyTorch: `nn.Module`, `Dataset`/`DataLoader`, autograd, single-GPU training loop, `torch.optim`, LR schedulers.
- Reading and reproducing a paper's training curve on a single GPU.
- FastAPI + Docker packaging of a trained model.
- MLflow experiment tracking, run comparison, artifact logging.
- Classical ML metrics + evaluation discipline.

### Infrastructure engineering fundamentals — [`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) (Level 20)

- Docker: multi-stage builds, image hardening, non-root, buildx.
- Kubernetes: pods, deployments, services, StatefulSets, DaemonSets, resource requests/limits, taints/tolerations, node affinity, PVCs. GPU device plugin at the operator altitude.
- One major cloud (AWS / GCP / Azure) at the operator altitude: IAM, VPC, object storage, VM/instance types (especially GPU SKUs), managed K8s.
- Terraform (or comparable IaC) for cluster + storage provisioning.
- Linux systems administration: cgroups, namespaces, systemd, kernel networking basics, `perf` / `ftrace` awareness.

### Engineering-craft baseline — [`ai-infra-junior-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning) (Level 10)

- Git fluency, PR discipline, code review.
- Python packaging discipline (`pyproject.toml`, `pip` / `uv`, virtualenv, editable installs).
- Reading and writing YAML config, Makefiles, shell.
- Networking primitives — TCP, DNS, TLS, HTTP, socket basics.
- Bash / observability instincts: `htop`, `iostat`, `nvidia-smi`, `dmesg`, `journalctl`.

## Adjacent knowledge assumed (surface familiarity, not depth)

- Transformer architecture at the level of "I can trace a forward + backward pass through a decoder-only block". The mod-101 opener rebuilds intuition for parallelism reasoning but does not re-teach transformer internals — see [`fine-tuning-engineer-learning/mod-101`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning) for the practitioner-altitude teardown.
- Fine-tuning workflow at the customer altitude — SFT, LoRA, DPO. The Training Pipeline Engineer builds the platform underneath these workflows; the workflow itself is owned by [`fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning). Familiarity is enough.
- HPC vocabulary — MPI, OpenMP, NUMA, NUMA-aware placement, IB verbs. Depth is developed in mod-105; surface familiarity going in is expected.
- CUDA at the *user* altitude — `nvidia-smi`, streams, memcpy, understanding "what a warp is". Kernel authoring is out of scope and is owned by [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) (peer at Level 35).

## Skills explicitly NOT prerequisite (this track owns the depth)

- Sharded training strategies (FSDP2 / DeepSpeed ZeRO / Megatron-LM 3D-parallel) — owned in **mod-101 and mod-102**.
- Training-scale data pipelines (WebDataset / MosaicML Streaming / Ray Data / distributed tokenization / dedup at scale) — owned in **mod-103**.
- Training-cluster scheduling (SLURM / Kueue / Volcano / KubeRay / MPI Operator / TorchX) — owned in **mod-104**.
- NCCL / IB / RoCEv2 / GPUDirect / parallel filesystems for training — owned in **mod-105**.
- Distributed checkpointing (DCP) / elastic training / fault-tolerance playbooks — owned in **mod-106**.
- MFU / mixed-precision (BF16 / FP8) / FlashAttention v2/v3 integration / communication-compute overlap — owned in **mod-107**.
- Training-run observability at scale — owned in **mod-108**.
- Chinchilla-scale compute economics — owned in **mod-109**.
- Training-platform architecture and cross-team leadership at Level 35 — owned in **mod-110** and the capstone project.

## Suggested entry check

Before starting mod-101, a learner should be able to:

1. Write a single-GPU PyTorch training loop for a decoder-only transformer from scratch, using their own tokenizer + `Dataset` + `DataLoader`, in under 3 hours.
2. Package that training script into a Docker image, run it in a Kubernetes Job on a real GPU pod, and stream logs back with `kubectl logs -f`.
3. Read the OPT-175B logbook headers and identify at least three operational failure classes that the trainers had to plan for.
4. Reason about — even if not implement — what would go wrong if the same 7B model were split across 8 GPUs with data parallel vs. tensor parallel vs. pipeline parallel.

A learner who cannot yet do (1) or (2) should route back through [`ml-engineer-learning`](https://github.com/ml-engineering-curriculum/ml-engineer-learning) and [`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning). A learner comfortable with (1)–(2) but unsure about (3)–(4) is exactly the target audience for **mod-101**.
