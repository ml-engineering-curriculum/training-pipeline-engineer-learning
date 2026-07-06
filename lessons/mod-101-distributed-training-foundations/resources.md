# Resources for mod-101-distributed-training-foundations

Primary sources first, tooling and docs second. These are the references
the chapters and exercises are grounded in. Skim the framework docs
(PyTorch, NCCL) as-needed while working the exercises; the papers reward
a full read.

## Primary papers — distributed-training foundations

- **Li, S., et al. (2020). "PyTorch Distributed: Experiences on
  Accelerating Data Parallel Training."** *Proceedings of the VLDB
  Endowment.* The reference for DDP's gradient bucketing and overlap
  design. Read alongside chapter 2.
- **Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). "ZeRO:
  Memory Optimizations Toward Training Trillion Parameter Models."**
  *SC20.* The original ZeRO paper — introduces the ZeRO-1 / ZeRO-2 /
  ZeRO-3 memory decomposition. Core reading for chapter 3.
- **Ren, J., et al. (2021). "ZeRO-Offload: Democratizing Billion-Scale
  Model Training."** *USENIX ATC.* The CPU-offload extension of ZeRO.
- **Rajbhandari, S., et al. (2021). "ZeRO-Infinity: Breaking the GPU
  Memory Wall for Extreme Scale Deep Learning."** *SC21.* NVMe-based
  offload and the "infinity" ladder for extreme-scale models.
- **Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-Billion
  Parameter Language Models Using Model Parallelism."** arXiv:1909.08053.
  The canonical tensor-parallel paper. Read alongside chapter 4.
- **Narayanan, D., et al. (2021). "Efficient Large-Scale Language Model
  Training on GPU Clusters Using Megatron-LM."** *SC21.* The follow-up
  that shows how to compose TP + PP + DP into 3D-parallel, including
  interleaved-1F1B pipeline scheduling.
- **Huang, Y., et al. (2019). "GPipe: Efficient Training of Giant Neural
  Networks using Pipeline Parallelism."** *NeurIPS.* The original
  GPipe pipeline schedule and bubble analysis.
- **Narayanan, D., et al. (2019). "PipeDream: Generalized Pipeline
  Parallelism for DNN Training."** *SOSP.* The 1F1B pipeline schedule.
- **Korthikanti, V., et al. (2022). "Reducing Activation Recomputation
  in Large Transformer Models."** arXiv:2205.05198. Introduces
  sequence-parallel for Megatron.
- **Lepikhin, D., et al. (2020). "GShard: Scaling Giant Models with
  Conditional Computation and Automatic Sharding."** arXiv:2006.16668.
  Expert parallel and the routing all-to-all.
- **Fedus, W., Zoph, B., & Shazeer, N. (2021). "Switch Transformers:
  Scaling to Trillion Parameter Models with Simple and Efficient
  Sparsity."** *JMLR.* The engineering follow-up to GShard.
- **Rajbhandari, S., et al. (2022). "DeepSpeed-MoE: Advancing
  Mixture-of-Experts Inference and Training to Power Next-Generation AI
  Scale."** *ICML.*

## Case-study papers (chapter 7)

- **Grattafiori, A., et al. (2024). "The Llama 3 Herd of Models."**
  Meta AI technical paper. The 405B section covers the 4-D parallelism
  (TP × CP × PP × DP), FP8 mixed precision, and training reliability.
- **Le Scao, T., et al. (2022). "BLOOM: A 176B-Parameter Open-Access
  Multilingual Language Model."** arXiv:2211.05100. Megatron-DeepSpeed
  3D-parallel with ZeRO-1 on the Jean Zay supercomputer.
- **Zhang, S., et al. (2022). "OPT: Open Pre-trained Transformer
  Language Models."** arXiv:2205.01068. Read with the accompanying
  **OPT-175B logbook** — the definitive public record of large-scale
  training operational realism.

## Collective-cost and NCCL references

- **Patarasuk, P., & Yuan, X. (2009). "Bandwidth Optimal All-reduce
  Algorithms for Clusters of Workstations."** *JPDC.* The
  ring-all-reduce derivation.
- **Sanders, P., Speck, J., & Träff, J. L. (2009). "Two-tree Algorithms
  for Full Bandwidth Broadcast, Reduction and Scan."** *Parallel
  Computing.* The double-binary-tree algorithm NCCL uses at scale.
- **NVIDIA NCCL Documentation.** https://docs.nvidia.com/deeplearning/nccl/
  — collective operations, algorithm and protocol selection,
  environment variables (`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_DEBUG`,
  `NCCL_TOPO_FILE`), tuning guide.
- **NVIDIA `nccl-tests`.** https://github.com/NVIDIA/nccl-tests —
  reference benchmark suite. Used directly in exercise 3.

## Scaling-law references cited for the FLOP model

- **Kaplan, J., et al. (2020). "Scaling Laws for Neural Language
  Models."** arXiv:2001.08361. Origin of the `6 · N · D` FLOP count
  used in comm-to-compute reasoning.
- **Hoffmann, J., et al. (2022). "Training Compute-Optimal Large
  Language Models."** arXiv:2203.15556. The Chinchilla paper. Refines
  the compute-optimal parameter / data trade.

## Framework docs (primary reference for the code paths)

- **PyTorch `torch.distributed` overview.**
  https://pytorch.org/docs/stable/distributed.html
- **PyTorch DDP.**
  https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
  and the "PyTorch Distributed Overview" tutorial.
- **PyTorch FSDP2 (`fully_shard`).**
  https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html and
  the FSDP2 examples in `torch/distributed/_composable/fsdp/`.
- **PyTorch DeviceMesh & DTensor.**
  https://pytorch.org/docs/stable/distributed.tensor.html and
  https://pytorch.org/docs/stable/distributed.device_mesh.html
- **PyTorch `torch.distributed.tensor.parallel`.**
  https://pytorch.org/docs/stable/distributed.tensor.parallel.html
- **PyTorch elastic (`torchrun`).**
  https://pytorch.org/docs/stable/elastic/run.html
- **Megatron-LM.** https://github.com/NVIDIA/Megatron-LM — README,
  `pretrain_gpt.py`, and the parallel layer implementations.
- **DeepSpeed.** https://www.deepspeed.ai/ — ZeRO configuration,
  ZeRO-Offload, ZeRO-Infinity, and the DeepSpeed training tutorials.
- **torchtitan.** https://github.com/pytorch/torchtitan — PyTorch's
  reference large-scale training implementation using FSDP2 + TP + PP
  through DeviceMesh.

## Recommended reading order for a first pass

1. PyTorch `torch.distributed` overview (30 min).
2. Chapter 1–2 of this module + Li et al., 2020, DDP paper (2 h).
3. Chapter 3 + Rajbhandari et al., 2020, ZeRO paper (2 h).
4. Chapter 4 + Shoeybi et al., 2019, Megatron paper + Narayanan et al.,
   2021, follow-up (3 h).
5. Chapter 5 + Patarasuk & Yuan, 2009 + NCCL docs' collective operations
   chapter (2 h).
6. Chapter 6 + `torch.distributed.fsdp.fully_shard` API docs (1 h).
7. Chapter 7 + one of the three case-study papers of your choice
   (2–4 h; Llama 3 first if you can only pick one).

The exercises assume you have done at least items 1–5 before starting.
