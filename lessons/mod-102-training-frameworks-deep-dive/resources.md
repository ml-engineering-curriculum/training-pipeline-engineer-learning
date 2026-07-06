# Resources for mod-102-training-frameworks-deep-dive

Primary sources first, framework documentation second, reference
implementations third. These are the references the chapters and
exercises are grounded in. Skim the framework docs while working
the exercises; the papers reward a full read.

## Primary papers

### ZeRO family (chapters 3 and 4)

- **Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020).
  "ZeRO: Memory Optimizations Toward Training Trillion Parameter
  Models."** *SC20.* The original ZeRO paper — ZeRO-1 / ZeRO-2 /
  ZeRO-3 memory decomposition. The mental model behind DeepSpeed
  and FSDP both.
- **Ren, J., et al. (2021). "ZeRO-Offload: Democratizing
  Billion-Scale Model Training."** *USENIX ATC.* Introduces the
  CPU offload of optimizer state and the CPU-Adam kernel.
- **Rajbhandari, S., et al. (2021). "ZeRO-Infinity: Breaking the
  GPU Memory Wall for Extreme Scale Deep Learning."** *SC21.*
  NVMe offload, memory-centric tiling, bandwidth-centric
  partitioning.

### Megatron and 3D parallelism (chapters 5 and 6)

- **Shoeybi, M., et al. (2019). "Megatron-LM: Training
  Multi-Billion Parameter Language Models Using Model
  Parallelism."** arXiv:1909.08053. The canonical tensor-parallel
  paper. `ColumnParallelLinear` / `RowParallelLinear` are
  introduced here.
- **Narayanan, D., et al. (2021). "Efficient Large-Scale Language
  Model Training on GPU Clusters Using Megatron-LM."** *SC21.*
  The 3D-parallel composition (TP + PP + DP) and the
  interleaved-1F1B pipeline schedule.
- **Korthikanti, V., et al. (2022). "Reducing Activation
  Recomputation in Large Transformer Models."** arXiv:2205.05198.
  Introduces sequence-parallel in Megatron.
- **Huang, Y., et al. (2019). "GPipe: Efficient Training of Giant
  Neural Networks using Pipeline Parallelism."** *NeurIPS.* The
  original GPipe schedule and bubble analysis.
- **Narayanan, D., et al. (2019). "PipeDream: Generalized
  Pipeline Parallelism for DNN Training."** *SOSP.* The 1F1B
  pipeline schedule.

### JAX / GSPMD (chapter 7)

- **Xu, Y., et al. (2021). "GSPMD: General and Scalable
  Parallelization for ML Computation Graphs."** arXiv:2105.04663.
  The compiler-driven sharding model JAX uses on TPU and GPU.
- **Lepikhin, D., et al. (2020). "GShard: Scaling Giant Models
  with Conditional Computation and Automatic Sharding."**
  arXiv:2006.16668. GSPMD's precursor and the origin of the
  `PartitionSpec`-style annotation model.

### Case-study papers referenced in exercises and chapter 8

- **Grattafiori, A., et al. (2024). "The Llama 3 Herd of
  Models."** Meta AI technical paper. The 405B section covers a
  4-D parallelism (TP × CP × PP × DP), FP8 mixed precision, and
  training reliability. The reference example for
  Megatron-style 3D-parallel at scale.
- **Le Scao, T., et al. (2022). "BLOOM: A 176B-Parameter
  Open-Access Multilingual Language Model."** arXiv:2211.05100.
  Megatron-DeepSpeed 3D-parallel with ZeRO-1 on the Jean Zay
  supercomputer.
- **Zhang, S., et al. (2022). "OPT: Open Pre-trained Transformer
  Language Models."** arXiv:2205.01068. Read alongside the
  **OPT-175B logbook** — a training-team operational-realism
  document.

### MoE (chapter 8 sidebar)

- **Fedus, W., Zoph, B., & Shazeer, N. (2021). "Switch
  Transformers: Scaling to Trillion Parameter Models with Simple
  and Efficient Sparsity."** *JMLR.*
- **Rajbhandari, S., et al. (2022). "DeepSpeed-MoE: Advancing
  Mixture-of-Experts Inference and Training to Power
  Next-Generation AI Scale."** *ICML.*

## Framework documentation

### PyTorch FSDP2 and DTensor (chapter 2)

- **`torch.distributed.fsdp.fully_shard`.**
  https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- **PyTorch DTensor.**
  https://pytorch.org/docs/stable/distributed.tensor.html
- **PyTorch DeviceMesh.**
  https://pytorch.org/docs/stable/distributed.device_mesh.html
- **`torch.distributed.tensor.parallel`.**
  https://pytorch.org/docs/stable/distributed.tensor.parallel.html
- **PyTorch Distributed Checkpoint (DCP).**
  https://pytorch.org/docs/stable/distributed.checkpoint.html
- **`torch.profiler`.**
  https://pytorch.org/docs/stable/profiler.html

### DeepSpeed (chapters 3 and 4)

- **DeepSpeed homepage and tutorials.** https://www.deepspeed.ai/
- **DeepSpeed ZeRO configuration reference.**
  https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training
- **DeepSpeed ZeRO-Offload / ZeRO-Infinity tutorials.**
  https://www.deepspeed.ai/tutorials/zero-offload/ and
  https://www.deepspeed.ai/tutorials/zero/
- **DeepSpeedExamples repository.**
  https://github.com/microsoft/DeepSpeedExamples — reference
  training configurations, including model conversion scripts.

### Megatron-LM / Megatron-Core (chapters 5 and 6)

- **Megatron-LM.** https://github.com/NVIDIA/Megatron-LM — README,
  `pretrain_gpt.py`, `tools/preprocess_data.py`, and the parallel
  layer implementations.
- **Megatron-Core documentation.** Distributed with the
  Megatron-LM repository under `megatron/core/`; also referenced
  from NeMo's docs as `mcore`.

### NeMo (chapter 6)

- **NVIDIA NeMo repository and docs.**
  https://github.com/NVIDIA/NeMo — main entry point; the
  `nemo/collections/llm` package contains the LLM training and
  recipe surface for NeMo 2.0.
- **NeMo LLM guide.**
  https://docs.nvidia.com/nemo-framework/user-guide/latest/ —
  installation, configuration, and current supported recipes.
- **PyTorch Lightning documentation.**
  https://lightning.ai/docs/pytorch/stable/ — for the Trainer /
  Strategy / Callback abstractions NeMo layers on.

### JAX / Flax (chapter 7)

- **JAX distributed arrays and automatic parallelization
  tutorial.**
  https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- **`jax.jit` API reference.**
  https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
- **`jax.experimental.shard_map` documentation.**
  https://docs.jax.dev/en/latest/notebooks/shard_map.html
- **`jax.sharding` module.**
  https://docs.jax.dev/en/latest/jax.sharding.html
- **JAX profiler.**
  https://docs.jax.dev/en/latest/profiling.html
- **Flax Linen docs.** https://flax-linen.readthedocs.io/
- **Flax NNX docs.** https://flax.readthedocs.io/en/latest/nnx_basics.html
- **Optax (JAX optimizer library).**
  https://optax.readthedocs.io/

### NCCL and low-level references (all chapters)

- **NVIDIA NCCL documentation.**
  https://docs.nvidia.com/deeplearning/nccl/ — collective
  operations, algorithm and protocol selection, environment
  variables. Also referenced from mod-101 chapter 5.

## Reference implementations

- **torchtitan.** https://github.com/pytorch/torchtitan — PyTorch's
  reference large-scale training implementation using FSDP2 + TP +
  PP through DeviceMesh. Primary reference for the exercise-1
  FSDP2 leg.
- **DeepSpeedExamples/pretraining.**
  https://github.com/microsoft/DeepSpeedExamples/tree/master/training —
  DeepSpeed ZeRO-3 pretraining reference configurations.
- **NVIDIA/Megatron-LM `pretrain_gpt.py`.**
  https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py —
  Megatron-LM's canonical GPT pretraining entry point.
- **NVIDIA/NeMo `examples/nlp/language_modeling`.**
  https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/language_modeling
  — NeMo GPT pretraining scripts and configs.
- **Google/MaxText.** https://github.com/google/maxtext — Google's
  reference JAX pretraining implementation. Runs on both TPU and
  GPU; the JAX analogue of torchtitan.

## Recommended reading order for a first pass

1. Chapter 1 of this module (30 min).
2. Chapter 2 + PyTorch `fully_shard` and DTensor docs (1 h).
3. Chapter 3 + ZeRO paper (Rajbhandari et al., 2020) +
   DeepSpeed ZeRO tutorial (2 h).
4. Chapter 4 + ZeRO-Offload paper (Ren et al., 2021) +
   ZeRO-Infinity paper (Rajbhandari et al., 2021) (2 h).
5. Chapter 5 + Megatron-LM paper (Shoeybi et al., 2019) +
   the 3D-parallel follow-up (Narayanan et al., 2021) (3 h).
6. Chapter 6 + NeMo LLM guide's overview page (1 h). Skim the
   recipes; do not read all of them.
7. Chapter 7 + JAX distributed-arrays tutorial + `shard_map`
   docs (2 h).
8. Chapter 8 + one case-study paper (Llama 3 if you can only
   pick one) (2 h).

The exercises assume you have done at least chapters 1–5's
references, and that you have exercise 1 from mod-101 as
background muscle memory.
