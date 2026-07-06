# The Training-Framework Landscape

There are three ways to write a modern large-scale training job. You
can pick up PyTorch and call `torch.distributed` primitives yourself.
You can adopt an opinionated engine (DeepSpeed, Megatron-LM, NeMo)
that fills in more of the loop for you. Or you can use JAX and let a
compiler decide where the collectives go. Every stack in this module
sits somewhere on that spectrum, and each was designed to solve a
different flavor of the same problem.

This chapter is the map. It explains what each stack is optimized for,
what they share underneath, and how they relate to the algorithms from
mod-101 (DDP, FSDP2, ZeRO-3, TP, PP, EP). The goal is that by the end
you can look at a training-job repo and, from the imports alone, guess
which trade-offs it made.

## Why five stacks

At first glance the stacks look redundant. All of them can train a
transformer. All of them wrap `torch.distributed` or an equivalent.
So why does the field maintain five?

Each stack was born in a different context, and those contexts baked
in different design commitments:

- **PyTorch FSDP2** is Meta's answer to "we want ZeRO-3 as a
  first-class PyTorch feature that composes with the rest of the
  ecosystem." It is aggressively imperative: you keep writing
  `nn.Module`s and a `torch.optim` optimizer, and `fully_shard`
  changes what tensors those modules hold under the hood.
- **DeepSpeed** started at Microsoft as the reference implementation
  of the ZeRO paper. It is organized around a `deepspeed.initialize`
  call and a JSON configuration file. Almost every knob — ZeRO stage,
  offload targets, gradient accumulation, mixed precision — is a JSON
  field, not Python code.
- **Megatron-LM** started at NVIDIA as the reference implementation
  of the tensor-parallel paper (Shoeybi et al., 2019). It is
  organized around parallelism-aware layers
  (`ColumnParallelLinear`, `RowParallelLinear`,
  `VocabParallelEmbedding`) and expects you to construct the model
  out of those layers. TP, PP, and DP compose into 3D-parallel by
  configuration.
- **NeMo** is NVIDIA's opinionated training framework built on top
  of Megatron-Core (the extracted-library form of Megatron-LM) and
  PyTorch Lightning. It targets *productized* training: recipes for
  Llama-style pretraining, SFT, and RLHF, with Hydra/YAML
  configuration and a Lightning `Trainer`.
- **JAX/Flax** is Google's stack. Instead of imperatively inserting
  collectives, you write a pure function of parameters and inputs,
  annotate its arguments with sharding, and let the XLA compiler
  (via GSPMD) rewrite it into a distributed program. `pjit` was the
  original entry point; in current JAX (`jax.Array` unified array,
  post-0.4) the same job is expressed with
  `jax.jit(in_shardings=..., out_shardings=...)` and `shard_map`.

The stacks differ less in *what* they compute than in *who decides
what to compute where*: you (PyTorch), a config file (DeepSpeed,
Megatron-LM, NeMo), or a compiler (JAX).

## The common substrate

Almost every PyTorch-family stack in this module sits on the same
runtime:

- `torch.distributed` for process groups.
- NCCL for GPU-to-GPU collectives.
- CUDA / cuDNN for the actual kernels.
- Optionally `torch.compile` / Inductor for kernel fusion.

Under the hood, FSDP2, DeepSpeed ZeRO-3, and Megatron-LM's DP axis are
all issuing NCCL `all-gather`, `reduce-scatter`, and `all-reduce`
against process groups you can enumerate with
`torch.distributed.get_process_group_ranks`. The cost model from
mod-101 chapter 5 (α + β, ring vs. tree, effective vs. nominal
bandwidth) is the same in all of them. The difference is entirely in
what layer of the stack decides which collective happens when.

JAX is the exception: it uses XLA, which on NVIDIA GPUs still calls
NCCL underneath but does so through the compiler's own runtime rather
than through `torch.distributed`. In practice, when you profile a JAX
job on GPUs, you still see NCCL kernels on the timeline.

## The stacks at a glance

The following table is deliberately opinionated. Each row is
approximate — every stack can be pushed outside its comfort zone —
but the "sweet spot" column captures why each stack tends to be
chosen.

| Stack                | Sweet spot                                                    | Primary sharding API                                     | Config style                          | 3D-parallel? | Notes                                       |
|----------------------|---------------------------------------------------------------|----------------------------------------------------------|---------------------------------------|:------------:|---------------------------------------------|
| PyTorch FSDP2        | Custom `nn.Module`s at 1B–70B on a small–medium cluster       | `fully_shard(module, mesh=...)` on `nn.Module` blocks    | Python                                | Via 2-D DeviceMesh + `torch.distributed.tensor.parallel` | The default for new PyTorch-native training code. |
| DeepSpeed            | Existing `transformers` models needing memory reduction, or extreme offload (NVMe) | `deepspeed.initialize(config=...)`                        | JSON                                  | Via `DeepSpeed-Megatron` bridge or pipeline modules | Big win when you can't rewrite the model to use TP-aware layers. |
| Megatron-LM          | Pre-training at 7B+ where 3D-parallel is required                | `ColumnParallelLinear` / `RowParallelLinear` / TP+PP+DP config | CLI flags + Python                    | Yes (native)  | The reference implementation for TP and interleaved-1F1B. |
| NeMo                 | Recipe-driven productized training and post-training pipelines   | Megatron-Core underneath, wrapped in PL Trainer          | Hydra / YAML                          | Yes (via Megatron-Core) | Buys you data + logging + eval infrastructure; costs you an abstraction layer to debug through. |
| JAX / Flax           | Compiler-driven sharding, TPU-first, GPU-viable                 | `jax.jit(in_shardings=NamedSharding(mesh, spec))`         | Python                                | Yes (GSPMD)   | The mental model *inverts*: you say what data is where; the compiler picks the collectives. |

A team can and does mix these. A common composition is Megatron-LM's
TP + PP + DP for a pretraining run, with the resulting checkpoint
converted for HuggingFace `transformers` fine-tuning downstream. NeMo
explicitly bridges that pipeline. Chapter 5 comes back to this.

## What is the "same" training job across stacks?

The bake-off in exercise 1 runs a ~3B decoder-only transformer through
three stacks and expects the loss curves to *match* (within
numerical tolerance) on the same synthetic dataset. That is a
non-trivial claim, because each stack composes the algorithm slightly
differently. The invariants that must hold across all three:

1. **Global batch size** is identical. If FSDP2 runs `micro_batch=4`
   × 8 ranks and DeepSpeed runs `micro_batch=8` × 4 ranks with 2
   gradient-accumulation steps, both must produce a global batch of
   32. Otherwise the loss trajectory is not comparable.
2. **Optimizer semantics** are identical. AdamW with the same
   `(lr, betas, eps, weight_decay, warmup)`. The FP32 master copy of
   the parameter must exist in all three; without it you are
   comparing Adam-in-BF16 to Adam-in-FP32.
3. **Mixed-precision policy** is the same. Params in BF16, reductions
   in FP32 (or all in BF16, but consistently). See chapter 2 for the
   FSDP2 knob and chapter 3 for the DeepSpeed equivalent.
4. **Sequence length, tokenizer, and data ordering** are identical.
   A fixed seed on a `DistributedSampler` (PyTorch) has to match the
   fixed seed on the Megatron `MMapDataset`.

If any of these drift, the "same" model diverges to a different loss
without any actual bug in the parallelism, and the bake-off tells you
nothing about the frameworks. Chapters 2, 3, and 5 name the knobs in
each stack that enforce these invariants.

## The observability gap

A subtler difference between the stacks is what a *training engineer*
sees when the job misbehaves:

- **FSDP2** — you see PyTorch stack traces and a `torch.profiler`
  timeline. Collectives appear as `NCCL:all_gather` /
  `NCCL:reduce_scatter` events. Failures are usually shape mismatches
  or NCCL timeouts.
- **DeepSpeed** — you see DeepSpeed engine logs, an
  `engine.step()`-shaped stack, and the ZeRO-3 partitioner's own
  events. Failures often manifest as OOM inside the offload
  transfer or as a config mismatch that DeepSpeed refuses at
  `initialize`.
- **Megatron-LM** — you see Megatron's own logging (training
  configuration dump, iteration timings, TFLOPs estimates), and the
  parallel-groups are explicit. Failures often manifest as a group
  mismatch (e.g., `pipeline_model_parallel_size` incompatible with
  world size).
- **NeMo** — you see PyTorch Lightning's `Trainer` output on top of
  Megatron-Core's logs. The extra layer is helpful when it maps to
  what you want (`trainer.fit(model, datamodule)`) and painful when
  it doesn't (Lightning callbacks obscure a low-level Megatron
  error).
- **JAX** — you see JAX/XLA errors (often HLO dumps), and profile
  through `jax.profiler` producing traces you view in TensorBoard or
  Perfetto. Sharding mismatches surface as `NotImplementedError`s
  from XLA's SPMD partitioner rather than as runtime shape errors.

Observability shows up in the decision doc (chapter 8). A team that
has never read an XLA HLO dump before will pay a real cost adopting
JAX, and that cost belongs in the trade-off analysis.

## What to keep in mind while reading the rest of this module

Two lenses to hold:

1. **The five-collective decomposition still applies.** No matter what
   the stack's surface API says, its forward pass is issuing some
   combination of `all-gather`, `reduce-scatter`, `all-reduce`, and
   sometimes `all-to-all`. When you get lost in a config file, pull
   back to "which collective is this system issuing, on which process
   group, at what payload size" — that is the same question mod-101
   chapter 5 asks, in the same language.
2. **The abstraction layer is a design choice.** DeepSpeed's JSON
   config buys you fewer lines of code and costs you a debugger you
   understand. Megatron-LM's TP-aware layers buy you a proven 3D
   implementation and cost you the ability to drop in any HuggingFace
   architecture. JAX's compiler buys you sharding-by-declaration and
   costs you a compilation step and an unfamiliar debugger. None of
   these trades is free, and the module treats them as first-class
   engineering decisions rather than as taste.

## Summary

- Five stacks exist because they baked in different trade-offs at
  birth. PyTorch FSDP2 is imperative and PyTorch-native; DeepSpeed is
  config-driven with strong offload; Megatron-LM is the canonical
  3D-parallel implementation; NeMo wraps Megatron-Core in Lightning
  for productized training; JAX/Flax is compiler-driven and TPU-first.
- The PyTorch-family stacks all share `torch.distributed` + NCCL
  underneath. The mod-101 cost model applies to all of them.
- The bake-off in exercise 1 tests whether you can enforce the
  invariants that make cross-stack comparisons meaningful.
- Observability differs sharply across stacks and is a real input
  into the "which stack" decision.
