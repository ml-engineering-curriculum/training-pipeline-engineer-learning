# Megatron-LM and 3D Parallelism in Practice

Megatron-LM (Shoeybi et al., 2019, "Megatron-LM: Training
Multi-Billion Parameter Language Models Using Model Parallelism";
Narayanan et al., 2021, "Efficient Large-Scale Language Model
Training on GPU Clusters Using Megatron-LM") is the reference
implementation of tensor-parallel and 3D-parallel training. If the
Megatron paper is where the algorithm lives, the Megatron-LM
repository is where the production implementation lives. Almost
every 70B-plus pretraining run in the open literature — Llama 3,
BLOOM, MT-NLG — either used Megatron-LM directly or a fork of it.

This chapter walks the Megatron-LM building blocks:
parallelism-aware layers, the way TP × PP × DP compose into 3D
parallel, the interleaved-1F1B pipeline schedule, and the more
recent Megatron-Core library (the reusable extract that NeMo, in
chapter 6, sits on top of).

## The three parallelism axes as configured

Megatron-LM asks you to explicitly configure the three axes at
launch:

```bash
python pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --num-layers 80 \
  --hidden-size 8192 \
  --num-attention-heads 64 \
  --micro-batch-size 2 \
  --global-batch-size 1024 \
  --sequence-length 8192 \
  --seq-length 8192 \
  --max-position-embeddings 8192 \
  --train-iters 300000 \
  --bf16 \
  --distributed-backend nccl \
  ...
```

The three sizes multiply to give total GPU count. On a `world_size`
of 512, the example above uses TP=8, PP=4, and DP is inferred as
`512 / (8 * 4) = 16`. That decomposition is not decoration — it
literally chooses which process group each rank belongs to for
tensor-parallel all-reduces vs. pipeline-parallel sends vs.
data-parallel all-reduces.

There is an ordering convention Megatron enforces: for a world
laid out as an outer DP × PP × TP hyper-rectangle, the TP group
is the *innermost* (fastest-changing rank), the PP group is next,
and the DP group is outermost. On a typical NVLink-per-node
cluster with 8 GPUs per node, TP=8 fills exactly one node, which
puts every TP all-reduce over NVLink (fast) and every PP
send/recv across nodes (slow but infrequent).

## The parallelism-aware layers

The Megatron implementation of tensor-parallel does not use
`torch.distributed.tensor.parallel`; it uses hand-written
parallelism-aware layers that predate DTensor. The three you must
know:

- **`VocabParallelEmbedding`** — an embedding table sharded on the
  vocab dimension. Forward is a local lookup for the tokens in
  this rank's vocab slice, followed by an `all-reduce` to combine
  the outputs across the TP group. Handles the mask for tokens
  that map to a shard held by another rank.
- **`ColumnParallelLinear`** — a linear layer whose output
  dimension is sharded across the TP group. Weight is
  `[out_features / TP, in_features]`, input is replicated across
  TP, and no communication is needed on forward (each rank
  computes its slice of the output). Backward requires an
  `all-reduce` on the input gradient. Optionally emits the sharded
  output directly (for chaining into a following `RowParallelLinear`),
  or gathers it via `all-gather` for chaining into a non-parallel
  layer.
- **`RowParallelLinear`** — a linear layer whose input dimension is
  sharded across the TP group. Weight is
  `[out_features, in_features / TP]`, each rank owns its slice of
  the input, and after the local matmul an `all-reduce` combines
  the partial output. Backward is symmetric.

The Megatron block for a decoder-only transformer uses these in a
specific pattern:

- QKV projection: `ColumnParallelLinear` — output is sharded on the
  head dimension.
- Output projection after attention: `RowParallelLinear` — takes
  the head-sharded activation and reduces back.
- MLP up-projection (`gate_proj` / `up_proj` in Llama-style
  gated MLPs): `ColumnParallelLinear`.
- MLP down-projection: `RowParallelLinear`.
- LM head: `ColumnParallelLinear` over vocab, paired with
  `VocabParallelEmbedding` at the input.

The pattern (column then row) is intentional: two consecutive
matmuls with a sharded activation between them require only *one*
all-reduce (on the row's output), not two. That is the fundamental
Megatron insight for TP throughput.

## Sequence-parallel and context-parallel

Two later refinements that both compose with tensor-parallel:

- **Sequence-parallel** (Korthikanti et al., 2022, "Reducing
  Activation Recomputation in Large Transformer Models") — inside
  a TP group, shard the *activations* of LayerNorm / Dropout on
  the sequence dimension instead of replicating them. This changes
  the TP all-reduce into an `all-gather` + `reduce-scatter`
  pattern and roughly halves activation memory in these layers.
  In Megatron, it is enabled by `--sequence-parallel`.
- **Context-parallel** — shard the sequence dimension across an
  additional axis, so that a very long sequence (32k, 128k+ tokens)
  can be split across multiple GPUs. Used prominently by Llama 3
  for its 128k context; introduced into Megatron-Core as CP with
  a Ring Attention-style algorithm. Enabled by
  `--context-parallel-size`.

Both require your model to be built out of Megatron layers, because
the sharding of activations is baked into the layer implementation.

## Pipeline parallelism and interleaved-1F1B

Pipeline parallel in Megatron is not a wrapper around your model;
it is a *scheduler* that runs a partitioned model in a specific
micro-batch order. The two schedules Megatron implements:

- **1F1B** (One-Forward-One-Backward), from PipeDream (Narayanan
  et al., 2019). After warmup, each rank alternates between one
  forward and one backward micro-batch, keeping activation memory
  low.
- **Interleaved-1F1B**, from Narayanan et al., 2021. Each pipeline
  stage is *split further* into `virtual_pipeline_model_parallel`
  chunks, and the ranks interleave through them. This reduces the
  pipeline bubble at the cost of more sends/receives.

Config knobs:

- `--pipeline-model-parallel-size P` — number of stages.
- `--num-layers-per-virtual-pipeline-stage L` — enables interleaved
  scheduling with virtual pipeline chunks.
- `--pipeline-model-parallel-split-rank` — for encoder-decoder
  models, where the encoder stops and the decoder starts.

The bubble analysis (from Narayanan et al., 2021) is the reason
you tune these: with `M` micro-batches and `P` stages, the
non-interleaved 1F1B bubble is roughly `(P - 1) / M`, so you need
`M >> P` for pipeline efficiency. Interleaved reduces this by a
factor of `V` (virtual chunks per stage).

## The 3D composition

3D-parallel means composing all three axes. On the 512-GPU example
above with TP=8, PP=4, DP=16:

- **TP=8** intra-node. Every attention/MLP block emits
  activation-sized all-reduces on NVLink.
- **PP=4** across nodes. Micro-batches flow through the pipeline;
  each rank sends its stage's output activations and receives
  the next stage's input activations from adjacent stages. Two
  sends per micro-batch.
- **DP=16** across the outer axis. At the end of each global batch,
  ranks that hold corresponding parameter shards `all-reduce` (or
  `reduce-scatter` with ZeRO-1) their gradients across the DP
  group.

Megatron-LM historically implemented plain DP (with optional
Distributed Optimizer for ZeRO-1-style optimizer sharding). Recent
releases add Distributed Optimizer with a broader range of ZeRO
behaviors. For a first-order 3D-parallel run, `--distributed-optimizer`
gives you the memory savings without extra configuration.

## The training loop, viewed from outside

Unlike DeepSpeed or FSDP2, Megatron-LM does not present a
"wrap-your-model" API. You use its main pretraining scripts
(`pretrain_gpt.py`, `pretrain_bert.py`, `pretrain_t5.py`) or
adapt one of them. The scripts wire together:

- Model builder — assembles a `GPTModel` out of Megatron's
  parallel layers. The block structure is fixed; the number of
  layers/heads/hidden is configurable.
- Data pipeline — Megatron's own `GPTDataset`, which reads from
  the pre-tokenized `.bin` + `.idx` format the repo's
  preprocessing script produces (see `tools/preprocess_data.py`).
- Optimizer builder — Megatron's `MegatronOptimizer` wraps
  `Adam` / `AdamW` with the Distributed Optimizer and per-parameter
  main-grad accumulation for FP32 gradient reduction.
- Training loop — an iteration-based loop that logs at fixed
  intervals, checkpoints via Megatron's `save_checkpoint`, and
  handles validation.

If you want a novel architecture, you have to build it out of
Megatron's parallel primitives yourself. That is a real cost — it
is not "point at a HuggingFace `LlamaForCausalLM` and go". The
Megatron-Core library (below) somewhat lowers that bar.

## Megatron-Core: the extracted library

Historically Megatron-LM was a monolithic training repository.
Megatron-Core (`megatron.core.*` in the same GitHub repo) is a
factored-out library of the parallel primitives so that
downstream frameworks — NeMo, TransformerEngine, custom
research forks — can build models out of them without adopting the
whole training script.

The Megatron-Core surface includes:

- `megatron.core.tensor_parallel` — `ColumnParallelLinear`,
  `RowParallelLinear`, `VocabParallelEmbedding`.
- `megatron.core.parallel_state` — the process-group setup
  (`initialize_model_parallel(tp, pp, cp, ep, ...)`).
- `megatron.core.transformer` — a configurable transformer block
  and language-model definition, TP-aware.
- `megatron.core.pipeline_parallel` — the schedule engine (1F1B
  and interleaved-1F1B) exposed as a `schedule_forward_backward`
  API.
- `megatron.core.optimizer` — the Distributed Optimizer and the
  `chained_optimizer` for mixed-parameter groups.
- `megatron.core.dist_checkpointing` — the sharded checkpoint
  format.

NeMo (chapter 6) is what you get when you wrap Megatron-Core in
PyTorch Lightning and add a Hydra config surface.

## Common Megatron pitfalls

Practical things that trip a first-time Megatron user:

- **World-size mismatch.** `world_size` must be exactly
  `TP × PP × DP × CP × EP`. Off-by-one gets you a hard error at
  `initialize_model_parallel`.
- **TP crossing a node.** As mod-101 chapter 6 warned, TP over
  IB/RoCE is catastrophic. Verify with `NCCL_DEBUG=INFO` that TP
  groups stay intra-node.
- **Data preprocessing.** Megatron's data loader wants pre-tokenized
  `.bin`/`.idx` blobs. The `tools/preprocess_data.py` script is the
  reference; skipping this and trying to plug in an ad hoc
  DataLoader breaks the pipeline schedule that assumes fixed-shape
  batches.
- **Deterministic dropout.** With TP, the RNG has to agree for
  dropout applied to a TP-replicated tensor. Megatron manages this
  with `get_cuda_rng_tracker()`; if you write a custom layer, you
  must use it.
- **Checkpoint format.** Megatron's sharded checkpoints are not
  HuggingFace-loadable directly. The conversion scripts in
  `tools/checkpoint/` (or the equivalent in NeMo) handle this.
- **Interleaved-1F1B and imbalanced stages.** If your model's
  layers are not evenly divisible across pipeline stages, the
  slowest stage bottlenecks the pipeline. Prefer layer counts that
  divide cleanly by `PP × V`.

## When to use Megatron-LM directly vs. NeMo vs. FSDP2

- **Use Megatron-LM directly** when you are doing large-scale
  pretraining (7B+), you want the reference 3D-parallel
  implementation, and you are comfortable working in the Megatron
  script ecosystem. The Llama 3 405B paper is the canonical
  example of this workflow.
- **Use NeMo** when you want the Megatron capabilities plus the
  Lightning `Trainer`, the built-in dataloading recipes, and the
  post-training pipelines (SFT, RLHF, alignment). Chapter 6.
- **Use FSDP2** when you can express your parallelism as DP + TP
  (no pipeline needed), you want to keep imperative PyTorch, and
  your model is a custom `nn.Module` you own. `torchtitan` shows
  this composition can go quite far — a 405B-class model is
  demonstrated in the repository — but for production 3D-parallel
  the Megatron / NeMo pathway is the well-trodden one.

Exercise 1 asks you to run the same 3B model in all three; the
choice of "which stack for team X" (chapter 8) grows out of that
hands-on comparison.

## Summary

- Megatron-LM is the reference 3D-parallel implementation.
  Parallelism is configured via CLI (`--tensor-model-parallel-size`,
  `--pipeline-model-parallel-size`) and enforced by
  parallelism-aware layers (`ColumnParallelLinear`,
  `RowParallelLinear`, `VocabParallelEmbedding`).
- The column-then-row pattern for MLPs and attention emits one
  TP `all-reduce` per pair of matmuls; sequence-parallel adds
  activation shard collectives on LN/Dropout.
- Pipeline scheduling is 1F1B or interleaved-1F1B, with bubbles
  scaling as `(P-1)/M`.
- Megatron-Core is the extracted library. NeMo (chapter 6) and
  research forks build on top of it.
- Megatron is not "plug and play" — you construct models out of
  its layers or use one of its `pretrain_*.py` scripts. That cost
  is real and factors into the stack-choice decision.
