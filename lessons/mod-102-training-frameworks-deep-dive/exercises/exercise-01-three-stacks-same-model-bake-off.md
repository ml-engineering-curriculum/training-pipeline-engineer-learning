# exercise-01: Three-Stacks Same-Model Bake-Off

**Estimated effort:** 6 hours

## Objective

Run the *same* ~3B-parameter decoder-only transformer through three
training stacks — PyTorch FSDP2 (torchtitan-style), DeepSpeed
ZeRO-3, and Megatron-LM 3D-parallel — on the same hardware and the
same synthetic dataset, then produce a bake-off report comparing
peak HBM per GPU, tokens/sec/GPU, and step-time breakdown. The
deliverable is the artifact that will anchor exercise 6-style
decision docs (chapter 8) for the rest of your training-team
career: "here is what these three stacks actually cost, on our
hardware, on a real 3B model, when configured to be comparable."

This is the module's centerpiece exercise. Chapters 2, 3, and 5
give you the API surface for each stack; this exercise puts them
head-to-head.

## Prerequisites

- Chapters 1–5 of this module. Chapter 8 sets up the write-up
  rubric.
- Exercises from mod-101 (DDP → FSDP2 → Megatron-TP ladder) so
  the FSDP2 leg is not the first time you have seen `fully_shard`.
- A multi-node or multi-GPU environment with at least 8 GPUs and
  enough HBM to hold a 3B model in BF16 sharded 8-ways
  (comfortably: 8×80 GB H100; workable: 8×40 GB A100 with
  activation checkpointing enabled).
- Installed and importable: PyTorch ≥ 2.4, DeepSpeed (recent
  release), Megatron-LM (repository checked out and
  `pip install -e .`).
- `NCCL_DEBUG=INFO` and `torch.profiler` from mod-101 exercise 3.

## Problem statement

Your team is starting a new pretraining project. Engineering
leadership has said "run a 3B pilot to de-risk the 30B", and you
have been asked to pick the training stack. Before writing a
recommendation, you want three data points on your own hardware —
FSDP2, DeepSpeed ZeRO-3, and Megatron-LM 3D-parallel — running the
same model to the same loss.

The output is not "which is fastest". The output is the report
that would let the next engineer on the team make an informed
choice.

## Requirements

### The reference model

- **Architecture.** A decoder-only transformer, roughly Llama-style:
  RoPE positions, SwiGLU MLP, RMSNorm, grouped-query attention if
  your chosen Megatron/NeMo release supports it (otherwise vanilla
  MHA). Target parameter count: **~3B**. Suggested shape:
  `num_layers=26, hidden=2560, ffn=6912, heads=20, kv_heads=20 or 4,
  seq_len=2048, vocab≈32000`. Adjust to hit ~3B.
- **Precision.** BF16 parameters, FP32 optimizer master + moments,
  FP32 gradient reduction. Same policy in all three stacks.
- **Optimizer.** AdamW, `lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
  weight_decay=0.1`. Warmup 500 steps, cosine decay to 3e-5 over
  10000 steps.
- **Global batch.** 128 sequences × 2048 tokens = 262 144 tokens per
  step. Micro-batch and gradient-accumulation freely chosen per
  stack to match.

### The three legs

Implement the same model as three parallel executions:

1. **PyTorch FSDP2, torchtitan-style.**
   - Use `torch.distributed.fsdp.fully_shard` on each transformer
     block and on the root module.
   - Use a 1-D DeviceMesh for pure FSDP2. Add TP (2-D mesh with
     `torch.distributed.tensor.parallel`) only if a single GPU
     cannot hold a materialized block.
   - `MixedPrecisionPolicy(param_dtype=torch.bfloat16,
     reduce_dtype=torch.float32)`.
2. **DeepSpeed ZeRO-3.**
   - Wrap the same model class through `deepspeed.initialize`.
   - JSON config with `zero_optimization.stage=3`,
     `bf16.enabled=true`, `overlap_comm=true`,
     `reduce_bucket_size`, `stage3_prefetch_bucket_size` tuned
     to the model.
   - **Enforce the same effective global batch** — DeepSpeed's
     `train_batch_size` field is the guard.
3. **Megatron-LM 3D-parallel.**
   - Use `pretrain_gpt.py` (or a minimal fork) with `TP × PP × DP`
     multiplying to your world size.
   - On 8 GPUs, `TP=8, PP=1, DP=1` is the natural first
     configuration.
   - `--bf16 --distributed-optimizer`, sequence length 2048.
   - Pre-tokenize the dataset into Megatron's `.bin`/`.idx` format
     via `tools/preprocess_data.py`.

### The synthetic dataset

- Generate a fixed synthetic corpus of ~100 M tokens using a
  deterministic RNG. A trivial `n-gram`-shuffled Wikipedia dump or
  even `numpy.random.randint(0, vocab, size=(100_000_000,))`
  serialized to disk is enough — the loss must decrease with a
  reasonable learning rate; you are not building a real LM.
- Use the *same tokenizer* across all three stacks. A SentencePiece
  model trained on this corpus or a stock Llama-family tokenizer
  are both fine.
- Same seed for the data loader. Verify by hashing the first 1000
  token IDs each rank sees on step 0.

### The invariants (from chapter 1)

Before you run any bake-off, confirm all three configurations
agree on:

1. **Global batch size** (in tokens) is identical.
2. **Optimizer** is AdamW with identical hyperparameters.
3. **Mixed-precision policy** is BF16 params / FP32 reductions.
4. **Sequence length, tokenizer, and data ordering** are identical.

Any mismatch here makes the comparison meaningless.

### Measurements

For each stack, capture:

- **Peak HBM per GPU.** `torch.cuda.max_memory_allocated()` at end
  of 100 steps (all three stacks are PyTorch-based; the same call
  works).
- **Tokens/sec/GPU.** `(global_batch_tokens / step_time) /
  world_size`. Report the median of 100 steady-state steps
  (steps 21–120, discarding warmup).
- **Step-time breakdown.** From a `torch.profiler` trace: fraction
  of the step in compute vs. communication vs. optimizer. Name
  the top three time sinks in each stack.
- **First-100-step loss curve.** Should match across stacks to
  within numerical noise (BF16 rounding + fused-vs-non-fused
  optimizer differences). If any stack diverges, that stack has a
  bug — fix before benchmarking.

### The bake-off report

Deliver a 500–800-word markdown document with:

1. **Setup.** Model architecture (parameter count formula
   confirmed), hardware, and the invariants you enforced.
2. **The three configurations.** For each stack: the shape of the
   sharding, the config file (or CLI args), and the ~10-line
   training loop or launch command.
3. **Results table.** One row per stack, columns: peak HBM/GPU,
   tokens/s/GPU, step time, comm/compute ratio.
4. **Step-time breakdown.** For each stack, a short paragraph
   explaining the top time sinks and whether they were expected.
5. **The convergence check.** A plot (or a table of loss at steps
   1, 10, 50, 100) showing the three loss curves overlaid.
6. **Discussion.** For each stack, one paragraph on: what
   surprised you, what would you change if you had another day,
   which regime this stack would be your first choice for.
7. **References.** At least one citation per stack (paper or
   official docs).

### Acceptance criteria

- Three working configurations that reach step 100 without divergence
  on the same synthetic corpus, at the same global batch size.
- The three loss curves match within numerical tolerance for the
  first 100 steps.
- Peak HBM, throughput, and step-time breakdown are recorded in
  the results table with actual numbers — not "TBD".
- The step-time breakdown for each stack correctly identifies
  the dominant collective(s) predicted by chapters 2/3/5.
- The report cites at least one primary source per stack
  (FSDP2 docs / DeepSpeed + ZeRO paper / Megatron-LM paper).

## Starter guidance

- **Do the FSDP2 leg first.** It has the shortest launch surface
  and lets you validate the model architecture and the synthetic
  data pipeline before adding the extra machinery of DeepSpeed
  and Megatron.
- **Keep the model architecture in one shared Python module** and
  build minimal adapters per stack. Do not maintain three copies
  of the transformer block — one bug in one and the loss curves
  will not match.
- **Reduce sequence length and steps for the first pass.** Get all
  three stacks to *run* at 512 seq / 20 steps before tuning
  bench-sized configurations. Bake-off numbers only matter after
  the correctness check passes.
- **Use `torch.profiler` for FSDP2 and DeepSpeed; use Megatron's
  own timing logs plus `torch.profiler` for Megatron.** Megatron's
  `--log-timers-to-tensorboard` and per-iteration timing dumps
  are the primary source for its own step-time breakdown.
- **Watch for tokenizer differences.** Even one whitespace-handling
  difference in the SentencePiece config makes the dataset different
  across runs. Hash the first 1000 token IDs; if they differ, stop
  and fix.
- **Global batch size is the loudest bug source.** In FSDP2 the
  global batch is `micro_batch × world_size × grad_accum`; in
  DeepSpeed it is the JSON `train_batch_size`; in Megatron it is
  `--global-batch-size`. All three must resolve to the same
  number in tokens.
- **Time budget guidance.** Assume ~2 hours per stack to configure
  and validate, plus 30 minutes on the report. If you are running
  over, cut the model to 1.3B and rerun rather than skipping the
  comparison legs.

## Stretch goals

- **Add a 4th leg with FSDP2 + TP=2 on a 2-D DeviceMesh** and
  compare it to Megatron-LM's TP=8. The comparison at fixed
  `world_size / TP` isolates the FSDP2 vs. Megatron sharding
  implementation difference from the DP overhead.
- **Turn on `torch.compile` on the FSDP2 leg** and record the
  throughput delta. Note whether it introduced recompilations
  (which would show up as sudden step-time inflations in the
  profiler trace).
- **Run each stack on 16 GPUs** (2 nodes) if you have the
  hardware, and record the strong-scaling number vs. 8 GPUs.
  The comm-to-compute ratio predicted by mod-101 chapter 5 tells
  you what to expect; verify.
- **Swap the ~3B model for a ~7B model** on the same hardware.
  FSDP2 without TP will now OOM on 8×40 GB A100; document what
  changes when you introduce TP=2 or drop to DeepSpeed ZeRO-3
  with CPU offload.
