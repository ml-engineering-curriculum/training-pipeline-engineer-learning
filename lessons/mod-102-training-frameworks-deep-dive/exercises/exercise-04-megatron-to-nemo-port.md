# exercise-04: Megatron-LM → NeMo Port

**Estimated effort:** 4 hours

## Objective

Take the Megatron-LM 3D-parallel configuration from exercise 1 and
port it to NeMo (Megatron-Core + PyTorch Lightning), verifying that
the ported job runs the same model with matching loss curves for
the first 100 steps. Along the way, produce a mapping table of
Megatron CLI flags → NeMo YAML keys and a short reflection on what
NeMo's abstractions added, hid, or made harder.

## Prerequisites

- Chapters 5 and 6 of this module.
- Exercise 1 completed — you have a working Megatron-LM
  `pretrain_gpt.py` configuration for the 3B model.
- NeMo installed (see the NeMo docs for the currently-pinned
  release). You may use the NGC PyTorch/NeMo container or an
  install from source; either is fine as long as you can `import
  nemo` and reach the LLM examples.
- The same synthetic dataset and tokenizer from exercise 1. NeMo
  and Megatron-LM share the `.bin`/`.idx` format, so no re-tokenizing
  is required.
- At least 8 GPUs. Same shape as the exercise-1 Megatron leg
  (e.g. TP=8, PP=1, DP=1 on a single 8-GPU node).

## Problem statement

Your team has a stable Megatron-LM pretraining pipeline for the
3B pilot but is planning a post-training pipeline (SFT + DPO) that
NeMo already ships as recipes. Before committing to two
frameworks, you want to see whether NeMo can drive the *same*
pretraining run — so you port the Megatron-LM config over, prove
the loss curves match, and then decide whether to consolidate.

You will *not* re-implement any parallel layers. NeMo uses
Megatron-Core underneath, and Megatron-Core is exactly what the
existing Megatron-LM job uses. The port is a config-shape change,
not a training-implementation change.

## Requirements

### Part A — The configuration mapping

Produce a `megatron-to-nemo-mapping.md` file that lists, for
every non-trivial CLI flag in your exercise-1 Megatron
configuration, the corresponding NeMo YAML key. Use the table in
chapter 6 as your starting point and extend it with anything your
specific configuration used (e.g. `--num-query-groups`,
`--rotary-percent`, `--swiglu`, `--use-distributed-optimizer`,
`--recompute-granularity`, etc.).

For each mapped flag, note:

- The exact CLI argument value from your Megatron config.
- The NeMo YAML key path (e.g. `model.tensor_model_parallel_size`).
- Any transformation required (e.g. `--global-batch-size 128`
  maps to `model.global_batch_size: 128`; but Megatron's
  `--seq-length` may map to `model.encoder_seq_length` depending
  on NeMo release).

Cite the NeMo docs page (or the recipe YAML) that each mapping is
sourced from. If a specific flag has no NeMo equivalent, say so
and describe the workaround.

### Part B — The port

Author a NeMo config YAML that reproduces your exercise-1
Megatron configuration:

```yaml
trainer:
  devices: 8
  num_nodes: 1
  accelerator: gpu
  precision: bf16
  max_steps: 100          # matching the exercise-1 loss-comparison window
  log_every_n_steps: 1
  val_check_interval: 100
  limit_val_batches: 0

exp_manager:
  exp_dir: ./nemo_experiments
  name: 3b-port
  create_wandb_logger: false

model:
  micro_batch_size: <your value>
  global_batch_size: <your value>
  tensor_model_parallel_size: <your value>
  pipeline_model_parallel_size: <your value>
  sequence_parallel: <your value if set>

  encoder_seq_length: 2048
  max_position_embeddings: 2048
  num_layers: 26
  hidden_size: 2560
  num_attention_heads: 20
  ffn_hidden_size: 6912
  # activation, rope base, norm type, etc. to match your Megatron config

  tokenizer:
    library: sentencepiece
    model: <path to the exercise-1 tokenizer>

  optim:
    name: distributed_fused_adam
    lr: 3.0e-4
    betas: [0.9, 0.95]
    weight_decay: 0.1
    sched:
      name: CosineAnnealing
      warmup_steps: 500

  data:
    data_prefix: [1.0, <path to the exercise-1 .bin/.idx>]
    seq_length: 2048
    num_workers: 2
```

Launch NeMo with the same GPU count and TP/PP configuration as
the Megatron run.

`<!-- needs-research: verify the exact NeMo YAML key names for the
currently-pinned NeMo release before shipping the solution;
some fields differ between NeMo 1.x and 2.0. -->`

### Part C — The parity check

- Fix the random seed identically for both runs
  (`trainer.random_seed` in NeMo, `--seed` in Megatron; also
  `numpy` and `torch` seeds).
- Run **100 steps** of both configurations on the same dataset.
- Log training loss per step from both runs.
- Overlay the two loss curves in a plot. They should track to
  within numerical tolerance (BF16 rounding + fused-vs-non-fused
  optimizer noise). Persistent divergence indicates a mapping
  bug.
- If divergence appears, walk the config table and identify the
  discrepancy. Common culprits: `add_bias_linear`,
  `layernorm_epsilon`, `gated_linear_unit` vs. `swiglu`, tokenizer
  encoding differences.

### Part D — The reflection

Deliver a 400–600-word `nemo-port-reflection.md` covering:

1. **The mapping.** Which flags were 1:1 obvious, which required
   translation, and which had no NeMo equivalent.
2. **The startup diff.** Compare launching the two runs. Megatron
   is `torchrun pretrain_gpt.py <100+ flags>`; NeMo is
   `python megatron_gpt_pretraining.py --config-path=... trainer.devices=...`.
   Which was easier to configure? Easier to debug when a value was
   wrong?
3. **The observed abstractions.** What did the PyTorch Lightning
   `Trainer` do that Megatron-LM's own loop didn't? Callbacks?
   Checkpointing? Logging? Give one concrete example.
4. **What NeMo hid.** Point to a specific Megatron-LM behavior you
   understood well but that NeMo obscures behind a callback,
   YAML default, or Strategy. Not necessarily bad — but call it
   out.
5. **A recommendation.** Given your team and the pretraining +
   post-training plan, which of the two would you consolidate on?
   Reference chapter 8's rubric.

## Acceptance criteria

- A complete mapping table (part A) covering every non-trivial
  Megatron flag your exercise-1 configuration used.
- A working NeMo YAML (part B) that launches on the same 8 GPUs.
- A loss-curve comparison over 100 steps with curves matching to
  within numerical tolerance. If they diverge, part D must
  explain what you found and (ideally) how you fixed it.
- Reflection (part D) cites at least one primary NeMo doc page
  (e.g. NeMo LLM API docs, a recipe README, or the NeMo YAML
  reference).

## Starter guidance

- **Prefer to start from a NeMo recipe close to your model.** If
  your 3B model is Llama-family, find the Llama pretraining recipe
  under `nemo.collections.llm.recipes.llama*` (2.0) or the
  Megatron-GPT example under `examples/nlp/language_modeling/`
  (1.x) and edit its YAML to match your dimensions.
- **Do not fight `distributed_fused_adam` vs. `adamw` vs.
  `mcore_distributed_optimizer`.** The names have moved across
  NeMo releases. Pick the one your NeMo docs recommend for the
  current release; if the optimizer type is not the same as
  Megatron's, expect a bit of numerical drift in the loss curve.
- **Verify the resolved config.** NeMo prints the resolved Hydra
  config at startup. Read it. It is where you catch a wrong
  `--override` from the CLI silently mutating a field you
  intended to leave alone.
- **Timing budget.** Assume ~90 min on the mapping table (part A),
  ~60 min on the working YAML (part B), ~45 min running the
  parity check (part C), and ~30 min on the write-up (part D).
- **Do not port the whole 300k-step pretraining schedule.** 100
  steps is the parity window; do not spend hours on data schedule
  or checkpointing configuration that is not on the critical
  path for the exercise.

## Stretch goals

- **Port to NeMo 2.0** (if your starting point was NeMo 1.x, or
  vice-versa) and record the API surface differences. That
  characterization is directly useful in the decision doc for
  chapter 8.
- **Enable the Transformer Engine FP8 path** in NeMo
  (`transformer_engine: true` in the model config, along with an
  FP8 precision plugin at the trainer level) and re-run the
  parity check. Note whether the loss curve stays within
  tolerance.
- **Point NeMo at your Megatron-LM checkpoint** using the NeMo
  conversion utilities (`nemo.collections.llm.import_ckpt` in
  2.0, or `scripts/checkpoint_converters/` in 1.x). Resume from
  step 100 in NeMo using the converted Megatron-LM state and
  verify the loss stays flat at resumption — a real
  integration-readiness signal.
- **Layer a callback.** Add a `Callback` that logs the norm of a
  chosen parameter every 10 steps. This exercise gives you a
  concrete taste of what NeMo's abstraction ladder buys and where
  it lives.
