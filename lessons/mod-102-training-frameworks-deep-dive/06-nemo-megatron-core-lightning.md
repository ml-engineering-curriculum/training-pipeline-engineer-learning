# NeMo: Megatron-Core Wrapped in PyTorch Lightning

NVIDIA NeMo is what you get when you take Megatron-Core (chapter 5)
and wrap it in PyTorch Lightning, add Hydra/YAML configuration, and
package end-to-end recipes for LLM pretraining and post-training
(SFT, DPO, RLHF). Roughly: Megatron-LM is the reference training
*implementation*; NeMo is the reference training *product*.

The trade-off NeMo asks you to make is straightforward: it takes
more layers away between you and NCCL, but it gives you a
`Trainer`, a dataloading pipeline, a checkpoint system, and an
evaluation harness that you would otherwise have to build.
Exercise 4 asks you to port a Megatron-LM job to NeMo and reason
about what those extra layers do. This chapter builds the map for
that port.

> **Version note.** NeMo has evolved substantially. NeMo 1.x used
> `NeMoMegatronBaseModel`, YAML configs, and a `MegatronTrainer`.
> NeMo 2.0 (introduced in 2024) reorganized the API around
> `nemo.collections.llm` and PyTorch Lightning 2.x-native
> `Trainer` + `Strategy` objects. Both APIs still ship. The
> abstractions described here are conceptual — where APIs differ,
> the current release's docs are authoritative. `<!-- needs-research:
> confirm exact class names and module paths for NeMo 2.0 vs 1.x
> APIs against the currently pinned NeMo release before writing
> code -->`

## The abstraction ladder

NeMo layers four kinds of abstraction on top of Megatron-Core, in
order from user-facing to internal:

1. **Recipes / examples.** `nemo.collections.llm.recipes` (2.0)
   packages full training runs (Llama-family pretraining, SFT,
   PEFT variants) as ready-to-launch scripts. In 1.x the
   equivalent is `examples/nlp/language_modeling/megatron_gpt_pretraining.py`
   and friends.
2. **Model classes.** `MegatronGPTModel`,
   `MegatronBertModel`, `MegatronT5Model` (1.x) or
   `nemo.collections.llm.GPTModel` (2.0). These wrap a
   Megatron-Core transformer with PyTorch Lightning's
   `LightningModule` — meaning they define `training_step`,
   `validation_step`, `configure_optimizers`, and the surrounding
   plumbing rather than an iteration-based loop.
3. **Strategy classes.** NeMo provides Lightning
   `Strategy` implementations that know how to launch and manage a
   Megatron-Core parallel job. `MegatronStrategy` (2.0) or
   `NLPDDPStrategy` / `NLPFSDPStrategy` / `MegatronTrainerStrategy`
   (1.x) set up the process groups, apply the parallelism plan,
   and orchestrate pipeline-parallel micro-batch scheduling.
4. **Megatron-Core underneath.** All the actual parallel primitives
   — `ColumnParallelLinear`, `parallel_state`, the pipeline
   scheduler, the Distributed Optimizer — come from
   `megatron.core.*`.

The one-line summary: NeMo is the *composition* of a Lightning
`Trainer` + a Megatron `Strategy` + a Megatron `LightningModule` +
a Hydra `Config`. If you understand each of those four things, you
understand what NeMo does.

## Configuration: Hydra / YAML

NeMo uses Hydra (Facebook's config framework) for its configs.
That means:

- Configs are YAML files with structured schemas.
- You override any leaf field on the CLI:
  `python train.py model.hidden_size=8192
  trainer.devices=8 model.tensor_model_parallel_size=4`.
- Configs compose via YAML defaults lists (e.g. a `llama3` model
  config includes a `common/precision/bf16.yaml`).

A minimal NeMo GPT pretraining config (1.x-flavored) looks
approximately like:

```yaml
trainer:
  devices: 8
  num_nodes: 1
  accelerator: gpu
  precision: bf16
  max_steps: 300000
  log_every_n_steps: 10
  val_check_interval: 1000

exp_manager:
  exp_dir: ./nemo_experiments
  name: gpt-3b
  create_wandb_logger: true

model:
  micro_batch_size: 4
  global_batch_size: 128
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 1
  virtual_pipeline_model_parallel_size: null

  encoder_seq_length: 2048
  max_position_embeddings: 2048
  num_layers: 26
  hidden_size: 2560
  num_attention_heads: 32
  ffn_hidden_size: 10240
  init_method_std: 0.02

  tokenizer:
    library: sentencepiece
    model: /path/to/tokenizer.model

  optim:
    name: distributed_fused_adam
    lr: 3.0e-4
    weight_decay: 0.1
    betas: [0.9, 0.95]
    sched:
      name: CosineAnnealing
      warmup_steps: 2000

  data:
    data_prefix: [1.0, /path/to/pretokenized_data]
    seq_length: 2048
    num_workers: 2
```

Notice what is *not* in that YAML: no explicit
`initialize_model_parallel(tp, pp)`, no `pretrain_gpt.py`-style CLI
flags for parallelism, no explicit `Adam` construction. The
Strategy and Model classes translate the YAML into Megatron-Core
calls under the hood.

## Mapping from Megatron-LM configuration to NeMo

For the port in exercise 4, the following table is the
mapping most training-team users need. `<!-- needs-research:
double-check the NeMo 2.0 field names below against current docs;
1.x uses model.* fields, 2.0 restructures under trainer/model
namespaces. -->`

| Megatron-LM (CLI flag)                             | NeMo (YAML key)                                       |
|----------------------------------------------------|-------------------------------------------------------|
| `--tensor-model-parallel-size`                     | `model.tensor_model_parallel_size`                    |
| `--pipeline-model-parallel-size`                   | `model.pipeline_model_parallel_size`                  |
| `--num-layers-per-virtual-pipeline-stage`          | `model.virtual_pipeline_model_parallel_size` (derived) |
| `--sequence-parallel`                              | `model.sequence_parallel: true`                       |
| `--context-parallel-size`                          | `model.context_parallel_size`                         |
| `--expert-model-parallel-size`                     | `model.expert_model_parallel_size`                    |
| `--micro-batch-size`                               | `model.micro_batch_size`                              |
| `--global-batch-size`                              | `model.global_batch_size`                             |
| `--sequence-length` / `--seq-length`               | `model.encoder_seq_length`                            |
| `--num-layers` / `--hidden-size` / `--num-attention-heads` | `model.num_layers` / `model.hidden_size` / `model.num_attention_heads` |
| `--distributed-optimizer`                          | Enabled by default via the strategy in modern releases |
| `--optimizer AdamW --lr X`                         | `model.optim.name: distributed_fused_adam / lr: X`    |
| `--bf16`                                           | `trainer.precision: bf16` (or `bf16-mixed`)           |
| `--tokenizer-type … --vocab-file …`                | `model.tokenizer.library / model.tokenizer.model`     |
| `--train-iters`                                    | `trainer.max_steps`                                   |
| `--save … / --load …`                              | `exp_manager` + Lightning callbacks (`ModelCheckpoint`) |
| `--data-path`                                      | `model.data.data_prefix`                              |
| `--data-impl mmap`                                 | (default; NeMo uses Megatron's mmap dataset)          |

## What the Lightning `Trainer` gives you

The features you inherit by adopting Lightning:

- **`Trainer.fit(model, datamodule)`** rather than an iteration
  loop. Handy for post-training pipelines (SFT, DPO) that already
  compose as Lightning modules.
- **Callbacks** — `ModelCheckpoint`, `LearningRateMonitor`,
  `EarlyStopping`, and NeMo-specific ones. In particular NeMo's
  `AsyncFinalizerCallback` and `StraggerCallback` (2.0) address
  training-run stability at scale.
- **Loggers** — TensorBoard, Weights & Biases, MLflow, all built
  into `trainer.logger`.
- **Distributed launch** — `Trainer(devices=8, num_nodes=4,
  strategy=MegatronStrategy(...))` handles process-group init and
  torchrun-style launching for you.
- **Precision plugins** — `precision: bf16-mixed` /
  `precision: transformer-engine` (integrates TE for FP8 on H100).

## What the Lightning `Trainer` costs you

Real friction points to plan for:

- **The stack trace is longer.** A NaN loss surfaces through
  `Trainer -> Strategy -> MegatronGPTModel.training_step ->
  Megatron-Core forward -> parallel layer`. Reading that trace is a
  skill.
- **Callbacks can hide behavior.** A callback silently changing
  learning rate schedule or triggering an unexpected checkpoint
  save at step boundaries is easy to miss in the config.
- **YAML drift.** With enough overrides on the CLI, the effective
  config diverges from the file in the repo. NeMo prints the
  resolved config at startup; read it.
- **Version lock-in.** NeMo pins Lightning, Megatron-Core, and TE
  versions tightly. Upgrading NeMo often forces an upgrade of the
  whole stack.
- **Custom architectures require plumbing.** A novel attention
  variant needs to be written as a Megatron-Core layer *and*
  registered with the NeMo Model class *and* exposed via the
  config schema. That is real work — Megatron-LM lets you edit a
  Python file; NeMo asks you to edit several.

## Post-training features NeMo bundles

The reason many teams pick NeMo over Megatron-LM for a
post-training run is that it packages:

- **Supervised fine-tuning (SFT).** `SFTModel` with instruction
  templates, sequence packing, and loss masking.
- **PEFT.** LoRA / adapters, with sharding-aware injection into a
  Megatron-Core model.
- **Preference optimization.** DPO and related algorithms
  (`nemo-aligner` / `nemo.collections.nlp.models.rl` in older
  releases). `<!-- needs-research: RLHF library naming has changed
  across NeMo releases; verify current module path before writing
  code. -->`
- **Reward-model training.** A separate `Model` for training
  reward models with Megatron-Core.

If you build these from scratch on Megatron-LM you will end up
recreating most of what NeMo already ships. That is a real
argument for NeMo in a productized post-training pipeline; it is
less clearly an argument for NeMo in a novel-architecture
research pretraining run.

## The porting workflow

The mechanics of taking a Megatron-LM job and porting it to NeMo:

1. **Inventory the Megatron config.** Every CLI flag →
   corresponding YAML key using the table above.
2. **Pick a NeMo model class.** For a decoder-only GPT-style
   model, `MegatronGPTModel` (1.x) or `nemo.collections.llm.GPTModel`
   (2.0). For LLaMA-family models specifically, a subclass or a
   recipe in `nemo.collections.llm.recipes.llama*`.
3. **Convert the Megatron checkpoint** (optional). NeMo can
   consume Megatron's `.pt`/sharded checkpoints via the
   conversion utilities in `scripts/checkpoint_converters/` (1.x)
   or `nemo.collections.llm.import_ckpt` (2.0).
4. **Wire the data pipeline.** NeMo's dataloader still uses
   Megatron's `MMapIndexedDataset`, so pre-tokenized `.bin`/`.idx`
   shards from Megatron `preprocess_data.py` work directly. You
   are just changing where the shards are pointed to.
5. **Launch with a Lightning strategy.**
   `Trainer(strategy=MegatronStrategy(...), devices=..., num_nodes=...)`
   or the equivalent recipe launcher. Verify at startup that the
   resolved parallelism sizes and world size match your
   Megatron-LM run.
6. **Verify the loss curve.** If you configured every knob correctly,
   the first N steps of loss from a fresh initialization should
   match the Megatron-LM baseline to within numerical tolerance
   (BF16 rounding + optimizer fused-vs-non-fused differences
   dominate the residual). If they diverge, the port has a bug.

Exercise 4 walks through exactly this workflow.

## When NeMo helps and when it hurts

Trade-off summary you can use in the decision doc (chapter 8):

**NeMo helps when:**

- You want a productized pipeline: pretraining → SFT → PEFT →
  DPO with the same codebase and checkpoints flowing through.
- You want the built-in Lightning callbacks / loggers /
  checkpoint manager.
- You want NVIDIA-tuned defaults for H100-class hardware (fused
  optimizers, TE FP8, activation checkpointing recipes).
- The models you want to train are close to the recipes NeMo
  already ships (Llama-family, Mistral-family, GPT-family).

**NeMo hurts when:**

- You are prototyping a novel architecture that does not fit the
  Megatron-Core layer palette without invasive changes.
- You want to debug at the collective level and dislike the extra
  Lightning frames in the stack.
- Your team has no prior Lightning experience and is not eager to
  learn a second training framework.
- You need to iterate quickly on the training loop itself
  (custom curriculum, custom sampler, dynamic mixed precision) —
  overriding the Lightning `training_step` from a YAML is more
  friction than the equivalent Python edit in Megatron-LM.

## Summary

- NeMo is Megatron-Core wrapped in PyTorch Lightning with Hydra
  configuration and end-to-end recipes for pretraining and
  post-training.
- The abstractions are recipes → Model class → Strategy →
  Megatron-Core. Reading a NeMo run means being fluent in all
  four.
- The Megatron-LM → NeMo port is largely a CLI-flag → YAML-key
  translation, with model-class selection and Lightning strategy
  configuration on top.
- NeMo is the right choice for productized post-training and for
  NVIDIA-tuned defaults; Megatron-LM is the right choice when you
  want less indirection and more control over the training loop.
