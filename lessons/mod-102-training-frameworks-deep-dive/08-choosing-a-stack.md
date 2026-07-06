# Choosing a Stack: The Decision Doc

By the time you have finished exercises 1–5, you have felt each of
the five stacks (FSDP2, DeepSpeed, Megatron-LM, NeMo, JAX/Flax) on
the same hardware, on comparable models. This chapter is the
capstone: how to turn those hands-on impressions into a written
decision doc that a training platform team can defend to
engineering leadership.

The output for the module is a decision doc of the form:

> "Team X is starting training regime Y on cluster Z. We recommend
> stack S because [reasons]. We considered [others] and rejected
> them because [reasons]. Our exit criteria for revisiting this
> choice are [criteria]."

This chapter gives you the rubric behind that sentence.

## Why a decision doc at all

Stack choice is a decision with real switching cost. Once a team's
training pipeline is in DeepSpeed JSON configs, migrating to FSDP2
is not a one-week task — it is a project that touches data
pipelines, checkpointing, logging, evaluation, and the CI that runs
smoke jobs. A written decision doc has three purposes:

1. **Forces the analysis to happen once.** You have to name the
   regime, the constraints, the alternatives, and the evidence.
2. **Creates an artifact for future re-evaluation.** Six months
   later, when someone asks "should we still be on DeepSpeed?",
   the answer is "here is the doc; here are the exit criteria;
   here is what has changed." You compare against the original
   analysis rather than re-litigating from scratch.
3. **Makes the choice legible to non-training-team stakeholders.**
   The infrastructure team, the finance team, and the platform-
   owners can read the doc and know why the choice was made.

## The five inputs to any stack choice

The rubric has five inputs. Every serious decision doc must have a
paragraph on each.

### 1. Team

- **Prior expertise.** Does the team already have engineers who
  have shipped in PyTorch? DeepSpeed? Megatron-LM? JAX? Prior art
  is worth more than any abstract capability. A team of 5 PyTorch
  natives will be net-productive on FSDP2 in a week. The same team
  will lose a month adopting JAX before matching their PyTorch
  throughput.
- **Team size.** A one-engineer research team should not be
  running Megatron-LM directly — the operational surface is too
  large. A 10-engineer platform team can afford it.
- **Debugging bandwidth.** Whoever is on-call for the training run
  needs to be able to read the stack trace when it breaks. That
  argues against layers of abstraction the team has not internalized.

### 2. Training regime

- **Pretraining vs. post-training.** Pretraining runs weeks;
  every 1% throughput matters; complexity is worth it. Fine-tuning
  runs hours; complexity is a cost that dominates the savings.
- **Model size and shape.** Under ~7B on 8 GPUs, FSDP2 is
  overkill. 7B–70B, FSDP2 or DeepSpeed. 70B+, expect 3D-parallel
  (Megatron / NeMo).
- **Sequence length.** 8k context → any stack works.
  32k+ context → you need context-parallel (Megatron / NeMo), or
  a JAX equivalent. FSDP2 currently does not ship a first-class CP.
- **Sparsity / MoE.** MoE training benefits from
  expert-parallel. DeepSpeed-MoE and Megatron-Core-MoE ship this
  natively; FSDP2 + MoE is possible but hand-assembled.
- **Fine-tuning shape.** LoRA / adapters mostly bypass the
  sharding-difficulty spectrum; a plain DDP or FSDP2 setup with
  the base model frozen is fine.
- **Post-training pipeline.** SFT → DPO → RLHF pipelines are
  where NeMo's bundled recipes pay off; hand-rolling them on
  Megatron-LM is real work.

### 3. Hardware / cluster shape

- **GPU count and topology.** How many GPUs per node, how many
  nodes, what fabric between nodes?
- **HBM per GPU.** 80 GB (H100) vs. 24 GB (RTX 4090 lab box) vs.
  40 GB (A100 40 GB). The tighter the HBM budget, the more you
  need offload; the more offload matters, the more DeepSpeed wins
  vs. FSDP2 (chapter 4).
- **NVMe presence.** ZeRO-Infinity only pays off if you have
  provisioned NVMe. If not, that rung of the ladder is unavailable.
- **Fabric.** IB at ~200–400 Gb/s vs. RoCEv2 vs. Ethernet-only. TP
  across nodes is bad on any of them; PP is more tolerant.
- **TPUs.** If the target hardware is TPU, JAX is the primary
  supported stack. Sidesteps a chunk of this decision.

### 4. Ecosystem constraints

- **Checkpoint interoperability.** Which model formats do you need
  to consume (HuggingFace `transformers`, Megatron `.pt`, NeMo
  `.nemo`, orbax for JAX)? Every stack ships conversion utilities
  in some direction, but "you can always convert" is a real
  operational cost.
- **Downstream serving.** If the trained model has to serve on
  vLLM, TensorRT-LLM, or a proprietary inference stack, you need
  a checkpoint format compatible with those. Some conversion paths
  are one-line; others are custom scripts.
- **Model-zoo access.** If your regime is "fine-tune the latest
  Llama-N release the day it drops", you want the stack with the
  fastest support cycle — historically HuggingFace + DeepSpeed
  and NeMo have been quick; Megatron-LM catches up next; FSDP2 +
  torchtitan-style code follows.
- **Frameworks the rest of the org uses.** Consistency has value.
  A platform team that already runs 20 training jobs in DeepSpeed
  loses something when a new team adds a JAX pipeline for one job.

### 5. Observability and reliability

- **Logging integration.** W&B, TensorBoard, MLflow: all stacks
  can be wired up, but Lightning (NeMo) makes it declarative.
- **Failure mode legibility.** How readable are the errors when a
  run goes wrong? PyTorch native > DeepSpeed > Megatron-LM > NeMo
  > JAX (roughly, and for a PyTorch-native team).
- **Checkpointing at scale.** Distributed checkpointing (mod-106)
  is a real thing — PyTorch DCP, DeepSpeed sharded checkpoints,
  Megatron sharded checkpoints, NeMo's, orbax for JAX. Choose a
  stack whose checkpoint story you can operate.
- **Elastic training / preemption tolerance.** Owned by mod-106;
  worth flagging that DeepSpeed and PyTorch elastic each have
  their own answers.

## The decision matrix

Not a formula, but the rules of thumb the module supports. Use as
a starting point and override with your team's actual constraints.

| Regime                                                          | Recommended default                    | Notes                                                                 |
|-----------------------------------------------------------------|----------------------------------------|-----------------------------------------------------------------------|
| <1B fine-tuning, small cluster                                  | DDP or single-GPU                      | Sharded frameworks are pure overhead here                              |
| 1B–7B pretraining, 8–64 GPU                                     | FSDP2                                  | PyTorch-native, low-friction                                          |
| 7B–70B pretraining, 32–256 GPU, no NVMe pressure                | FSDP2 (with TP if a layer overflows)   | `torchtitan`-shaped code                                              |
| 7B–70B pretraining, tight HBM, want offload                     | DeepSpeed ZeRO-3 + ZeRO-Offload        | CPU Adam kernel wins                                                  |
| Training a model bigger than aggregated HBM but <total DRAM     | DeepSpeed ZeRO-Infinity (NVMe path OFF, DRAM ON) | Chapter 4's decision procedure                                |
| Training a model bigger than aggregated DRAM                    | DeepSpeed ZeRO-Infinity (NVMe path ON) | Expect meaningful step-time slowdown                                  |
| 70B+ pretraining requiring 3D parallel                          | Megatron-LM (research team) or NeMo (platform team) | Depends on how bundled you want the pipeline                        |
| 70B+ with productized post-training (SFT / PEFT / DPO)          | NeMo                                    | Recipes pay for the abstractions                                     |
| Any pretraining on TPU                                          | JAX (MaxText-flavored)                  | Only serious option                                                  |
| Long-context (32k+) pretraining                                 | Megatron / NeMo (CP)                    | FSDP2 lacks first-class context-parallel                             |
| MoE at any serious scale                                        | DeepSpeed-MoE or NeMo (Megatron-Core-MoE) | Native EP + tuned kernels                                          |
| Team has zero PyTorch and heavy JAX experience                  | JAX                                     | Team expertise dominates                                             |

The critical caveat: the recommended default is not a claim about
"the best stack for regime X, ignoring team." It is a starting
point. The team + expertise input can flip almost any row.

## The decision doc template

A workable template — the exercise-1 deliverable and the
module capstone should follow this shape:

```markdown
# Decision doc: training stack for [team] / [regime]

## Context
- Team: <who they are, size, prior training experience>
- Regime: <model size, dataset, training scheme, timelines>
- Cluster: <GPUs, nodes, fabric, HBM, DRAM, NVMe>
- Ecosystem: <what checkpoints they need to consume/produce, downstream serving>

## Recommendation
We recommend **[stack]** because:
1. [Regime-driven reason]
2. [Cluster-driven reason]
3. [Team-driven reason]

## Alternatives considered
- **[Alternative 1]**: rejected because …
- **[Alternative 2]**: rejected because …
- **[Alternative 3]**: on the table as a fallback if …

## Evidence
Bake-off numbers on our hardware (from exercise 1 or from a spike):
| Stack | Peak HBM / GPU | Throughput (tokens/s/GPU) | Step time | Notes |
|-------|----------------|---------------------------|-----------|-------|
| ...   | ...            | ...                       | ...       | ...   |

## Risks and mitigations
- [Risk]: [mitigation]
- ...

## Exit criteria (when do we revisit this choice?)
- If [condition] we should reconsider [alternative]
- If [condition] we should reconsider [alternative]
- Otherwise, revisit in [N] months regardless

## Implementation plan
Phase 1: …
Phase 2: …
...

## References
- [links to primary docs, papers, our exercise 1 write-up, cost model]
```

## Anti-patterns to avoid

Common failure modes in decision docs:

- **"We picked stack S because it's what we know."** Sometimes
  correct, but only when you have named team expertise as an
  explicit input and weighed it against alternatives. "It's what
  we know" without that framing hides the trade-off.
- **"We picked stack S because the paper used it."** Papers use
  the stack the authors happened to have; that is not an
  endorsement of your regime.
- **Benchmarking against your target scale using a toy model.**
  A 125M model on 2 GPUs will lie to you about a 70B model on 256
  GPUs. Benchmark at your target scale, or (if too expensive) at
  a scale where the parallelism regime you actually care about is
  already active.
- **Ignoring switching costs.** "We can always switch later" is
  usually only true for models. Training pipelines, CI, data
  loaders, checkpoint formats, monitoring dashboards — all have
  real switching cost, and the doc should name them.
- **No exit criteria.** A decision without exit criteria is a
  decision that can never be un-made. State when to revisit.

## Two worked examples (in outline)

Both examples are illustrative rubrics — you fill in the numbers
from your own bake-off in exercise 1.

### Example 1: 3-engineer research team, 70B pretraining on 128 H100s, IB fabric, no NVMe

- Regime: 70B pretraining, long timeline (2 months), custom
  architecture variant.
- Team: PyTorch-native, one engineer with Megatron experience.
- Cluster: 16 nodes × 8 H100 (80 GB), IB HDR, no NVMe.
- Ecosystem: existing HuggingFace `transformers` code, want
  compatible checkpoints.

Likely recommendation: **Megatron-LM** (not NeMo, because the
custom architecture is easier to hand-write in Megatron-Core
layers than to plug into a Lightning `LightningModule` and a
NeMo recipe). Fallback: FSDP2 + TP if the team's Megatron engineer
is a bottleneck.

Exit criteria: revisit if the architecture stabilizes and moves to
production post-training — NeMo becomes attractive; or if a new
CP-heavy long-context regime is added — Megatron's native CP is a
plus.

### Example 2: 10-engineer platform team, Llama-family SFT + DPO pipeline, 32 H100s

- Regime: 8B/70B SFT + DPO on Llama-family bases, monthly runs.
- Team: mixed PyTorch + Lightning experience, no JAX.
- Cluster: 4 nodes × 8 H100, IB, healthy NVMe.
- Ecosystem: downstream serving on vLLM; Llama-family
  checkpoints from HuggingFace as inputs.

Likely recommendation: **NeMo** — the SFT/DPO recipes, model-zoo
integration, and Lightning Trainer buy the team an operational
pipeline they would otherwise re-implement. Fallback: DeepSpeed
+ HuggingFace `transformers` for the 8B path if NeMo integration
proves fragile.

Exit criteria: revisit if NeMo's Llama-family recipes ever fall
behind by more than a release; or if the team decides to train
custom architectures where NeMo's abstractions become friction.

## Summary

- A stack choice is a decision with real switching cost. Write it
  down.
- The five inputs are team, regime, hardware, ecosystem, and
  observability/reliability. Every serious decision doc has a
  paragraph on each.
- The decision matrix in this chapter is a starting point, not a
  formula. Team expertise can override every recommendation.
- The doc template names the recommendation, the alternatives,
  the evidence, the risks, and — crucially — the exit criteria.
- The most common failure mode is not picking the wrong stack;
  it is picking any stack without documenting *why*, which
  guarantees the choice will be re-litigated at every future
  team change.
