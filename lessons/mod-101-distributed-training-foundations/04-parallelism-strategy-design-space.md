# The Parallelism Strategy Design Space

Data, tensor, pipeline, sequence, expert, and hybrid sharded parallel — six
axes, each with its own communication pattern, memory footprint, and
happy-path cluster shape. This chapter enumerates the design space, explains
what each axis does at the collective level, and gives you a decision
procedure for picking the right combination for a specific `(model, cluster)`
pair. Chapter 5 turns the qualitative picks here into quantitative cost
estimates.

## Motivation

At Llama-3-405B scale (Grattafiori et al., 2024, "The Llama 3 Herd of
Models"), the training run uses tensor parallel *inside* a node, pipeline
parallel across a group of nodes, context / sequence parallel to handle
long sequences, and full data parallel across the outer axis — a 4-D
strategy. You cannot invent that from scratch. You pick each axis by
answering:

- What memory problem is this axis solving?
- What collective does it introduce, over which links?
- What does it cost me in comm-to-compute ratio?

## The six axes

### 1. Data parallel (DP)

Already covered in chapter 2 (DDP) and chapter 3 (FSDP/ZeRO-3 as
memory-sharded DP). Splits the *batch* across ranks. The
sharded-optimizer variants (FSDP2, ZeRO-1/2/3) also split state.

- **Collective**: all-reduce (DDP) or all-gather + reduce-scatter (FSDP).
- **Solves**: throughput; also memory when sharded.
- **Cost**: comm scales with parameter size; independent of batch or
  sequence length.

### 2. Tensor parallel (TP)

Split individual weight matrices *across ranks* so the matmul itself is
distributed. The canonical scheme is from Megatron-LM (Shoeybi et al.,
2019, "Megatron-LM: Training Multi-Billion Parameter Language Models Using
Model Parallelism"): for a transformer block, split the two MLP linears
column-then-row and the QKV projection column-wise, so the block requires
**two all-reduces per layer** on the forward pass and two more on the
backward.

- **Collective**: all-reduce (Megatron style) inside each transformer
  block, on the activation-sized tensor. **This is the dominant cost.**
- **Solves**: per-layer memory (parameters and their activations get
  smaller by TP factor).
- **Cost**: comm is proportional to `activation_size * TP`, and happens
  *inside* every forward and backward — much more frequent than DP.
- **Cluster fit**: keep TP inside one node (typically TP ≤ 8 on an
  8-GPU host) so its all-reduces stay on NVLink. Cross-node TP is
  usually a mistake.

Megatron's follow-up (Narayanan et al., 2021, "Efficient Large-Scale
Language Model Training on GPU Clusters Using Megatron-LM") is the
canonical reference for how to compose TP with pipeline and data parallel;
you should read it before designing anything past a single node.

### 3. Pipeline parallel (PP)

Split the *layers* across ranks so each rank owns a range of layers
("stages"). Micro-batches flow through the pipeline; while stage `k` is
processing micro-batch `m`, stage `k-1` is already processing micro-batch
`m+1`. Point-to-point sends/recvs move activations between stages on
forward and gradients between stages on backward.

- **Collective**: point-to-point `send`/`recv` of activations across
  stage boundaries — not a group collective.
- **Solves**: parameter memory across depth (each rank only owns a slice
  of layers) and lets you scale past what TP × DP can fit inside one node.
- **Cost**: **pipeline bubble**. If you have `S` stages and `M`
  micro-batches, the fraction of ideal throughput lost to the fill/drain
  bubble is roughly `(S - 1) / (M + S - 1)` for the classical
  fill-then-drain schedule (GPipe; Huang et al., 2019). Interleaved-1F1B
  and zero-bubble schedules reduce this — Megatron's paper and the
  interleaved-1F1B derivation are the definitive references.
- **Cluster fit**: PP is what stitches TP islands into a wider training
  job. Each pipeline stage typically *is* a TP group.

### 4. Sequence / context parallel (SP)

Split the **sequence dimension** of activations across ranks so each rank
sees only `seq_len / SP` tokens for the attention and MLP computation.
Introduced for Megatron in Korthikanti et al., 2022 ("Reducing Activation
Recomputation in Large Transformer Models"), and generalized into
"context parallel" in more recent framework docs.

- **Collective**: all-to-all (for attention's Q/K/V exchange under
  context parallel) plus reduce-scatter/all-gather at the boundaries of
  layer-norm / dropout that would otherwise need the full sequence.
- **Solves**: **activation memory** at long sequence length. Not a
  parameter-memory optimization — this is the axis you reach for when the
  KV cache and per-token activations dominate.
- **Cost**: extra collectives, but on shorter tensors than TP's activation
  all-reduces. Worth it when sequence length is the memory bottleneck.
- **Cluster fit**: usually layered on top of TP within the same node
  (they share the same fast-fabric group).

### 5. Expert parallel (EP)

Only relevant for Mixture-of-Experts (MoE) models. Different experts are
placed on different ranks; tokens are routed to their assigned experts
with an **all-to-all**, processed, then routed back with a second
all-to-all.

- **Collective**: two all-to-alls per MoE layer (dispatch + combine).
- **Solves**: parameter memory (experts are massive; you shard the expert
  dimension across ranks).
- **Cost**: all-to-all traffic is highly sensitive to load imbalance
  between experts. See the GShard paper (Lepikhin et al., 2020) and the
  Switch Transformer paper (Fedus et al., 2021) for the routing math, and
  DeepSpeed-MoE (Rajbhandari et al., 2022) for the engineering side.
- **Cluster fit**: EP is usually another axis in a 3D or 4D mesh; the
  Llama 3 paper shows how these axes are composed.

### 6. Hybrid Sharded Data Parallel (HSDP)

Not a new axis so much as a decomposition of the DP axis into a fast inner
axis (shard-on-NVLink) and a slower outer axis (replicate-across-fabric).
Chapter 3 introduced it; the key idea is that you take the FSDP algorithm
and put the *shard* group on the fast link and the *replicate* group on the
slow link.

- **Collective**: intra-shard-group all-gather + reduce-scatter (fast),
  inter-replica all-reduce on gradients (slow, on the outer axis, so
  smaller absolute size).
- **Solves**: FSDP's all-gather over the entire fabric is expensive at
  large `N`; HSDP shrinks the all-gather cost by scoping it to a fast
  group.

## Putting it together: 2D, 3D, and 4D meshes

A common composition:

- **Node-level**: TP = 8, one GPU per shard, all-reduces on NVLink.
- **Pod-level**: PP = 8 across nodes, each stage is a full node.
- **Cluster-level**: DP = the rest, sharded (FSDP-style) or replicated
  (DDP-style).

Total ranks = `TP × PP × DP`. In PyTorch this is a `DeviceMesh` with
three named dimensions; in JAX it is a `Mesh` with three axes and a
`PartitionSpec` per tensor. Adding SP promotes it to 4D
(`TP × SP × PP × DP`); adding EP for MoE promotes to 5D.

Two rules of thumb for laying out this mesh:

- **Fast collectives on fast links.** TP's all-reduces stay on NVLink.
  PP's point-to-points and DP's all-gathers cross RDMA. That is the whole
  reason 3D-parallel beats 1D-DP once the model is big enough.
- **Powers of two.** Every axis size divides the world size and the
  model dimensions cleanly. You will occasionally break this rule for
  cluster-shape reasons; make sure the paper trail says why.

## The decision procedure

Given `(hidden_size H, num_layers L, seq_len T, batch B, cluster shape C)`,
work top-down:

1. **Can one GPU hold a full parameter shard including its optimizer
   state?** If yes → start with DDP or FSDP2 depending on how tight you are
   on memory. If no → you *must* introduce TP or PP; keep reading.
2. **Can one node (`GPU × 8`) hold a full layer's parameters plus its
   activations?** If yes → TP within the node covers you; use FSDP or HSDP
   across nodes.
3. **Can one node hold one *layer*'s parameters but not the entire
   model?** → TP for the layer, PP across nodes for depth.
4. **Is the sequence length long enough that activation memory
   dominates?** → add SP on top of TP.
5. **Is it a MoE architecture?** → add EP with a plan for load balancing.
6. **Do you have more nodes than the strategy so far consumes?** → wrap
   the whole thing in DP (typically FSDP or HSDP).

Every step introduces at least one new collective. Chapter 5 turns those
collectives into wall-clock numbers.

## What Llama 3, BLOOM, and OPT-175B actually chose

We will look at these in depth in chapter 6, but as a preview:

- **Llama 3 405B** (Grattafiori et al., 2024) — 4D parallel:
  TP × CP × PP × DP with FP8 mixed precision. Uses interleaved
  pipeline scheduling to hide the bubble. Discusses network
  reliability and training-loop robustness in the paper.
- **BLOOM-176B** (Le Scao et al., 2022, "BLOOM: A 176B-Parameter Open-
  Access Multilingual Language Model") — Megatron-DeepSpeed 3D-parallel:
  TP inside a node, PP across nodes, DP on the outside, with ZeRO-1.
- **OPT-175B** (Zhang et al., 2022, "OPT: Open Pre-trained Transformer
  Language Models" and its logbook) — fully sharded data-parallel via
  Megatron + FSDP-like sharding, with extensive failure-mode
  documentation in the logbook.

The design procedure above is what those teams executed; you can retrace
their choices by reading the paper with the axes in this chapter on hand.

## Summary

- Six axes: DP, TP, PP, SP, EP, HSDP. Each solves a specific memory or
  throughput problem and pays for it with a specific collective.
- Map fast collectives (TP all-reduces, FSDP all-gathers) onto fast links
  (NVLink); map slow collectives (DP all-reduces) onto slow links (RDMA).
- Real production strategies compose 3–5 axes as a device mesh. The
  decision procedure is top-down: figure out what will not fit on the
  next-larger unit, and add the axis that solves *that* problem.
