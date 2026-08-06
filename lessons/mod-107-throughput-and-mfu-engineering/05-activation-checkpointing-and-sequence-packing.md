# Activation Checkpointing, Gradient Checkpointing, and Sequence Packing

Memory is the second-tightest budget of a training step (after step
time). Once you have picked BF16/FP8 and FlashAttention, the next
question is: *how do I fit the working set into HBM without spilling
into slow paths?* The three tools this chapter covers — activation
checkpointing, gradient checkpointing, and sequence packing — are
how you trade compute for memory (the first two) or reshape work to
avoid wasted padding (the third).

The goal is to hit a fixed HBM budget while paying as little MFU as
possible. Every technique here has a compute cost; the game is
picking the cheapest combination.

## The training-step memory budget

Per-GPU HBM at a given step is consumed by roughly four things. On
H100 SXM5 you have 80 GB (or 141 GB on H200) to divide among:

- **Sharded parameters.** With FSDP2, each rank holds `total_params /
  world_size * bytes_per_param`. In BF16 with 128 ranks and a 30B
  model, that is `30e9 / 128 * 2 = 468 MB` — small.
- **Sharded optimizer state.** AdamW is two FP32 moments per
  parameter: `2 * total_params / world_size * 4` bytes. Same 30B on
  128 ranks: `2 * 30e9 / 128 * 4 = 1.87 GB`. Still small on the per-
  rank axis; it is the *total* fleet cost that dominates.
- **Gradient buffers.** During the backward pass, gradients are
  materialized before reduce-scatter. Size is comparable to the
  parameter shard.
- **Activations saved for backward.** *This is the dominant term,*
  and it grows with batch size, sequence length, and (crucially)
  model depth. Un-checkpointed, a 30B decoder-only model at seq
  4096 with a typical micro-batch can hit tens of gigabytes of
  activations per rank.

Activations are the dominant term because for every layer you must
retain the input to the layer (to compute the input-gradient in
backward). For an `L`-layer Transformer with hidden dim `d_model`
and batch × sequence total `T` tokens per rank:

```
activations_per_layer ≈ constant · T · d_model · bytes_per_activation
total_activations     ≈ L · activations_per_layer
```

The constant depends on how many intermediate tensors your specific
block architecture keeps (attention Q/K/V, MLP intermediate, layer-
norm outputs). Empirically 4–20× `T · d_model · 2` bytes for BF16 is
the range. The point is: activations scale with model *depth* and
*batch × sequence*, and they are what your knobs mostly target.

## Activation checkpointing (a.k.a. gradient checkpointing)

Activation checkpointing trades compute for memory: for a
checkpointed segment of the network, only the *inputs* to the
segment are saved during forward. During backward, the segment's
forward is *re-executed* on the fly to reproduce its intermediate
activations before computing gradients.

Primary reference: **Chen et al., 2016. "Training Deep Nets with
Sublinear Memory Cost."** arXiv 1604.06174. The original sublinear-
memory technique. Modern PyTorch exposes it through
`torch.utils.checkpoint.checkpoint`.

Cost model:

- **Memory saved:** the activations inside the checkpointed segment
  no longer sit in HBM. For full-block checkpointing of every
  Transformer block, activation memory drops to roughly the
  activation footprint of *one* block instead of `L` blocks —
  order-of-magnitude reduction.
- **Compute overhead:** each checkpointed segment runs its forward
  *twice* per step (once during actual forward, once during
  backward's recompute). If the segment is the whole block, this
  adds one full forward's worth of FLOPs per backward. Total step
  cost goes from "1 forward + 2 backward equivalents" to "2 forward
  + 2 backward" — roughly a 33% increase in HFU relative to MFU
  (chapter 1).

The 33% number is the "full activation checkpointing" ceiling — you
recompute *everything*. Two more selective schemes cost less:

- **Selective activation checkpointing.** Only the most expensive-
  to-recompute or most memory-heavy operations are recomputed;
  cheap-to-materialize ones are still cached. PyTorch's
  `SAC` (selective activation checkpointing) and Megatron's
  selective recompute schedules implement this. Typical cost: 10–20%
  extra FLOPs for close to the memory saving of full AC.
- **No checkpointing.** For short-context, low-batch training, the
  activations fit and you pay no compute penalty.

PyTorch usage (per-block):

```python
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def forward(self, x, ...):
        return checkpoint(self._forward_impl, x, ..., use_reentrant=False)

    def _forward_impl(self, x, ...):
        # real block body
        ...
```

Use `use_reentrant=False`. The reentrant path is legacy and does not
compose cleanly with autograd hooks that FSDP2 relies on.

FSDP2-friendly wrapper (from the FSDP2 tutorial):

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    CheckpointImpl,
)
apply_activation_checkpointing(
    model,
    check_fn=lambda m: isinstance(m, Block),
    checkpoint_wrapper_fn=lambda m: torch.utils.checkpoint.checkpoint_wrapper(
        m, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    ),
)
```

## The "gradient checkpointing" naming confusion

"Gradient checkpointing" and "activation checkpointing" refer to the
same technique. The community uses both names for the Chen et al.
2016 approach. HuggingFace's `model.gradient_checkpointing_enable()`
is the same mechanism. There is no separate "gradient checkpointing"
knob distinct from AC. If you see both terms in a design doc,
confirm they mean the same thing to the author.

## Sequence packing

Sequence packing is a completely different tool: it does not reduce
per-token activation cost — it eliminates the *wasted work* padding
short sequences to a maximum length. In LLM pretraining the input is
a stream of variable-length documents; the naive tokenizer output has
every batch element padded to `max_seq_len` with `[PAD]` tokens the
model computes over but ignores.

Two techniques:

- **Static packing (sample concatenation).** Concatenate multiple
  short sequences into one length-`max_seq_len` sequence with an
  attention mask that prevents cross-sequence attention. The model
  sees "one long sequence" that is actually many short ones. Every
  token is a real token; padding is minimized.
- **Dynamic packing (variable-length forward).** Same idea, but use
  a variable-length attention API (`flash_attn_varlen_func` from
  chapter 2) that natively takes cumulative-sequence-length arrays
  and skips the cross-sequence attention entirely. No padding, no
  mask materialization.

The lift depends on the sequence-length distribution of your dataset.
For a distribution with mean length 30% of max, naive padding wastes
~70% of the compute per step. Packing turns that back into real
tokens: same wall-clock, 3× the training tokens. That is a *3× MFU
improvement* on the numerator (chapter 1's `B · S` factor becomes
tokens-with-actual-content, not tokens-with-padding).

Packing interacts with sampler and loss:

- **Sampler.** The dataloader must produce packed batches where the
  cumulative sequence lengths (`cu_seqlens`) reach `max_seq_len`
  without exceeding it. This is bin-packing; a simple greedy
  algorithm works well.
- **Loss.** Standard cross-entropy over the flattened output
  produces the correct per-token loss automatically — no mask
  needed because there is no padding.
- **Attention.** Use `flash_attn_varlen_func` from chapter 2 to
  avoid materializing the `(seq_len × seq_len)` mask. For very
  long packed sequences, that mask alone would exceed HBM.
- **Position IDs.** Reset within each packed sub-sequence — every
  document starts at position 0. Otherwise your model sees monotonic
  position IDs that jump across documents.
- **Random shuffling.** Repack every epoch to avoid learning any
  packing artifact.

## The combined memory-target playbook

Given a fixed HBM budget per rank `M`, walk this decision list:

1. **Measure current per-rank HBM at your desired micro-batch and
   sequence length.** Use `torch.cuda.max_memory_allocated()` to
   isolate PyTorch-tracked memory vs. driver overhead. If you are
   below `M` with headroom, stop and buy the throughput win of a
   larger micro-batch.
2. **If over budget, first try sequence packing.** Compared to full
   activation checkpointing, packing has *zero* MFU cost on the
   packed tokens and may even raise MFU by removing padding waste.
   It only helps if your data has short sequences to pack; if you
   are already at full-length sequences, skip.
3. **If still over budget, try selective AC before full AC.**
   Selective AC's compute cost is 10–20%, vs. full AC's ~33%.
4. **If still over budget, apply full AC to every block.**
5. **If still over budget, reduce the micro-batch.** Rebalance by
   raising gradient accumulation steps to keep the effective batch
   size constant. This has near-zero compute cost per unit token but
   changes the shape of BatchNorm-like statistics (irrelevant for
   most LLM training that uses LayerNorm/RMSNorm).
6. **If still over budget, revisit whether the run should fit at
   this scale.** Dropping micro-batch below 1 sequence per rank
   requires model parallelism (tensor parallel, pipeline parallel)
   which is a mod-101 topic. This module owns the FSDP2 axis.

The order matters: sequence packing is *free MFU*; AC costs some
MFU; further batch reduction with gradient accumulation costs no
MFU per token but changes convergence dynamics slightly.

## What the numbers actually look like

Approximate memory savings on a typical decoder-only LLM at
`max_seq_len = 4096`, micro-batch 4 sequences per rank, hidden 4096,
32 layers, BF16:

| Configuration              | Activations HBM per rank | Compute overhead |
| -------------------------- | ------------------------ | ---------------- |
| No AC, no packing          | ~35 GB                   | 0                |
| No AC, packed sequences    | ~35 GB                   | 0 (more useful tokens) |
| Selective AC, packed       | ~10 GB                   | +15%             |
| Full AC, packed            | ~4 GB                    | +33%             |

Numbers are illustrative — validate on your own model with the
memory-instrumentation snippet below. The *shape* of the trade-off
is what to remember: full AC is the sledgehammer, selective AC is
the scalpel, and packing is orthogonal to both.

## Instrumenting activation memory

You cannot pick between the four rows in the table without measuring
activation memory. A minimal snippet:

```python
import torch, gc

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

loss = model(input_ids, labels=labels).loss
loss.backward()

peak_gb = torch.cuda.max_memory_allocated() / 1e9
alloc_gb = torch.cuda.memory_allocated() / 1e9
print(f"peak allocated: {peak_gb:.2f} GB, current: {alloc_gb:.2f} GB")

# For a per-module breakdown, use torch.cuda.memory._record_memory_history
# and torch.cuda.memory._dump_snapshot to produce a memory timeline
# that can be visualized at https://pytorch.org/memory_viz
```

The memory-viz snapshot is the tool that lets you see exactly which
tensors are held across the forward-to-backward boundary and which
disappear on autograd's own. A screenshot of the timeline before and
after enabling AC is the single most useful artifact for "did the
policy do what I thought it did".

## Interactions with the rest of the module

- **FlashAttention (chapter 2)** already reduces the attention
  activation footprint from `O(S²)` to `O(S)` by recomputing the
  softmax during backward. FA + no AC is often already the right
  answer for the attention block; do not additionally AC an FA
  block or you pay for the same recompute twice.
- **BF16 / FP8 (chapters 3, 4)** halve or quarter the per-tensor
  byte count vs. FP32. Enabling BF16 shrinks activation memory
  before any AC decision. Re-measure after enabling BF16 to see
  whether AC is still required.
- **`torch.compile` (chapter 6)** may add its own small memory
  overhead from cache buffers. Re-measure after compile.
- **DCP async save (mod-106)** uses a CPU pinned staging buffer, not
  extra HBM. It does not affect this budget.

## Summary

- Activations, not parameters, dominate per-rank HBM in a well-
  sharded FSDP2 setup. Every knob in this chapter targets that
  term.
- Activation checkpointing (aka "gradient checkpointing") trades
  compute for memory. Full AC costs ~33% extra FLOPs; selective AC
  costs 10–20%; use selective by default and full only when the
  budget demands it.
- Sequence packing eliminates padding waste and *raises* MFU while
  freeing memory. Use `flash_attn_varlen_func` for the attention
  path. Packing is orthogonal to AC and should be tried first.
- The decision order for a fixed memory target: measure → pack →
  selective AC → full AC → reduce micro-batch (with gradient
  accumulation) → model parallelism (out of scope for this
  chapter).
- Instrument the choice with `torch.cuda.max_memory_allocated`
  and the memory-viz timeline snapshots. Do not pick from the
  playbook without measurements.
