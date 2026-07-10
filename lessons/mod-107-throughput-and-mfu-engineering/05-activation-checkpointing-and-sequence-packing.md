# Activation Checkpointing and Sequence Packing

Chapter 1 named two gap terms this chapter closes:

- **Activation recomputation forward FLOPs** (5–15 pp of the
  HFU-vs-MFU gap). Activation checkpointing is a memory-for-compute
  trade: you save HBM by throwing away activations on forward and
  recompute them on backward. That extra forward is a real FLOP cost
  that shows up in HFU but not MFU.
- **Padded / short sequences** (5–20 pp of MFU). If your batch is 32
  documents of average length 500 padded to `max_seqlen=4096`, you
  are computing on 84% pad tokens. Sequence packing turns that batch
  into one packed 16 000-token block; suddenly you are computing on
  the tokens you meant to.

Both mechanisms are integration-layer choices with clear knobs. The
trick is knowing when each pays.

The primary references:

- **Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). "Training
  Deep Nets with Sublinear Memory Cost."** arXiv:1604.06174. The
  gradient-checkpointing paper; introduces the memory-for-compute
  trade and the ~√N-in-memory / ~1× extra forward pass.
- **Korthikanti, V., et al. (2022). "Reducing Activation
  Recomputation in Large Transformer Models."** arXiv:2205.05198.
  The Megatron paper that introduces *selective* recomputation and
  sequence parallelism.
- **PyTorch checkpointing docs.**
  https://pytorch.org/docs/stable/checkpoint.html
- **FlashAttention `varlen` API.**
  https://github.com/Dao-AILab/flash-attention (README's "Interface"
  section and `flash_attn_varlen_func` docstring).

## The memory / compute trade for activation checkpointing

Backprop through a transformer block needs the block's forward
activations (input, all intermediate tensors) so autograd can
compute gradients. By default PyTorch stores every intermediate
tensor. For a transformer block of hidden `h` and sequence `s` in
BF16, the per-block activation footprint is roughly:

```
mem_act_per_block ≈ s · h · (attention terms + MLP intermediate + norm) · 2 bytes
```

At `s = 8192`, `h = 8192`, this is on the order of ~200 MB per
block. With 80 blocks that is ~16 GB per rank *just* for stored
activations — comparable to the parameter memory of a mid-sized
model.

Activation checkpointing runs the block's forward *twice*: once as
a normal forward (but only the *inputs* to the block are stored),
and again on backward to reconstitute the intermediates so autograd
can walk them. The block outputs are always kept; only the
intermediates disappear.

Per the Chen et al., 2016 analysis:

- **Memory:** shrinks from `O(N · s · h)` (one N-block model, all
  intermediates) to `O(√N · s · h)` if you checkpoint every √N
  block, or `O(1)` extra intermediates if you checkpoint every
  block. In practice you checkpoint every block for the largest
  memory savings.
- **Compute:** adds ~1× the forward FLOPs of the checkpointed
  fraction. For every-block checkpointing on a full transformer,
  that is ~1× the whole-model forward, which is ~33% extra
  step FLOPs (forward = 1× cost, backward = 2×, forward+backward =
  3×; extra forward adds 1 more to the denominator: 4/3 ≈ 1.33).

So the MFU accounting: full activation checkpointing costs ~33%
extra HFU. It does not change MFU as long as the recomputed FLOPs
don't count in the numerator — which is the definition. The
question is whether you can trade that 33% for enough memory to
allow a larger batch that raises MFU by more than 33%. The answer is
often yes for large models on H100 80GB.

## The PyTorch API

Two APIs, in increasing order of framework opinionatedness.

### `torch.utils.checkpoint.checkpoint`

The low-level primitive:

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(torch.nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        y = self.attention(self.norm1(x)) + x
        y = self.mlp(self.norm2(y)) + y
        return y
```

Three flags:

- `use_reentrant=False`. Use the modern non-reentrant
  implementation. The reentrant one predates `torch.autograd.Function`
  improvements and is retained for backwards compatibility; do not
  use it in new code.
- `preserve_rng_state=True` (default). Save and restore the RNG
  state around the checkpoint so recomputation produces the same
  dropout / random values as the original forward.
- `context_fn=lambda: (contextlib.nullcontext(), contextlib.nullcontext())`
  for advanced use (e.g., recording NVTX ranges only on the second
  forward). Rarely needed.

### `apply_activation_checkpointing` for FSDP2

Under FSDP2 you usually apply checkpointing to every wrapped block
with a helper:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import fully_shard

# Wrap every TransformerBlock with checkpointing.
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
        m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    ),
    check_fn=lambda m: isinstance(m, TransformerBlock),
)

# Then shard.
for block in model.blocks:
    fully_shard(block, mesh=mesh, mp_policy=mp_policy)
fully_shard(model, mesh=mesh, mp_policy=mp_policy)
```

Order matters: apply checkpointing first, then FSDP2 shards the
checkpointed module. The stored input to the checkpoint boundary is
the FSDP2 *pre-all-gather* input, so the memory saving includes the
transient full parameters being freed after forward.

## Granularity: full-block vs. selective

Full-block checkpointing is the easy default. Selective is the
Korthikanti et al., 2022 refinement: check-point only the *expensive-
to-store, cheap-to-recompute* operations. Their canonical selection
for a transformer block:

- **Recompute**: attention (the `s × s` matrix is huge to store; the
  matmul is cheap to redo).
- **Keep**: MLP intermediates (large but the recompute cost is
  ~2× a matmul).

Selective checkpointing gets you most of the memory saving at a
fraction of the extra FLOPs. The Megatron paper reports 5×
memory reduction at 2.7% extra FLOPs (vs. full at 33%). At scale
this is a substantial MFU win because the "extra FLOPs" fraction of
the step time drops.

Under FSDP2 you get selective checkpointing by writing a
`policy_fn` in your `_forward`:

```python
class TransformerBlock(torch.nn.Module):
    def _forward(self, x):
        # Only the attention region is checkpointed.
        y = checkpoint(self._attn_forward, x, use_reentrant=False) + x
        y = self.mlp(self.norm2(y)) + y
        return y

    def _attn_forward(self, x):
        return self.attention(self.norm1(x))
```

The choice of *which* ops to checkpoint should be driven by a
memory report (see `torch.cuda.memory_stats()` or PyTorch's
`torch.cuda.memory._record_memory_history` + Chrome trace viewer)
plus a step-time A/B. Do not guess.

<!-- needs-research: cross-check current FSDP2 selective-checkpointing API surface against the latest torchtitan reference and update the code snippet if the API has moved to a more ergonomic form. -->

## When to checkpoint

The decision matrix:

| Scenario                                          | Checkpoint? |
|---------------------------------------------------|-------------|
| Model fits in HBM with room to double batch       | No          |
| Model fits, batch is already at compute-optimal   | No          |
| Model fits but batch is small → MFU low           | Yes, selective |
| Model does not fit at any batch                   | Yes, full   |
| Long context (`s ≥ 16k`) with FA                  | Selective; attention is the expensive one |

Two failure modes to watch for:

- **Checkpointing *and* small batch.** You paid 33% extra FLOPs to
  save memory you did not need. HFU goes up (more FLOPs per second)
  but MFU stays flat or drops.
- **Checkpointing *and* CPU offload.** FSDP2's `CPUOffloadPolicy`
  (see mod-102 chapter 4) already trades memory for compute. Layering
  checkpointing on top can compound the overhead. Measure both
  independently before combining.

## Sequence packing

Every transformer block does `O(B · s²)` attention work when the
batch is padded to `s`. If the average document is much shorter than
`s`, most of that work is on pad tokens. Sequence packing
concatenates multiple documents into one packed sequence of length
`s`, with a per-document attention mask that prevents cross-document
attention.

Numerically, the packed forward on 8 documents of average length 500
is *identical* to 8 separate forwards on those documents — as long
as the attention mask correctly blocks cross-document tokens. So
packing produces the same gradient (up to floating-point
associativity) as running the documents separately.

The MFU impact depends on how skewed your sequence-length
distribution is. For a natural-text pretraining corpus:

- Fixed-length batches (pad to `max_seqlen`): typical utilization
  60–80% (20–40% of the compute is on pad).
- Packed batches: typical utilization > 95%.

That translates to 5–20 pp of MFU on pretraining datasets.

## FlashAttention `varlen` and `cu_seqlens`

Eager attention with a block-diagonal mask still computes the full
`s × s` matrix; the mask only zeroes out the pad entries. To *not*
compute cross-document attention, you need a kernel that understands
variable-length sequences.

FA's `flash_attn_varlen_func` is the standard interface:

```python
from flash_attn import flash_attn_varlen_func
import torch

# 8 documents packed into one 4096-token sequence.
# doc_lengths = [512, 480, 600, 500, 520, 490, 508, 486]
# cu_seqlens: cumulative sums, prepended with 0.
# shape (batch+1,) = (9,) for one packed sequence.
cu_seqlens = torch.tensor(
    [0, 512, 992, 1592, 2092, 2612, 3102, 3610, 4096],
    dtype=torch.int32, device="cuda",
)
max_seqlen = 600

# q, k, v are flattened: (total_tokens, num_heads, head_dim).
# For one packed sequence of 4096 tokens: (4096, num_heads, head_dim).
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    dropout_p=0.0,
    causal=True,
)
```

The kernel internally partitions the work at document boundaries;
token `t_i` in document 3 attends only to tokens `t_j` with `j ≤ i`
inside document 3. No compute is spent on the cross-document zeroed
entries.

For pretraining setups, `cu_seqlens` typically has O(10–50) entries
per packed sequence — one per document.

## Packing at the data-loader level

Packing is a data-side concern; the loader assembles packed
batches and produces `cu_seqlens` per batch. Two patterns:

- **Static packing (offline).** Pre-tokenise the corpus into fixed
  packed shards during data prep. Each shard is a stream of packed
  `max_seqlen`-length blocks with a `cu_seqlens` sidecar. Loading is
  a simple sequential read; there is no per-batch bin-packing.
  MosaicML StreamingDataset (mod-103) supports this pattern
  directly.
- **Dynamic packing (online).** The loader maintains a bin-packing
  queue: as documents arrive from tokenisation it packs them into
  `max_seqlen` blocks greedily. Gives higher packing efficiency
  (approaches 100%) but adds loader complexity and non-determinism
  concerns (order-dependent packs). Ray Data or a custom collator can
  implement this.

Both patterns feed the same FA `varlen` API in the model. The choice
is a data-pipeline call (mod-103's scope); this chapter's concern is
that the model uses the `varlen` interface, not the fixed-length one.

## Interaction with checkpointing

Sequence packing and activation checkpointing compose naturally:

- Packed sequences have *higher* activation memory per batch (fewer
  pad tokens = more real activations). So packing tends to push you
  *toward* checkpointing.
- Checkpointing recomputes on the same packed inputs, so nothing
  about the `cu_seqlens` interface changes.

The composition is standard on production pretraining runs. The
Llama 3 pretraining report notes both packed sequences and
selective activation recomputation as concurrent MFU levers.

## Interaction with FSDP2 and TP

- **FSDP2.** Both checkpointing and packing are model-side; FSDP2
  operates on the sharded parameters and does not know or care
  about the sequence layout. Wrap the checkpointing on the block
  first, then `fully_shard`.
- **Tensor parallel.** Under Megatron TP the attention heads are
  head-sharded. FA `varlen` operates head-wise; the TP shard is
  orthogonal. The `cu_seqlens` tensor is broadcast across the TP
  group so all ranks agree on the document boundaries.
- **Context parallel (CP).** CP splits the sequence dimension
  across ranks. Packing composes with CP but the `cu_seqlens`
  layout gets more complex (each CP rank sees a contiguous slice
  of the packed sequence). Mod-101 chapter 4 covers CP; here it is
  enough to note that packing and CP are not mutually exclusive.

## Measuring the MFU delta

The A/B protocol:

1. Fixed batch, fixed dtype, no packing, no checkpointing. Measure
   MFU baseline.
2. Turn on packing (loader change, model uses `flash_attn_varlen_func`).
   Re-measure. Delta is the "pad reduction" MFU term.
3. If OOM at the larger batch enabled by packing, turn on
   checkpointing. Re-measure. Delta is the "checkpointing enabled
   larger batch" MFU term — this can be net positive even though
   checkpointing costs FLOPs, because the batch size increase
   raises useful FLOPs / step by more than 33%.

Cross-check: HFU should go up when you turn on checkpointing (more
compute per step), and MFU should also go up if the batch increase
paid off. If HFU rises but MFU falls or is flat, checkpointing did
not buy you enough memory to matter — remove it.

## Summary

- Activation checkpointing trades ~33% extra forward FLOPs (full-
  block) for a large HBM-activation reduction, per Chen et al., 2016.
  Use `torch.utils.checkpoint` with `use_reentrant=False`, or
  `apply_activation_checkpointing` under FSDP2.
- Selective checkpointing (Korthikanti et al., 2022): checkpoint
  attention, keep MLP intermediates. Gets most of the memory saving
  at ~3% extra FLOPs. Standard on Megatron / torchtitan pretraining.
- Checkpoint when the memory saving lets you grow the batch enough
  that the MFU gain beats the recompute cost. Do not checkpoint
  when the model already fits comfortably.
- Sequence packing concatenates variable-length documents into
  fixed-length packed sequences with a `cu_seqlens` mask.
  FA's `flash_attn_varlen_func` is the kernel; the mask blocks
  cross-document attention exactly.
- Packing eliminates pad-token compute; typical MFU lift 5–20 pp on
  pretraining corpora, larger on very-skewed sequence-length
  distributions.
- Both mechanisms compose with FSDP2, TP, CP, BF16, FP8, and each
  other. Prove every combination with an A/B MFU + HFU
  measurement — HFU rising while MFU stays flat is the signature
  of a costly configuration.
