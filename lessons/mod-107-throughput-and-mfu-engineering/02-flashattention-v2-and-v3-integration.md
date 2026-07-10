# FlashAttention v2 and v3: Integration, Not Authoring

FlashAttention is the single largest MFU-lift you can install by
changing a config flag. Chapter 1 named "attention kernel overhead"
as a 5–15 pp gap term; this chapter closes it. The scope is
deliberate: you are integrating a published kernel, not authoring
one. The kernel authoring itself — tile shapes, warp scheduling,
softmax numerics, the Hopper WGMMA async pipeline — lives in
`ai-infra-performance-learning` and in the papers this chapter cites.
Your job is to know when FA helps, which version to pick, how to
wire it in, and how to prove the MFU delta.

The two primary references you should keep open:

- **FlashAttention v2 paper.** Dao, T. (2023). "FlashAttention-2:
  Faster Attention with Better Parallelism and Work Partitioning."
  arXiv:2307.08691.
- **FlashAttention v3 paper.** Shah, J., Bikshandi, G., Zhang, Y.,
  Thakkar, V., Ramani, P., & Dao, T. (2024). "FlashAttention-3: Fast
  and Accurate Attention with Asynchrony and Low-precision."
  arXiv:2407.08608.
- **The reference repo.** https://github.com/Dao-AILab/flash-attention

## What FlashAttention actually changes

Naive scaled-dot-product attention (`softmax(QKᵀ / √d) V`) has two
inefficiencies that dominate at long sequence length `s`:

- It **materialises** the `s × s` attention matrix in HBM. The matrix
  is written out of tensor-core SRAM, read back for softmax, written
  back for the softmax result, and read back for the `V` multiply.
  For `s = 8192` and BF16, one head's attention matrix is 128 MB —
  many times the size of any GPU's SRAM/L2 cache.
- It runs three unfused kernels (matmul → softmax → matmul), each of
  which has its own launch overhead, its own dtype-conversion cost,
  and its own uncoalesced HBM traffic.

The result: naive attention is memory-bound at long `s`, and matmul
tensor cores sit idle waiting for HBM bytes.

FlashAttention rearranges the computation into a **tiled** loop over
`Q` blocks and `KV` blocks that keeps the working set in SRAM. The
mathematical core is the online-softmax algorithm (Milakov & Gimelshein,
2018, arXiv:1805.02867), which lets you compute softmax over a stream
of blocks without ever materialising the full row. The effects:

- **No `s × s` matrix in HBM.** HBM traffic scales as `O(s · d)`
  instead of `O(s²)`.
- **One fused kernel.** One launch, one dtype conversion, one HBM
  round trip.
- **Backward keeps only `O(s · d)` state**, using the log-sum-exp
  and softmax normaliser saved on forward, plus a recompute of the
  attention block on backward. Section 3 of the v1 paper and
  section 2 of the v2 paper are the canonical derivations.

Numerically FA is *exact*, not approximate. It computes the same
`softmax(QKᵀ/√d)V` up to floating-point associativity. It is not
"linear attention" or "sparse attention"; the model output is
bit-close to eager attention on the same inputs (small residual from
different accumulation orders).

## v2 vs. v3: what changed

**v2 (Dao, 2023)** is the version you already run on Ampere (A100)
and on Hopper (H100) if you have not enabled v3. Key changes over
v1:

- **Better warp scheduling.** The v1 kernel had one warp per Q-block;
  v2 splits work across warps within a threadblock more evenly and
  reduces the shared-memory footprint by keeping fewer intermediates.
- **Non-causal and causal both fast.** v1's causal path was much
  slower than the non-causal path; v2 closes the gap by splitting the
  causal computation across threadblocks correctly.
- **Reported ~2× throughput vs. v1 on A100 at long context.** See v2
  paper, Table 3.

**v3 (Shah et al., 2024)** is Hopper-specific and requires SM90 (H100
/ H200). Key changes over v2:

- **Warpgroup-level MMA (WGMMA) with async execution.** Hopper's
  WGMMA instruction lets a warpgroup issue a tensor-core matmul and
  proceed to non-tensor work (softmax normalisation, the
  correction step, the next block's `Q` load) while the matmul is in
  flight. v3 pipelines this: while WGMMA A is producing the next
  block, the CUDA cores execute softmax on the previous block. This
  is a producer-consumer pipeline expressed in inline PTX.
- **TMA (Tensor Memory Accelerator) for asynchronous HBM → shared
  memory loads.** The `Q`, `K`, `V` block loads no longer stall the
  warp; TMA copies happen in the background.
- **FP8 support.** Both operand and accumulator paths have FP8
  variants. This matters because FP8 attention pairs with FP8
  matmul in Transformer Engine (chapter 4); without an FP8 attention
  kernel, the encoder/decoder stack would bottleneck on the sole
  BF16 attention step.
- **Reported ~1.5–2× over v2 on H100 in BF16, and ~1.2× on top of
  that in FP8.** See v3 paper, Tables 1 and 2.

The rule of thumb:

- On A100: FA v2 or bust. There is no v3 for Ampere.
- On H100: FA v3 if the toolchain supports it; FA v2 otherwise. v3
  needs recent PyTorch (≥ 2.4) and a recent `flash-attn` release; if
  the pin is old, v2 is fine.

## When FA helps (and when it does not)

FA closes the "attention kernel overhead" term. The magnitude of the
closure depends on how much of your step's FLOPs are in attention:

- **Short-context runs** (`s ≤ 2048`, dense 3–8B model, TP≤4). The
  matmul terms in the MLP dominate. FA is a smaller MFU lift; it
  still helps (better HBM locality on the attention step) but expect
  1–3 pp of MFU, not 10.
- **Standard-context pretraining** (`s = 4096` or `8192`, 7–70B
  dense). FA is a substantial lift: 5–10 pp of MFU is typical,
  because attention is now a nontrivial fraction of the step and its
  HBM traffic dominates the memory-bound axis.
- **Long-context (`s ≥ 16k`).** FA is the difference between a run
  that completes and a run that OOMs. The `O(s²)` HBM allocation
  disappears; attention becomes memory-linear in `s`. Without FA,
  32k context is infeasible on H100; with FA, it is straightforward.
- **Very small `s` (≤ 128).** FA can be *slower* than eager because
  the block sizes are tuned for larger `s`. You would essentially
  never run pretraining at `s ≤ 128`, but it comes up in some
  fine-tuning scenarios; measure before assuming FA wins.

## The API surface

There are three integration paths, in increasing order of framework
opinionatedness:

### 1. Direct `flash_attn` calls

```python
from flash_attn import flash_attn_func

# Q, K, V shapes: (batch, seqlen, num_heads, head_dim)
out = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    causal=True,
    softmax_scale=None,       # defaults to 1/sqrt(head_dim)
    window_size=(-1, -1),     # sliding window; -1 means full
)
```

Use `flash_attn_func` when you own the attention module. It is the
minimum-abstraction path and gives you causal-mask, sliding-window,
and (in v2.5+) attention-sink support.

The `varlen` cousin is the one that pays off with sequence packing
(chapter 5):

```python
from flash_attn import flash_attn_varlen_func

# q, k, v shapes: (total_tokens, num_heads, head_dim), i.e. flattened
# cu_seqlens_{q,k}: cumulative sequence lengths, shape (batch+1,)
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0,
    causal=True,
)
```

The `cu_seqlens` API is what makes packed short-sequence batches
efficient: attention is block-diagonal per document, so
cross-document tokens never attend to each other. Chapter 5 details
the packing pipeline that feeds this API.

### 2. PyTorch `scaled_dot_product_attention` backend flag

PyTorch 2.x ships `torch.nn.functional.scaled_dot_product_attention`
(SDPA), which dispatches internally to one of: FlashAttention (v2 in
recent releases), memory-efficient attention (xFormers-style), or a
math (eager) fallback. You choose a backend via the context manager:

```python
import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend

with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    out = scaled_dot_product_attention(q, k, v, is_causal=True)
```

This path is what you want when the model was written in vanilla
PyTorch — no FA-specific code. It works for any `nn.MultiheadAttention`
or hand-rolled QKV projection that calls SDPA. Documented at
https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
and https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html.

The PyTorch SDPA backend has some limitations vs. calling
`flash_attn_func` directly:

- No `varlen` API (no packed sequences); PyTorch will always
  round to a rectangular `s × s` block per batch.
- FA v3 integration lags. If you need v3-on-Hopper explicitly, use
  the direct `flash_attn` path or a framework that vendors v3.
- `is_causal=True` is the only causal-mask shape supported without
  a fallback to the math backend.

### 3. HuggingFace `attn_implementation`

For HF Transformers-based training, the model constructor takes an
`attn_implementation` argument:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

Supported values include `"eager"`, `"sdpa"` (the PyTorch backend
selector above), and `"flash_attention_2"`. HF documents this at
https://huggingface.co/docs/transformers/attention (see also
https://huggingface.co/docs/transformers/perf_infer_gpu_one). Set
`"flash_attention_2"` when you want the direct FA path with
`varlen` support that HF's data collators can produce.

## Interaction with FSDP2, TP, and `torch.compile`

- **FSDP2.** FA is dtype-transparent as long as you keep `Q`, `K`,
  `V` in the same dtype. Under FSDP2 with
  `MixedPrecisionPolicy(param_dtype=torch.bfloat16)` (chapter 3), the
  QKV projection outputs BF16 and FA runs BF16 natively. There is no
  FSDP2-specific config for FA; it "just works."
- **Tensor parallel.** FA operates on the local attention heads; the
  head split under Megatron-style TP is entirely orthogonal. Wire it
  into the TP-sharded attention block after the QKV projection but
  before the output projection — exactly where eager attention would
  run.
- **`torch.compile`.** FA is a registered custom op with a proper
  meta-kernel, so `torch.compile` traces through it as an opaque call.
  It does not fuse with surrounding ops (the FA kernel is
  hand-written), but it also does not graph-break. Chapter 6 covers
  this in depth.
- **FP8.** FA v3 supports FP8 QKV. FA v2 does not. If you want FP8
  matmul (chapter 4) *plus* FP8 attention, v3 is required.

## The MFU delta measurement

The A/B protocol for attributing an MFU lift to FA:

```python
# Run A: eager attention
model_A = build_model(attn_implementation="eager")
mfu_A = measure_mfu(model_A, num_steps=100, warmup=20)

# Run B: FlashAttention v2
model_B = build_model(attn_implementation="flash_attention_2")
mfu_B = measure_mfu(model_B, num_steps=100, warmup=20)

print(f"FA v2 delta: {mfu_B - mfu_A:+.3f} MFU")
```

Cross-check with the torch profiler: the attention region on run B
should be shorter and the HBM traffic (visible in `nsys` NVTX
annotations) dramatically lower.

If the delta is smaller than expected (< 2 pp) the likely culprits:

- Your `s` is too short; attention was not on the critical path.
- Your MLPs are not being kept fed; the true bottleneck is
  elsewhere (see chapter 7 on comm overlap).
- Padded batches are dominating; measure the token-density of your
  batch and packed alternatives (chapter 5).

## What this chapter does not cover

The kernel itself: warp specialisation, the WGMMA producer-consumer
pipeline, the softmax numerics, the shared-memory bank-conflict
avoidance, the CUTLASS layout choices. That is
`ai-infra-performance-learning`'s scope. If you find yourself needing
to modify FA — new attention biases, custom masking that FA's
`window_size` cannot express, a variant with an ALiBi position bias —
that is the moment to escalate to the performance-engineer track or
to the FA maintainers' repo. Chapter 8 codifies this hand-off.

## Summary

- FlashAttention rearranges attention into a tiled fused kernel with
  no materialised `s × s` matrix; HBM traffic drops from `O(s²)` to
  `O(s · d)`. Numerically exact, not approximate.
- v2 (Dao, 2023, arXiv:2307.08691) is the Ampere and Hopper baseline.
  v3 (Shah et al., 2024, arXiv:2407.08608) is Hopper-only, uses
  WGMMA async and TMA, and adds FP8. Reported ~1.5–2× v3-over-v2 on
  H100 BF16 in the paper.
- Three integration paths: direct `flash_attn_func` /
  `flash_attn_varlen_func`, PyTorch `scaled_dot_product_attention`
  with the `sdpa_kernel` context manager, and HuggingFace
  `attn_implementation="flash_attention_2"`. Pick the highest-level
  one that meets your needs.
- Composes cleanly with FSDP2, TP, and `torch.compile`. Pairs with
  sequence packing via the `varlen` API (chapter 5) and with FP8
  matmul via v3 (chapter 4).
- Prove every FA claim with an A/B MFU measurement and a profiler
  trace. Delta of 5–10 pp is typical at `s = 4096`; smaller at short
  context; larger (up to "run does not OOM") at long context.
- Kernel authoring is out of scope for this module and this track;
  chapter 8 owns the hand-off to `ai-infra-performance-learning`.
