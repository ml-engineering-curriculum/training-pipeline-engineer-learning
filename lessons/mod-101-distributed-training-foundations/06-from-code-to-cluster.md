# From Code to Cluster: Running a 1B Job, then Porting to FSDP2 and Megatron-TP

The previous chapters were about *reasoning*. This chapter is about *doing*.
The goal is to walk a ~1B-parameter decoder-only transformer from a
single-GPU training loop to DDP on 2–8 GPUs, then to FSDP2, then to
Megatron-style tensor parallel — and at each transition, be able to explain
what changed in the code, what changed in the collective pattern, and what
changed in the observable step time and memory. Exercises 4 and 5 make you
execute this transition.

## Why 1B parameters as the anchor

- Small enough to fit on a single 80 GB GPU with a stock optimizer, so you
  can compare DDP to a single-GPU baseline. This lets you compute strong
  scaling directly.
- Large enough that FSDP2 has something to shard. Below ~500M the
  overhead of the sharding infrastructure dwarfs the savings.
- Same size class as the smallest Llama-family checkpoints, so the
  configuration is well-documented and comparable.

Concretely, a 1B decoder-only transformer at reasonable dims: `hidden ≈
2048`, `layers ≈ 24`, `heads ≈ 16`, vocab `~32k`, seq_len `~2048`.

## Step 0 — Single-GPU baseline

You should be able to write this in your sleep by now (mod prerequisites).
Key pieces the rest of the chapter references:

- A decoder-only `nn.Module` composed of `L` identical transformer blocks
  plus an embedding and an LM head.
- `torch.optim.AdamW` (fused Adam if the CUDA build supports it).
- BF16 autocast or a manual BF16 parameter cast — pick one and be
  consistent.
- A `DataLoader` reading pre-tokenized shards.

The step counter, throughput (tokens/sec), and GPU memory high-water mark
are the numbers you compare against every subsequent stage.

## Step 1 — DDP on 2–8 GPUs

Only three things change from the single-GPU loop:

1. `torch.distributed.init_process_group(backend="nccl")` at the top,
   `destroy_process_group()` at the bottom.
2. `model = DDP(model, device_ids=[local_rank])`.
3. `DataLoader` uses a `DistributedSampler` and its `set_epoch(epoch)` is
   called every epoch.

The loop body is identical. Launch with `torchrun --nproc-per-node=N`.

**Numbers to record**:

- Wall time per step at `N = 1, 2, 4, 8`.
- GPU memory high-water mark per rank (`torch.cuda.max_memory_allocated`).
- Ideal scaling would be linear; observe your actual scaling and identify
  where the loss comes from (all-reduce, dataloader, warmup).

Also record what does *not* change: per-rank memory. DDP is not a memory
optimization.

## Step 2 — FSDP2

The intent is: same batch size per rank, but memory drops substantially
because parameters + gradients + optimizer state are sharded.

Structural changes:

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

mp = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

for block in model.blocks:
    fully_shard(block, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)
```

Everything else — optimizer, loss, backward, step — is unchanged. This is
the payoff of FSDP2's per-parameter sharding: the training loop stays a
DDP-shaped loop.

**Numbers to record**:

- GPU memory per rank compared to DDP. Expect roughly a `1/N` reduction in
  parameter + optimizer memory (activations are unchanged).
- Step time. Expect it to go *up* on a small cluster where DDP already fit
  comfortably. This is the 1.5× wire volume from chapter 3 showing up in
  wall time.
- The comm-to-compute ratio: profile with `torch.profiler` and identify
  whether the all-gathers of layer `i+1` are being prefetched behind the
  compute of layer `i`.

Common failures you will run into and their causes:

- **`RuntimeError: <collective> op failed`** with mismatched shapes across
  ranks — you initialized the model differently on different ranks. Fix
  by seeding before `fully_shard` and initializing on `meta` device
  where possible.
- **Optimizer state is huge on rank 0** — you called
  `torch.save(state_dict())` without gathering shards. Use PyTorch DCP
  (see mod-106) or `full_state_dict` with explicit rank-0-only saving.
- **`ValueError: FSDP parameters are not properly sharded`** — you passed
  raw `nn.Parameter` slices somewhere expecting a full tensor. Fix by
  materializing with `.full_tensor()` at the boundary.

## Step 3 — Megatron-style tensor parallel

The FSDP2 version scales in the DP direction. To make each *layer* smaller,
you now introduce a tensor-parallel axis. On an 8-GPU node this typically
means TP=8 (all-inside-node NVLink) at first.

You have two implementation paths:

- **Use Megatron-LM directly.** Follow the Megatron-LM
  `pretrain_gpt.py` example, replacing your dataset with your loader. The
  model has to be rebuilt using Megatron's layers (`ColumnParallelLinear`,
  `RowParallelLinear`, `VocabParallelEmbedding`), which handle the
  all-reduces internally.
- **Use PyTorch `torch.distributed.tensor.parallel` with FSDP2.**
  Build a 2-D device mesh (`("tp", "dp")`), apply `parallelize_module`
  with `RowwiseParallel` / `ColwiseParallel` plans on the transformer
  block, and let FSDP2 shard on the DP dimension. This is the mesh path
  the PyTorch docs (see `torch.distributed.tensor.parallel`) endorse for
  new code, and it is what torchtitan uses.

Either way, the collective pattern you introduce is:

- Two all-reduces per forward per block (Megatron's column-then-row MLP
  and column QKV + row output projection).
- Two more on the backward.
- All-reduces happen on the **activation-sized** tensor, not the
  parameter-sized tensor. That's a very different budget from DDP's
  parameter all-reduce.

**Numbers to record**:

- The TP intra-node all-reduce time from `nccl-tests` at
  `payload = B_micro · T · H · sizeof(activation_dtype)`.
- Per-GPU memory: parameters shrink by `TP`, activations often shrink too
  if sequence-parallel is enabled.
- Step time and TFLOPs / GPU. Compare to the FSDP2-only number. If TP is
  set up correctly, throughput at the same global batch should go up
  because per-GPU compute now fits in a smaller working set (better cache
  behavior for matmuls).

Common failures:

- **Cross-node TP** — someone put TP=16 across two nodes. The all-reduce
  now traverses IB every layer, which is catastrophic. Verify with
  `NCCL_DEBUG=INFO` that the TP process group's ranks are all on the
  same host.
- **Divergent RNG in dropout** — with TP, `torch.nn.Dropout` needs its
  RNG to agree on the ranks that share the same activation slice.
  Megatron-LM tracks this with a "tensor-parallel RNG state"; PyTorch's
  `parallelize_module` needs the equivalent through `SequenceParallel`
  wrappers.
- **Loss curves diverge from DDP baseline** — usually because the
  activation-parallel path is not layered on top correctly. Sanity check
  by running with TP=1 on the TP-enabled code path and confirming
  bit-for-bit (up to nondeterminism) parity with the FSDP2-only baseline.

## The transition table

| Stage      | Where the state lives                         | Dominant collective              | Extra memory saving |
|------------|-----------------------------------------------|----------------------------------|---------------------|
| Single GPU | Everything on one GPU                          | none                             | baseline            |
| DDP        | Full copy of everything on every rank          | all-reduce per bucket             | none                |
| FSDP2      | 1/N of params/grads/opt-state per rank         | all-gather + reduce-scatter       | ~N× on optimizer    |
| FSDP2 + TP | 1/(N·TP) of params, activations sharded on TP  | + intra-node all-reduce per block | + per-layer memory  |

## Building the muscle

Exercises 4 and 5 are the codified version of steps 2 and 3. When you
finish them you should be able to answer, without looking at code:

- What memory line item shrank at each transition and why.
- Which collective got added at each transition and which link it lands on.
- What the observable step-time story looks like — where you *gained*
  performance and where you *paid* for the extra flexibility.

## Summary

- DDP → FSDP2 → FSDP2 + TP is the standard scaling ladder for a modern
  training platform.
- Each transition trades a specific memory line item for a specific
  additional collective.
- Do each transition with numbers in hand: memory before and after,
  step time before and after, comm-to-compute ratio before and after. The
  numbers are what the model, cluster, and strategy have to agree on.
