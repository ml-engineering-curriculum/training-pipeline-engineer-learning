# DDP and the All-Reduce Backward Pass

DistributedDataParallel (DDP) is the simplest useful thing you can do with
more than one GPU: replicate the model on every rank, split the batch, and
average the gradients before the optimizer step. Everything more elaborate
in this module — FSDP, ZeRO, tensor / pipeline parallel — is DDP with more
of the "average this thing between ranks" moved earlier in the step. So
you need to be able to draw the DDP forward + backward pass in your sleep.

## Motivation

DDP does *not* solve the memory problem — every rank still holds a full copy
of parameters, gradients, and optimizer state. It solves the *throughput*
problem: `N` GPUs process `N` times the tokens per step, and if the extra
communication is well-hidden, you get an ~`N`× speed-up on the compute-bound
regime.

For any model that fits on one GPU, DDP is the right first move. And even
when the model does not fit on one GPU, DDP's mental model — a synchronous
optimizer step that averages gradients — is exactly what FSDP and ZeRO also
implement; they just also shard the state so the memory footprint drops.

## The one-slide picture

For a batch of size `B` on `N` ranks:

1. Rank `r` reads the sub-batch `B/N` assigned to it.
2. Every rank runs an identical forward pass on its own sub-batch and
   computes a local loss.
3. Every rank runs an identical backward pass and produces a local gradient
   `g_r` for every parameter.
4. Ranks **all-reduce** the gradients so every rank now holds
   `(1/N) * Σ_r g_r` for every parameter. The reduction is `SUM` divided by
   `N` (or equivalently `AVG` where NCCL supports it).
5. Every rank runs an identical optimizer step. Because parameters,
   gradients, and RNG state are identical on entry, the parameters stay
   bit-identical (up to floating-point ordering) after the step. No
   parameter sync is needed.

The only step that actually crosses the network is step 4. That is *the*
DDP operation.

## What the all-reduce actually does on the wire

NCCL implements ring all-reduce (Baidu-style; the reference implementation
predates NCCL) as **reduce-scatter followed by all-gather**. On `N` ranks
with a payload of size `S`:

- Reduce-scatter phase: each rank sends `(N-1)` chunks of `S/N` bytes.
- All-gather phase: each rank sends another `(N-1)` chunks of `S/N` bytes.

So the total per-GPU send is `2 * (N-1)/N * S`, asymptotically `2 * S` as
`N` grows. See Patarasuk & Yuan (2009), "Bandwidth Optimal All-reduce
Algorithms for Clusters of Workstations", for the derivation, and NCCL's
own docs for the specific algorithms it picks (ring, tree, double binary
tree). We will use this `2(N-1)/N` figure a lot in chapter 5.

Two consequences that matter for later chapters:

- The per-GPU communication volume in DDP is **independent of `N` for
  reasonable `N`** — it approaches `2S`. So DDP scales well until latency
  or contention on the fabric dominates.
- Bandwidth-bound → prefer ring. Latency-bound (small tensors, many ranks)
  → prefer tree. NCCL picks per-collective; you can override with
  `NCCL_ALGO` for A/B testing.

## Where the all-reduce fits inside `backward()`

If you called `all_reduce` naively at the end of the backward pass you would
serialize `T_compute + T_comm`. DDP does not do that. Instead:

- On construction, DDP walks the model and assigns each parameter's gradient
  to a **bucket** — a contiguous chunk (default ~25 MB — check
  `DistributedDataParallel(bucket_cap_mb=...)`) that will be all-reduced as
  a single NCCL call.
- Each parameter registers a `post_accumulate_grad_hook` (or equivalent) so
  that when autograd finishes computing that parameter's gradient, DDP is
  notified.
- When every parameter in a bucket has finished its backward, DDP launches
  the all-reduce on that bucket **while the rest of the backward pass is
  still running**.
- At `optimizer.step()` time, every bucket's all-reduce has already been
  awaited (via the DDP reducer's join).

Because parameters finish backwards in *reverse* graph order — output layer
first, embedding last — the bucketing intentionally groups the output
layers together so their all-reduces can overlap with the embedding's
backward. This is the "gradient bucketing" pattern the DDP paper describes
(Li et al., 2020, "PyTorch Distributed: Experiences on Accelerating Data
Parallel Training", §3).

### The `no_sync()` gotcha

`DistributedDataParallel.no_sync()` is a context manager that skips the
all-reduce on backward for the duration of the block. It is the correct way
to accumulate gradients over `K` micro-batches without paying `K` all-reduce
costs — you only all-reduce on the *final* micro-batch. Using it is the
first optimization you should reach for when you have small per-step batches
and want to hide comm.

## Correctness invariants

A working DDP setup obeys:

1. Every rank sees the same total number of steps and hits the same
   `.backward()` / `optimizer.step()` calls in the same order (otherwise a
   collective will deadlock waiting for a peer that never posts).
2. The model, optimizer, and RNG are initialized to the same state on every
   rank (typically by seeding, then broadcasting rank-0's parameters at
   construction — `DistributedDataParallel` does the broadcast for you).
3. Every parameter that ends up with a gradient participates in the
   all-reduce. If some parameters are computed only on some ranks, you need
   `find_unused_parameters=True` (or, more efficiently, `static_graph=True`
   once you know the set is stable).
4. Data is sharded with `DistributedSampler` (or equivalent) so each rank
   sees a distinct slice of the epoch. This is where "epoch drift" bugs
   creep in — see mod-103 for the war stories.

## Minimum working example

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = build_model().cuda()
    model = DDP(model, device_ids=[local_rank])

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
                                         batch_size=micro_batch)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # deterministic per-epoch shuffle
        for step, batch in enumerate(loader):
            loss = model(batch).loss
            loss.backward()          # gradient bucketing + all-reduce here
            optim.step()
            optim.zero_grad()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Launch:

```bash
torchrun --nproc-per-node=8 train.py
```

You should be able to walk a colleague through every line of this file and,
for `loss.backward()`, describe which parameters finish first, when their
bucket becomes ready, when the NCCL all-reduce is enqueued, and when
`optimizer.step()` observes the averaged gradient. Exercise 1 will make you
do that on paper.

## What DDP does **not** solve

The reason we do not stop the module here:

- **Memory** — every rank still holds full model, gradient, and optimizer
  state. Chapter 3 solves this by sharding.
- **Activation memory at long context** — DDP does nothing for activations.
  Chapter 4 (sequence parallelism) and mod-107 (activation checkpointing)
  do.
- **Cross-node comm cost** — DDP's `2(N-1)/N * S` per-GPU is fine at 8 GPUs
  and painful at 8,192. Chapter 4 (HSDP, 3D-parallel) is the escape hatch.

## Summary

- DDP averages gradients between ranks with one all-reduce per bucket per
  step, overlapped with the tail of the backward pass.
- Ring all-reduce moves `2(N-1)/N * S` bytes per GPU total; this is the
  baseline every other strategy is measured against.
- DDP is the correct starting point for any model that fits on one GPU and
  the mental substrate for FSDP/ZeRO.
