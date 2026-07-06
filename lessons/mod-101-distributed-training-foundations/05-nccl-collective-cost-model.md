# The NCCL Collective Cost Model

The previous chapter picked a *strategy*. This chapter turns that strategy
into a number you can defend in a design review: how many bytes cross the
fabric per step, how long that takes on a given cluster, and where the
comm-to-compute ratio is going to bottleneck you. If you cannot do this
back-of-envelope calculation for your own strategy, you do not understand
the strategy.

## Motivation

Every parallelism decision — TP-8 vs TP-4, FSDP vs HSDP, pipeline stage
size — is ultimately a choice about how much wire time you are buying with
your memory-and-compute budget. NCCL's collectives have well-understood
scaling; the same `α + n·β` cost model that MPI has taught for decades
applies unchanged.

## The α + β model

The classical latency / bandwidth model of a message-passing collective:

- `α` — the fixed latency of one collective operation on this fabric,
  in seconds. Includes NIC + switch + software stack per-message overhead.
- `β` — the inverse bandwidth in seconds per byte. `1/β` is the achievable
  bandwidth in bytes/second (on a single link).
- For a payload of `n` bytes on `N` ranks, an algorithm's cost is
  `T(n, N) = f(N) · α + g(N) · n · β`.

The functions `f` and `g` depend on the algorithm. NCCL publishes
selectors that pick per collective; the important ones are:

- **Ring all-reduce (bandwidth-optimal for large `n`)**:
  - Time ≈ `2(N-1) · α + 2(N-1)/N · n · β`.
  - Latency term grows with `N`; bandwidth term is asymptotically `2n · β`.

- **Recursive-doubling / tree all-reduce (latency-optimal for small `n`)**:
  - Time ≈ `⌈log₂ N⌉ · (2α + 2n · β)`.
  - Latency term grows as `log N` instead of `N`. Bandwidth term is worse
    than ring, but for small `n` (small tensors, many ranks) the log
    latency wins.

- **Double binary tree** (NCCL's default at scale) — combines two trees
  that each move half the payload; latency is `~2 log N · α` with
  bandwidth close to ring's asymptote for large messages. See Sanders,
  Speck & Träff (2009) and NCCL's own docs for the algorithm.

- **Ring reduce-scatter / all-gather** — each is essentially half of the
  ring all-reduce: `(N-1) · α + (N-1)/N · n · β`.

**Rule of thumb**: for tensors larger than a few megabytes, everything is
bandwidth-bound and you use `2(N-1)/N · n · β` for all-reduce.
For very small tensors and many ranks (small gradient buckets, coordination
messages), latency dominates and ring is the wrong pick.

You can force NCCL's choice with `NCCL_ALGO=Ring|Tree|CollnetChain|CollnetDirect`
and its protocol with `NCCL_PROTO=Simple|LL|LL128`. Doing so is a debugging
tool, not a production knob; the auto-tuner is usually right.

## Topology and effective bandwidth

The "achievable bandwidth" per rank you feed into `β` is not the NIC's
line rate. It is:

- **Intra-node NVLink** — on H100 with NVSwitch, hundreds of GB/s per
  GPU pair, with all-to-all bisection bandwidth in the ~900 GB/s range.
  Consult the DGX/HGX H100 spec sheet or `nvidia-smi topo -m` on your
  hardware for the exact number; NVLink4 nominally delivers 900 GB/s
  bidirectional per GPU on H100.
- **Inter-node HDR IB** — 200 Gb/s per port ≈ 25 GB/s per link. On a
  typical fat-tree fabric with `k` GPUs per node and one NIC per GPU,
  bisection bandwidth scales with the number of NICs, not one link
  aggregated.
- **Inter-node NDR IB** — 400 Gb/s per port ≈ 50 GB/s per link.

Two "topology gotchas" that catch teams every year:

1. **NUMA / PCIe root-complex crossings.** If GPU-to-NIC pinning is
   wrong, a collective can transit the host memory controller instead of
   going straight through GPUDirect RDMA. Effective bandwidth drops
   multiplicatively. `nvidia-smi topo -m` and `NCCL_DEBUG=INFO`'s
   printed topology are your diagnostic tools. mod-105 goes into depth on
   this; here it is enough to know that "effective β" and "nominal β" are
   often different and you have to *measure*.
2. **PXN and rail alignment.** On multi-rail IB fabrics, NCCL's PXN
   (Peer eXchange Network) feature pipes traffic through the "right" NIC
   for each destination. Misconfiguration or an unrecognized topology
   makes NCCL fall back to a single NIC per node and cuts your effective
   inter-node bandwidth by the number of rails.

## The comm-to-compute ratio

The number that actually matters for scaling efficiency is
`T_comm / T_compute` per step. If it is much less than 1, you are
compute-bound and adding more GPUs will help. If it is at or above 1, you
are comm-bound and adding more GPUs will hurt (and may even slow the
job down absolute).

You compute it per strategy.

### DDP

Per step:

- Compute time: `T_compute ≈ 6 · P · tokens / (peak_FLOPS · MFU)`
  (the 6·P·tokens factor is the "6·N·D" FLOP count from Kaplan et al.,
  2020, "Scaling Laws for Neural Language Models", and refined in
  Chinchilla). MFU is the achieved fraction of peak, e.g. 0.4–0.55 for a
  well-tuned run.
- Comm time: `T_comm ≈ 2(N-1)/N · S · β` where `S` is the parameter size
  in bytes.
- Overlap: DDP bucket-overlaps comm with backward, so the observable
  step cost is roughly `max(T_compute, T_comm)` in the good case.

For a 1B-parameter BF16 model on 8 GPUs with ~30 GB/s effective all-reduce
bandwidth per rank, the gradient tensor is `1e9 * 2 bytes = 2 GB` and
`T_comm ≈ 2 * 7/8 * 2 GB / 30 GB/s ≈ 116 ms`. Compare against your
observed step compute; if the step is a few hundred milliseconds, you are
still comp-bound and the strategy scales.

### FSDP2 / ZeRO-3

Per step:

- Extra all-gather on backward → `T_comm ≈ 3(N-1)/N · S · β`.
- No `optimizer.step()` cross-rank comm.
- Overlap depends on the framework hiding the all-gather of layer `i+1`
  behind the compute of layer `i`. Both PyTorch FSDP2 and DeepSpeed do
  this by default (see `forward_prefetch` and equivalents).

### Tensor parallel (Megatron-style)

Per step, per transformer block:

- Two all-reduces on activations (forward), two on activation-gradients
  (backward). Payload per collective ≈ `B · T · H` (batch · seq · hidden)
  in the activation dtype.
- `T_comm_per_block ≈ 4 · 2(TP-1)/TP · (B · T · H) · β_intra`.
- On NVLink `β_intra` is very small, but this fires **per layer**, so it
  dominates once you cross the node boundary. That is why TP > 8 is
  usually a bad idea.

### Pipeline parallel

Per step:

- Point-to-point sends per stage boundary per micro-batch. Payload is one
  activation tensor of size `~B_micro · T · H`.
- The dominant cost is not the P2P — it is the **bubble**. For `S` stages
  and `M` micro-batches with the classical GPipe schedule, the bubble
  fraction is `(S - 1) / (M + S - 1)`. Interleaved-1F1B shrinks this by a
  factor of the number of interleaves (Narayanan et al., 2021).

## Sanity numbers you should know cold

Order-of-magnitude figures every training-platform engineer should have
internalized. Every one of these should be verified against your own
`nccl-tests` numbers on your own cluster — treat these as the shape of the
answer, not the exact answer.

| Link                     | Bandwidth per GPU (order of magnitude) |
|--------------------------|----------------------------------------|
| HBM3 (H100)              | ~3 TB/s                                |
| NVLink4 pair (H100)      | ~900 GB/s bidirectional                |
| NVSwitch fabric (DGX-H100) | ~900 GB/s bisection per GPU          |
| HDR InfiniBand (per port)| ~25 GB/s                               |
| NDR InfiniBand (per port)| ~50 GB/s                               |
| PCIe Gen5 x16            | ~64 GB/s                               |

Use those as the α+β constants until you have measured your own.

## `nccl-tests`: measuring the model against reality

`nccl-tests` (NVIDIA's benchmark suite) exercises each collective at a
sweep of payload sizes and reports:

- **algbw** — algorithmic bandwidth, `n / T`. Comparable across different
  values of `N` because it does not correct for the algorithm's `f(N)`.
- **busbw** — bus bandwidth, `n / T · f(N)`. Meant to represent the
  underlying link-level bandwidth the collective is achieving.

You should run `all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf`,
and `alltoall_perf` on every new cluster, at message sizes that match your
gradient / activation tensors. If your DDP or FSDP job runs at
substantially lower effective bandwidth than nccl-tests reports, the
scheduling / overlap is broken, not the fabric.

## What to do with the model

The cost model is a **decision tool**, not a benchmark. Its job is to tell
you, before you burn GPU-hours, whether:

- The comm cost of a proposed strategy fits in the compute time.
- Adding a TP or PP axis will help or hurt.
- The comm dip you observed on a real run is "as expected" or "something
  is broken".

Exercise 3 will have you build this model for a realistic model + cluster
shape and defend the choice against a benchmark.

## Summary

- Every collective has a cost `T = f(N) · α + g(N) · n · β`. For
  ring-based collectives on large tensors, the bandwidth term dominates
  and is `~2n · β` for all-reduce.
- The strategy you picked in chapter 4 lands each collective on a
  specific fabric link with a specific effective β. The comm-to-compute
  ratio is the single number that tells you whether the strategy scales.
- Sanity-check every model against `nccl-tests`. The delta between
  predicted and observed is where the interesting engineering lives.
