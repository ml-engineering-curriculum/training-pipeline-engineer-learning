# GPUDirect RDMA, GPUDirect Storage, and the Storage-to-GPU Path

The last three chapters were about the compute fabric — the RDMA
network the collectives run on. This chapter is about the second
fabric on the SuperPOD: the storage side. The question it answers is
"how does a training batch actually get from the parallel filesystem
or the object store into GPU HBM without paying the host memory bus
tax on every byte", and "how much of my per-step budget can the
loader afford to spend on that path before it starves the model".

Two primary references:

- **NVIDIA GPUDirect Storage documentation.**
  https://docs.nvidia.com/gpudirect-storage/
- **The `cuFile` API reference.**
  https://docs.nvidia.com/cuda/cufile-api/index.html

## Two GPUDirect families

The name "GPUDirect" is used for a family of features. The two that
matter to training platform engineers:

- **GPUDirect RDMA (GDR).** Peer-to-peer transfer between a GPU and a
  network HCA over PCIe, bypassing host memory entirely. When NCCL
  says "I did an IB write from GPU 0 to remote GPU 3", GDR is what
  makes that not go through the host CPU's memory controller. GDR
  requires an HCA that supports PeerDirect and a driver stack (MOFED
  on the Mellanox / NVIDIA side, `nvidia-peermem`) that exposes it.
  See the NVIDIA GPUDirect RDMA documentation.
- **GPUDirect Storage (GDS).** Peer-to-peer transfer between a
  storage NIC (or local NVMe) and a GPU. Same principle: bypass host
  DRAM, keep the data on the PCIe path. GDS is exposed by the
  `nvidia-fs` kernel module plus the `cuFile` user-space API, and it
  is supported by parallel filesystems (Lustre, WEKA, DDN AI400X, IBM
  Storage Scale) and by local NVMe (via specific block drivers).

There are older members of the GPUDirect family — GPUDirect P2P for
intra-node GPU-GPU DMA, GPUDirect Video for capture cards — but for
training the two above are what matters.

## Why bypassing host memory matters

The naive read path for a training batch is:

```
  Storage NIC/NVMe → host DRAM (kernel page cache) → host DRAM (userspace buffer)
                   → host DRAM (CUDA host-pinned buffer) → GPU HBM
```

Three copies through host DRAM, one PCIe DMA into the GPU. On a
DGX-class node with 8 GPUs each pulling multi-GB/s of dataset, that
saturates the host memory bandwidth and steals cycles from the
data-preprocessing that has to happen alongside training. It also
double-writes and double-reads the same bytes, wasting DRAM bandwidth.

With GDS:

```
  Storage NIC/NVMe → GPU HBM
```

One DMA, no host DRAM copies. On a well-provisioned NVMe or a
GDS-capable parallel filesystem client, this frees ~10s of GB/s of
host memory bandwidth per node and lets the CPU concentrate on
tokenization / augmentation instead of on being a memcpy engine.

## The GDS software stack

The layers, in the order they call each other on a `cuFileRead`:

1. **Application: `cuFile` API.** `cuFileHandleRegister`,
   `cuFileRead`, `cuFileWrite`, `cuFileBufRegister`. The public API
   for GDS. Documented under docs.nvidia.com/cuda/cufile-api/.
2. **`cuFile` user-space library.** Translates `cuFileRead` into
   ioctls against `nvidia-fs`.
3. **`nvidia-fs` kernel module.** Extends the VFS with an interface
   the underlying filesystem client can use to route reads directly
   to GPU memory instead of to a host page cache.
4. **Filesystem client.** Lustre client, WEKA client, DDN client, or
   the local NVMe block driver. Each vendor's client integrates with
   `nvidia-fs`; the specific supported versions are in the vendor's
   docs.
5. **Transport.** For remote filesystems, this is RDMA over the
   storage fabric (a separate IB or Ethernet fabric from the compute
   fabric — see chapter 2). For local NVMe, it is the PCIe NVMe
   protocol.
6. **Hardware.** Either the storage HCA or the NVMe SSD, DMAing
   directly into GPU HBM.

Every step is verifiable:

- `gdscheck.py` (ships with the `nvidia-fs` package) validates the
  stack end-to-end.
- `cuFileGetVersion` from the app confirms the library version.
- `dmesg | grep nvidia-fs` confirms the module loaded.
- The filesystem's own health checks (e.g., `lctl` for Lustre,
  `weka status` for WEKA) confirm the client is GDS-mode.

## When GDS actually helps

GDS is not free — it involves ecosystem coupling (filesystem +
driver + firmware versions), and it can be complex to bring up.
Reach for it when at least one of:

- **Per-node loader bandwidth is ≥ several GB/s.** At small dataset
  sizes the host memory bandwidth is not the bottleneck; a
  well-tuned page-cache-based path is fine.
- **Host CPU cycles are contested.** Tokenization on-the-fly,
  augmentation heavy enough to saturate a CPU core, or
  loader/tokenizer threads competing with the trainer for RAM
  bandwidth — GDS frees the host to do CPU work rather than being a
  memcpy engine.
- **You are checkpointing at scale.** Distributed checkpoint writes
  (mod-106 covers this) benefit substantially from GDS's ability to
  DMA parameter tensors from GPU HBM straight to the parallel
  filesystem without a host-buffer round trip.
- **You are doing GPU-side data decoding.** DALI-style pipelines
  that decode / decompress on the GPU want the compressed bytes
  already in GPU HBM. GDS is the fast path for that.

If none of the above applies (a small model with a small dataset
that already fits in the page cache), you can skip GDS and use a
plain buffered read path. Do not adopt technology you do not need.

## The storage-to-GPU throughput budget

The per-step budget calculation is the reason this chapter exists.
For a training step:

```
  T_step = max(T_compute, T_comm, T_loader, T_checkpoint)
                    ^-- if overlap is working
```

We introduced `T_compute` and `T_comm` in mod-101 chapter 5. Now we
add `T_loader`.

### The loader-side arithmetic

Per step, the loader must produce a batch of `B` samples times
`bytes_per_sample` bytes.

- **Text pretraining, packed tokens.** Say sequence length `T`
  tokens × 4 bytes per int32 token = `4 · T` bytes per sequence,
  times batch size `B` per rank = `4 · B · T` bytes per rank per
  step. For BF16 embedding lookups on the GPU, this is the entire
  read.
- **Vision / multimodal, decoded to GPU.** Larger — raw pixels or
  compressed frames + audio. Order of magnitude MBs per sample.
- **Vision / multimodal, decoded on CPU.** Bytes per sample after
  decoding; the compressed source is smaller but you pay CPU cycles
  for decoding.

The step-time budget for loader is roughly:

```
  T_loader = (B · bytes_per_sample) / (per_rank_read_bandwidth)
```

If `T_loader > T_compute`, the trainer will stall waiting on batches.
The loader-side of mod-103 (chapters 5 and 6) is what you use to make
sure that never happens; this chapter is what you use to compute
`per_rank_read_bandwidth` so the accounting closes.

### Where the per-rank read bandwidth comes from

Working up from the hardware:

- **Local NVMe.** Modern PCIe Gen4/Gen5 NVMe delivers ~5–14 GB/s per
  drive; per-node aggregate is typically ~30–100 GB/s depending on
  the number of drives. GDS lets multiple GPUs on the same node
  share that bandwidth without cross-copying via host DRAM.
- **Parallel filesystem client (Lustre / WEKA).** Per-client
  bandwidth is bounded by the storage NIC. On DGX SuperPOD the
  storage NICs are typically 2 × 400 Gb/s NDR → up to ~100 GB/s per
  node aggregate on the storage fabric. Real workloads see less
  than the nominal because of protocol overhead, cache misses on the
  server, and metadata operations.
- **Object store (S3 / GCS).** Per-node bandwidth is bounded by the
  in-region internet gateway; typical numbers are single-digit GB/s
  per node without a cache, higher through a proxy or the vendor's
  Mountpoint / Storage Gateway client.

For any given cluster, measure with `fio` (for local NVMe) and the
filesystem-specific benchmark tools (`ior` for Lustre, `weka
benchmark` for WEKA, or `s3-benchmark` / `warp` for S3). Do not
trust spec-sheet numbers.

### Composing the budget: a worked example

Assume the following on a DGX H100 node:

- 8 GPUs, each pulling a batch of `B_local = 4` sequences × `T = 8192`
  tokens × 4 bytes = 128 KiB per sequence, so ~512 KiB per rank per
  step of raw token I/O.
- Per-step compute is ~500 ms at MFU 0.5 on H100.
- Storage fabric delivers ~50 GB/s per client node aggregate at
  large sequential reads.

Loader budget:
- Per node per step: `8 · 512 KiB = 4 MiB` of raw tokens.
- At 50 GB/s that is ~80 µs of wire time — trivially inside 500 ms.

Text pretraining is compute-bound and the loader is a non-issue.
Contrast with a vision-language workload where each sample is a
decoded image of ~4 MiB × 8 GPUs × B_local = ~128 MiB per node per
step. That would be ~2.5 ms of storage-fabric time per step —
still well inside compute budget, but non-trivial and demanding of
prefetch. For high-resolution video the per-sample bytes go up
another order of magnitude and GDS becomes a hard requirement.

The point of the exercise is not the specific arithmetic; it is the
habit of writing the arithmetic down before choosing a storage tier.
Exercise 4 has you do exactly this for a workload you specify.

## Checkpoint I/O as the other write case

Checkpoints are the other big storage-side event. The write budget:

- **Checkpoint size.** For a `P`-parameter model in BF16 with a Adam
  optimizer state (FP32 master, two FP32 moments), roughly
  `12 · P` bytes for the sharded model + optimizer state, distributed
  across DP ranks. For a 70B model that is ~840 GB in aggregate.
- **Write cadence.** Every `k` steps, aligned with your
  fault-tolerance budget (mod-106).
- **Write time budget.** You want a checkpoint to land in seconds,
  not minutes, so the training loop can resume without a long stall.

Distributed checkpoint frameworks (`torch.distributed.checkpoint`,
DeepSpeed / Megatron savers) shard the write across all DP ranks in
parallel; the storage fabric aggregates them. GDS on the write path
is a substantial win here because the parameter and optimizer
tensors are already in HBM and there is no reason to stage them
through host DRAM.

mod-106 owns the checkpointing semantics; this chapter's contribution
is the fabric-side of the arithmetic.

## When to skip GDS

Honest counter-cases:

- **The workload is compute-bound and the loader has slack.** If
  `T_loader < 0.2 · T_compute` on a plain buffered read, you have no
  performance reason to adopt GDS. The engineering cost of the
  ecosystem coupling exceeds the benefit.
- **The cluster does not have a GDS-supported filesystem.** Some
  filesystems have partial or no GDS support; check the current
  compatibility matrix on the NVIDIA GDS docs before designing around
  it.
- **You are on a hyperscaler VM.** Public-cloud VM environments
  historically had limited or no GDS support; verify against the
  current provider docs.
- **You are still in the prototype phase.** Get the loader working
  and the model training first. Adopt GDS when the profiling
  actually shows a host-DRAM bottleneck.

## Summary

- GDR bypasses host DRAM on the GPU-to-network path. It is what
  makes NCCL's IB writes fast. On a properly configured SuperPOD it
  is on by default.
- GDS bypasses host DRAM on the storage-to-GPU path. It requires
  `nvidia-fs` + a supported filesystem/NVMe client + the `cuFile`
  API in the loader. It is worth adopting when the loader is a
  material fraction of the step budget or when host DRAM
  bandwidth is otherwise saturated.
- The per-step storage budget is `T_loader = payload / bandwidth`.
  Every workload should have that arithmetic written down before a
  storage tier is chosen; exercise 4 makes you do it.
- Distributed checkpoint write is the other big storage-side event.
  GDS on the write path is a substantial win for
  `torch.distributed.checkpoint`-style sharded writes; mod-106 owns
  the semantics.
- Do not adopt GDS you do not need. The engineering cost of the
  driver/firmware coupling is real; the fastest path is to measure
  the loader's fraction of step time first.
