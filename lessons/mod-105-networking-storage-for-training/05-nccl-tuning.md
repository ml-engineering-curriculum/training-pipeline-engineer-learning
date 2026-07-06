# Tuning NCCL for the Fabric

Every previous chapter in this module is about the fabric NCCL runs
on. This chapter is about the layer that decides *which* piece of
the fabric each byte crosses. NCCL is not magic — it makes decisions,
and every decision has knobs. Your job as a training platform engineer
is to know which knobs exist, which ones you own, and how to prove any
tuning change is a win with `nccl-tests`.

The two primary references you should keep open while reading this
chapter and while working exercise 2:

- **NCCL user guide and API reference.**
  https://docs.nvidia.com/deeplearning/nccl/
- **`nccl-tests` source and benchmarks.**
  https://github.com/NVIDIA/nccl-tests

## The tuning philosophy

Two rules that will save you dozens of GPU-hours:

1. **Do not tune blind.** Set `NCCL_DEBUG=INFO` on a small run first
   and read what NCCL actually did — algorithm, protocol, ring
   construction, PXN status. That output tells you what the auto-tuner
   picked and why. Half the "tuning" that shows up in tickets is
   forcing NCCL back to the choice it would have made on its own.
2. **The auto-tuner is usually right.** NCCL has algorithm selectors
   built on measured curves per topology per collective per payload.
   Override it only when (a) you can show, with `nccl-tests`, that the
   override is faster on your fabric, and (b) you can name the specific
   fabric feature that makes the auto-tuner wrong. "I read it on a
   forum" is not a reason.

## The knob taxonomy

NCCL exposes a lot of knobs. Group them into four families:

| Family | Example knobs | Owned by | When you touch it |
|--------|---------------|----------|-------------------|
| Algorithm & protocol | `NCCL_ALGO`, `NCCL_PROTO` | Application / platform | When measurement says the auto-tuner picked wrong for your payload / world size |
| Topology & routing | `NCCL_TOPO_FILE`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`, `NCCL_CROSS_NIC` | Platform | On fabric bring-up; on cluster changes |
| Rail alignment & PXN | `NCCL_PXN_DISABLE`, `NCCL_NET_GDR_LEVEL`, `NCCL_IB_QPS_PER_CONNECTION`, `NCCL_IB_SPLIT_DATA_ON_QPS` | Platform | On PXN debugging; on QP-count issues at scale |
| Diagnostics | `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`, `NCCL_DEBUG_FILE` | Everyone | Every incident |

We take these families in the same order.

## Algorithm selection: `NCCL_ALGO`

The algorithm is *how* NCCL structures the collective across ranks.
The values that matter for training:

- **`Ring`.** The bandwidth-optimal choice for large messages. Every
  rank forwards to its neighbor along a ring; the cost is
  `2(N-1)/N · n · β` for all-reduce (see mod-101 chapter 5).
- **`Tree`.** The latency-optimal choice for small messages.
  Recursive-doubling / double-binary-tree over ranks; `~log(N) · α`
  cost. Wins for small tensors (bucketed control messages,
  small-batch DDP with tiny per-bucket payload).
- **`CollnetChain` and `CollnetDirect`.** SHARP-based (in-network
  reduction on IB switches). Requires SHARP to be enabled on the
  fabric and the SHARP Aggregation Manager to be running. When
  available, wins on large all-reduces at high `N` because the
  reduction happens in the switches.
- **`NVLS` and `NVLSTree`.** NVLink SHARP variants that use NVSwitch3's
  in-switch reduction primitive. Wins on intra-node collectives at
  supported payload sizes.

The default is `Auto` — NCCL picks per collective per message size
based on its built-in tuner. `NCCL_ALGO` accepts a comma-separated
list: `NCCL_ALGO=Tree,Ring` means "allow Tree and Ring only".

When would you force it?

- **Ring for large-message all-gathers.** If `NCCL_DEBUG=INFO` shows
  NCCL picking `Tree` for a large all-gather and `nccl-tests` shows
  Ring is faster at your payload size, you have a case.
- **Disable Collnet during SHARP debugging.** If SHARP is misbehaving
  on the fabric, `NCCL_ALGO=Ring,Tree` gets you a baseline without
  SHARP so you can isolate the issue.
- **NVLS on a topology NCCL does not auto-detect.** Rare, but has
  happened on custom or non-reference builds.

Do not force `Ring` "just because it's simpler". At small message
sizes it will be measurably slower than Tree.

## Protocol selection: `NCCL_PROTO`

The protocol is *how* NCCL frames the data on the wire. Three values:

- **`Simple`.** Straight RDMA writes. Highest bandwidth for large
  messages.
- **`LL` (low-latency).** Sends 8-byte flags interleaved with 8 bytes
  of data so the receiver can poll without a synchronization step.
  Wins for very small messages at very low latency.
- **`LL128`.** 128-byte version of LL. Wins for small-to-medium
  messages; it is the default on many modern fabrics for that range.

NCCL picks per collective per size. Forcing `NCCL_PROTO=Simple`
during a debugging session is a common way to eliminate one axis of
variance while chasing a latency regression.

## The `algbw` / `busbw` reading exercise

Every `nccl-tests` output line looks like:

```
  size (B)    time (us)   algbw (GB/s)   busbw (GB/s)
   1048576         42         24.94          22.19
```

Two things to remember:

- **`algbw = n / T`.** The wire-rate the algorithm achieved for the
  input payload. Comparable across N because it does not correct for
  algorithm scaling.
- **`busbw = algbw × f(N)`.** Corrected for the algorithm-specific
  scaling factor (e.g., `2(N-1)/N` for ring all-reduce). Comparable
  against the underlying link bandwidth.

Rule of thumb: `busbw` at large payload should be within ~10% of your
nominal per-link bandwidth on a healthy fabric. If it is not, no
algorithm change will help — the fabric or the topology config is
broken.

## Topology and routing knobs

NCCL discovers your topology at initialization. Sometimes the
discovery is wrong (a virtualized environment, a custom PCIe layout,
a NIC pin the kernel does not expose). The knobs:

- **`NCCL_TOPO_FILE=<path>`.** Point at an XML file NCCL uses instead
  of auto-detection. Use `NCCL_TOPO_DUMP_FILE=/tmp/topo.xml` to have
  NCCL emit its auto-detected topology, edit it, and feed it back. See
  the NCCL user guide's topology chapter for the schema.
- **`NCCL_IB_HCA=<list>`.** Which HCAs NCCL should use. Format is
  `mlx5_0,mlx5_1,...` or `^mlx5_0` to exclude. Set this when the
  storage HCA and the compute HCA share a card and NCCL is picking the
  wrong port. On DGX SuperPOD compute nodes, exclude the storage HCA
  by name.
- **`NCCL_IB_GID_INDEX=<n>`.** RoCEv2-specific. The GID index selects
  the L3 address / VLAN. Get this wrong on RoCEv2 and NCCL will fail
  to bring up QPs; get it right by consulting `show_gids` output on
  the host.
- **`NCCL_SOCKET_IFNAME=<iface>`.** Which host interface NCCL uses for
  its out-of-band bootstrap. Set to the management interface;
  leaving it on the compute interface accidentally routes bootstrap
  traffic through the compute fabric.
- **`NCCL_CROSS_NIC=<0|1|2>`.** Whether a rank can send through a NIC
  other than its "own" NIC. Interacts with PXN below.

Every one of these knobs is a "set once, verify at bring-up" thing.
If you find yourself flipping them in a training loop, something is
wrong.

## Rail alignment and PXN

**PXN** (Peer eXchange Network) is NCCL's mechanism for
rail-alignment on multi-rail fabrics. The problem PXN solves:

- Suppose rank A on node 1 wants to send to rank B on node 2, but A
  is on rail 3 and B is on rail 5.
- Without PXN, that message goes through a leaf switch that connects
  rail 3 to rail 5. If most jobs are hitting the same leaf switch,
  you have congestion.
- With PXN, NCCL first hops the data over NVLink on node 1 from
  rank-A's GPU to the GPU that owns rail 5 (call it rank A'), and
  then sends A' → B on rail 5. That keeps the fabric traffic on the
  right rail and lets the NVLink hop absorb the rail change.

The relevant knobs:

- **`NCCL_PXN_DISABLE=1`.** Turn PXN off. Use this only when
  diagnosing whether PXN itself is the cause of a slowdown; production
  training runs on multi-rail fabrics should leave PXN on.
- **`NCCL_NET_GDR_LEVEL`.** Controls when GPUDirect RDMA is enabled
  vs. when NCCL falls back to staging through host memory. On DGX
  SuperPOD leave at default; it will infer.
- **`NCCL_IB_QPS_PER_CONNECTION` and `NCCL_IB_SPLIT_DATA_ON_QPS`.**
  Multi-QP configuration. At very high per-rank bandwidth (NDR+) or
  with certain workloads you may want more than one QP per rank pair
  to hit line rate; NCCL splits payload across them.

Verifying PXN is working:

- `NCCL_DEBUG=INFO` prints a line per rank pair that includes the
  PXN state and the intermediate rank if PXN routing is used.
- `NCCL_DEBUG_SUBSYS=NET` gives you the transport subsystem's
  detailed log.
- Compare `nccl-tests` at `NCCL_PXN_DISABLE=1` and without. The
  delta *is* the PXN benefit on your fabric.

Chapter 8 has the runbook for the "PXN misconfiguration" failure
mode.

## Subcommunicators and rail-aware collectives

The single biggest application-side lever for training-cluster
performance is picking subcommunicators (process groups) that align
with the fabric.

Two canonical examples:

### HSDP: replicate group vs. shard group

Under HSDP, you split your ranks into two axes:

- **Shard group.** The subgroup that does the FSDP all-gather /
  reduce-scatter. Typically one shard group per SU, so the
  shard-group all-gathers stay inside the SU and never cross the
  spine boundary.
- **Replicate group.** The subgroup that does the DDP-style
  all-reduce on gradients. Typically one rank per shard, so the
  replicate-group all-reduces run on same-rail cross-SU rails —
  entirely on the rail-optimized fat-tree.

The result: expensive collectives stay in the fast domain; cross-SU
traffic is on the slow-but-still-fast-enough replicate all-reduce.
This is a strategy choice you make in the application (see mod-101
chapter 4), but it *only* pays off if NCCL and the scheduler
cooperate. Chapter 8 discusses the failure mode where a scheduler
places your gang across SUs in a way that breaks this.

### Tensor-parallel intra-node group

TP-8 groups always fit inside one DGX. The corresponding process
group's collectives use NVLink / NVSwitch — never the RDMA fabric.
Verify this by inspecting `NCCL_DEBUG=INFO` on a TP-only collective;
you should see NVLS or NVLink transports, never IB.

## The two feedback loops

Everything above is theory. In practice you have two feedback loops
you use every day:

### Loop 1: `nccl-tests`

`nccl-tests` is the ground truth for "what does the fabric do at
this payload size in this collective at this world size". Every
tuning change should be validated with a sweep:

```
mpirun -np <N> \
  -x NCCL_DEBUG=INFO \
  ./all_reduce_perf -b 1K -e 1G -f 2 -g 1
```

- `-b 1K -e 1G -f 2` sweeps payload from 1 KiB to 1 GiB doubling.
- `-g 1` is one GPU per MPI rank.
- `-x NCCL_DEBUG=INFO` prints the algorithm/protocol NCCL picked.

Read `busbw` at both ends of the range. At the small end you are
measuring latency; at the large end you are measuring bandwidth. A
tuning change that improves one and regresses the other is a
trade-off, not a win.

### Loop 2: `NCCL_DEBUG=INFO` on a real training run

For a real training job, running with `NCCL_DEBUG=INFO` (and
`NCCL_DEBUG_FILE=/path/to/nccl.log` so it does not spam stderr) gives
you:

- The topology NCCL discovered.
- The rings and trees it constructed per process group.
- The transport (IB / NVLink / P2P) it chose per rank pair.
- The PXN routing and intermediate hops.

Read that log at least once per new cluster / new NCCL version /
new NCCL knob. It is the fastest way to catch a discovery bug before
you lose GPU-hours to it.

The `NCCL_DEBUG_SUBSYS` variable narrows what NCCL logs to a specific
subsystem — `NCCL_DEBUG_SUBSYS=NET,GRAPH,COLL` is a good starting
combination when diagnosing a specific pattern.

## An honest checklist for a new cluster

Every time you bring up a new fabric, run this sequence:

1. **Confirm the topology NCCL discovered.** `NCCL_DEBUG=INFO` on a
   trivial 2-node collective. Verify HCA names, NUMA affinity, GPU
   affinity.
2. **Baseline `nccl-tests` all-reduce, all-gather, reduce-scatter,
   alltoall.** Payload sweep from 1 KiB to 1 GiB. Save the output
   as your reference; every subsequent regression compares against
   it.
3. **PXN check on a cross-rail collective.** Compare `busbw` with
   and without `NCCL_PXN_DISABLE=1` at a payload size that
   dominates your training step. Confirm PXN is a win.
4. **SHARP check (IB only).** Run all-reduce with and without
   `NCCL_ALGO=CollnetDirect`. If SHARP is up and provisioned, the
   Collnet run should be faster at large payloads and higher world
   sizes.
5. **`ib_write_bw` between representative node pairs.** Chapter 4
   introduced this. Confirms the fabric is delivering line rate at
   the verbs layer before NCCL is even involved.
6. **Record the baselines in the platform's runbook.** Chapter 8's
   failure-mode runbook builds on these numbers.

Every one of those six steps is a step in exercise 1 or exercise 2.

## Summary

- NCCL is a decision-making layer between your process groups and the
  fabric. Every decision has a knob; the auto-tuner is usually right
  and you override with numbers, not intuition.
- The four knob families are algorithm/protocol, topology/routing,
  rail alignment/PXN, and diagnostics. Own each family before you
  touch it in production.
- `nccl-tests` is the ground-truth benchmark. Every tuning change is
  validated with a payload sweep; every new cluster starts with a
  baseline that later regressions are compared against.
- Subcommunicators (HSDP shard vs. replicate groups, TP intra-node
  groups) are the highest-impact application-side lever. NCCL's job
  is to route them onto the right fabric; your job is to build them
  so NCCL can.
- `NCCL_DEBUG=INFO` is the first tool you reach for on any incident.
  Chapter 8's runbook opens every entry with reading it.
