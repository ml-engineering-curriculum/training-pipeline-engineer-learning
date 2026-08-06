# exercise-04: GPUDirect Storage Throughput Budget

**Estimated effort:** 3 hours

## Objective

Write the storage-to-GPU throughput budget for a real training step —
in numbers you can defend — and then decide, in numbers, whether
GPUDirect Storage (GDS) is worth adopting for that workload. The
deliverable is a one-page "storage budget memo" that names the
per-step bytes, the available bandwidth on the intended tier, the
resulting slack (or shortage), and a ship / no-ship recommendation for
GDS.

The exercise is the counterweight to exercise 2's NCCL focus:
collectives are one input to `T_step`; the loader is the other. If
you cannot write the loader-side of the budget, you cannot argue
about whether the fabric-side has room to move.

## Prerequisites

- Chapter 6 of this module (GDS software stack, per-step budget).
- Chapter 7 of this module for the parallel-filesystem context you
  will apply the budget against (you will pick a target tier in the
  report; chapter 7 is the vocabulary).
- Familiarity with the mod-103 loader model (data loader, prefetch
  queue, `T_loader` vs. `T_compute`). If you have not done mod-103,
  the summary in chapter 6 of this module is enough context to work
  the exercise.
- Access to at least one node with GPU + a storage tier to measure
  against — either local NVMe, a parallel-filesystem client, or a
  cloud-attached FS (FSx, filestore, blob). If you cannot measure at
  all, you can still do the arithmetic and label numbers as
  vendor-datasheet-only, but the report is stronger with real
  measurements.
- The following tools available on the test node: `fio` (for local
  NVMe), the vendor benchmark for your FS (`ior` for Lustre, `weka
  benchmark` for WEKA), `gdsio` (ships with the `nvidia-fs` package)
  if GDS is installed, and `gdscheck.py`.

## Problem statement

You are the training-platform lead. A training team is proposing a
workload and asking whether the platform's current storage tier can
sustain it, and whether the extra engineering cost of adopting GDS is
warranted. Your job is to answer both, in a memo that a director
could sign off on.

Pick one of the following workloads as the subject of your memo, or
substitute one your team actually runs (and characterize it in the
same terms):

- **Workload D: text pretraining, packed tokens.** 70B dense
  decoder, sequence length 8192 tokens, 8 GPUs per node, per-GPU
  micro-batch 4 sequences, ~500 ms per step compute at MFU ~0.5
  on H100. Dataset stored as WebDataset tars on a parallel FS or
  as MDS shards.
- **Workload E: vision-language pretraining.** Multimodal model,
  images decoded on the GPU via DALI, per-GPU batch 32 samples,
  each sample ~4 MiB compressed, ~300 ms per step compute.
- **Workload F: video pretraining.** Short-clip video model, batch
  4 clips per GPU, each clip ~64 MiB pre-decode, per-GPU compute
  ~800 ms per step.
- **Workload G: distributed-checkpoint save.** Not a training-step
  workload but the write-side budget: 70B model + Adam state,
  distributed checkpoint every 1000 steps, target write time ≤ 10
  seconds so the run does not stall.

The workload's numbers do not have to be exact; they need to be
specific and internally consistent. If you use a public model's exact
dims (e.g., Llama 3 70B), cite the source. Every subsequent
calculation refers back to these inputs.

## Requirements

Deliver a single memo (`storage-budget-<workload>.md`) with the
following sections.

### 1. Workload inputs

Table (or short list) with:

- Model size, precision, framework.
- Sequences / images / clips per GPU per step.
- Bytes per sample as observed on the storage tier (raw, not
  post-decode where the two differ).
- Steps per hour and hours per full run.
- Compute time per step (`T_compute`) as a defended estimate. Cite
  MFU and the FLOP count basis (Kaplan `6 · N · D`, or measured).
- Checkpoint size (for workload G) or checkpoint cadence (for D–F if
  applicable).

State the assumption behind each number in one line. This is the
input to every downstream calculation.

### 2. Bytes-per-step arithmetic

Compute:

- **Per-rank per-step bytes read** for the loader. Show the
  arithmetic: `batch × bytes_per_sample`.
- **Per-node per-step bytes read** (8× for a DGX-class node).
- **Bytes per second** at your step time: `bytes_per_step /
  T_compute` (assuming the loader is on the critical path).
- **Bytes per hour, per full run.** So the storage vendor can size
  capacity, not just throughput.

For workload G, do the same for checkpoint writes: per-rank shard
size, per-node write bytes, and required aggregate write bandwidth
to hit the target write time.

### 3. Available bandwidth on the intended tier

Measure — do not assume from vendor spec sheets alone — the per-node
achievable throughput on the storage tier you plan to use. Pick one
of the four cases below (or as many as apply to your platform) and
run the measurement:

- **Local NVMe.** `fio --name=seqread --rw=read --bs=1M --iodepth=32
  --numjobs=8 --size=64G --direct=1`. Publish per-drive and
  aggregate throughput and IOPS.
- **Lustre client.** `ior -w -r -t 4m -b 8g -F -a POSIX` (write and
  read; adjust block and transfer size to workload). Publish
  per-client sequential read throughput.
- **WEKA client.** `weka benchmark` per the vendor's docs; capture
  per-client read throughput at large sequential reads.
- **FSx for Lustre / cloud FS.** The FS-native benchmark (or `ior`
  against the mount). Publish per-client throughput and the
  per-TiB throughput tier configured on the FS.

For each measurement:

- The command you ran (verbatim).
- The result (aggregate throughput, IOPS if relevant).
- The nominal cap (per-drive line rate, per-NIC port bandwidth,
  per-TiB tier for FSx).
- The efficiency ratio (measured / nominal). Chapter 1's "effective
  vs. nominal" framing again.

If you have GDS installed, additionally run:

- `gdscheck.py` to verify the stack.
- `gdsio -f <mounted_file> -d 0 -s 4G -x 4 -r` (per the current
  `nvidia-fs` docs) to measure GDS-mode throughput to GPU HBM.

Publish the GDS-mode number alongside the non-GDS number from the
same tier and highlight the delta.

### 4. Slack calculation

For each workload, compute:

- `T_loader = per_node_bytes_per_step / per_node_available_bw`.
- The ratio `T_loader / T_compute`. This is the fraction of the step
  time the loader would consume if it were on the critical path.
- The overlap headroom: how many samples deep the prefetch queue
  must be to hide `T_loader` behind `T_compute`. If `T_loader >
  T_compute`, no prefetch depth will save you — call this out.

For workload G, compute the write time analogously:

- `T_checkpoint = checkpoint_size_per_node / write_bandwidth`.
- Compare to the target write-time budget from section 1.

State the verdict in one line per workload: "loader has X ms of
slack per step", or "loader consumes X% of compute budget", or
"loader exceeds compute budget by X — must fix".

### 5. GDS decision

Given sections 2–4, argue in one paragraph:

- **Does this workload need GDS today?** Chapter 6's four "when GDS
  helps" bullets are the reference. Point at the specific one that
  makes the case (per-node bandwidth ≥ several GB/s, host CPU
  contested, GPU-side decode, checkpoint scale). If none apply,
  state that and recommend deferral.
- **What is the incremental cost?** Filesystem support, driver
  versions, `cuFile` in the loader code, ongoing compatibility
  vigilance. Chapter 6's software stack section is the reference.
- **What is the measured or expected win?** Numbers from section 3
  or a citation to vendor / community benchmarks for a similar
  workload. If the win is < 10% on end-to-end step time, say so —
  a "GDS is not worth it here" recommendation is fine and often
  correct at prototype scale.
- **The recommendation.** Adopt, defer, or reject. State the
  operational owner if adopted (who upgrades the driver stack, who
  monitors compatibility, who owns the loader-side `cuFile` code).

### 6. Failure-mode watch-items

Chapter 6 warned about ecosystem coupling (driver / firmware /
filesystem-client versions). Chapter 8 covers fabric-side failure
modes. Name the two or three things that would degrade the storage
budget if they happened, and how you would detect them:

- Filesystem-client version drift breaking GDS.
- Storage-fabric congestion during a checkpoint burst.
- A metadata storm from an untuned DataLoader worker count.
- Any hot spot specific to your chosen storage tier.

Each item is one bullet, one detection signal.

## Starter guidance

- The arithmetic is the point. Do not skip it because "it is
  obvious" — writing the numbers down is what makes the memo
  defensible.
- Round aggressively. `~4 GB/s`, `~500 ms`, `~50 GB/s` per node.
  Precision beyond one significant figure implies confidence you
  probably do not have.
- For text workloads (D), the loader is almost never the
  bottleneck; expect to write a "GDS is not worth it" recommendation
  and defend it in numbers. For vision (E), it is often close; for
  video (F), it is often decisive.
- Do not confuse `bytes_per_sample` at rest with `bytes_per_sample`
  post-decode. For E/F, the storage tier is sized on the raw
  bytes; the GPU DMAs the raw bytes; the decoded tensor is a
  downstream cost that lives in HBM.
- For workload G (checkpoint), do not forget that distributed
  checkpoint frameworks shard the write across DP ranks. The
  aggregate write bandwidth is what closes the budget, not the
  per-node one.
- If you cannot measure GDS on your cluster (because it is not
  installed or supported), you can still write sections 1–4 and
  section 6, and in section 5 present the decision as
  "measurement-blocked, request access to a GDS-capable cluster
  before signing off". That is a valid outcome.

## Acceptance criteria

- Section 1 has numeric inputs with cited sources for any public
  model dims used.
- Section 2 shows the bytes-per-step arithmetic explicitly, not just
  the result.
- Section 3 has at least one measured tier with the command,
  result, and efficiency ratio. Any unmeasured tier is clearly
  labelled as datasheet-only.
- Section 4 has an explicit `T_loader / T_compute` number and a
  verdict.
- Section 5 has an explicit adopt / defer / reject recommendation
  with a named operational owner if adopting.
- Section 6 has at least two watch-items with detection signals.
- No invented benchmark numbers. Every measurement is reproducible.

## Stretch goals

- Repeat the memo for a second workload at the opposite end of the
  loader-intensity spectrum (say D and F). Compare the
  recommendations; they should differ. If they don't, the analysis
  is wrong somewhere.
- Add a "next-gen tier" section: what would the memo say if the
  storage tier were upgraded to the vendor's next generation (e.g.,
  Gen5 NVMe, or a WEKA cluster with 2× the server count)? Which
  workloads would flip from "GDS-optional" to "GDS-mandatory" or
  vice versa?
- Integrate with mod-103: pick the specific loader implementation
  (WebDataset, Ray Data, MDS) the target training team uses and
  audit whether its current `num_workers`, prefetch depth, and
  bucket strategy match your budget. A budget that assumes a
  prefetch depth of 8 but the loader has depth 2 is not a budget
  that will hold.
- Turn the memo into a per-team template your platform publishes:
  training teams fill in section 1 with their workload, the
  platform team fills in sections 3–6 with the current tier's
  data. The template is the deliverable.
