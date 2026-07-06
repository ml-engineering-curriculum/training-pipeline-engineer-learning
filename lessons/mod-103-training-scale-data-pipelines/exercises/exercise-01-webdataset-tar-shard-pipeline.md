# exercise-01: WebDataset Tar-Shard Pipeline That Saturates 8 GPUs

**Estimated effort:** 4 hours

## Objective

Build a WebDataset tar-shard pipeline that keeps an 8-GPU training
node fed for a synthetic training loop whose per-step compute time
you know exactly. Prove — with measurements, not vibes — that the
pipeline sustains the target sample throughput without stalling the
GPUs, and identify the specific knob (`num_workers`,
`prefetch_factor`, `shuffle` depth, shard size) that is the
bottleneck.

The deliverable is a short report showing the measured step time,
GPU idle time, and worker utilization at three configurations, plus
the config you would ship to production.

## Prerequisites

- Chapters 1 and 2 of this module.
- A machine with 8 GPUs (or a comfortable proxy — 4 or 2 GPUs work
  for the pattern, but pick a target you can hit).
- Object-store bucket you can read and write from the compute
  node's VPC / project. S3, GCS, or an S3-compatible endpoint (MinIO
  running locally is fine).
- `webdataset` (recent version), `torch`, and an S3 client library
  (`boto3` or `aiobotocore`) installed in the training env.

## Problem statement

You are on the training-platform team. A researcher hands you a
1-billion-sample multimodal training corpus at 10 KB per sample
(~10 TB). They plan to run a Llama-3-class training loop on it. On
one 8×H100 node the per-step compute time (from mod-101 style
profiling) is 0.30 s at a batch of 8 · 4 · 2048 samples equivalent.

Your job is to design and prove out a WebDataset pipeline that
saturates that node's compute — GPU idle time on the
data-side ≤ 5 % — and to defend the shard size, worker count,
prefetch factor, and shuffle depth you chose.

## Requirements

1. **Generate a synthetic corpus.**
   - 10 000 shards, each a tar file with ~2 000 samples of
     ~10 KB each (`~20 MB` shards — deliberately too small at
     first).
   - Each sample is a fixed-format record: `<index>.bin`
     (10 KB of random bytes) and `<index>.json` (a tiny label).
   - Upload to your object-store bucket under
     `s3://your-bucket/exercise01/tiny/`.
2. **Build a WebDataset reader pipeline.**
   - Iterates the shard URLs deterministically.
   - Uses `split_by_node` + `split_by_worker`.
   - Passes each sample through a **synthetic training step**
     that `.sleep()`s for 0.30 s to model compute (or actually
     runs a small `matmul` at the right cost — up to you).
3. **Measure three configurations.**
   - **Config A** — the "researcher default": `shard = 20 MB`,
     `num_workers = 4`, `prefetch_factor = 2`, `shuffle(1000)`.
   - **Config B** — larger shards: rebuild the corpus at
     `shard ≈ 500 MB` (roughly 50 K samples/shard);
     `num_workers = 8`, `prefetch_factor = 4`, `shuffle(2000)`.
   - **Config C** — your production candidate. Justify every
     knob change relative to B.
4. **Report** per config:
   - Median and p95 step time over 500 steps (after a 100-step
     warm-up).
   - GPU idle fraction — the wall time not spent inside the
     compute step, divided by total wall time.
   - Per-worker CPU utilization during steady state.
   - The first stall you see and what triggered it (network,
     decode, shuffle drain).
5. **Ship a config.** In a one-page write-up, propose the shard
   size, worker count, prefetch depth, and shuffle depth you would
   put into production, and explain from your data why.

## Starter guidance

- Do **not** skip the "researcher default" run. It exists to
  reproduce the stall pattern. You need to have seen a stall
  before you can claim you eliminated one.
- Instrument stall detection cheaply: log a wall-clock timestamp
  around every fetch call and around every training step. Anything
  where fetch wall time exceeds step wall time by more than a
  small margin is a stall.
- The synthetic step's `sleep()` should include a `torch.cuda`
  synchronization or you will not see the stall pattern the real
  training loop sees.
- Object-store egress on a shared VPC has meaningful tail
  latency; run each config for at least 5 minutes of steady state
  to see it.
- Set `NCCL_DEBUG=OFF` — it's not the subject of this exercise
  and adds noise.

## Acceptance criteria

- Config A produces measurable GPU idle time (> 20 %) or
  measurable step-time inflation. This is the failure baseline.
- Config B produces meaningfully less GPU idle time than A. Your
  report identifies which knob change was responsible.
- Config C hits ≤ 5 % GPU idle time, sustained across 500 steps,
  with p95 step time within 10 % of median.
- The report cites either the `webdataset` README's guidance on
  shard sizing or the S3 performance guidelines for at least one
  design decision, and includes the actual bandwidth numbers you
  observed from your object store.
- All three configs run against the same synthetic training step
  and the same synthetic corpus; only the knobs differ.

## Stretch goals

- Rerun Config C with `shardshuffle=False` and quantify the
  degradation from removing the shard shuffle.
- Force one worker to sleep for an extra 200 ms every 100 shards
  (simulating an S3 tail-latency spike) and check whether the
  pipeline's prefetch depth is enough to hide it.
- Compare `webdataset.WebLoader` vs. wrapping the pipeline in
  `torch.utils.data.DataLoader` directly and measure the
  difference.
- Repeat Config C with `num_workers = 16` and confirm the
  point of diminishing returns (or worse — over-fanout hurting
  performance).
