# From FLOPs to GPU-Hours to Dollars

Chapter 1 gave you `C` in FLOPs. This chapter turns `C` into wall-clock
time, GPU-hours, and dollars on a specific cluster shape. The
arithmetic is not deep, but the constants matter and the sensitivity
analysis is what your director actually wants to see.

If mod-107 owns the *engineering* of MFU (kernels, precision, overlap),
this chapter owns the *economics* of MFU — every point you gain shows up
here as a dollar figure. Chapter 4 makes the ROI framing explicit.

## The pipeline

    C (FLOPs) → wall-clock hours → GPU-hours → dollars
              ↑                  ↑            ↑
        (aggregate FLOPs/s)  (× G GPUs)   (× $/GPU-hr)

Concretely, for a cluster of `G` GPUs at effective FLOPs/s `F_eff` per
GPU:

    T_wall = C / (G · F_eff)          # seconds
    T_wall_hours = T_wall / 3600
    GPU_hours = G · T_wall_hours
    Dollars = GPU_hours · price_per_gpu_hour

Each step has one constant. Let us pin them down.

## Peak FLOPs/s per GPU

Peak is what the datasheet promises. It is a hard upper bound; you will
never reach it, but you need the number to compute MFU.

| GPU     | Precision | Peak FLOPs/s (dense) | Source                                 |
|---------|-----------|----------------------|----------------------------------------|
| A100 SXM | BF16 / FP16 | ~312 TFLOPS         | NVIDIA A100 datasheet                  |
| A100 SXM | TF32       | ~156 TFLOPS         | NVIDIA A100 datasheet                  |
| H100 SXM | BF16 / FP16 | ~989 TFLOPS         | NVIDIA H100 SXM datasheet              |
| H100 SXM | FP8        | ~1979 TFLOPS        | NVIDIA H100 SXM datasheet (FP8 dense)  |
| H200 SXM | BF16 / FP16 | ~989 TFLOPS         | NVIDIA H200 datasheet                  |
| B200    | BF16 / FP16 | ~2250 TFLOPS        | NVIDIA Blackwell architecture material |

<!-- needs-research: cross-check B200 dense BF16 peak; NVIDIA publishes
several numbers depending on sparse / dense and MXFP4 / FP8 modes -->

Two subtleties every engineer confuses at least once:

- **NVIDIA quotes the "sparse" TFLOPS separately.** For training we
  care about dense, because the training loop does not exploit
  structured sparsity. If the datasheet says "989 / 1979 TFLOPS" for
  H100 BF16, the first is dense.
- **Peak assumes tensor cores.** If any part of your training loop
  falls back to CUDA cores (some layernorms, some elementwise ops,
  legacy attention paths), it will not hit peak. The MFU number
  absorbs this.

For the rest of this chapter we use H100 SXM BF16 at 989 TFLOPS as the
canonical GPU.

## Effective FLOPs/s: peak × MFU

**MFU (Model FLOPs Utilization)** is defined as:

    MFU = (achieved model FLOPs/s) / (peak dense FLOPs/s)

where "achieved model FLOPs/s" uses the `6 · N · D / T_step` accounting
from chapter 1. The PaLM paper (Chowdhery et al., 2022) formalized
this metric; mod-107 chapter 1 is where you engineer for it.

Published MFU numbers you should carry around:

| Run                          | MFU (BF16 unless noted)   | Source                                     |
|------------------------------|---------------------------|--------------------------------------------|
| PaLM 540B on TPU v4          | ~46.2% (aggregate)        | Chowdhery et al., 2022, PaLM paper         |
| Megatron-LM 3D parallel       | ~52% peak reported        | Narayanan et al., 2021, SC21               |
| MPT-7B on A100                | ~40–45% (typical)         | MosaicML MPT-7B blog                       |
| Llama 3 405B on H100         | ~38–41% (BF16 + FP8 mix)  | Grattafiori et al., 2024                   |

<!-- needs-research: pull exact Llama 3 published MFU / hardware-FLOPs
utilisation numbers from the model card; there are two related metrics
(MFU vs. HFU) reported in the paper -->

For budgeting, **45–55% MFU on a well-tuned dense FSDP2 stack on H100 is
the working assumption**. If your team is new to the framework, budget
40%. If you already have a battle-tested MFU story (mod-107 tells you
how to build one), budget 55%.

    F_eff = peak_FLOPs_per_gpu × MFU
    F_eff (H100 BF16 @ 50% MFU) = 989 TFLOPS × 0.50 ≈ 495 TFLOPS
                                = 4.95 × 10^14 FLOPs/s

Keep both `F_peak` and `F_eff` in your notes. The MFU is the leverage
point chapter 4 quantifies.

## Aggregate cluster FLOPs/s

For a homogeneous cluster of `G` GPUs, aggregate effective throughput is
just:

    F_cluster = G · F_eff

This is where the linear-scaling assumption enters. It is *not* trivially
true: as you scale `G`, comm-to-compute ratio grows and MFU tends to
sag. mod-101's chapter 5 collective-cost model is the precise story;
here we take a pragmatic shortcut and use a constant MFU with an
adjustment table:

| Cluster shape        | Realistic MFU on H100 BF16 (FSDP2 dense LM) |
|----------------------|---------------------------------------------|
| 8 GPUs (single node) | 55–60%                                      |
| 64 GPUs (8 nodes)    | 50–55%                                      |
| 512 GPUs (64 nodes)  | 45–55%                                      |
| 2048 GPUs            | 40–50%                                      |
| 8192 GPUs            | 35–45%                                      |

<!-- needs-research: cross-check the 8192-GPU MFU numbers against Llama
3, Megatron, and Databricks/Mosaic published runs -->

The sag is real. At 8192 GPUs with a well-engineered fabric and 4-D
parallelism, holding 40% is a serious achievement (this is roughly
where Llama 3's 405B training ran; Grattafiori et al., 2024, section 3).

## Wall-clock: `T = C / F_cluster`

Given `C` and `F_cluster`:

    T_wall_seconds = C / F_cluster
    T_wall_hours = T_wall_seconds / 3600
    T_wall_days = T_wall_hours / 24
    T_wall_weeks = T_wall_days / 7

You then translate that to GPU-hours:

    GPU_hours = G · T_wall_hours

Because `GPU_hours = C / F_eff`, note that **GPU-hours are independent
of `G`**. Doubling `G` halves wall-clock but not GPU-hours — assuming
MFU holds, which is the caveat above. When MFU sags with scale,
GPU-hours grow with cluster size, and this is the underlying reason
"bigger cluster = more expensive per token" at frontier scale.

## Worked example 1: 7B model, 200 B tokens, 512 × H100

The canonical mid-scale pretraining run.

**Inputs.**

- `N = 7 × 10^9`
- `D = 2 × 10^11` (200 B tokens — roughly Chinchilla for 7B; less than
  Llama 2's overtraining)
- `C = 6 · N · D = 6 × 7e9 × 2e11 = 8.4 × 10^21 FLOPs`
- Cluster: `G = 512` H100 SXM at 989 TFLOPS peak BF16
- MFU: 50%
- `F_eff = 989 TFLOPS × 0.5 ≈ 495 TFLOPS = 4.95 × 10^14 FLOPs/s`
- `F_cluster = 512 × 4.95e14 ≈ 2.53 × 10^17 FLOPs/s`

**Arithmetic.**

| Quantity                         | Value                                        |
|----------------------------------|----------------------------------------------|
| `T_wall_seconds = C / F_cluster` | `8.4e21 / 2.53e17 ≈ 3.32 × 10^4` s           |
| `T_wall_hours`                   | `~9.2 h`                                     |
| `T_wall_days`                    | `~0.38 d` (roughly 9 hours)                  |
| `GPU_hours = G · T_wall_hours`   | `512 × 9.2 ≈ 4,714 GPU-h`                    |
| Dollars @ $3/GPU-hr H100 on-demand | `~$14,140`                                 |
| Dollars @ $2/GPU-hr H100 reserved | `~$9,430`                                  |
| Dollars @ $1/GPU-hr H100 spot     | `~$4,710`                                  |

<!-- needs-research: verify current on-demand H100 SXM hourly for AWS
EC2 P5, GCE A3, Azure ND H100 v5, and CoreWeave. Round-order figures
used above are teaching numbers, not quotes -->

That is a *very* short run: 9 hours at Chinchilla-optimal 200B tokens.
This is why 7B-scale runs at Chinchilla are typically research
experiments, not products. Push `D` up to 2 T (Llama 2 style,
overtrained ~14×) and both wall-clock and dollars scale by 10×.

## Worked example 2: 70B model, 1.4 T tokens, 2048 × H100

The frontier-adjacent pretraining run — a Chinchilla-scaled 70B, so
directly comparable to Chinchilla's own 70B / 1.4 T recipe.

**Inputs.**

- `N = 7 × 10^10`
- `D = 1.4 × 10^12`
- `C = 6 · 7e10 · 1.4e12 = 5.88 × 10^23 FLOPs`
- Cluster: `G = 2048` H100 SXM
- MFU: 45% (larger cluster, more comm pressure)
- `F_eff = 989 TFLOPS × 0.45 ≈ 445 TFLOPS = 4.45 × 10^14 FLOPs/s`
- `F_cluster = 2048 × 4.45e14 ≈ 9.11 × 10^17 FLOPs/s`

**Arithmetic.**

| Quantity                         | Value                                        |
|----------------------------------|----------------------------------------------|
| `T_wall_seconds`                 | `5.88e23 / 9.11e17 ≈ 6.45 × 10^5` s          |
| `T_wall_hours`                   | `~179 h`                                     |
| `T_wall_days`                    | `~7.5 d`                                     |
| `GPU_hours`                      | `2048 × 179 ≈ 366,000 GPU-h`                 |
| Dollars @ $3/GPU-hr on-demand     | `~$1.10 M`                                  |
| Dollars @ $2/GPU-hr reserved      | `~$733 k`                                   |
| Dollars @ $1.50/GPU-hr large-vol reserved | `~$549 k`                          |

<!-- needs-research: verify large-volume reserved discount tiers for
H100 (typically negotiated, not published) -->

At 2048 H100s for a week, this is a ~$0.5–1.1M run depending on the
buying mix. Chapter 3 covers the mix; chapter 5 puts it in a feasibility
study.

## Worked example 3: 13B model, 300 B tokens, 128 × H100

The "mid-size fine-tune / small pretraining" shape most product teams
actually run.

- `N = 1.3 × 10^10`
- `D = 3 × 10^11`
- `C = 6 · 1.3e10 · 3e11 = 2.34 × 10^22 FLOPs`
- `G = 128`, MFU 55%, `F_eff = 544 TFLOPS`, `F_cluster ≈ 6.96 × 10^16`
- `T_wall = 2.34e22 / 6.96e16 ≈ 3.36 × 10^5 s ≈ 93 h ≈ 3.9 days`
- `GPU_hours = 128 × 93 ≈ 11,970`
- Dollars @ $2.50/GPU-hr reserved-ish ≈ **$30 k**

Chapter 5's worked feasibility study uses this shape.

## Sensitivity: what moves the number

The dominant sources of uncertainty in the dollar figure, ranked:

**1. MFU.** A 55% → 40% drop is a 27% wall-clock (and dollar) increase.

| MFU  | `F_eff` (H100 BF16) | Relative wall-clock |
|------|---------------------|---------------------|
| 60%  | 593 TFLOPS          | 0.83× baseline      |
| 55%  | 544 TFLOPS          | 0.91× baseline      |
| 50%  | 495 TFLOPS          | 1.00× baseline      |
| 45%  | 445 TFLOPS          | 1.11× baseline      |
| 40%  | 396 TFLOPS          | 1.25× baseline      |
| 35%  | 346 TFLOPS          | 1.43× baseline      |

Chapter 4 monetises this table.

**2. GPU generation.** Going from H100 to A100 is roughly a 3× wall-clock
bump for the same token budget in BF16 (A100 peak BF16 ~312 TFLOPS vs.
H100 SXM ~989 TFLOPS per NVIDIA datasheets, ~3.17× ratio), *before* MFU
differences. In practice you often gain a couple of MFU points on A100
because the fabric is less stressed, but the peak-ratio dominates. If
you can only get A100 capacity, take that budget and multiply GPU-hours
by ~3×; then compare dollar-per-GPU-hour.

**3. Precision.** Going from BF16 to FP8 on H100 doubles peak. The
realised speedup is smaller (typically 1.3–1.5× at end-to-end training
because not every op is FP8-safe), but it is real. FP8 also requires
Transformer Engine or an equivalent kernel path; mod-107 has the story.
For budgeting: if the framework supports FP8 on H100, budget for a
1.3–1.5× effective throughput bump, then verify.

**4. Attention long-context.** If `T = 32 k` or above, the `O(T²)`
attention FLOPs stop being negligible. Add them explicitly and expect
20–40% more compute than `6 · N · D` predicts.

**5. Overtraining ratio.** Doubling `D` doubles `C` doubles dollars.
This is a decision you make deliberately (chapter 6 covers the anchor
recipes), but it is the largest single lever on the dollar figure.

## Cluster-shape sizing: how to pick `G`

Given `C`, you have a one-parameter family: choose `G` and read off
wall-clock and dollars. The right `G` is set by the intersection of
three constraints:

- **Wall-clock deadline.** `T_wall_max = C / (G · F_eff)` gives a lower
  bound on `G`.
- **Capacity available.** You cannot book more GPUs than the cluster
  has. mod-104's quota model tells you what you can actually
  provision.
- **Comm-to-compute regime.** Past a point (typically hundreds of GPUs
  per TP × PP × DP factorisation), MFU sags and `F_cluster` stops
  scaling linearly with `G`. Chapter 5 of mod-101 gives you the model;
  the empirical sag is what shows up in the table above.

A pragmatic sizing procedure:

1. Compute the *minimum* `G` for your deadline at your budgeted MFU.
2. Round up to the nearest configuration your platform actually
   supports (8, 32, 64, 128, 512, 1024, 2048 GPUs).
3. Cross-check that MFU assumption at the rounded-up size against the
   sag table.
4. If wall-clock is comfortable, take the smallest cluster shape that
   fits — smaller clusters are cheaper per GPU-hour (less reserved
   commitment) and more elastic.
5. If wall-clock is tight, jump to the next shape and re-derive.

## What the arithmetic omits

At this altitude, the answer is *not* the dollar figure — it is the
range. Two omissions worth naming explicitly:

- **Failure and restart cost.** mod-106's availability budget shows up
  here as a multiplier: budget 10–20% headroom on `T_wall` for a
  well-run cluster, more if reliability is unproven. Frontier-scale runs
  (Llama 3, OPT-175B) publish failure rates that translate directly into
  GPU-hour tax.
- **Data pipeline and staging.** The GPU has to be fed. mod-103 owns
  the loader; mod-105 owns the storage tier. A poorly-staged corpus can
  eat 20% of wall-clock in the first epoch even with a perfectly-tuned
  training loop. Budget it explicitly; do not let it fall out of the
  MFU number.

Both are captured as line items in the chapter 5 feasibility study
template.

## Summary

- `T_wall = C / (G · F_eff)`. `F_eff = F_peak × MFU`. `GPU_hours = G ·
  T_wall_hours` — which, when MFU is constant, is independent of `G`
  and equal to `C / F_eff`.
- Canonical constants: H100 SXM BF16 peak is 989 TFLOPS (NVIDIA
  datasheet). A well-tuned FSDP2 dense LM on H100 sits at 45–55% MFU
  in the 64–512-GPU range and sags to 35–45% past a few thousand GPUs.
- A 7B / 200B-token Chinchilla run on 512 H100s at 50% MFU is ~4.7 k
  GPU-hours (~9 hours wall-clock, ~$10–15 k on-demand). A 70B / 1.4T-token
  Chinchilla run on 2048 H100s at 45% MFU is ~366 k GPU-hours (~7.5
  days wall-clock, ~$0.7–1.1 M).
- The dollar figure is dominated by (1) MFU, (2) GPU generation, (3)
  precision, (4) attention long-context if `T ≥ 32k`, and (5) how much
  you overtrain past Chinchilla. Chapter 3 handles the buying mix
  (reserved / spot / on-prem); chapter 4 monetises the MFU knob.
- Every published price in this chapter carries a
  `<!-- needs-research -->` marker because cloud pricing moves. Use
  round-order figures for teaching, current vendor quotes for
  procurement.
