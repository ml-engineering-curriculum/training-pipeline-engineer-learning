# From FLOPs to GPU-Hours

Chapter 1 ended with three numbers: `N`, `D`, and `C = 6 · N · D`.
This chapter turns the last one into GPU-hours. The formula is
short — divide `C` by the effective per-GPU throughput — but every
one of the three factors on the bottom is a place where teams
routinely mis-estimate by 30–50%. This chapter is the arithmetic
plus the traps.

Downstream, chapter 3 turns GPU-hours into dollars and wall-clock;
chapter 5 shows how much each MFU point is worth in each unit. But
the GPU-hours number is the primary sizing figure — it is what
gets quoted in every capacity ask, every RFC, and every retro. Get
it right first.

## The core identity

For a cluster of `G` accelerators, each with dense peak `P` FLOPs
at the training precision, sustaining an end-to-end **MFU** of
`μ`, the wall-clock time to burn a compute budget `C` is

```
T_seconds = C / (G · P · μ)
```

and the GPU-hours consumed are

```
GPU_hours = G · T_seconds / 3600  =  C / (P · μ · 3600)
```

Note the collapse: **`G` cancels out of the GPU-hours calculation.**
Doubling the fleet halves the wall-clock but leaves the GPU-hours
budget unchanged (in a first-order model — chapter 3 revisits the
scaling losses). This is a useful sanity check: a proposal that
predicts fewer GPU-hours for a bigger cluster is wrong.

The GPU-hours arithmetic thus depends on exactly two numbers you
have to defend: `P` (the vendor peak) and `μ` (the sustained MFU
you will actually get). Both are places to get burned.

## `P`: use the dense vendor peak at the training precision

The rules from mod-107 chapter 1 apply. The `P` in the denominator
is:

- **The vendor's published peak** for the accelerator, at the
  precision the matmuls actually run in during training. BF16 is
  the current default; FP8 is Hopper/Blackwell-and-later. Choose
  the precision that matches your recipe, not a wishlist.
- **The dense number**, not the "with sparsity" number. Training
  runs do not use 2:4 structured sparsity; the sparsity-doubled
  figure is inference-marketing.

Reference peaks to keep in a table:

- NVIDIA A100 (SXM4, no sparsity): 312 TFLOP/s BF16/FP16. Source:
  NVIDIA A100 Tensor Core GPU datasheet.
- NVIDIA H100 (SXM5, no sparsity): 989 TFLOP/s BF16/FP16;
  1979 TFLOP/s FP8. Source: NVIDIA H100 Tensor Core GPU
  datasheet.
- NVIDIA H200 (SXM5): identical math peak to H100; HBM3e delivers
  more bandwidth, which helps memory-bound kernels but does not
  change `P`. Source: NVIDIA H200 datasheet.
- NVIDIA B200 (Blackwell): consult the current Blackwell
  datasheet; the FP4 / FP8 / FP16 numbers moved substantially
  from Hopper.
- AMD MI300X: 1307 TFLOP/s BF16 dense per AMD's Instinct MI300X
  product page (verify against the datasheet you install with).
- Google TPU v5p: consult the current Google TPU v5p spec page;
  the BF16 peak is published there.

Cite the URL in the feasibility study. A dollar figure attached
to an unsourced `P` is a dollar figure that survives one
reviewer.

## `μ`: use the MFU you will *sustain*, not peak

This is where most cost estimates go wrong. The instantaneous
peak MFU an experienced team hits during a benchmark run is
routinely 20–40% higher than the *end-to-end* MFU averaged over
the full training run, once losses from restarts, straggler
recovery, framework warm-up, evaluation passes, and checkpoint
writes are folded in.

Chapter 5 turns the difference between "peak MFU during a good
100-step window" and "sustained MFU across the run" into a
dollar figure. For now, two numbers to keep separate:

- **Nominal MFU (`μ_nom`)** — what you measure during
  instrumented steady-state windows. This is the mod-107
  chapter 1 number. For dense LLMs on Hopper BF16 with FA
  and reasonable overlap, community values sit in the 40–55%
  band; the Llama 3 paper reports its observed BF16 MFU in
  section 3.4.
- **Sustained MFU (`μ_sus`)** — nominal MFU multiplied by
  goodput (the fraction of wall-clock that is productive
  training). This is the number that ends up in the GPU-hours
  denominator.

The relationship, taken from mod-106 chapter 8's goodput SLO
formulation:

```
μ_sus = μ_nom · goodput
```

where `goodput = productive_step_time / total_wall_clock_time`.
Goodput on a well-run large H100 pretraining sits in the
`0.85–0.95` band; the Llama 3 paper (§3.3.2) documents a
concrete example. On a spot-instance run with a lot of
preemption, goodput can drop below `0.60`; chapter 4 quantifies
this.

The rule of thumb for a first-pass estimate:

- **A well-tuned dense LLM on H100 BF16, reserved capacity, mature
  team**: `μ_nom ≈ 0.45`, `goodput ≈ 0.90`, `μ_sus ≈ 0.40`.
- **The same recipe, first pretraining run for a new team**:
  `μ_nom ≈ 0.35`, `goodput ≈ 0.80`, `μ_sus ≈ 0.28`.
- **Same recipe, aggressive spot-heavy tier**: `μ_sus ≈ 0.20–0.25`
  depending on how frequently the run gets preempted.

These are anchors, not standards. Do the measurement (mod-107
exercise 1) on your own setup and use *your* number in *your*
budget. But when you have no measurement — writing a budget for
a not-yet-existing cluster — quote the anchor and the source.

## The parallelism strategy multiplier

The `G · P · μ_sus` product implicitly assumes every GPU in the
cluster is doing useful matmul work every step. Real parallelism
strategies impose a scaling loss that shows up as a decrease in
`μ_sus` at larger world sizes — the exposed communication
fraction grows with the collective's participant count.

A useful accounting trick: split `μ_sus` into a small-scale
efficiency `μ_ref` (measured at, say, 8 or 16 GPUs) and a
**scaling efficiency `η(G)`** that decays with the world size.

```
μ_sus(G) = μ_ref · η(G)
```

`η(G)` is the fraction of small-scale MFU that survives the
scale-up. Public anchors:

- **Well-tuned FSDP2 dense-LLM run on Hopper**: `η(1024) ≈ 0.90–
  0.95`, `η(4096) ≈ 0.80–0.90`, `η(16384) ≈ 0.70–0.85`
  depending on fabric and overlap discipline. Grattafiori et al.
  (2024, § 3) report specific numbers for Llama 3 at 16 K GPUs.
- **3D-parallel Megatron run**: similar overall shape, with the
  tensor-parallel size (TP=8) staying inside a node and pipeline
  and data-parallel splits going across. See Shoeybi et al.
  (2019, arXiv 1909.08053) and its follow-ons for the
  measurement protocol.

Two implications:

- **Do not extrapolate small-cluster MFU straight to large-
  cluster GPU-hours.** A 60% MFU measured at 8 H100s does not
  give you 60% at 4096. Multiply by `η(G)` before using.
- **Bigger cluster = shorter wall-clock but fewer GPU-hours only
  if `η` stays close to 1.** If doubling `G` cuts `η` by 30%,
  you paid a GPU-hours *tax* to buy wall-clock. Chapter 3 turns
  this into a dollars-vs-schedule trade-off.

## Attention correction for long context

The `C = 6 · N · D` formula (chapter 1) is the dominant term for
short-context runs. At long context (`S ≥ 8 K` or so with a
30 B+ model), the attention quadratic term becomes a substantial
fraction of the per-step FLOPs and *must be added to the
numerator*.

The mod-107 chapter 1 formula:

```
F_step ≈ 6 · N · B · S  +  12 · L · H · d_head · B · S²
```

The compute budget `C_run` for the full training run then
integrates this over `D` tokens (`D = B · S · num_steps`):

```
C_run ≈ 6 · N · D  +  12 · L · H · d_head · D · S
       = 6 · N · D · (1 + (2 · L · H · d_head · S) / N)
```

The correction factor `(2 · L · H · d_head · S) / N` grows
linearly in `S`. At `S = 2K` for a 30B model it is negligible
(a few percent); at `S = 32K` it can be 20–40% of the base
budget. Compute it explicitly for your recipe before quoting a
`C` for a long-context run.

Long-context runs also often use FlashAttention v3's IO-aware
scheduling to hide the extra bandwidth cost. That does not
change the FLOP count; it changes the MFU you can achieve on
that FLOP count. Do both corrections separately.

## Recompute (HFU) is a separate axis

If your recipe uses full activation checkpointing, the GPU
actually executes roughly `1.33 · C` FLOPs — the extra `0.33`
is the extra forward pass per backward step. Two ways to handle
this in the GPU-hours calculation, both defensible:

- **MFU-based (numerator = `C`, denominator includes the extra
  work implicitly via a lower MFU).** This is the version that
  matches the mod-107 chapter 1 MFU-vs-HFU distinction.
- **HFU-based (numerator = `C_executed ≈ 1.33 · C`, denominator
  is the same).** This charges the training run for the
  recompute directly.

The two calculations should give the same total GPU-hours if
you are consistent. What breaks people is mixing them — using
the MFU number in the denominator but the HFU-scaled FLOPs in
the numerator. Pick one convention and stick to it in the
feasibility study.

## A worked example, end-to-end

The 34 B Chinchilla-optimal recipe from chapter 1:

- `N = 34 · 10^9`, `D = 680 · 10^9`, `C = 1.39 · 10^23` FLOPs.
- Cluster: H100 SXM5, dense BF16, `P = 989 · 10^12` FLOP/s.
- Assumed `μ_sus = 0.40` (mature team, reserved capacity).

GPU-hours:

```
GPU_hours = C / (P · μ_sus · 3600)
          = 1.39e23 / (989e12 · 0.40 · 3600)
          ≈ 97 600 H100-hours
```

Sanity check the wall-clock at three cluster shapes:

- 128 H100s: `97 600 / 128 ≈ 763 hours ≈ 31.8 days`.
- 512 H100s: `97 600 / 512 ≈ 191 hours ≈ 8.0 days` (if `η(512)
  ≈ 1`; add a scaling-loss correction if not).
- 2048 H100s: `97 600 / 2048 ≈ 48 hours ≈ 2.0 days` (if `η(2048)
  ≈ 1`; realistically apply `η(2048) ≈ 0.85` and it lengthens to
  ~2.4 days and adds ~15% to the GPU-hours total via the lower
  `μ_sus`).

The inference-optimal recipe (`D = 4.76 T`) scales linearly:

```
GPU_hours ≈ 97 600 · (4.76 / 0.68) ≈ 683 000 H100-hours
```

Both numbers are the input to chapter 3's dollar arithmetic. The
inference-optimal recipe is 7× the training cost — which is why
the trade-off against the inference-lifetime FLOPs in chapter 5
matters.

## GPU-hours vs. GPU-days: pick one unit and stick with it

Community publications quote in whichever unit is convenient.
Some anchors:

- Meta's Llama 3 paper (§ 3): 39.3 M H100-hours for the 70 B
  pretraining run.
- Meta's Llama 2 paper (Touvron et al. 2023, arXiv 2307.09288,
  table 2): 1.7 M / 184 K / 3.31 M A100-hours for the 7 B /
  13 B / 70 B variants.
- MPT-7B (MosaicML blog): about 9.5 days of continuous training
  on 440 A100-40GB GPUs (multiply out for the GPU-hours figure).

Convention for this module: **GPU-hours** as the primary unit,
GPU-days only for wall-clock comparisons. GPU-hours is the unit
your capacity team quotes leases in.

Also always name the accelerator: an H100-hour is roughly 3.2×
the compute of an A100-hour at BF16, and roughly 6.3× at FP8.
Mixing accelerator generations without conversion is a common
mistake in cross-team cost discussions.

## Common estimation errors and their sizes

- **Using peak MFU instead of sustained MFU.** Undersizes the
  budget by 20–40%. Fix by computing `goodput` from mod-106
  chapter 8's SLO or from a comparable published run.
- **Using the sparsity-doubled peak `P`.** Undersizes the budget
  by ~50%. Fix by citing the datasheet dense number.
- **Ignoring the scaling-efficiency term at large cluster
  sizes.** Undersizes the budget by 10–30% for runs at 1 K+
  GPUs. Fix by measuring `η(G)` at the target world size,
  or by quoting the Llama-3-report numbers as a placeholder
  and flagging the uncertainty.
- **Forgetting the attention quadratic at long context.** Under-
  sizes the budget by 20–40% for `S ≥ 32 K` on 30 B+ models. Fix
  by computing the correction factor explicitly.
- **Silent unit confusion between GPU-hours and GPU-days.**
  Undersizes / oversizes by 24×. Fix by naming the unit in
  every sentence.
- **Mixing MFU and HFU conventions in numerator and denominator.**
  Undersizes or oversizes by ~33% depending on direction. Fix
  by picking one convention and putting it in the feasibility
  study's front matter.

Each of these is a single-integer-multiplier error. Multiple of
them compound. A feasibility study that has *two* of these
errors going the same direction is off by ~2× and gets the
wrong capacity decision.

## Summary

- `GPU_hours = C / (P · μ_sus · 3600)`. This is the primary
  sizing formula for the rest of the module.
- `P` is the vendor **dense** peak at the **training** precision;
  cite the datasheet URL.
- `μ_sus` is `μ_nom · goodput`. Community anchors: 0.40 for a
  mature H100 BF16 run, 0.28 for a first run, 0.20–0.25 on a
  spot-heavy tier.
- Scaling efficiency `η(G)` decays with world size; measure it
  at the target scale or quote a published anchor and flag the
  uncertainty.
- Add the attention quadratic to `C` for `S ≥ 8 K`; add the HFU
  correction if you full-activation-checkpoint (or fold it into
  MFU — pick a convention).
- The `G` in the numerator cancels — GPU-hours does not depend
  on cluster size to first order. Wall-clock does. Chapter 3
  puts a dollar figure on both.
- Always name the accelerator (H100-hour, A100-hour). Cross-
  generation quoting without conversion is a common failure.
