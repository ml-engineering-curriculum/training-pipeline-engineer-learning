# Fault Tolerance as a Scale Problem

The first five modules of this track taught you how to make a training
job go *fast*. This module is about making it not stop. At the scale
these clusters run at, "runs to completion without operator
intervention" stops being the default and becomes a designed property
of the platform. If you skip that design work, your GPUs still spin,
but the tokens they produce do not accumulate — a failed step is a
step you paid for and got nothing back for.

Everything in this module — DCP, torchrun elasticity, incident
classification, straggler detection, goodput SLOs — is a tool for
buying back the tokens that failures otherwise consume. Chapter 1
frames why the problem is worth this much design.

## Why individual reliability does not scale

At a single-node training scale you can assume the box is up. A DGX
system's mean time between hardware faults is measured in months; if
your job runs for a week, the odds are on your side.

At cluster scale the arithmetic inverts. If a single GPU has an
annualized failure probability of `p`, a job spanning `G` GPUs has a
compound probability of at least one failure per unit time that grows
roughly linearly in `G`. Two consequences fall out:

- **Multi-thousand-GPU jobs almost never run to completion.** With
  ~1 000 GPUs in the collective and per-GPU MTBF in the years, the
  effective time-to-first-failure of the *job* is measured in hours to
  days. This is not a hypothetical: every published large-model
  training logbook — OPT-175B, BLOOM, GPT-NeoX-20B, Llama 3 — reports
  it directly.
- **Recovery cost has to be designed against, not tolerated.** If a
  failure costs an hour of GPU time to recover from (checkpoint load,
  rendezvous, warm-up) and failures come every four hours, you have
  quietly ceded a quarter of your capacity. You will not get it back
  by buying more hardware; you get it back by shortening recovery.

The rest of this chapter names the two numbers that make that
arithmetic concrete and introduces the SLO shape chapter 8 turns into
a production budget.

## Primary sources you should read

Before anything in this module, read these logbooks in full. Every
subsequent chapter is grounded in one of them:

- **Zhang, S., et al. (2022). "OPT: Open Pre-trained Transformer
  Language Models."** The paper is at
  https://arxiv.org/abs/2205.01068. Meta AI also released the
  **OPT-175B chronicles** as a dated logbook alongside the
  code — https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf.
  This is the canonical account of what a real large-model training
  run's failure stream looks like day-by-day.
- **Grattafiori, A., et al. (2024). "The Llama 3 Herd of Models."** The
  tech report at https://arxiv.org/abs/2407.21783 documents, in
  section 3.3.2, a 54-day 16 384-H100 pretraining run and its 466
  interruptions (419 unexpected). The interruption taxonomy in the
  paper is the closest thing the community has to a published incident
  rate for a modern frontier run.
- **Chowdhery, A., et al. (2023). "PaLM: Scaling Language Modeling with
  Pathways."** https://arxiv.org/abs/2204.02311 — introduced the term
  "goodput" for this class of work (see section 5) and defined it as
  "useful throughput relative to peak throughput", i.e. how much of
  the machine's compute actually contributed to gradient updates.
- **Jouppi, N., et al. (2023). "TPU v4: An Optically Reconfigurable
  Supercomputer for Machine Learning."** — chapter 8's "goodput"
  discussion started here for TPU pods; the reasoning transfers
  directly to GPU pods.

Have those three papers (OPT chronicles, Llama 3, PaLM) at hand for
chapter 7. Every recovery pattern in this module was authored to fix
a failure mode one of them documents.

## The two numbers that dominate everything

Two numbers set the shape of your fault-tolerance design. Pick them
early and revisit them at every major cluster change.

### 1. Mean time between failures (MTBF) at your job's world size

For a `G`-GPU job on hardware with per-GPU annualized failure rate
`λ`, the job-level failure rate is approximately `G · λ` under the
"any component failure kills the run" assumption. That assumption is
correct for classic synchronous training without elasticity; the
whole point of the elastic + DCP design later in the module is to
break it.

The Llama 3 report (arXiv 2407.21783, §3.3.2) gives the direct
number for a 16 384-H100 run: 466 interruptions over 54 days, of
which 419 were unexpected — roughly one unexpected interruption
every ~3 hours of wall clock. That is a *published* frontier
number; treat it as your worst-case reference until your own runs
prove different.

### 2. Recovery time from a fault

Recovery time is the sum of five things:

1. **Detection.** How long from "the run stops making progress" to
   "the platform knows". If your loop is `wait for NCCL timeout`,
   this is 10 to 30 minutes of dead GPU time depending on how you
   set `torch.distributed`'s `timeout` argument.
2. **Diagnosis.** How long to decide whether to fence a node,
   restart, or escalate. Chapters 5 and 6 are about shortening this.
3. **Rendezvous.** How long to re-form the process group with a
   possibly-different world size. Chapter 4 is about this.
4. **Checkpoint load.** How long to read the last DCP checkpoint
   from storage. Chapter 2 is about the cost model, chapter 3 about
   how DCP makes this scale-out.
5. **Warm-up.** JIT recompiles, NCCL topology re-discovery, LR
   warm-up if you jumped enough steps to matter. Usually the
   shortest of the five.

Multiply this recovery time by the number of interruptions per day
and you get the fraction of the cluster's capacity you burn on
recovery instead of training. That fraction is what chapter 8's SLO
budget quantifies.

## Goodput vs. throughput vs. uptime

Three terms that get mixed up on-call, defined once so you can use
them precisely:

- **Throughput** is tokens (or samples) per second when a step is
  running.
- **Uptime** is the fraction of wall-clock time the job is not
  crashed. A stalled-but-not-crashed job (rendezvous hang, straggler
  ballooning step time) has 100% uptime and 0 useful work.
- **Goodput** is the tokens per wall-clock hour that actually
  contribute to a persisted checkpoint you would resume from. It is
  the product of throughput, uptime, and "did we roll back to a
  checkpoint before losing this work" — restart-and-retry work does
  not count.

Every design decision in this module tries to raise goodput without
touching throughput. That is why "faster checkpoint" and "shorter
detection" both count: they reduce the divisor in `useful work / wall
clock`.

PaLM's paper (arXiv 2204.02311, §5) is the reference for the term;
the TPU v4 paper (Jouppi et al. 2023) uses the same framing for its
optical-pod interconnect. Chapter 8 formalizes the SLO built on it.

## The three failure classes this module treats separately

Not every failure has the same shape, and confusing them is the most
common way to mis-invest engineering effort. Chapter 5 is the full
taxonomy; here is the three-way split that motivates the rest of the
module:

- **Fail-stop.** A rank crashes. The process is gone, the collective
  hangs, the platform notices. The fix is rendezvous + DCP resume.
  Chapters 3 and 4 cover this class end-to-end.
- **Fail-slow.** A rank is still up but is running slower than its
  peers (thermal throttle, ECC-remapped HBM page, network flap that
  auto-recovers to a lower rate). The collective still completes —
  it just drags. Chapter 6 covers detection and quarantine for this
  class.
- **Silent corruption.** A rank is up, is running at rate, but its
  numeric output is wrong. Nobody notices at the framework layer;
  the model diverges hours or days later. Chapter 6 covers detection.

Fail-stop is the *easy* case in a mature platform. The Llama 3 report
puts 78% of its interruption count in GPU / host / component
failures — mostly fail-stop after crash. Fail-slow and silent
corruption are what a good platform team invests disproportionately
in, because their downside (silently ruined training) is much larger
than the raw incident count would suggest.

## What this module deliberately does not cover

- **Distributed-training semantics** (DDP, FSDP2, 3D-parallel) —
  owned by mod-101. This module *uses* the process-group abstraction
  to place checkpoint I/O and rendezvous.
- **Framework internals** (Megatron, DeepSpeed, torchtitan) — owned by
  mod-102. torchtitan's checkpoint helpers wrap DCP; DeepSpeed has
  its own checkpoint format; this module treats them as consumers
  of the same underlying storage tier.
- **The fabric itself** — owned by mod-105. NCCL timeouts are covered
  as a *symptom*; the diagnosis of *why* NCCL timed out (a NIC drop,
  a PXN misconfiguration) lives in mod-105 chapter 8.
- **Cluster orchestration and scheduling** — owned by mod-104. This
  module runs on top of a scheduler that has already granted a gang;
  what happens when the gang shrinks mid-run is chapter 4's problem.
- **Observability and metrics collection plumbing** — owned by
  mod-108. This module *emits* the counters (per-step time histograms,
  ECC delta counters, goodput ratio) that mod-108 turns into
  dashboards.
- **Cost accounting on the recovered-vs-lost token axis** — owned by
  mod-109. Chapter 8's SLO is the input; mod-109 turns it into a
  dollar number.

## Summary

- At multi-thousand-GPU scale, fail-stop failures are guaranteed on
  the timescale of a run; individual reliability does not compose to
  cluster reliability.
- The two design numbers are per-job MTBF (roughly `G · λ`) and
  recovery time (detect + diagnose + rendezvous + load + warm-up).
  Their product sets the goodput ceiling.
- "Goodput" is tokens per wall-clock hour that made it into a
  persisted checkpoint. Uptime and throughput can both look healthy
  while goodput collapses; this module targets goodput directly.
- Three failure classes deserve separate design: fail-stop (chapters
  3–4), fail-slow (chapter 6), and silent corruption (chapter 6).
  Their incident *rates* differ; their per-incident *cost* differs
  even more.
- The primary sources (OPT chronicles, Llama 3 report, PaLM paper)
  are the empirical ground truth chapter 7 draws on. Read them once
  before touching the rest of the module.
