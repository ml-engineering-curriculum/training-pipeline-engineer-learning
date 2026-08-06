# Lessons from OPT-175B and Llama 3: Turning Logbooks Into Runbooks

Chapter 5 gave you an incident taxonomy; chapter 6 gave you the
detectors. This chapter grounds both in the two most detailed
public accounts of frontier-scale pretraining: the **OPT-175B
chronicles** and the **Llama 3 herd of models** technical report.
Everything before this chapter is theory backed by APIs; this
chapter is the empirical evidence that the theory is worth building
against.

The point of this chapter is not "these papers are interesting" (they
are). The point is: **every failure they describe is a runbook entry
you should own before your own team hits it**. Read this chapter with
your platform's incident-response document open beside it. Every
lesson below is a "does your runbook cover this? if not, why not?"
question.

## The two primary sources

- **OPT-175B chronicles / logbook** (Meta AI, 2022). Published as a
  dated logbook in the `metaseq` repository at
  https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf.
  The accompanying paper is Zhang, S., et al. (2022), "OPT: Open
  Pre-trained Transformer Language Models", https://arxiv.org/abs/2205.01068.
  Read the logbook front-to-back at least once; it is short by
  logbook standards (~50 pages, dated entries), and it is the
  single most honest account of a real large-model training run's
  incident stream that has been published.
- **The Llama 3 Herd of Models** (Meta, 2024). Grattafiori, A.,
  et al., https://arxiv.org/abs/2407.21783. Section 3.3.2
  ("Training infrastructure, at scale") reports the incident
  taxonomy and rates for a 54-day 16 384-H100 pretraining run.
  This is the modern-frontier reference; read it after the OPT
  chronicles to see how much (and how little) has changed.

Two supporting references that you should have skimmed:

- **BLOOM training paper (Le Scao et al., 2022), "BLOOM: A 176B-Parameter
  Open-Access Multilingual Language Model", https://arxiv.org/abs/2211.05100.**
  Section on training compares infra choices to OPT and covers a
  distinct set of incidents.
- **Google TPU v4 paper (Jouppi et al., 2023), "TPU v4: An Optically
  Reconfigurable Supercomputer for Machine Learning", ISCA 2023,
  https://dl.acm.org/doi/10.1145/3579371.3589350.** Introduces
  the "goodput" framing at TPU-pod scale that chapter 8 formalizes
  for GPU pods.

## What the OPT-175B chronicles say

The OPT-175B logbook covers the 992-A100 pretraining of a 175B
parameter model in late 2021. It is written as a dated engineering
log — day-by-day entries in the voice of the on-call engineers. Some
of the highest-leverage observations, cited to the chronicles PDF and
paraphrased so you can go find them:

- **Hardware failures required frequent restarts.** The logbook
  reports a stream of GPU / node hardware failures across the 992-GPU
  cluster, each requiring restarts, occasional model surgery, and
  operator judgment about whether to resume or roll back. The
  volume was high enough that "hardware failure" is one of the top
  categories the log tags.
- **Loss spikes and divergences required LR reduction and
  rollback.** Multiple entries describe a spike, a rollback, and a
  restart with a lower learning rate. The team learned to keep
  frequent checkpoints specifically because rollbacks were common
  enough that a large checkpoint interval would have amplified the
  cost.
- **Silent hangs required manual detection.** Several entries
  describe the run appearing to make progress in the logs while
  actually being stalled. The response was a per-step throughput
  check that would page if the ratio dropped.
- **Restart configuration bugs cost more than the original
  failure.** Several entries describe restarts that then had their
  own issues — a wrong config, a bad checkpoint pick, a scheduler
  that placed the gang badly. The takeaway: the recovery flow needs
  its own testing, not just the training flow.

The chronicles' overall tone is instructive: this was a very
capable team on carefully-chosen hardware, and the run was still a
running battle with the failure stream. If you take away one thing
from the chronicles, take away that **a competent team plus
best-in-class hardware plus a well-known model still has
incident-driven downtime measured in double-digit percent**. That
is the number chapter 8's SLO is authored against.

Read the chronicles PDF for the tone; the exact incident counts and
rates are less important than the *shape* of the log, which is
"almost every day has an entry".

## What the Llama 3 report says

The Llama 3 report (2024) is more concise and more quantitative.
Section 3.3.2 reports specific numbers for the 54-day 16 384-H100
pretraining run. Key numbers, cited to the paper:

- **466 total job interruptions in 54 days.** That is about **8.6
  interruptions per day** on average.
- **419 of those were unexpected**, i.e. not planned by the
  operators. About **7.8 unexpected interruptions per day**, or
  roughly **one every ~3 hours** on the wall clock.
- **The remaining 47 were expected**, e.g. planned maintenance and
  operator-triggered restarts.
- **GPU-related issues account for approximately 58% of the
  unexpected interruptions**, split across faulty GPUs and
  GPU-adjacent components (HBM3 issues, thermal, etc.). Adding CPU
  and host-side issues brings the "host hardware" category
  higher.
- **Non-hardware categories** — dataset issues, framework bugs,
  storage issues, network incidents — make up the remainder.

The paper explicitly names the design choices that made this
tractable, and each maps directly to a chapter of this module:

- **Frequent DCP checkpoints** (chapter 3) to bound work loss per
  failure.
- **Automated failure detection and node quarantine** (chapter 6)
  so that a bad node does not repeatedly re-fire.
- **A specific goodput target** the team measured against
  (chapter 8), reported in the paper as **above 90% effective
  training time** during the pretraining run.

The paper does not spell out the underlying rendezvous / elastic
mechanism as explicitly, but the operational shape is exactly the
chapter 4 flow: fail-stop failure → automated detection → shrink /
reshape → resume from a recent checkpoint.

## Translating logbook lessons into runbook entries

Every observation above should generate one or more entries in
your platform's runbook. The mapping:

| Logbook lesson | Runbook entry class | Where in this module |
|----------------|---------------------|----------------------|
| Frequent hardware failures | Class 4 (hardware fault) with DCGM automation | Chapter 6 |
| Loss spikes + LR rollback | Class 1 (loss spike) with checkpoint-rollback playbook | Chapter 5 |
| Silent hangs | Class 3 (NCCL timeout) with step-time watchdog | Chapter 5, chapter 6 |
| Restart-config bugs | Recovery-path testing as a first-class deliverable | Chapters 4 and 5 |
| ~1 unexpected interruption / 3 hours at scale | Availability budget derivation | Chapter 8 |
| >90% effective training time as a target | Goodput SLO shape | Chapter 8 |
| GPU-heavy interruption mix | Node quarantine + DCGM investment | Chapter 6 |

Exercise 3 asks you to produce your platform's version of this
mapping, with a specific playbook per class grounded in this table.

## Two patterns from the literature worth adopting

Beyond the specific incidents, both papers surface **operational
patterns** that are worth adopting whether or not you ever hit the
scale that motivated them:

### 1. The daily / weekly logbook

The OPT chronicles' form — a plain-text, dated logbook of events,
decisions, and their reasoning — is the piece of infrastructure that
made incident recurrence tractable. Six weeks into the run, someone
was hitting a symptom the team had already seen at week 2, and the
logbook was what told them.

A logbook is not a metrics dashboard; it is a **narrative** log of
what the on-call thought was happening and what they did about it.
It is one of the few pieces of a training platform that is
deliberately low-tech: a Markdown file per run, appended to, in
timestamp order.

Adopt this before you need it. The zero-cost version:
`runs/<run-id>/logbook.md`, appended by the on-call, reviewed at
weekly training-team standups.

### 2. Recovery-path testing

Both papers implicitly (OPT) and explicitly (Llama 3, via the
frequent-checkpoint design) treat the recovery path as its own
tested system. The pattern:

- Have a synthetic-failure drill. Once a week, kill a random node
  in a canary training job. Measure recovery time. Alert if it
  regresses beyond the SLO's budget.
- Have a DCP-load smoke test. On every new PyTorch / DCP / model
  release, run "save at N ranks, load at N/2 ranks" end-to-end.
- Have a rendezvous-timeout drill. Once a quarter, verify that a
  simulated split-brain (network partition on the rendezvous
  endpoint) does not corrupt state.

Exercise 2 is the drill-authoring equivalent of this pattern for
elastic reshape; exercise 4 is the drill-authoring equivalent for
straggler / SDC detection.

## What the papers do not tell you

Two things the public accounts underspecify, and where your
platform has to build its own institutional knowledge:

- **The cost accounting behind the "keep going" vs. "roll back"
  decision.** Both papers describe rollbacks but do not publish the
  decision rubric. Your team's rubric — LR-drop threshold, spike
  size threshold, rollback-N-checkpoints policy — has to be
  authored locally. Chapter 5's decision-line rule is what you
  start from.
- **The specific silent-corruption rate.** Neither paper publishes
  a hard number for SDC rate. The Meta and Google SDC papers in
  `resources.md` give you background but not a workload-specific
  number. Chapter 6's replay drill is how your platform develops
  its own estimate.

## Reading order

If you have never read either paper, read them in this order for
the first pass:

1. **OPT chronicles**, front-to-back, ~1.5 hours. Skim the
   day-by-day entries; the point is the *volume* and *shape* of
   the events, not any single entry.
2. **Llama 3 report §3.3.2**, ~30 minutes. The quantitative
   anchor. Cross-reference each number to a chapter of this
   module.
3. **PaLM §5 (goodput discussion)**, ~15 minutes. Enough context
   for chapter 8.
4. **BLOOM training section**, ~30 minutes. A distinct-team
   perspective on similar-shape problems.

Exercise 3 asks you to author a mapping doc that translates the
Llama 3 §3.3.2 taxonomy into your platform's runbook. That is the
concrete exercise this chapter builds toward.

## Summary

- The OPT-175B chronicles and Llama 3 §3.3.2 are the two
  primary-source empirical logbooks for large-scale pretraining
  incidents. Read both.
- Llama 3's 54-day run at 16 384 H100s reports 466 total
  interruptions (419 unexpected), ~58% GPU-related, with an
  operational target of >90% effective training time. Use those
  numbers as your worst-case reference until your own runs prove
  different.
- Every logbook lesson maps to a runbook entry: hardware faults →
  DCGM quarantine, loss spikes → rollback + LR drop, silent hangs
  → step-time watchdog, restart bugs → recovery-path testing.
- Two adopt-early patterns: a plain-text run logbook and a
  scheduled recovery-path drill. Both are cheap; both compound.
- What the papers do not tell you (rollback rubric, SDC rate) is
  what your platform has to build. Chapters 5 and 6 give you the
  starting shape.
