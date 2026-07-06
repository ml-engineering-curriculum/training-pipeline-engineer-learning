# exercise-02: Parallelism-Strategy Selection Rubric

**Estimated effort:** 4 hours

## Objective

For three concrete `(model, cluster)` scenarios, design and defend a
parallelism strategy across the DP / TP / PP / SP / EP / HSDP axes,
sketching the DeviceMesh and predicting the comm-to-compute ratio for
each. The output is a rubric-based decision document — the artifact you
would attach to an internal RFC when a training team proposes a new
model size.

## Prerequisites

- Chapters 1 through 5 of this module, especially the decision procedure
  in chapter 4 and the α + β cost model in chapter 5.
- Have skimmed the Megatron-LM paper (Shoeybi et al., 2019) and the
  follow-up "Efficient Large-Scale" paper (Narayanan et al., 2021).

## Problem statement

You are the training-platform lead. Three requests come in on the same
Monday:

- **Team A** wants to train a ~7B dense decoder-only transformer on a
  single node of 8 × H100 (SXM, NVSwitch, 80 GB HBM3), no cross-node
  fabric available for this team's quota.
- **Team B** wants to train a ~70B dense model on 4 nodes × 8 × H100
  (NVSwitch inside each node, HDR InfiniBand between nodes).
- **Team C** wants to train a ~200B parameter *sparse* MoE model with
  64 experts on 16 nodes × 8 × H100 (NVSwitch + NDR InfiniBand).

For each team, you must produce a written strategy the team can execute
against.

## Requirements

Deliver one document (Markdown, PDF, or slide deck) with a section per
team. Each section must contain:

1. **The strategy** — an axis-by-axis breakdown:
   `(DP, TP, PP, SP, EP, HSDP)` values, chosen framework (FSDP2 /
   Megatron-LM / DeepSpeed / etc.), and the DeviceMesh sketch.
2. **The forcing argument** — for each axis you introduce, one paragraph
   explaining what memory or throughput problem it solves and why the
   next-simpler option would have failed. Use chapter 4's decision
   procedure explicitly.
3. **The cost budget** — for at least the dominant collective on each
   axis, an α + β estimate using order-of-magnitude bandwidth numbers
   from chapter 5. State assumptions clearly (batch size, sequence
   length, hidden size).
4. **The comm-to-compute ratio** — a single number per team, with the
   arithmetic behind it. Cite the `6 · P · tokens` FLOP count from
   Kaplan et al., 2020 for the compute side. Assume a MFU you consider
   defensible and justify the choice.
5. **The failure modes** — for each strategy, list the two or three
   operational failure modes you expect to see first (see chapter 4 and
   the OPT logbook references in chapter 7 for inspiration).
6. **The alternative you rejected** — one design you seriously
   considered and why it lost. This is what turns the document from a
   picture into a decision.

## Starter guidance

- Do not try to be exhaustive. Two competing strategies per team is
  enough to make the argument concrete.
- Use round-number model dimensions: e.g., 7B ≈ (hidden 4096, layers 32),
  70B ≈ (hidden 8192, layers 80), 200B MoE ≈ (hidden 8192, layers 60,
  E=64). If you use a public model's exact dims, cite them.
- For Team C, spend real time on the expert-parallel routing math and
  the all-to-all cost. GShard (Lepikhin et al., 2020) and Switch
  Transformer (Fedus et al., 2021) are the primary references.
- Keep TP inside a node unless you can defend a specific reason to
  break the rule. If you do break it, quantify the penalty.
- Sketch the DeviceMesh both mathematically (`(TP=8, PP=8, DP=4)`)
  and visually (a small diagram showing which rank sits on which
  physical GPU).

## Acceptance criteria

- Each team has a specific, defensible strategy with numeric axis
  sizes and a stated framework.
- Each axis choice is justified against chapter 4's decision procedure
  in one paragraph.
- The comm-to-compute ratio for each team includes the arithmetic and
  the sanity numbers used. It does not have to match reality to three
  significant digits, but it must be internally consistent.
- At least one team writeup names a specific *rejected* alternative and
  gives the reason (numeric where possible).
- The document cites at least three primary sources across the three
  teams.

## Stretch goals

- Add a fourth team: a 1B model on a heterogeneous cluster of 4 × A100
  40 GB + 4 × H100 80 GB. Reason about whether the strategy must degrade
  to the slower device or whether the heterogeneity can be exploited.
- Re-do Team B's design for a cluster with RoCEv2 instead of IB.
  Quantify what changes in the comm budget and whether the strategy
  needs to move.
- For Team C, extend to a scheduler-level failure mode: what happens
  when one expert's assigned rank fails mid-step? Cross-reference
  mod-106 (fault tolerance) if you already have it open.
