# Case Studies: Reading Llama 3, BLOOM, and OPT-175B

Every parallelism strategy in this module has a paper behind it. Every
paper is also a *design record* — the author explaining, in their own
words, why they made the choice they did given the model, the cluster,
and the calendar. This chapter's goal is to teach you to read those three
papers the way a training-platform engineer reads them: extracting the
strategy, the constraints that forced it, and the failure modes it had to
plan for.

By the end of this chapter you should be able to summarize each paper's
parallelism choice in one paragraph and, given a hypothetical alternative
cluster shape, argue what the paper's team would probably have chosen
instead.

## The reading protocol

For each paper, walk in with this checklist and take notes:

1. **Model shape.** Parameter count, layer count, hidden size, sequence
   length, vocab, MoE or dense.
2. **Cluster shape.** How many GPUs? Of what generation? On what fabric
   (NVLink? IB? RoCE?)? What storage?
3. **Parallelism strategy.** DP × TP × PP × SP × EP — with the axis
   sizes. Which framework(s) implemented it?
4. **The forcing constraint.** Why this strategy and not the obvious next
   simpler thing? Was memory the wall? Communication? Wall-clock deadline?
5. **Failure modes documented.** Hardware failures, loss spikes,
   silent-corruption episodes, cluster degradation.
6. **What they published, what they omitted.** Every paper is also a
   sales pitch. Notice what a peer training-platform engineer would want
   to know that the paper does not tell you.

## Llama 3 (Meta, 2024)

**Reference:** Grattafiori et al., 2024, "The Llama 3 Herd of Models"
(the technical paper released with the Llama 3 launch).

Reading focus points:

- The **4-D parallelism** used for the 405B variant: TP × CP × PP × DP,
  layered as described in chapter 4. Identify each axis's size and which
  fabric link it lands on.
- The paper's description of **FP8 mixed precision** on H100 and how it
  composed with the parallelism plan. What has to be BF16 or FP32 for
  numerical stability, and what can safely be FP8?
- The section on **network and training reliability** — how the team
  handled hardware faults, silent corruption, and restart cost. This is
  the piece that most other papers gloss over; Llama 3 is unusually
  detailed here.
- The **context parallelism** discussion — how sequence-parallel
  ("context parallel") was scaled to long context.

Given the size of the paper, prioritize the model + system sections and the
appendix on training infrastructure. It is fine to skim the eval-heavy
sections for this module.

## BLOOM-176B (BigScience, 2022)

**Reference:** Le Scao et al., 2022, "BLOOM: A 176B-Parameter Open-Access
Multilingual Language Model".

Reading focus points:

- **Megatron-DeepSpeed** as the framework: TP inside nodes, PP across
  nodes, DP on the outer axis, layered with ZeRO-1. This is a canonical
  3D-parallel setup and worth walking through against the design procedure
  in chapter 4.
- Cluster shape (Jean Zay supercomputer). The paper reports the fabric
  and the memory / storage decisions in enough detail that you can
  reconstruct the per-GPU comm budget yourself.
- **Multilingual data considerations**. These are not directly a
  parallelism concern, but the choice of a large vocabulary (~250k)
  interacts with the embedding-sharding strategy and is worth noting.
- **Logbook / retrospective material**. The BigScience blog posts and
  the associated engineering write-ups describe the operational side
  (failures, restarts, throughput ramp) more thoroughly than the paper
  itself.

## OPT-175B (Meta AI, 2022)

**Reference:** Zhang et al., 2022, "OPT: Open Pre-trained Transformer
Language Models" (paper) and the accompanying **OPT-175B logbook**
released alongside the model.

Reading focus points:

- **Framework and parallelism.** OPT was trained with
  Megatron-LM's tensor parallel plus Fully Sharded Data Parallel
  (FSDP-like) — the paper describes the specifics.
- **The logbook**. This is the reason OPT is on the reading list. It is
  the most detailed public record we have of what actually happens
  during a large-scale training run: loss spikes, learning-rate resets,
  divergence recoveries, hardware failures, and the human decisions made
  in response. For a training-platform engineer, the logbook is more
  valuable than the paper.
- **Correlate the incidents in the logbook with the cost model in
  chapter 5.** Where is comm cost showing up as an observable? Where does
  hardware degradation manifest as a loss / throughput signal? mod-106
  will build the incident-classification playbook you use to react to
  these; this chapter is the reading that motivates it.

## The comparative exercise

After you have read each paper, fill in this table (this is exactly what
exercise 2 asks you to do more formally):

| Dimension       | Llama 3 (405B) | BLOOM (176B) | OPT (175B) |
|-----------------|----------------|--------------|------------|
| Params          | 405 B           | 176 B         | 175 B       |
| Framework       | ?               | Megatron-DeepSpeed | Megatron-LM + FSDP-style |
| TP              | ?               | ?             | ?           |
| PP              | ?               | ?             | ?           |
| DP + sharding   | ?               | ZeRO-1 (DP with opt-state sharding) | FSDP-style |
| Cluster fabric  | ?               | HDR IB (Jean Zay) | ?         |
| Notable failure modes documented | ? | ? | Loss spikes, learning-rate resets, hardware failures (logbook) |

Fill in the exact numbers from the primary sources. The table forces you to
compare like-with-like across three shops that made three different
choices with the same fundamental constraints.

## What to take away

The reason these three papers are on the reading list, and not any other
three:

- **Llama 3** shows what the state of the art in 2024 does when a team
  has H100s, an FP8 recipe, and enough compute for a 4-D mesh with
  extensive fault-tolerance instrumentation.
- **BLOOM** shows an academic/consortium run on shared HPC hardware. The
  strategy is 3D-parallel on Megatron-DeepSpeed with ZeRO-1, and the
  cluster constraints are transparent.
- **OPT** shows what an open, retrospective, engineering-first
  publication looks like. The logbook is the closest thing the field
  has to a shared incident-response record for a large training run.

Together, they cover the space of "what has actually been done at scale"
and give you the vocabulary to reason about other public runs (Falcon,
DeepSeek, Mixtral, etc.) as they come out.

## Summary

- Read each of Llama 3, BLOOM, and OPT-175B against the design procedure
  in chapter 4 and the cost model in chapter 5.
- The takeaways are not "copy this strategy" — they are "understand why
  this team, under these constraints, made this call."
- The OPT logbook is the reference this track keeps coming back to for
  operational realism. It is worth returning to at the start of every
  mod-106 (fault tolerance) reading session.
