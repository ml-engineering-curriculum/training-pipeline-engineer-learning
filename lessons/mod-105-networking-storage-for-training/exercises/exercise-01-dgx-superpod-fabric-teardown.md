# exercise-01: DGX SuperPOD Fabric Teardown

**Estimated effort:** 3 hours

## Objective

Read the current NVIDIA DGX SuperPOD Reference Architecture PDF and
produce a written "teardown" — a diagram plus a short design memo —
that proves you can reconstruct the fabric mental model from primary
documentation. The deliverable is the artifact you would send to a
new-hire training platform engineer on day one to bring them up to
speed on your pod.

This exercise anchors the rest of the module. Every subsequent
exercise assumes you have the pod's topology in your head; this one
is the read-and-write pass that puts it there.

## Prerequisites

- Chapters 1, 2, and 3 of this module.
- A copy of the current DGX SuperPOD Reference Architecture PDF for
  Hopper (H100 / H200) or Blackwell (B200) systems, downloaded from
  the NVIDIA DGX SuperPOD documentation index
  (https://docs.nvidia.com/dgx-superpod/). Use the Hopper document
  if you are not sure — the chapters were written against it.
- A drawing tool of your choice (Excalidraw, draw.io, plain
  paper-and-scan).

## Problem statement

Your training platform team is onboarding a new engineer. Their first
day, they will be asked to answer three questions about your pod:
"What is the largest gang that fits in one SU?", "What is the
nominal aggregate all-reduce bandwidth per GPU at the SU boundary?"
and "Which rail does host `dgx-042`'s HCA 0 land on?" If they cannot
answer these, they cannot debug the fabric. Your job is to build the
document that lets them.

## Requirements

1. **Read the reference architecture end-to-end.** The whole PDF, not
   just the summary. Do not skip the storage-fabric section or the
   management-fabric section; you need to be able to name the four
   fabrics and their isolation.

2. **Produce three diagrams.** Together they compose the pod picture.

   - **Diagram A: one DGX node.** Show 8 GPUs, the 4 NVSwitch3 chips
     (or the current generation's equivalent), the 8 compute-fabric
     HCAs (one per GPU, labelled by rail), and the storage / management
     NICs. Annotate the NVLink aggregate bandwidth per GPU. Cite the
     page and section of the reference architecture your numbers came
     from.
   - **Diagram B: one SU.** Show the count of DGX nodes in an SU, the
     count of leaf switches per rail, the count of spine switches per
     rail (inside the SU), and how rail `k` on every node connects to
     that rail's leaf switches. Annotate the SU's aggregate GPU count
     and the nominal per-rail bandwidth.
   - **Diagram C: full SuperPOD (SU-to-SU spine).** Show how SUs
     compose. Annotate the number of SUs supported at the current
     generation and the aggregate GPU count of the largest supported
     pod.

3. **Produce a design memo (600–1200 words).** It must contain the
   following sections:

   - **The four fabrics.** Name compute, storage, in-band management,
     out-of-band management. For each, name the physical medium, the
     approximate bandwidth per node, and its role.
   - **Rail alignment.** Explain, in your own words, what "the
     rail-optimized fat-tree" means and why it matters for training
     collectives. Reference chapter 3 of this module.
   - **What is one SU?** State the node count, GPU count, leaf/spine
     switch count, and aggregate bisection bandwidth at the SU
     boundary. Reference the specific pages of the PDF that gave you
     each number.
   - **What is the SuperPOD?** Same, one level up. State the largest
     supported pod at the current generation and the total GPU count.
   - **The three onboarding questions.** Answer each of the three
     from the problem statement, with the numbers derived above.
     Show your work.

4. **Highlight one thing you disagree with or find under-specified.**
   Every reference architecture leaves choices to the implementer.
   Pick one — a knob, a rate, a rack topology, a firmware requirement,
   a cabling constraint — that you would want to see clarified before
   you signed off on a build. Explain why. This is the part of the
   memo that proves you *read* the document rather than skimmed it.

## Starter guidance

- Do the reading in the order: cluster overview → node hardware →
  compute fabric → storage fabric → management fabrics → software
  stack. Every generation's PDF is organized roughly this way.
- Diagram A is the same across most modern DGX generations
  (Hopper/Blackwell); the SU and pod diagrams differ across
  generations. Use the specific PDF you have in hand.
- If you are working on a cloud (AWS/Azure/GCP) instead of an
  on-prem pod, run the same exercise against the hyperscaler's
  reference architecture for their equivalent product (AWS EC2 P5 /
  P5e capacity blocks, Azure ND-series topology guides, GCP A3
  cluster docs). Note the differences from the NVIDIA reference in
  your memo.
- Do not paraphrase the PDF — write in your own words. If you are
  quoting a table verbatim, cite the page.
- The memo is short on purpose. This is not a research report; it
  is the onboarding note.

## Acceptance criteria

- Three diagrams (A, B, C) that a colleague could read without the
  PDF in front of them.
- The memo names all four fabrics correctly and does not confuse
  compute with storage.
- Every quantitative claim (bandwidth, node count, switch count)
  cites either a specific page of the PDF or the specific
  vendor-doc URL it came from.
- The three onboarding questions are answered with numbers and a
  reasoning chain, not just "8" / "a lot" / "somewhere".
- The "one thing you disagree with" is specific enough that a
  reviewer could act on it, not just "the docs could be clearer".
- Nothing in the memo is invented. If a number is not in the PDF
  and you cannot verify it against another primary source, leave a
  `TBD` and open a question rather than guessing.

## Stretch goals

- Cross-reference against a second generation's PDF (Hopper vs.
  Blackwell). What changed between generations, and why? Write a
  200-word addendum.
- Cross-reference against a hyperscaler's equivalent
  (AWS EC2 P5 / Azure ND-series / GCP A3). Where do the reference
  architectures agree and where do they diverge?
- Produce a fourth diagram: the storage fabric, showing MDS / OSS
  hosts (for a Lustre-backed pod) or WEKA server hosts (for a
  WEKA-backed pod). Note the storage NIC count and the storage
  fabric's isolation from the compute fabric.
- Turn the memo into a wiki page suitable for your team's internal
  documentation, with cross-links to the specific pages of the PDF
  and to chapters 2 and 3 of this module.
