# Incident Review Anchored in OPT-175B and BLOOM

Every large-scale training run in this field's short history has
been an *incident record*. The papers are polished; the logbooks
and reliability appendices are the real artifact. Two of them —
the OPT-175B logbook (Zhang et al., 2022, arXiv:2205.01068,
released with the Meta AI OPT logbook posted alongside the model)
and the BLOOM paper's reliability discussion (Le Scao et al.,
2022, arXiv:2211.05100) — are the most public detailed records
we have of what actually happens when a foundation-model
pretraining run meets the messy reality of a shared HPC cluster.
Llama 3 (Grattafiori et al., 2024) makes it three; §6 of that
report contains the closest thing to a modern reliability
appendix any frontier lab has published.

This chapter reads those three artifacts as *incident reviews* and
turns them into a template you can apply to your own runs. The
exercise-05 deliverable is a full post-mortem in that template,
grounded in one of these three source records. mod-106's fault
tolerance material teaches you the mechanisms; this chapter
teaches you the artifact.

Two framing points before the template.

First: **an incident review is a written artifact, not a
conversation.** The value shows up months later, when someone
encounters a similar failure and reads yours. Verbal debriefs are
not incident reviews.

Second: **the incident review is blameless.** The reader is trying
to learn what happened and how to prevent it, not who to blame.
Every reviewed incident is a system failure — the human at the
keyboard is one component of the system, and the system's design
is what let a human error become a run failure.

## What each shop's incident-review artifact contains

The three publicly available records are shaped slightly
differently, but they converge on the same content:

**OPT-175B logbook.** A running text log of the training run,
posted publicly. It contains dated entries, symptoms, hypotheses,
actions taken, and outcomes. It is the closest thing the field
has to a *raw* incident-response record. The formal paper (Zhang
et al., 2022) references the logbook and summarises the categories
of failures the run hit: loss spikes and divergence, hardware
faults (notably GPU / node crashes forcing restarts from
checkpoints), silent numeric issues, and the operational choices
around learning-rate resets and gradient clipping.

**BLOOM.** The published paper (Le Scao et al., 2022) has a
section on the training infrastructure and the reliability of the
Jean Zay supercomputer during the run. The BigScience associated
engineering write-ups (blog posts and slides across the BigScience
public wiki) go into more operational depth on hardware failure
rate, restart cost, and the specific software failures encountered
in Megatron-DeepSpeed. Reading the paper and the associated
BigScience engineering material together gives you the
incident-review perspective.

**Llama 3.** §6.3 of the Llama 3 paper (Grattafiori et al., 2024)
gives statistics: how many interruptions, what fraction were
hardware-caused vs. software-caused, what the mean time to
detection was, and how they built the tooling to bring the
detection window down. It is the least conversational of the
three but the most modern; it is written by a team that had
learned from OPT and BLOOM and built the reliability
instrumentation into the platform from day one.

## What they have in common

All three artifacts converge on the same content categories:

- **Timeline of events.** When did the symptom appear, when was
  it noticed, when was action taken?
- **Symptom.** What did the operator actually see — a loss spike,
  a NaN, a NCCL timeout, a node marked down?
- **Telemetry snapshot.** What did the dashboards show at the
  time — loss curve, gradient norm, GPU utilisation, DCGM error
  counters, NCCL debug logs?
- **Hypothesis space.** What did the responders think might be
  causing the symptom?
- **Root cause.** What was actually the cause? (Sometimes the
  reviews are honest that root cause is unknown or uncertain.)
- **Mitigations attempted.** What did they try before it worked?
- **Recovery cost.** How much wall-clock, how many steps rewound,
  what did the loss curve look like on resume?
- **Durable fix.** What went into the runbook, the platform
  config, the code, the hardware maintenance schedule?
- **Runbook update.** How does the next occurrence get handled
  faster?

The template below formalises exactly this.

## The nine-section incident-review template

Every training-run post-mortem your platform produces uses this
shape. Ninety percent of the value comes from filling in every
section honestly; the remaining ten percent comes from the
"durable fix owner" line making the fix actually happen.

| Section                    | What it contains                                                                                     | Common failure mode                                              |
|----------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| 1. Incident window         | Start (first symptom time), acknowledgement time, mitigation time, resolution time. UTC.             | "Sometime Tuesday afternoon" — imprecise timestamps              |
| 2. Initial symptom         | What the operator actually saw. One sentence.                                                        | Author skips ahead to root cause without naming the symptom      |
| 3. Telemetry snapshot      | Loss curve, gradient norm, MFU, GPU util, DCGM, NCCL_DEBUG at the incident window. Screenshots or CSVs. | Missing telemetry — post-mortem cannot be reproduced             |
| 4. Hypothesis space        | Numbered list of hypotheses considered, with why each was ruled in/out.                              | Only the winning hypothesis is listed — reader can't learn       |
| 5. Verified root cause     | What was actually true. Include how it was verified.                                                 | "Probably X" without verification                                |
| 6. Mitigation applied      | What resolved the incident. Include the exact command / config change.                               | "Restarted the job" — no specifics                               |
| 7. Recovery cost           | Steps rewound, GPU-hours lost, wall-clock impact, checkpoint used to resume.                         | Unstated cost — org can't budget                                 |
| 8. Durable-fix owner       | Named human + ticket + due date. Not "we should."                                                     | No owner — fix does not happen                                   |
| 9. Runbook update          | Link to the runbook entry (new or updated) with the symptom → diagnosis → fix flow.                  | Runbook change is a promise, not a link                          |

The nine sections are ordered roughly by information density. A
reader in a hurry reads 1–3 and knows what happened; a reader
learning reads 4–7 and understands the reasoning; a reader
following up reads 8–9 to see whether the loop closed.

## Case: OPT-175B loss-spike-then-diverge sequence

Reading the OPT logbook through the template.

**Incident window.** The logbook records several loss-spike
incidents across the 3-month training run, with representative
entries in the first weeks of training. Exact timestamps are in
the log; the shape of each incident is: loss spikes over the
course of a few dozen steps, then either recovers or diverges into
NaN.

**Initial symptom.** Training loss increases sharply over a short
step window (rather than the expected slow decline), sometimes
recovering, sometimes progressing to NaN loss and irrecoverable
divergence.

**Telemetry snapshot.** The team plotted loss curves, gradient
norm, and per-parameter gradient statistics. The characteristic
signature was a large gradient norm event immediately preceding
the loss spike — the "step was too large" pattern that any
Adam-family optimiser can exhibit when the loss surface has a
locally very steep region.

**Hypothesis space.** The team considered (a) an actual data
issue — a problematic shard containing pathological tokens; (b) a
numerics issue in the FP16 mixed-precision path; (c) an optimiser
issue (too-large learning rate for the current state); (d) a
hardware issue introducing bit flips.

**Verified root cause.** Multiple root causes across incidents.
Some were traced to numerical instability under FP16 with the
particular gradient scaling in use; some to the learning-rate
schedule interacting with a temporarily steep loss region; the
team's mitigations tell you which was which. The logbook is
honest about uncertainty.

**Mitigations applied.** Learning-rate resets (rewinding to a
recent checkpoint and lowering LR briefly), gradient clipping
adjustments, occasional short skips over shard ranges, and — for
the terminal FP16 issues — a partial move toward stabler
precision handling.

**Recovery cost.** Non-trivial; the logbook records multiple
rewinds of thousands of steps, cumulative GPU-hour cost estimable
from the run's total step budget vs. wall-clock — see the paper's
§4.1 for the aggregate view. This is the material fact that a
platform team draws on: divergence recovery costs *days* of
wall-clock and material fractions of the total run budget.

**Durable fix.** The team codified LR-reset and clipping
strategies into the training script, adopted a monitoring
approach for gradient norms, and (later in the field) the
techniques moved into FSDP2 and Megatron-Core as first-class
support for gradient-norm clipping strategies.

**Runbook update.** For a modern platform following this history:
the runbook entry for "loss spike" reads (a) confirm gradient
norm spike precedes loss spike, (b) rewind to checkpoint N-1, (c)
if divergence: rewind further and lower LR for M steps, (d) if
persistent: mod-107 numerics check, (e) escalate if not resolved
in K steps.

## Case: OPT-175B hardware-fault sequence

**Incident window.** Multiple entries throughout the run; the
logbook records node failures at rates broadly consistent with
"one failure per hundreds of node-hours in aggregate," but the
specific counts and dates are in the log.

**Initial symptom.** Job hangs or NCCL timeout; a specific node
becomes unresponsive; sometimes a specific GPU throws
uncorrectable ECC.

**Telemetry snapshot.** NCCL_DEBUG output showing collective
stalled on a particular rank; DCGM showing that rank's node
recently degraded; sometimes hardware health monitoring flagged
the node before the collective stalled.

**Hypothesis space.** (a) NIC issue, (b) GPU HBM ECC, (c) OS
kernel issue on the node, (d) transient network congestion.

**Verified root cause.** Varied per incident; hardware faults
were the most frequently confirmed root cause.

**Mitigations applied.** Quarantine the node, resume from the
last checkpoint on a replacement, replay the affected steps.
mod-106 chapters 3–4 cover the elastic-restart mechanics.

**Recovery cost.** Time to detect + time to restart + time to
replay to the pre-fault step. Modern platforms drive this down
into minutes with fast detection and DCP async save; older
platforms spent hours per event.

**Durable fix.** Hardware quarantine automation, faster
detection (DCGM into Prometheus into a paging rule),
elastic-restart plumbing so a replacement node is automatic.

**Runbook update.** For a modern platform: "NCCL timeout on rank
R → check DCGM for node hosting rank R → quarantine node →
elastic restart on replacement → log the fault to the
hardware-failure ledger."

## Case: BLOOM hardware-failure sequence

BLOOM ran on the Jean Zay supercomputer. The reliability material
in Le Scao et al. (2022) and the BigScience engineering
write-ups discusses concrete hardware failure rates on a shared
HPC platform. Applying the template:

**Initial symptom.** Similar to OPT — node hangs, NCCL stalls,
occasional GPU HBM ECC events.

**Telemetry snapshot.** NCCL debug, Jean Zay's cluster telemetry
tools, and the run's own monitoring.

**Hypothesis space.** Same as OPT.

**Root cause.** Hardware faults on a shared supercomputer, with
the added dimension that the platform was shared with other
users — so failures sometimes came from adjacent jobs' effects
on the shared fabric.

**Mitigation.** Restart from checkpoint, sometimes wait for the
adjacent job to clear, occasionally coordinate with Jean Zay's
operators for a node replacement.

**Recovery cost.** Real; the paper discusses restart frequency
and the effect on wall-clock schedule. Multi-week rescheduling
followed serious hardware events.

**Durable fix.** Later work in the field (Llama 3's
reliability instrumentation, mod-106's stateful sampler and DCP
resharding) directly addresses the gap.

**Runbook update.** For a modern platform: the runbook adds a
"shared cluster" branch — check whether the fault is your job's
or adjacent-job-caused, coordinate with the shared operators as
needed.

## Case: Llama 3 §6 reliability instrumentation

Llama 3 is the newest of the three and is the least
narrative-shaped, but it is the most useful piece to extract
platform lessons from because the team explicitly designed the
platform around the OPT / BLOOM lessons.

Key facts to lift into your platform:

- The team measured **mean time to interruption** and
  **fraction of time in productive step execution** (a goodput
  measure — mod-106 chapter 7 talks about this).
- Fault detection was instrumented to the *node* granularity in
  near-real-time; a node failure was quarantined and the job
  restarted from the last checkpoint automatically.
- The frequency of hardware faults was not zero — training at
  the 10 000+ GPU scale for weeks means multiple failures per
  day even on modern hardware.
- The team wrote reliability tooling as a *platform*
  responsibility, not a per-run responsibility. This is exactly
  the level-35 altitude framing that this module argues for.

Reading §6.3 with the template in hand teaches you not to write
your incident review the same shape — Meta's engineers already
did that work — but rather to build the *instrumentation* that
lets an incident review be filled in quickly.

## Practice: applying the template to your own runs

The exercise-05 deliverable is a full post-mortem for either
(a) an OPT-175B logbook incident of your choice, filled in from
the log's raw text, or (b) a BLOOM hardware-failure sequence,
filled in from the paper + BigScience engineering material. Use
the nine-section template exactly. Be honest about what you cannot
extract from the source; a section reading "the source does not
report — my platform would answer this by …" is a legitimate
answer.

A common inexperience pattern is skipping section 4 (the
hypothesis space). Do not. The hypothesis space is where the
reader learns to think like an on-caller: what were the
plausible things it could have been, and how would you have
ruled them in or out? A post-mortem that lists only the winning
hypothesis teaches nothing except the conclusion.

Another common pattern is under-specifying section 8 (durable
fix owner). "The team will consider improving X" is not an
owner; "Alice files a ticket by end of week for X, due in Sprint
32" is. If your post-mortems do not close their durable-fix
loops, run the same fires next quarter.

## What your platform's incident-review process looks like

Wrap the template in a process:

1. **Every P0/P1 incident produces a review within one week** of
   resolution. Non-negotiable.
2. **Review is drafted by the incident commander** (chapter 1
   § war-room protocol) with input from the SME leads.
3. **Review is read in a scheduled meeting** attended by the
   platform team and any affected peer teams. Reading it aloud
   catches gaps; sending it around asynchronously does not.
4. **Every review has an assigned durable-fix owner**; the owner
   reports back the next week on progress. Reviews without
   closed loops go on the "reviews to revisit" list.
5. **Runbook updates land in the same PR** as the review. Not
   a promise; a diff.
6. **Reviews are published internally** — the platform wiki has a
   post-mortem folder that engineers can read. This is the
   long-run compounding value.

The frontier labs that talk about their incident practices
publicly (Meta, Google, others) all describe some variant of
this. If you have not seen a mature review process before, model
after it.

## Summary

- The OPT-175B logbook, the BLOOM reliability discussion, and the
  Llama 3 §6.3 appendix are the three most useful public incident
  records the field has. Read them as incident reviews, not as
  papers.
- The nine-section template (incident window, symptom, telemetry,
  hypothesis space, root cause, mitigation, recovery cost,
  durable-fix owner, runbook update) is the shape every training
  post-mortem takes.
- OPT-175B teaches the loss-spike-then-diverge and hardware-fault
  patterns. BLOOM adds the shared-cluster dimension. Llama 3
  teaches what platform instrumentation makes those reviews easy
  to fill in.
- Post-mortems are blameless, written, and time-boxed to one week
  after resolution. Every review has a named durable-fix owner
  with a ticket and a due date.
- Runbook updates land in the review PR — not as a promise, as a
  diff.
- The compounding value shows up months later: the next on-caller
  reads your review and skips the hypothesis-space exploration
  you already did.
