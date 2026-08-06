# The Training-Run Incident Review

Every mature training platform ships incidents. Frontier
training runs, historically, are 20–50% goodput (Meta's OPT-
175B chronicles, BigScience BLOOM logbook, Grattafiori et al.,
2024 §3.3.2) — meaning half or more of the wall-clock is lost
to hardware failures, silent corruption, loss spikes,
convergence issues, and their recovery. The incident is the
platform's normal operating condition, not an exception.

What separates a mature platform from a chaotic one is not
fewer incidents; it is what happens *after* one. The mature
practice is the written *incident review*: a blameless
retrospective, published within a fixed window, tied to a
canonical taxonomy, that produces named action items with named
owners. This chapter is the template.

Two open-report anchors ground the chapter's examples:

- **OPT-175B logbook** (Zhang et al., 2022; Meta/facebookresearch
  metaseq chronicles), the canonical
  narrative record of loss spikes, hardware losses, and
  recovery decisions in a 175 B pretraining run. Referenced
  throughout as the "spike" anchor.
- **BLOOM training logbook** (BigScience, 2022; the
  bigscience-workshop repository's `train/tr11-176B-ml/` logs),
  the parallel narrative for BLOOM-176B. Referenced as the
  "hardware failure" anchor.

Both are cited in resources.md. Read at least the OPT-175B
chronicles once before writing your first incident review — the
tone is the reference.

## Why write incident reviews

Three mechanisms make written retrospectives worth the effort:

- **Institutional memory.** The same failure recurs six months
  later; the review from last time is what the on-call reads
  in the runbook. Absent a written record, every recurrence is
  first-time debugging.
- **Systemic action items.** A postmortem's action items are
  the mechanism that turns individual incidents into platform
  improvements. Without them, the platform improves only
  through personal memory.
- **Cross-team trust.** The consuming teams (chapter 4) need to
  know that when the platform breaks, the platform *responds*
  visibly. The published postmortem is the response.

The alternative — a Slack post-hoc discussion that fades in a
week — has none of these properties. Write the review down.

## Blameless discipline

Blameless does not mean *actorless*. It means the review names
what happened and why the *system* allowed it, without naming
who to punish. Two rules:

- **Individuals are named as narrators, not causes.** "The
  on-call engineer merged the deploy" is a fact of the
  timeline; "the on-call engineer's decision caused the
  outage" is a cause attribution and is out of scope. The
  system that let the deploy through without a required check
  is the cause.
- **The review team owns the process, not the outcome.** The
  author's job is to make the incident's mechanism visible; it
  is not to assign fault. If leadership wants an accountability
  conversation, that is a separate meeting.

Blameless discipline is what makes people speak. A postmortem
culture where the on-call fears being named as the cause is a
culture where the on-call under-reports. Cite Google SRE Book
§16 (referenced in resources.md) if leadership pushes back on
blameless framing.

## Severity, and when to write a review

Not every incident gets a full review. Use the chapter 2 Sev
ladder:

| Sev | Review artifact                              |
|----:|----------------------------------------------|
|   1 | Full incident review (this template), published within 2 business days of resolution. |
|   2 | Full incident review, published within 5 business days. |
|   3 | Short review (the timeline + one-paragraph analysis + action items), published within 10 business days. |
|   4 | Logged; batched into a monthly trend review; no per-incident postmortem. |

The public-postmortem discipline for Sev 1 and 2 is what makes
the chapter 1 incident contract real. If the incident does not
get written up on the SLA, the contract is not being kept.

## The eight-section template

Every training-run incident review has the same eight sections
in the same order.

### 1. Summary (5 lines)

- **Title.** Date-prefixed and short: "2026-08-05: Foundation
  cluster run FOUND-2026-08-03 loss-spiked at step 47800 and
  did not recover; run terminated after 6 h."
- **Severity.** Sev-N.
- **Duration.** Detect → mitigate → resolve wall-clock.
- **Impact.** Which runs, which teams, how many GPU-hours.
- **Root-cause class.** One of the five canonical classes
  (below).

The summary is the paragraph leadership reads. Everything else
is for the platform team.

### 2. Timeline (bullet list, timestamped)

Every event in the incident window, timestamped to the minute
in UTC. Do not editorialise; state facts.

Structure each line: `HH:MM  UTC  what-happened  (who / how
observed)`. Example:

```
14:03  UTC  training loss on FOUND-2026-08-03 spikes from 2.14
             to 3.72 in one step (mod-108 chapter 2 dashboard).
14:03  UTC  Alertmanager fires L-spike alert (mod-108 chapter 6
             signature 2 rule).
14:04  UTC  On-call (name) acknowledges.
14:05  UTC  On-call inspects gradient-norm panel; sees no
             upstream spike. Cross-references DCGM ECC panel;
             no ECC events.
14:12  UTC  Loss remains at 3.72 across 8 steps. Skip-and-
             resume decision made from mod-106 chapter 5
             runbook rung 3.
14:14  UTC  Rollback to checkpoint at step 47500; resume.
14:17  UTC  Training resumes; loss returns to 2.14.
15:41  UTC  Loss spikes again at step 47812, same shape.
15:41  UTC  Sev-2 declared; incident channel opened.
15:47  UTC  Second rollback; two additional replays; loss
             spikes in the same window every time.
17:31  UTC  Escalated to Sev-1; foundations lead paged; run
             halted pending root-cause analysis.
19:22  UTC  Data-team on-call identifies the shard at token
             offset 47812 as containing a
             UTF-8-decoding-corrupt Wikipedia page ingested
             from a bad crawl on 2026-07-30.
20:00  UTC  Shard removed from the manifest; run resumes from
             step 47500 with the new manifest; loss stable.
23:00  UTC  Run declared recovered; on-call handed off.
```

The timeline is the anchor. Everything else in the review
references it by timestamp.

### 3. Impact (short paragraph + table)

- Which runs were affected (by `run_id`).
- Which teams were affected.
- GPU-hours lost (measurable from mod-108 chapter 2's goodput
  panel).
- Dollar impact (chapter 5 of mod-109 arithmetic:
  `hours_lost × $/GPU-hour × active_GPUs`).
- Any downstream deadlines missed.

Table format is fine; a paragraph is fine. Numbers must be
cited to the source dashboard or metadata store.

### 4. Detection and diagnosis

*How* the incident was detected and diagnosed. Named metrics,
named alerts, named runbook rungs.

- Which alert fired first, and against which threshold.
- Which mod-108 chapter 6 signature the pattern matched.
- Which mod-106 chapter 5 runbook rung the on-call executed.
- Which diagnostic step ultimately identified the cause.
- Which diagnostic steps did *not* help (equally important).

This section is what improves the runbook.

### 5. Root cause (multi-cause)

The training-run incident review does not seek "the" root
cause; it enumerates *contributing causes* along the
mechanism-detection-response axis. The canonical five classes:

- **Hardware.** GPU / NIC / node hardware failure. Xid events,
  ECC-DBE, NCCL transport errors. BLOOM logbook §3 is the
  reference set.
- **Data.** Bad shard, corrupted tokenisation, dataset drift.
  OPT-175B's loss-spike investigations touch this class.
- **Software / recipe.** Framework bug, optimiser
  misconfiguration, precision policy interaction (mod-107
  chapter 3). OPT-175B chronicles reference many.
- **Infrastructure.** Scheduler, storage, network partition,
  cluster-wide event outside a single run. Chapter 2 of this
  module.
- **Operator / process.** A change made outside the RFC / RFC-
  window process (chapter 3), an admission invariant bypass, a
  runbook step skipped. Named honestly.

The mature review lists *all* contributing causes across
mechanism (why did it happen?), detection (why didn't we catch
it sooner?), and response (why did recovery take as long as it
did?). The five-class taxonomy is a filter across mechanism;
detection and response often have their own contributing
causes in different classes.

### 6. Contributing factors (unstructured list)

Beyond the contributing causes, list environmental factors
that made the incident worse than it needed to be:

- **Runbook gap.** The runbook did not name this pattern; the
  on-call had to derive from first principles.
- **Alert gap.** No alert existed on the signal that would
  have detected earlier.
- **Ownership gap.** The affected artifact (the shard) had no
  owner in the metadata store.
- **Documentation gap.** The relevant knowledge was in
  someone's head, not the platform docs.
- **Overload.** The on-call was concurrently on another
  incident; response was delayed.

These factors do not "cause" the incident but shape its cost.
Name them.

### 7. Action items

Every action item has: (owner, due-date, tracking-id).

Structure by class:

- **Runbook additions.** New rungs, updated thresholds,
  cross-links.
- **Detection improvements.** New alerts, adjusted thresholds,
  new mod-108 chapter 6 signature entries.
- **Prevention.** RFC to change the platform's behaviour (link
  chapter 3), admission invariant additions (chapter 2),
  interface changes (chapter 4).
- **Process changes.** On-call training, escalation-ladder
  tweaks.

Action items without owners and dates are not action items —
they are wishes. Every item is trackable in the platform's
issue tracker; the review links each.

### 8. Sign-off

- Author (typically the on-call, or the incident commander).
- Reviewer (the platform lead or delegate).
- Affected-team representative (chapter 4 named individual).
- Publication date.

## Worked example: OPT-175B-style loss-spike review

An abbreviated example, in the template, for a synthetic
incident modeled on the OPT-175B loss-spike class. This is
*illustrative* — the OPT-175B chronicles themselves are the
primary reference (Zhang et al., 2022;
https://arxiv.org/abs/2205.01068; and the logbook at
https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf).

```
# Incident Review: FOUND-2026-08-03 loss-spike at step 47800

Severity:  Sev-1
Duration:  detect 14:03 UTC → mitigate 20:00 UTC → resolve
           23:00 UTC (9h)
Impact:    FOUND-2026-08-03, foundations team;
           ~4,608 H100-hours lost (512 GPUs × 9 h)
Class:     Data (mechanism); Detection gap; Runbook gap

## 1. Summary
FOUND-2026-08-03 loss-spiked at step 47800 from 2.14 → 3.72
and did not recover across three rollbacks. Root cause traced
to a UTF-8-corrupt shard ingested from a bad crawl on
2026-07-30. Shard removed from manifest; run resumed. Lost
~4,608 H100-hours (~$18K at current per-hour reserved rate).

## 2. Timeline
(as above)

## 3. Impact
See table above. Foundations Q4 deadline is unaffected;
timeline slip within tolerance.

## 4. Detection and diagnosis
- Detection: L-spike alert fired at 14:03 within one step of
  the spike (mod-108 chapter 6 signature 2).
- Diagnosis: runbook rung 3 (rollback and replay) was
  executed correctly but did not distinguish
  data-deterministic from stochastic spikes. It took a
  second replay before the on-call suspected data.
- What did not help: DCGM ECC panels (no hardware event),
  NCCL warning log (no comm issue).
- What identified the cause: data-team on-call cross-
  referenced the sampler's token offset against the last-
  ingested shard, spotting the 2026-07-30 crawl.

## 5. Root cause
- Mechanism: data. UTF-8-decoding-corrupt shard.
- Detection: gap. The data pipeline (mod-103) did not
  detect the decode failure; the trainer downstream
  presented it as a loss spike.
- Response: gap. Runbook did not distinguish
  data-deterministic from stochastic loss spikes; three
  rollbacks were expended before switching to
  data-side investigation.

## 6. Contributing factors
- Runbook gap: no rung for "spike recurs at same offset
  after rollback ⇒ suspect data".
- Alert gap: shard-ingest pipeline had no UTF-8 validation
  alert.
- Ownership gap: the shard's manifest entry had no
  data-team owner label.

## 7. Action items
- Runbook: add rung 3b "spike recurs at same token offset
  after rollback ⇒ suspect data; escalate to data-team
  on-call". Owner: (name). Due 2026-08-12. Track PLAT-1041.
- Detection: shard-ingest UTF-8 validation with alert on
  failure rate > 0. Owner: (name). Due 2026-08-20. Track
  DATA-833.
- Detection: publish "token-offset → shard" query in the
  metadata store (mod-108 chapter 7 addition). Owner:
  (name). Due 2026-08-20. Track PLAT-1042.
- Process: data-team ownership label mandatory in shard
  manifests. RFC-047. Owner: (name). Due 2026-08-31.
- Runbook: cross-link OPT-175B chronicles' loss-spike
  narrative in mod-106 chapter 5. Owner: (name). Due
  2026-08-15.

## 8. Sign-off
- Author: (name), on-call
- Reviewer: (name), platform lead
- Affected team: (name), foundations
- Published: 2026-08-08
```

## Worked example: BLOOM-style hardware-failure review

Sketch of a hardware-failure incident modeled on the BLOOM
logbook's pattern (BigScience,
https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml).
The class is Hardware; response and detection may or may not
have gaps depending on the mod-106 chapter 4 elastic-training
maturity.

```
# Incident Review: FOUND-2026-08-14 node-failure cascade

Severity:  Sev-2
Duration:  02:41 UTC → 04:12 UTC (91 min)
Impact:    FOUND-2026-08-12, foundations team;
           ~192 H100-hours lost (elastic recovery from step
           N-1 successful; 1.5h of GPU-hours across 128
           GPUs)
Class:     Hardware (mechanism)

## 1. Summary
Node ND-047 (H100 SXM5) hit Xid 79 (GPU fell off the bus) at
02:41 UTC during FOUND-2026-08-12 pretraining. Gang
preempted, elastic recovery kicked in (mod-106 chapter 4),
run resumed at 04:12 UTC with 128 GPUs (was 256; node retired
from the pool). Run continued to completion.

## 2. Timeline
02:41 UTC  Xid 79 event on ND-047 (DCGM stream).
02:41 UTC  NCCL timeout on all-reduce; PyTorchJob pod
             restart-loops.
02:42 UTC  Xid alert fires; on-call paged.
02:44 UTC  On-call ack; classifies per mod-106 chapter 5
             runbook rung 1 (hardware).
02:45 UTC  Node ND-047 drained; retired to `broken` pool
             for hardware team.
02:48 UTC  torchrun elastic reconfigure triggered; new
             world size 128.
04:12 UTC  Run resumed from step N-1 checkpoint; loss on
             trajectory.

## 3. Impact
Foundations run schedule unaffected. Retired node
represents 4× H100 GPUs, entering hardware-team RMA queue.

## 4. Detection and diagnosis
- Detection: Xid alert fired within 60 s (mod-108 chapter
  3, DCGM field XID_ERRORS).
- Runbook rung 1 (hardware, Xid-classified) executed
  correctly.
- Elastic reconfigure completed within budget (mod-106
  chapter 4 SLO 15 min; actual 84 min due to storage
  contention during checkpoint read — see contributing
  factor below).

## 5. Root cause
- Mechanism: hardware. Xid 79 signature; NVIDIA's own
  guidance is that the node is likely to recur; retire.
- Detection: adequate.
- Response: contributing gap on elastic-reconfigure
  latency (84 min vs. 15 min SLO).

## 6. Contributing factors
- Checkpoint read contention: two other teams were also
  saving async checkpoints at the same window; storage
  IOPS saturated.
- The `broken` pool label was manually set; automating
  this via mod-106 chapter 5 detection would speed hardware
  team's RMA workflow.

## 7. Action items
- Storage: rate-limit concurrent async checkpoint saves
  cluster-wide (RFC-051). Owner: (name). Due 2026-09-15.
- Automation: auto-label `broken` pool on Xid 79 / 74 / 61
  events (script + mod-108 chapter 3 alerting). Owner:
  (name). Due 2026-08-31.
- Runbook: cross-link BLOOM logbook §3 hardware section
  in mod-106 chapter 5. Owner: (name). Due 2026-08-20.
- Post-mortem: hardware-team RMA feedback loop — publish
  monthly the ratio of RMA'd nodes returned as
  reproducible-fault vs. no-fault-found. Owner: (name).
  Due 2026-Q4 recurring.

## 8. Sign-off
- Author: (name), on-call
- Reviewer: (name), platform lead
- Affected team: (name), foundations
- Published: 2026-08-16
```

## Learning from the two anchors

Read both anchors before authoring your first review — they
teach the *tone* the industry expects.

- The **OPT-175B chronicles** are notable for the honesty of the
  narrative: what was tried, what failed, what recovery cost.
  The tone is descriptive, not defensive. Emulate.
- The **BLOOM logbook** is notable for the routine-ness of
  hardware failures at scale. The takeaway: hardware failures
  are not incidents in the sense of "unusual events"; they are
  the normal operating condition, and the platform's design
  target is elastic recovery within a bounded window, not
  hardware perfection.

Both are open, published, and cited in the field. Your team's
reviews should be discoverable in the same way (internally, at
minimum) and cite these two as the tradition.

## Cadence — the monthly incident-trends review

Beyond per-incident postmortems, the platform lead runs a
monthly trends review:

- **Sev-1 count** (target: 0), **Sev-2 count** (target: < 4/mo
  at a mid-sized platform), **Sev-3 count** (informational).
- **Root-cause distribution** across the five classes
  month-over-month. A hardware-heavy month may drive a
  hardware-refresh RFC; a data-heavy month may drive a
  data-pipeline hardening RFC.
- **Action-item completion rate.** Items overdue are the
  metric that fails first when the platform is under-resourced.
- **Runbook coverage.** How many Sev-2s in the month had a
  runbook rung that matched? Rising number = maturing
  platform.

The monthly review is the aggregate view; the per-incident
reviews are the atoms. Both go into the archive.

## Common failure modes

- **The review that names an individual as the cause.**
  Blameless discipline is violated; future on-calls
  under-report. Rewrite before publication.
- **The review with no action items.** The incident happened;
  nothing changed; it will happen again. If the review
  produced no action items, either the incident was already
  covered by a completed action item (say so and cite it) or
  the review is incomplete.
- **Action items with no owner or date.** Not action items;
  wishes.
- **The review that arrives four weeks late.** The team has
  already moved on; the review is not read; the incident is
  not learned from. Publish on the Sev SLA.
- **The review that treats the runbook as invisible.** The
  runbook is the *point*; every review either exercises a
  runbook rung (and validates it) or reveals a gap (and adds
  one).
- **Reviews stored in Confluence pages that are not indexed.**
  Six months later nobody can find the review. Index in the
  platform docs; number sequentially; publish to a fixed URL.
- **Aggregate trends never reviewed.** The monthly cadence
  slips; nothing rolls up; the review process is per-incident
  reactive rather than pattern-driven.

## Summary

- Every training-run incident above Sev-3 gets a written review
  within a fixed publication window (2 / 5 / 10 business days
  for Sev 1 / 2 / 3). The window is the chapter 1 incident
  contract in action.
- The eight-section template — summary, timeline, impact,
  detection & diagnosis, root cause (across five classes),
  contributing factors, action items, sign-off — is the
  invariant. Every review has all eight.
- Blameless discipline: individuals are named as narrators of
  the timeline, not as causes. The *system* that let the event
  through is the cause.
- Root cause is enumerated across mechanism, detection, and
  response — not "the" cause. The five-class mechanism
  taxonomy (hardware, data, software/recipe, infrastructure,
  operator/process) is the filter.
- Action items have owner + date + tracking-id or they are
  wishes.
- OPT-175B chronicles and BLOOM logbook are the two open
  anchors. Read both; emulate their tone. Your team's reviews
  are discoverable in the same way.
- Monthly incident-trends review aggregates the per-incident
  reviews and drives RFCs (chapter 3) — a hardware-heavy month
  drives a refresh RFC, a data-heavy month drives a pipeline
  RFC.
