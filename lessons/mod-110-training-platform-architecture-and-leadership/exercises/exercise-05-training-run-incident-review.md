# exercise-05: Training-Run Incident Review

**Estimated effort:** 3 hours

## Objective

Author a full **incident review** for a real, open-report-
anchored training-run outage. Fit the eight-section
postmortem template from chapter 6, respect blameless
discipline, and produce action items with named owners, dates,
and tracking IDs. Deliverable: one Markdown document
(`incident-review-<slug>.md`), one machine-readable action-
item file (`action-items.yaml` or `.csv`), and one addition
to the runbook (`runbook-addition.md`) that closes the loop
back into mod-106 chapter 5.

## Prerequisites

- Chapter 6 (the training-run incident review) of this
  module.
- Chapter 2 (the Sev ladder that classifies the incident).
- Mod-106 chapter 5 (the incident-response runbook the review
  cross-references). Chapter 6's action items land here.
- Mod-108 chapter 3 (metrics), chapter 6 (loss-spike
  signatures), and chapter 7 (metadata store). The review
  cites these by name.
- Read at least **one** of the two anchor logbooks in full
  before starting; skim the other:
  - **OPT-175B chronicles** — Zhang et al. (2022), arXiv
    2205.01068 (https://arxiv.org/abs/2205.01068), and the
    logbook PDF at
    https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf.
  - **BLOOM training logbook** — BigScience, 2022;
    https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml
    (the `chronicles/` and `README.md` files in that
    directory).
  The tone in these two documents is the reference. Read one
  cover-to-cover before writing.

## Problem statement

Pick **one** of the two anchor incident classes and author
the review. Both are open, well-documented outages of
frontier training runs; both are the reference incidents
chapter 6 grounds against.

### Option A — OPT-175B-class loss spike (data / recipe mechanism)

Anchored on the OPT-175B chronicles. The class is a **loss
spike that recurs after rollback**, ultimately traced to a
data or recipe mechanism (a bad shard, a bad optimiser step,
a precision-policy interaction) rather than to hardware.

Ground the review in a specific event from the OPT-175B
chronicles' log — pick one loss-spike entry, cite its
approximate date and step from the chronicles, and re-author
it as if it happened on *your* platform (the platform in
exercise-01). You are not re-writing Meta's postmortem; you
are writing the postmortem your platform's Sev-ladder
requires when the same class of event happens.

### Option B — BLOOM-class hardware-failure cascade

Anchored on the BLOOM logbook. The class is a **node or GPU
hardware failure** that triggers a NCCL timeout, requires
elastic recovery, and either recovers cleanly (Sev-2) or
cascades into a longer outage due to a secondary contributing
factor (Sev-1). The BLOOM logbook documents this class
across dozens of instances during the tr11 pretraining run.

Ground the review in a specific event from the BLOOM logbook
— pick one hardware-failure entry, cite its approximate date
from the chronicles, and re-author it as if it happened on
*your* platform.

Whichever option you pick, the deliverables and acceptance
criteria are identical.

## Requirements

Ship one directory `mod-110-ex05/` with:

- `incident-review-<slug>.md` — the full incident review (the
  eight-section template).
- `action-items.yaml` or `action-items.csv` — machine-
  readable extract of §7 of the review, one row per action.
- `runbook-addition.md` — the new or amended mod-106 chapter
  5 runbook rung(s) the review demands (chapter 6's "the
  review is what improves the runbook" pattern).
- `trends-review-entry.md` — the one-paragraph entry this
  incident would produce in the monthly incident-trends
  review.

### 1. The incident review (`incident-review-<slug>.md`)

Length target: 3–5 pages when rendered. All eight sections
from chapter 6, in order:

1. **Summary (5 lines).** Title (date-prefixed), Severity,
   Duration (detect → mitigate → resolve wall-clock),
   Impact (runs, teams, GPU-hours), Root-cause class (one of
   the five canonical).
2. **Timeline (bullet list, timestamped).** Every event
   in UTC, minute-resolution. Structure: `HH:MM UTC
   what-happened (who / how observed)`. At least 12 entries
   spanning detect → ack → diagnose → mitigate → resolve →
   hand-off. Include at least one entry that names the
   mod-108 chapter 6 signature that fired (option A) or the
   Xid class (option B).
3. **Impact (paragraph + table).** Which runs (`run_id`),
   which teams, GPU-hours lost, dollar impact (mod-109
   chapter 5 arithmetic:
   `hours_lost × $/GPU-hour × active_GPUs`), any downstream
   deadlines missed.
4. **Detection and diagnosis.** Which alert fired first and
   its threshold. Which mod-108 chapter 6 signature (loss
   spike) or mod-106 chapter 5 rung (hardware) matched.
   Which diagnostic step ultimately identified the cause,
   and which diagnostic steps did *not* help.
5. **Root cause (multi-cause).** Enumerate across mechanism,
   detection, and response. Mechanism must map to one of the
   five canonical classes (hardware / data / software-recipe
   / infrastructure / operator-process). Detection and
   response gaps may map to different classes.
6. **Contributing factors (unstructured list).** At least 3:
   pick from runbook gap, alert gap, ownership gap,
   documentation gap, overload, or any other honest
   environmental factor. These are *not* causes; they shape
   the cost.
7. **Action items.** Every item has owner, due-date,
   tracking-id. Grouped by class (runbook / detection /
   prevention / process). At least 5 total; at least one in
   each of runbook, detection, and prevention.
8. **Sign-off.** Author (on-call or incident commander),
   Reviewer (platform lead or delegate), Affected-team
   representative, Publication date. Publication date must
   respect the Sev SLA (Sev-1: 2 business days from
   resolution; Sev-2: 5 business days).

Blameless discipline is enforced: individuals named as
narrators of the timeline, not as causes. Any sentence that
attributes cause to a person fails acceptance and must be
rewritten to attribute cause to the system.

### 2. The action-items file (`action-items.yaml` / `.csv`)

Machine-readable extract of §7. One row per action item.
Required columns/fields:

- `id` — a stable identifier (e.g., `INC-2026-08-05-01`).
- `class` — one of `runbook`, `detection`, `prevention`,
  `process`.
- `description` — one sentence.
- `owner` — role or name (role placeholders are OK).
- `due_date` — ISO calendar date.
- `tracking_id` — a fabricated but plausible tracker ID
  (`PLAT-1041`, `DATA-833`, `INFRA-402`) — the point is
  that every action is trackable.
- `linked_rfc` — RFC number if the action requires an RFC
  under chapter 3; otherwise `null`.

The file must parse (`python -c 'import yaml, csv; ...'` or
equivalent) and have at least one row per class.

### 3. The runbook addition (`runbook-addition.md`)

The review closes the loop by delivering the runbook change
its action items promise. Chapter 6 is explicit: "the review
is what improves the runbook."

Format: the exact Markdown to be inserted into or amended
onto mod-106 chapter 5. Cite the rung number you are adding
(`rung 3b`) or amending (`rung 5 amended`), the trigger
condition, the diagnostic steps in order, the escalation
target, and cross-links to (a) the signature or Xid class
that identifies the pattern, and (b) this incident review.

Length target: half a page. A runbook rung is short, dense,
and immediately actionable at 03:00; write to that reader.

### 4. The trends-review entry (`trends-review-entry.md`)

One paragraph, chapter 6 monthly-trends format. Structure:

- Sev (contributes to the Sev-1 / Sev-2 / Sev-3 count).
- Root-cause class (contributes to the class distribution
  the RFC pipeline reads).
- Whether the incident was covered by an existing runbook
  rung (contributes to the runbook-coverage metric).
- Whether it produced an RFC (contributes to the prevention
  pipeline).

The paragraph is what rolls up into the monthly review; the
per-incident file remains linked but not read by the monthly
reviewers directly. Write for the roll-up reader.

## Starter guidance

- **Read the anchor logbook before writing.** Chapter 6 is
  explicit: "the tone is the reference." A first draft
  written without reading OPT-175B or BLOOM will read as
  defensive or apologetic; those chronicles read as
  descriptive. Emulate.
- **Write the timeline first.** Every other section
  references the timeline by timestamp. If the timeline is
  fabricated as you write §3–§7, the review is not credible.
  Author §2 completely, then write the rest by referencing
  it.
- **Blameless is not passive-voice.** "The check was
  skipped" is passive and lazy; "the admission webhook did
  not require the RFC-linked feasibility-study reference for
  training-high jobs" is blameless and specific. The system
  is named; the individual is not.
- **Root cause is not "the" cause.** Chapter 6 is
  categorical: mechanism, detection, and response each get
  their own contributing cause. A review with a single
  "root cause" line has skipped detection and response and
  is incomplete.
- **Action items with no owner or date are not action
  items — they are wishes.** Chapter 6 says so explicitly.
  If the owner is `TBD`, the item is not real; assign a
  role placeholder (`<Runbook Owner>`, `<Data Team Lead>`)
  and a specific calendar date.
- **The runbook addition is the point.** If §7's action items
  do not deliver `runbook-addition.md`, the review has
  produced a document and nothing else. The runbook is the
  artifact that catches the next occurrence.
- **Cite the anchor's specific entry.** "Modeled on the
  OPT-175B chronicles, log entry from ~2021-12-04 loss
  spike at step ~62 K" is a citation; "inspired by
  OPT-175B" is not. Precise citations are what separate
  training on the real incident from writing fiction.
- **Do not invent Xid numbers, ECC signatures, or CUDA
  errors.** Cite NVIDIA's Xid documentation
  (https://docs.nvidia.com/deploy/xid-errors/index.html) or
  mark `<!-- needs-research: ... -->`. Fabricated hardware
  error codes fail acceptance.

## Acceptance criteria

- All eight sections present in order; no section is TBD.
- Summary is ≤ 5 lines and names all five required fields.
- Timeline has at least 12 timestamped entries in UTC,
  minute resolution, covering detect → ack → diagnose →
  mitigate → resolve → hand-off. Every timestamp is
  internally consistent (mitigate is after ack; resolve
  is after mitigate).
- Impact section states GPU-hours lost with the arithmetic
  exposed and a dollar figure computed via mod-109 chapter
  5's formula.
- Detection-and-diagnosis section names the alert that
  fired, its threshold, the mod-108 chapter 6 signature
  (option A) or Xid class (option B) matched, the runbook
  rung executed, and at least one diagnostic step that did
  not help.
- Root cause section enumerates mechanism, detection, and
  response, each with its own contributing cause. Mechanism
  maps to one of the five canonical classes.
- Contributing-factors list has at least 3 items; each item
  is a system-level factor (runbook gap, alert gap, etc.),
  not an individual.
- Action items: at least 5 total; every item has owner,
  due-date, and tracking-id; at least one item in each of
  runbook, detection, and prevention.
- Sign-off block has author, reviewer, affected-team
  representative, and a publication date that respects the
  Sev SLA (2 business days for Sev-1, 5 for Sev-2).
- `action-items.yaml` / `.csv` parses cleanly and matches
  the review's §7 line-for-line.
- `runbook-addition.md` is short (≤ 1 page), names the rung
  number, cites the trigger, lists diagnostic steps in
  order, and cross-links back to the review.
- `trends-review-entry.md` is one paragraph and names Sev,
  class, runbook coverage, RFC.
- Blameless discipline is respected: no sentence in the
  review attributes cause to a named individual.
- The chosen anchor incident is cited with a specific
  reference (log entry date or step, arXiv paper section,
  or logbook file path).

## Stretch goals

- **The other anchor.** Do the exercise twice — once for the
  OPT-175B class, once for the BLOOM class. The two together
  cover the mechanism space chapter 6's five-class taxonomy
  is built for.
- **The joint postmortem.** Pick an incident that crosses one
  of the peer-track boundaries from chapter 4 (data
  pipeline, ml-platform serving, security). Author the joint
  postmortem: two teams, one review, both sides of the
  boundary named. Reference the exercise-03 escalation
  runbook if you did that exercise.
- **The RFC that comes out.** Take the highest-blast-radius
  action item from §7 and author its RFC (chapter 3, using
  the exercise-02 template). This is the actual closure
  loop chapter 3 predicts: incident → RFC → migration →
  runbook.
- **The monthly trends review.** Take four fabricated
  incidents (this one plus three you invent, distributed
  across the five classes), roll them up into a monthly
  trends review document (chapter 6's monthly cadence
  section). What RFC does the class distribution suggest?
- **Runbook regression test.** Write a small test (Python or
  shell) that a monthly job could run to verify each runbook
  rung named in `runbook-addition.md` still resolves against
  a known synthetic incident. The runbook that is not
  tested rots; catching the rot is a platform-mature
  behaviour.
- **Publication practice.** Publish the review to a real
  destination — a personal blog, an internal wiki, a
  numbered incidents/ directory. The publication step is
  what makes the chapter 1 incident contract real; do it
  once so you have felt the weight of it.
