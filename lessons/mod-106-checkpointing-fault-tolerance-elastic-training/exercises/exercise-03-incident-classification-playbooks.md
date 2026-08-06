# exercise-03: Incident Classification Playbooks

**Estimated effort:** 3 hours

## Objective

Turn chapters 5 and 7 into a set of runbook-quality playbooks that a
teammate can actually use on-call. Each of the five incident classes
gets its own page, following the **symptom → tools → likely causes
→ decision → post-mortem note** shape from chapter 5. Deliverables:
five playbook Markdown files plus a translation appendix mapping the
Llama 3 §3.3.2 incident categories onto your platform's response.

This exercise is deliberately writing, not code. The value of the
deliverable is that a new on-call engineer can be handed the folder
and successfully respond to a real incident from it.

## Prerequisites

- Chapters 5 and 7 of this module.
- Chapters 3 and 4 (the recovery mechanisms your decision lines
  will reference).
- Chapter 6 (the detection primitives your tools lines will
  reference), even if you have not yet run exercise 4.
- Access to your platform's existing on-call docs (whatever they
  are, however incomplete). If you do not have any, an equivalent
  "starting from scratch" is acceptable — say so in the
  deliverable's introduction.
- mod-105 chapter 8 open in a tab; every NCCL / fabric tool listed
  here traces back to it.

## Problem statement

Your on-call rotation currently handles training-run incidents
based on institutional knowledge — one senior engineer wrote a wiki
page three years ago, everyone has read it once, and everyone
diagnoses from muscle memory. The pager rings, the on-call figures
it out, the fix works, the ticket closes. But the shape of the
response varies with who is on-call, restart-without-diagnosis
happens more often than it should, and last week two teammates
independently mis-classified an SDC-in-disguise as a NaN incident
and rolled back to a checkpoint that was already corrupted.

Your job is to author the five class-level playbooks that make the
response consistent, and to publish the OPT / Llama 3 findings from
chapter 7 in a translated form your team can consume.

## Requirements

Deliver a directory (`playbooks/`) containing:

- `README.md` — one-page index with the class list, when to page,
  and the shared "before you touch anything" pre-flight checklist.
- `class-1-loss-spike.md`
- `class-2-nan-inf.md`
- `class-3-nccl-timeout.md`
- `class-4-hardware-fault.md`
- `class-5-silent-corruption.md`
- `appendix-a-logbook-translation.md` — Llama 3 §3.3.2 taxonomy
  mapped to your platform's incident classes.
- `appendix-b-recovery-testing.md` — the recovery-path drill
  schedule for the platform (chapter 7's pattern).

### The playbook template

Every class file follows this template exactly. The reason for the
exact-shape requirement is that a runbook is a piece of
infrastructure: consistency in shape is what makes it fast to read
during an incident.

```
# Class N: <Class name>

## When you get here
- Symptom: <one paragraph>
- Severity: <sev-1 / sev-2 / sev-3, with definition>
- Time-to-first-action target: <minutes>

## Pre-flight: what to check in the first 60 seconds
- <5 bullets max, each a runnable command or a place to look>

## Tools you will use
- Command / dashboard / log path — what it tells you
- Command / dashboard / log path — what it tells you
- (more)

## Likely causes, in order of probability
- <cause> — <one line on how to confirm>
- <cause> — <one line on how to confirm>
- (more)

## Decision tree
- If <observation>: <decision, prescriptively stated>
- If <observation>: <decision>
- Otherwise: <escalation path>

## The rollback / recovery command
- The exact command(s) to run, copy-paste-runnable
- The verification step that confirms the recovery worked

## What NOT to do
- <specific anti-pattern for this class>
- <specific anti-pattern for this class>

## Post-mortem note template
- Rank / node / step affected:
- Root cause (one sentence, not "NCCL was slow"):
- Blast radius (steps lost, GPU-hours consumed):
- Detection lag (from first-onset signal to page):
- Follow-up (hardware ticket / code fix / recipe change):
```

Each class page must be filled in for **your** platform — the
specific dashboards, log paths, commands, and escalation names.
Chapter 5's runbook entries are the starting template; the exercise
is to make them concrete.

### The class assignments (from chapter 5)

- **Class 1: loss spike / divergence.** Decision line covers
  keep-going vs. rollback-and-drop-LR.
- **Class 2: NaN / Inf.** Decision line covers immediate-kill and
  the rank/hardware suspicion path.
- **Class 3: NCCL timeout / hang.** Decision line covers "is this a
  dead rank, a slow rank, or a fabric issue" and routes to
  chapter 4's elastic reshape or mod-105 chapter 8's fabric
  runbook.
- **Class 4: hardware fault.** Decision line covers node drain,
  scheduler quarantine, elastic reshape around the drained node.
- **Class 5: silent data corruption.** Decision line covers
  replay-drill triggering and multi-checkpoint rollback.

### The pre-flight checklist

The shared "first 60 seconds" checklist in `README.md`. Suggested
content — every on-call should run these before anything else:

1. Do NOT restart anything until you have snapshotted state (log,
   NCCL debug, DCGM, agent logs).
2. Check the run's logbook (chapter 7's pattern) for a similar
   past incident.
3. Note the current step, the last DCP checkpoint step, and the
   pre-crash loss trajectory.
4. Establish the on-call's incident ownership in the paging tool.

### Appendix A: logbook translation

For each unexpected-interruption category from Llama 3 §3.3.2,
identify:

- Which of your platform's five classes it maps to.
- The specific detector (chapter 6) that would catch it.
- The specific playbook decision line it routes into.
- A rough estimate of the fraction of your platform's own incidents
  it would explain, given your current cluster and workload.

The point of the appendix is to force the platform team to
articulate that Llama 3's incident distribution is not necessarily
your distribution — but every category is a category *you* need a
plan for, whether or not it dominates your rate.

### Appendix B: recovery-path testing

A schedule for the recovery-path drills from chapter 7. At minimum:

- **Weekly:** synthetic kill on a canary training job; verify
  elastic reshape completes within budget.
- **Monthly:** DCP resume compat smoke test on the current
  PyTorch / DCP / model release.
- **Quarterly:** rendezvous split-brain simulation.
- **Per major release:** end-to-end recovery from the newest DCP
  format onto the next-oldest supported version.

Each entry names an owner, a run duration, an alert channel, and
the specific budget the drill's outcome is compared against.

## Starter guidance

- **Do not template-fill.** A generic "check the logs" runbook is
  no better than no runbook. Every command, dashboard link, and
  escalation name has to be specific to your platform.
- **Read the OPT chronicles first.** The tone of the runbook
  should match the pragmatic voice of the OPT log. A runbook that
  reads like a compliance document will not be used at 3 AM.
- **The decision line matters more than the tools line.** A
  reader in an incident state will skim tools; they will read the
  decision. Make it prescriptive: "roll back to `/ckpt/step-X`",
  not "consider rollback".
- **Include chapter 4's flow in class 3 and class 4.** Both routes
  end in elastic reshape; the decision is whether to trigger a
  drain first.
- **Include chapter 6's replay drill in class 5.** SDC playbooks
  without a replay drill are aspirational; a real class-5 response
  requires that drill.
- **Number every command.** In an incident, the on-call runs
  commands in order. Numbering the tools line as "1., 2., 3."
  makes it a script, not a list.

## Acceptance criteria

- All five class playbooks are present, each following the exact
  template shape.
- Every playbook cites specific commands, dashboards, and
  escalation contacts — no `<TODO>` placeholders in the final.
- `README.md` has the shared pre-flight checklist and the
  when-to-page ladder.
- Appendix A maps every Llama 3 §3.3.2 category to a class + a
  detector + a decision line. No category is unmapped.
- Appendix B has a recovery-drill schedule with owners and
  budgets.
- A teammate who has read chapters 5 and 7 but has *not* seen
  your platform can read the playbooks and respond to a
  reasonably-described incident from them alone. Have a teammate
  actually try this and note any gaps in the appendix.
- No invented facts. If you do not have DCGM deployed yet, the
  class 4 playbook says so and names the intended detector for
  when it lands.

## Stretch goals

- **Post-mortem template automation.** Take the post-mortem note
  template at the end of each playbook and turn it into a form
  (a GitHub issue template, a Google Doc, a Notion page) that
  fires when an incident closes. Include an SLO-cost calculation
  that estimates the incident's contribution to chapter 8's
  budget.
- **Playbook drill.** Run a game-day exercise where you inject a
  synthetic failure of each class into a canary job and time how
  long it takes a teammate (who has read the playbooks but not
  seen the specific incident) to respond correctly. Iterate the
  playbooks based on where they slowed down or went off-script.
- **Cross-runbook consistency check.** Read mod-105 chapter 8's
  fabric runbook and your chapter 5-based playbooks together.
  Every entry that appears in both (e.g., NCCL timeout as a
  training-side symptom vs. a fabric-side cause) should have
  matching, non-duplicative content. Where they disagree,
  reconcile.
- **Automated post-mortem harvester.** Grep the last 90 days of
  incidents from your ticket system, categorize each by class,
  compute your platform's actual incident-mix histogram, and
  compare against Llama 3 §3.3.2. Where you differ, that is
  either interesting or a mis-classification opportunity.
