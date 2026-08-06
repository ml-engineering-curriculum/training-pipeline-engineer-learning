# exercise-02: torchrun Elastic Reshape Drill

**Estimated effort:** 3 hours

## Objective

Turn chapter 4's elastic-training walkthrough into a live drill: a
`torchrun`-launched training job with `--nnodes=MIN:MAX`, a
rendezvous backend you can point at, and a repeatable "kill a node
mid-epoch, watch the job reshape and resume" script. Deliverable: a
launch spec plus a written incident-timeline report covering the
sequence of events from the moment of the kill to the moment
training resumes. The exercise builds on exercise 1's trainer — you
should not need to write a new model.

## Prerequisites

- Chapters 3 and 4 of this module.
- A completed exercise 1 (a DCP-checkpointing trainer with a
  `Stateful` `AppState`). This exercise is what makes the recovery
  half of chapter 4's flow observable.
- At least 3 nodes' worth of GPUs (ideally 4). You need enough that
  you can lose one and still be above the `MIN` rendezvous
  threshold. If you only have one physical node, you can simulate
  by running two agents against different GPU subsets, but note
  the limitation in the report.
- A working `torchrun` and `torch.distributed.elastic.rendezvous`
  install. Either the built-in `c10d` backend or an accessible
  etcd cluster works.
- `py-spy` (or an equivalent live-Python-stack tool) installed on
  the training nodes for the failure-detection portion.

## Problem statement

An elastic-training platform is only as good as its documented
recovery time. Nobody trusts a runbook that says "should work" —
they trust one that says "on this cluster, on this date, node
kill-at-step-N took T seconds from SIGKILL to first post-resume
step". Your job is to produce that document for the cluster you
have.

You will:

1. Launch the exercise-1 trainer under `torchrun --nnodes=MIN:MAX`
   with a rendezvous backend of your choice.
2. Wait for the trainer to be past a checkpoint boundary and into
   a productive step.
3. Kill one node (SIGKILL the agent, or `kill -9` the workers on
   it — pick one method and document it).
4. Instrument the timeline: `kill`, NCCL timeout, agent
   re-rendezvous, worker relaunch, DCP load, first resumed step.
5. Repeat under three configurations, gathering the timeline for
   each:
   - Default `init_process_group` NCCL timeout (10 min).
   - Aggressive timeout (60 s).
   - Aggressive timeout + a step-time watchdog that fails faster
     than NCCL for a known-fatal signal.
6. Write up the results as a runbook-quality timeline report.

## Requirements

Deliver a single directory containing:

- `launch/` — the `torchrun` launch specs used (as shell scripts or
  a small Python launcher).
- `traces/` — the raw logs from every rank for every run, plus the
  `torchrun` agent logs and the rendezvous backend logs.
- `report.md` — the timeline report described below.

### 1. Launch spec

Publish the exact `torchrun` command line you used, including:

- `--nnodes=MIN:MAX` values and the reasoning.
- `--rdzv-backend` and `--rdzv-endpoint`.
- `--rdzv-id` (or `--rdzv-conf` values you set).
- `--max-restarts` and the reasoning.
- Any `--rdzv-conf timeout=` or `join_timeout=` values you set.

Also publish the `init_process_group(..., timeout=...)` value used
in the training script for each of the three configurations, and
where in the script the watchdog fires for configuration 3.

### 2. Baseline: N-rank steady-state timeline

Before any kill, publish 10 minutes of steady-state training at
world size `N`. Report:

- Mean and P99 step time.
- Number of DCP saves in the window and their mean cost.
- Any rendezvous re-generations that happened during the window
  (should be zero on a healthy setup).

This is your reference; every subsequent timeline is compared
against it.

### 3. The three kill drills

For each of the three configurations:

- **Configuration:** the NCCL timeout, any watchdog specifics, the
  rendezvous parameters.
- **Kill target:** which node, at which step, by which method
  (`kill -9 <torchrun PID>` vs. `kill -9 <worker PID>` vs. host
  `reboot`). Prefer the same target across configurations for a
  clean comparison.
- **Timeline table** with wall-clock offsets from the kill:
  - t=0: SIGKILL sent.
  - t=? : first log line on surviving nodes indicating "something
    is wrong" (a per-step-time watchdog, a `py-spy` dump you
    initiated, or an NCCL timeout).
  - t=? : NCCL timeout fires on surviving workers.
  - t=? : agents detect worker failure and initiate re-rendezvous.
  - t=? : new rendezvous closes; new generation number issued.
  - t=? : new workers launched.
  - t=? : `init_process_group` succeeds on the new world size.
  - t=? : DCP `load` completes.
  - t=? : first post-resume training step.
- **Delta budget:** each line-item's cost in seconds, matched to
  chapter 1's five-component recovery breakdown (detection,
  diagnosis, rendezvous, load, warm-up).
- **Loss trace across the boundary.** Attach the per-step loss
  log; visually confirm the resumed loss picks up near where the
  pre-kill trace left off (some drift is expected due to
  floating-point non-determinism; see exercise 1 report).

### 4. Comparative interpretation

One paragraph per axis:

- **Effect of the NCCL timeout.** How much did going from 10 min
  to 60 s change total recovery time? Was there any downside (any
  false-positive rendezvous during steady-state)?
- **Effect of the watchdog.** Did the watchdog beat NCCL to the
  detection in configuration 3? By how much? Did it introduce any
  false positives?
- **Which configuration would you ship to production, and why?**
  A defensible answer includes the trade-off between fast
  detection and false-positive risk from chapter 4's discussion.

### 5. Failure modes hit and diagnosed

At least three real issues that came up during the drills. Suggested
starting points: a rendezvous that timed out because MIN was too
strict, a DCP checkpoint that could not be loaded because of an
atomic-rename bug leftover from exercise 1, a split-brain during a
network partition simulation, an agent that failed to kill its dead
workers cleanly and left orphan processes. For each: symptom,
diagnosis, fix.

## Starter guidance

- **Do the baseline first.** Do not skip section 2. Without a
  clean steady-state baseline you cannot tell whether a subsequent
  reshape did or did not perturb the healthy behavior.
- **Kill only one node per drill.** The exercise is about the
  single-failure recovery path, not multi-failure. Multi-failure is
  a stretch goal below.
- **Instrument the training script with structured logs at every
  lifecycle boundary.** At `init_process_group` completion, at DCP
  `load` completion, at the first step of the loop. Include
  wall-clock timestamps to millisecond precision.
- **Record the `torchrun` agent logs.** By default the agent logs
  go to the same directory as the worker logs; make sure your
  `--log-dir` (or equivalent) is captured. The agent is the source
  of truth for "when did the rendezvous re-close".
- **For configuration 3's watchdog**, the simplest correct
  implementation is a per-step timing loop that all-reduces a
  timing tensor every K steps and aborts the process if the max
  crosses a threshold. Chapter 6's step-time histogram is what you
  are implementing early.
- **Do not restart between drills without wiping the rendezvous
  state.** With c10d the store lives with the endpoint process;
  with etcd, purge the `--rdzv-id` key. Otherwise stale membership
  can cause a "phantom" ghost rank at the next rendezvous.
- **Cite chapter 4's flow explicitly in the timeline table.** The
  report should be readable by a teammate who has read the
  chapter and can follow along.

## Acceptance criteria

- The launch specs are complete enough to be reproduced by
  another engineer.
- Section 2's baseline includes step-time numbers and confirms
  zero rendezvous re-generations during steady state.
- Sections 3.1, 3.2, and 3.3 each have a full timeline table with
  numeric time offsets and a matched delta budget.
- Section 4 makes a ship-recommendation decision and defends it
  against chapter 4's trade-off.
- The three failure modes in section 5 are real and diagnostic —
  a "no bugs found" section is a red flag.
- Every timeline number is reproducible from the saved traces.
- No invented numbers. If a drill could not run (e.g., only one
  physical node and elastic-simulate had a limitation), the
  report says so.

## Stretch goals

- **Multi-failure drill.** Kill a second node while the reshape
  from the first is in flight. Does the rendezvous handle it? At
  what world size does the second failure drop you below MIN? Add
  the timeline.
- **Split-brain drill.** Partition the rendezvous backend from
  half the compute nodes (`iptables` or a `tc netem` on the
  management interface). Verify that with the c10d backend one
  side "wins" and the other stalls; verify that with etcd the
  Raft quorum determines the outcome. Document what your
  production choice would be.
- **Auto-scale drill.** Configure the scheduler (mod-104 chapter
  4 patterns) to add a fresh node to the pool while the job is
  running. Verify the new node joins on the next rendezvous
  generation (i.e., not until a failure or an explicit
  rendezvous refresh). Discuss how you would trigger a "graceful
  scale-up" that does not require a failure.
- **Nightly-drill automation.** Turn the kill script into a
  nightly cron / CI job that runs against a canary training job
  and alerts if the total recovery time regresses beyond a
  budget you set. This is chapter 7's recovery-path-testing
  pattern.
