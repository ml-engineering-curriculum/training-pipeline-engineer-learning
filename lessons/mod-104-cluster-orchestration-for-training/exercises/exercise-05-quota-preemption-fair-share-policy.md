# exercise-05: Quota, Preemption, and Fair-Share Policy

**Estimated effort:** 3 hours

## Objective

Design and document a multi-tenant policy for a shared training cluster
using the four levers from chapter 8 — quota, fair-share, priority,
preemption — and implement it end-to-end on either SLURM or
Kubernetes (Kueue + Volcano). Then run a small scenario suite that
demonstrates the policy makes the right decisions under contention.

## Prerequisites

- Chapter 8. Chapters 3, 4 (Kueue + Volcano) if you go the Kubernetes
  route; chapter 2 if you go the SLURM route.
- A cluster you can create tenants / namespaces / accounts on.
- Enough capacity to run at least three simultaneous mock training
  jobs (a `sleep` container per pod is fine; the point is the
  policy, not the model).

## Problem statement

You are the platform owner for a 24-GPU (or 24-mock-GPU) shared
training cluster. Three teams share it:

- `foundations` — deep-pocketed pretraining team, needs a guarantee
  and the highest priority.
- `finetune` — steady day-to-day fine-tuning workload, medium
  priority.
- `research` — bursty experimentation, lowest priority, tolerates
  preemption.

Design the policy and demonstrate it survives the four scenarios in
"Requirements" without a human touching the scheduler.

## Requirements

1. **Policy design (`POLICY.md`).**
   - 400–800 words.
   - A table of tenants × (guarantee, cap, priority tier,
     preemptable-by, preempts-what).
   - A rationale paragraph for the priority ordering, citing
     chapter 8.
   - A "what breaks first" paragraph — when the cluster is fully
     subscribed, which team's job queues, and why.
   - A "what we did not do" paragraph — one or two policies you
     considered and rejected, with reasons.
2. **Implementation (`manifests/` or `slurm/`).**
   - Kubernetes path: ClusterQueues (with cohort borrowing),
     LocalQueues, PriorityClasses, Volcano scheduler config with
     `gang` + `preempt` + `proportion` (or `drf`).
   - SLURM path: `sacctmgr` script for accounts and associations,
     `slurm.conf` snippet for QoS and preemption, `topology.conf`
     if applicable.
3. **Scenario suite (`SCENARIOS.md`).** Run each of these and paste
   evidence (kubectl / squeue output at three time points per
   scenario):
   - **S1: Idle → guarantee.** With the cluster idle, submit
     `finetune`'s guaranteed-size job. It admits within seconds.
   - **S2: Guarantee + borrow.** With `finetune` at guarantee and
     the rest idle, submit `finetune`'s cap-size job. It admits
     the borrowed portion.
   - **S3: Guarantee reclaim.** With `finetune` at cap (guarantee +
     borrow), submit `foundations`'s guarantee-size job. The borrow
     is reclaimed (preempted) from `finetune`; `foundations` starts;
     the preempted `finetune` job re-queues.
   - **S4: Priority preemption within cohort.** With the cluster at
     capacity running mostly `research` jobs, submit a
     `foundations` `training-high` job. A `research` gang is
     preempted; `foundations` starts; `research` re-queues.
4. **Post-run analysis.**
   - For each scenario, one sentence: "policy made the right call
     because __". Cite the relevant chapter-8 lever.
   - One paragraph: what would you monitor to detect the policy
     misbehaving in production (queue times, preemption rate,
     fair-share error)?

## Starter guidance

- For the Kubernetes path, reuse the manifest set from exercise 2 and
  extend it with priority classes and preemption. You do not need to
  rebuild.
- For the SLURM path, `sacctmgr show associations` and `sprio -l` are
  the two commands that will save your debugging time.
- Use "training jobs" that are just `sleep` in a container / bash.
  This exercise is *only* about the scheduler; no model needed.
- For scenario S3, make the borrow duration long enough (a `sleep
  600` container) that you can observe the preemption in real time.
- If your cluster does not preempt cleanly under your first policy,
  do not paper over it — surface the misconfiguration in
  `POLICY.md` and fix.

## Acceptance criteria

- `POLICY.md` has the tenant × lever table and the rationale, and it
  reads like an RFC you would ship to a platform team.
- Every scenario in `SCENARIOS.md` has three observed time points
  (before submission, at preemption/admission, after settle) with
  paste-ins from the actual scheduler.
- The reclaim scenario (S3) demonstrably runs — the previously
  borrowed pods / job are evicted and the reclaimed capacity is used
  by the incoming guarantee.
- The priority-preemption scenario (S4) shows evicted `research`
  pods and a re-queued (not lost) job.
- Every claim in `POLICY.md` about the scheduler's behaviour cites
  either Kueue Preemption docs
  (https://kueue.sigs.k8s.io/docs/concepts/preemption/), Volcano
  scheduling docs (https://volcano.sh/en/docs/), or SLURM's
  Preemption guide (https://slurm.schedmd.com/preempt.html).

## Stretch goals

- Add a `preemptible` tier that runs on spot / burst capacity and
  demonstrate that it is preempted by any of the three named tenants
  at any priority.
- Implement fair-share explicitly: on Kubernetes with Volcano's
  `drf` plugin, or on SLURM with `PriorityWeightFairshare`. Run a
  saturation scenario (all three teams over their guarantee at once)
  and show the fair-share output.
- Turn checkpointing on (mod-106 exercise-style, or a mocked
  checkpoint-and-resume `sleep` script) and show a preempted job
  resuming from its checkpoint on next admission. This is what
  makes preemption invisible to the researcher.
- Instrument a "policy dashboard" — a Grafana or Prometheus panel
  showing per-tenant queue time, per-tenant usage vs. guarantee, and
  preemption rate. Sketch is fine; do not have to be production-
  quality.
