# exercise-05: Run-Time Signature Catalog

**Estimated effort:** 3 hours

## Objective

Turn chapter 6 into a *drillable* pattern library. For each of the
five canonical signatures — divergence, loss spike, throughput
cliff, straggler, silent corruption — produce a reproducible
synthetic incident, the disambiguation query, the shape of the
curve on the dashboard, and a runbook-linked catalog entry
on-call can actually use. Deliverable: a signature catalog
document, an incident-injection harness that reproduces each
signature against the exercise-1 dashboard, and a written
incident-review template.

## Prerequisites

- Chapter 6 of this module — the five signatures and the pattern
  for each (curve shape, correlated metrics, disambiguation
  query, first-response action). Do not start without it.
- Chapter 2 (dashboard) and chapter 3 (DCGM stack) so the
  dashboard rendering the signatures actually exists. Ideally
  you completed exercises 1 and 2 first; if not, at least have
  the dashboards standing.
- Mod-106 chapter 5 (incident classification) and chapter 6
  (straggler / SDC detection) — the runbook the catalog entries
  point into.
- A small training loop with the metrics-emitter and the DCGM
  stack from earlier exercises reachable, plus rank-level control
  so you can inject synthetic incidents.

## Problem statement

Chapter 6's argument is that pattern recognition beats first-
principles debugging at 3 AM: on-call has under a minute to
name the class and reach for the right runbook. That skill only
develops if the on-call has *seen* each signature on the
dashboard before it shows up in production. The pattern library
is the artifact that trains the skill.

Two things fall out of that argument. First, on-call needs a
one-page catalog they carry: shape, correlate, query, runbook.
Second, the platform team needs a reproducible way to *inject*
each signature so the dashboard's rendering of it can be
verified and the on-call can be drilled against it. This
exercise builds both.

## Requirements

Ship one directory:

- `signature_catalog/`
  - `CATALOG.md` — the on-call-facing catalog (the artifact you
    print and hand to a new on-call).
  - `injector/` — a Python package that injects each of the five
    signatures into a running training loop.
    - `injector/__init__.py`
    - `injector/divergence.py`
    - `injector/spike.py`
    - `injector/cliff.py`
    - `injector/straggler.py`
    - `injector/sdc.py`
    - `injector/cli.py` — a `signature-inject` CLI.
  - `dashboards/signature-drill.json` — a Grafana dashboard
    variant with panels aligned to each signature's shape and
    disambiguation query.
  - `runbooks/` — one file per signature with the runbook
    pointer and the specific query text that resolves it.
    - `runbooks/01-divergence.md`
    - `runbooks/02-loss-spike.md`
    - `runbooks/03-throughput-cliff.md`
    - `runbooks/04-straggler.md`
    - `runbooks/05-silent-corruption.md`
  - `screenshots/` — captured screenshots (or exported panel
    PNGs) of the dashboard rendering each injected signature.
    Attach in each catalog entry.
  - `INCIDENT_REVIEW.md` — the incident-review template the team
    uses to add new entries to the catalog.

### 1. The on-call catalog (`CATALOG.md`)

The artifact on-call reads. For each of the five signatures, one
page with:

- **60-second version of the curve shape** — one paragraph and
  a screenshot from `screenshots/`.
- **Correlated metrics** — the co-moving panels and which
  direction they move.
- **Disambiguation query** — the single Prometheus / tracker /
  dashboard query that confirms the class in one call. Include
  the exact query text and the interpretation of a positive
  result.
- **First-response action** — a link into `runbooks/` and the
  pointer into mod-106 chapter 5 or chapter 6.
- **Distinguishing feature vs. adjacent signatures** — e.g., "a
  spike is fast; divergence is slow" or "a straggler shows
  dispersion; a cliff shows median-shift".
- **Common false positives** — the shape that looks like this
  signature but is not (e.g., an LR warm-up boundary can mimic
  a spike; a scheduled checkpoint save can mimic a cliff).

Constraint: each signature fits on one page. If a signature is
running long, cut prose — the on-call at 3 AM will not read
paragraph seven.

Include the summary table from chapter 6's "two-page summary"
at the top so the catalog is skimmable.

### 2. The injection harness (`injector/`)

A Python package the training loop imports:

```python
from injector import Injector

inj = Injector.from_env()  # reads INJECT_SIGNATURE=spike, etc.

for step in range(NUM_STEPS):
    with inj.step(step):
        loss = train_step(...)
```

Or as a standalone CLI (`signature-inject`) that attaches to a
running loop via an out-of-band control channel (Unix socket,
Redis pub/sub, whatever's cheap).

Each signature has a reproducible implementation:

- **`divergence.py`** — over ~500 steps, gradually scale a
  target-layer's gradient scaling so gradient norm rises
  monotonically and loss walks upward. The training loop
  continues; nothing crashes. This is signature 1.
- **`spike.py`** — at a chosen step, multiply the loss (or the
  gradient) by a factor of `5–10×` for one step, then release.
  Optionally: recur every `N` steps to simulate a data-range-
  correlated spike. This is signature 2.
- **`cliff.py`** — starting at a chosen step, add a fixed sleep
  (or a synthetic CUDA workload) to every step so median step
  time doubles and stays doubled. Dispersion should remain
  ~1.0 (cluster-wide, not per-rank). This is signature 3.
- **`straggler.py`** — starting at a chosen step, add the same
  sleep to one designated rank only. Dispersion should climb
  above 1.10–1.20 and the same rank should be the max-time
  rank for every subsequent step. This is signature 4.
- **`sdc.py`** — starting at a chosen step, on one designated
  rank, quietly perturb the loss tensor by a small factor
  (e.g., 0.999×) that does not visibly change the global
  loss but does show up in per-rank loss and eventually in a
  cross-rank weight-hash all-reduce. Requires the per-rank
  loss emission from exercise 3. This is signature 5.

Each injector is idempotent and reversible — a follow-up call
disables it, so drills can chain (inject → observe → resolve →
inject next).

### 3. The signature-drill dashboard (`dashboards/signature-drill.json`)

A Grafana dashboard variant of exercise-1's dashboard with:

- The exercise-1 three-row layout (liveness / efficiency /
  hardware) so the injected incident appears in the same panels
  the on-call would see in production.
- Five extra "focus" panels, one per signature, each showing
  the disambiguation query from `CATALOG.md` for the signature
  and a rendered version of the co-moving correlate.
- A dashboard-level annotation that fires whenever the
  injection harness starts or stops a signature (so the drill
  timeline is visible on the dashboard).

### 4. The runbook entries (`runbooks/`)

One file per signature. Each contains:

- The disambiguation query as executable text.
- The step-by-step first-response actions from chapter 6 /
  mod-106 chapter 5 or 6.
- A specific `bundle-diff` call (from exercise 4) the on-call
  runs when the signature appears after a recent config
  change — chapter 6 named "do not skip the bundle diff" as a
  what-not-to-do; the runbook makes the diff the first step.
- Named links out to mod-106 chapter 5 for classes 1–3 and
  chapter 6 for stragglers and SDC.

### 5. The incident-review template (`INCIDENT_REVIEW.md`)

Chapter 6 called the signature catalog a *living* document —
"every incident that does not match extends or adds a signature".
The template is what turns a real incident into a catalog delta:

- Date, `run_id`, on-call.
- Which signature the on-call named at the time; which
  signature it turned out to be.
- What the disambiguation query returned.
- What the first-response action was; what it should have been.
- What the catalog should say to catch the same incident faster
  next time (a new co-moving metric, a tighter disambiguation
  query, a note in "common false positives").
- A PR against `CATALOG.md` proposing the change.

## Starter guidance

- **Drill against the real dashboard, not a synthetic view.**
  The point of the exercise is to reproduce the shape *the
  on-call sees*, not a bespoke test panel. Use the exercise-1
  dashboard as the primary rendering surface; the signature-
  drill dashboard is an addition, not a replacement.
- **Sub-second injection is a distraction.** Chapter 6's
  signatures develop over minutes to hours. Inject at the
  cadence the real thing takes; a spike is one step, but a
  divergence is 500 steps and a straggler is many minutes of
  sustained dispersion. Fast-drilling a divergence loses the
  shape.
- **Ground each catalog entry in a paper or a documented
  incident.** Chapter 6's references (OPT-175B chronicles,
  Llama 3 herd paper, Dixit et al. SDC paper, PaLM appendix F)
  are the citation set. If you cannot ground a claim, mark
  `<!-- needs-research: ... -->` per the module guardrails.
- **Do not merge classes.** A loss spike that recovers in one
  step and a divergence over 500 steps are different classes
  with different runbooks; treating them as one signature
  hides the disambiguation query that separates them. Chapter
  6 is deliberately five entries; keep it five.
- **Verify each injection by observation, not by "the script
  ran".** Point the dashboard at the loop and confirm the
  panel shape matches the catalog screenshot. If it doesn't,
  either the injector is wrong or the dashboard is missing a
  panel the catalog assumes.
- **Test the runbook by handing it to someone unfamiliar.** If
  a teammate given only `CATALOG.md` + `runbooks/` cannot
  resolve one of your injections against a live loop, the
  catalog is not yet the artifact chapter 6 asks for.

## Acceptance criteria

- `CATALOG.md` contains one page per signature (5 total) with:
  curve-shape paragraph, screenshot from `screenshots/`,
  correlated metrics, exact disambiguation query, runbook link,
  distinguishing feature vs. adjacent signatures, and common
  false positives.
- Each of the five injectors reproduces its signature against a
  running training loop; the dashboard renders the expected
  shape and the injection is annotated on the dashboard
  timeline.
- The dashboard `signature-drill.json` imports cleanly into
  Grafana and renders both the exercise-1 layout and the five
  focus panels.
- Each `runbooks/` file has an executable disambiguation query,
  the first-response steps, the `bundle-diff` command to run,
  and named links out to mod-106 chapters 5 and 6.
- The SDC injector demonstrably surfaces on the per-rank loss
  view *and* on a cross-rank weight-hash all-reduce (the
  cheapest of the two triggers first, per chapter 6).
- `INCIDENT_REVIEW.md` is a usable template — a teammate can
  fill it in for a real incident and produce a
  `CATALOG.md`/`runbooks/` PR from it.
- A dry-run drill against an unfamiliar teammate has them
  correctly name at least four of the five injected signatures
  using only the catalog. Note the result in the writeup.

## Stretch goals

- Author a *sixth* signature you observed but chapter 6 does
  not name. Ground it in a paper or an OSS training logbook.
  Extend the catalog and injector; treat the addition as the
  proof that the catalog is a living document.
- Wire the injection CLI into a chaos-scheduler (Chaos Mesh,
  Litmus, or the equivalent) so drills can be scheduled during
  low-stakes windows. Requires a rendezvous between the
  scheduler and the training loop; sketch the design in a
  `chaos_design.md`.
- Cross-reference each signature entry with the metadata store
  (chapter 7) `incident` table: an incident recorded there
  should link into the catalog entry, and a catalog entry
  should be reachable from any incident of that class.
- Build a "signature classifier" that takes a `run_id` and a
  time window and returns the most likely signature by running
  every disambiguation query and scoring them. This is the on-
  call assistant a mature platform team eventually builds; the
  catalog is the training data. Ship a small version.
- Fold the catalog's `bundle-diff` steps into an automated
  post-incident report generator that pulls the run's
  reproducibility bundle (exercise 4), diffs against the last
  healthy run's bundle, and attaches the report to the
  incident review. The whole chain — signature → runbook →
  bundle-diff → post-mortem — becomes one command.
