# Build vs. Buy at Level-35 Altitude

At level 35 the build-vs-buy call is not "should we adopt a
library?" — that is a level-25 conversation. It is "should this
organisation own a training platform at all, and if so, how much
of the stack?" The answer commits the org to years of engineering
investment, an on-call rotation, a hiring plan, and a capex or
opex profile that shows up on the CFO's dashboard. It is one of
the highest-leverage decisions a training platform architect
makes.

This chapter is the framework for that call. Four options are on
the table. The framework has seven decision axes. The chapter
ends with two worked examples the exercise-04 doc grades against.

## The four options

The design space collapses into four positions along an ownership
gradient:

**Option A — in-house Megatron-LM fork.** You take the upstream
Megatron-LM tree, fork it, own the merge cost, author your own
launcher SDK on top, run your own scheduler stack, operate your
own storage tier, and integrate your own kernel work. Maximum
control, maximum ownership overhead. This is the shape frontier
labs (OpenAI, Anthropic, Meta AI's foundational-model group) sit
at. Reference: NVIDIA's Megatron-LM
(`github.com/NVIDIA/Megatron-LM`).

**Option B — NeMo integration.** You adopt NVIDIA's NeMo Framework
(Megatron-Core + Lightning wrapper) as the training stack, layer
your own launcher SDK on top, run your own scheduler and storage,
and inherit NVIDIA's recipe library. You still own the platform,
but you do not own the framework internals; when Megatron-Core
changes, you follow. Reference: NVIDIA NeMo Framework docs
(`docs.nvidia.com/nemo-framework/`).

**Option C — Databricks / Mosaic managed platform.** You use
Mosaic AI Model Training on Databricks: they run the cluster, they
run the scheduler, they run the storage, they ship a launcher.
You bring the data and the training recipe. Your ownership drops
to a training-config author's responsibility; the platform is
theirs. Reference: Databricks Mosaic AI product page
(`www.databricks.com/product/machine-learning/mosaic-ai`). <!-- needs-research: confirm exact product URL and specific service naming for the "training" surface within Mosaic AI as of 2026 -->

**Option D — Together AI or other managed pretraining vendor.**
Similar to C but from a specialised vendor whose product is
specifically "we train large models for you." You send data and
recipe; they send back weights + a run report. Ownership is even
lower than C in practice, because the vendor takes on the
end-to-end responsibility including the run reliability. Reference:
Together AI product docs. <!-- needs-research: confirm current Together AI "custom model training" or pretraining-as-a-service product URL and SLA terms -->

The four options are not strictly ordered — the trade is
multi-dimensional — but they roughly follow ownership decreasing
left to right and vendor risk increasing right to left. Every
option has real production users; none of them is wrong in the
abstract.

## The seven decision axes

Every serious build-vs-buy decision doc has a paragraph on each
axis. Skip an axis only if you can name why it does not apply.

### 1. Team size

Below ~8 engineers on the platform team, Option A is basically
untenable. Megatron-LM is not a library you can dabble in; the
merge cost alone (against upstream) is a full-time job for one
engineer, and you need at least three more to cover framework
internals, scheduler operations, and storage. Below ~4 engineers,
Options C and D become the honest choice — a two-person team that
owns a training platform is a two-person team that is not
sleeping.

The rule of thumb by team size:

| Platform team size | Realistic option set        |
|--------------------|-----------------------------|
| 1–3 engineers      | C or D only                 |
| 4–8 engineers      | B, C, or D                  |
| 8–15 engineers     | A, B, C, or D               |
| 15+ engineers      | A or B (C/D would waste headcount) |

These numbers are not laws; they are calibrated against public
signals. Meta's OPT-175B team was in the 15–25 range with the
launcher / infra pieces distributed across a larger org (Zhang et
al., 2022, § acknowledgements). Databricks' Mosaic team runs a
platform for many external customers with a large internal
platform group. A product team with 3 engineers running Option A
in 2026 is doing something else instead of shipping.

### 2. Cluster size

Cluster size shapes what the vendor economics look like.

| Cluster size          | Ownership implication                                       |
|-----------------------|-------------------------------------------------------------|
| < 64 GPUs             | Vendor is almost always cheaper. Option C or D              |
| 64–512 GPUs           | Break-even zone. Option B is common; Option C viable        |
| 512–4 000 GPUs        | Own the platform (B or A). Vendor markup dominates          |
| 4 000+ GPUs           | Option A becomes attractive; you have amortised the cost    |

The mechanism: a managed vendor's markup is roughly constant per
GPU-hour. Your own platform's per-GPU-hour operating cost is
dominated by fixed engineering headcount. At small cluster sizes
the fixed headcount is a large per-GPU-hour cost; at large
cluster sizes it becomes negligible. mod-109 has the actual
model; this chapter uses the qualitative shape.

### 3. Model-training cadence

How often does the org do a "real" training run?

| Cadence                                    | Ownership implication                       |
|--------------------------------------------|---------------------------------------------|
| One-off pretraining, no post-training      | Option D — rent it                          |
| 1–3 pretraining runs per year              | Option C or B                               |
| 3–10 pretraining runs per year             | Option B                                    |
| Continuous pretraining + heavy post-training | Option B or A                             |
| Frontier-lab cadence (multiple runs in-flight) | Option A                                |

Cadence matters more than absolute size, because the fixed cost of
standing up a platform is amortised over runs. A team that does
one 100 M-GPU-hour pretraining run in a year and nothing else
should be renting. A team that does fifty 10 M-GPU-hour fine-tunes
in a year should own.

### 4. Custom-kernel needs

Do you need to author or integrate custom kernels?

- **No custom kernels.** Any option works.
- **Integrating other people's kernels** (FlashAttention v3,
  Transformer Engine, community Triton kernels). Option A, B, or C
  — vendors ship these.
- **Authoring your own kernels** for architecture variants or
  novel research (e.g. a custom MoE routing kernel). Option A or
  B; C/D make custom kernel authoring painful because you do not
  own the container or the launcher.

The `ai-infra-performance-learning` peer track's boundary shows
up here. If that peer track exists and is authoring kernels, you
need Option A or B to integrate their work efficiently. If you do
not have that peer track, kernel authoring is unlikely to be a
real need.

### 5. Regulatory and data-residency constraints

Where do the training data and the trained weights live, and who
has read access?

| Constraint                                     | Ownership implication                       |
|------------------------------------------------|---------------------------------------------|
| No external cloud, air-gapped                  | Option A or B, on-prem                      |
| Regulated industry (health, finance, defence)  | A or B; specific vendors on C/D may qualify with contracts |
| No cross-border transfer of raw data           | A/B on-prem in-region; C/D only if vendor honours residency |
| Public research, permissive licensing          | Any option                                  |
| Sensitive PII in training data                 | A/B strongly preferred; audit trail is easier |

The security peer track's contract (chapter 3, § 5a) is the actual
governing document; this axis is the summary version. If the
security peer's contract requires an audit trail that Option C
cannot provide (because their audit surface is not exposed to
you), Option C is off the table regardless of the economics.

### 6. Cost model — CapEx vs. OpEx

This is the axis your CFO cares about most, and it is the axis
platform architects most consistently get wrong on the first
pass.

- **In-house on-prem (A/B)** — high up-front CapEx (hardware,
  networking, floor space, power provisioning), moderate OpEx
  (engineering headcount, power, cooling, upgrades). Depreciation
  schedule 3–5 years. Rebooking as OpEx via a leaseback is
  possible.
- **In-house in-cloud (A/B on cloud GPUs)** — no CapEx; OpEx
  scales linearly with GPU-hours. Reserved-instance discounts
  possible for planned runs; spot instances possible with elastic
  training (mod-106).
- **Managed vendor (C/D)** — pure OpEx, per-GPU-hour rate with
  a vendor markup, no fleet-ownership overhead. Often minimum
  commitments (reserved capacity) apply.

Naïve dollar-per-GPU-hour comparisons favour Options A/B on-prem
at scale and Options C/D at low scale. The break-even is complex
because CapEx amortisation depends on utilisation, and utilisation
depends on how good you are at gang preemption and fair-share
(chapter 1). A well-run platform at 70 %+ sustained utilisation
beats managed vendor pricing at 500+ GPUs; a poorly-run platform
at 30 % utilisation does not. mod-109 owns the actual model.

### 7. Lock-in risk

What does the exit path look like?

- **Option A.** Your fork is your fork; the lock-in is to your
  own code. Exit is possible by re-merging with upstream, but
  that has a cost.
- **Option B.** Lock-in is to NVIDIA (Megatron-Core + NeMo). Exit
  is a rewrite to FSDP2 + `torchtitan`, which has a cost but is a
  known path.
- **Option C.** Lock-in is to Databricks. Exit means re-writing
  training pipelines and re-hosting the data. Multi-vendor
  strategies (Databricks + own cluster) partially mitigate.
- **Option D.** Lock-in is to the vendor of choice. Exit means
  changing vendors or re-implementing in-house. If the vendor
  goes out of business or changes terms, exit cost is now,
  unplanned.

Lock-in risk is not a reason to reject any option, but it is a
reason to name the exit path in the decision doc. An RFC that
picks Option C without documenting the exit path is missing a
section.

## The decision doc template

Same shape as mod-102 chapter 8's stack-choice template, but with
build-vs-buy axes:

```markdown
# Decision doc: training-platform ownership for [org / team]

## Context
- Org: <size, industry, regulatory posture>
- Training regime: <pretraining vs. post-training, cadence, model scale>
- Cluster: <owned or rented, size, generation, fabric>
- Team: <current platform headcount, hiring plan>
- Downstream teams: <who consumes the platform, how many>

## Recommendation
We recommend **Option [X]** because:
1. [axis-driven reason]
2. [axis-driven reason]
3. [axis-driven reason]

## Options considered
- **Option A (in-house Megatron fork):** [analysis]
- **Option B (NeMo integration):** [analysis]
- **Option C (Databricks / Mosaic managed):** [analysis]
- **Option D (Together AI or other managed pretraining):** [analysis]

## Decision axes
| Axis                    | This org's answer                | Weight |
|-------------------------|----------------------------------|--------|
| Team size               | [n engineers on platform team]   | High   |
| Cluster size            | [n GPUs]                         | High   |
| Training cadence        | [n runs/year, scale]             | Med    |
| Custom kernel needs     | [none / integration / authoring] | Med    |
| Regulatory constraints  | [none / regulated / air-gapped]  | High if present |
| Cost model              | [CapEx tolerated / OpEx only]    | Med    |
| Lock-in risk tolerance  | [named exit path]                | Low unless flagged |

## Cost model
[per-option 3-year TCO estimate, from mod-109]

## Risks and mitigations
- [Risk]: [mitigation]

## Exit criteria
- If [condition] we should reconsider [alternative option]

## Sign-off
- Platform lead
- Finance / cost owner
- Security peer lead
- Consumer teams' leads
```

The document is 5–10 pages when done well. Sub-5 pages usually
means the axes were treated as a checkbox; over 10 usually means
the author was arguing rather than analysing.

## Worked example 1 — product-company platform team

Ten-person platform team at a product company doing 3–5
pretraining runs per year on a 512-H100 cluster.

- **Team size:** 10. All four options open.
- **Cluster size:** 512 H100 GPUs. Break-even zone.
- **Training cadence:** 3–5 pretraining runs/year, plus continuous
  fine-tunes downstream. Own-the-platform territory.
- **Custom kernel needs:** integration only; the org does not have
  a `ai-infra-performance` peer authoring custom kernels.
- **Regulatory:** no specific constraint beyond standard product
  security posture.
- **Cost model:** OpEx-preferred (cloud GPUs on reserved-capacity
  contract).
- **Lock-in risk:** moderate tolerance; the org is comfortable
  with NVIDIA-adjacent lock-in.

Analysis:

- Option A (in-house Megatron fork). Team size supports it, but
  the org has no custom-kernel needs and no "we must own the
  framework internals" requirement. The merge cost is pure
  overhead. Reject.
- Option B (NeMo integration). Team size supports it, cluster
  size supports it, cadence supports it, NeMo's recipe library
  is a good fit for a mixed pretraining + fine-tuning workload.
  Kernel integration comes for free through Megatron-Core.
  Strong candidate.
- Option C (Databricks / Mosaic managed). Cluster size is
  slightly above the break-even zone; the vendor markup on 512
  H100s over three years is real money. Also constrains the
  team's operational learning: level-35 platform engineers
  develop skill by operating the platform, and Option C offloads
  that operation. Reject unless the org's cost model prefers OpEx
  strongly.
- Option D (Together AI or similar). Overkill for a team with a
  cluster already. Reject.

**Recommendation:** Option B (NeMo integration), with an in-house
launcher SDK on top and the platform team owning the scheduler
stack, storage tier, and observability. Exit path to Option A is
possible in 2–3 years if the team grows past 15 and starts
authoring kernels. Exit path to Option C is possible if the org
downsizes the cluster below 200 GPUs.

## Worked example 2 — frontier lab

Twenty-four-person platform team at a frontier lab with 20 000
H100 GPUs and continuous pretraining runs in flight.

- **Team size:** 24. All options open.
- **Cluster size:** 20 000 H100. Well past the vendor break-even.
- **Training cadence:** continuous; multiple pretraining runs
  in-flight simultaneously.
- **Custom kernel needs:** authoring; the lab has an internal
  `ai-infra-performance` group producing bespoke kernels.
- **Regulatory:** internal security posture, no external cloud
  for weights.
- **Cost model:** CapEx tolerated; the cluster is owned.
- **Lock-in risk:** avoid; a frontier lab does not accept vendor
  lock-in on the core training stack.

Analysis:

- Option A (in-house Megatron fork). Team supports it, cluster
  supports it, cadence supports it, custom-kernel needs
  practically require it (integration friction with NeMo's
  recipe layer becomes a bottleneck at this altitude),
  regulatory posture requires it. Recommend.
- Option B (NeMo integration). Would work at smaller scale, but
  the recipe-library abstraction becomes a headwind when the
  team is authoring architectural variants weekly. Reject.
- Options C and D. Rejected on ownership and regulatory grounds.

**Recommendation:** Option A (in-house Megatron-LM fork), with a
full in-house launcher SDK, a bespoke checkpoint service, and a
tight integration with the `ai-infra-performance` peer's kernel
pipeline. Exit path is theoretical (Option B) if the lab shrinks
or the training frontier moves — but at this scale and cadence,
Option A is the honest answer for the next 3–5 years.

## Common failure modes

- **"We picked A because we can."** A frontier-lab-scale decision
  at product-company scale. The team ends up with a Megatron fork
  and three engineers to maintain it, and the fork rots. The
  correct answer is usually B or C.
- **"We picked C because it's fastest to ship."** A managed
  vendor at frontier-lab cadence. The vendor markup on 5 000+
  GPU-hours per week dominates the CFO's dashboard within a
  quarter, and the platform team spends its time managing the
  vendor relationship instead of the platform.
- **"We picked B because everyone picks B."** Sometimes correct,
  but only when you have named the axes and B legitimately wins.
  "Everyone picks B" without that framing is a herd choice.
- **No exit criteria.** A decision without exit criteria is a
  decision that can never be un-made. Publish "revisit if X, Y,
  or Z."
- **Cost model uses a single-year cost.** A three-year TCO with
  reserved-capacity assumptions and depreciation gives a very
  different answer than a naïve first-year cost. Use mod-109's
  model.
- **No security peer sign-off.** Options C and D transfer parts
  of the security surface to a vendor. That transfer requires
  the security peer's explicit acknowledgement (chapter 3, § 5).

## Summary

- Four options span the ownership gradient: in-house Megatron
  fork (A), NeMo integration (B), Databricks / Mosaic managed
  (C), Together / other managed pretraining vendor (D).
- Seven decision axes: team size, cluster size, training cadence,
  custom-kernel needs, regulatory constraints, cost model,
  lock-in risk. Every serious doc has a paragraph on each.
- Rules of thumb: small team → C or D; product company at
  cluster 100–1 000 GPUs → B; frontier lab → A. Override with
  the specifics.
- The decision doc names the recommendation, the options, the
  axes, the cost model, the risks, the exit criteria, and the
  sign-off list. Any missing section is a red flag on review.
- Two worked examples: a 10-engineer product-company team with
  512 H100s ships Option B; a 24-engineer frontier lab with
  20 000 H100s ships Option A.
- The most common failure mode is picking the option that
  matches the team's aspiration rather than its actual size and
  cadence.
