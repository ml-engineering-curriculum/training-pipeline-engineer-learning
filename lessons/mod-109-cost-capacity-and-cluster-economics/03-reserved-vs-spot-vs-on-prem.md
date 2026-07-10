# Reserved vs. Spot vs. On-Prem: The Buying Menu

Chapter 2 gave you GPU-hours and a hand-wavy dollar figure. This chapter
is where you turn that dollar figure into a defensible procurement plan.
Every GPU-hour you spend was bought under some contract; the contract
choice moves the total cost by 3–5× and also determines your reliability
posture, your capacity risk, and how much of your engineering time goes
into elasticity.

The chapter is organised as three axes crossed against each other:

- **Term.** On-demand vs. reserved (1–3 year commitment) vs. spot / preemptible.
- **Sharing.** Dedicated (a reserved SuperPOD carved out for one org)
  vs. shared multi-tenant (mod-104's Kueue / Volcano / SLURM
  fair-share).
- **Ownership.** Cloud (rent DGX / EC2 P5 / GCE A3 / Azure ND) vs.
  on-prem (build your own IB fabric).

Every training-run decision picks a point in this 3D space. The rest of
the chapter walks the axes.

## Axis 1 — Term: on-demand, reserved, spot

The three canonical price points on a cloud GPU, in decreasing order:

| Term        | Typical discount vs. on-demand | Preemption risk | Capacity guarantee |
|-------------|--------------------------------|-----------------|--------------------|
| On-demand   | 0% (baseline)                  | none            | best-effort        |
| Reserved 1 yr | ~30–40% off                  | none            | contracted         |
| Reserved 3 yr | ~40–60% off                  | none            | contracted         |
| Spot / preemptible | ~60–80% off             | high (minutes)  | none               |

<!-- needs-research: verify current H100 reserved-instance discounts on
AWS EC2 P5, GCE A3, Azure ND H100 v5, CoreWeave. The 30–60% range is
teaching order-of-magnitude, not a quote -->

The three price points are not just discounts on the same thing — they
carry very different operational contracts.

### On-demand: the reference price

On-demand pricing is what you use for a short experiment, a burst run,
or a proof of concept. It costs the most but you can turn it on and off
instantly and you owe nothing after the run ends.

Every dollar figure in chapter 2 was quoted at on-demand ("$2–5/GPU-hr"
range for H100 SXM as of publication). Treat on-demand as the sticker
price; everything else is a discount off it.

<!-- needs-research: verify current on-demand H100 SXM hourly for AWS
EC2 P5 (p5.48xlarge), GCE A3 (a3-highgpu-8g), Azure ND H100 v5, and
CoreWeave. Numbers move quarterly; do not cite without checking -->

### Reserved: buying capacity in exchange for commitment

Reserved instances (RIs), Capacity Reservations, Committed Use Discounts
(GCP), and Reserved Capacity (Azure) are all versions of the same
contract: **you commit to pay for `N` GPUs for `T` months (usually 12,
24, or 36); the vendor gives you a discount and a promise that the
GPUs will be there when you ask.**

Two flavours worth distinguishing:

- **Convertible reserved.** You can swap the reservation across
  instance types (say, from an older GPU generation to a newer one)
  during the term. Smaller discount.
- **Standard / non-convertible.** Locked to a specific instance type.
  Larger discount.

Two questions to answer before signing:

1. **What is your utilisation floor?** If you can commit that the
   reserved fleet will be at least 60–70% utilised over the term, a
   reserved-heavy mix is a strong buy. If your workload is bursty (a
   two-week pretraining every quarter with quiet weeks in between),
   reserved is often *worse* than on-demand because you pay for idle
   GPUs.
2. **What is your capacity risk?** For frontier hardware (H100 in 2023,
   B200 in 2024) capacity is genuinely scarce. Reservations are the
   only way to guarantee you can get GPUs at all. The "discount" is
   partly a discount and partly an insurance premium against not
   being able to book.

At the platform level, reserved capacity is what you carve into
guaranteed quota (mod-104 chapter 8). Bursts above the reservation land
on on-demand or spot.

### Spot / preemptible: cheap capacity with a knife hanging over it

Spot instances (AWS), preemptible VMs (GCP), and low-priority VMs
(Azure) are the cloud's mechanism for selling capacity that would
otherwise be idle. Discount is typically 60–80%; the catch is that
the vendor can reclaim the instance with 30 seconds to 2 minutes of
notice.

This is where mod-106 becomes a hard prerequisite. Spot is only viable
if:

- Your job checkpoints frequently enough that a preemption costs
  seconds-to-minutes of forward progress, not hours.
- Your rendezvous (torchrun elastic, KubeRay autoscaling, MPI
  Operator with a recovery plan) can absorb a node leaving and rejoin
  or reshape without a human in the loop.
- Your data pipeline is deterministic and resumable (mod-103's stateful
  sampler contract).

Without those three, spot for training is malpractice. The 60–80%
discount is real, but the loss of a run that had been going for a week
because the spot bid was pulled is more than 60–80% of your budget.

**A composite pattern that works.** Take your reserved fleet as the
"always on" gang for the critical run. Layer spot on top as an
opportunistic booster for the same job under mod-106's elastic
protocol, so the reserved gang keeps the run alive and the spot layer
adds throughput when it happens to be available. Kueue's
`admissionChecks` + priority classes are the operational mechanism;
mod-104 chapter 8 covers the policy shape.

## Axis 2 — Sharing: dedicated vs. multi-tenant

Independent of term, you have to decide who else uses the fleet.

### Dedicated: your own SuperPOD

A dedicated SuperPOD (or its cloud equivalent — AWS EC2 UltraCluster
reserved, GCP TPU Pod, Azure ND H100 v5 SuperPOD) is a fabric-and-
storage island that belongs to exactly one org. NVIDIA DGX SuperPOD
reference architectures (mod-105 chapter 2) are the canonical shape.

Pros:

- Predictable performance. No noisy-neighbour, no fabric contention.
- Reserved capacity by definition; you can plan against it.
- Custom fabric tuning (NCCL topology, PXN, subnet manager).

Cons:

- You pay for it whether you use it or not.
- Utilisation is your problem. Frontier labs have entire teams whose
  job is to keep the SuperPOD busy 24/7.
- Non-trivial to grow or shrink; scaling means procuring new SUs.

For a shop whose main product is one very large model, dedicated is
almost always the right answer. For a shop whose main product is many
smaller models, shared usually wins.

### Multi-tenant shared

A shared cluster is exactly what mod-104 taught you to run: quotas,
fair-share, gang preemption, priority classes. Multiple teams,
one fabric.

Pros:

- Utilisation is amortised across teams. A 512-GPU cluster that would
  sit idle for one team runs at 80–90% utilisation for four teams.
- Bursty workloads see each other's idle capacity. Team A's finished
  sweep frees GPUs for team B's fine-tune.
- Cheaper per useful GPU-hour when utilisation is high.

Cons:

- Noisy-neighbour: another team's poorly-tuned job can eat fabric
  bandwidth and drag your MFU.
- Preemption is a real risk you have to design for (mod-106).
- Policy is now an engineering problem in its own right (mod-104
  chapters 8 and 9).

The crossover is *utilisation*. If a single team can hold the fleet at
>80% utilisation, dedicated wins on simplicity. If you cannot, shared
wins on economics — and you pay for the sharing with a mod-104-scale
policy stack.

### The pragmatic hybrid

Real orgs run both: a dedicated reserved carve-out for the flagship
pretraining run, plus a shared multi-tenant cluster for research and
fine-tune workloads. mod-110's platform-architecture altitude is where
you design the split; here it is enough to note that the split exists
and both sides carry their own economics.

## Axis 3 — Ownership: cloud vs. on-prem

The most consequential decision, and the one that gets debated for
months.

### Cloud

Renting GPU-hours from AWS, GCP, Azure, CoreWeave, Lambda, or a
neocloud. You get:

- Instant scaling (subject to capacity).
- No CapEx. All spend is OpEx.
- No power / cooling / datacenter / IB technician overhead.
- Vendor-managed fabric and storage (with some tuning access).

You give up:

- The margin the vendor takes.
- Long-term cost predictability (prices move; renewals move; regions
  fill).
- Some fabric visibility. You can tune NCCL, but you cannot rewire the
  spine.

Cloud is the right answer if:

- You are pre-product and cannot forecast usage 2 years out.
- Your workload is bursty enough that a dedicated on-prem cluster would
  sit idle.
- You do not have a datacenter team.
- Frontier hardware is your bottleneck — cloud is often the *only* place
  H100 was available in 2023, and B200 in 2024.

### On-prem

Buying and operating your own DGX SuperPOD or equivalent. You get:

- Long-term dollar-per-GPU-hour that beats any cloud reservation
  once utilisation is high enough.
- Full fabric control. Your own InfiniBand spine, your own subnet
  manager, your own storage tier (mod-105).
- Data locality. Your corpus does not egress out of a cloud region.

You give up:

- 12–24 months of lead time for a serious build.
- CapEx that dominates the first two years of your budget.
- A DC-ops team you did not have before.
- Immobility. Once bought, you have this fleet whether or not H100 is
  the right chip in 24 months.

### The crossover

The rough economic model:

    total_cloud_3yr = GPU_hours_per_year × $/GPU-hr_reserved × 3
    total_onprem_3yr = CapEx + 3 × OpEx_per_year

`CapEx` for on-prem includes: GPUs + servers + IB fabric + storage +
datacenter buildout + spares. Order of magnitude, an H100 SuperPOD SU
(32 DGX H100 nodes = 256 GPUs) runs a few $10–20 M in CapEx before
opex.

<!-- needs-research: verify current CapEx for a 256-GPU H100 SuperPOD SU
including servers, IB spine, and storage; SemiAnalysis and CoreWeave
whitepapers publish estimates that move quarterly -->

`OpEx` is power (H100 SXM is ~700W per GPU, plus cooling overhead),
datacenter rent, and a small ops team.

A widely-cited working figure: **on-prem crosses over a 3-year reserved
cloud contract at around ~$5–10 M CapEx break-even for a 256-GPU H100
cluster.**

<!-- needs-research: verify current break-even from SemiAnalysis and
CoreWeave whitepapers; the crossover moves as cloud reserved rates and
CapEx costs change quarterly -->

The subtlety: this is a break-even at 100% utilisation. If you can only
keep the fleet at 60% utilised, the on-prem number effectively grows by
`1 / 0.6 = 1.67×` on a per-useful-GPU-hour basis. That is what pushes
most product companies to cloud until their sustained utilisation is
high.

The additional subtlety: **the strategic value of on-prem is not just
dollar cost.** It is (a) capacity guarantee independent of the cloud's
own capacity crunch, (b) data locality for sensitive corpora, and (c)
fabric-level control. All three can be worth a per-GPU-hour premium.

### Neoclouds: the middle ground

CoreWeave, Lambda, Together, Run.ai (pre-acquisition), and the other
GPU-specialist clouds are a middle ground between hyperscaler cloud
and on-prem. They typically offer:

- Better dollar-per-GPU-hour than hyperscalers for reserved H100/H200.
- Bare-metal or thinner virtualisation, closer to on-prem performance.
- Faster procurement than hyperscalers when frontier capacity is scarce.

<!-- needs-research: verify current CoreWeave / Lambda / Together
published H100 pricing tiers and note the vendor URLs -->

For a training-platform team that does not want the on-prem overhead but
wants better economics than a hyperscaler, a neocloud reserved contract
often wins. The trade-off is a smaller vendor with less regional
coverage and less mature IAM / networking integration with the rest of
your cloud footprint.

## Composing the axes

The three axes multiply. A workable framework:

**Step 1. Baseload vs. burst.** Estimate the number of GPU-hours you
will run in a steady month for the next 12–24 months. Baseload is what
you buy reserved. Burst is what you buy on-demand or spot.

**Step 2. Reservation split.** Of the baseload, what fraction should be
reserved for term-length savings? The classic answer is 60–80%
reserved + 20–40% on-demand / spot for burst.

**Step 3. Sharing.** Of the reserved capacity, how much is dedicated to
a flagship run vs. multi-tenant? mod-104 policy determines who sees the
multi-tenant slice.

**Step 4. Ownership.** Above a stable utilisation floor (roughly:
sustained 60% utilisation on hundreds of GPUs for 24 months), evaluate
on-prem or neocloud. Below it, cloud reserved is the pragmatic answer.

A concrete example. A product company running a quarterly fine-tune of a
13B model plus continuous research sweeps:

- Baseload: 128 GPUs sustained (research + fine-tune).
- Burst: up to 512 GPUs during the two-week fine-tune window each
  quarter.
- Reserved: 128 GPUs on a 1-year commitment (cloud reserved or
  neocloud reserved). Sized to baseload.
- On-demand / spot: burst up to 512 during fine-tune windows. Spot
  only after mod-106's elastic protocol is proven.
- Dedicated vs. shared: shared for research, dedicated carve-out for
  the fine-tune (via mod-104 quotas).
- Ownership: cloud reserved (utilisation floor too low for on-prem).

A different example. A frontier lab targeting continuous pretraining
at 4096 GPUs:

- Baseload: 4096 GPUs at ~90% utilisation, 24 months out.
- Reserved: 100% reserved (any less and you cannot even book the
  capacity).
- Dedicated vs. shared: dedicated SuperPOD. Multi-tenant across
  research and pretraining teams via mod-104 policy inside the same
  fabric.
- Ownership: on-prem or a hyperscaler multi-year strategic contract.
  At this scale and utilisation, the on-prem CapEx amortises and the
  strategic value of fabric control is material.

## Interaction with mod-106: spot only works if elasticity is real

Repeat, because it matters: **the spot discount is meaningless if a
preemption destroys the run.** Every dollar in the "spot" column of your
plan is conditional on:

- Async DCP checkpointing with a target checkpoint interval that limits
  loss-of-progress on preemption to minutes (mod-106 chapter 3).
- torchrun rendezvous or KubeRay autoscaling that can absorb a node
  leaving without a human (mod-106 chapter 5).
- Deterministic, resumable data pipeline (mod-103 chapter 5).
- Reliability posture with an availability budget that treats spot
  preemption as an expected event (mod-106 chapter 8).

If any of those is not real, cross the spot column out of your plan.
It is not that spot is bad — it is that spot without elasticity is a
promise of engineering debt that will be paid in a middle-of-the-night
incident.

## Common failure modes

- **Buying reserved before you know your baseload.** A 3-year reserved
  contract locked in during a hype cycle is a very expensive way to
  discover that your team's actual utilisation is 40%.
- **Optimising on discount instead of dollar-per-useful-GPU-hour.** A
  60% discount at 40% utilisation is worse than a 30% discount at 90%
  utilisation.
- **Not budgeting egress.** Cloud egress costs on the training corpus
  and checkpoint bucket are a real line item, especially in cross-cloud
  or hybrid setups. They do not show up in "$/GPU-hr".
- **Not budgeting fabric capacity.** A cloud vendor may sell you 512
  H100s but the fabric may not be co-located in one rail-optimised
  fat-tree island. If your NCCL topology has to hairpin across a
  broader network, MFU sags — and the sag is invisible in the sticker
  price. mod-105 has the diagnostic.
- **Cross-region reserved for capacity you cannot get anywhere.** Some
  reserved capacity is *guaranteed* by contract in a region that does
  not yet have the physical hardware. Read the fine print.

## Summary

- Cloud GPU capacity comes in three price tiers: on-demand (baseline,
  most flexible), reserved (30–60% discount for a 1–3 year commitment
  and capacity guarantee), and spot / preemptible (60–80% discount with
  minutes-of-notice preemption). Order-of-magnitude discounts; verify
  current vendor quotes.
- Dedicated (own SuperPOD) vs. multi-tenant (shared cluster under
  mod-104 policy) is orthogonal to term. Dedicated wins on simplicity
  and performance when utilisation is high; shared wins on economics
  when it is not.
- On-prem crosses over a 3-year reserved cloud contract at roughly
  $5–10 M CapEx break-even for a 256-GPU H100 SU, and only at
  sustained high utilisation. Below that, cloud is the pragmatic
  answer. Neocloud reserved is the middle ground.
- Spot capacity is only viable if mod-106's DCP + torchrun-elastic +
  stateful-sampler story is real. Without it, the discount is illusory
  and the operational cost is paid in incidents.
- The right procurement plan is baseload (reserved) + burst (on-demand
  or spot) + strategic (dedicated vs. shared vs. on-prem) chosen against
  a specific utilisation forecast. Chapter 5's feasibility study
  template is where you write this down formally.
