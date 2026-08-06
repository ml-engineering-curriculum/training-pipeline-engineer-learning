# Reserved vs. Spot, Dedicated vs. Shared, On-Prem vs. Cloud

Chapter 3 stopped at "the per-GPU-hour rate depends on the lease
tier and the ownership model". This chapter is that trade-off in
detail. It is the negotiation the training-pipeline engineer has
with the capacity team every quarter: cheaper per hour, but with
preemption risk and lower goodput; or more expensive per hour,
but predictable throughput; or a large multi-year capex
commitment with the lowest amortised rate and the highest
downside if the roadmap changes.

The framing to keep: the per-GPU-hour rate is one axis; the
*goodput* the training run achieves on that tier is the other.
A cheap tier that halves your goodput can end up more expensive
than the reserved tier. Do the arithmetic in dollars *per
productive GPU-hour*, not in listed prices.

## The lease tier ladder

Every hyperscaler cloud and every specialty GPU cloud
publishes some subset of the same tier ladder. From most
expensive per hour (and most flexible) to least:

### On-demand

Pay by the hour, cancel at any time, no capacity guarantee. The
listed on-demand rate is the "shelf price" — nearly always the
highest of the tiers. Best for genuinely bursty workloads
(short experiments, one-off evaluations, dev / debug work).

Two operational facts you have to plan around:

- **Capacity is not guaranteed.** During a supply crunch, an
  on-demand `p5.48xlarge` request in a specific AZ can fail
  outright ("InsufficientInstanceCapacity"). For a
  multi-node run, this is the failure mode: 15 of the 16
  nodes come up, one does not, and the run cannot start.
- **The instance itself is not preempted** — once you have it,
  you have it. But if it goes down (host failure), you have
  no priority on replacement.

Chapter 3's per-hour rates for on-demand come from the vendors'
public pricing pages. Cite the URL and the date.

### Reserved instances / commitments (1-year, 3-year)

You commit to a specific SKU and quantity for a specific term.
In exchange for the commitment, the per-hour rate drops
substantially — hyperscalers publish typical discounts of the
order of 40–70% off on-demand for a 3-year all-upfront commit,
lower for 1-year and no-upfront variants. Exact numbers move
frequently; consult:

- AWS: https://aws.amazon.com/ec2/pricing/reserved-instances/
- Azure: https://azure.microsoft.com/pricing/reserved-vm-instances/
- GCP: https://cloud.google.com/compute/docs/instances/reservations-overview

Two subtleties:

- **Commit-vs-consumption model.** Some tiers (AWS Savings Plans,
  GCP Committed Use Discounts) commit to a *dollar spend* per
  hour or per year; others (traditional Reserved Instances)
  commit to a specific SKU. The dollar-spend model is more
  flexible across SKUs but has its own rules about which SKUs
  count.
- **Reserved does not mean physical dedication.** A reserved
  instance still runs on shared physical hardware unless you
  add a dedicated-host option. For training, this rarely
  matters — the SKU-level performance is unaffected — but the
  distinction matters for compliance.

### Capacity blocks / capacity reservations (short-term dedicated)

Newer to the market: bookable, short-duration, guaranteed-
capacity blocks of GPUs. Priced *above* on-demand per hour but
below the on-demand-plus-cancellation-risk expectation. You
book a block of hundreds or thousands of GPUs for hours to
weeks; the provider guarantees availability for the window.

The relevant products:

- **AWS Capacity Blocks for ML** —
  https://aws.amazon.com/ec2/capacityblocks/. Book multi-GPU
  clusters for training in defined windows.
- **NVIDIA DGX Cloud** — https://www.nvidia.com/en-us/data-
  center/dgx-cloud/. NVIDIA-managed, multi-cloud DGX-shape
  clusters, purchased through a commit.
- **Google Cloud Dynamic Workload Scheduler** —
  https://cloud.google.com/blog/products/compute/introducing-
  dynamic-workload-scheduler. Two modes: Flex Start (best-
  effort with queueing, at a discount) and Calendar (booked
  in advance, guaranteed).

Best for: training runs that need a specific cluster shape for
a specific window and cannot afford an on-demand capacity
gamble but do not want a 1–3-year commit.

### Spot / preemptible instances

Idle capacity sold at a steep discount — typically 60–90% off
on-demand. The catch: the provider can reclaim the instance
with minutes of notice. For training, this means restarts.

- AWS Spot: https://aws.amazon.com/ec2/spot/
- Azure Spot VMs: https://azure.microsoft.com/pricing/spot-
  advisor/
- GCP Preemptible / Spot VMs: https://cloud.google.com/spot-
  vms

Spot economics is where mod-106 (checkpointing, elastic
training, goodput) intersects mod-109. The dollar comparison
below quantifies it.

### On-prem / owned hardware

Chapter 3's amortisation formula applies. The lowest amortised
per-hour rate at high utilisation; the highest downside if the
utilisation drops or the hardware becomes obsolete. Multi-year
capex commitment.

## The dollars-per-productive-GPU-hour comparison

Naively comparing list prices across tiers is misleading. The
right unit is dollars per *productive* GPU-hour, which folds in
goodput:

```
$_per_productive_GPU_hour = $_per_GPU_hour_listed / goodput
```

Recall (mod-106 chapter 8; mod-109 chapter 2):

```
goodput = productive_wall_clock / total_wall_clock
```

Public anchors for goodput on different tiers:

- **Reserved / on-prem, mature team, well-tuned run**: 0.85–0.95.
- **On-demand from a hyperscaler, no unusual issues**: similar
  to reserved (the instance itself is stable).
- **Capacity Block for ML for a bounded window**: similar to
  reserved.
- **Spot / preemptible with restart-per-preemption**: 0.30–0.60
  depending on preemption frequency and your restart tail
  time.
- **Spot / preemptible with mod-106 chapter 4 elastic-training
  reshape** (survive a node loss without a full restart): can
  push back into the 0.70–0.85 range, but only if the elastic
  path is well-tested.

Example: suppose the *listed* rates are

- Reserved 3-year H100: `$2.00/hr`
- On-demand H100: `$4.00/hr`
- Spot H100 (typical steady-state discount): `$1.20/hr`

Naive comparison suggests spot is 40% cheaper than reserved. But
if spot goodput on this workload is 0.50 (a plausible number for
a non-elastic run at typical preemption rates) and reserved
goodput is 0.90:

```
Reserved: $2.00 / 0.90 ≈ $2.22 per productive GPU-hour
On-demand: $4.00 / 0.90 ≈ $4.44 per productive GPU-hour
Spot: $1.20 / 0.50 = $2.40 per productive GPU-hour
```

Spot is *more* expensive than reserved for this run — and *only*
because of goodput. With mod-106 elastic training pushing spot
goodput to 0.80, spot drops to `$1.20 / 0.80 = $1.50` per
productive GPU-hour, which is now the cheapest tier. This is
why chapter 5 quantifies the dollar value of each MFU or
goodput point — the multipliers are large.

**The rule**: never publish a tier comparison in list-price
dollars. Always publish in dollars per productive GPU-hour, with
the assumed goodput per tier called out in the table.

## The four tier decisions to make per training run

### 1. Big, planned, single run (main pretraining)

The single-run canonical decision:

- **Duration is known** (days to months).
- **Cluster shape is known** (hundreds to thousands of GPUs).
- **Preemption tolerance is low** — a restart of a 30-day run
  loses hours to days of throughput.

Best fit: **Capacity Blocks for ML** or **reserved capacity**
if the organisation has the commit budget. On-prem if the
programme is stable and the fleet is otherwise utilised.
Avoid spot unless the elastic-training story (mod-106 chapter 4)
is well-tested for this workload.

### 2. Continuous experimental workload

- **Duration is indefinite** — the team is running dozens of
  small-medium experiments a week.
- **Cluster shape varies** — 8 GPUs one day, 64 the next.
- **Preemption tolerance is moderate** — an experiment restart
  is annoying but survivable.

Best fit: a **reserved baseline** sized to the median week,
plus **on-demand** or **spot** for the bursts above baseline.
The reserved commit is sized to the *quiet quarter*, not the
peak — over-commitment on reserved is wasted spend.

### 3. Occasional very large run (frontier attempt)

- **Duration is weeks to months.**
- **Cluster shape is very large** (multi-thousand GPUs).
- **Preemption tolerance is low.**
- **Return is high but uncertain** — one-off strategic project.

Best fit: **DGX Cloud** or a **capacity block** booked in
advance, or **on-prem via a SuperPOD-shape purchase** if the
organisation is building a permanent frontier capability. Spot
is not viable at this shape — the preemption exposure
compounds.

### 4. Recovery / restart / evaluation workload

- **Duration is short** (hours).
- **Cluster shape is smaller than the main run.**
- **Preemption tolerance is high** — a preempted evaluation
  can just re-run.

Best fit: **spot / preemptible** with a simple retry loop. The
cheapest tier is genuinely the cheapest here because the
preemption penalty is small.

## Dedicated vs. shared

Independent of the lease-tier axis: within a given ownership
(on-prem or reserved-cloud), you can either dedicate the
cluster to a single workload or share it across many tenants
via a scheduler.

### Dedicated

- Predictable wall-clock: no queue delay, no priority
  preemption.
- Idle time is your cost. If the run does not use all of `G`
  every hour, the unused GPU-hours are still on your bill /
  amortisation.
- Fits: a single large run that dominates the fleet.

### Shared / multi-tenant

- A scheduler (SLURM, Kubernetes with Kueue / Volcano /
  KubeRay / MPI Operator — see mod-104) packs multiple tenants
  onto the same fleet.
- Idle time is filled with other tenants' work; the *fleet*
  utilisation goes up, which lowers the per-GPU-hour cost for
  each tenant.
- Queue delay and priority contention are new costs. A
  workload that needs a 512-GPU block might wait hours for it
  to be free.
- Fits: a research organisation with many small–medium
  workloads and a few large ones.

The trade-off, precisely:

```
dedicated:  100% availability, ~50–70% utilisation typical
shared:     variable availability (queue), 80–95% utilisation
```

The GPU-hours a shared cluster delivers per dollar are higher —
but a specific run in that cluster spends part of its wall-
clock in the queue. Chapter 5 shows how to translate queue
delay into dollars for a run whose schedule matters (product
launch, publication deadline).

### The multi-tenancy contract

If the cluster is shared, the scheduler policy (mod-104
chapters 5–6) becomes part of the cost model. Two questions
your feasibility study has to answer explicitly:

- **What priority tier does this run get?** High priority → less
  queue delay, less preemption risk; low priority → cheaper
  per hour in the internal cost-allocation model but more
  wait time.
- **What preemption policy applies?** If a higher-priority
  workload arrives, does this run get preempted? How is
  its work re-billed?

These are policy questions owned by the platform team
(mod-110), but the cost study inherits their answers.

## On-prem vs. cloud, in one decision list

The question that every training-platform org revisits every
year. The concrete inputs:

- **Cost of capital** — organisation-specific; determines how
  aggressive the amortisation window can be.
- **Utilisation forecast** — the fraction of the fleet that
  will be productively occupied. Below 70% and on-prem
  economics get shaky.
- **Roadmap stability** — the probability that the current-
  generation hardware is still competitive in 3 years. If a
  new generation halves per-flop cost every 2 years (Hopper
  → Blackwell → Rubin), a locked-in 5-year on-prem commit
  can be a strategic mistake.
- **Team scale** — operating a large on-prem fleet requires
  a real platform team (mod-104, mod-105, mod-110). A team
  under ~5 platform engineers is usually better off leasing.
- **Data gravity** — if the training data is already in one
  cloud, moving it out is an egress cost. If the data is on-
  prem, moving it to the cloud is an ingress cost (usually
  free) plus storage cost.
- **Compliance** — some workloads (health, finance, sovereign)
  must run on hardware with specific residency guarantees;
  this narrows or eliminates the cloud option.

The pattern that most established training organisations
converge on (as of 2026):

- **A baseline on-prem or long-committed reserved fleet** sized
  to the "known year-round" workload.
- **Cloud-reserved or Capacity Block** for the *planned* large
  runs above baseline.
- **Cloud on-demand or spot** for the *unplanned* bursts and
  the low-priority workloads.

The specific proportions vary. The feasibility study should
name which portion of the run falls into which tier.

## Failure modes

- **Comparing list prices instead of dollars per productive
  GPU-hour.** Systematically favors spot on paper and reserved
  in practice. Fix by putting the goodput assumption in the
  same table.
- **Assuming spot goodput without measuring it.** Spot goodput
  varies by AZ, by SKU, by time of year, and by workload. A
  "0.80 goodput on spot" claim without measurement is
  speculation. Fix by running an instrumented pilot before
  committing.
- **Over-committing on reserved.** Reserved is a *use-it-or-
  lose-it* obligation. Commit to the *quiet quarter*
  utilisation, not the peak. Excess bursts go on on-demand or
  spot.
- **Under-counting the operational cost of on-prem.** A DGX
  cluster does not run itself. Include the platform-team cost
  in the amortisation, or the on-prem number is fictitious.
- **Not modeling queue delay on a shared cluster.** If a
  product launch depends on the run completing by date D, and
  the run needs an 8-hour block that averages 4 hours of
  queue, the wall-clock overrun is a real cost — potentially
  a bigger one than the compute itself.
- **Ignoring egress in a cross-cloud recipe.** Training in one
  cloud on data stored in another can easily add 10–20% to
  the total bill via egress. Storage in the same region as
  compute is nearly always cheaper.
- **Locking into a multi-year on-prem commit right before a
  generation transition.** The economics of Hopper systems
  purchased in 2026 look different in 2027 with Blackwell
  widely available. Match the amortisation window to the
  competitive lifetime, not the purchase cost recovery.

## Summary

- The lease tier ladder — on-demand, reserved, capacity block,
  spot, on-prem — differs in list price by a factor of 3–5×.
  In dollars per *productive* GPU-hour, the differences shrink
  once goodput is folded in.
- The rule: compare tiers in `$_per_GPU_hour_listed / goodput`,
  not in the vendor's shelf rate.
- Spot is *usually* the cheapest tier only when mod-106
  chapter 4's elastic-training story is well-tested; otherwise
  the goodput penalty erases the discount.
- Dedicated vs. shared trades wall-clock predictability against
  fleet utilisation. Sharing lowers per-tenant cost; queue
  delay is a new expense the feasibility study must model.
- On-prem is a multi-year commitment that pays off at high
  utilisation and long roadmap stability. Cloud reserved sits
  in between; cloud on-demand / spot is for burst and
  experimental workloads. A mixed portfolio is the typical
  answer.
- The training-pipeline engineer owns the *arithmetic*; the
  platform-team owns the *policy* (mod-104, mod-110). The
  feasibility study names both.
