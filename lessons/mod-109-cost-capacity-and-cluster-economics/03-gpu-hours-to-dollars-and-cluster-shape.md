# GPU-Hours to Dollars and the Cluster-Shape Choice

Chapter 2 landed you at a defensible GPU-hours number. This
chapter turns it into three connected outputs the feasibility
study needs: a **dollar cost**, a **cluster shape** (how many
GPUs of what kind in what topology), and a **wall-clock
schedule**. All three come from the same three-way trade — the
GPU-hours are fixed by chapter 2, but you get to pick how they
are laid down in time (fewer GPUs longer, more GPUs shorter) and
who owns the underlying hardware.

The lease pricing is a moving target. Rather than embed dollar
figures that decay within a quarter, this chapter gives the
*form* of the calculation, cites the vendor pricing pages, and
walks through worked examples with round hypothetical numbers.
Fill in the current pricing when you use the templates.

## The dollar arithmetic

For a lease-based (cloud) run:

```
$_run = GPU_hours · $_per_GPU_hour
      = C / (P · μ_sus · 3600) · $_per_GPU_hour
```

For an on-prem run, `$_per_GPU_hour` is not a lease rate but an
amortised cost that includes capex, colo, power, cooling, and
staff. The next section works both.

The first-order sanity check: a doubling of `$_per_GPU_hour`
doubles `$_run`. A doubling of `μ_sus` halves `$_run`. Chapter 5
formalises the second identity into "each MFU point is worth $X
at this run scale".

## The published cluster shapes: 2026 lay of the land

Every hyperscaler cloud sells H100-generation training in a
handful of standard instance shapes. Consult the current pricing
pages for the numbers; the shapes below are the *product SKUs*
you should recognise on sight.

### AWS

- **`p5.48xlarge`** — 8× H100 SXM5 (80 GB), 3.2 Tbps EFA
  networking. The standard AWS H100 training instance. Pricing:
  https://aws.amazon.com/ec2/instance-types/p5/ and
  https://aws.amazon.com/ec2/pricing/on-demand/. On-demand,
  reserved (1-year, 3-year), Savings Plans, and Spot tiers all
  apply — see chapter 4.
- **`p5e.48xlarge`** — 8× H200 (141 GB HBM3e). Same math peak
  as p5, more HBM and more HBM bandwidth. Useful for larger
  models or longer contexts on the same host.
- **`p5en.48xlarge`** — 8× H200 with 3.2 Tbps EFA v3.
- **AWS UltraClusters** — the tightly-coupled multi-thousand-GPU
  topology built out of the same instances. Bookable through
  Capacity Blocks for ML (see chapter 4) or via long-term
  contracts. Reference:
  https://aws.amazon.com/hpc/ultraclusters/.

### Google Cloud

- **`a3-highgpu-8g`** — 8× H100. The current H100 offering as of
  the a3 family. Documentation:
  https://cloud.google.com/compute/docs/gpus and
  https://cloud.google.com/compute/gpus-pricing.
- **`a3-megagpu-8g`** — same 8× H100 shape with higher intra-
  cluster networking bandwidth.
- **`a3-ultragpu-8g`** — 8× H200.
- **`a4-highgpu-8g`** — B200 shape (verify current pricing and
  availability).
- **TPU v5p / v5e / v6e** — Google's TPU alternative. Different
  peak-FLOPs and different networking topology (ICI mesh, not
  Ethernet/InfiniBand). Distinct math from GPU pricing but the
  same *shape* of the chapter's argument applies. Reference:
  https://cloud.google.com/tpu.

### Azure

- **`Standard_ND_H100_v5`** — 8× H100 SXM5 with 400 Gb/s
  InfiniBand per GPU. Documentation and pricing:
  https://learn.microsoft.com/azure/virtual-machines/nd-h100-v5-series
  and https://azure.microsoft.com/pricing/details/virtual-machines/linux/.
- **`Standard_ND_H200_v5`** — 8× H200 equivalent.
- **Azure ND-series** in general is the ML/HPC line.

### GPU-focused specialty providers

- **CoreWeave, Lambda, Nebius, Crusoe, RunPod, Together, and
  the DGX Cloud offering from NVIDIA on multiple hyperscalers.**
  Prices, availability, and lease tiers vary significantly.
  Some (CoreWeave, Lambda, Crusoe) publish on-demand H100 rates
  well below the hyperscaler on-demand tier; some (DGX Cloud)
  bundle NVIDIA software support. Verify current rates on the
  provider's public price sheet.

### On-prem reference systems

- **NVIDIA DGX H100**: 8× H100 SXM5, 640 GB HBM3, NVLink Switch
  fabric internal, 3.2 Tbps InfiniBand external. Publicly-listed
  as of 2024–2026 in the several-hundred-thousand-USD range per
  chassis; consult NVIDIA's DGX product page and your reseller
  quote for the current figure.
- **NVIDIA HGX H100**: the reference-design board without the
  DGX-branded chassis. Purchased by hyperscalers and OEMs
  (Supermicro, Dell, HPE, Lenovo) as the basis of custom
  systems.
- **NVIDIA DGX SuperPOD**: the tightly-integrated multi-node
  design (typically 32-node / 256-GPU building blocks). The
  "buy the whole cluster" version of on-prem.

The specific dollar-per-hour numbers move; the SKU names and
their peak-FLOPs per node do not. Cite the pricing pages
in the feasibility study, don't inline a stale number.

## The on-prem amortisation formula

For a run on hardware your organisation owns (or has leased for
a multi-year term), the per-GPU-hour cost is not a lease rate. It
is the amortisation of the purchase cost plus the operating
overheads, divided over the utilised hours.

The full expression:

```
$_per_GPU_hour = (
    (capex / lifetime_years / 8760)      # hardware amortisation
  + (colo_per_GPU_per_hour)              # rack + datacenter fees
  + (power_kW · $_per_kWh · PUE / G)     # power drawn + cooling
  + (staff_cost_per_year / (G · 8760))   # ops staff amortised
) / utilisation_fraction
```

Term by term, with representative order-of-magnitude anchors
(fill in your own numbers before quoting):

- **`capex`**: purchase price of the node (8-GPU DGX or HGX-based
  system). Reference sizes are in the several-hundred-thousand-
  USD range per 8-GPU node as of 2024–2026; get an OEM quote
  for the current figure.
- **`lifetime_years`**: typical amortisation window for
  accelerator hardware. 3 years is aggressive (matches vendor
  refresh cadence), 5 years is conservative (matches classical
  server depreciation and often fits accounting policy).
- **`8760`**: hours in a year. Multiply by `lifetime_years` for
  the total lease-equivalent hours.
- **`colo_per_GPU_per_hour`**: rack space, cross-connect, and
  facility fees per GPU per hour. Small compared to power and
  hardware; still worth including.
- **`power_kW`**: per-GPU sustained power draw. H100 SXM5 is
  700 W TDP per GPU; add ~30% for the rest of the node (CPU,
  DRAM, networking, PSU losses). Roughly 1 kW per H100 sustained.
- **`PUE`**: power usage effectiveness — datacenter overhead for
  cooling and power distribution. Efficient hyperscaler
  datacenters run PUE ~1.1–1.2; older colo can be 1.5+. Publish
  the PUE you assumed.
- **`utilisation_fraction`**: fraction of the amortisation window
  the GPU is actually running a productive workload (i.e.,
  `goodput` times cluster utilisation). A well-run internal
  cluster is 0.70–0.90; a research cluster with lots of idle
  time can be 0.40 or below. This is often the single biggest
  driver of on-prem cost — the same hardware at 90% utilisation
  is half the per-GPU-hour of the same hardware at 45%.

The last term is why "on-prem is cheaper per hour" is not
automatically true. The reserved cloud instance is more
expensive per calendar hour, but it stops billing when you turn
it off — the on-prem cost accrues whether or not a run is
running.

## Cluster-shape choice: three-way trade

Once the GPU-hours number is fixed (chapter 2), you get to
choose:

- **`G`** — how many GPUs to allocate to the run.
- **`T`** — wall-clock time (equals `GPU_hours / G`, subject to
  the scaling-loss caveat below).
- **Node topology and lease tier** — what SKU, from what
  provider, on what lease.

The three trade-offs to walk through in the feasibility study:

### 1. Wall-clock vs. GPU-hours (the `η(G)` curve)

Doubling `G` halves `T` to first order. But `μ_sus` decays with
`G` via the scaling-efficiency term `η(G)` from chapter 2, so
GPU-hours actually rises. The break-even point where the
scaling loss overwhelms the wall-clock speedup is where you
stop scaling. For dense H100 FSDP2 runs, this often sits in the
1024–4096 GPU band before diminishing returns become sharp; for
tightly-optimized 3D-parallel Megatron runs it can be much
higher (Llama 3's 16K-GPU regime is a documented example).

Publishing rule: quote the wall-clock at *three* cluster sizes,
mark the assumed `η(G)` for each, and let the reviewer see the
knee.

### 2. Dedicated vs. shared cluster

- **Dedicated** — the cluster is yours for the run duration.
  Predictable, easier to reason about, but you pay for idle
  time.
- **Shared / multi-tenant** — the cluster runs several workloads
  under a scheduler (SLURM, Kubernetes with Kueue / Volcano —
  see mod-104). Per-GPU-hour cost is lower because idle time is
  filled with other tenants' work, but priority contention and
  queue delays reduce your run's wall-clock predictability.

The trade is scheduling latency vs. amortised cost. Chapter 4
walks through how the cloud tier translates this into an
instance-lease choice; on-prem, it is a scheduler policy
choice.

### 3. On-prem vs. cloud

Beyond the arithmetic — same `$_per_GPU_hour` computed both
ways — the real difference is *flexibility vs. commitment*:

- **On-prem** is a multi-year capex commitment. You can amortise
  down toward a low `$_per_GPU_hour` if you run the fleet hot
  for years, but you cannot deprovision if the roadmap changes
  or a new GPU generation makes yours obsolete two years in.
  Best when: the training programme is stable, the team is
  large enough to operate the fleet, and utilisation is high.
- **Cloud on-demand** is elastic and has no lock-in. Expensive
  per-hour, no commitment, best for burst / experimental /
  irregular workloads.
- **Cloud reserved (1-year / 3-year commit, capacity blocks,
  savings plans)** sits in between: cheaper than on-demand,
  still elastic within the commit period. Chapter 4 is entirely
  about this tier.

The typical answer for a heavy-serving product team: a mix of
on-prem or long-committed reserved for the *baseline* workload,
and on-demand or spot for *bursts* and *research*. The
feasibility study should name which tier is assumed for which
part of the run.

## A worked example

Continuing the 34 B Chinchilla-optimal recipe (chapter 1) with
`GPU_hours ≈ 97 600` (chapter 2):

Assume — as *placeholders* only — that the current effective
reserved H100 rate is `$3.00` per H100-hour and the on-prem
amortised rate for a healthy internal cluster is `$1.20` per
H100-hour. (These numbers move; substitute the current ones from
the vendor pricing pages and your own capex quote before
quoting to a decision-maker.)

Reserved cloud:

```
$_run = 97 600 · $3.00 ≈ $293 000
```

On-prem (assuming amortisation-window utilisation makes the run
"free" beyond the amortised rate — i.e., the fleet is running
at 80%+ utilisation and this run is one of many that share the
same amortisation):

```
$_run = 97 600 · $1.20 ≈ $117 000
```

Wall-clock at three cluster shapes, assuming `η` from chapter 2:

- 128 H100 nodes (16 × 8-GPU): ~32 days
- 512 H100 nodes: ~8 days at `η = 1.00`, ~9 days at `η = 0.90`
- 2048 H100 nodes: ~2 days at `η = 1.00`, ~2.4 days at
  `η = 0.85`

The feasibility study puts all three rows in a table with the
`η` column explicit, and quotes the two dollar figures as a
*range* rather than a point.

## Cost side-terms that are not GPU-hours

The GPU-hours × per-hour rate is the dominant term but not the
only term. A rigorous feasibility study includes:

- **Storage** — the pretraining data lives somewhere. At 4.76 T
  tokens, tokenized as int32, that is roughly 19 TB before
  compression and shard packing; typical object-storage costs
  are small compared to GPU costs but not zero.
- **Data staging and preprocessing** — mod-103 owns the
  pipeline; the cost of the CPU/storage tier that runs it is
  a line item.
- **Egress** — moving data between regions, between clouds, or
  from a cloud to on-prem, is billed per byte. Large token
  corpora make this material; a 100 TB cross-region egress
  can be several thousand dollars alone.
- **Checkpoint storage** — chapter 5 covers the cadence
  trade-off. Retention policy determines the total; the
  monthly bill scales linearly with retained checkpoint bytes.
- **Evaluation runs** — every checkpoint you evaluate at scale
  is another (smaller) GPU-hours line item.
- **Development and pre-run experiments** — the small
  experiments that shape the recipe consume compute too. A rule
  of thumb: the *dev budget* for a large run is 5–15% of the
  main-run budget. Quote it separately so the reviewer sees the
  full ask.
- **Staff time** — a large training run is a multi-engineer,
  multi-month effort. If the organisation costs internal
  engineering time, this can be tens of percent of the
  hardware line.

The main-run hardware cost is the dollar number that ends up on
the slide. The side terms are what turn "the training run costs
$293 K" into "the training programme costs $370 K over 6 weeks"
in the feasibility study.

## Failure modes

- **Stale price numbers.** Cloud GPU pricing is dropping ~15–30%
  per year as of the H100/B200 generations. A dollar figure
  based on last year's pricing overstates cost significantly.
  Cite the URL and the date you checked.
- **Comparing on-demand to on-prem amortised as if they are the
  same tier.** On-demand cloud is not what "reserved" costs; you
  will overstate cloud cost 2–4× by using the on-demand column.
  Chapter 4 disambiguates.
- **Not counting utilisation on on-prem.** An on-prem cluster
  running at 45% utilisation is roughly 2× the amortised rate
  of the same cluster at 90%. Publish the utilisation number
  behind your on-prem quote.
- **Wall-clock quoted without the scaling-loss correction.** A
  linear-scaling wall-clock at 4 K GPUs is fiction. Fold in
  `η(G)` and quote the range.
- **Egress and storage omitted for cross-region training.** If
  the data lives in region A and the compute is in region B,
  the egress bill can rival the GPU bill for the first run.
- **Bundling dev-budget compute into the main-run number.**
  Makes the main-run cost look higher and leaves the dev budget
  invisible when someone else reviews the recipe.

## Summary

- `$_run = GPU_hours · $_per_GPU_hour`. For lease-based runs,
  the per-hour rate is a vendor price sheet. For on-prem, it
  is an amortisation of capex + colo + power + staff, divided
  by utilisation.
- Cite pricing to the vendor URL and date; do not inline stale
  numbers.
- Cluster shape is a three-way trade: wall-clock (via `η(G)`),
  dedicated vs. shared, and lease tier. The feasibility study
  publishes all three axes explicitly.
- The dominant cost line is GPU-hours × per-hour rate. Side
  terms — storage, staging, egress, checkpoints, evaluation,
  dev budget, staff — sum to another 20–50% of the main line
  on typical large runs.
- Publish a *range*, not a point. A "$293K to $370K" range with
  the drivers named is more useful than a "$317K" point.
- Chapter 4 goes deeper on lease-tier arithmetic (reserved,
  spot, dedicated capacity). Chapter 5 goes deeper on the MFU-
  to-dollars translation that lets you show the goodput impact
  on this line.
