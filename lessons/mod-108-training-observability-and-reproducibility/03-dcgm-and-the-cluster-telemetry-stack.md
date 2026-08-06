# DCGM and the Cluster Telemetry Stack

Chapter 2 said Row 3 of the dashboard reads from `dcgm-exporter`
scraped by Prometheus. This chapter is the plumbing behind that
sentence: what DCGM is, which of its field IDs matter for a training
run, how the exporter and Prometheus stack fit into a Kubernetes /
Slurm cluster, and the specific gotchas that turn a "we have DCGM"
claim into a "we do not actually have the metrics we thought we did"
incident.

The primary references for everything in this chapter, held in one
place so you do not have to hunt for them:

- **NVIDIA Data Center GPU Manager (DCGM) documentation.**
  https://docs.nvidia.com/datacenter/dcgm/latest/ — field IDs,
  policies, `dcgmi` CLI, group management, host-engine architecture.
  Pin the version you install; field IDs and metric names can
  change between releases.
- **`dcgm-exporter` on GitHub.**
  https://github.com/NVIDIA/dcgm-exporter — the Prometheus exporter
  that wraps DCGM. Ships as a container image; runs as a DaemonSet
  on Kubernetes or as a systemd unit on bare metal.
- **Prometheus documentation.** https://prometheus.io/docs/ —
  scrape configs, recording rules, external labels, storage
  compaction. The canonical reference for the metric store.
- **Grafana documentation.** https://grafana.com/docs/ — dashboards
  as code, provisioning, template variables.
- **NVIDIA GPU Operator (for Kubernetes).**
  https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/
  — the operator that installs the NVIDIA driver, container toolkit,
  and `dcgm-exporter` as a coordinated bundle on a Kubernetes
  cluster.

The chapter walks you from "install DCGM on a node" to "the dashboard
in chapter 2 renders" — the middle contains the design decisions
worth naming.

## What DCGM actually is

DCGM is a userspace daemon (`nv-hostengine`) that runs on each GPU
node. It talks to the NVIDIA driver via NVML and exposes:

- **Continuous field-value telemetry** — a fixed catalog of ~200
  numeric fields per GPU (temperature, clocks, ECC counters,
  utilization, memory, NVLink counters, PCIe counters, throttle
  reasons). Each field has a stable numeric ID (`DCGM_FI_DEV_*`)
  and a documented meaning. The full list is in the DCGM API
  reference.
- **Group-scoped policies** — actions (log, page, run a diagnostic)
  that fire on threshold crossings of specific fields. Rarely used
  in a Prometheus-based stack because alerting moves to
  Alertmanager.
- **Bundled diagnostics** (`dcgmi diag -r <level>`) — the
  pre-canned health tests that stress GPUs, HBM, NVLink, and
  compute. Levels roughly go from "quick sanity check" to "run for
  an hour". Consult the current DCGM docs for the level meanings on
  the version you have installed.
- **Xid event surfacing** — DCGM catches NVIDIA driver Xid errors
  (documented at https://docs.nvidia.com/deploy/xid-errors/) and
  makes them queryable and streamable.

The `dcgmi` CLI is the interactive front-end. `dcgmi discovery -l`
lists GPUs, `dcgmi dmon -e <field_ids>` streams field values,
`dcgmi diag -r <level>` runs a health test.

## The field IDs that matter for a training run

You do not want to scrape all ~200 fields — the exporter emits one
Prometheus time series per (GPU, field) pair, and a `10^4`-GPU
fleet scraping 200 fields at 10-second cadence is `2 × 10^5`
samples/second, which is `10^8` samples in a work-day. Pick a
focused subset. The default set that `dcgm-exporter` ships with is a
sensible starting point, and it is documented in the exporter's
repository under a `dcp-metrics-included.csv` (or similarly named
file — check the current release for the exact filename).

The groups you almost always want:

- **Utilization** — `DCGM_FI_DEV_GPU_UTIL`,
  `DCGM_FI_DEV_MEM_COPY_UTIL`. On-call's first sanity check.
- **Memory** — `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE`,
  `DCGM_FI_DEV_FB_TOTAL`. HBM usage vs. capacity.
- **Temperature and power** — `DCGM_FI_DEV_GPU_TEMP`,
  `DCGM_FI_DEV_MEMORY_TEMP` (where available),
  `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_SLOWDOWN_TEMP`.
- **Clocks and throttling** — `DCGM_FI_DEV_SM_CLOCK`,
  `DCGM_FI_DEV_MEM_CLOCK`, `DCGM_FI_DEV_CLOCKS_EVENT_REASONS`
  (older DCGM releases called this `THROTTLE_REASONS` — check the
  exporter's current metric name for your version).
- **ECC** — `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL`,
  `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL`, and the retired-page /
  row-remap counters (`DCGM_FI_DEV_RETIRED_*` and
  `DCGM_FI_DEV_ROW_REMAP_*` families). Mod-106 chapter 6 lists
  which severity band each one falls into.
- **NVLink** — `DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL`,
  `DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL`, and the
  per-link bandwidth/utilization counters. Rising CRC errors mean
  a link is degrading.
- **PCIe** — `DCGM_FI_DEV_PCIE_REPLAY_COUNTER`. A rising counter is
  the "host–GPU link is unhealthy" signal.
- **Xid events** — surfaced as `DCGM_FI_DEV_XID_ERRORS` or a similar
  field on modern releases. Confirm the exact name for your
  exporter version.

**Do not scrape the sample-level DCP profiling fields (the ones in
the "profiling" group prefixed with `DCGM_FI_PROF_*`) casually.**
They are expensive on the GPU side because they enable the GPU's
profiling hardware. On some hardware/driver combinations, enabling
profiling metrics causes small performance loss. Turn them on
deliberately, per-workload, when the training team asks — not by
default.

## The exporter architecture

`dcgm-exporter` is a small Go binary that:

1. Connects to the local DCGM host engine.
2. Reads a configuration file listing the fields to expose.
3. Polls DCGM at a configurable interval (`--collect-interval`).
4. Serves the current values as Prometheus text-format metrics on
   an HTTP endpoint (default `:9400/metrics`).

Deployment:

- **On Kubernetes**: a `DaemonSet` alongside the NVIDIA GPU Operator
  installs it on every GPU node. The `ServiceMonitor` (if using
  Prometheus Operator) or a static `scrape_config` (if using
  vanilla Prometheus) tells Prometheus to scrape it. The GPU
  Operator install path is documented at
  https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/.
- **On Slurm / bare metal**: a systemd unit on each node, plus a
  Prometheus target list generated by whatever service discovery
  the site uses (Consul, static file, or a Slurm-side integration
  that publishes node membership).

The two labels every scrape needs:

- `Hostname` (or `node`, `instance` — pick one and be consistent).
  The dashboard uses this to attribute a metric to a node.
- `gpu` (the ordinal within the node) and `UUID` (the GPU's
  persistent UUID). The UUID survives reboots and node reshuffles;
  the ordinal does not. Chapter 6's silent-corruption diagnosis
  needs the UUID to correlate this run's ECC events with the same
  physical GPU's history.

Adding `run_id` as a label on DCGM metrics is possible via
Prometheus recording rules or by joining against a separate
"which run is scheduled where" metric. Chapter 7 covers the join;
the DCGM stream itself does not know about runs.

## The Prometheus tier

Prometheus is the metric store. Two decisions matter more than the
rest:

### Retention

Prometheus's on-disk retention is set with `--storage.tsdb.retention.time`
and defaults to 15 days. For DCGM you want more. The reason is
mod-106 chapter 6's silent-corruption diagnosis: you want to be
able to correlate "this GPU produced a NaN today" against "this
GPU's SBE count started climbing six weeks ago".

Two ways to get long retention:

- **A long-term store**: Cortex, Thanos, Mimir, or a managed
  service (Amazon Managed Prometheus, Grafana Cloud, etc.). These
  push chunks to object storage and query them via a compatible
  API. Chapter-3 grade solution for a real cluster.
- **A local Prometheus with a big disk and a longer retention
  flag.** Works for a few thousand GPUs and a few months of data.
  Below the threshold where a long-term store is worth its
  operational cost.

Pick one; document the retention window in the metric-store's
own documentation. Six months is a defensible default for DCGM;
one week is defensible for the per-step trainer metrics (chapter
4 covers the researcher's tracker separately).

### Recording rules

Prometheus recording rules pre-compute expensive expressions into
new time series. Two you almost always want for DCGM:

- A per-node rollup: `sum by (Hostname) (DCGM_FI_DEV_GPU_UTIL)` /
  `count by (Hostname) (up{job="dcgm-exporter"})` gives you the
  average GPU utilization per node without evaluating the sum at
  every dashboard render.
- A per-fleet rollup: the same, without the `by (Hostname)` clause,
  gives the fleet-wide roll-up.

Recording rules also let you smooth noisy signals. `rate(
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL[1h])` as a recording rule turns the
raw counter into an ECC-error-rate time series that alertmanager
can threshold cleanly.

## The Grafana tier

Grafana reads Prometheus and renders. Three patterns worth
standardizing:

### Dashboards as code

The dashboards you build should live in JSON in the same repo the
training code lives in (or a `platform` repo of its own).
Grafana's provisioning API (`grafana.ini` `[dashboards]` block, or
the Grafana Operator on Kubernetes) will re-load them from disk;
avoid the web UI as the primary editor. Chapter 2's
"dashboard is code" claim depends on this.

### Template variables

`$run_id`, `$node`, `$gpu` as template variables at the top of
every dashboard let the same JSON serve every run. Grafana's
`query` variable type queries Prometheus for the current set of
values (e.g., `label_values(up{job="dcgm-exporter"}, Hostname)`)
and populates the dropdown. This is what makes historical-run
diagnosis possible from the same dashboard URL.

### Alerting boundary

Grafana has alerting; so does Prometheus (via Alertmanager). Pick
one — running both leads to duplicate pages. The community default
is Alertmanager for anything the on-call gets paged on and
Grafana's alerting for softer signals that email the owner. The
important part is the choice being explicit.

## The training-loop side of the stack

DCGM covers the hardware. The training loop's per-step metrics
(chapter 2's Row 1) need their own path into Prometheus. Two
mechanisms:

- **`prometheus_client` on rank 0.** The Python
  `prometheus_client` library lets rank 0 expose an HTTP
  `/metrics` endpoint from which Prometheus scrapes. Simple; works
  for scalars where rank 0 is authoritative (loss, LR).
- **Push mode via the Prometheus Pushgateway.** For metrics rank 0
  emits at a cadence Prometheus's scrape interval cannot catch,
  or for metrics from ranks that are not addressable from
  Prometheus. The Pushgateway is documented at
  https://github.com/prometheus/pushgateway; it is designed for
  batch-job-style emissions.

Both need the `run_id` label. Both need the trainer to *stop
emitting* at run end so the metric series does not stay hot with
stale values.

The per-rank system metrics (per-rank step time, per-rank HBM) are
higher-volume and can either be pushed to the Pushgateway or
funnelled through an intermediate collector (a small OpenTelemetry
Collector, for example) that batches and forwards. The scale of
the fleet decides which pattern is right; chapter 4 revisits this
under the researcher's experiment-tracker lens.

## Gotchas the runbook exists to catch

Named, so you can grep the runbook for them later.

- **Field IDs change between DCGM releases.** A dashboard that
  references `DCGM_FI_DEV_THROTTLE_REASONS` breaks silently when
  the exporter is upgraded to a version that emits
  `DCGM_FI_DEV_CLOCKS_EVENT_REASONS`. Pin the exporter version;
  gate upgrades on a dashboard-render check.
- **Ordinals are not stable across reboots.** `gpu="0"` on Tuesday
  is not necessarily `gpu="0"` on Wednesday after a driver
  restart. Use `UUID` for any longitudinal analysis.
- **DCGM host engine can lag under load.** If DCGM's poll interval
  is set below the host engine's ability to fetch NVML data,
  values stall. On a busy node, `dcgmi dmon` shows the same
  values for many seconds — that is a symptom, not a Prometheus
  bug. Raise the poll interval.
- **Profiling metrics are not free.** As above, the `DCGM_FI_PROF_*`
  family costs GPU-side profiling overhead. Do not scrape them
  by default.
- **DCGM does not know about training runs.** Attribution of a
  DCGM metric to a `run_id` requires joining against scheduler
  data (which pod/job is on which node). Chapter 7's metadata
  store covers the join; the exporter alone does not.
- **`dcgm-exporter` on containerized workloads needs the right
  device plugin permissions.** The pod needs access to the DCGM
  socket or to run privileged with the NVIDIA runtime. The GPU
  Operator sets this up correctly by default; a hand-rolled
  DaemonSet is easy to get wrong.
- **Alerting on absolute ECC counts, not deltas.** A GPU that has
  had 3 SBE errors since the beginning of time is fine; a GPU
  with 3 in the last hour is a signal. Use `rate(...)` or
  `increase(...)`, not the raw counter.

## The recovery-from-metric-loss failure mode

A category of incident worth naming on its own: **metric loss
during the incident**. Prometheus is designed to be a best-effort
metrics store; when the exporter is unreachable (network partition,
node reboot), the samples for that window are simply missing. In a
training-observability context, this is a problem — the metric you
most want during an incident is the one that was being emitted
right as the incident started.

Two mitigations, both worth having:

- **Buffer at the source.** OpenTelemetry Collector in agent mode,
  configured with an in-memory queue and a file-based fallback,
  buffers metric samples during a Prometheus scrape failure. Not
  perfect (a node reboot loses the buffer) but reduces the
  "missing five minutes" pattern.
- **Federate to a second Prometheus.** A second Prometheus in a
  different failure domain scrapes the first (or scrapes the
  exporters directly at lower resolution). Two independent copies
  of the metrics; one survives most failure modes.

For the incident-post-mortem case where both mitigations lost the
sample, the trainer's own log (chapter 5's reproducibility bundle
includes a pointer to the log storage location) is the source of
truth. This is one of the reasons chapter 4's experiment tracker
is not redundant with Prometheus — the tracker is the researcher-
tier record, and it has its own durability profile.

## Summary

- DCGM is the userspace daemon on each node that exposes GPU
  telemetry via a fixed catalog of numeric field IDs.
  `dcgm-exporter` bridges it to Prometheus; the NVIDIA GPU
  Operator installs both on Kubernetes.
- Scrape a focused subset of fields (utilization, memory,
  temp/power, clocks/throttle, ECC, NVLink, PCIe, Xid). Do not
  scrape the `DCGM_FI_PROF_*` profiling family by default — it
  costs training-time overhead.
- Label DCGM metrics with `Hostname` and `UUID`; the GPU
  *ordinal* is not stable across reboots. Longitudinal analysis
  uses `UUID`.
- Prometheus retention for hardware metrics should be measured in
  months, not days, because silent-corruption diagnosis needs
  historical baselines. Cortex/Thanos/Mimir or a managed service
  is the pattern once one Prometheus is not enough.
- Dashboards live as Grafana JSON in version control. Template
  variables (`$run_id`, `$node`, `$gpu`) make historical
  diagnosis possible. Alert from Alertmanager or Grafana, not
  both.
- The training loop's own per-step scalars enter Prometheus via
  `prometheus_client` (rank 0) or the Pushgateway; per-rank
  system metrics fan through an intermediate collector at scale.
- The named gotchas — field-ID drift, unstable ordinals, host-
  engine lag, profiling overhead, DCGM-does-not-know-about-runs,
  absolute-count alerting — are what turn a nominally working
  stack into a broken one. The runbook has to catch all of them.
