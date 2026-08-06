# exercise-02: DCGM + Prometheus + Grafana Integration

**Estimated effort:** 3 hours

## Objective

Stand up the GPU-telemetry pipeline described in chapter 3, from
`dcgm-exporter` running on every GPU node to `Row 3` of the
exercise-01 dashboard rendering real DCGM data. Deliverable: a
running stack (Kubernetes or Docker Compose), a documented set
of scraped fields, a set of Prometheus recording rules for
per-node and per-fleet rollups, and a `stack_notes.md` that
records the specific gotchas from chapter 3 you had to work
through.

## Prerequisites

- Chapter 3 of this module.
- Access to at least one GPU node with the NVIDIA driver
  installed. A single node with 1–2 GPUs is sufficient to
  exercise the plumbing; more nodes let you exercise the roll-up.
- A Kubernetes cluster with GPU nodes (preferred) or bare-metal
  nodes where you can install systemd units.
- The Grafana + Prometheus stack from exercise 1 (or a fresh
  install for this exercise).
- Read access to the NVIDIA GPU Operator docs
  (https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)
  and the `dcgm-exporter` repository README
  (https://github.com/NVIDIA/dcgm-exporter).

## Problem statement

Chapter 3 said "install DCGM, scrape it, render it" in a paragraph
and then spent the rest of the chapter naming the specific
gotchas that turn a nominally-working stack into a broken one.
This exercise is the operational cash-out: stand the stack up,
scrape the fields chapter 3 said matter, roll them up correctly,
and write down every gotcha you hit so the next teammate does not
hit it.

## Requirements

Ship one directory:

- `gpu_telemetry/`
  - `install.md` — the specific commands you ran to install the
    stack, from a clean environment. Reproducible.
  - `dcgm-exporter-config.yaml` — the field-list configuration
    (which `DCGM_FI_*` fields are scraped).
  - `prometheus.yml` — the scrape configuration.
  - `recording_rules.yml` — recording rules for per-node and
    per-fleet rollups.
  - `alerting_rules.yml` — the minimum alert set from chapter 3.
  - `dashboard-hardware.json` — a Grafana dashboard focused on
    the hardware panels (roll-up plus a drill-down per-node
    view).
  - `stack_notes.md` — the notes-to-self on gotchas and
    resolutions.

### 1. The install

Use the GPU Operator on Kubernetes if available. If you are on
bare metal, install DCGM + `dcgm-exporter` as systemd units. In
either case, document every command in `install.md` so a teammate
can reproduce your environment.

Verify the install with:

```bash
dcgmi discovery -l          # sees the GPUs
curl localhost:9400/metrics # dcgm-exporter is scraping
```

### 2. The scraped field list

Configure `dcgm-exporter` to scrape a focused subset (chapter 3):

- Utilization: `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_MEM_COPY_UTIL`.
- Memory: `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE`,
  `DCGM_FI_DEV_FB_TOTAL`.
- Temperature and power: `DCGM_FI_DEV_GPU_TEMP`,
  `DCGM_FI_DEV_MEMORY_TEMP` (if available on your GPU),
  `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_SLOWDOWN_TEMP`.
- Clocks and throttling: `DCGM_FI_DEV_SM_CLOCK`,
  `DCGM_FI_DEV_MEM_CLOCK`, `DCGM_FI_DEV_CLOCKS_EVENT_REASONS`
  (or the older `_THROTTLE_REASONS` name; check what your
  version emits).
- ECC: `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL`,
  `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL`, plus the row-remap /
  retired-page families as available.
- NVLink: `DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL`,
  `DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL`.
- PCIe: `DCGM_FI_DEV_PCIE_REPLAY_COUNTER`.
- Xid: `DCGM_FI_DEV_XID_ERRORS` (or the equivalent name in your
  version).

**Do not scrape the `DCGM_FI_PROF_*` profiling family by
default.** Note this decision explicitly in `stack_notes.md`
(chapter 3 rationale).

### 3. Recording rules

Author at least these recording rules in `recording_rules.yml`:

- Per-node average GPU utilization.
- Per-node maximum GPU temperature.
- Per-node SBE ECC error *rate* (per hour) — chapter 3 called
  out "alerting on absolute counts" as a gotcha. Use `rate()` or
  `increase()`.
- Per-node DBE ECC error *count* (increase over 1h).
- Per-fleet median GPU utilization and top-10 hottest GPUs.

These are what the dashboard's roll-up panels query; putting them
in recording rules makes the dashboard render fast.

### 4. Alerting rules

Author at least these Alertmanager rules in `alerting_rules.yml`:

- Any DBE ECC event on any GPU (severity: page).
- Sustained SBE ECC rate above threshold for `>1 h` (severity:
  ticket).
- Any Xid event of specific severities (consult chapter 3 refs).
- GPU temperature above `SLOWDOWN_TEMP` sustained for `>10 min`.
- `dcgm-exporter` job down for `>2 min` (metrics loss).
- Persistent throttle-reason of `THERMAL` or `HW_SLOWDOWN` for
  `>10 min`.

Each rule must have an `annotations` block with a runbook link
pointing at mod-106 chapter 5 or chapter 6 as appropriate.

### 5. The hardware dashboard

`dashboard-hardware.json` is a Grafana dashboard with two views:

- **Roll-up view**: fleet-wide summary — top-N hot GPUs, ECC
  heatmap, throttle-reason count, NVLink error rate per node.
- **Per-node drill-down view**: templatized on `$node`, showing
  every scraped field for the selected node. Templating exposes
  a dropdown of nodes populated from `label_values(up{job="dcgm-exporter"},
  Hostname)`.

Both views should render on a fresh Prometheus (with the recording
rules loaded); do not hard-code node names.

### 6. The gotcha notes (`stack_notes.md`)

Write down every gotcha you hit. Chapter 3's named list is the
starting point; add anything specific to your environment. At
minimum:

- Field-ID versioning: what version of `dcgm-exporter` are you
  running, and which metric names differ from what the docs
  show?
- Ordinals vs. UUIDs: which of your labels uses which; if any of
  your dashboards use `gpu` (ordinal), explain why it's safe.
- DCGM host-engine load: at what poll interval does the exporter
  keep up on your hardware? Does anything stall at higher
  frequency?
- Profiling metrics: explicit note that you did not enable them
  and why.
- If applicable: how you attached `run_id` (or equivalent
  scheduler label) to the DCGM metrics — Prometheus recording
  rule joining against a `kube_pod_info` metric, or equivalent.
- If applicable: which fields were unavailable on your GPU
  generation and how you handled the "no data" state on the
  dashboard.

## Starter guidance

- **Pin the `dcgm-exporter` version.** Everything downstream —
  metric names, field IDs, dashboard queries — depends on it.
  Note the version in `install.md`.
- **Test one field before the whole list.** Get
  `DCGM_FI_DEV_GPU_TEMP` scraping first; confirm it in
  Prometheus; then add the rest.
- **Confirm labels before writing dashboards.** Different
  installs label the metric with `Hostname`, `node`, `instance`,
  or `kubernetes_node`. Whichever your stack emits, use it
  consistently.
- **Trigger a real signal to test the alert.** Increase your
  GPU's `SLOWDOWN_TEMP` threshold in a test environment (or run
  a synthetic workload to raise temperature) to verify the
  alert path end-to-end. Do not ship alerting rules you have
  not seen fire.
- **Do not skip the drill-down view.** A per-node view is what
  on-call opens when the roll-up panel flags a specific node;
  building only the roll-up leaves the on-call flying blind on
  the next click.

## Acceptance criteria

- `dcgm-exporter` is running on every GPU node in your test
  environment; `curl :9400/metrics` returns the configured
  fields.
- Prometheus is scraping successfully; the fields appear in
  `Status → Targets` and `Status → Rules`.
- All recording rules evaluate without error.
- The hardware dashboard renders both the roll-up and the per-
  node drill-down views on a fresh browser (no state cached).
- All alerting rules have runbook links; you have triggered at
  least one alert (in a test environment) end-to-end and
  confirmed it reached Alertmanager.
- `stack_notes.md` names every chapter 3 gotcha you encountered
  and how you resolved it.
- The `DCGM_FI_PROF_*` decision is documented explicitly.
- The `install.md` is reproducible: a teammate following it from
  a clean environment ends up with the same stack.

## Stretch goals

- Wire in a long-term store (Mimir, Thanos, or Cortex) and
  configure retention for hardware metrics at ≥ 6 months. Verify
  by querying a metric that is older than the local Prometheus's
  retention window.
- Add an OpenTelemetry Collector between exporters and
  Prometheus configured with a file-based buffer for
  metric-loss-during-incident mitigation (chapter 3 recovery
  section). Verify by killing Prometheus for two minutes and
  confirming the samples arrive afterward.
- Federate to a second Prometheus in a different failure domain
  (a second small VM with its own scrape config). Confirm that
  killing either Prometheus leaves the other with the data.
- Add a `run_id` label to DCGM metrics via a Prometheus
  recording rule that joins against `kube_pod_info` (or the
  Slurm equivalent). Now the hardware dashboard can filter by
  training run — the join chapter 3 flagged as non-trivial.
- Fork the alert set into two tiers: paged alerts (severity 1)
  and ticket alerts (severity 2). Defend the classification in
  `stack_notes.md`. If any alert cannot be defended as either,
  delete it.
