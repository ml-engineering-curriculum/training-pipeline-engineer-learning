# DCGM, dcgm-exporter, Prometheus, and Grafana

Chapter 1 fixed the mental model and the panel budget. This chapter
wires up the cluster-level half of it. The stack you will learn is
the one large-scale training shops converge on, in this order:

1. **NVIDIA Data Center GPU Manager (DCGM)** on every node —
   authoritative per-GPU health and utilization telemetry.
2. **`dcgm-exporter`** — the DCGM-to-Prometheus adapter that
   runs as a DaemonSet (Kubernetes) or systemd unit (SLURM /
   bare-metal).
3. **Prometheus** — pull-based time-series store scraping
   `dcgm-exporter`, node exporter, and your framework's metric
   endpoint.
4. **Grafana** — dashboards on top of Prometheus. This is where
   panels 5, 6, 7, and 8 from chapter 1 live.

Everything below is anchored in the vendor's own documentation:

- NVIDIA DCGM: https://docs.nvidia.com/datacenter/dcgm/
- `dcgm-exporter`: https://github.com/NVIDIA/dcgm-exporter
- Prometheus documentation: https://prometheus.io/docs/
- Grafana documentation: https://grafana.com/docs/grafana/latest/
  <!-- needs-research: verify canonical Grafana docs URL. -->

## Why DCGM and not `nvidia-smi`

`nvidia-smi` is a debugging tool for a human at a shell. DCGM is the
production telemetry system for a fleet:

- DCGM keeps a *cached* view of every GPU's state (SM util, memory
  util, temperature, power, ECC errors, XID events, PCIe /
  NVLink throughput, throttling reasons) and streams it via a
  single well-defined API — no per-poll shell fork, no
  best-effort parsing of `nvidia-smi -q -x`.
- DCGM has a formal *field catalog* — every metric is an integer
  field ID with documented semantics — so the meaning of "the
  number labelled SM utilization" is stable across driver
  versions.
- DCGM ships **health checks** and **diagnostics** (`dcgmi health
  --check`, `dcgmi diag -r 3`) that are the same commands you
  run in an incident and in a nightly regression check.

The `nvidia-smi` output is fine for a one-off ssh; every dashboard
in this module goes through DCGM.

## Deployment shape

The stack has three deployment shapes depending on your platform:

- **Kubernetes.** `dcgm-exporter` runs as a DaemonSet in a
  privileged pod that mounts the NVIDIA device plugin's socket.
  Prometheus discovers it via `ServiceMonitor` or the
  Prometheus Operator's pod-monitor annotations. This is what the
  NVIDIA GPU Operator installs by default.
- **SLURM / bare-metal.** `dcgm-exporter` runs as a systemd unit
  on every compute node. Prometheus static-configures the scrape
  target list (or discovers it via Consul / file_sd).
- **Cloud.** The managed GPU node images (EKS AMI, GKE
  node-image, AKS) ship the GPU Operator preinstalled;
  Prometheus lives in your managed observability stack
  (Managed Prometheus, Google Cloud Managed Prometheus, Azure
  Monitor).

Regardless of shape, the observable surface is the same: an HTTP
endpoint (default `:9400/metrics`) that exposes DCGM fields in
Prometheus text format.

## Which DCGM fields to scrape

DCGM exposes hundreds of fields. Scraping all of them at 1-second
resolution across a 1000-node cluster will crush Prometheus. The
minimal set that supports every panel in chapter 1:

| Concern | DCGM field | Cardinality note |
|---------|-----------|------------------|
| SM utilization (%) | `DCGM_FI_DEV_GPU_UTIL` | per-GPU |
| HBM used bytes | `DCGM_FI_DEV_FB_USED` | per-GPU |
| Memory copy util (%) | `DCGM_FI_DEV_MEM_COPY_UTIL` | per-GPU |
| Power draw (W) | `DCGM_FI_DEV_POWER_USAGE` | per-GPU |
| GPU temperature (°C) | `DCGM_FI_DEV_GPU_TEMP` | per-GPU |
| HBM temperature (°C) | `DCGM_FI_DEV_MEMORY_TEMP` | per-GPU |
| SBE ECC (correctable) | `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` | per-GPU |
| DBE ECC (uncorrectable) | `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` | per-GPU |
| Latest XID event | `DCGM_FI_DEV_XID_ERRORS` | per-GPU |
| PCIe TX / RX bytes | `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` / `_RX_` | per-GPU |
| NVLink bandwidth per link | `DCGM_FI_DEV_NVLINK_BANDWIDTH_L0` … `L17` | per-GPU × per-link |
| Throttle reasons bitmask | `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` | per-GPU |

Verify exact field-name spelling against the DCGM API reference for
the driver version you run — the field naming has been stable but
the *set* of fields grows every driver release.
<!-- needs-research: pin the DCGM field-catalog page URL for the
current driver family (currently under docs.nvidia.com/datacenter/
dcgm/latest/dcgm-api/dcgm-api-field-ids.html — verify). -->

## Cardinality control

At 8 GPUs × 18 NVLinks × 1000 nodes × 30 fields at 1 s resolution,
naïve scraping produces > 4 million active series. Prometheus can
handle it; your Grafana rendering budget cannot. Three levers:

- **Scrape interval per field group.** Utilization and step
  timing at 5–10 s, temperature and power at 15–30 s, ECC / XID
  counters at 60 s. `dcgm-exporter` supports per-field
  intervals via its config file.
- **Aggregate at the recording-rule layer.** Recording rules like
  `dcgm_sm_util_avg_by_node = avg by (node) (DCGM_FI_DEV_GPU_UTIL)`
  cut down what dashboards query. Panels 6 uses the per-GPU
  series (heatmap); panel 8 uses the recording-rule roll-up.
- **Drop labels you do not need at query time.** By default
  `dcgm-exporter` includes `Hostname`, `UUID`, `pci_bus_id`,
  `container`, `pod`, `namespace`. Drop the ones your queries
  never group by.

Rule of thumb: the top-level dashboard's Prometheus queries should
all complete in under one second. If they do not, you added
cardinality without adding an aggregation.

## A working `dcgm-exporter` scrape config

Prometheus-side scrape configuration for a Kubernetes deployment
that uses the Prometheus Operator's `PodMonitor` CRD:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: dcgm-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: dcgm-exporter
  namespaceSelector:
    matchNames: [gpu-operator]
  podMetricsEndpoints:
    - port: metrics
      interval: 10s
      scrapeTimeout: 5s
      relabelings:
        - sourceLabels: [__meta_kubernetes_pod_node_name]
          targetLabel: node
        - sourceLabels: [__meta_kubernetes_pod_name]
          targetLabel: pod
      metricRelabelings:
        # Drop per-container labels we do not group by.
        - action: labeldrop
          regex: (container|namespace|pod_ip)
```

For a SLURM / bare-metal cluster, the equivalent lives in
`prometheus.yml`:

```yaml
scrape_configs:
  - job_name: dcgm-exporter
    scrape_interval: 10s
    static_configs:
      - targets:
          - compute-001.dc.example.net:9400
          - compute-002.dc.example.net:9400
          # …one entry per compute node…
    relabel_configs:
      - source_labels: [__address__]
        regex: '([^:]+):.*'
        target_label: node
        replacement: '$1'
```

The naming conventions here follow the Prometheus best-practice
guide (see chapter 1); every label is bounded and every metric
comes with its unit implied by the DCGM field's documented type.

## Alerts that pull the right lever

Alerting is where DCGM stops being telemetry and becomes an
operational surface. Four alerts every training cluster runs, no
exceptions:

1. **Persistent low SM utilization (straggler).**

   ```
   avg_over_time(DCGM_FI_DEV_GPU_UTIL[5m]) < 30
     and on(node,gpu) training_job_running == 1
   ```

   If SM util is below 30% for 5 minutes on a GPU that a training
   job is scheduled on, the job is comm-bound *or* the GPU is a
   straggler. Route to the training-platform on-call. See chapter
   5 for the runbook.

2. **XID error (auto-quarantine).**

   ```
   increase(DCGM_FI_DEV_XID_ERRORS[1m]) >= 1
   ```

   XID codes ≥ 79 (see the NVIDIA XID error reference)
   <!-- needs-research: NVIDIA XID error catalog URL —
   https://docs.nvidia.com/deploy/xid-errors/ but verify. -->
   are usually unrecoverable. Route to node cordon + drain
   automation. Chapter 5 covers XID triage.

3. **HBM temperature above throttle threshold.**

   ```
   DCGM_FI_DEV_MEMORY_TEMP > 90
   ```

   H100 HBM throttles around 95 °C. A 90 °C alert gives you time
   to catch airflow / cooling issues before the run starts
   throttling and MFU tanks silently.

4. **Uncorrectable ECC (DBE).**

   ```
   increase(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL[1m]) >= 1
   ```

   A double-bit ECC error is silent corruption territory. Route
   to node quarantine; the affected run needs to roll back to the
   last DCP checkpoint (mod-106 owns that flow).

Each alert threshold has to be tuned to your cluster's normal
range. Get the baseline from chapter 1's dashboard before you
turn these on with paging; a noisy alert on day one destroys
your on-call's trust in the whole stack.

## What DCGM does *not* tell you

DCGM sees the driver's view of the GPU. It does not see:

- **What your kernels are actually doing.** For that, use the
  PyTorch profiler, Nsight Systems, or CUPTI-based traces
  (owned by mod-107).
- **NCCL comm time per collective.** DCGM shows NVLink bandwidth
  utilization, which tells you *how much* wire the collective
  used, not *how long* your framework spent inside it. For that,
  use `torch.cuda.Event` timing around `dist.all_reduce()` (see
  chapter 3) or the Nsight Systems NCCL plugin.
- **Framework-level state.** Loss, gradient norm, learning rate:
  those are the experiment tracker's job in chapter 3.
- **Storage throughput per rank.** DCGM sees PCIe bytes, not
  which file they came from. Chapter 6 of mod-105 owns the
  storage-throughput view.

The right way to think about it: DCGM is the *fabric baseline* —
everything below your training-loop code — and the experiment
tracker is the *code baseline*. Together they cover everything.

## Bring-up checklist

Before you turn paging on:

1. `dcgm-exporter` running on every GPU node, verified with a
   `curl :9400/metrics | head`.
2. Prometheus scraping all nodes at 10 s interval, verified via
   the `up{job="dcgm-exporter"}` query.
3. Panels 5 (per-rank step-time heatmap), 6 (SM/HBM heatmap), 7
   (NCCL comm overlay), and 8 (XID / ECC / node-down) live in
   Grafana with threshold lines and click-through URLs to the
   experiment tracker.
4. The four alerts above configured on a *staging* alertmanager
   route (no paging yet) for one week; tune thresholds against
   the observed noise.
5. Move alerts to the real on-call route only after the staging
   week produces zero false pages on healthy runs.

That checklist ships one time per cluster. After that, it's the
regression check every driver / DCGM / dcgm-exporter upgrade goes
through.

## Summary

- DCGM is the authoritative per-GPU telemetry source; `nvidia-smi`
  is a debugging tool. Fleet dashboards go through DCGM.
- `dcgm-exporter` scrapes DCGM into Prometheus over an HTTP
  endpoint. Deploy as a DaemonSet on Kubernetes or a systemd
  unit on SLURM / bare-metal.
- Scrape the minimum field set: SM util, HBM used, memory copy
  util, power, temperature, ECC (SBE + DBE), XID, PCIe TX/RX,
  per-link NVLink bandwidth, throttle reasons.
- Control cardinality with per-field scrape intervals, recording
  rules that aggregate to node roll-ups, and dropping labels
  Grafana never groups on.
- Standing alerts every training cluster runs: persistent low SM
  util, XID error, HBM over 90 °C, DBE ECC.
- DCGM does not see kernel behavior, NCCL comm time,
  framework-level loss / gradient norm, or per-file storage
  throughput. Those live in the experiment tracker (chapter 3),
  the PyTorch profiler (mod-107), and mod-105's storage view.
