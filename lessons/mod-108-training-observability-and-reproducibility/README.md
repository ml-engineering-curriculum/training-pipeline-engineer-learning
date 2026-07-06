# mod-108-training-observability-and-reproducibility: Training Observability and Reproducibility at Scale

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 15 hours

## Learning objectives

- Design a training-run dashboard covering loss curves, gradient norms, throughput, MFU, GPU utilization, thermals, and NCCL health
- Wire DCGM into per-node GPU telemetry and Prometheus / Grafana for cluster-wide roll-ups
- Integrate W&B / TensorBoard / MLflow at pretraining scale — including sharded logging, sub-sampled metrics, and offline sync patterns
- Author a run reproducibility bundle: seed + config + dataset hash + tokenizer hash + framework versions + container digest + hardware manifest
- Detect and diagnose the canonical run-time signatures: divergence, loss spike, throughput cliff, straggler, silent-corruption
- Design an experiment-metadata store that lets the fine-tuning-engineer / model-evaluation-engineer teams pull run provenance

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
