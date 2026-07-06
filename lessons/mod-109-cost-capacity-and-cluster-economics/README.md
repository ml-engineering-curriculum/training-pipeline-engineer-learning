# mod-109-cost-capacity-and-cluster-economics: Cost, Capacity, and Cluster Economics for Training Runs

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 14 hours

## Learning objectives

- Apply Chinchilla / Kaplan scaling laws to translate a product target (parameter count × quality target) into a data budget
- Convert data + parallelism strategy into GPU-hours and dollars for a target cluster shape
- Reason about reserved-vs-spot / dedicated-vs-shared / on-prem-vs-cloud economics for training runs
- Model the cost impact of MFU improvements and checkpointing / restart budgets
- Author a training-run feasibility study (Chinchilla-scaled compute → cluster shape → dollar cost → wall-clock schedule)
- Reason about scaling-up (frontier) vs. scaling-down (cost-optimised) recipes with MPT-7B and Llama 3 as anchors

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
