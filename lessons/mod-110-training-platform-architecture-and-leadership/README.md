# mod-110-training-platform-architecture-and-leadership: Training-Platform Architecture and Cross-Team Leadership

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 15 hours

## Learning objectives

- Architect a multi-tenant training-cluster platform for 5–15 downstream teams: quotas, gang preemption, priority classes, on-call, escalation
- Author RFCs and design docs for training-platform surface changes (framework upgrade, hardware refresh, storage migration)
- Design hand-off contracts with peer platform tracks: fine-tuning-engineer (consumer of the platform), ai-infra-ml-platform-engineer (owns serving + registry), ai-infra-mlops-engineer (owns CI/CD + post-training pipeline), ai-infra-performance-engineer (authors kernels the platform consumes), ai-infra-security-engineer (owns training-data provenance and cluster boundary)
- Run a build-vs-buy decision at level-35 altitude (Megatron-LM in-house vs. NeMo integration vs. Databricks/Mosaic-hosted vs. Together-hosted training)
- Author an incident review for a real (open-report-anchored) training-run outage: OPT-175B loss spike or BLOOM hardware failure
- Design a training-platform migration plan (e.g. FSDP1 → FSDP2, or Megatron-LM → torchtitan) with rollback, versioning, and researcher-side compatibility windows

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
