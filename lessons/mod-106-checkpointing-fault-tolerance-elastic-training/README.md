# mod-106-checkpointing-fault-tolerance-elastic-training: Checkpointing, Fault Tolerance, and Elastic Training

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 18 hours

## Learning objectives

- Implement a PyTorch Distributed Checkpoint (DCP) save/load with async save, resharding across a different world size, and a stateful sampler
- Design a torchrun rendezvous + elastic reshape flow that survives a node crash mid-epoch
- Classify training-run incidents (loss spike, NaN, NCCL timeout, hardware fault, silent-corruption) and codify recovery playbooks
- Instrument straggler detection, silent-data-corruption detection, and automatic node quarantine
- Read the OPT-175B logbook and Llama 3 failure statistics and translate them into on-call runbooks
- Author an SLO for 'goodput' (useful training tokens per wall-clock hour) and derive an availability budget from it

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
