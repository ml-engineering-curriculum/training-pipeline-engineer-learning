# exercise-03: NCCL Collective Cost Model

**Estimated effort:** 3 hours

## Objective

Build an executable version of the α + β collective-cost model from
chapter 5, calibrate it against `nccl-tests` on the cluster you have
access to, and use it to predict the per-step comm cost of a specific
DDP and FSDP2 training configuration. The point is to turn the
whiteboard model into a spreadsheet or notebook you can actually
consult when designing a training job.

## Prerequisites

- Chapter 5 (this module).
- Read access to `nccl-tests` (github.com/NVIDIA/nccl-tests) or a
  cluster where it is already installed.
- ≥ 2 GPUs on ≥ 1 node. Multi-node access is ideal but not required —
  you can still calibrate intra-node numbers.

## Problem statement

Your team keeps guessing at whether a proposed strategy will fit in the
comm budget. That is unacceptable at scale. You are going to replace the
guesswork with a small, testable model, and demonstrate it agrees with
real numbers on the cluster to within a factor of 2 for the collectives
you care about.

## Requirements

1. **Author the cost model.** In a Python notebook or a small library,
   implement:
   ```python
   def all_reduce_cost(n_bytes, world_size, alpha_s, beta_s_per_byte, algo="ring"): ...
   def all_gather_cost(n_bytes, world_size, alpha_s, beta_s_per_byte, algo="ring"): ...
   def reduce_scatter_cost(n_bytes, world_size, alpha_s, beta_s_per_byte, algo="ring"): ...
   def all_to_all_cost(n_bytes, world_size, alpha_s, beta_s_per_byte): ...
   ```
   Use the formulas from chapter 5. `n_bytes` is the total payload the
   caller thinks about (e.g., the parameter tensor size for DDP's
   all-reduce), not the per-rank shard.
2. **Calibrate α and β from `nccl-tests`.**
   - Run `all_reduce_perf` at a sweep of message sizes (e.g., 1 KiB up
     to 1 GiB, doubling each step) at `-g 1 -n N` where `N` is your
     world size.
   - For each collective's output, extract `time_us` at small message
     sizes (to fit α) and `algbw` at large message sizes (to fit β).
   - Do this twice: once for a purely intra-node communicator, once for
     an inter-node communicator (if you have one).
3. **Predict a real training config.**
   - For a 1B BF16 dense model, `S = 2e9 bytes`. Predict the DDP
     all-reduce cost per step at your world size.
   - For the same model under FSDP2, predict per-step
     all-gather + reduce-scatter cost.
   - Predict the ratio to `T_compute` given the `6 · P · tokens` FLOP
     model, an MFU you commit to (`0.4` is a defensible start on
     H100), and the peak TFLOPs of your GPUs.
4. **Compare to reality.**
   - Run a single DDP step on the model above (use exercise 1's setup)
     and read the observed all-reduce wall time out of the profiler.
     Report `predicted / observed`. Do the same for FSDP2's collectives.
5. **Explain the delta.** Where the prediction and reality differ by
   more than 2×, write one paragraph on the likely cause: bucketing,
   overlap, unexpected algorithm choice (Ring vs. Tree vs. NVLS), NIC
   pinning, PXN, or model bug.

## Starter guidance

- Do not try to model every algorithm — get ring correct first, then
  add tree if you find yourself needing it for small-message latency.
- `NCCL_DEBUG=INFO` prints which algorithm and protocol NCCL picked
  for each collective; put that in your notebook so the assumptions
  in the cost model match the runtime.
- Use `torch.cuda.synchronize()` around timed sections; wall-clock
  measurements without sync are meaningless.
- Warm up NCCL communicators before timing (throw away the first two
  iterations).

## Acceptance criteria

- The notebook / library runs end-to-end with a single command and
  produces both a table of predicted collective costs and a table of
  observed costs.
- α and β are fitted from real `nccl-tests` output, not copied from a
  spec sheet. The fitting method is documented and reproducible.
- Predicted / observed collective cost falls within 2× for at least
  the collectives that dominate the DDP and FSDP2 configurations.
- The write-up correctly identifies which term (α or β) dominates for
  each configuration and defends the choice of algorithm term with
  reference to chapter 5 or the NCCL docs.

## Stretch goals

- Extend the model with a `topology_penalty` factor and calibrate it
  against a deliberately misconfigured run (e.g., disable PXN via
  `NCCL_PXN_DISABLE=1` and rerun `nccl-tests`).
- Add a tree-algorithm variant and identify the crossover message size
  at which ring stops winning for your world size. Explain why.
- Fold the model into a `Makefile` target so `make predict-ddp
  MODEL_BYTES=2e9 WORLD_SIZE=8` prints the predicted step cost.
