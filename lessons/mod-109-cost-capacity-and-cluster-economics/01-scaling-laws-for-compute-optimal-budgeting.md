# Scaling Laws for Compute-Optimal Budgeting

Every conversation about a training run eventually reduces to three
numbers: the parameter count `N`, the training-token count `D`, and the
compute `C` in FLOPs. Everything else — the cluster shape, the framework,
the reserved-vs-spot mix, the wall-clock schedule — is downstream of
those three. This chapter teaches you the scaling laws that let you set
`N`, `D`, and `C` on the back of an envelope, before you write a single
YAML file. The rest of the module then turns that envelope into a cluster
plan and a dollar number.

If mod-101 taught you *how* the flops move across the fabric, this
chapter teaches you *how many* flops you actually need in the first
place. It is a budgeting instrument, not a training recipe.

## Why you need a budget before you touch a config

A training run is expensive in three ways at once: dollar cost, wall-clock
time, and cluster capacity opportunity cost. Every one of them scales
with `C`. If you cannot estimate `C` for your product target before you
begin, you cannot answer any of the questions your director will ask:

- "Can we do this on the 512-GPU cluster we already have, or do we have
  to burst?"
- "Will we hit the demo deadline in eight weeks?"
- "What do we lose in quality if we cut the budget by 40%?"

The scaling laws in this chapter give you a defensible answer to all
three. They will not be exact — no scaling law ever is — but they will
put you inside a factor of two of the truth, which is enough to sort
cluster shapes and enough to write the feasibility study in chapter 5.

## The `6 · N · D` FLOP model

The workhorse formula:

    C ≈ 6 · N · D

- `N` is the number of trainable parameters (embeddings, transformer
  blocks, LM head — everything you count in `sum(p.numel() for p in
  model.parameters())`).
- `D` is the number of training tokens seen (one pass over the corpus at
  batch-size accounting; not "documents", not "characters").
- The `6` factor decomposes as `2` (multiply-add counted as two FLOPs) ×
  `3` (one forward pass + two backward-pass matmuls per weight — one for
  the input gradient, one for the weight gradient). Kaplan et al. (2020),
  "Scaling Laws for Neural Language Models" (arXiv:2001.08361), Appendix
  D, derives this cleanly for transformer LMs; the same accounting
  appears in Hoffmann et al. (2022), the Chinchilla paper
  (arXiv:2203.15556), Appendix F.

**What is included.** The `6 · N · D` count is a *dense-matmul FLOP*
estimate. It counts every parameter contributing to every token's
forward and backward pass. For a standard dense decoder-only transformer
with tied or untied embeddings, it is within a few percent of what a
FLOP counter (PyTorch `FlopCounterMode`, `deepspeed.profiling.flops_profiler`,
`torch.profiler`) will tell you.

**What is not included.** Attention's quadratic term in sequence length
is *not* in `6 · N · D`. For most training regimes with `T ≤ 4 k` and
`d_model = 4096`, the attention term is a few percent of the matmul
FLOPs and safely ignorable at budgeting altitude. Once you push `T` to
32 k or 128 k (long-context pretraining, video, code with long files),
the attention `O(T²)` term becomes a first-class cost and you must add
it back in. Chinchilla's own appendix shows the correction; Chowdhery et
al. (2022, PaLM) discuss it explicitly.

**When it fails outright.** Two cases where `6 · N · D` is wrong by
enough to matter for a budget:

- **Mixture-of-experts (MoE).** Only `k` of the `E` experts are active
  per token. The "activated" parameter count `N_active ≈ N_dense +
  (k / E) · N_experts` is what enters `6 · N · D` for training FLOPs, not
  the total parameter count. Switch Transformer (Fedus et al., 2021),
  GShard (Lepikhin et al., 2020), and Mixtral (Jiang et al., 2024) all
  report cost against activated params, not total.
- **Sparse attention / MoE-routing all-to-all overhead.** These add
  communication cost that is invisible to the FLOP model but very
  visible in wall-clock time. Budget them separately if you are going
  down that road.

For the rest of this module we assume a dense transformer LM. When you
step outside that shape, correct `N` and re-derive.

## Kaplan (2020): loss as a power law in compute

Kaplan et al. (2020) fit language-modeling loss on a broad sweep of
model shapes and dataset sizes and found, empirically, that:

    L(C) ≈ A · C^(-α_C)

where `L` is cross-entropy loss, `C` is training compute in FLOPs, `A`
and `α_C` are fit constants, and — critically — the same relation holds
in `N` alone and in `D` alone with their own exponents `α_N`, `α_D`:

    L(N) ≈ A_N · N^(-α_N)
    L(D) ≈ A_D · D^(-α_D)

Two consequences you should burn into your intuition:

1. **Loss is smooth in compute.** Doubling `C` decreases loss by a
   predictable factor. There is no "wall" you crash into within the
   range of production training runs.
2. **There is a compute-optimal frontier.** For a fixed `C`, you can
   spend it on more parameters or more data. Kaplan's original fit
   suggested `N ≫ D` (spend it on parameters), which drove the "just
   make it bigger" era: GPT-3 at 175B params on ~300B tokens (Brown et
   al., 2020).

Kaplan's constants were later revised, but the *shape* of the law — power
law in `C`, smooth over many decades — has held up across every
follow-up study for dense transformer LMs.

## Chinchilla (2022): the compute-optimal recipe

Hoffmann et al. (2022), "Training Compute-Optimal Large Language
Models" (arXiv:2203.15556), re-ran the Kaplan sweep with three fitting
methodologies and reached a striking conclusion: for a fixed compute
budget `C`, the loss-minimising split is **roughly `D ≈ 20 · N`
tokens per parameter**, not the parameter-heavy split Kaplan had
suggested.

The paper's headline result is the Chinchilla model itself: **a 70B
model trained on 1.4T tokens** hits the same loss as, and outperforms
on downstream evals, Gopher's 280B model trained on 300B tokens. Same
compute budget, four times fewer parameters, four times more data.

In practice, "20 tokens per parameter" is the rule of thumb you carry
around. The precise ratio depends on the loss you are targeting and on
the fit methodology (Chinchilla's three approaches agree on the shape,
disagree on the exact constant), but for budgeting `20 · N` is the
right anchor.

A canonical table you should have in your head:

| Model size `N` | Chinchilla-optimal `D` | Approx `C = 6 · N · D` |
|----------------|------------------------|------------------------|
| 1 B            | 20 B tokens            | ~1.2 × 10^20 FLOPs     |
| 7 B            | 140 B tokens           | ~5.9 × 10^21 FLOPs     |
| 13 B           | 260 B tokens           | ~2.0 × 10^22 FLOPs     |
| 30 B           | 600 B tokens           | ~1.1 × 10^23 FLOPs     |
| 70 B           | 1.4 T tokens           | ~5.9 × 10^23 FLOPs     |
| 175 B          | 3.5 T tokens           | ~3.7 × 10^24 FLOPs     |
| 400 B          | 8 T tokens             | ~1.9 × 10^25 FLOPs     |

Chapter 2 turns that FLOP number into GPU-hours on real hardware.

## Overtraining: when you deliberately break Chinchilla

Chinchilla optimises for loss at a fixed *training* compute budget. It
does **not** optimise for inference cost, latency, or memory footprint.
Most production shops trade some training-optimal-ness for inference
efficiency by training a smaller model on more data than Chinchilla
prescribes:

- **Llama 2 (Touvron et al., 2023)** trained a 7B on 2 T tokens
  (~286 tokens/param, well above Chinchilla's 20).
- **Llama 3 (Grattafiori et al., 2024)** trained the 8B on ~15 T tokens
  (~1900 tokens/param).
- **MPT-7B (MosaicML, 2023)** trained on ~1 T tokens (~143 tokens/param).
  <!-- needs-research: verify exact MPT-7B token count in the MosaicML
  blog post -->

The trade you are making: extra training FLOPs are cheap at frontier
scale compared to the inference savings from a smaller model over its
deployment lifetime. This is the "inference-first" motivation for
overtraining. Chapter 6 comes back to it with MPT-7B and Llama 3 as the
two anchor recipes.

The takeaway for budgeting: `D ≈ 20 · N` is the *floor* if you want
Chinchilla-optimal loss for the compute you spent. Above that ratio you
are trading training compute for inference cost. Below it you are
under-training, and the model will be worse than a smaller model with
the same compute.

## From product target to compute budget: three worked examples

Every feasibility study starts here. You have a product target; you
need `C`.

**Example A. "We want a 7B model competitive with Llama 2 7B."**

- Set `N = 7 × 10^9`.
- Chinchilla-optimal `D ≈ 20 · N = 1.4 × 10^11 = 140 B tokens`.
- `C = 6 · N · D ≈ 6 × 7e9 × 1.4e11 ≈ 5.9 × 10^21 FLOPs`.
- If you want to match Llama 2's ~2 T token training (overtrained ~14×
  Chinchilla), multiply `C` by 14 → `~8.3 × 10^22 FLOPs`.

Chapter 2 turns each of those into a GPU-hour count.

**Example B. "We want a 30B model that matches a specific eval target."**

- Set `N = 3 × 10^10`.
- Chinchilla-optimal `D ≈ 6 × 10^11 = 600 B tokens`.
- `C ≈ 1.08 × 10^23 FLOPs`.
- If your eval target is aggressive (frontier-adjacent), plan on 2–3×
  Chinchilla — call it `3 × 10^23 FLOPs`.

**Example C. "We want a 70B model at Llama-3-70B quality."**

- Llama 3 reports training the 70B on ~15 T tokens (Grattafiori et al.,
  2024, section 3). That is ~214 tokens/param — well into the
  overtrained regime.
- `C ≈ 6 × 7e10 × 1.5e13 ≈ 6.3 × 10^24 FLOPs`.

You now have a `C` for each. The rest of the module answers: *how many
GPU-hours, on what hardware, for how many dollars, over what
wall-clock?*

## Sensitivity: how tight does `C` need to be?

Two questions worth asking every time you cite a `C`:

- **How wrong is `6 · N · D`?** For a dense transformer with `T ≤ 4 k`
  and any reasonable `d_model`, expect a few percent overhead from
  attention and layernorm, sometimes as much as 5–10% for short-context
  models with wide hidden. Kaplan's Appendix D and Chinchilla's Appendix
  F are the reference derivations.
- **How wrong is `D ≈ 20 · N`?** Chinchilla's three fits agree that the
  ratio is somewhere in the 10–30 range depending on the exact loss you
  are targeting. Anchor on 20, but do not defend the second significant
  figure.

For feasibility work at the tens-of-millions-of-dollars altitude, you
want `C` within `±50%`. That is easily inside what `6 · N · D` with
`D ≈ 20 · N` gives you. When someone tries to pin you down to a `±5%`
compute estimate, push back — it is inside the noise of the scaling-law
fit, not the noise of your arithmetic.

## The three prices of `C`

`C` alone is not what your director cares about. `C` implies three
downstream numbers, and every feasibility study has to make them
explicit:

- **Dollar cost** — `C` in FLOPs times `$/FLOP` on the cluster shape you
  can actually book. Chapter 2 does the arithmetic.
- **Wall-clock** — `C` divided by the aggregate effective FLOPs/s of your
  cluster. Chapter 2 does this too.
- **Opportunity cost** — a cluster busy on this run is not available for
  the next one. On a shared multi-tenant cluster, this is real and shows
  up as queue time for other teams (mod-104's fair-share policy is the
  operational mechanism; the economic model is here).

Every subsequent chapter in this module is downstream of these three
numbers.

## What this chapter is not

- It is **not** a recipe for training a model. Setting `N`, `D`, and
  `C` picks the budget; picking the framework (mod-102), the data mix
  (mod-103), the parallelism strategy (mod-101), and the reliability
  posture (mod-106) all still have to happen.
- It is **not** a substitute for eval. The scaling laws predict loss;
  loss is a proxy for downstream eval performance, not a guarantee. If
  your eval target is `MMLU > 65` at 7B, no scaling law tells you
  whether you will hit it — you still have to train and measure.
- It is **not** the last word on scaling. Scaling laws are empirical
  fits over the data that existed when the paper was written. If you
  are pushing well past frontier-scale, or into a new modality, expect
  the constants to move. Refit locally on a small ladder before you
  commit hundreds of millions of dollars.

## Summary

- `C ≈ 6 · N · D` is the workhorse FLOP model for dense transformer LM
  training. It is derived cleanly in Kaplan (2020) Appendix D and
  Chinchilla (2022) Appendix F. It ignores attention's `O(T²)` term
  (fine below ~4 k context) and is wrong for MoE (use activated
  parameters instead).
- Kaplan et al. (2020) established that LM loss is a power law in `C`,
  `N`, and `D` — smooth, predictable, and monotone across the range of
  production training.
- Chinchilla (Hoffmann et al., 2022) refined the compute-optimal split
  to `D ≈ 20 · N`. Above that, you are overtrained-for-inference; below,
  you are under-trained. Most production runs deliberately overtrain
  (Llama 3, MPT-7B) because inference cost dominates deployment
  economics.
- A product target — parameter count and quality bar — turns into a
  compute budget `C` in FLOPs in three lines of arithmetic. That number
  is the input to every other decision in this module: cluster shape
  (chapter 2), buying mix (chapter 3), MFU-uplift ROI (chapter 4),
  feasibility study (chapter 5), and anchor recipe (chapter 6).
- Defend `C` to within a factor of two, not to the second significant
  figure. Scaling-law fits do not warrant more precision than that at
  budgeting altitude.
