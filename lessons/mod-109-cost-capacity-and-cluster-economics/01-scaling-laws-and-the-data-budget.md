# Scaling Laws and the Data Budget

Every module in this track so far has treated the training run as a
given: the parameter count `N` and token count `D` were inputs, and
your job was to turn them into a stable, efficient training loop.
This module inverts the question. A product team walks in with a
target — "a 34B chat model that beats our current baseline on our
internal eval" — and asks *what does this cost, and by when can we
have it*. The very first arithmetic between that request and a
budget is picking `N` and `D`, and the tool for that is the
**neural-network scaling laws**.

The wrong pick here is very expensive. Under-training a large model
(too many parameters for the data budget) wastes hardware; over-
training a small model (too many tokens for the parameter count)
wastes wall-clock. Kaplan et al. (2020) and Hoffmann et al. (2022 —
the "Chinchilla" paper) gave the field two calibrations of the
same underlying trade-off. This chapter is the one you use before
any dollar figure gets written down.

## The compute ledger: `C ≈ 6 · N · D`

Both scaling-law papers rest on one accounting identity, the same
one mod-107 chapter 1 used for MFU. For a dense decoder-only
Transformer trained on `D` tokens with `N` non-embedding
parameters, the total training FLOPs cost is approximately

```
C ≈ 6 · N · D
```

The `6` decomposes as `2 (forward) + 4 (backward, one for input
gradients, one for weight gradients)`, each of which is `N · D`
FLOPs across the run. The attention term is separately quadratic
in sequence length but drops out of the *ratio* arguments the
scaling laws use, so it is usually written as a correction and
folded into the effective `N` for accounting.

Two useful re-arrangements:

- `D = C / (6 · N)` — given a compute budget `C` and a fixed
  parameter count `N`, how many tokens can you afford.
- `N = C / (6 · D)` — given a compute budget `C` and a fixed data
  budget `D`, how big can you make the model.

The scaling laws are what tell you the optimal *ratio* between
those two knobs when both are free to move. See Hoffmann et al.
(2022) equation 3, arXiv 2203.15556, for the derivation.

## Kaplan (2020): loss scales as a power law in `N`, `D`, `C`

Kaplan, McCandlish, et al. (2020, arXiv 2001.08361) fit
power-law relationships between test loss and the three axes:

- `L(N)` — loss as a function of parameter count, with data and
  compute unconstrained.
- `L(D)` — loss as a function of dataset size, with parameters
  and compute unconstrained.
- `L(C)` — loss as a function of compute, with `N` and `D` chosen
  optimally.

Their empirical claim: each axis had a clean power-law fit over
several orders of magnitude, and the compute-optimal recipe put
more of the marginal compute into *larger models* than into
*more data*. The specific advice was to grow `N` roughly as
`C^0.73` and `D` roughly as `C^0.27` — parameters much faster
than tokens.

This became the field's default for a couple of years and
motivated GPT-3-era runs that were parameter-heavy relative to
their data budgets. The Kaplan paper is still the right first
reference for the *shape* of the argument: loss is a smooth
power law over the ranges we care about, and there is such a
thing as a compute-optimal `(N, D)` for a given `C`.

## Chinchilla (2022): revise the ratio to roughly `20 tokens per parameter`

Hoffmann, Borgeaud, et al. (2022, "Training Compute-Optimal
Large Language Models", arXiv 2203.15556) redid Kaplan's fit
with a wider sweep of model sizes and training-token counts,
and a more careful learning-rate schedule that decays to a
small value at the target token count. Their revised claim: for
a fixed compute budget `C`, the loss-minimising choice of `N`
and `D` grows them at roughly equal rates —

```
N ∝ C^0.5,   D ∝ C^0.5
```

which in practice comes out to a ratio of roughly

```
D / N ≈ 20 tokens per parameter
```

for the compute-optimal frontier. The Chinchilla model itself is
the empirical anchor: 70 B parameters trained on 1.4 T tokens
(exactly the 20:1 ratio), outperforming the 280 B-parameter
Gopher model that had been trained on only 300 B tokens (a 1:1
ratio). See Hoffmann et al. §3 (three approaches all converging
on the same conclusion) and table 3 for the fitted constants.

The "20 tokens per parameter" number is a rule of thumb, not a
theorem. The paper's own three fitting approaches produce
slightly different exponents (approach 1: `~19:1`, approach 2:
`~20:1`, approach 3: `~20:1`). Community follow-ups (see
Muennighoff et al. 2023 "Scaling Data-Constrained Language
Models" arXiv 2305.16264, and the DeepMind / Anthropic / Meta
follow-on work) have refined the constants for specific data
mixtures, tokenisers, and architectures — but the *shape* of the
conclusion, "grow `N` and `D` together, not `N` alone", has held
up.

## Compute-optimal ≠ inference-optimal

The Chinchilla frontier minimises training loss for a fixed
training compute budget. That is not the same as minimising the
total lifetime cost of the model when you factor in serving.

The intuition: a model that is used for `T` inference tokens over
its lifetime pays FLOPs proportional to `2 · N · T` on the
inference side. If `T` is large — the model is going to be
served heavily — then reducing `N` at the cost of *more training
tokens per parameter* buys you cheaper inference for the whole
serving lifetime. This is why every post-2023 open-weight
release is *deliberately* trained past the Chinchilla point.

Concrete anchors:

- **Llama 1 (Touvron et al. 2023, arXiv 2302.13971)**: 7 B / 13 B
  / 33 B / 65 B on 1 T / 1 T / 1.4 T / 1.4 T tokens. The 7 B
  model at 1 T tokens is `~143` tokens per parameter — roughly
  7× the Chinchilla-optimal ratio. Meta's stated motivation
  (§ 1): "the focus of this work is to train a series of language
  models that achieve the best possible performance at various
  inference budgets".
- **Llama 3 (Grattafiori et al. 2024, arXiv 2407.21783)**: 8 B /
  70 B / 405 B on ~15 T tokens each (see § 3). The 8 B model is
  `~1875` tokens per parameter — roughly 90× Chinchilla-optimal.
  This is an aggressive inference-optimal recipe: heavy
  investment in training compute to make each future inference
  call cheaper.
- **Chinchilla-optimal reference (Hoffmann 2022)**: 70 B on 1.4 T
  tokens, exactly 20:1. This is the *training-optimal* baseline
  the field still measures against.
- **MPT-7B (MosaicML 2023, cost-optimised recipe)**: 7 B on 1 T
  tokens, roughly 143:1. Chapter 6 uses this as one of the two
  anchor recipes for the scaling-down conversation. MosaicML's
  blog post on MPT-7B is a public reference for its cost figure
  and recipe details.

The lesson: the scaling laws give you the training-optimal
*floor* — do not train a large model on fewer tokens than
Chinchilla says. But the actual `D / N` ratio you pick should be
above that floor by a factor set by the serving-vs-training
economics of the product. Chapter 5 formalises the serving-
lifetime term; chapter 6 works both anchors end-to-end.

## From product target to `(N, D)`: the decision list

The concrete procedure a training-pipeline engineer runs when
asked "what's the data budget for the 34 B model":

### 1. Get the parameter count `N` from the product side

The product team owns the *what* of the model — the parameter
count, architecture family, tokenizer. Your input is the *cost*
of each choice. If they hand you "somewhere between 30 B and
40 B parameters", turn it into a range and price both ends.

### 2. Pick the `D / N` ratio from the serving-lifetime lens

- If the model is a *research artifact* or a checkpoint that
  will not be served heavily (fine-tuning base, ablation), the
  Chinchilla-optimal 20:1 is the right ratio.
- If the model will be *served at scale* (millions of user
  tokens per day for a year+), pick a ratio well above 20:1.
  The Llama 1 recipe (~140:1 for a 7 B) and the Llama 3 recipe
  (~1800:1 for an 8 B) are the public anchors. See chapter 5
  for the arithmetic.
- If you are *data-constrained* — you do not have enough unique
  high-quality tokens to hit the ratio — Muennighoff et al.
  (2023, arXiv 2305.16264) fits the correction for repeating
  data. Do not silently ignore the constraint; document it in
  the feasibility study (chapter 7).

### 3. Compute the FLOP budget `C = 6 · N · D`

This is the single number chapter 2 turns into GPU-hours. Do
this in isolation from the cluster shape — you want to know the
compute cost independent of what hardware you happen to have.

### 4. Sensitivity-check by re-doing the math at `N ± 20%`

Parameter counts are approximate at design time (embedding
count, tokenizer size, GLU vs. non-GLU MLP width all move `N`
by ±5–15%). Recompute `C` at the low and high ends of the
plausible `N` range. If the answer is that `C` moves by 40% and
therefore the dollar cost moves by 40%, either the product team
needs to freeze `N` more tightly or the feasibility study needs
to hedge its dollar range accordingly.

### 5. Freeze the recipe before writing the dollar number

A dollar figure attached to a floating `(N, D)` is not a
budget; it is a wish. Every subsequent chapter of this module
assumes `(N, D)` are fixed, and every reader downstream of the
feasibility study will treat the dollar number as a
commitment. Do the freezing here, in writing, before doing the
GPU-hours conversion.

## A worked example

The product team wants a 34 B-parameter dense decoder-only chat
model. They expect to serve it at a rate of `~50 M user tokens
per day` (roughly 500 K daily active users × 100 tokens each).
The model will run for at least 12 months in production before
being replaced by a next-generation model.

### Chinchilla-optimal (training-optimal) recipe

- `N = 34 · 10^9`
- `D = 20 · N = 6.8 · 10^11 = 680 B` tokens
- `C = 6 · N · D = 6 · 34e9 · 680e9 ≈ 1.39 · 10^23` FLOPs

### Inference-optimal recipe (following Llama 1's factor)

Pick `D / N = 140` (roughly Llama 1 7 B's ratio, applied to a
34 B model):

- `N = 34 · 10^9`
- `D = 140 · N = 4.76 · 10^12 = 4.76 T` tokens
- `C = 6 · N · D ≈ 9.71 · 10^23` FLOPs

The inference-optimal recipe is roughly 7× the training compute.
Chapter 5 shows how to check that trade-off against the serving-
lifetime FLOPs — for a 12-month serving window at 50 M user
tokens per day, cumulative inference FLOPs are

```
2 · N · T = 2 · 34e9 · (50e6 · 365) ≈ 1.24 · 10^21 FLOPs
```

which is only ~0.9% of the training-optimal budget and ~0.1% of
the inference-optimal budget. In this case, the inference-
lifetime term is small compared to training and the extra
training compute for a lower-`N` model would not pay for itself.
The right recipe here is closer to Chinchilla-optimal than to
Llama 3-scale. Chapter 5 walks through the same calculation for
a heavier-serving product where the answer flips.

Write both recipes into the feasibility study; do not commit to
one before chapter 5's dollar-lens comparison and chapter 6's
recipe anchors have been layered on.

## Failure modes to avoid

- **Publishing a Chinchilla number without noting it is training-
  optimal, not inference-optimal.** Every real product that
  will be served past a handful of users trains past Chinchilla.
  Presenting Chinchilla-optimal as *the* target biases the team
  toward under-training.
- **Copy-pasting the Kaplan `C^0.73 / C^0.27` split.** Superseded
  by Chinchilla for anything but historical comparison; using
  it produces parameter-heavy, data-starved runs.
- **Applying `6 · N · D` to an MoE architecture without
  correction.** The `N` in the formula is *active* parameters,
  not total. For MoE, use the number of experts per token
  routing pattern. Getting this wrong is a common source of
  4×–8× cost errors on MoE runs.
- **Applying `6 · N · D` to a non-Transformer architecture
  (state-space model, RWKV, hybrid) without redoing the
  accounting.** The `6` factor is Transformer-specific. Redo
  from first principles.
- **Ignoring embeddings and tokenizer size.** `N` is
  conventionally non-embedding parameters. For small models
  (~1 B and under) with a large tokenizer (~100 K+ tokens),
  the embedding table is a substantial fraction of total
  parameters and materially moves `D` under a Chinchilla
  computation. Say which convention you used.
- **Treating scaling-law constants as universal across data
  mixtures and tokenisers.** The Chinchilla constants were fit
  on a specific data mix; a very different corpus (heavy code,
  low-quality web scrape) will have different constants. Use
  Chinchilla for the *shape* and internal ablations to refine.

## Summary

- The compute ledger `C ≈ 6 · N · D` is the single identity
  every subsequent chapter of this module rests on. Know it
  cold.
- Kaplan (2020) gave the field the first power-law fit; Chinchilla
  (Hoffmann 2022) revised it to the "grow `N` and `D` together,
  roughly 20 tokens per parameter" recipe that is the current
  default for the training-optimal frontier.
- Compute-optimal ≠ inference-optimal. Real product runs
  deliberately train past the Chinchilla point to shrink the
  parameter count for serving. Llama 1 (~140:1) and Llama 3
  (~1800:1 at 8 B) are the public anchors for inference-
  optimal recipes; chapter 6 uses them as case studies.
- The training-pipeline engineer's procedure: get `N` from the
  product team, pick `D / N` from the serving-lifetime lens,
  compute `C`, sensitivity-check, then freeze the recipe before
  writing a dollar number.
- The output of this chapter is `(N, D, C)` — three numbers with
  units and a defensible recipe. Chapter 2 converts `C` to GPU-
  hours.
