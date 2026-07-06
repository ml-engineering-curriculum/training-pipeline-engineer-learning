# The Training-Data Mental Model: Feed Rate, Step Time, and the I/O Budget

Before we look at any specific loader (WebDataset, MosaicML Streaming, Ray
Data) it is worth building the mental model that every subsequent chapter
in this module hangs on. A training-data pipeline is not "a `DataLoader`
that returns tensors". It is a **subsystem that must sustain a byte-rate
matched to the compute rate of the accelerator fleet** — for as many
epochs as the run lasts, across restarts, without ever letting the GPUs
starve and without ever silently changing the sample stream.

This chapter defines the units, the constraints, and the failure surface
that the rest of the module operates in.

## Motivation: the dataloader as a compute-competing subsystem

The distributed-training foundations from mod-101 assumed that a training
step consists of forward + backward + all-reduce + optimizer, and that
every one of those pieces has a well-understood cost. There is a fifth
line item that mod-101 was silent on: **producing the next batch of
tokens**.

At the scale mod-101 name-checked (7 B → 175 B parameters, thousands to
hundreds of thousands of tokens per step per GPU) the loader can no longer
be treated as free.

Two numbers dominate:

- **Per-step compute time** `t_step` — the wall-clock cost of a full
  forward + backward + step, from mod-101. On an 8×H100 node running a
  Llama-3-class model this is typically in the low hundreds of
  milliseconds.
- **Per-step byte demand** `B_step` — the payload the loader must hand
  the model for that step: `global_batch_tokens × sizeof(sample)`. For
  pretraining, `sizeof(sample)` is dominated by the tokenized `input_ids`
  (typically 2 or 4 bytes per token depending on vocab size); for
  multimodal training it can be many kilobytes per sample.

The loader has to sustain `B_step / t_step` bytes per second **on
average**, and it has to keep the GPU queue non-empty even when
individual fetches spike. If either invariant is violated for even a
handful of consecutive steps you see it as GPU utilization dropping and
your throughput chart flattening.

The training-pipeline engineer's job is to guarantee both invariants
across a run that lasts weeks, on a fleet of hundreds to tens of
thousands of GPUs, over a corpus of tens to hundreds of billions of
tokens, without cheating on determinism.

## The three units that matter

Everything in this module reduces to three numbers you should be able to
compute from first principles:

1. **Sample throughput target** — `samples/sec` (or `tokens/sec`) that
   the loader must sustain to keep the accelerators fed. Derived from
   `global_batch_size / t_step`.
2. **Byte throughput target** — `bytes/sec` off the storage subsystem
   (S3, GCS, Lustre, WEKA, local NVMe). Derived from the sample
   throughput times the on-disk sample size.
3. **Working set** — how much data is *in flight* between "the shard
   sits at rest in object storage" and "the tensor is on the GPU". This
   is the total capacity you must budget across prefetch buffers,
   per-worker queues, staging cache, and the shuffle buffer.

The chapters that follow put concrete numbers behind these units for
each toolchain.

## The pipeline is a hierarchy of queues

A production training-data pipeline is a **series of producer / consumer
stages** with a queue between each pair:

```
S3 / GCS (cold object store)
      │  network read
      ▼
[stage A] on-cluster staging     (Lustre, WEKA, or node-local NVMe)
      │  local read + decode
      ▼
[stage B] loader worker process  (WebDataset pipeline, StreamingDataset iterator, ...)
      │  cross-process transport
      ▼
[stage C] per-rank prefetch buffer (`num_workers`, `prefetch_factor`)
      │  `.to(device)`
      ▼
[stage D] GPU input queue        (a batch or two of `input_ids` in HBM)
      │  training step
      ▼
GPU compute
```

Every arrow is a link with a bandwidth and a latency. Every stage is a
queue with a depth and an eviction policy. When the pipeline stalls,
it is because a downstream queue has drained and one specific arrow
above it was too slow. The debugging protocol in chapter 5 (failure
modes) is essentially "find which queue drained and which arrow
starved it".

Two useful invariants:

- **Compute time must dominate the deepest queue's drain time.** If
  the shuffle buffer holds 30 seconds of samples and the step time is
  0.3 s, you have 100 steps of runway when the upstream stalls. If it
  holds 3 seconds, you have 10 steps. Design for the *slowest expected
  disruption*, not the median.
- **Every stage must be able to backpressure.** A loader that silently
  drops samples when the GPU is fast is a determinism bug; a loader
  that silently duplicates samples when the GPU is slow is an epoch-drift
  bug. Chapter 5 shows both in the wild.

## Why "just use torch.utils.data.DataLoader with a big Dataset" fails

The single-node PyTorch tutorial pattern —

```python
dataset = MyMapStyleDataset("/mnt/data/")
loader = DataLoader(dataset, batch_size=B, num_workers=W, shuffle=True)
```

— falls over at training-platform scale for reasons that are worth
naming explicitly, because each one motivates a specific piece of
tooling later in the module:

1. **Random-access I/O over a corpus of billions of small files** is
   the worst-case pattern for both object stores and parallel
   filesystems. WebDataset and MosaicML Streaming exist to convert this
   into sequential shard reads. See chapters 2 and 3.
2. **A single global shuffle** is impossible over a 100 B-token
   corpus — you can't hold indices for all of it in memory, and you
   can't afford a random seek per sample. Both toolchains substitute a
   two-level scheme (shuffle shards globally, sample within a bounded
   window per worker). Shuffle collapse (chapter 6) is what happens
   when this bounded window is too small.
3. **`num_workers` fanout scales poorly** — every worker holds a
   full copy of the sample index and its own decoders, tokenizers, and
   file handles. At 128 workers × 8 ranks × 4 nodes this becomes a
   fork-bomb-shaped memory problem long before it becomes a throughput
   problem.
4. **Resume is broken by default**. `DataLoader` with `shuffle=True`
   does not persist its state across an interrupt. If your training
   job dies at step 400 000 of 1 M and you restart, you have no
   correctness guarantee about which samples you've seen. Chapter 3
   walks through the state that MosaicML Streaming persists to make
   resume actually deterministic.
5. **Tokenization is not a per-sample step at scale.** For a ~100 B
   token corpus, tokenizing on-the-fly in the loader worker is
   ~100 B tokenizer calls per epoch, which is neither cheap nor
   reproducible. You tokenize **once** offline, in a distributed job,
   with the tokenizer's ID and hash pinned. Chapter 4 (Ray Data) is
   how that offline job is built.

Every subsequent chapter in this module is either "how does toolchain X
solve one of these problems" or "here is the failure mode you get when
you don't".

## The two invariants you must not violate

Almost every serious data-pipeline incident in a production training
run traces back to a violation of one of these two invariants:

**Invariant 1 — determinism.** Given a fixed corpus, tokenizer, and
seed, two runs of the same job (including a resumed run) must
consume the same sequence of samples in the same order. Anything less
and you cannot reproduce a loss curve, cannot bisect a regression,
and cannot cleanly diff a checkpoint against its predecessor.

**Invariant 2 — coverage.** Over one epoch, every sample in the
declared corpus appears exactly the number of times the sampler
promises. This sounds trivial. In practice it's violated by:
mid-run shard failures that are retried non-deterministically;
`drop_last=True` interacting with sharded samplers; workers that
silently skip corrupted samples; and shuffle windows that pull from
"whatever is loaded so far" rather than the declared shard set. Epoch
drift (chapter 6) is the diagnostic name for this.

Every design decision in this module — shard layout, sampler,
prefetch, staging — is evaluated against these two invariants first
and throughput second. Throughput without determinism is not a
training pipeline; it is a random-batch generator.

## The scope of this module

This module is opinionated about the modern stack: WebDataset for
image/multimodal, MosaicML Streaming for LLM pretraining, Ray Data
for the offline tokenization / dedup job, and a
manifest-hash-runbook discipline layered on all three. It does *not*
survey every loader; NVIDIA DALI, `tf.data`, `torchdata` v1, HF
`datasets.Dataset.to_iterable_dataset`, and Grain are all real and
all used in production, but they compose with the same mental model
this chapter builds. If you understand chapters 2–6 you can port the
patterns to any of them.

## Summary

- A training-data pipeline is a compute-competing subsystem with a
  strict byte-rate contract.
- The pipeline is a stack of queues; a stall is always a specific
  queue draining and a specific arrow starving it.
- Determinism and coverage are non-negotiable invariants — every
  toolchain in this module is evaluated on them before throughput.
- The naive `DataLoader` pattern fails at scale on five specific
  axes; each axis is the motivation for a later chapter.
