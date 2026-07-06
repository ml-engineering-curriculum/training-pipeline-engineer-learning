# exercise-04: A Staging Layer Sized to Per-Step Compute

**Estimated effort:** 3 hours

## Objective

Design and measure a staging layer between an object store (tier 0)
and a training loader (tier 2, node-local NVMe) for a corpus that is
deliberately larger than the local hot tier. Prove that the staging
layer's byte-rate contract is met at the target per-step compute
time, characterize the failure mode when the working set exceeds
the cache, and pick a production configuration.

The deliverable is a working staging layer, a report showing
first-fetch latency, cache-hit rate, and observed step time under
three cache-size configurations, and a runbook stub for the layer.

## Prerequisites

- Chapter 5 of this module (chapter 1 for the byte-rate
  contract).
- A machine with node-local NVMe you can size (a scratch
  directory works as long as you can enforce a size limit).
- An MDS stream from exercise-02 or an equivalent WebDataset
  corpus large enough that the corpus does not fit on the
  local hot tier — target ratio ~2× (corpus is 2× the cache
  size).
- Ability to instrument the loader (or wrap it) to record
  per-fetch wall time.

## Problem statement

Your researcher's training loop has step time `t_step = 0.30 s`
and per-step byte demand `B_step = 5 MB` (a moderately sized
sample regime — bigger than tokenized text, smaller than raw
video). Their corpus lives in S3. Their box has 8 GPUs and
1.5 TB of local NVMe. The corpus is 3 TB.

Design a staging layer that keeps the training loop fed at
steady state, meets the byte-rate contract, and does not thrash
when the working set cycles.

## Requirements

1. **Compute the byte-rate contract.**
   - Derive the per-node byte throughput requirement from
     `t_step`, `B_step`, and 8 GPUs.
   - Derive a first-order tail-latency budget: the deepest
     downstream queue in your loader is the shuffle buffer or
     the `prefetch_factor`-sized queue; a single fetch must
     take less than the drain time of that queue at steady
     state.
2. **Implement a rolling per-node cache.**
   - Option A: use `StreamingDataset`'s built-in
     `local` + `cache_limit` with `predownload=N`.
   - Option B: implement your own cache manager as a small
     process alongside the loader (LRU + size limit +
     prefetch cursor).
   - Either way, record: shard fetch wall time (first-fetch
     latency), whether the fetch was a hit or miss, and
     cache utilization.
3. **Run three configurations.**
   - **Config S** — small cache: 500 GB (~17 % of corpus).
   - **Config M** — medium cache: 1.5 TB (~50 % of corpus).
   - **Config L** — large cache: 3 TB (full corpus fits).
   - For each, measure: median and p99 first-fetch latency,
     steady-state cache hit rate, GPU idle fraction, and
     total wall time to complete one epoch of a synthetic
     training loop.
4. **Characterize thrashing.**
   - Identify which config (if any) has a working set larger
     than the cache and see it in the metrics as low hit rate
     and elevated fetch wall time.
   - Report the ratio (working set / cache size) below which
     hit rate stays > 95 %.
5. **Object-store hygiene.**
   - Confirm your reader uses multi-part GETs for shards
     larger than 8 MB.
   - Confirm your shards are spread across multiple S3
     prefixes (or verify the loader is not saturating a single
     prefix's request budget).
6. **Runbook stub.** Ship a `staging.md` capturing:
   - The byte-rate contract you computed.
   - Cache size, eviction policy, `predownload` value.
   - Metric names for first-fetch latency, serve time, hit
     rate, and cache utilization.
   - How to alarm when the working set exceeds the cache.

## Starter guidance

- Instrument early. A tiny wrapper around the fetch call that
  logs `(shard_id, hit_or_miss, wall_ms)` is enough. Everything
  else can be derived from that log.
- Use a synthetic training loop that `sleep()`s for `t_step`
  and consumes the shard's samples in memory. You are testing
  the staging layer, not the training kernel.
- Do not conflate "cold-start" with "steady-state". Discard the
  first N shards of every run when computing steady-state
  metrics.
- For the thrashing config, run for at least two full working-set
  cycles to see the drop clearly.
- If you don't have a real Lustre or WEKA mount, tier 1 is out
  of scope for this exercise; tier 2 (node-local NVMe) is
  enough to see the pattern.

## Acceptance criteria

- The byte-rate contract is computed with units and shown in
  the report.
- All three configs run against the same corpus and the same
  synthetic training loop; only the cache size differs.
- The report shows the hit-rate curve as a function of
  (working set / cache size) and identifies the knee.
- Config L achieves > 99 % hit rate at steady state and
  meets the tail-latency budget.
- Config S demonstrates a measurable degradation, and the
  report attributes it to cache thrashing with evidence
  (rising p99 first-fetch, rising GPU idle time).
- Runbook stub includes metric names that a mod-108-style
  observability pipeline could consume without modification.

## Stretch goals

- Add a coordinated prefetch daemon: a small process that reads
  the loader's declared shard schedule from `index.json` and
  prewarms the cache ahead of the read cursor. Compare to
  the loader-driven cache.
- Simulate an object-store tail-latency spike (add jitter to
  the fetch path) and measure how the different configs
  handle it.
- Replace the LRU with LFU or with a "hot vs. warm" tier
  scheme; measure the difference and defend the trade.
- If you have Lustre or WEKA access, add tier 1 and repeat
  Config M with tier 1 as the shared warm cache.
- Show that a mis-sized cache (Config S) produces a *lower*
  effective object-store bandwidth than a well-sized one, due
  to request-count effects. Cite the S3 performance guide.
