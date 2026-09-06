# exercise-04: Reproducibility Bundle Spec + Emitter

**Estimated effort:** 3 hours

## Objective

Turn chapter 5 into a concrete reproducibility bundle: a versioned
JSON schema, a rank-aware emitter the training loop calls at run
start, a verify-at-emit harness, and a `bundle-diff` CLI a
teammate can point at two bundles six weeks apart to attribute a
divergence. Deliverable: a specification document, a Python
package that emits and verifies bundles, an example bundle
produced by a real (or realistic) run, and a written team-level
reproducibility contract.

## Prerequisites

- Chapter 5 of this module. The seven fields, three tiers, and
  the emit/verify recipe are the spec you are implementing.
- Chapter 1 (the anti-pattern that motivates the bundle).
- A training loop or a small representative script you can hook
  into at run start; the loop from exercise 3 works. Nothing
  about the exercise requires a real large-scale run — the emitter
  is what you are demonstrating.
- Access to `docker` or `crane` for resolving container tags to
  digests (chapter 5 field 6).
- Familiarity with the PyTorch reproducibility notes
  (https://pytorch.org/docs/stable/notes/randomness.html) — the
  source of the seed-manifest structure.

## Problem statement

Chapter 5 argued that the config file alone does not reproduce
the run, and named seven fields the bundle must hold: seed,
config, dataset hash, tokenizer hash, framework versions,
container digest, hardware manifest. Every field is there because
a specific past failure would have been caught by it. This
exercise is the operational cash-out — build the emitter, verify
it works, and ship the artifact the metadata store (chapter 7)
records.

You will also decide which reproducibility tier your team
targets (bitwise / loss-curve equivalence / recipe equivalence),
and design the bundle so a `bundle-diff` against a re-run's
bundle tells you whether the deviations are compatible with the
tier or not.

## Requirements

Ship one directory:

- `repro_bundle/`
  - `SPEC.md` — the versioned schema specification.
  - `pyproject.toml` (or `setup.py`) — packages the module.
  - `repro_bundle/__init__.py` — the emitter and verify
    functions.
  - `repro_bundle/collectors/` — one file per collector (seed,
    config, dataset, tokenizer, framework, container, hardware).
  - `repro_bundle/cli.py` — the `bundle-diff` CLI.
  - `tests/` — unit tests for each collector against synthetic
    inputs.
  - `examples/bundle.example.json` — a real bundle emitted by
    running the code against a small training loop.
  - `CONTRACT.md` — the team-level reproducibility contract.

### 1. The specification (`SPEC.md`)

A written spec (2–4 pages) that:

- Names the current `schema_version` (start at `1`).
- Enumerates every field chapter 5 requires, with:
  - Its JSON path in the bundle (e.g., `dataset.manifest_hash`).
  - Its type and format (SHA-256 hex, ISO-8601 UTC timestamp,
    OCI image digest `sha256:...`, etc.).
  - Its required / optional status.
  - The specific failure mode it defends against (see chapter 5
    "failure modes and how the bundle catches them").
- Names the three reproducibility tiers and which fields are
  load-bearing for each.
- Documents the schema-versioning policy: how to add a field
  (bump `schema_version`, keep readers backwards-compatible for
  N previous versions), how to deprecate a field.
- Names the expected-to-differ allow-list for `bundle-diff`
  (hardware UUIDs, timestamps, `run_id`).

### 2. The emitter (`repro_bundle/__init__.py` + collectors)

A Python package the training loop imports:

```python
from repro_bundle import emit_bundle, verify_at_emit

if dist.get_rank() == 0:
    bundle_uri = f"s3://runs/{run_id}/bundle.json"
    emit_bundle(run_id=run_id, output_uri=bundle_uri, tier="loss_curve")
    verify_at_emit(bundle_uri)  # fails the run at step 0 if the bundle is broken
```

`emit_bundle` must:

- Call each of the seven collectors (each in its own file under
  `collectors/`).
- Include `schema_version` and `run_id` at the top of the JSON.
- Write atomically (`write_atomic`) — temp path + rename. A
  half-written bundle must not be observable.
- Emit only from rank 0. Every rank contributes its slice
  (hardware manifest, sampler fingerprint); rank 0 gathers and
  writes.

Each collector implements chapter 5's field:

- **`seed.py`** — the seed manifest. `random`, `numpy`,
  `torch` (CPU + CUDA per-device), `DataLoader.worker_init_fn`
  derivation, `DistributedSampler.seed`. Record each subsystem →
  seed value.
- **`config.py`** — the raw config file bytes, the git commit +
  remote of the training-code repo, and a filtered env-var dump
  (allow-list of vars that actually matter: `TORCH_NCCL_*`,
  `NCCL_ALGO`, `NCCL_PROTO`, `CUDA_LAUNCH_BLOCKING`, etc.).
- **`dataset.py`** — the manifest hash, the sampler-order
  fingerprint (first N `(shard_id, offset)` tuples), and the
  first-batch token hash. Design the collector so the manifest
  hash is computable without loading all shards into memory.
- **`tokenizer.py`** — the tokenizer artifact URI, SHA-256 of
  its bytes, vocab size, and a test-vector hash (tokens for
  "The quick brown fox jumps over the lazy dog.").
- **`framework.py`** — `torch.__version__`, `torch.version.cuda`,
  `torch.cuda.nccl.version()`, `torch.backends.cudnn.version()`,
  Python version, full requirements-lock content, and a
  `canary_kernel_hash` from a fixed matmul + attention. The
  canary hash is the same test chapter 6 signature 5's detector
  will use.
- **`container.py`** — the OCI image digest (`sha256:...`), not
  the tag. If not running in a container, records
  `container.mode = "bare_metal"` plus OS release, kernel
  version, and the full requirements lock.
- **`hardware.py`** — per-node CPU / DRAM / kernel / NIC
  firmware; per-GPU model / UUID / firmware / topology
  (`nvidia-smi topo -m`); fabric switch identifiers if
  discoverable; NCCL topology dump (from `NCCL_TOPO_DUMP_FILE`).

### 3. `verify_at_emit`

Runs immediately after `emit_bundle`. Loads the bundle back from
storage, walks every field, and confirms:

- Every referenced artifact URI is reachable (config file,
  tokenizer file, dataset manifest).
- Every hash matches the referenced bytes when re-computed.
- The `canary_kernel_hash` matches when re-run.
- The `schema_version` matches the emitter's version.

If any check fails, the training loop terminates at step 0 with
a clear message. A silent bundle-emission bug is worse than a
loud crash at step 0.

### 4. `bundle-diff` CLI

A CLI exposed via `pyproject.toml`'s `[project.scripts]`:

```
$ bundle-diff runs/run-abc123/bundle.json runs/run-def456/bundle.json --tier loss_curve
```

Behavior:

- Loads both bundles, walks every field.
- Prints one line per field: `EQUAL`, `DIFFERENT`, or `SKIPPED`
  (for allow-listed expected-to-differ fields).
- Groups `DIFFERENT` lines by whether they are compatible with
  the requested tier. A framework minor version bump is a
  known deviation for tier 3 but a blocker for tier 1.
- Exits `0` if all differences are compatible with the tier,
  non-zero otherwise. Wire this into CI for re-runs.

### 5. Example bundle (`examples/bundle.example.json`)

Emit a real bundle by running the emitter against a small
training loop. Redact anything sensitive (internal hostnames,
private image registries) with `<redacted>` markers, but leave
the structure intact.

### 6. The reproducibility contract (`CONTRACT.md`)

A short (1–2 pages) document the team can adopt:

- Every production run emits a bundle. No bundle, no run — the
  emitter is called before step 0 and blocks on failure.
- Every checkpoint records its `bundle_uri`.
- Every downstream consumer (fine-tuning, eval, model-registry
  push) records the bundle URI its input was built against.
- The reproducibility tier is declared per run and recorded in
  the bundle.
- Bundle format changes are versioned; readers accept at least
  the previous version.

## Starter guidance

- **Start with the schema, not the emitter.** Write `SPEC.md`
  first, agree on the fields and types, then write collectors
  against the spec. Otherwise every collector negotiates its own
  format and the diff CLI is unwritable.
- **Rank 0 writes; everyone contributes.** The hardware manifest
  needs one slice per rank (each rank knows its own GPU UUID);
  the sampler fingerprint is rank-local by design; the seed
  manifest is global. Use `dist.gather_object` to collect the
  slices at rank 0. Do not try to make rank 0 introspect the
  other ranks.
- **`write_atomic` is not optional.** A run that crashes mid-
  bundle-emission and leaves a partial JSON file is worse than
  one with no bundle. Write to `<uri>.tmp` and rename; on cloud
  storage, use the provider's atomic-copy semantics.
- **Use content digests, not tags, for containers.** `docker
  inspect <tag>` gives you the digest. Register the digest in
  the bundle; the tag can be additionally recorded but is not
  authoritative.
- **Test the diff CLI against a known-different pair.** Run the
  emitter twice on the same host and confirm the "everything
  equal except hardware UUIDs and timestamps" case works.
  Change something (git branch, seed, one env var) between runs
  and confirm the diff surfaces it in the right bucket for the
  tier.
- **Do not treat the tracker (chapter 4) as the bundle store.**
  The tracker is a discovery layer. The bundle URI in object
  storage is the source of truth; the tracker just points to it.

## Acceptance criteria

- `SPEC.md` exists and enumerates all seven chapter 5 fields
  with type, format, required/optional status, and the failure
  mode each defends against.
- The Python package installs cleanly (`pip install -e .`) and
  the tests pass.
- `emit_bundle` writes a valid JSON bundle atomically from rank
  0 in a multi-rank run; each collector produces the expected
  field group.
- `verify_at_emit` catches at least two synthetic emitter bugs
  (a tampered hash, a missing referenced file) — demonstrate
  these in the tests.
- `bundle-diff` runs against two example bundles, groups
  differences by tier compatibility, and exits with the right
  code.
- `examples/bundle.example.json` is committed and validates
  against `SPEC.md`.
- `CONTRACT.md` names the emit-before-step-0 rule, the
  checkpoint linkage, the downstream-record rule, and the
  schema-version policy.
- The `canary_kernel_hash` is the same fixed test chapter 6's
  SDC detector will use — name the shared test explicitly in
  the spec.

## Stretch goals

- Add a `bundle-lint` CLI that walks a bundle and warns on
  weak-signal fields (config uses a git commit not on `main`;
  container is a `latest` tag; a required env var was recorded
  as empty). Wire it into the pre-run hook so the bundle is
  reviewed before the run starts spending GPU-hours.
- Emit the bundle to a content-addressed store — hash the
  bundle bytes and write to `bundle-store/<sha256>/bundle.json`
  in addition to the run-scoped URI. Deduplicates identical
  bundles from repeated runs and makes bundle-provenance
  queries easy.
- Sign the bundle with `cosign` or another signing tool and
  verify the signature in `verify_at_emit`. Downstream teams
  now know the bundle they pulled was the bundle the training
  loop emitted (not one edited by hand).
- Register the bundle URI into a metadata store (chapter 7's
  `run` table) via the same emitter code path. Now the
  fine-tuning team's query for "which bundle was checkpoint X
  built under" returns the URI directly.
- Extend the collector for a *fabric* fingerprint beyond NCCL's
  topology dump: query the InfiniBand subnet manager (SM) for
  the LID-to-node map, or the Ethernet fabric's LLDP
  neighbourship graph. A fabric-topology change during a re-run
  is one of chapter 5's failure modes; capture it explicitly.
