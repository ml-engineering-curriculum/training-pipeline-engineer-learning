# The Reproducibility Bundle

Chapter 1's third anti-pattern was "we can reproduce the run from the
config alone". This chapter is the specification of what a bundle
actually needs to hold so that six months from now, on possibly-
different hardware and possibly-different framework versions, a
teammate can re-run the recipe and get either a bitwise-identical or
a statistically-equivalent result depending on the reproducibility
tier you targeted.

Two motivations, both practical:

- **Debugging six weeks after the fact.** A quarterly retrospective
  needs to A/B a good run against a bad run. Without a bundle, you
  are guessing at what differed.
- **Downstream teams (fine-tuning-engineer, model-evaluation-
  engineer) picking up the weights.** Their eval and fine-tune
  runs need to know what the base model was trained under. Chapter
  7 turns this bundle into the schema they query.

The bundle is *not* the checkpoint. The checkpoint is what was
trained; the bundle is *how* it was trained. Both are load-bearing,
both need durable storage, and the bundle is the smaller and cheaper
of the two.

## The three tiers of reproducibility

Before enumerating the bundle contents, agree on what "reproducible"
means. There are three tiers, in decreasing order of strictness:

### Tier 1: bitwise reproducibility

Running the recipe again on identical hardware, with identical
software versions and the same seed, produces the *same weights bit-
for-bit*. This is achievable for small runs; it is essentially
unachievable at multi-thousand-GPU pretraining scale because
non-associativity of floating-point addition combined with NCCL's
non-deterministic reduction ordering means every all-reduce has a
possibly-different result. See the PyTorch reproducibility docs at
https://pytorch.org/docs/stable/notes/randomness.html for the
concrete list of sources.

Bitwise reproducibility is a target for *unit tests* of a training
step, not for full runs. Chapter 6's silent-corruption detection
uses it for a small "canary" workload.

### Tier 2: loss-curve equivalence

Running the recipe again produces a loss curve that stays within a
tolerance band of the original for the duration of training. This is
the tier that most pretraining runs target. Achievable with:

- Same seed for all sources of randomness (PyTorch, NumPy, Python
  `random`, CUDA `torch.cuda.manual_seed_all`, the sampler's own
  seed).
- Same data ordering (the sampler is stateful and its state is
  checkpointed alongside the model — mod-106 chapter 3).
- Same framework version, same NCCL version, same fabric topology.
- Same hardware model (H100 vs. H200 has the same math throughput
  but different HBM bandwidth; a run may still land at the same
  loss curve, but demonstrate it, don't assume it).

The bundle's job is to record enough that a re-run can *demonstrate*
tier-2 equivalence. That requires more than a config file.

### Tier 3: recipe-equivalence

Running "the recipe" again produces a model that scores within a
tolerance band on the downstream eval suite. This is the loosest
tier and the most common in practice. Bit-exact weights are not
required, and even loss-curve deviations of a few percent are
acceptable if the eval outcomes agree.

Chapter 7's metadata store records which tier a given run targeted;
downstream consumers filter by tier when they need a specific one.

## The seven fields the bundle must hold

The learning objective lists these; here is the operational content
of each.

### 1. Seed

Not one seed — a *seed manifest*. Modern training loops draw from at
least four RNG sources:

- Python's `random` (via `random.seed`).
- NumPy's global RNG (via `numpy.random.seed`).
- PyTorch's CPU RNG (via `torch.manual_seed`).
- CUDA per-device RNG (via `torch.cuda.manual_seed_all`).

Plus, typically:

- The DataLoader's `worker_init_fn` — each worker has its own RNG
  and its seed must be a documented derivation from the master
  seed.
- The distributed sampler's seed (`DistributedSampler(seed=...)`).
- The optimizer's own RNG for cases where the optimizer draws
  random state (rare; Adam does not, but some auxiliary
  regularizers do).

Record all of them, either as one master seed and a documented
derivation function, or as an explicit map from `subsystem →
seed_value`. If you record only "we set seed 42", a re-runner does
not know what "seed 42" propagated to.

### 2. Config

The full recipe: model config, optimizer config, LR schedule,
batch size, sequence length, mixed-precision policy, activation-
checkpointing policy, sharding strategy, gradient-clipping
threshold, every knob the run consumes. The convention is a single
YAML or JSON file, immutable per-run, versioned in git. Modern
frameworks (torchtitan, torchrun scripts, Megatron, DeepSpeed) have
their own config formats; the bundle records the raw file, not a
parsed re-serialization.

Two useful additions:

- **The git commit hash and repo URL of the training-code repo.**
  A config that references an internal `MyModel` class is only
  reproducible if the class definition is pinned.
- **Any environment-variable overrides.** `TORCH_NCCL_*`,
  `CUDA_LAUNCH_BLOCKING`, `NCCL_ALGO`, and friends can silently
  change results if the re-runner has different defaults. Dump
  the full env-var set at run start; filter to a documented
  allow-list of "these actually matter".

### 3. Dataset hash

The dataset hash covers three levels:

- **The dataset manifest hash**: a hash over the list of shards
  (URIs plus sizes plus per-shard content hashes). If a shard is
  silently re-materialized on the parallel filesystem with a
  different byte layout, the manifest hash catches it.
- **The sampler order hash**: the sampler is stateful; its
  deterministic ordering must be reproducible from the seed. Hash
  the first N step's `(shard_id, offset)` tuples the sampler
  produces at initialization — this is a fingerprint of the data
  path the run *would* take.
- **The tokenized-batch hash for a sanity sample**: hash the first
  batch's tokenized content (bytes) so a re-runner can confirm
  their tokenizer is producing the same tokens the original run
  saw.

The failure mode this specifically defends against: a shard on
parallel FS whose contents changed silently between the original
run and the re-run. mod-103 covers the data pipeline; the bundle
records the fingerprint mod-103 emits.

### 4. Tokenizer hash

Tokenizers are versioned artifacts: a `tokenizer.json` (Hugging
Face's `tokenizers` library format), a SentencePiece `.model` file,
a Tiktoken vocab. The bundle records:

- **The tokenizer artifact URI and a SHA-256 hash of its bytes.**
- **The tokenizer's vocabulary size** (defends against a re-runner
  loading a tokenizer of the same name but a different version).
- **A test-vector hash**: tokenize a fixed string ("the quick brown
  fox…") and hash the resulting token IDs. This is a fingerprint
  that catches "the tokenizer changed but the vocab size did
  not".

Tokenizer versioning is under-invested in most teams. Chapter 7
turns it into a first-class metadata field.

### 5. Framework versions

Not just PyTorch. The bundle records a `requirements-lock.txt` (or
equivalent — Poetry lock, `uv` lock, `conda list`) that pins every
Python dependency to a specific version. In addition, the specific
versions of:

- **PyTorch** — `torch.__version__` plus the CUDA build variant
  string (`torch.version.cuda`).
- **NCCL** — `torch.cuda.nccl.version()`. NCCL algorithm choices
  can change results across minor versions.
- **CUDA runtime and driver** — from `torch.version.cuda` and
  `nvidia-smi --query-gpu=driver_version`.
- **cuDNN** — `torch.backends.cudnn.version()`.
- **Flash-Attention, Transformer Engine, and any hand-installed
  ops** — their pip metadata plus their git commit hash if
  installed from source.

The bundle also records the *result* of a short deterministic
kernel test: run a fixed matmul and a fixed attention on the
current stack and hash the output. Chapter 6's silent-corruption
detection uses the same test as an in-run health check.

### 6. Container digest

If the run ran in a container (Docker, Podman, Singularity,
Enroot), record the **content-addressable digest** — the SHA-256
`sha256:...` of the image manifest, not the tag. Tags are
mutable; a tag can be re-pushed pointing at different content, and
a `pytorch:24.06-py3` from March is not necessarily the same as a
`pytorch:24.06-py3` fetched in September if someone re-pushed
under the same tag.

The digest is what registry pulls resolve to. `docker inspect
<image>` and `crane manifest <image>` both surface it. Slurm sites
using Enroot or Singularity have equivalent digest concepts.

If the run ran outside a container (bare metal), record the
equivalent: OS release, kernel version, and the full
`requirements-lock.txt`.

### 7. Hardware manifest

The per-node inventory of the training gang:

- **Per-node**: hostname, CPU model, DRAM size, kernel version,
  NIC firmware version, PCIe topology (`nvidia-smi topo -m`).
- **Per-GPU**: model, UUID, firmware version, current NVLink
  topology, `nvidia-smi` snapshot at run-start.
- **Fabric**: switch identifiers the nodes are attached to, ideally
  down to rail-plane assignment (mod-105 chapter 3).
- **NCCL detected topology**: `NCCL_TOPO_DUMP_FILE` output; NCCL
  will emit its XML topology view if you set this env var. Pin
  this in the bundle so a re-run can catch a topology change.

The hardware manifest is the field most often skipped because it
feels tedious. It is also the field that most reliably catches
"the run failed to reproduce because we were on H100 SXM5 the
first time and H100 PCIe the second time and the fabric was
different".

## Emitting the bundle: the recipe

The bundle is emitted by the *training loop* at run start (before
the first step) and again at run end (with any final state). The
runtime pattern:

```python
def emit_bundle(run_id: str, output_uri: str):
    bundle = {
        "run_id": run_id,
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": collect_seed_manifest(),
        "config": {
            "raw": open(CONFIG_PATH, "rb").read().decode(),
            "git_commit": subprocess.check_output(
                ["git", "-C", REPO, "rev-parse", "HEAD"]
            ).decode().strip(),
            "git_remote": subprocess.check_output(
                ["git", "-C", REPO, "remote", "get-url", "origin"]
            ).decode().strip(),
            "env": dict(os.environ),
        },
        "dataset": {
            "manifest_hash": hash_manifest(DATA_MANIFEST_PATH),
            "sampler_first_100": fingerprint_sampler(sampler, n=100),
            "first_batch_hash": hash_first_batch(train_loader),
        },
        "tokenizer": {
            "uri": TOKENIZER_URI,
            "sha256": sha256_file(TOKENIZER_PATH),
            "vocab_size": tokenizer.vocab_size,
            "test_vector_hash": hash_tokens(
                tokenizer.encode("The quick brown fox jumps over the lazy dog.")
            ),
        },
        "framework": {
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "nccl": ".".join(str(v) for v in torch.cuda.nccl.version()),
            "cudnn": torch.backends.cudnn.version(),
            "python": sys.version,
            "requirements_lock": open("requirements.lock").read(),
            "canary_kernel_hash": run_canary_kernel_and_hash(),
        },
        "container": {
            "image_digest": os.environ.get("CONTAINER_DIGEST", "unknown"),
            "image_tag": os.environ.get("CONTAINER_TAG", "unknown"),
        },
        "hardware": collect_hardware_manifest(),
    }
    write_atomic(output_uri, json.dumps(bundle, indent=2))
```

`collect_seed_manifest`, `hash_manifest`, `fingerprint_sampler`,
`collect_hardware_manifest`, and `run_canary_kernel_and_hash` are
platform-specific helpers exercise 4 asks you to author.

Two operational details worth emphasizing:

- **`write_atomic`**. The bundle URI is the source of truth for
  the run; a half-written bundle is worse than no bundle. Write to
  a temp path and rename.
- **Only rank 0 writes**. Every rank collects the info it can (its
  own hardware manifest slice, the fingerprint of its shard). Rank
  0 gathers into the full bundle and writes.

## Verifying a bundle

A bundle you cannot verify is a bundle you cannot trust. Author two
verification harnesses alongside the emitter:

### Verify-at-emit

Immediately after `emit_bundle` returns, load the bundle back from
storage, walk its fields, and confirm every referenced artifact
(config file, tokenizer, dataset manifest) is reachable and has
the recorded hash. Fail the run at step 0 if the bundle is not
self-consistent. This is a cheap early-warning that catches most
emission bugs.

### Verify-for-rerun

A CLI that takes a bundle URI and an "attempt" run's bundle URI
and diffs them field by field:

```
$ bundle-diff runs/run-abc123/bundle.json runs/run-def456/bundle.json
seed: EQUAL
config.git_commit: DIFFERENT  (abc123 vs. def567)
config.env.NCCL_ALGO: DIFFERENT  (Ring vs. Tree)
dataset.manifest_hash: EQUAL
dataset.first_batch_hash: DIFFERENT  ← blocks tier-2 rerun
tokenizer.sha256: EQUAL
framework.torch: DIFFERENT  (2.4.0 vs. 2.5.1)
framework.nccl: DIFFERENT  (2.20 vs. 2.21)
framework.canary_kernel_hash: EQUAL
container.image_digest: DIFFERENT
hardware.gpus[0].uuid: DIFFERENT
```

Every "DIFFERENT" line is a possible source of divergence. A tier-2
rerun that expects loss-curve equivalence and shows any `DIFFERENT`
above the "expected-to-differ" allow-list (hardware UUIDs will
obviously differ across allocations) is doing so with known risk.

## Storage and retention

The bundle is small — kilobytes to low megabytes per run. Storage
choices:

- **Object storage** (S3, GCS, Azure Blob) with lifecycle policies.
  The URI pattern `s3://<bucket>/runs/<run_id>/bundle.json`
  becomes the join key the metadata store (chapter 7) records.
- **Git repository** for runs whose bundles you specifically want
  to code-review — a `bundles/` subtree of the training repo, with
  the bundle written as a PR when the run kicks off. Overkill
  for most runs; useful for the small set of "publishable" runs.
- **Content-addressed storage** — hash the bundle's own bytes and
  store at `bundle-store/<sha256>/bundle.json`. Deduplicates
  identical bundles from re-runs. Complicates the lookup path;
  usually not worth it for bundles.

Retention should be *at least* as long as the model weights are
retained. Deleting the bundle while keeping the weights leaves a
model in the store that no team can reproduce.

## The bundle is not the logbook

Distinct artifacts, both persistent:

- **Bundle**: emitted programmatically, immutable, describes
  starting conditions. Chapter 5's scope.
- **Logbook**: written by humans (and some automation), append-
  only, describes what *happened* during the run — the LR bumped
  down at step 152 000 after a spike, the node quarantined at step
  340 200. The OPT-175B chronicles are the reference (mod-106
  chapter 7). Lives in the metadata store as a linked artifact.

You need both. The bundle tells the re-runner what to try; the
logbook tells them what to expect and what to look out for.

## The reproducibility contract

A team-level contract that codifies the bundle's role:

- **Every production run emits a bundle.** No bundle, no run
  allowed to complete (the training loop refuses to start).
- **Every checkpoint records its bundle URI.** The metadata
  store's checkpoint row (chapter 7) has a `bundle_uri` column.
- **Every downstream consumer** (fine-tuning, eval, model-registry
  push) that references a checkpoint also records the bundle URI
  it was built against.
- **Bundle format changes are versioned.** `schema_version: 1`
  today; when you add a field, bump it and keep read compatibility
  for at least one previous version.

The contract's practical value is discoverability: the fine-tuning
team pulling a base model can look at the bundle URI in the
metadata store and instantly know what they are picking up.

## Failure modes and how the bundle catches them

Named, so you can defend the fields.

- **Silent shard re-materialization.** Caught by `dataset.
  manifest_hash` and `dataset.first_batch_hash`.
- **Tokenizer swap under the same name.** Caught by `tokenizer.
  sha256` and `test_vector_hash`.
- **PyTorch minor upgrade that changed a kernel.** Caught by
  `framework.torch` and `framework.canary_kernel_hash`.
- **Container tag re-push.** Caught by `container.image_digest`.
- **Fabric-topology change (rail-plane reshuffle).** Caught by
  `hardware` (specifically NCCL topology dump).
- **Env-var change that flipped an NCCL algorithm.** Caught by
  `config.env`.
- **A researcher's local git branch instead of trunk.** Caught by
  `config.git_commit` + `config.git_remote`.

Each failure mode above has, at some point, wasted weeks of a
platform team's post-mortem effort. Each is caught by a specific
bundle field. The fields exist because the failure modes exist.

## Summary

- The reproducibility bundle is the "how" the checkpoint's "what"
  points at. It is emitted per-run, kilobytes to low megabytes,
  stored in object storage with retention at least as long as the
  weights.
- Three reproducibility tiers: bitwise (small runs only), loss-
  curve equivalence (the practical target for pretraining), and
  recipe-equivalence (eval-score band). Bundle records the tier
  the run targeted.
- Seven fields: seed manifest, config (+ git commit + env),
  dataset hash, tokenizer hash, framework versions (+ canary
  kernel hash), container digest, hardware manifest.
- Rank 0 gathers and writes atomically at run start. Verify-at-
  emit catches emission bugs at step 0; verify-for-rerun is the
  diff CLI teammates use six weeks later.
- The bundle is not the logbook. The logbook is human-written
  and describes what happened; the bundle is programmatic and
  describes starting conditions. Both are load-bearing.
- The reproducibility contract makes the bundle unskippable: no
  run without a bundle, every checkpoint references its bundle
  URI, every downstream artifact records the bundle it was built
  against. Chapter 7's metadata schema encodes this contract.
- Every field defends against a specific failure mode observed
  in practice; skip a field and you accept its failure mode.
