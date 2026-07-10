# The Reproducibility Bundle

If chapter 3 answers "what did the run do while it ran?", this
chapter answers "how do we make it possible to run it again?" and
"how do we prove the run we say we did is the one we actually did?"
Both questions have the same answer: a **reproducibility bundle** —
a small, well-typed set of artifacts, produced by every run, that
downstream consumers (fine-tuning, evaluation, audit, the on-call
next week) can use to reproduce or verify the run without you
present.

The reproducibility bundle is the training-side counterpart to
mod-103 chapter 7's data-pipeline runbook. mod-103 owned the
corpus; this chapter owns the run. Together they cover the entire
"what was trained on what, and how" question.

Primary references:

- **PyTorch reproducibility guide:**
  https://pytorch.org/docs/stable/notes/randomness.html
- **HuggingFace tokenizers:**
  https://github.com/huggingface/tokenizers
- **OCI image specification:**
  https://github.com/opencontainers/image-spec
- **nvidia-smi query interface:**
  https://developer.nvidia.com/nvidia-system-management-interface
  <!-- needs-research: canonical URL for nvidia-smi -q -x reference -->

## What "reproduce" actually means

"Reproduce this training run" is under-specified. Three
progressively stronger meanings, each of which the bundle has to
support:

1. **Bitwise reproduction** — same weights out, byte for byte. On
   modern GPUs this is almost impossible: NCCL reductions are
   non-associative in floating point, kernel autotuning is
   non-deterministic, mixed-precision loss scaling depends on
   observed dynamic range, and some CUDA kernels only guarantee
   *within tolerance* determinism. See the PyTorch reproducibility
   notes.
2. **Statistical reproduction** — same loss curve within noise,
   same final task metrics within a few % relative. This is what
   the reproducibility bundle actually enables when the hardware
   is a match.
3. **Provenance reproduction** — you can *prove* the run described
   in the bundle is the run that produced weights W. Same seed,
   same config, same data, same tokenizer, same code, same
   container, same fleet shape → the audit passes. You may not be
   able to reproduce W bitwise, but you can prove W was produced
   under the recorded conditions.

The bundle supports 2 and 3 comprehensively, and 1 as much as the
underlying stack allows. When you tell a stakeholder "this run is
reproducible", you should be specific about which meaning applies.

## The bundle contents

Every training run emits exactly one bundle, at run finish (or on
elastic checkpoint), to a stable object-store prefix. The bundle
is a directory:

```
runs/<run_id>/
├── run_manifest.json      # the top-level record — see below
├── config/
│   ├── training.yaml      # the raw training config (hydra / omegaconf / JSON)
│   └── config.sha256      # sha256 of training.yaml
├── seeds.json             # seed set (see below)
├── data/
│   ├── shard_manifest.json     # from mod-103 chapter 7
│   └── tokenizer.json          # from HF tokenizers, or vendor equivalent
├── code/
│   ├── requirements.lock  # pip freeze / uv lock output
│   └── git.json           # {repo, commit, dirty: bool, diff: str}
├── container/
│   └── image.json         # {registry, image, digest, size}
├── hardware/
│   ├── nvidia_smi.xml     # `nvidia-smi -q -x` capture
│   ├── dcgm_diag.json     # `dcgmi diag -r 3 --json` capture
│   ├── nccl_info.txt      # NCCL_DEBUG=INFO opening banner
│   ├── topo.xml           # NCCL_TOPO_DUMP_FILE
│   └── nodes.json         # {node_name, fw_versions, driver, nccl, cuda}
└── metrics/
    ├── final_loss.json    # {train_loss, eval_loss_by_task, tokens_seen}
    └── tracker.json       # {tracker: wandb/mlflow, run_url, run_id}
```

Every file is small (all together ≤ a few MB per run). The bundle
is *not* the checkpoint — checkpoints live under the same run
prefix but in a different subdirectory (`checkpoints/step-N/`) and
are owned by mod-106.

## The run_manifest.json schema

The top-level `run_manifest.json` is the single index that a
downstream consumer reads first. It is small enough to inline:

```json
{
  "schema_version": "training_run.v1",
  "run_id": "3b-pretrain-2026-07-09-1234",
  "job_id": "slurm-45678",
  "started_at": "2026-07-09T14:22:04Z",
  "ended_at":   "2026-07-11T02:17:31Z",
  "status": "completed",
  "config": {
    "path": "config/training.yaml",
    "sha256": "1a2b3c…"
  },
  "seeds": {
    "path": "seeds.json",
    "sha256": "…"
  },
  "data": {
    "shard_manifest_sha256": "…",
    "tokenizer_sha256": "…",
    "tokens_seen": 300000000000,
    "epochs": 1.03
  },
  "code": {
    "repo": "github.com/acme/training",
    "commit": "abcdef01",
    "dirty": false,
    "requirements_lock_sha256": "…"
  },
  "container": {
    "registry": "nvcr.io",
    "image": "nvidia/pytorch:24.05-py3",
    "digest": "sha256:…",
    "size_bytes": 12345678900
  },
  "hardware": {
    "fleet_shape": {"nodes": 64, "gpus_per_node": 8, "gpu": "H100-SXM5-80GB"},
    "driver": "550.90.07",
    "cuda": "12.4",
    "nccl": "2.21.5",
    "flash_attn": "2.6.1",
    "transformer_engine": "1.7"
  },
  "metrics": {
    "final_train_loss": 1.834,
    "final_eval_loss": {"c4_val": 2.041},
    "mfu_p50": 0.44
  },
  "artifacts": {
    "checkpoint_uri": "s3://runs/3b-pretrain-2026-07-09-1234/checkpoints/step-100000/",
    "checkpoint_manifest_sha256": "…",
    "tracker_run_url": "https://wandb.ai/acme/pretrain/runs/3b-pretrain-2026-07-09-1234"
  }
}
```

Two invariants make this document useful:

- **Every referenced file's sha256 is included in the document.**
  A consumer can verify "I fetched what the manifest describes"
  in one pass through the fetched bytes.
- **Every hash is either verifiable from an official tool or
  transparently computed.** Container digest from the registry
  (`docker inspect` / `crane digest`), file hashes from `sha256sum`,
  shard hashes from mod-103's manifest, container/registry
  digests from OCI's canonical form.

Chapter 6 formalizes this schema as `training_run.v1.json` — the
hand-off contract with fine-tuning and evaluation.

## Seeds: the whole set, not just one

A common bundle bug: recording `seed: 42` in the config and calling
it done. That number is not enough. The seed set that actually
governs a PyTorch training run is:

```python
import os, random, numpy as np, torch

def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional determinism flags — trade throughput for repeatability.
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
```

Additionally, per-worker seeding of the DataLoader:

```python
def _seed_worker(worker_id: int):
    # PyTorch computes a base seed per worker; layer library seeds on top.
    base = torch.initial_seed() % 2**32
    random.seed(base + worker_id)
    np.random.seed(base + worker_id)

loader = torch.utils.data.DataLoader(
    ds, worker_init_fn=_seed_worker,
    generator=torch.Generator().manual_seed(seed),
    ...
)
```

`seeds.json` captures the whole picture:

```json
{
  "master_seed": 42,
  "python_hash_seed": "42",
  "torch_manual_seed": 42,
  "torch_cuda_manual_seed_all": 42,
  "numpy_seed": 42,
  "python_random_seed": 42,
  "dataloader_generator_seed": 42,
  "dataloader_worker_init_seed_offset": "torch.initial_seed() + worker_id",
  "torch_deterministic_algorithms": false,
  "cudnn_benchmark": true,
  "cudnn_deterministic": false
}
```

`torch_deterministic_algorithms` is the flag that most surprises
teams. When true, some ops (scatter, non-deterministic reductions,
FP16 accumulation) throw at runtime or fall back to slower paths.
When false, the run is faster but no longer bitwise-reproducible
even on identical hardware. Record which choice this run made.

## Dataset and tokenizer hashes

Refer out to mod-103 chapter 7 for the shard manifest format. The
bundle simply *includes the manifest* and its sha256; if the
consumer wants to verify shard-by-shard, they use mod-103's tools.

For the tokenizer, the canonical form for a HuggingFace tokenizer
is the `tokenizer.json` file it emits from
`AutoTokenizer.save_pretrained(...)`. The sha256 of that file is
the tokenizer's identity for the bundle. Vocab-file-only formats
(sentencepiece `spm.model`) hash the model file.

Two invariants your training loop asserts before step 0:

1. The corpus manifest's sha256 matches the value recorded in the
   config.
2. The tokenizer's sha256 matches the manifest's
   `tokenizer_sha256`.

Any mismatch fails the launch. This is mod-103's teeth applied to
the training side.

## Framework, driver, and library versions

The versions that matter for reproducibility are much larger than
`torch.__version__`. A minimum working set:

```python
import torch, sys, platform
version_info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_git_version": torch.version.git_version,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "nccl": ".".join(str(x) for x in torch.cuda.nccl.version()),
}
```

For the rest — `flash-attn`, `transformer-engine`, `deepspeed`,
`megatron-core`, `nemo` — dump `pip freeze` (or `uv pip freeze`)
into `requirements.lock`. The container digest already pins these,
so the freeze is redundant with the container; you keep both
because someone will eventually re-derive the container from the
lockfile.

## Container digest, not container tag

A container tag (`nvcr.io/nvidia/pytorch:24.05-py3`) is a mutable
alias. A container digest (`sha256:…`) is not. If your bundle
records `pytorch:24.05-py3` and the vendor re-issues that tag with
a driver bump, your bundle is silently wrong.

Two commands that give you the pinned digest:

```
# From a running node:
docker inspect --format '{{index .RepoDigests 0}}' <image>

# From the registry (no local pull):
crane digest nvcr.io/nvidia/pytorch:24.05-py3
```

Record both the tag (for human legibility) and the digest (for
verification). Consumers pull by digest.

The OCI image specification defines the digest scheme; see
https://github.com/opencontainers/image-spec. Every registry that
implements OCI produces the same digest for the same bytes,
which is why this is a stable identity across environments.

## Hardware manifest

The last block. `nvidia-smi -q -x` produces a machine-readable XML
dump of every GPU on the node — driver, VBIOS, memory config,
inforom checksums, PCIe topology. `dcgmi diag -r 3 --json` runs a
suite of hardware diagnostics and dumps their pass/fail outcomes.
`NCCL_DEBUG=INFO`'s opening banner records the topology NCCL saw
at bring-up. `NCCL_TOPO_DUMP_FILE=/tmp/topo.xml` gives you a
canonical topology description NCCL used for algorithm selection.

Every file lands under `hardware/`. The exercise for this chapter
walks through wiring these into your run's `postbatch` / `epilog`
hook.

The reason this block matters: when a run cannot be reproduced,
the first thing an audit looks at is whether the hardware was the
same. If the answer is "we do not have a record of what hardware
we used", the audit stops there.

## Failure modes the bundle catches

The bundle prevents specific classes of incident. Four worth
naming:

- **"Which version of the tokenizer?"** — the config says
  `tokenizer=Llama3`, but Llama3 tokenizers ship with vocabulary
  patches over time. The tokenizer sha256 tells you which one.
- **"We upgraded PyTorch and now runs diverge from a month ago."**
  — the framework block is the diff.
- **"The container digest changed under us."** — the digest
  block plus the launch assertion detect this before step 0.
- **"Did we train on the deduped corpus or the pre-dedup one?"** —
  mod-103's manifest, included by hash, answers this in one line
  of diff.

Each incident type resolved from the bundle is a several-day
archaeological dig avoided. That is the ROI on this whole chapter.

## Bring-up checklist

The bundle-emission code paths every training-cluster owns:

1. `set_all_seeds()` called before `torch.distributed.init_process_group`
   with a seed pulled from the config; the whole seed set recorded to
   `seeds.json`.
2. Corpus / tokenizer hash asserts run before step 0; a mismatch
   fails the launch.
3. Container digest resolved at launch (`docker inspect` on the
   running container) and recorded to `container/image.json`.
4. `nvidia-smi -q -x`, `dcgmi diag -r 3 --json`, and
   `NCCL_TOPO_DUMP_FILE` written to `hardware/` in the launch
   prolog.
5. On run finish, `run_manifest.json` assembled and uploaded to
   the object-store prefix, and the tracker's config is updated
   with its URL.

Once those five are wired, every run — pretraining, fine-tune,
smoke test — produces a valid bundle without any per-run effort.

## Summary

- "Reproduce" has three meanings: bitwise (rarely possible on
  GPUs), statistical (what the bundle enables when hardware
  matches), and provenance (what the bundle proves regardless).
- The bundle is a small set of files: `run_manifest.json`, seeds,
  config + hash, data manifest + tokenizer hash, code lockfile +
  git state, container digest, hardware capture, final metrics.
- Seeds are a *set* — Python, NumPy, torch, torch.cuda,
  PYTHONHASHSEED, dataloader worker seed. One seed field in the
  config is not enough.
- Dataset and tokenizer identity go through sha256; mod-103 owns
  the shard-manifest format and the launch-time assertion is
  applied here.
- Framework versions include torch, CUDA, cuDNN, NCCL, plus the
  full `pip freeze`. Container digest (OCI sha256) is the
  authoritative pin.
- Hardware manifest is `nvidia-smi -q -x`, `dcgmi diag`, NCCL
  banner, and `NCCL_TOPO_DUMP_FILE`. Without this, no audit
  completes.
- The bundle is small (a few MB) and lives alongside the
  checkpoint in the object store. Chapter 6 formalizes the
  hand-off contract that consumes it.
