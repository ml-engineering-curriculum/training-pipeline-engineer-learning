# Resources for mod-103-training-scale-data-pipelines

Primary sources first, tooling and docs second. The chapters and
exercises are grounded in these references; skim the framework docs
as-needed while working the exercises, read the papers cover-to-cover
at least once before the runbook exercise.

## Primary papers — corpus, dedup, and pretraining data

- **Lee, K., et al. (2022). "Deduplicating Training Data Makes Language
  Models Better."** *ACL.* The reference for why exact + near-dup
  matters for LLM pretraining, with MinHash-LSH parameter choices.
- **Rae, J. W., et al. (2021). "Scaling Language Models: Methods,
  Analysis & Insights from Training Gopher."** arXiv:2112.11446.
  The Gopher paper's dataset appendix walks the MassiveText
  cleaning + dedup pipeline in detail.
- **Wenzek, G., et al. (2019). "CCNet: Extracting High Quality
  Monolingual Datasets from Web Crawl Data."** *LREC 2020.* The
  canonical pipeline for cleaning and dedup on Common Crawl.
- **Penedo, G., et al. (2023). "The RefinedWeb Dataset for Falcon
  LLM: Outperforming Curated Corpora with Web Data, and Web Data
  Only."** arXiv:2306.01116. The RefinedWeb pipeline and Falcon
  team's dedup choices.
- **Touvron, H., et al. (2023). "Llama 2: Open Foundation and
  Fine-Tuned Chat Models."** arXiv:2307.09288. Section on
  pretraining data preparation and dedup.
- **Grattafiori, A., et al. (2024). "The Llama 3 Herd of Models."**
  Meta AI technical paper. The 15 T-token corpus assembly and its
  dedup / decontamination pipeline.
  <!-- needs-research: cite the current Llama 3 paper URL and dedup section when authoring. -->
- **Kaplan, J., et al. (2020). "Scaling Laws for Neural Language
  Models."** arXiv:2001.08361. Sets the compute-vs.-data budget
  the whole staging discussion hangs on.
- **Hoffmann, J., et al. (2022). "Training Compute-Optimal Large
  Language Models."** arXiv:2203.15556. The Chinchilla paper —
  refines the token / parameter trade the corpus size targets.

## Data-loader white papers and design notes

- **Aizman, A., Maltby, G., & Breuel, T. (2019). "High Performance
  I/O For Large Scale Deep Learning."** IEEE Big Data 2019. The
  WebDataset design paper.
- **The MosaicML Streaming design blog series.** The mosaicml
  engineering blog covers `num_canonical_nodes`, shuffle
  algorithms (`py1s`, `py1b`, `py1br`, `py1e`, `py2s`), and the
  resume-state design that MDS depends on.
  <!-- needs-research: cite the specific mosaicml/streaming blog posts and versioned docs pages when authoring. -->

## Library documentation (primary reference for the code paths)

- **WebDataset (`github.com/webdataset/webdataset`).** README, the
  `WebDataset` and `WebLoader` API references, and the
  `split_by_node` / `split_by_worker` semantics.
- **MosaicML Streaming (`github.com/mosaicml/streaming`).** The
  `MDSWriter` and `StreamingDataset` API reference; the format
  specification; the `streaming.util.merge_index` function; the
  `state_dict` / `load_state_dict` contract on `StreamingDataset`.
- **Ray Data (`docs.ray.io/en/latest/data/data.html`).** Dataset
  API, `read_parquet`, `map_batches`, actor pool configuration
  (`concurrency`, `num_cpus`), `groupby`, `write_parquet`, and
  streaming execution semantics.
- **HuggingFace `tokenizers`.** The Rust-backed tokenizer
  library. Serialization format (`tokenizer.json`), normalizers,
  and the vocab / special-token API used in chapter 4.
- **PyArrow.** `pyarrow.parquet` for row-group-level reads and
  `pyarrow.fs.S3FileSystem` for object-store I/O. Multi-part
  read config lives here.
- **PyTorch `torch.utils.data.DataLoader`.** For
  `num_workers`, `prefetch_factor`, `persistent_workers`,
  and the interaction with `IterableDataset` (which is what
  both WebDataset and StreamingDataset present).
- **`fsspec` and `smart_open`.** The generic filesystem
  abstraction under `webdataset`, `pyarrow`, and much of the
  Ray Data stack.

## Object-store and staging references

- **AWS: "Best practices design patterns: optimizing Amazon S3
  performance."** The S3 performance guide — request-rate targets
  per prefix, multi-part transfer patterns, and event-driven
  configuration.
  <!-- needs-research: cite the current AWS documentation URL and rate numbers when authoring. -->
- **GCP: "Cloud Storage best practices."** The GCS analog of the
  above.
  <!-- needs-research: cite the current Cloud Storage docs URL when authoring. -->
- **AWS Mountpoint for Amazon S3.** A FUSE-based S3 client suitable
  as a per-node staging layer.
- **Alluxio.** Distributed caching layer with a POSIX-style
  interface; a productionized version of the coordinated
  prefetch daemon pattern in chapter 5.
- **DDN Lustre and WEKA (product documentation).** For the
  parallel-filesystem side of the staging story. The tuning
  guides are the ones you consult when tier-1 first-fetch
  latency starts climbing.

## Reference codebases

- **`mosaicml/composer`.** The MosaicML training framework;
  reference for MDS integration with a real training loop.
- **`mosaicml/llm-foundry`.** LLM pretraining recipes built on
  Composer + Streaming.
- **`pytorch/torchtitan`.** PyTorch's reference large-scale
  training implementation; uses WebDataset-style shard readers
  and demonstrates the loader / trainer boundary at scale.
- **`ray-project/ray/python/ray/data`.** Ray Data source; the
  best reference for actor-pool and `map_batches` semantics.
- **`nvidia/DALI`.** NVIDIA's alternative loader stack.
  Out of scope for this module but worth a skim to see the
  same problems solved with GPU-side decoding.
- **`grain-project/grain`.** Google's alternative Python-first
  training loader. Same problems, different mental model; useful
  contrast.

## Recommended reading order for a first pass

1. Chapter 1 of this module + the WebDataset README (30 min).
2. Chapter 2 + Aizman et al., 2019 (WebDataset paper) (1 h).
3. Chapter 3 + the MosaicML Streaming docs + at least one of the
   Streaming design blog posts (1.5 h).
4. Chapter 4 + Lee et al., 2022 (dedup paper) + the Ray Data
   user guide (1.5 h).
5. Chapter 5 + the S3 performance guide + one Lustre / WEKA
   tuning guide (1 h).
6. Chapter 6 (30 min).
7. Chapter 7 + a corpus-appendix section of the Llama 3 or
   RefinedWeb paper for what a real runbook artifact set looks
   like (1 h).

Exercises assume you have done at least items 1–4 before
starting; exercise-05 additionally assumes chapter 6.
