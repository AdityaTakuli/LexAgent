# LexAgent v3.0 — Harvard Caselaw Access Project (CAP) Edition

`V3_with_CAP_as_main` is a standalone, clean edition of the LexAgent Multi-Agent Debate Framework where **Harvard's Caselaw Access Project (CAP)** is the **Primary Authority Database** for dense retrieval, BM25 lexical search, and citation confidence verification.

---

## 🌟 Key Features of V3

1. **Harvard CAP as the Primary Authority Core**:
   - Primary vector store (`cap_chroma_corpus`) and BM25 index (`cap_bm25_index.pkl`) built directly from CAP casebodies (U.S. Reports Volume 410).
   - Preserves official court reporters, volume numbers (e.g. `410 U.S. 113`), first/last pages, and opinion text.
   - CCE (Citation Confidence Engine) verifies both official reporter citations and case titles against the CAP verified catalog.
2. **One Master Runner to Rule Them All**:
   - `master_run.py`: Single entry point for database building, phase testing, benchmarking (25-30 cases), and ablation studies.
   - Interactive terminal menu when run without flags.
3. **Colab Master Notebook**:
   - `LexAgent_V3_Master_Execution.ipynb`: Clean one-click execution cells for Google Colab (Tesla T4 GPU).

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
cd V3_with_CAP_as_main
pip install -r requirements.txt
```

### 2. Using the Master Runner (`master_run.py`)

#### Option A: Interactive Menu (Recommended)
Run without any arguments to open the interactive control menu:
```bash
python master_run.py
```

#### Option B: CLI Flags (Full Unlimited Run)
```bash
# 1. Build the primary CAP database in three separate stages (see "Staged CAP Data Pipeline" below)
python master_run.py --download-cap --volumes landmark --workers 16
python master_run.py --chunk-cap    --volumes landmark
python master_run.py --build-db     --volumes landmark --batch-size 256

# 2. Run Phase 1, Phase 2, and Phase 3 verification smoke tests
python master_run.py --test-phases

# 3. Run the zero-LLM Citation Stress-Test Protocol (Novelty 5 Diagnostic Gate)
python master_run.py --stress-test

# 4. Run the full benchmark across all 30 constitutional cases
python master_run.py --benchmark --benchmark-limit 30

# 5. Run the standard single-agent RAG baseline across all 30 cases
python master_run.py --baseline --benchmark-limit 30

# 6. Run all 5 ablation studies (Full vs w/o Defense vs w/o CCE vs w/o Memory vs w/o DDC) across 30 cases
python master_run.py --ablation --ablation-limit 30

# 7. Run the complete pipeline end-to-end (Build -> Tests -> Stress Test -> Benchmark -> Ablation)
python master_run.py --all --benchmark-limit 30 --ablation-limit 30
```

---

## 🧱 Staged CAP Data Pipeline

The database is built in three separate stages. Each one is resumable, and none of them does another stage's work.

| Stage | Command | Does | Never does |
|---|---|---|---|
| 1. Download | `python master_run.py --download-cap --volumes all --workers 16 --checkpoint-interval 64` | CAP network → validated raw JSON in `cap_raw_volumes/` | chunking, embedding, ChromaDB |
| 2. Chunk | `python master_run.py --chunk-cap --volumes all --checkpoint-interval 64` | verified raw JSON → `cap_chunks_cache/` | network access, embedding |
| 3. Index | `python master_run.py --build-db --volumes all --batch-size 256` | verified chunks → BM25, known cases, ChromaDB embeddings (GPU) | network access, chunking |

`--volumes all` means volumes 1 to 600 inclusive. CAP publishes U.S. Reports volumes 1 to 572 only, so later volumes are recorded as `UNAVAILABLE` by the download stage and are never counted as complete.

**Volumes CAP does not have (573 and later).** An optional stage 1b fills them from the Supreme Court's own U.S. Reports PDFs (<https://www.supremecourt.gov/opinions/USReports.aspx>), writing the same raw cache format so stages 2 and 3 need no changes:

```bash
python master_run.py --download-us-reports-pdf --volumes 573-600
```

| Volumes | Source | Notes |
|---|---|---|
| 1–572 | CAP (`static.case.law`) | All cases, including orders. |
| 573–587 | Official bound-volume PDFs | Reported opinions and opinions in chambers; the orders section is not imported. Volume 587 is a page proof. |
| 588–593 | Official preliminary prints | Same coverage. Volume 593 is incomplete: only part 1 has been printed. |
| 594–600 | none | The Court has published only the orders of these volumes in U.S. Reports form. |

- It never replaces a volume that is already verified complete, so CAP volumes are untouched.
- Each imported case records where it came from: `authority_identity` is `US_REPORTS_OFFICIAL_PDF` instead of `CAP_CANONICAL`, and the raw file lists the PDF, its SHA-256 and the page range. The chunk `source` field stays `CAP`, so the agents see these passages labelled like every other one.
- The PDFs drop the letters of "fi", "fl" and "ffi" ligatures ("offcer", "traffc"). The importer restores them exactly from the embedded fonts, moves footnotes to the end of each case and rejoins hyphenated words.
- A PDF that does not split cleanly into cases is recorded as `FAILED`, not imported partially.
- Checked against CAP on volume 572, which both publish: every reported opinion was found at the same page with the same decision date, and a median 99% of CAP's words appear in the imported text. The imported text is longer because it includes the Court's syllabus.
- Needs `pymupdf` (in `requirements.txt`).

**Resume is the default.** Run the same command again after a crash, a power loss or a network failure:

- A volume is skipped only when its cache file carries an explicit `"status": "complete"` and passes validation (readable JSON, matching volume, every case in the CAP metadata accounted for, no failed case downloads). A file that merely exists is not trusted.
- A volume that fails is recorded as `FAILED` with the reason, and the run moves on to the next volume. The next run retries only the volumes that are not complete, and re-fetches only the cases that are missing.
- The chunk stage never downloads. If a raw volume is missing or unverified it reports `Volume X is not available/verified. Run the download stage`.
- The index stage never downloads or chunks. It stops if a target volume has no verified chunk cache, unless `--allow-incomplete` is passed.

**Checkpoints.** `cap_download_checkpoint.json` and `cap_chunk_checkpoint.json` hold the state of every volume (`NOT_STARTED`, `IN_PROGRESS`, `COMPLETE`, `FAILED`, `UNAVAILABLE`), the completed and failed lists, a `not_completed` record with the reason for each volume, and one batch entry per 64 volumes (1–64, 65–128, …, 577–600). They are written atomically. The embedding stage keeps its own chunk-level checkpoint, `cap_indexing_checkpoint.json`.

**Nothing is deleted unless you ask.**

| Flag | Effect |
|---|---|
| `--overwrite` | Stage 3 only: wipes the ChromaDB collection and rebuilds BM25, known cases and the corpus JSON. Never touches raw or chunk caches. |
| `--redownload` | Stage 1: re-downloads the selected volumes even if verified. |
| `--rechunk` | Stage 2: re-chunks the selected volumes even if verified. |

A corrupt or unopenable ChromaDB stops the build with an error instead of being purged automatically.

### How a citation is verified (CCE)

A full corpus holds about 180,000 reports, and four in five are one-line order entries (certiorari denied, rehearing denied) that carry the same case name as an opinion. Looking a case up by name alone therefore often lands on an order entry. The build writes `cap_case_catalog.json`, a list of every report with its cite, year and length, and the CCE uses it to work in this order:

1. **The citation.** Does the named case exist, and which report is it? The reporter cite (volume and first page) settles it when given; otherwise the substantial report with that name is taken, never an order entry. Different spellings of a name are recognised ("INS v. Chadha", "Planned Parenthood of Southeastern Pa. v. Casey"). A name that merely shares words with a real case does not count as existing, and a reporter cite that belongs to a different case is recorded as a mismatch.
2. **The holding.** Only for a citation that exists: the passages of that opinion most relevant to the claimed holding are checked for entailment. `LEXAGENT_CCE_SCAN_LIMIT` (default 8,000 characters, three passages) sets how much of the opinion is checked.

A case whose opinion text is not in the corpus is `UNVERIFIABLE`, not `MISATTRIBUTED`.

### Test set

`get_benchmark_cases()` returns only cases whose target opinion is in the corpus (at least 500 words of it). Cases whose source text lies wholly outside the corpus are left out and listed at the start of each run, because no answer to them could be grounded. On the 593-volume corpus that removes 4 of the 100: Dobbs, Bruen and Students for Fair Admissions (volumes 594–600) and Buckley v. Valeo (CAP's volume 424 begins at page 295). The stress test's "real off-corpus" class is likewise built from cases that are actually absent from the corpus in use.

**Tests** (fake CAP server, no network or GPU needed):
```bash
python -m unittest discover -s tests -v
```

---

## ⚡ Running on an Offline Server with High-End Nvidia GPU

This branch (`offline-run`) is pre-configured for high-end Nvidia GPUs (**A100**, **H100**, **RTX 4090**, **RTX 3090**, **A6000**):

```bash
# 1. Set environment variables for offline / high-VRAM execution
export CUDA_VISIBLE_DEVICES=0
export LEXAGENT_PRECISION=bfloat16           # Native unquantized 16-bit for top legal quality
export LEXAGENT_BATCH_SIZE=256              # Dense indexing batch size
export LEXAGENT_WORKERS=16                  # Parallel thread workers
export LEXAGENT_CCE_SCAN_LIMIT=8000         # Scan up to 8,000 characters of opinion text in CCE
# export TRANSFORMERS_OFFLINE=1             # Set if server has no external internet access
# export HF_HUB_OFFLINE=1                   # Set if server has no external internet access

# 2. Run the complete suite without limits
python master_run.py --all --benchmark-limit 30 --ablation-limit 30
```

---

## 📂 Directory Structure

```text
V3_with_CAP_as_main/
├── config.py                           # Central configuration (all paths point to CAP)
├── requirements.txt                    # Project dependencies
├── README.md                           # This guide
├── master_run.py                       # Single master execution runner (CLI + Interactive)
├── LexAgent_V3_Master_Execution.ipynb  # One-click Colab Master Notebook
│
├── src/
│   ├── agents/                         # Prosecutor, Defense, Reflection, Judge
│   │   ├── prosecutor_agent.py
│   │   ├── defense_agent.py
│   │   ├── reflection_agent.py
│   │   ├── judge_agent.py
│   │   ├── prompts.py
│   │   └── state.py
│   │
│   ├── data/                           # Ingestion & Chunking
│   │   ├── cap_ingest.py               # CAP download stage, chunk stage, cache validation
│   │   ├── cap_checkpoint.py           # Per-volume state ledger, batch checkpoints, atomic writes
│   │   ├── us_reports_pdf.py           # Volumes CAP does not publish, from the official Supreme Court PDFs
│   │   ├── case_catalog.py             # Every report in the corpus; resolves a citation to its opinion
│   │   ├── build_corpus.py             # Corpus loaders
│   │   ├── chunking.py                 # Legal chunking with overlap
│   │   ├── corpus_schema.py            # CorpusDocument & CitationRecord schemas
│   │   └── download_datasets.py        # CaseHOLD & LegalBench downloader for evaluation
│   │
│   ├── retrieval/                      # Hybrid retrieval over CAP
│   │   ├── dense_retriever.py          # Legal-BERT ChromaDB retriever
│   │   ├── bm25_retriever.py           # BM25 sparse retriever
│   │   └── hybrid_retriever.py         # Reciprocal Rank Fusion (RRF k=60)
│   │
│   ├── novel/                          # Core novel components
│   │   ├── cce.py                      # CCE with CAP reporter & title verification
│   │   ├── ddc.py                      # Dynamic Debate Controller
│   │   └── adaptive_memory.py          # Semantic verdict cache & risk tracker
│   │
│   ├── models/                         # Quantized Mistral-7B + Legal-BERT loader
│   │   └── load_model.py
│   │
│   ├── graph/                          # LangGraph state machine
│   │   └── lexagent_graph.py
│   │
│   ├── evaluation/                     # Metric calculators
│   │   ├── evaluator.py
│   │   ├── citation_accuracy.py        # CAS
│   │   ├── hallucination_rate.py       # HR
│   │   └── debate_convergence.py       # DCR
│   │
│   └── utils/
│       └── logger.py
│
├── runs/
│   ├── build_db.py                     # Stage 3: index verified chunk caches (BM25 + ChromaDB)
│   ├── phase1_run.py                   # Phase 1 retrieval test
│   ├── phase2_run.py                   # Phase 2 fixed debate test
│   └── phase3_run.py                   # Phase 3 full DVA+ debate test
│
├── experiments/
│   ├── benchmark_cases.py              # Benchmark questions; only those the corpus can support are used
│   ├── run_25_cases_benchmark.py       # Benchmark runner with scorecards, and the single-agent baseline
│   ├── run_ablation.py                 # 4-variant ablation study runner (--variants A2 re-runs one)
│   ├── run_stress_test.py              # Citation stress test (no LLM)
│   ├── run_cce_evidence_study.py       # Which passages should the citation checker read? (no LLM)
│   └── bootstrap.py                    # Validated component loader
│
├── tests/                              # Unit tests: no network, GPU or model needed
├── ui/                                 # Web UI: FastAPI server + React app (see ui/README.md)
└── paper/                              # IEEE-format paper, figures, and the script that fills in its numbers
```

## 🔎 Forensic audit and post-audit experiments

The first full evaluation scored the debate below a single-agent baseline. A forensic audit of the
code, the saved results and the run logs (`data/results/forensic_audit/`, nine reports) traced that
result to a verdict cache that answered different questions, to two measures that did not measure
what their names said, and to a citation checker that could not tell a correct holding from a wrong one.

Six repairs followed in two steps, and a new set of experiments on the repaired code
([`experiments/v2/README.md`](experiments/v2/README.md)):

| Repair | In short |
|---|---|
| Reproducible, inspectable runs | per-case seeds and a full saved trace of every case |
| Verdict cache | answers only the identical question; off in evaluation |
| Measures | existence, cite correctness, target cited and citation counts reported separately |
| Citation checker | existence and holding support reported separately; the support verdict acts only after a support model has been validated on held-out citations |

| Debate protocol (second step) | a ledger keeps what every round verified, the judge reads it, the controller stops when a citation is supported (`config.DEBATE_PROTOCOL = "ledger"`; the default stays `"v1"`) |
| Retrieval (second step) | a second index with an embedder trained for retrieval and word-token BM25 (`runs/build_retrieval_v2.py`) |

The earlier result files are kept unchanged as the record of the pre-audit runs
(`data/results/EXPERIMENT_MANIFEST.json`); post-audit runs live in `data/results/v2/`.

**Result (8 October 2026).** The repairs make the debate a much better debate: 62.5% of its answers
rest on the target decision instead of 34.4%, in 43 s instead of 116 s, and no answer cites a
non-existent case. It is still not better than a single agent: on the same retrieval the
single-agent baseline reaches 81.6% and is ahead on citation recall, citation precision, grounded
answers, ROUGE-L and BERTScore, in 3 s. The corrected evaluation itself contained one more error,
found at the end and fixed from the saved raw outputs: the baseline's parser dropped citations the
model wrote as plain strings. Details and tables: [`experiments/v2/README.md`](experiments/v2/README.md),
`data/results/v2/REPORT.md`, `data/results/v2/STANDARD_METRICS.md`, and the paper.

```powershell
.venv\Scripts\python.exe experiments\v2\runner.py --queue    # first post-audit queue (resumable)
.venv\Scripts\python.exe experiments\v2\runner.py --final    # final system, its baselines, standard measures
.venv\Scripts\python.exe experiments\v2\score.py             # tables for whatever has finished
.venv\Scripts\python.exe experiments\v2\standard_metrics.py  # retrieval, citation and overlap measures
```

## 🖥️ Web UI

A chat-style interface that shows the debate stage by stage, with the evaluation results and the
corpus as charts. Build it once and start the server (details in [`ui/README.md`](ui/README.md)):

```powershell
cd ui\web; npm install; npm run build; cd ..\..
.venv\Scripts\python.exe ui\server.py --port 8000     # http://127.0.0.1:8000
```

## 📄 Paper

[`paper/lexagent_ieee.pdf`](paper/lexagent_ieee.pdf) is an IEEE-format write-up of the system and its
evaluation. Its numbers are not typed by hand: `paper/make_results.py` computes them from the saved
result files, so the paper follows the experiments. Rebuild with `paper\build.ps1`
(see [`paper/README.md`](paper/README.md)).

---

## 🏛️ Storage Paths & Environment Configuration

Set the environment variable `LEXAGENT_DATA_DIR` to customize the storage target (defaults to `/content/drive/MyDrive/LexAgentV3` on Colab):

```bash
export LEXAGENT_DATA_DIR="/path/to/data"
export LEXAGENT_CAP_VOLUME="410"        # Default U.S. Reports volume
```

Persistent files created by the download and chunk stages:
* `cap_raw_volumes/`: One validated raw JSON per CAP volume (`cap_raw_us_<N>.json`).
* `cap_chunks_cache/`: One chunk cache per volume (`cap_chunks_us_<N>.json`).
* `cap_download_checkpoint.json`, `cap_chunk_checkpoint.json`: Per-volume state and 64-volume batch checkpoints.

Persistent files created during `--build-db`:
* `cap_build_state.json`: Which corpus the BM25 index, known cases and corpus JSON were built from.
* `cap_indexing_checkpoint.json`: Chunk-level progress of the embedding pass.
* `cap_chroma_corpus/`: ChromaDB persistent dense vector store.
* `cap_bm25_index.pkl`: Serialized BM25 index over CAP chunks.
* `cap_known_cases.pkl`: Set of canonical CAP case names & citations for CCE.
* `cap_corpus.json`: Canonical dump of chunked documents.
* `cap_corpus_manifest.json`: Build manifest with hashes.
* `chroma_memory/`: Adaptive Memory collections (`verdict_cache`, `citation_risk_index`).
* `results/`: Benchmark and ablation output scorecards.
