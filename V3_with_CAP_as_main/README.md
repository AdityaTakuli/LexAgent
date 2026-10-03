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
# 1. Build the primary CAP database (Landmark 27 vols, Modern 285 vols, or All 600+ vols)
python master_run.py --build-db --volumes landmark --workers 16 --batch-size 256

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
│   │   ├── cap_ingest.py               # CAP downloader, parser, and corpus builder
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
│   ├── build_db.py                     # Standalone DB builder script
│   ├── phase1_run.py                   # Phase 1 retrieval test
│   ├── phase2_run.py                   # Phase 2 fixed debate test
│   └── phase3_run.py                   # Phase 3 full DVA+ debate test
│
└── experiments/
    ├── benchmark_cases.py              # Curated 30-case benchmark suite
    ├── run_25_cases_benchmark.py       # Benchmark runner with scorecards
    ├── run_ablation.py                 # 4-variant ablation study runner
    └── bootstrap.py                    # Validated component loader
```

---

## 🏛️ Storage Paths & Environment Configuration

Set the environment variable `LEXAGENT_DATA_DIR` to customize the storage target (defaults to `/content/drive/MyDrive/LexAgentV3` on Colab):

```bash
export LEXAGENT_DATA_DIR="/path/to/data"
export LEXAGENT_CAP_VOLUME="410"        # Default U.S. Reports volume
```

Persistent files created during `--build-db`:
* `cap_chroma_corpus/`: ChromaDB persistent dense vector store.
* `cap_bm25_index.pkl`: Serialized BM25 index over CAP chunks.
* `cap_known_cases.pkl`: Set of canonical CAP case names & citations for CCE.
* `cap_corpus.json`: Canonical dump of chunked documents.
* `cap_corpus_manifest.json`: Build manifest with hashes.
* `chroma_memory/`: Adaptive Memory collections (`verdict_cache`, `citation_risk_index`).
* `results/`: Benchmark and ablation output scorecards.
