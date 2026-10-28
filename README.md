# LexAgent

**Adversarial multi-agent debate with opinion-grounded citation verification for legal question answering.**

Language models invent legal authorities, and they credit real cases with holdings those cases never
contained. LexAgent tackles both problems. A prosecutor agent and a defense agent debate a question
over retrieved U.S. Supreme Court opinions. Inside the debate loop, a citation checker tests every
cited case against the case catalog and the opinion's own text, a controller decides whether another
round is needed, and a judge writes the final answer from what was verified.

- **Corpus:** 593 volumes of the United States Reports, 528,283 passages. Harvard's Caselaw Access
  Project supplies volumes 1–572, and the Supreme Court's official PDFs supply volumes 573–593.
- **Paper:** [`V3_with_CAP_as_main/paper/lexagent_ieee.pdf`](V3_with_CAP_as_main/paper/lexagent_ieee.pdf) (IEEE format)
- **Web UI:** chat with the debate as it runs, and browse the evaluation, the benchmark cases and the corpus

> A research prototype. Its answers are not legal advice.

---

## How it works

```text
question
   │
   ▼
Hybrid retrieval ── dense (BGE / Legal-BERT, ChromaDB) + BM25, fused by reciprocal rank
   │
   ▼
┌───────────────────────── debate round ─────────────────────────┐
│  Prosecutor ──► Defense ──► Citation checker ──► Reflection    │
│  (answers       (attacks     (does the case exist?   (finds    │
│   with cites)    with its     does its opinion       gaps)     │
│                  own          support the claim?)              │
│                  evidence)                                     │
└───────────────────────────────┬────────────────────────────────┘
                                ▼
               Debate controller: another round, or stop?
                                │
                                ▼
               Judge: final answer, resting only on supported citations
```

| Component | Code |
|---|---|
| Agents (prosecutor, defense, reflection, judge) | `src/agents/` |
| Citation Confidence Engine (existence + holding support) | `src/novel/cce.py` |
| Debate ledger (keeps what each round verified) | `src/novel/ledger.py` |
| Dynamic Debate Controller | `src/novel/ddc.py` |
| Adaptive memory (verdict cache, citation risk list) | `src/novel/adaptive_memory.py` |
| LangGraph pipeline | `src/graph/lexagent_graph.py` |
| Retrieval (v1 Legal-BERT, v2 BGE + word-token BM25) | `src/retrieval/` |
| Staged, crash-resumable corpus pipeline | `src/data/` |

The language model is either an OpenAI-compatible API (the default when a key is set) or
Mistral-7B-Instruct running on your own GPU.

---

## Results

On 96 questions about landmark Supreme Court decisions:

| System | Answers resting on the target decision | Seconds per question |
|---|---:|---:|
| LexAgent, first version | 34.4% | 116 |
| **LexAgent, final** (after the audit repairs) | **62.5%** | **43** |
| Single-agent RAG baseline (same retriever) | 81.6% | 3 |

**This is a negative result, and we report it as one.** A forensic audit of the first evaluation
found several problems. A verdict cache had answered 34 of the 96 questions with the stored answer to
a different question. Two measures did not measure what their names said. The citation checker
accepted only 22.9% of correct citations. Six repairs fixed these, and they nearly doubled how often
the debate's answer rests on the right decision. No final debate answer cites a non-existent case.
Even so, a single agent on the same retrieval still does better on this benchmark, with a 7B
generator. The full analysis is in the paper and in
[`experiments/v2/README.md`](V3_with_CAP_as_main/experiments/v2/README.md).

---

## Quickstart

Everything lives in [`V3_with_CAP_as_main/`](V3_with_CAP_as_main/).

```bash
cd V3_with_CAP_as_main
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # full research stack (GPU)
# or, for the web server with an API model on CPU only:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install -r requirements-server.txt

cp .env.example .env                       # then set LEXAGENT_LLM_API_KEY, or LEXAGENT_LLM_BACKEND=local
```

**1. Build the corpus** (three resumable stages; `landmark` is the 27 volumes behind the core
benchmark cases, `all` is the full corpus):

```bash
python master_run.py --download-cap --volumes landmark
python master_run.py --chunk-cap    --volumes landmark
python master_run.py --build-db     --volumes landmark
```

**2. Run it**

```bash
python master_run.py                        # interactive menu
python master_run.py --test-phases          # smoke tests: retrieval, fixed debate, full debate
python master_run.py --benchmark            # benchmark
python master_run.py --baseline             # single-agent RAG baseline
python master_run.py --ablation             # w/o defense, checker, memory, controller
python master_run.py --stress-test          # citation stress test (no LLM)
```

**3. Web UI**

```bash
pip install -r ui/requirements.txt
cd ui/web && npm install && npm run build && cd ../..
python ui/server.py --port 8000             # http://127.0.0.1:8000
```

`http://127.0.0.1:8000/#/chat?replay=1` replays a recorded debate without loading any model.

**Tests** (no network, GPU or model needed):

```bash
python -m unittest discover -s tests -v
```

---

## Repository layout

```text
.
├── V3_with_CAP_as_main/      # the current system (start here)
│   ├── master_run.py         # single entry point: corpus build, tests, benchmark, ablation
│   ├── config.py             # all settings; reads .env
│   ├── src/                  # agents, citation checker, controller, memory, retrieval, data pipeline
│   ├── runs/                 # corpus indexing and phase smoke tests
│   ├── experiments/          # benchmark, baseline, ablation, stress test, post-audit experiments (v2/)
│   ├── tests/                # unit tests
│   ├── ui/                   # FastAPI server + React/TypeScript front end
│   ├── deploy/               # Hugging Face Space and Cloudflare tunnel scripts
│   ├── paper/                # IEEE paper; every number is generated from the saved results
│   ├── COMPARISON_MATRIX.md  # LexAgent against commercial tools and research systems
│   └── Review.md             # the external review that started the audit
└── legacy/v2_phase1/         # the earlier v2 Phase-1 prototype, kept for reference
```

## Documentation

| Document | Covers |
|---|---|
| [`V3_with_CAP_as_main/README.md`](V3_with_CAP_as_main/README.md) | Full guide: corpus pipeline, citation checking, GPU settings, storage paths |
| [`experiments/v2/README.md`](V3_with_CAP_as_main/experiments/v2/README.md) | The audit repairs and the post-audit experiments |
| [`ui/README.md`](V3_with_CAP_as_main/ui/README.md) | Web UI pages, API and development |
| [`paper/README.md`](V3_with_CAP_as_main/paper/README.md) | Building the paper from the results |
| [`COMPARISON_MATRIX.md`](V3_with_CAP_as_main/COMPARISON_MATRIX.md) | Capability comparison with related systems |
