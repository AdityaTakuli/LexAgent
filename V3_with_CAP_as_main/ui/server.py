# LexAgent v3.0 | ui/server.py
"""Web UI server for LexAgent.

Serves three things:
  - the built React app (ui/web/dist),
  - the saved evaluation results and corpus statistics as JSON,
  - a live debate: POST /api/chat streams the pipeline node by node (Server-Sent Events).

Run from the V3_with_CAP_as_main directory:

    python ui/server.py                 # http://127.0.0.1:8000
    python ui/server.py --port 8080 --preload

The model stack (Mistral-7B, Legal-BERT, the NLI model, BM25 and the corpus) is loaded on the
first question, or at start-up with --preload. It needs the GPU to itself: do not start a
chat while a benchmark or ablation run is using it.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from config import (  # noqa: E402
    ACTIVE_LLM_NAME,
    LLM_API_MODELS,
    LLM_BACKEND,
    CAP_CHUNK_CHECKPOINT_PATH,
    CAP_DOWNLOAD_CHECKPOINT_PATH,
    CASE_CATALOG_PATH,
    CORPUS_MANIFEST_PATH,
    DDC_MAX_ROUNDS,
    EMBEDDER_NAME,
    INDEXING_CHECKPOINT_PATH,
    NLI_MODEL_NAME,
    RESULTS_DIR,
    TAU_ACCEPTANCE_THRESHOLD,
)

WEB_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "dist")
SAMPLE_TRANSCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_transcript.json")
TRANSCRIPT_DIR = os.path.join(RESULTS_DIR, "ui_transcripts")

# The order in which the debate graph visits its nodes, used to announce the stage that starts next.
NEXT_STAGE = {"memory_check": "retrieve", "retrieve": "prosecutor", "inject_llm": "prosecutor", "prosecutor": "defense",
              "defense": "cce", "cce": "reflection", "reflection": "ddc", "judge": "memory_store"}


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# ══════════════════════════════════════════════════════
# The engine: the model stack and the compiled debate graph
# ══════════════════════════════════════════════════════
class Engine:
    """Loads the pipeline once and runs one debate at a time."""

    def __init__(self) -> None:
        self.state = "cold"  # cold | loading | ready | error
        self.message = "The model is not loaded yet."
        self.loaded_seconds: Optional[float] = None
        self.graph = None
        self.graphs: Dict[str, Any] = {}  # model name -> debate graph (API backend: one per model)
        self._components: Optional[dict] = None
        self._memory = None
        self._load_lock = threading.Lock()
        self.run_lock = threading.Lock()

    def status(self) -> dict:
        return {"state": self.state, "message": self.message, "loaded_seconds": self.loaded_seconds, "busy": self.run_lock.locked()}

    def load(self) -> None:
        """Load the pipeline (blocking). Safe to call from several threads; only one load happens."""
        with self._load_lock:
            if self.state == "ready":
                return
            self.state, self.message = "loading", "Loading the corpus, the retrievers and the models..."
            started = time.monotonic()
            try:
                from experiments.bootstrap import load_lexagent_components
                from src.graph.lexagent_graph import build_lexagent_graph
                from src.novel.adaptive_memory import AdaptiveMemory

                components = load_lexagent_components(load_llm=True)
                # A namespace of its own, so chat answers never mix with benchmark runs.
                memory = AdaptiveMemory(namespace="ui_chat")
                self.graph = build_lexagent_graph(
                    components["llm"], components["hybrid"], components["cce"], components["ddc"], memory, components["embedder"]
                )
                self._components, self._memory = components, memory
                self.graphs = {ACTIVE_LLM_NAME: self.graph}
                self.loaded_seconds = round(time.monotonic() - started, 1)
                self.state, self.message = "ready", f"Ready (loaded in {self.loaded_seconds:.0f}s)."
            except Exception as e:
                self.state, self.message = "error", f"{type(e).__name__}: {e}"
                raise

    def graph_for(self, model: Optional[str]) -> Any:
        """The debate graph that writes with `model`. Search, citation checks and the debate itself
        are shared; only the language model and the answer cache differ per model."""
        if not model or model == ACTIVE_LLM_NAME or LLM_BACKEND != "api":
            return self.graph
        with self._load_lock:
            if model not in self.graphs:
                import copy

                from src.graph.lexagent_graph import build_lexagent_graph
                from src.novel.adaptive_memory import AdaptiveMemory

                c = self._components
                llm = copy.copy(c["llm"])  # same endpoint, key and settings
                llm.model = model
                safe = re.sub(r"[^A-Za-z0-9_-]", "_", model)
                # Its own answer cache, so a question answered by one model is not replayed for another.
                memory = AdaptiveMemory(namespace=f"ui_chat_{safe}", client=self._memory.client)
                self.graphs[model] = build_lexagent_graph(llm, c["hybrid"], c["cce"], c["ddc"], memory, c["embedder"])
            return self.graphs[model]

    def load_in_background(self) -> None:
        if self.state in ("cold", "error"):
            threading.Thread(target=self._safe_load, daemon=True).start()

    def _safe_load(self) -> None:
        try:
            self.load()
        except Exception:
            pass


ENGINE = Engine()


def _initial_state(query: str) -> dict:
    return {
        "query": query, "jurisdiction": "US_Federal", "debate_round": 0, "debate_history": [],
        "retrieved_passages": [], "entity_map": {},
        "prosecutor_argument": "", "prosecutor_citations": [],
        "defense_argument": "", "defense_citations": [], "defense_challenges": [], "defense_concede": False,
        "challenge_strength": 0.0, "defense_counter_passages": [], "evidence_overlap_ratio": 0.0,
        "reflection_gaps": [], "reflection_queries": [], "reflection_score": 0.0,
        "citation_confidence_report": {}, "avg_ccs": 0.0, "high_risk_citations": [],
        "ddc_decision": "", "ddc_reason": "",
        "judge_verdict": "", "judge_confidence": 0.0, "judge_verified_citations": [], "judge_uncertain_citations": [],
        "judge_reasoning": "", "memory_cache_hit": False, "cached_answer": None, "final_answer": "", "structured_output": {},
    }


def _passage(doc: Any) -> dict:
    meta = getattr(doc, "metadata", None) or {}
    volume, page = str(getattr(doc, "volume", "") or ""), int(getattr(doc, "first_page", 0) or 0)
    return {
        "doc_id": getattr(doc, "doc_id", ""),
        "case_name": getattr(doc, "case_name", "") or "(unnamed)",
        "cite": f"{volume} {getattr(doc, 'reporter', '') or 'U.S.'} {page}" if volume and page else "",
        "year": int(getattr(doc, "year", 0) or 0),
        "court": getattr(doc, "court", "") or "",
        "text": (getattr(doc, "text", "") or "")[:700],
        "score": meta.get("rrf_score"),
    }


def _citation(cite: Any) -> dict:
    data = cite.to_dict() if hasattr(cite, "to_dict") else dict(cite or {})
    return {key: data.get(key) for key in ("case_name", "court", "year", "holding", "self_confidence", "agent_source", "reporter_cite")}


def _plain(value: Any) -> Any:
    """Make a node's output JSON-safe: dataclasses become dicts, unknown objects become strings."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _events_for(node: str, update: dict, round_number: int) -> List[dict]:
    """Turn one graph node's output into the events the UI renders."""
    update = update or {}
    if node == "memory_check":
        if update.get("memory_cache_hit"):
            return [{"type": "memory", "hit": True, "answer": update.get("cached_answer") or update.get("final_answer") or ""}]
        return [{"type": "memory", "hit": False}]
    if node == "retrieve":
        return [{"type": "retrieval", "round": round_number, "passages": [_passage(d) for d in update.get("retrieved_passages", [])]}]
    if node == "prosecutor":
        return [{"type": "prosecutor", "round": round_number, "argument": update.get("prosecutor_argument", ""),
                 "citations": [_citation(c) for c in update.get("prosecutor_citations", [])]}]
    if node == "defense":
        return [{"type": "defense", "round": round_number, "argument": update.get("defense_argument", ""),
                 "citations": [_citation(c) for c in update.get("defense_citations", [])],
                 "challenges": _plain(update.get("defense_challenges", [])),
                 "concede": bool(update.get("defense_concede", False)),
                 "challenge_strength": float(update.get("challenge_strength", 0.0) or 0.0),
                 "evidence_overlap_ratio": float(update.get("evidence_overlap_ratio", 0.0) or 0.0),
                 "counter_passages": [_passage(d) for d in update.get("defense_counter_passages", [])]}]
    if node == "cce":
        checks = []
        for key, result in (update.get("citation_confidence_report") or {}).items():
            cite = result.get("citation") or {}
            checks.append({
                "key": key, "agent": cite.get("agent_source") or key.split("::")[0], "case_name": cite.get("case_name", ""),
                "reporter_cite": cite.get("reporter_cite", ""), "holding": cite.get("holding", ""),
                "tier": result.get("tier"), "ccs": result.get("ccs"), "existence": result.get("existence"), "support": result.get("support"),
                "evidence": result.get("evidence", ""), "evidence_source": result.get("evidence_source", ""),
                "cite_check": result.get("cite_check", ""), "best_match": result.get("best_match", ""),
            })
        return [{"type": "cce", "round": round_number, "avg_ccs": float(update.get("avg_ccs", 0.0) or 0.0), "checks": checks}]
    if node == "reflection":
        return [{"type": "reflection", "round": round_number, "gaps": _plain(update.get("reflection_gaps", [])),
                 "queries": _plain(update.get("reflection_queries", [])), "score": float(update.get("reflection_score", 0.0) or 0.0)}]
    if node == "ddc":
        return [{"type": "ddc", "round": round_number, "decision": update.get("ddc_decision", ""), "reason": update.get("ddc_reason", "")}]
    if node == "judge":
        return [{"type": "judge", "verdict": update.get("judge_verdict", ""), "confidence": float(update.get("judge_confidence", 0.0) or 0.0),
                 "reasoning": update.get("judge_reasoning", ""), "final_answer": update.get("final_answer", "") or update.get("judge_verdict", ""),
                 "verified": [_citation(c) for c in update.get("judge_verified_citations", [])],
                 "uncertain": [_citation(c) for c in update.get("judge_uncertain_citations", [])],
                 "rejected": _plain((update.get("structured_output") or {}).get("rejected_citations", []))}]
    return []


def run_debate(query: str, model: Optional[str] = None) -> Iterator[dict]:
    """Run one debate and yield UI events as each stage finishes."""
    started = time.monotonic()
    if ENGINE.state != "ready":
        yield {"type": "stage", "stage": "loading", "message": "Loading the corpus and the models. The first question takes about two minutes."}
        ENGINE.load()
    model = model or ACTIVE_LLM_NAME
    yield {"type": "stage", "stage": "memory_check", "round": 1, "model": model}

    round_number = 1
    for chunk in ENGINE.graph_for(model).stream(_initial_state(query), stream_mode="updates"):
        for node, update in chunk.items():
            for event in _events_for(node, update, round_number):
                yield event
            if node == "ddc":
                if (update or {}).get("ddc_decision") == "CONTINUE":
                    round_number += 1
                    yield {"type": "stage", "stage": "retrieve", "round": round_number}
                else:
                    yield {"type": "stage", "stage": "judge", "round": round_number}
            elif node == "memory_check" and (update or {}).get("memory_cache_hit"):
                pass
            elif node in NEXT_STAGE and node != "inject_llm":
                yield {"type": "stage", "stage": NEXT_STAGE[node], "round": round_number}
    yield {"type": "done", "rounds": round_number, "seconds": round(time.monotonic() - started, 1)}


# ══════════════════════════════════════════════════════
# Saved results and corpus statistics
# ══════════════════════════════════════════════════════
_CACHE: Dict[str, Any] = {}


def _corpus_stats() -> dict:
    """Corpus figures from the build manifest, the stage ledgers and the case catalog (cached)."""
    stamp = tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (CORPUS_MANIFEST_PATH, CASE_CATALOG_PATH, CAP_DOWNLOAD_CHECKPOINT_PATH))
    if _CACHE.get("corpus_stamp") == stamp:
        return _CACHE["corpus"]

    manifest = _read_json(CORPUS_MANIFEST_PATH) or {}
    download = _read_json(CAP_DOWNLOAD_CHECKPOINT_PATH) or {}
    chunk = _read_json(CAP_CHUNK_CHECKPOINT_PATH) or {}
    indexing = _read_json(INDEXING_CHECKPOINT_PATH) or {}
    from_pdf = manifest.get("volumes_from_official_us_reports_pdf") or []
    indexed = int(manifest.get("volumes_indexed") or 0)

    reports = opinions = 0
    catalog = _read_json(CASE_CATALOG_PATH)
    if catalog:
        from src.data.case_catalog import MIN_OPINION_WORDS
        reports = len(catalog["reports"])
        opinions = sum(1 for row in catalog["reports"] if row[6] >= MIN_OPINION_WORDS)

    stats = {
        "chunks": int(manifest.get("chunks") or 0),
        "reports": reports,
        "opinions": opinions,
        "order_entries": reports - opinions,
        "volumes_target": int(download.get("target_volume_count") or 0),
        "volumes_indexed": indexed,
        "volumes_from_cap": indexed - len(from_pdf),
        "volumes_from_official_pdf": len(from_pdf),
        "volumes_unavailable": manifest.get("volumes_unavailable") or [],
        "built_at": manifest.get("created_at"),
        "stages": [
            {"name": "Download", "detail": "CAP and official PDFs into raw JSON", "counts": download.get("counts") or {}, "updated_at": download.get("updated_at")},
            {"name": "Chunk", "detail": "raw JSON into retrieval chunks", "counts": chunk.get("counts") or {}, "updated_at": chunk.get("updated_at")},
            {"name": "Index", "detail": "chunks into BM25 and ChromaDB", "indexed": indexing.get("last_indexed_index"), "total": indexing.get("total_chunks"), "updated_at": indexing.get("updated_at")},
        ],
        "models": {"generator": ACTIVE_LLM_NAME, "embedder": EMBEDDER_NAME, "nli": NLI_MODEL_NAME, "tau": TAU_ACCEPTANCE_THRESHOLD, "max_rounds": DDC_MAX_ROUNDS},
    }
    _CACHE.update(corpus=stats, corpus_stamp=stamp)
    return stats


def _stress_summary(data: Optional[dict]) -> Optional[dict]:
    if not data:
        return None
    return {key: data.get(key) for key in ("total_evaluated", "power", "false_acceptance_rate_misattributed", "false_acceptance_rate_casehold",
                                           "fabrication_catch_rate", "false_fabrication_rate", "elapsed_seconds", "per_type_counts")}


def _gpu() -> Optional[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        name, total, used, util = [part.strip() for part in out.split(",")]
        return {"name": name, "memory_total_mb": int(total), "memory_used_mb": int(used), "utilization": int(util)}
    except Exception:
        return None


# ══════════════════════════════════════════════════════
# HTTP API
# ══════════════════════════════════════════════════════
app = FastAPI(title="LexAgent UI", docs_url=None, redoc_url=None)
# Pages served from another origin (the Vite dev server, or the public site) may call the API.
# Extra origins: LEXAGENT_ALLOWED_ORIGINS="https://a.example,https://b.example"
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"] + [
    origin.strip().rstrip("/") for origin in os.environ.get("LEXAGENT_ALLOWED_ORIGINS", "").split(",") if origin.strip()
]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None  # one of LLM_API_MODELS; the default model when left out


@app.get("/api/status")
def api_status() -> dict:
    return {"engine": ENGINE.status(),
            "models": {"options": LLM_API_MODELS if LLM_BACKEND == "api" else [ACTIVE_LLM_NAME], "default": ACTIVE_LLM_NAME}, "gpu": _gpu() if LLM_BACKEND == "local" else None,  # only a local model needs the GPU
            "corpus": {key: _corpus_stats()[key] for key in ("chunks", "volumes_indexed", "reports")},
            "sample_transcript": os.path.exists(SAMPLE_TRANSCRIPT)}


@app.post("/api/engine/load")
def api_engine_load() -> dict:
    ENGINE.load_in_background()
    return ENGINE.status()


RESULT_FILES = ("benchmark_100_cases_results.json", "baseline_standard_rag_results.json", "ablation_summary.json",
                "citation_stress_test_results.json", "citation_stress_test_results_before_cce_fix.json")


def _mtimes(paths: List[str]) -> tuple:
    return tuple(os.path.getmtime(path) if os.path.exists(path) else 0 for path in paths)


def _test_set() -> tuple:
    """The benchmark split into evaluable and excluded cases. Cached: it loads the whole case catalog."""
    from experiments.benchmark_cases import split_benchmark_by_source_text

    stamp = _mtimes([CASE_CATALOG_PATH])
    if _CACHE.get("test_set_stamp") != stamp:
        _CACHE.update(test_set=split_benchmark_by_source_text(), test_set_stamp=stamp)
    return _CACHE["test_set"]


@app.get("/api/overview")
def api_overview() -> dict:
    """Everything the dashboard shows. Recomputed only when a result file or the corpus changes."""
    stamp = _mtimes([os.path.join(RESULTS_DIR, name) for name in RESULT_FILES] + [CORPUS_MANIFEST_PATH, CASE_CATALOG_PATH, CAP_DOWNLOAD_CHECKPOINT_PATH, INDEXING_CHECKPOINT_PATH])
    if _CACHE.get("overview_stamp") != stamp:
        _CACHE.update(overview=_build_overview(), overview_stamp=stamp)
    return _CACHE["overview"]


def _build_overview() -> dict:
    from experiments.benchmark_cases import BENCHMARK_100_CASES

    benchmark = _read_json(os.path.join(RESULTS_DIR, "benchmark_100_cases_results.json")) or {}
    baseline = _read_json(os.path.join(RESULTS_DIR, "baseline_standard_rag_results.json")) or {}
    evaluable, excluded = _test_set()

    def mean(rows: List[dict], key: str) -> Optional[float]:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        return sum(values) / len(values) if values else None

    def summarise(rows: List[dict]) -> dict:
        rows = [row for row in rows if row.get("status") != "ERROR"]
        return {
            "cases": len(rows),
            **{key: mean(rows, key) for key in ("hr", "cfr", "oor", "mar", "ucr", "ass", "ar", "ap", "haa", "judge_confidence", "debate_rounds", "total_cites")},
            "converged": sum(1 for row in rows if row.get("dcr_status") == "CONVERGED"),
        }

    return {
        "benchmark": summarise(benchmark.get("per_case") or []) if benchmark else None,
        "benchmark_seconds": (benchmark.get("aggregate") or {}).get("elapsed_seconds"),
        "baseline": summarise(baseline.get("per_case") or []) if baseline else None,
        "ablation": _read_json(os.path.join(RESULTS_DIR, "ablation_summary.json")),
        "stress": _stress_summary(_read_json(os.path.join(RESULTS_DIR, "citation_stress_test_results.json"))),
        "stress_before_fix": _stress_summary(_read_json(os.path.join(RESULTS_DIR, "citation_stress_test_results_before_cce_fix.json"))),
        "test_set": {
            "total": len(BENCHMARK_100_CASES),
            "evaluable": len(evaluable),
            "excluded": [{"id": c["id"], "target_case": c["target_case"], "topic": c["topic"], "reason": c["excluded_reason"]} for c in excluded],
        },
        "corpus": _corpus_stats(),
    }


@app.get("/api/cases")
def api_cases() -> dict:
    from experiments.benchmark_cases import BENCHMARK_100_CASES

    benchmark = {row["case_id"]: row for row in (_read_json(os.path.join(RESULTS_DIR, "benchmark_100_cases_results.json")) or {}).get("per_case", [])}
    baseline = {row["case_id"]: row for row in (_read_json(os.path.join(RESULTS_DIR, "baseline_standard_rag_results.json")) or {}).get("per_case", [])}
    keep = ("hr", "cfr", "oor", "mar", "ucr", "ass", "ar", "ap", "haa", "dcr_status", "judge_confidence", "debate_rounds", "total_cites", "fake_cites", "fake_names", "status")
    rows = []
    for case in BENCHMARK_100_CASES:
        if case["id"] not in benchmark and case["id"] not in baseline:
            continue
        rows.append({
            "id": case["id"], "topic": case["topic"], "query": case["query"], "target_case": case["target_case"],
            "reporter_citation": case.get("reporter_citation", ""), "ground_truth_holding": case["ground_truth_holding"],
            "lexagent": {key: benchmark[case["id"]].get(key) for key in keep} if case["id"] in benchmark else None,
            "baseline": {key: baseline[case["id"]].get(key) for key in keep} if case["id"] in baseline else None,
        })
    return {"cases": rows}


@app.get("/api/suggestions")
def api_suggestions() -> dict:
    from experiments.benchmark_cases import get_benchmark_cases

    picks = [case for case in get_benchmark_cases(100) if case["id"] in (9, 4, 26, 16, 3, 20)]
    return {"suggestions": [{"topic": case["topic"], "query": case["query"]} for case in picks]}


@app.get("/api/transcripts/sample")
def api_sample_transcript() -> JSONResponse:
    data = _read_json(SAMPLE_TRANSCRIPT)
    if data is None:
        raise HTTPException(status_code=404, detail="No sample transcript has been recorded yet.")
    return JSONResponse(data)


@app.post("/api/chat")
def api_chat(request: ChatRequest) -> StreamingResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="The question is empty.")
    model = request.model or ACTIVE_LLM_NAME
    allowed = LLM_API_MODELS if LLM_BACKEND == "api" else [ACTIVE_LLM_NAME]
    if model not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown model {model!r}. Choose one of: {', '.join(allowed)}.")
    if not ENGINE.run_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another question is being answered. One debate runs at a time.")

    events: "queue.Queue[Optional[dict]]" = queue.Queue()
    transcript: List[dict] = []
    started = time.monotonic()

    def worker() -> None:
        try:
            for event in run_debate(query, model):
                event["t"] = round(time.monotonic() - started, 2)
                transcript.append(event)
                events.put(event)
        except Exception as e:
            events.put({"type": "error", "message": f"{type(e).__name__}: {e}", "t": round(time.monotonic() - started, 2)})
        finally:
            ENGINE.run_lock.release()
            events.put(None)
            if transcript and transcript[-1].get("type") == "done":
                try:
                    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
                    record = {"query": query, "model": model, "recorded_at": datetime.now(timezone.utc).isoformat(), "events": transcript}
                    name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
                    for target in (os.path.join(TRANSCRIPT_DIR, name), os.path.join(TRANSCRIPT_DIR, "latest.json")):
                        with open(target, "w", encoding="utf-8") as handle:
                            json.dump(record, handle, ensure_ascii=False, indent=1)
                except OSError:
                    pass

    threading.Thread(target=worker, daemon=True).start()

    def stream() -> Iterator[str]:
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"  # long model calls: keep the connection open
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if os.path.isdir(WEB_DIST):
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="LexAgent web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preload", action="store_true", help="Load the model stack at start-up instead of on the first question")
    args = parser.parse_args()

    if not os.path.isdir(WEB_DIST):
        print(f"[ui] The web app has not been built yet ({WEB_DIST}). See ui/README.md.")
    if args.preload:
        ENGINE.load_in_background()
    # Compute the dashboard data once now, so the first page load does not wait for it.
    threading.Thread(target=lambda: api_overview(), daemon=True).start()
    print(f"[ui] LexAgent UI on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
