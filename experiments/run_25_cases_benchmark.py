# LexAgent v3.0 | run_25_cases_benchmark.py
"""
Comprehensive 25-30 Case LexAgent v3.0 DVA+ Benchmark Runner (CAP Edition).
Executes benchmark evaluation across constitutional and statutory questions
against the primary Harvard CAP authority database.
"""

import os
import sys
import time
import json
import warnings
import logging
import argparse
from typing import List, Dict, Any, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["CHROMA_TELEMETRY"] = "false"
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from fuzzywuzzy import fuzz
except ImportError:
    try:
        from thefuzz import fuzz
    except ImportError:
        import difflib
        class FuzzFallback:
            @staticmethod
            def ratio(s1, s2):
                return int(difflib.SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio() * 100)
        fuzz = FuzzFallback()

try:
    from sentence_transformers import util
except ImportError:
    util = None

from config import (
    CHROMA_CORPUS_PATH,
    DRIVE_CHROMA_CORPUS_PATH,
    LOCAL_CHROMA_CORPUS_PATH,
    BM25_INDEX_PATH,
    KNOWN_CASES_PATH,
    RESULTS_DIR,
)
from src.utils.chroma_sync import ensure_chroma_ready
from src.models.load_model import load_embedder, load_mistral_7b
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.novel.cce import CitationConfidenceEngine
from src.novel.ddc import DynamicDebateController
from src.novel.adaptive_memory import AdaptiveMemory
from src.graph.lexagent_graph import build_lexagent_graph
from experiments.benchmark_cases import BENCHMARK_30_CASES


def silence_all_logs():
    logging.getLogger().setLevel(logging.ERROR)
    for name in list(logging.root.manager.loggerDict.keys()):
        l = logging.getLogger(name)
        l.setLevel(logging.ERROR)
        l.propagate = False


def _extract_case_names(citations_list: list) -> List[str]:
    names = []
    for c in citations_list or []:
        if isinstance(c, dict):
            name = c.get("case_name", "")
        else:
            name = getattr(c, "case_name", "")
        if name and name.strip():
            names.append(name.strip())
    return names


def evaluate_single_case(
    result: dict,
    case_info: dict,
    known_cases: set,
    embedder: Any,
    cas_threshold: float = 0.70,
    fuzzy_threshold: int = 85
) -> dict:
    all_cites = set()
    for name in _extract_case_names(result.get("prosecutor_citations", [])):
        all_cites.add(name)
    for name in _extract_case_names(result.get("defense_citations", [])):
        all_cites.add(name)
    for name in _extract_case_names(result.get("judge_verified_citations", [])):
        all_cites.add(name)

    total_cites = len(all_cites)
    fake_cites = 0
    fake_names = []
    normalized_known = {c.lower().strip() for c in known_cases if c}

    for cite in all_cites:
        cite_lower = cite.lower().strip()
        if cite_lower in normalized_known:
            continue
        best_r = max((fuzz.ratio(cite_lower, kc) for kc in normalized_known), default=0)
        if best_r < fuzzy_threshold:
            fake_cites += 1
            fake_names.append(cite)

    case_hr = round(fake_cites / total_cites, 4) if total_cites > 0 else 0.0

    # CAS
    case_cas = 0.0
    gt_holding = case_info.get("ground_truth_holding", "")
    final_answer = result.get("final_answer", "")

    if gt_holding and final_answer and embedder and util:
        try:
            p_vec = embedder.encode(final_answer[:500], convert_to_tensor=True)
            g_vec = embedder.encode(gt_holding[:500], convert_to_tensor=True)
            sim = float(util.cos_sim(p_vec, g_vec))
            case_cas = round(sim, 4)
        except Exception:
            case_cas = 0.0

    # DCR
    dcr_status = "CONVERGED" if (result.get("defense_concede", False) or result.get("judge_confidence", 0.0) >= 0.70) else "CONTESTED"

    return {
        "case_id": case_info.get("id"),
        "topic": case_info.get("topic"),
        "hr": case_hr,
        "cas": case_cas,
        "dcr_status": dcr_status,
        "evidence_overlap_ratio": float(result.get("evidence_overlap_ratio", 0.0)),
        "judge_confidence": result.get("judge_confidence", 0.0),
        "debate_rounds": result.get("debate_round", 0) + 1,
        "total_cites": total_cites,
        "fake_cites": fake_cites,
        "fake_names": fake_names,
    }


def run_benchmark(limit: int = 25, model_tuple=None, embedder=None, memory_namespace: Optional[str] = None) -> dict:
    silence_all_logs()
    print("\n" + "=" * 75)
    print(f"  LEXAGENT v3.0 (CAP EDITION) — {limit}-CASE BENCHMARK EVALUATION")
    print("=" * 75)

    import pickle
    import chromadb

    if not os.path.exists(KNOWN_CASES_PATH):
        raise FileNotFoundError(f"Database catalog not found at {KNOWN_CASES_PATH}. Please run DB build first.")

    with open(KNOWN_CASES_PATH, "rb") as f:
        known_cases = pickle.load(f)
    print(f"  Loaded {len(known_cases)} canonical CAP cases and citations.")

    if embedder is None:
        embedder = load_embedder()

    if model_tuple is None:
        _, _, llm = load_mistral_7b()
    else:
        llm = model_tuple[2]

    # Initialize Chroma & HybridRetriever
    active_path = ensure_chroma_ready(LOCAL_CHROMA_CORPUS_PATH, DRIVE_CHROMA_CORPUS_PATH)
    client = chromadb.PersistentClient(path=active_path)
    colls = [c.name for c in client.list_collections()]
    coll_name = "cap_authorities" if "cap_authorities" in colls else (colls[0] if colls else "cap_authorities")
    collection = client.get_or_create_collection(coll_name)
    
    dense_retriever = DenseRetriever(collection, embedder)

    from src.data.build_corpus import load_corpus
    from config import CORPUS_JSON_PATH
    corpus_chunks = load_corpus(CORPUS_JSON_PATH) if os.path.exists(CORPUS_JSON_PATH) else []
    
    if os.path.exists(BM25_INDEX_PATH) and corpus_chunks:
        bm25_retriever = BM25Retriever.load_index(BM25_INDEX_PATH, corpus_chunks)
    else:
        bm25_retriever = BM25Retriever(corpus_chunks)

    hybrid = HybridRetriever(dense_retriever, bm25_retriever)
    cce = CitationConfidenceEngine(collection, embedder, KNOWN_CASES_PATH)
    ddc = DynamicDebateController()

    # Isolate memory per benchmark run to avoid returning stale cached answers on re-runs
    if memory_namespace is None:
        memory_namespace = f"bench_{int(time.time())}"
    memory = AdaptiveMemory(namespace=memory_namespace)

    graph = build_lexagent_graph(llm, hybrid, cce, ddc, memory, embedder)

    test_cases = BENCHMARK_30_CASES[:limit]
    scorecards = []
    start_time = time.time()

    for idx, case in enumerate(test_cases, 1):
        print(f"\n[{idx}/{len(test_cases)}] Case #{case['id']}: {case['topic']}")
        print(f"      Q: {case['query'][:90]}...")

        state_input = {
            "query": case["query"],
            "jurisdiction": case.get("jurisdiction", "US_Federal"),
            "debate_round": 0,
            "debate_history": [],
            "retrieved_passages": [],
            "entity_map": {},
            "prosecutor_argument": "",
            "prosecutor_citations": [],
            "defense_argument": "",
            "defense_citations": [],
            "defense_challenges": [],
            "defense_concede": False,
            "challenge_strength": 0.0,
            "defense_counter_passages": [],
            "evidence_overlap_ratio": 0.0,
            "reflection_gaps": [],
            "reflection_queries": [],
            "reflection_score": 0.0,
            "citation_confidence_report": {},
            "avg_ccs": 0.0,
            "high_risk_citations": [],
            "ddc_decision": "",
            "ddc_reason": "",
            "judge_verdict": "",
            "judge_confidence": 0.0,
            "judge_verified_citations": [],
            "judge_uncertain_citations": [],
            "judge_reasoning": "",
            "memory_cache_hit": False,
            "cached_answer": None,
            "final_answer": "",
            "structured_output": {},
        }

        try:
            result = graph.invoke(state_input)
            card = evaluate_single_case(result, case, known_cases, embedder)
            card["status"] = "SUCCESS"
            scorecards.append(card)
            print(f"      -> HR: {card['hr']:.2%} | CAS: {card['cas']:.4f} | Conf: {card['judge_confidence']:.2f} | Status: {card['dcr_status']}")
        except Exception as e:
            print(f"      [!] Execution error on case {case['id']}: {e}")
            scorecards.append({
                "case_id": case['id'], "topic": case['topic'],
                "status": "ERROR", "error": str(e),
                "hr": None, "cas": None, "dcr_status": "ERROR",
                "judge_confidence": None, "debate_rounds": 1,
                "total_cites": 0, "fake_cites": 0, "fake_names": []
            })

    elapsed = time.time() - start_time

    # Compute aggregate metrics (exclude crashed cases from hallucination/accuracy means)
    valid_scorecards = [s for s in scorecards if s.get("status") != "ERROR" and s.get("hr") is not None]
    n_crashed = len(scorecards) - len(valid_scorecards)

    avg_hr = sum(s["hr"] for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    avg_cas = sum(s["cas"] for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    avg_conf = sum(s["judge_confidence"] for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    avg_eor = sum(s.get("evidence_overlap_ratio", 0.0) for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    dcr_rate = sum(1 for s in valid_scorecards if s.get("dcr_status") == "CONVERGED") / len(valid_scorecards) if valid_scorecards else 0.0

    print("\n" + "=" * 75)
    print("  AGGREGATE BENCHMARK SCORECARD SUMMARY")
    print("=" * 75)
    print(f"  Total Cases Evaluated:       {len(scorecards)}")
    print(f"  Completed Successfully:      {len(valid_scorecards)}")
    print(f"  Crashed/Failed Cases:        {n_crashed}")
    print(f"  Mean Hallucination Rate (HR): {avg_hr:.2%}" + (" (excluding crashes)" if n_crashed > 0 else ""))
    print(f"  Mean Citation Accuracy (CAS): {avg_cas:.4f}")
    print(f"  Mean Evidence Overlap (EOR):  {avg_eor:.2%}")
    print(f"  Mean Judge Confidence:       {avg_conf:.2f}")
    print(f"  Debate Convergence Rate (DCR):{dcr_rate:.2%}")
    print(f"  Total Execution Time:        {elapsed:.1f}s")
    print("=" * 75)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"benchmark_{limit}_cases_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "aggregate": {
                "total_cases": len(scorecards),
                "completed_cases": len(valid_scorecards),
                "crashed_cases": n_crashed,
                "mean_hr": avg_hr,
                "mean_cas": avg_cas,
                "mean_eor": avg_eor,
                "mean_conf": avg_conf,
                "dcr": dcr_rate,
                "elapsed_seconds": elapsed,
            },
            "per_case": scorecards
        }, f, indent=2)
    print(f"\nDetailed scorecard saved to: {out_path}")

    return {
        "total_cases": len(scorecards),
        "completed_cases": len(valid_scorecards),
        "crashed_cases": n_crashed,
        "mean_hr": avg_hr,
        "mean_cas": avg_cas,
        "dcr": dcr_rate,
        "scorecards": scorecards
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25, help="Number of benchmark cases to evaluate (default: 25)")
    parser.add_argument("--memory-namespace", type=str, default=None, help="Memory namespace for benchmark isolation")
    parser.add_argument("--reuse-memory", action="store_true", help="Reuse global memory without run isolation")
    args = parser.parse_args()

    mem_ns = "" if args.reuse_memory else args.memory_namespace
    run_benchmark(limit=args.limit, memory_namespace=mem_ns)
