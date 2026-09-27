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
    normalized_known = {c.lower().strip() for c in known_cases if c}

    # Federal known landmark list for off-corpus vs fabricated distinction
    known_scotus_registry = {
        "lockett v. ohio", "argersinger v. hamlin", "milliken v. bradley",
        "moran v. burbine", "gertz v. robert welch, inc.", "arkansas v. sanders",
        "california v. acevedo", "city of arlington v. fed. commc'ns comm'n",
        "bellotti v. baird", "first national bank v. bellotti", "milkovich v. lorain journal co.",
        "g.m. leasing corp. v. united states", "kolender v. lawson",
        "marbury v. madison", "mcculloch v. maryland", "gibbons v. ogden",
        "loving v. virginia", "tinker v. des moines", "new york v. quarles",
    }

    fake_cites = 0
    fake_names = []
    fabricated_cites = 0
    off_corpus_cites = 0

    for cite in all_cites:
        cite_lower = cite.lower().strip()
        if cite_lower in normalized_known:
            continue
        best_r = max((fuzz.ratio(cite_lower, kc) for kc in normalized_known), default=0)
        if best_r < fuzzy_threshold:
            fake_cites += 1
            fake_names.append(cite)
            # Differentiate genuine landmark cases outside local slice from pure fabrications
            is_real_scotus = any(fuzz.ratio(cite_lower, sc) >= 80 for sc in known_scotus_registry)
            if is_real_scotus:
                off_corpus_cites += 1
            else:
                fabricated_cites += 1

    case_hr = round(fake_cites / total_cites, 4) if total_cites > 0 else 0.0
    case_cfr = round(fabricated_cites / total_cites, 4) if total_cites > 0 else 0.0
    case_oor = round(off_corpus_cites / total_cites, 4) if total_cites > 0 else 0.0

    # Misattribution Rate (MAR) from CCE report
    cce_report = result.get("citation_confidence_report", {})
    misattributed_count = sum(1 for v in cce_report.values() if isinstance(v, dict) and v.get("tier") == "MISATTRIBUTED")
    supported_count = sum(1 for v in cce_report.values() if isinstance(v, dict) and v.get("tier") == "SUPPORTED")
    cce_total = len(cce_report) if cce_report else total_cites
    case_mar = round(misattributed_count / cce_total, 4) if cce_total > 0 else 0.0
    case_haa = round(supported_count / cce_total, 4) if cce_total > 0 else 0.0

    # Answer Semantic Similarity (ASS) — formerly misnamed as CAS
    case_ass = 0.0
    gt_holding = case_info.get("ground_truth_holding", "")
    final_answer = result.get("final_answer", "")

    if gt_holding and final_answer and embedder and util:
        try:
            p_vec = embedder.encode(final_answer[:500], convert_to_tensor=True)
            g_vec = embedder.encode(gt_holding[:500], convert_to_tensor=True)
            sim = float(util.cos_sim(p_vec, g_vec))
            case_ass = round(sim, 4)
        except Exception:
            case_ass = 0.0

    # Authority Recall (AR) & Authority Precision (AP)
    gold_target = case_info.get("target_case", "").lower().strip()
    gold_found = any(fuzz.ratio(gold_target, c.lower().strip()) >= 80 for c in all_cites) if gold_target else False
    case_ar = 1.0 if gold_found else 0.0
    case_ap = round((1.0 if gold_found else 0.0) / total_cites, 4) if total_cites > 0 else 0.0

    # Unsupported Claim Rate (UCR)
    rejected_cites = len(result.get("structured_output", {}).get("rejected_citations", []))
    case_ucr = round(rejected_cites / (total_cites + rejected_cites), 4) if (total_cites + rejected_cites) > 0 else 0.0

    # DCR: True convergence requires consensus or verified confidence without early crash
    dcr_status = "CONVERGED" if (result.get("defense_concede", False) or result.get("judge_confidence", 0.0) >= 0.70) else "CONTESTED"

    return {
        "case_id": case_info.get("id"),
        "topic": case_info.get("topic"),
        "hr": case_hr,
        "cfr": case_cfr,
        "oor": case_oor,
        "mar": case_mar,
        "ucr": case_ucr,
        "ass": case_ass,
        "cas": case_ass,  # Kept as alias for backward compatibility
        "ar": case_ar,
        "ap": case_ap,
        "haa": case_haa,
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
    avg_cfr = sum(s.get("cfr", 0.0) for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    avg_oor = sum(s.get("oor", 0.0) for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    avg_mar = sum(s.get("mar", 0.0) for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    avg_ucr = sum(s.get("ucr", 0.0) for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0

    avg_ass = sum(s.get("ass", 0.0) for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    avg_ap = sum(s.get("ap", 0.0) for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    avg_ar = sum(s.get("ar", 0.0) for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    avg_haa = sum(s.get("haa", 0.0) for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0

    avg_conf = sum(s["judge_confidence"] for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    avg_eor = sum(s.get("evidence_overlap_ratio", 0.0) for s in valid_scorecards) / len(valid_scorecards) if valid_scorecards else 0.0
    dcr_rate = sum(1 for s in valid_scorecards if s.get("dcr_status") == "CONVERGED") / len(valid_scorecards) if valid_scorecards else 0.0

    print("\n" + "=" * 75)
    print("  AGGREGATE BENCHMARK SCORECARD SUMMARY")
    print("=" * 75)
    print(f"  Total Cases Evaluated:           {len(scorecards)}")
    print(f"  Completed Successfully:          {len(valid_scorecards)}")
    print(f"  Crashed/Failed Cases:            {n_crashed}")
    print(f"  Mean Hallucination Rate (HR):    {avg_hr:.2%}")
    print(f"    ├─ Citation Fabrication Rate (CFR): {avg_cfr:.2%}")
    print(f"    ├─ Off-Corpus Precedent Rate (OOR): {avg_oor:.2%}")
    print(f"    └─ Misattribution Rate (MAR):       {avg_mar:.2%}")
    print(f"  Unsupported Claim Rate (UCR):    {avg_ucr:.2%}")
    print(f"  Answer Semantic Similarity (ASS):{avg_ass:.4f}")
    print(f"  Authority Recall (AR):           {avg_ar:.2%}")
    print(f"  Authority Precision (AP):        {avg_ap:.2%}")
    print(f"  Holding Attribution Acc (HAA):   {avg_haa:.2%}")
    print(f"  Mean Evidence Overlap (EOR):     {avg_eor:.2%}")
    print(f"  Mean Judge Confidence:           {avg_conf:.2f}")
    print(f"  Debate Convergence Rate (DCR):   {dcr_rate:.2%}")
    print(f"  Total Execution Time:            {elapsed:.1f}s")
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
                "mean_cfr": avg_cfr,
                "mean_oor": avg_oor,
                "mean_mar": avg_mar,
                "mean_ucr": avg_ucr,
                "mean_ass": avg_ass,
                "mean_cas": avg_ass,
                "mean_ar": avg_ar,
                "mean_ap": avg_ap,
                "mean_haa": avg_haa,
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
        "mean_cfr": avg_cfr,
        "mean_oor": avg_oor,
        "mean_mar": avg_mar,
        "mean_ass": avg_ass,
        "mean_ar": avg_ar,
        "dcr": dcr_rate,
        "scorecards": scorecards
    }


def run_single_agent_rag_baseline(limit: int = 25) -> dict:
    """Executes a Standard Single-Agent RAG Baseline (Retrieve -> Mistral-7B Prompt) for comparison."""
    silence_all_logs()
    print("\n" + "=" * 75)
    print(f"  STANDARD SINGLE-AGENT RAG BASELINE ({limit} Cases)")
    print("=" * 75)

    import pickle
    import chromadb

    with open(KNOWN_CASES_PATH, "rb") as f:
        known_cases = pickle.load(f)

    embedder = load_embedder()
    _, _, llm = load_mistral_7b()

    active_path = ensure_chroma_ready(LOCAL_CHROMA_CORPUS_PATH, DRIVE_CHROMA_CORPUS_PATH)
    client = chromadb.PersistentClient(path=active_path)
    colls = [c.name for c in client.list_collections()]
    coll_name = "cap_authorities" if "cap_authorities" in colls else (colls[0] if colls else "cap_authorities")
    collection = client.get_or_create_collection(coll_name)

    from src.data.build_corpus import load_corpus
    corpus_chunks = load_corpus(CORPUS_JSON_PATH) if os.path.exists(CORPUS_JSON_PATH) else []
    dense_retriever = DenseRetriever(collection, embedder)
    if os.path.exists(BM25_INDEX_PATH) and corpus_chunks:
        bm25_retriever = BM25Retriever.load_index(BM25_INDEX_PATH, corpus_chunks)
    else:
        bm25_retriever = BM25Retriever(corpus_chunks)
    hybrid = HybridRetriever(dense_retriever, bm25_retriever)

    test_cases = BENCHMARK_30_CASES[:limit]
    scorecards = []

    for i, case in enumerate(test_cases, 1):
        print(f"  [{i:2d}/{limit}] Baseline Case: {case['topic']}...")
        passages = hybrid.retrieve(case["query"])
        context_str = "\n".join([f"[{j+1}] {p.case_name}: {p.text[:300]}" for j, p in enumerate(passages[:4])])
        prompt = (
            f"[INST] You are an attorney. Answer the following legal question based on the retrieved precedents.\n"
            f"Question: {case['query']}\n"
            f"Precedents:\n{context_str}\n"
            f"Provide your answer and cite relevant case names in JSON format: {{\"final_answer\": \"...\", \"citations\": [{{\"case_name\": \"...\"}}]}} [/INST]"
        )
        try:
            raw = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
            import re
            m = re.search(r'\{.*\}', str(raw), re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {"final_answer": str(raw), "citations": []}
            res = {
                "final_answer": parsed.get("final_answer", str(raw)),
                "prosecutor_citations": parsed.get("citations", []),
                "defense_citations": [],
                "judge_verified_citations": [],
                "evidence_overlap_ratio": 0.0,
                "judge_confidence": 0.70,
                "debate_round": 0,
            }
            card = evaluate_single_case(res, case, known_cases, embedder)
            card["status"] = "SUCCESS"
            scorecards.append(card)
        except Exception as e:
            scorecards.append({"case_id": case["id"], "topic": case["topic"], "status": "ERROR", "error": str(e), "hr": None})

    valid = [s for s in scorecards if s.get("status") != "ERROR" and s.get("hr") is not None]
    avg_hr = sum(s["hr"] for s in valid) / len(valid) if valid else 0.0
    avg_ass = sum(s.get("ass", 0.0) for s in valid) / len(valid) if valid else 0.0

    out_path = os.path.join(RESULTS_DIR, "baseline_standard_rag_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": {"total_cases": len(scorecards), "mean_hr": avg_hr, "mean_ass": avg_ass}, "per_case": scorecards}, f, indent=2)
    print(f"\nBaseline Standard RAG results saved to: {out_path}")
    print(f"Baseline Mean HR: {avg_hr:.2%} | Mean ASS: {avg_ass:.4f}")
    return {"mean_hr": avg_hr, "mean_ass": avg_ass}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25, help="Number of benchmark cases to evaluate (default: 25)")
    parser.add_argument("--memory-namespace", type=str, default=None, help="Memory namespace for benchmark isolation")
    parser.add_argument("--reuse-memory", action="store_true", help="Reuse global memory without run isolation")
    parser.add_argument("--baseline", action="store_true", help="Run standard single-agent RAG baseline instead of debate")
    args = parser.parse_args()

    if args.baseline:
        run_single_agent_rag_baseline(limit=args.limit)
    else:
        mem_ns = "" if args.reuse_memory else args.memory_namespace
        run_benchmark(limit=args.limit, memory_namespace=mem_ns)

