# LexAgent v3.0 | run_ablation.py
"""
Ablation Study Suite — 4 Variants Removing One Core Component Each
===================================================================
  A1: w/o Defense node (Prosecutor -> CCE -> Reflection -> DDC -> Judge)
  A2: w/o CCE (bypass CCE; avg_ccs=1.0, unverified assumption)
  A3: w/o Adaptive Memory (stateless pass-through)
  A4: w/o DDC (fixed 1-round termination)

Compares results to show each novel component's contribution to hallucination reduction.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    CHROMA_CORPUS_PATH,
    DRIVE_CHROMA_CORPUS_PATH,
    LOCAL_CHROMA_CORPUS_PATH,
    BM25_INDEX_PATH,
    KNOWN_CASES_PATH,
    CORPUS_JSON_PATH,
    RESULTS_DIR,
)
from src.utils.chroma_sync import ensure_chroma_ready
from src.agents.state import LexAgentState
from src.agents.prosecutor_agent import prosecutor_node
from src.agents.defense_agent import defense_node, make_defense_node
from src.agents.reflection_agent import reflection_node
from src.agents.judge_agent import judge_node
from src.novel.cce import CitationConfidenceEngine, make_cce_node
from src.novel.ddc import DynamicDebateController, make_ddc_node
from src.novel.adaptive_memory import AdaptiveMemory, make_memory_nodes
from src.graph.lexagent_graph import (
    _make_retrieve_node,
    _make_inject_llm_node,
    _route_after_memory_check,
    _route_after_ddc,
)
from src.models.load_model import load_embedder, load_mistral_7b
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.logger import get_logger
from experiments.benchmark_cases import BENCHMARK_30_CASES
from experiments.run_25_cases_benchmark import evaluate_single_case

logger = get_logger(__name__)


# ─── CLEAN ABLATION GRAPH BUILDER ─────────────────────────────

def build_ablation_graph(variant: str, llm, hybrid, cce, ddc, memory, embedder):
    memory_check_node, memory_store_node = make_memory_nodes(memory, embedder)
    cce_node_fn = make_cce_node(cce)
    ddc_node_fn = make_ddc_node(ddc)

    builder = StateGraph(LexAgentState)
    builder.add_node("retrieve", _make_retrieve_node(hybrid))
    builder.add_node("inject_llm", _make_inject_llm_node(llm))
    builder.add_node("prosecutor", prosecutor_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("ddc", ddc_node_fn)
    builder.add_node("judge", judge_node)

    # 1. Memory integration (A3: true removal of memory check/store)
    if variant != "A3":
        builder.add_node("memory_check", memory_check_node)
        builder.add_node("memory_store", memory_store_node)
        builder.set_entry_point("memory_check")
        builder.add_conditional_edges("memory_check", _route_after_memory_check,
                                      {"cache_hit": END, "proceed": "retrieve"})
        builder.add_edge("judge", "memory_store")
        builder.add_edge("memory_store", END)
    else:
        builder.set_entry_point("retrieve")
        builder.add_edge("judge", END)

    builder.add_edge("retrieve", "inject_llm")
    builder.add_edge("inject_llm", "prosecutor")

    # 2. Defense node integration (A1: true removal of Defense node from graph)
    if variant != "A1":
        builder.add_node("defense", make_defense_node(hybrid))
        builder.add_edge("prosecutor", "defense")
        after_defense = "defense"
    else:
        after_defense = "prosecutor"

    # 3. CCE node integration (A2: true removal of CCE node from graph)
    if variant != "A2":
        builder.add_node("cce", cce_node_fn)
        builder.add_edge(after_defense, "cce")
        builder.add_edge("cce", "reflection")
    else:
        builder.add_edge(after_defense, "reflection")

    builder.add_edge("reflection", "ddc")

    # 4. DDC termination routing (A4: fixed single-round termination)
    if variant == "A4":
        builder.add_edge("ddc", "judge")
    else:
        builder.add_conditional_edges("ddc", _route_after_ddc,
                                      {"CONTINUE": "retrieve", "TERMINATE": "judge"})

    return builder.compile()


def run_ablation_study(limit: int = 30):
    test_cases = BENCHMARK_30_CASES if (limit is None or limit <= 0 or limit >= len(BENCHMARK_30_CASES)) else BENCHMARK_30_CASES[:limit]
    print("\n" + "=" * 75)
    print(f"  LEXAGENT v3.0 ABLATION STUDIES ({len(test_cases)} Cases per Variant — Full Run)")
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

    cce = CitationConfidenceEngine(collection, embedder, KNOWN_CASES_PATH)
    ddc = DynamicDebateController()
    ablation_run_id = int(time.time())

    variants = [
        ("Full", "Full LexAgent v3.0 (DVA+)"),
        ("A1", "w/o Defense Node (No adversarial cross-examination)"),
        ("A2", "w/o CCE (No citation confidence verification)"),
        ("A3", "w/o Adaptive Memory (Stateless inference)"),
        ("A4", "w/o DDC (Fixed 1-round termination)"),
    ]

    # Evaluates on full benchmark suite or specified limit
    summary_table = []

    for v_code, v_name in variants:
        print(f"\nEvaluating Variant [{v_code}]: {v_name}...")
        variant_memory = AdaptiveMemory(namespace=f"ablation_{ablation_run_id}_{v_code}")
        graph = build_ablation_graph(v_code, llm, hybrid, cce, ddc, variant_memory, embedder)

        cards = []
        for case in test_cases:
            state_input = {
                "query": case["query"], "jurisdiction": "US_Federal",
                "debate_round": 0, "debate_history": [],
                "retrieved_passages": [], "entity_map": {},
                "prosecutor_argument": "", "prosecutor_citations": [],
                "defense_argument": "", "defense_citations": [],
                "defense_challenges": [], "defense_concede": False,
                "challenge_strength": 0.0, "defense_counter_passages": [],
                "evidence_overlap_ratio": 0.0, "reflection_gaps": [],
                "reflection_queries": [], "reflection_score": 0.0,
                "citation_confidence_report": {}, "avg_ccs": 0.0,
                "high_risk_citations": [], "ddc_decision": "",
                "ddc_reason": "", "judge_verdict": "",
                "judge_confidence": 0.0, "judge_verified_citations": [],
                "judge_uncertain_citations": [], "judge_reasoning": "",
                "memory_cache_hit": False, "cached_answer": None,
                "final_answer": "", "structured_output": {},
                "use_defense": (v_code != "A1"),
                "use_cce": (v_code != "A2"),
                "use_memory": (v_code != "A3"),
            }
            try:
                res = graph.invoke(state_input)
                card = evaluate_single_case(res, case, known_cases, embedder)
                card["status"] = "SUCCESS"
                cards.append(card)

            except Exception as e:
                logger.error("Execution error on variant [%s] case %s: %s", v_code, case.get("id"), e)
                cards.append({
                    "case_id": case.get("id"),
                    "topic": case.get("topic", ""),
                    "status": "ERROR",
                    "error": str(e),
                    "hr": None,
                    "cas": None,
                    "judge_confidence": None,
                })

        valid_cards = [c for c in cards if c.get("status") != "ERROR" and c.get("hr") is not None]
        n_crashed = len(cards) - len(valid_cards)

        m_hr = sum(c["hr"] for c in valid_cards) / len(valid_cards) if valid_cards else 0.0
        m_cas = sum(c["cas"] for c in valid_cards) / len(valid_cards) if valid_cards else 0.0
        m_conf = sum(c["judge_confidence"] for c in valid_cards) / len(valid_cards) if valid_cards else 0.0

        summary_table.append({
            "variant": v_code,
            "description": v_name,
            "total_cases": len(cards),
            "completed_cases": len(valid_cards),
            "crashed_cases": n_crashed,
            "mean_hr": m_hr,
            "mean_cas": m_cas,
            "mean_confidence": m_conf,
        })
        crash_msg = f" (Crashed: {n_crashed})" if n_crashed > 0 else ""
        print(f"  -> Result: Mean HR={m_hr:.2%} | Mean CAS={m_cas:.4f} | Conf={m_conf:.2f} | Completed: {len(valid_cards)}/{len(cards)}{crash_msg}")

    print("\n" + "=" * 96)
    print(f"  {'Variant':<8} | {'Completed':<10} | {'Crashed':<8} | {'Mean HR':<10} | {'Mean CAS':<10} | {'Confidence':<12} | {'Description'}")
    print("-" * 96)
    for row in summary_table:
        comp_str = f"{row['completed_cases']}/{row['total_cases']}"
        print(f"  {row['variant']:<8} | {comp_str:<10} | {row['crashed_cases']:<8} | {row['mean_hr']:<10.2%} | {row['mean_cas']:<10.4f} | {row['mean_confidence']:<12.2f} | {row['description']}")
    print("=" * 96)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ablation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)
    print(f"\nAblation results saved to: {out_path}")
    return summary_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30, help="Cases per ablation variant (default: 30 for full offline suite, 0 for all)")
    args = parser.parse_args()
    run_ablation_study(limit=args.limit)
