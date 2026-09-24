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
from src.agents.defense_agent import defense_node
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


# ─── ABLATION STUBS ─────────────────────────────

def _stub_defense_node(state: LexAgentState) -> dict:
    return {
        "defense_argument": "",
        "defense_citations": [],
        "defense_challenges": [],
        "defense_concede": True,
        "challenge_strength": 0.0,
    }


def _make_stub_cce_node():
    def stub_cce_node(state: LexAgentState) -> dict:
        report = {}
        all_cites = state.get("prosecutor_citations", []) + state.get("defense_citations", [])
        for i, c in enumerate(all_cites):
            c_name = c.get("case_name", "") if isinstance(c, dict) else getattr(c, "case_name", "")
            report[f"cite_{i}_{c_name}"] = {
                "ccs": 1.0,
                "tier": "VERIFIED",
                "citation": {"case_name": c_name},
                "evidence": "A2 Ablation: Unverified assumption"
            }
        return {
            "citation_confidence_report": report,
            "avg_ccs": 1.0,
        }
    return stub_cce_node


def _make_stub_memory_nodes():
    def stub_memory_check(state: LexAgentState) -> dict:
        return {"memory_cache_hit": False, "high_risk_citations": []}
    def stub_memory_store(state: LexAgentState) -> dict:
        return {}
    return stub_memory_check, stub_memory_store


def build_ablation_graph(variant: str, llm, hybrid, cce, ddc, memory, embedder):
    memory_check_node, memory_store_node = make_memory_nodes(memory, embedder)
    cce_node_fn = make_cce_node(cce)
    ddc_node_fn = make_ddc_node(ddc)

    # Apply stubs per variant
    if variant == "A1":
        active_defense = _stub_defense_node
    else:
        active_defense = defense_node

    if variant == "A2":
        active_cce = _make_stub_cce_node()
    else:
        active_cce = cce_node_fn

    if variant == "A3":
        active_mem_check, active_mem_store = _make_stub_memory_nodes()
    else:
        active_mem_check, active_mem_store = memory_check_node, memory_store_node

    builder = StateGraph(LexAgentState)
    builder.add_node("memory_check", active_mem_check)
    builder.add_node("retrieve", _make_retrieve_node(hybrid))
    builder.add_node("inject_llm", _make_inject_llm_node(llm))
    builder.add_node("prosecutor", prosecutor_node)
    builder.add_node("defense", active_defense)
    builder.add_node("cce", active_cce)
    builder.add_node("reflection", reflection_node)
    builder.add_node("ddc", ddc_node_fn)
    builder.add_node("judge", judge_node)
    builder.add_node("memory_store", active_mem_store)

    builder.set_entry_point("memory_check")
    builder.add_conditional_edges("memory_check", _route_after_memory_check,
                                  {"cache_hit": END, "proceed": "retrieve"})
    builder.add_edge("retrieve", "inject_llm")
    builder.add_edge("inject_llm", "prosecutor")
    builder.add_edge("prosecutor", "defense")
    builder.add_edge("defense", "cce")
    builder.add_edge("cce", "reflection")
    builder.add_edge("reflection", "ddc")

    if variant == "A4":
        # Fixed 1-round: always terminate directly to Judge
        builder.add_edge("ddc", "judge")
    else:
        builder.add_conditional_edges("ddc", _route_after_ddc,
                                      {"CONTINUE": "retrieve", "TERMINATE": "judge"})

    builder.add_edge("judge", "memory_store")
    builder.add_edge("memory_store", END)
    return builder.compile()


def run_ablation_study(limit: int = 10):
    print("\n" + "=" * 75)
    print(f"  LEXAGENT v3.0 ABLATION STUDIES ({limit} Cases per Variant)")
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
    collection = client.get_collection("cap_authorities" if "cap_authorities" in colls else "legal_corpus")

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
    memory = AdaptiveMemory()

    variants = [
        ("Full", "Full LexAgent v3.0 (DVA+)"),
        ("A1", "w/o Defense Node (No adversarial cross-examination)"),
        ("A2", "w/o CCE (No citation confidence verification)"),
        ("A3", "w/o Adaptive Memory (Stateless inference)"),
        ("A4", "w/o DDC (Fixed 1-round termination)"),
    ]

    test_cases = BENCHMARK_30_CASES[:limit]
    summary_table = []

    for v_code, v_name in variants:
        print(f"\nEvaluating Variant [{v_code}]: {v_name}...")
        graph = build_ablation_graph(v_code, llm, hybrid, cce, ddc, memory, embedder)

        cards = []
        for case in test_cases:
            state_input = {
                "query": case["query"], "jurisdiction": "US_Federal",
                "debate_round": 0, "debate_history": [],
                "retrieved_passages": [], "entity_map": {},
                "prosecutor_argument": "", "prosecutor_citations": [],
                "defense_argument": "", "defense_citations": [],
                "defense_challenges": [], "defense_concede": False,
                "challenge_strength": 0.0, "reflection_gaps": [],
                "reflection_queries": [], "reflection_score": 0.0,
                "citation_confidence_report": {}, "avg_ccs": 0.0,
                "high_risk_citations": [], "ddc_decision": "",
                "ddc_reason": "", "judge_verdict": "",
                "judge_confidence": 0.0, "judge_verified_citations": [],
                "judge_uncertain_citations": [], "judge_reasoning": "",
                "memory_cache_hit": False, "cached_answer": None,
                "final_answer": "", "structured_output": {},
            }
            try:
                res = graph.invoke(state_input)
                cards.append(evaluate_single_case(res, case, known_cases, embedder))
            except Exception as e:
                cards.append({"hr": 0.0, "cas": 0.0, "judge_confidence": 0.0})

        m_hr = sum(c["hr"] for c in cards) / len(cards) if cards else 0.0
        m_cas = sum(c["cas"] for c in cards) / len(cards) if cards else 0.0
        m_conf = sum(c["judge_confidence"] for c in cards) / len(cards) if cards else 0.0

        summary_table.append({
            "variant": v_code,
            "description": v_name,
            "mean_hr": m_hr,
            "mean_cas": m_cas,
            "mean_confidence": m_conf,
        })
        print(f"  -> Result: Mean HR={m_hr:.2%} | Mean CAS={m_cas:.4f} | Conf={m_conf:.2f}")

    print("\n" + "=" * 80)
    print(f"  {'Variant':<8} | {'Mean HR':<10} | {'Mean CAS':<10} | {'Confidence':<12} | {'Description'}")
    print("-" * 80)
    for row in summary_table:
        print(f"  {row['variant']:<8} | {row['mean_hr']:<10.2%} | {row['mean_cas']:<10.4f} | {row['mean_confidence']:<12.2f} | {row['description']}")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ablation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)
    print(f"\nAblation results saved to: {out_path}")
    return summary_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Cases per ablation variant (default: 10)")
    args = parser.parse_args()
    run_ablation_study(limit=args.limit)
