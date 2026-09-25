# LexAgent v3.0 | phase3_run.py
"""Phase 3 Smoke Test: Full DVA+ Protocol pipeline with active CCE, DDC, and Adaptive Memory."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.bootstrap import load_lexagent_components
from src.graph.lexagent_graph import build_lexagent_graph


def run_phase3_test(query: str = "Does the Fourteenth Amendment protect abortion privacy?"):
    print("\n" + "=" * 70)
    print("  PHASE 3 SMOKE TEST: FULL DVA+ PIPELINE (CCE + DDC + MEMORY)")
    print("=" * 70)
    print(f"  Query: '{query}'")

    comp = load_lexagent_components(load_llm=True)
    graph = build_lexagent_graph(
        comp["llm"], comp["hybrid"], comp["cce"], comp["ddc"], comp["memory"], comp["embedder"]
    )

    state_input = {
        "query": query,
        "jurisdiction": "US_Federal",
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

    result = graph.invoke(state_input)

    print("\n  DEBATE DYNAMICS:")
    print(f"  Rounds Completed: {result.get('debate_round', 0) + 1}")
    print(f"  DDC Final Decision: {result.get('ddc_decision', '')} ({result.get('ddc_reason', '')})")
    print(f"  Average Citation Confidence (CCS): {result.get('avg_ccs', 0.0):.4f}")
    print(f"  Evidence Overlap Ratio (EOR): {result.get('evidence_overlap_ratio', 0.0):.2%}")

    verified = result.get("judge_verified_citations", [])
    print(f"\n  VERIFIED CITATIONS ({len(verified)}):")
    for c in verified:
        c_name = c.case_name if hasattr(c, "case_name") else c.get("case_name", "")
        conf = c.self_confidence if hasattr(c, "self_confidence") else c.get("self_confidence", 0.0)
        print(f"  - {c_name} (Confidence: {conf:.2f})")

    print("\n  FINAL VERDICT:")
    print("  " + str(result.get("final_answer", "")))
    print(f"  Judge Grounded Confidence: {result.get('judge_confidence', 0.0):.2f}")
    print("=" * 70)
    print("  PHASE 3 SMOKE TEST PASSED!")
    print("=" * 70)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Does the Fourteenth Amendment protect abortion privacy?")
    args = parser.parse_args()
    run_phase3_test(args.query)
