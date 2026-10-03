# LexAgent v3.0 | phase2_run.py
"""Phase 2 Smoke Test: Fixed 1-round debate demonstration over CAP database."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langgraph.graph import END, StateGraph
from experiments.bootstrap import load_lexagent_components
from src.agents.prosecutor_agent import prosecutor_node
from src.agents.defense_agent import defense_node, make_defense_node
from src.agents.reflection_agent import reflection_node
from src.agents.judge_agent import judge_node
from src.agents.state import LexAgentState
from src.graph.lexagent_graph import _make_inject_llm_node, _make_retrieve_node


def phase2_cce(state):
    cites = state.get("prosecutor_citations", []) + state.get("defense_citations", [])
    return {
        "citation_confidence_report": {
            f"stub::{i}": {"tier": "UNCERTAIN", "ccs": 0.5, "citation": cite.to_dict()}
            for i, cite in enumerate(cites)
        },
        "avg_ccs": 0.5
    }


def phase2_ddc(state):
    return {"ddc_decision": "TERMINATE", "ddc_reason": "Phase 2 fixed single-round control"}


def build_phase2_graph(llm, hybrid):
    graph = StateGraph(LexAgentState)
    graph.add_node("retrieve", _make_retrieve_node(hybrid))
    graph.add_node("inject_llm", _make_inject_llm_node(llm))
    graph.add_node("prosecutor", prosecutor_node)
    graph.add_node("defense", make_defense_node(hybrid))
    graph.add_node("cce", phase2_cce)
    graph.add_node("reflection", reflection_node)
    graph.add_node("ddc", phase2_ddc)
    graph.add_node("judge", judge_node)

    graph.set_entry_point("retrieve")
    for left, right in [("retrieve", "inject_llm"), ("inject_llm", "prosecutor"),
                        ("prosecutor", "defense"), ("defense", "cce"),
                        ("cce", "reflection"), ("reflection", "ddc"), ("ddc", "judge")]:
        graph.add_edge(left, right)
    graph.add_edge("judge", END)
    return graph.compile()


def run_phase2_test(query: str = "Does the Fourteenth Amendment protect abortion privacy?"):
    print("\n" + "=" * 70)
    print("  PHASE 2 SMOKE TEST: FIXED 1-ROUND DEBATE (PROSECUTOR & DEFENSE)")
    print("=" * 70)
    print(f"  Query: '{query}'")

    components = load_lexagent_components(load_llm=True)
    graph = build_phase2_graph(components["llm"], components["hybrid"])

    result = graph.invoke({
        "query": query,
        "jurisdiction": "US_Federal",
        "debate_round": 0,
        "debate_history": [],
        "high_risk_citations": []
    })

    print("\n  PROSECUTOR ARGUMENT:")
    print("  " + str(result.get("prosecutor_argument", ""))[:300] + "...")
    print("\n  DEFENSE COUNTER-ARGUMENT:")
    print("  " + str(result.get("defense_argument", ""))[:300] + "...")
    print("\n  FINAL JUDICIAL VERDICT:")
    print("  " + str(result.get("final_answer", "")))
    print(f"  Judge Confidence: {result.get('judge_confidence', 0.0):.2f}")
    print("=" * 70)

    # Correctness assertions per Review.md Section 14
    assert result is not None, "Pipeline execution returned None"
    assert len(str(result.get("final_answer", "")).strip()) > 0, "Final answer is empty"
    assert "judge_confidence" in result, "judge_confidence missing from state"
    assert "prosecutor_argument" in result, "prosecutor_argument missing from state"
    assert "defense_argument" in result, "defense_argument missing from state"

    print("  PHASE 2 EXECUTION & CORRECTNESS CHECK PASSED!")
    print("=" * 70)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Does the Fourteenth Amendment protect abortion privacy?")
    args = parser.parse_args()
    run_phase2_test(args.query)
