# LexAgent v3.0 | test_ablation_switches.py
"""Tests that the ablation switches reach the debate nodes.

LangGraph drops any input key that the state schema does not name. The switches were once
missing from the schema, so the judge and the debate controller always saw "component is on",
even in the variant that removes the citation checker. No model, GPU or database is used.

    python -m unittest discover -s tests -v
"""

import logging
import os
import sys
import tempfile
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("LEXAGENT_DATA_DIR", tempfile.mkdtemp(prefix="lexagent_test_data_"))
logging.disable(logging.INFO)

from langgraph.graph import END, StateGraph  # noqa: E402

from src.agents.judge_agent import judge_node  # noqa: E402
from src.agents.state import LexAgentState  # noqa: E402
from src.novel.ddc import DynamicDebateController  # noqa: E402

SWITCHES = ("use_defense", "use_cce", "use_memory")


def state_seen_by_a_node(state_input: dict) -> dict:
    """Run a one-node graph on the real state schema and return the state that node received."""
    seen = {}

    def probe(state):
        seen.update(state)
        return {"debate_round": 0}

    builder = StateGraph(LexAgentState)
    builder.add_node("probe", probe)
    builder.set_entry_point("probe")
    builder.add_edge("probe", END)
    builder.compile().invoke(state_input)
    return seen


class FakeJudgeLLM:
    """Returns a fixed ruling that relies on one case and reports its own confidence."""

    def invoke(self, prompt):
        return (
            '{"verdict": "AFFIRMED", "legal_holding": "A warrant is required.", '
            '"verified_citations": [{"case_name": "Riley v. California", "holding": "A warrant is required.", "confidence": 0.8}], '
            '"uncertain_citations": [], "rejected_citations": [], "confidence": 0.9, "reasoning": "The cited case controls."}'
        )


class AblationSwitchTests(unittest.TestCase):
    def test_switches_survive_in_the_graph_state(self):
        seen = state_seen_by_a_node({"query": "q", **{name: False for name in SWITCHES}})
        for name in SWITCHES:
            self.assertIn(name, seen, f"{name} was dropped before it reached a node")
            self.assertIs(seen[name], False)

    def test_switches_default_to_on_when_not_given(self):
        seen = state_seen_by_a_node({"query": "q"})
        for name in SWITCHES:
            self.assertTrue(seen.get(name, True))

    def test_controller_ignores_citation_scores_when_the_checker_is_off(self):
        # With the checker removed there are no scores: the empty-report average of 0.0 must not
        # be read as "weak citations" and keep the debate going.
        state = {"avg_ccs": 0.0, "reflection_gaps": ["an open point"], "challenge_strength": 0.0,
                 "debate_round": 0, "citation_confidence_report": {}}
        controller = DynamicDebateController()
        self.assertEqual(controller.decide({**state, "use_cce": True})[0], "CONTINUE")
        self.assertEqual(controller.decide({**state, "use_cce": False})[0], "TERMINATE")

    def test_judge_keeps_its_own_citations_and_confidence_when_the_checker_is_off(self):
        state = {"query": "May police search a phone without a warrant?", "llm": FakeJudgeLLM(),
                 "citation_confidence_report": {}, "debate_history": [], "debate_round": 0}
        with_checker = judge_node({**state, "use_cce": True})
        without_checker = judge_node({**state, "use_cce": False})
        # Checker on, nothing verified: the judge may not rely on the case, and confidence is capped.
        self.assertEqual(with_checker["judge_verified_citations"], [])
        self.assertLessEqual(with_checker["judge_confidence"], 0.20)
        # Checker off: the judge's own citation stands and so does its own confidence.
        self.assertEqual([c.case_name for c in without_checker["judge_verified_citations"]], ["Riley v. California"])
        self.assertAlmostEqual(without_checker["judge_confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
