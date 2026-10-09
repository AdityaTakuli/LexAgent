# LexAgent v3.0 | test_final_repairs.py
"""Tests for repairs 6 and 7 of the forensic audit.

Repair 6 (ledger protocol): what a debate verified in an earlier round stays available, the judge
reads the record of all rounds and is held to it, and the controller stops when another round has
nothing to repair. Repair 7 (retrieval v2): fusion with a per-case limit.

The earlier protocol ("v1") must stay exactly as it was, because every run made before these
repairs used it. No model, GPU or database is used.

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
from src.agents.state import LexAgentState, debate_protocol  # noqa: E402
from src.novel.cce import make_cce_node  # noqa: E402
from src.novel.ddc import DynamicDebateController  # noqa: E402
from src.novel.ledger import format_for_judge, has_reliable_citation, ledger_key, merge_round, reliable_entries  # noqa: E402
from src.retrieval.retrieval_v2 import fuse, word_tokens  # noqa: E402


def entry(name, tier, ccs=0.5, doc_id=None, holding="A holding.", agent="prosecutor", evidence="the court held so"):
    return {"tier": tier, "ccs": ccs, "resolved_doc_id": doc_id, "resolved_name": name if doc_id else None,
            "resolved_cite": "1 U.S. 1" if doc_id else None, "existence_verdict": "EXISTS", "support_verdict": tier,
            "support_score": ccs, "evidence_window": evidence,
            "citation": {"case_name": name, "holding": holding, "agent_source": agent, "reporter_cite": ""}}


class FakeChecker:
    """Returns a prepared report, as the citation checker would for one round."""

    def __init__(self, report):
        self.report = report

    def run_on_state(self, state):
        return self.report, 0.5


class FakeJudge:
    def __init__(self, verified, uncertain=()):
        self.verified, self.uncertain, self.prompts = list(verified), list(uncertain), []

    def invoke(self, prompt):
        import json
        self.prompts.append(str(prompt))
        return json.dumps({"verdict": "AFFIRMED", "legal_holding": "A warrant is required.",
                           "verified_citations": [{"case_name": n, "holding": "h"} for n in self.verified],
                           "uncertain_citations": [{"case_name": n, "note": "n"} for n in self.uncertain],
                           "rejected_citations": [], "confidence": 0.9, "reasoning": "r"})


class LedgerTests(unittest.TestCase):
    def test_a_supported_citation_survives_a_later_round_that_does_not_repeat_it(self):
        ledger, new = merge_round({}, {"a": entry("Riley v. California", "SUPPORTED", 0.6, "doc1")}, 1)
        self.assertEqual(new, 1)
        ledger, new = merge_round(ledger, {"b": entry("Carpenter v. United States", "NOT_SHOWN", 0.1, "doc2")}, 2)
        self.assertEqual(new, 0)
        self.assertEqual({e["case_name"]: e["tier"] for e in ledger.values()},
                         {"Riley v. California": "SUPPORTED", "Carpenter v. United States": "NOT_SHOWN"})
        self.assertTrue(has_reliable_citation(ledger))

    def test_a_later_worse_verdict_does_not_replace_an_earlier_better_one(self):
        ledger, _ = merge_round({}, {"a": entry("Riley v. California", "SUPPORTED", 0.6, "doc1", evidence="first passage")}, 1)
        ledger, new = merge_round(ledger, {"a": entry("Riley v. California", "NOT_SHOWN", 0.05, "doc1", evidence="other passage")}, 2)
        kept = ledger["report::doc1"]
        self.assertEqual((kept["tier"], kept["evidence"], kept["round"], kept["rounds_cited"]), ("SUPPORTED", "first passage", 1, [1, 2]))
        self.assertEqual(new, 0)

    def test_a_later_better_verdict_replaces_an_earlier_worse_one_and_counts_as_new(self):
        ledger, new = merge_round({}, {"a": entry("Riley v. California", "NOT_SHOWN", 0.05, "doc1")}, 1)
        self.assertEqual((new, has_reliable_citation(ledger)), (0, False))
        ledger, new = merge_round(ledger, {"a": entry("Riley v. California", "SUPPORTED", 0.7, "doc1")}, 2)
        self.assertEqual((new, ledger["report::doc1"]["tier"], ledger["report::doc1"]["round"]), (1, "SUPPORTED", 2))

    def test_one_decision_cited_under_two_spellings_is_one_entry(self):
        report = {"a": entry("Riley v. California", "SUPPORTED", 0.6, "doc1"), "b": entry("Riley v. Cal.", "NOT_SHOWN", 0.1, "doc1")}
        ledger, _ = merge_round({}, report, 1)
        self.assertEqual(list(ledger), ["report::doc1"])
        self.assertEqual(ledger["report::doc1"]["tier"], "SUPPORTED")

    def test_an_unresolved_citation_is_keyed_by_its_cleaned_name(self):
        self.assertEqual(ledger_key(entry("  Smith  v.  JONES. ", "FABRICATED")), "name::smith v jones")
        ledger, _ = merge_round({}, {"a": entry("", "FABRICATED")}, 1)
        self.assertEqual(ledger, {})

    def test_the_input_ledger_is_not_modified(self):
        first, _ = merge_round({}, {"a": entry("Riley v. California", "NOT_SHOWN", 0.1, "doc1")}, 1)
        snapshot = {k: dict(v) for k, v in first.items()}
        merge_round(first, {"a": entry("Riley v. California", "SUPPORTED", 0.9, "doc1")}, 2)
        self.assertEqual(first, snapshot)

    def test_the_judge_reads_verdicts_in_words_and_the_evidence_of_supported_citations(self):
        ledger, _ = merge_round({}, {"a": entry("Riley v. California", "SUPPORTED", 0.6, "doc1", evidence="a warrant is generally required"),
                                     "b": entry("Smith v. Jones", "FABRICATED"), "c": entry("Katz v. United States", "NOT_SHOWN", 0.1, "doc3")}, 1)
        text = format_for_judge(ledger)
        self.assertLess(text.index("Riley v. California"), text.index("Katz v. United States"))
        self.assertIn("SUPPORTED", text)
        self.assertIn("a warrant is generally required", text)
        self.assertIn("NOT SHOWN", text)
        self.assertIn("NOT FOUND", text)
        self.assertNotIn("CCS", text)
        self.assertEqual([e["case_name"] for e in reliable_entries(ledger)], ["Riley v. California"])


class StateTests(unittest.TestCase):
    def test_the_protocol_and_the_ledger_reach_the_nodes(self):
        seen = {}

        def probe(state):
            seen.update(state)
            return {"debate_round": 0}

        builder = StateGraph(LexAgentState)
        builder.add_node("probe", probe)
        builder.set_entry_point("probe")
        builder.add_edge("probe", END)
        builder.compile().invoke({"query": "q", "protocol": "ledger", "citation_ledger": {"k": {"tier": "SUPPORTED"}},
                                  "seen_passage_ids": ["p1"], "round_new_passages": 3, "round_new_supported": 1})
        self.assertEqual((seen.get("protocol"), seen.get("citation_ledger"), seen.get("seen_passage_ids"),
                          seen.get("round_new_passages"), seen.get("round_new_supported")),
                         ("ledger", {"k": {"tier": "SUPPORTED"}}, ["p1"], 3, 1))

    def test_a_state_without_a_protocol_runs_as_before(self):
        self.assertEqual(debate_protocol({}), "v1")
        with self.assertRaises(ValueError):
            debate_protocol({"protocol": "something else"})

    def test_the_checker_node_writes_the_ledger_only_under_the_ledger_protocol(self):
        node = make_cce_node(FakeChecker({"a": entry("Riley v. California", "SUPPORTED", 0.6, "doc1")}))
        self.assertNotIn("citation_ledger", node({"protocol": "v1", "debate_round": 0}))
        update = node({"protocol": "ledger", "debate_round": 1, "citation_ledger": {}})
        self.assertEqual((update["round_new_supported"], update["citation_ledger"]["report::doc1"]["round"]), (1, 2))


class ControllerTests(unittest.TestCase):
    def decide(self, **state):
        base = {"protocol": "ledger", "debate_round": 0, "reflection_gaps": ["gap"], "challenge_strength": 0.8,
                "avg_ccs": 0.1, "citation_confidence_report": {"a": {"ccs": 0.05}}, "round_new_passages": 8, "round_new_supported": 0}
        base.update(state)
        return DynamicDebateController().decide(base)[0]

    def test_a_supported_citation_ends_the_debate_whatever_the_scores_and_the_challenge(self):
        supported = {"report::doc1": {"tier": "SUPPORTED", "ccs": 0.3}}
        self.assertEqual(self.decide(citation_ledger=supported), "TERMINATE")

    def test_nothing_supported_a_gap_and_something_new_continues(self):
        self.assertEqual(self.decide(citation_ledger={"report::doc1": {"tier": "NOT_SHOWN"}}), "CONTINUE")

    def test_nothing_supported_and_nothing_new_ends_the_debate(self):
        self.assertEqual(self.decide(citation_ledger={}, round_new_passages=0, round_new_supported=0), "TERMINATE")

    def test_nothing_supported_and_no_gap_ends_the_debate(self):
        self.assertEqual(self.decide(citation_ledger={}, reflection_gaps=[]), "TERMINATE")

    def test_the_hard_stops_still_apply(self):
        self.assertEqual(self.decide(citation_ledger={}, debate_round=2), "TERMINATE")
        self.assertEqual(self.decide(citation_ledger={}, defense_concede=True), "TERMINATE")

    def test_the_earlier_protocol_is_unchanged(self):
        # A supported citation next to weak scores and a gap: v1 continues, the ledger protocol stops.
        state = {"debate_round": 0, "reflection_gaps": ["gap"], "challenge_strength": 0.0, "avg_ccs": 0.3,
                 "citation_confidence_report": {"a": {"ccs": 0.3}}, "citation_ledger": {"k": {"tier": "SUPPORTED"}}}
        self.assertEqual(DynamicDebateController().decide({**state, "protocol": "v1"})[0], "CONTINUE")
        self.assertEqual(DynamicDebateController().decide(dict(state))[0], "CONTINUE")
        self.assertEqual(DynamicDebateController().decide({**state, "protocol": "ledger"})[0], "TERMINATE")

    def test_without_the_checker_the_ledger_rule_is_not_used(self):
        state = {"protocol": "ledger", "use_cce": False, "debate_round": 0, "reflection_gaps": ["gap"], "challenge_strength": 0.8,
                 "avg_ccs": 0.0, "citation_ledger": {}}
        self.assertEqual(DynamicDebateController().decide(state)[0], "CONTINUE")  # strong challenge and a gap, as in v1


class JudgeTests(unittest.TestCase):
    def ledger(self):
        report = {"a": entry("Riley v. California", "SUPPORTED", 0.6, "doc1"),
                  "b": entry("Katz v. United States", "NOT_SHOWN", 0.1, "doc2"),
                  "c": entry("United States v. Jones", "SUPPORTED", 0.5, "doc3"),
                  "d": entry("Smith v. Maryland", "MISATTRIBUTED", 0.0, "doc4")}
        return merge_round({}, report, 1)[0]

    def run_judge(self, llm, **extra):
        state = {"query": "Is a warrant required?", "llm": llm, "protocol": "ledger", "citation_ledger": self.ledger(),
                 "debate_history": [], "reflection_gaps": [], "citation_confidence_report": {}}
        state.update(extra)
        return judge_node(state)

    def test_the_answer_rests_on_what_an_earlier_round_supported(self):
        # The last round's report is empty: under v1 nothing could be relied on.
        out = self.run_judge(FakeJudge(["Riley v. California"]))
        self.assertEqual([c.case_name for c in out["judge_verified_citations"]], ["Riley v. California"])
        self.assertEqual(out["judge_verified_citations"][0].reporter_cite, "1 U.S. 1")
        self.assertGreater(out["judge_confidence"], 0.35)

    def test_a_citation_whose_holding_was_not_shown_is_reported_as_uncertain_not_dropped(self):
        out = self.run_judge(FakeJudge(["Katz v. United States"]))
        self.assertEqual(out["judge_verified_citations"], [])
        self.assertEqual([c.case_name for c in out["judge_uncertain_citations"]], ["Katz v. United States"])
        self.assertLessEqual(out["judge_confidence"], 0.35)

    def test_contradicted_and_unknown_citations_are_not_counted(self):
        out = self.run_judge(FakeJudge(["Smith v. Maryland", "Miranda v. Arizona"]))
        self.assertEqual((out["judge_verified_citations"], out["judge_uncertain_citations"]), ([], []))
        self.assertLessEqual(out["judge_confidence"], 0.20)

    def test_two_shared_words_are_not_enough_to_stand_in_for_a_supported_case(self):
        # v1 matched on any two shared words: "United States v. Smith" would have passed as "United States v. Jones".
        out = self.run_judge(FakeJudge(["United States v. Smith"]))
        self.assertEqual(out["judge_verified_citations"], [])

    def test_an_abbreviated_name_of_a_supported_case_is_recognised(self):
        out = self.run_judge(FakeJudge(["Riley v. Cal."]))
        self.assertEqual([c.case_name for c in out["judge_verified_citations"]], ["Riley v. California"])

    def test_a_case_listed_twice_is_counted_once(self):
        out = self.run_judge(FakeJudge(["Riley v. California", "Riley v. California"], uncertain=["Riley v. California"]))
        self.assertEqual((len(out["judge_verified_citations"]), len(out["judge_uncertain_citations"])), (1, 0))

    def test_the_judge_is_shown_the_record_of_all_rounds_and_no_score_thresholds(self):
        llm = FakeJudge(["Riley v. California"])
        out = self.run_judge(llm)
        prompt = llm.prompts[0]
        self.assertIn("Citation Record (all rounds of the debate)", prompt)
        self.assertIn("Riley v. California", prompt)
        self.assertIn("NOT SHOWN", prompt)
        self.assertNotIn("CCS >= 0.75", prompt)
        self.assertEqual(out["structured_output"]["confidence_components"]["decisions_cited_in_the_debate"], 4)

    def test_the_earlier_protocol_still_reads_the_last_report_only(self):
        llm = FakeJudge(["Riley v. California"])
        state = {"query": "q", "llm": llm, "protocol": "v1", "citation_ledger": self.ledger(), "debate_history": [],
                 "reflection_gaps": [], "citation_confidence_report": {}}
        out = judge_node(state)
        self.assertEqual(out["judge_verified_citations"], [])
        self.assertIn("CCS >= 0.75", llm.prompts[0])

    def test_without_the_checker_the_judge_relies_on_its_own_citations(self):
        out = self.run_judge(FakeJudge(["Miranda v. Arizona"]), use_cce=False)
        self.assertEqual([c.case_name for c in out["judge_verified_citations"]], ["Miranda v. Arizona"])


class FusionTests(unittest.TestCase):
    def test_reciprocal_rank_fusion_prefers_what_both_rankings_return(self):
        parents = ["a", "b", "c", "d"]
        fused = fuse([[0, 1, 2], [2, 3, 0]], top=4, per_case=0, parents=parents)
        self.assertEqual([i for i, _ in fused][:2], [0, 2])

    def test_at_most_two_passages_of_one_case(self):
        parents = ["x", "x", "x", "x", "y", "z"]
        fused = fuse([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], top=4, per_case=2, parents=parents)
        self.assertEqual([i for i, _ in fused], [0, 1, 4, 5])

    def test_a_list_shortened_by_the_limit_is_filled_from_deeper_candidates(self):
        from src.retrieval.retrieval_v2 import fuse_and_fill
        parents = ["x", "x", "x", "x", "y", "z", "w"]
        dense, lexical = [0, 1, 2, 3, 4, 6], [0, 1, 2, 3, 5, 6]
        # The first four of each ranking are one case: two passages survive the limit.
        self.assertEqual([i for i, _ in fuse([dense[:4], lexical[:4]], 4, 2, parents)], [0, 1])
        filled = [i for i, _ in fuse_and_fill([dense, lexical], 4, 2, parents, depth=4)]
        self.assertEqual(filled[:2], [0, 1])           # the top of the list is unchanged
        self.assertEqual(filled[2], 6)                 # the next passage both rankings return
        self.assertEqual(len(filled), len(set(filled)))
        self.assertEqual(len(filled), 4)
        self.assertEqual(sum(1 for i in filled if parents[i] == "x"), 2)

    def test_word_tokens_drop_punctuation_and_case(self):
        self.assertEqual(word_tokens("Miranda v. Arizona, 384 U.S. 436 (1966)"), "miranda v arizona 384 u s 436 1966")



class StandardMeasureTests(unittest.TestCase):
    """The measures of experiments/v2/standard_metrics.py on hand-made cases (no model)."""

    def setUp(self):
        from rouge_score import rouge_scorer

        from experiments.v2 import standard_metrics
        self.module = standard_metrics
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.case = {"id": 1, "ground_truth_holding": "Police need a warrant to search a phone."}
        self.target = {"doc_id": "doc_riley", "name": "Riley v. California"}

    def exists(self, name, cite=""):
        known = {"Riley v. California": ("EXISTS", "doc_riley", True), "Katz v. United States": ("EXISTS", "doc_katz", True),
                 "Order v. Entry": ("EXISTS", "doc_order", False), "Far v. Away": ("NOT_IN_CORPUS", None, False)}
        verdict, doc_id, opinion = known.get(name, ("NOT_FOUND", None, False))
        return {"existence_verdict": verdict, "resolved_doc_id": doc_id, "resolved_is_opinion": opinion}

    def support(self, supported_pairs):
        class Fake:
            def supports(self, doc_id, statement):
                return (doc_id, statement) in supported_pairs
        return Fake()

    def score(self, answer, citations, supported_pairs=(), retrieved=("doc_other", "doc_riley"), in_prompt=None):
        trace = {"retrieved": [{"parent_doc_id": p} for p in retrieved], "passages_in_prompt": in_prompt or len(retrieved)}
        row = {"case_id": 1, "final_answer": answer, "answer_citations": [{"case_name": n} for n in citations], "trace": trace, "seconds": 4.0}
        return self.module.standard_case(row, self.case, self.target, self.exists, self.support(set(supported_pairs)), self.rouge)

    def test_a_supported_answer_is_grounded(self):
        answer = "Police need a warrant to search a phone. The exception for arrests does not cover digital data."
        first, second = self.module.split_sentences(answer)
        out = self.score(answer, ["Riley v. California"], [("doc_riley", first)])
        self.assertEqual((out["grounded"], out["misgrounded"], out["no_authority"]), (1.0, 0.0, 0.0))
        self.assertEqual((out["sentences"], out["sentences_supported"], out["sentence_support"]), (2, 1, 0.5))
        self.assertEqual((out["citations_judged"], out["citations_supporting"], out["authority_accuracy"]), (1, 1, 1.0))

    def test_an_existing_case_that_supports_nothing_makes_the_answer_misgrounded(self):
        answer = "Police need a warrant to search a phone."
        out = self.score(answer, ["Riley v. California", "Katz v. United States"], [("doc_riley", answer)])
        self.assertEqual((out["grounded"], out["misgrounded"]), (0.0, 1.0))
        self.assertEqual((out["citations_judged"], out["citations_supporting"]), (2, 1))

    def test_a_case_that_does_not_exist_is_fabricated_and_misgrounded(self):
        out = self.score("Police need a warrant to search a phone.", ["Smith v. Nobody"])
        self.assertEqual((out["has_fabricated"], out["citations_fabricated"], out["misgrounded"], out["authority_accuracy"]), (1.0, 1, 1.0, 0.0))

    def test_an_answer_without_a_citation_has_no_authority(self):
        out = self.score("Police need a warrant to search a phone.", [])
        self.assertEqual((out["no_authority"], out["grounded"], out["sentence_support"], out["citations"]), (1.0, 0.0, 0.0, 0))

    def test_citations_that_cannot_be_checked_are_left_out_of_precision(self):
        out = self.score("Police need a warrant to search a phone.", ["Far v. Away", "Order v. Entry"])
        self.assertEqual((out["citations"], out["citations_judged"], out["citations_cannot_be_checked"]), (2, 0, 2))
        self.assertEqual((out["misgrounded"], out["grounded"]), (0.0, 0.0))

    def test_retrieval_measures_use_the_rank_of_the_target_opinion(self):
        out = self.score("Police need a warrant to search a phone.", [], retrieved=("a", "b", "doc_riley", "c", "d"), in_prompt=2)
        self.assertEqual((out["hit_at_1"], out["hit_at_4"], out["hit_at_8"]), (0.0, 1.0, 1.0))
        self.assertAlmostEqual(out["reciprocal_rank"], 1 / 3)
        self.assertEqual(out["target_in_context"], 0.0)  # third passage, and the generator was given two

    def test_a_debate_row_uses_its_first_round_and_counts_its_generator_calls(self):
        row = {"case_id": 1, "final_answer": "Police need a warrant to search a phone.", "answer_citations": [],
               "trace": {"rounds": [{"retrieved": [{"parent_doc_id": "doc_riley"}]}, {"retrieved": [{"parent_doc_id": "x"}]}], "llm_calls": [1, 2, 3]}}
        out = self.module.standard_case(row, self.case, self.target, self.exists, self.support(set()), self.rouge)
        self.assertEqual((out["hit_at_1"], out["llm_calls"], out["target_in_context"]), (1.0, 3.0, 1.0))

    def test_run_summary_uses_micro_rates_and_leaves_out_skipped_debates(self):
        records = [
            {"skipped": False, "citations": 2, "citations_judged": 2, "citations_supporting": 1, "citations_fabricated": 0, "citations_cannot_be_checked": 0,
             "sentences": 3, "sentences_supported": 2},
            {"skipped": False, "citations": 2, "citations_judged": 1, "citations_supporting": 1, "citations_fabricated": 1, "citations_cannot_be_checked": 1,
             "sentences": 1, "sentences_supported": 0},
            {"skipped": True, "citations": 9, "citations_judged": 9, "citations_supporting": 9, "citations_fabricated": 9, "citations_cannot_be_checked": 0,
             "sentences": 9, "sentences_supported": 9},
        ]
        out = self.module.summarise_run(records)
        self.assertEqual((out["cases"], out["skipped"], out["citations"]), (3, 1, 4))
        self.assertAlmostEqual(out["citation_precision"], 2 / 3)
        self.assertAlmostEqual(out["citation_recall"], 2 / 4)
        self.assertAlmostEqual(out["fabricated_citation_rate"], 1 / 4)

    def test_sentences_are_not_cut_at_case_name_abbreviations(self):
        text = "In Riley v. California, 573 U.S. 373 (2014), the Court required a warrant. The exception does not apply to data."
        self.assertEqual(len(self.module.split_sentences(text)), 2)
        self.assertEqual(self.module.split_sentences("Yes."), [])

    def test_token_f1(self):
        self.assertAlmostEqual(self.module.token_f1("a warrant is required", "police need a warrant"), 0.5)
        self.assertEqual(self.module.token_f1("", "police need a warrant"), 0.0)

class EndToEndTests(unittest.TestCase):
    """The whole debate graph on the real state schema, with a scripted model, retriever and checker.

    Round one: the prosecutor cites Riley, which is supported. Round two (if it happens): it cites
    Katz only, whose holding is not shown. The answer of the judge relies on Riley.
    """

    class Model:
        def __init__(self):
            self.judge_prompts, self.prosecutor_calls = [], 0

        def invoke(self, prompt):
            import json
            head = str(prompt)[:400]
            if "SENIOR JUDGE" in head:
                self.judge_prompts.append(str(prompt))
                return json.dumps({"verdict": "AFFIRMED", "legal_holding": "A warrant is required to search a phone.",
                                   "verified_citations": [{"case_name": "Riley v. California", "holding": "A warrant is required."}],
                                   "uncertain_citations": [], "rejected_citations": [], "confidence": 0.9, "reasoning": "Riley controls."})
            if "REFLECTION AGENT" in head:
                return json.dumps({"gaps": ["scope of the exception"], "targeted_queries": ["search incident to arrest digital data"],
                                   "argument_quality": 0.6, "continue_debate": True, "reason": "open"})
            if "DEFENSE ATTORNEY" in head:
                return json.dumps({"challenges": ["the exception applies"], "counter_argument": "No warrant is needed.",
                                   "counter_citations": [], "challenge_strength": 0.8, "concede": False})
            self.prosecutor_calls += 1
            name = "Riley v. California" if self.prosecutor_calls == 1 else "Katz v. United States"
            return json.dumps({"argument": "A warrant is required.", "overall_confidence": 0.8,
                               "citations": [{"case_name": name, "holding": "A warrant is required.", "self_confidence": 0.9}]})

    class Retriever:
        def __init__(self):
            self.calls = 0

        def retrieve(self, query, jurisdiction=""):
            from src.data.corpus_schema import CorpusDocument
            self.calls += 1
            return [CorpusDocument(doc_id=f"case{self.calls}::chunk::{i}", text="text", source="CAP", case_name=f"Case {self.calls}") for i in range(2)]

        def get_entity_map(self, results):
            return {}

    class Checker:
        def run_on_state(self, state):
            report = {}
            for citation in state.get("prosecutor_citations", []) + state.get("defense_citations", []):
                supported = citation.case_name == "Riley v. California"
                report[citation.case_name] = entry(citation.case_name, "SUPPORTED" if supported else "NOT_SHOWN", 0.5 if supported else 0.05,
                                                   "doc_riley" if supported else "doc_katz")
            return report, 0.3

    def run_debate(self, protocol):
        from experiments.run_ablation import build_ablation_graph
        from experiments.v2.runner import initial_state

        model, retriever = self.Model(), self.Retriever()
        graph = build_ablation_graph("A3", model, retriever, self.Checker(), DynamicDebateController(), None, None)
        final = graph.invoke(initial_state({"query": "Is a warrant required to search a phone?"}, "A3", protocol))
        return final, model, retriever

    def test_under_the_ledger_protocol_the_debate_stops_once_a_citation_is_supported(self):
        final, model, retriever = self.run_debate("ledger")
        self.assertEqual(model.prosecutor_calls, 1)
        self.assertEqual([c.case_name for c in final["judge_verified_citations"]], ["Riley v. California"])
        self.assertIn("Citation Record (all rounds of the debate)", model.judge_prompts[0])
        self.assertEqual(final["citation_ledger"]["report::doc_riley"]["tier"], "SUPPORTED")
        self.assertIn("A supported citation exists", final["ddc_reason"])

    def test_under_the_earlier_protocol_the_supported_citation_of_round_one_is_lost(self):
        final, model, retriever = self.run_debate("v1")
        self.assertEqual(model.prosecutor_calls, 3)           # every debate ran to the maximum
        self.assertEqual(final["judge_verified_citations"], [])  # Riley was supported in round one and is gone
        self.assertEqual(final.get("citation_ledger"), {})


class BaselineCitationParsingTests(unittest.TestCase):
    """A single agent that writes its citations as plain strings has still cited them."""

    def test_citations_written_as_objects(self):
        from src.evaluation.metrics_v2 import citations_from_baseline_output
        raw = 'Sure. {"final_answer": "Yes.", "citations": [{"case_name": "Terry v. Ohio", "reporter_cite": "392 U.S. 1"}, {"case_name": ""}]}'
        self.assertEqual(citations_from_baseline_output(raw), [{"case_name": "Terry v. Ohio", "reporter_cite": "392 U.S. 1"}])

    def test_citations_written_as_plain_strings(self):
        from src.evaluation.metrics_v2 import citations_from_baseline_output
        raw = '{"final_answer": "Yes.", "citations": ["Miranda v. Arizona, 384 U.S. 436 (1966)", "Edwards v. Arizona", "Berghuis v. Thompkins (2010)", 7]}'
        self.assertEqual(citations_from_baseline_output(raw), [
            {"case_name": "Miranda v. Arizona", "reporter_cite": "384 U.S. 436"},
            {"case_name": "Edwards v. Arizona", "reporter_cite": ""},
            {"case_name": "Berghuis v. Thompkins", "reporter_cite": ""}])

    def test_no_json_is_reported_as_unknown_and_no_citation_list_as_empty(self):
        from src.evaluation.metrics_v2 import citations_from_baseline_output
        self.assertIsNone(citations_from_baseline_output("The answer is yes."))
        self.assertIsNone(citations_from_baseline_output('{"final_answer": "cut off'))
        self.assertEqual(citations_from_baseline_output('{"final_answer": "Yes."}'), [])

    def test_scoring_reads_baseline_citations_again_and_keeps_what_the_run_saved(self):
        from experiments.v2.score import reparse_baseline_citations
        rows = [
            {"variant": "baseline", "answer_citations": [], "pool_citations": [],
             "trace": {"raw_output": '{"final_answer": "Yes.", "citations": ["Terry v. Ohio", "Florida v. Royer"]}'}},
            {"variant": "baseline", "answer_citations": [{"case_name": "Terry v. Ohio"}], "pool_citations": [{"case_name": "Terry v. Ohio"}],
             "trace": {"raw_output": '{"final_answer": "Yes.", "citations": [{"case_name": "Terry v. Ohio"}]}'}},
            {"variant": "baseline", "answer_citations": [{"case_name": "Kept v. AsSaved"}], "trace": {"raw_output": "no json at all"}},
        ]
        report = reparse_baseline_citations(rows)
        self.assertEqual((report["answers"], report["answers_whose_citations_changed"]), (3, 1))
        self.assertEqual((report["citations_as_saved_by_the_run"], report["citations_read_from_the_raw_output"]), (2, 4))
        self.assertEqual([c["case_name"] for c in rows[0]["answer_citations"]], ["Terry v. Ohio", "Florida v. Royer"])
        self.assertEqual(rows[0]["answer_citations_as_run"], [])
        self.assertEqual(rows[0]["pool_citations"], rows[0]["answer_citations"])
        self.assertEqual([c["case_name"] for c in rows[2]["answer_citations"]], ["Kept v. AsSaved"])

    def test_debate_rows_are_not_touched(self):
        from experiments.v2.score import reparse_baseline_citations
        rows = [{"variant": "Full", "answer_citations": [{"case_name": "Terry v. Ohio"}], "trace": {"rounds": []}}]
        self.assertIsNone(reparse_baseline_citations(rows))
        self.assertNotIn("answer_citations_as_run", rows[0])


if __name__ == "__main__":
    unittest.main()
