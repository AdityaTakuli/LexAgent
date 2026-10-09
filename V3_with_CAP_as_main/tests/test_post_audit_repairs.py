# LexAgent v3.0 | test_post_audit_repairs.py
"""Tests for the repairs made after the forensic audit.

  repair 2  the verdict cache answers only the same question
  repair 3  measures that say what they measure (src/evaluation/metrics_v2.py)
  repair 4  the citation checker states existence and support separately, and the support verdict
            acts on the rest of the system only through an open validation gate

No model, GPU or corpus is used: the index, the embedder and the NLI model are small fakes.

    python -m unittest discover -s tests -v
"""

import logging
import os
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("LEXAGENT_DATA_DIR", tempfile.mkdtemp(prefix="lexagent_test_data_"))
logging.disable(logging.INFO)

from src.data.case_catalog import CaseCatalog, Report  # noqa: E402
from src.data.corpus_schema import CitationRecord  # noqa: E402
from src.evaluation import metrics_v2  # noqa: E402
from src.novel.adaptive_memory import AdaptiveMemory, make_memory_nodes, normalize_question  # noqa: E402

REPORTS = [
    Report(doc_id="riley", name="Riley v. California", volume="573", first_page=373, year=2014, chunks=2, words=5000),
    Report(doc_id="katz", name="Katz v. United States", volume="389", first_page=347, year=1967, chunks=1, words=4000),
    Report(doc_id="katz-order", name="Katz v. United States", volume="386", first_page=954, year=1967, chunks=1, words=20),
]


class SameVectorEmbedder:
    """Every text gets the same vector: under a similarity rule every question matches every other."""

    def encode(self, text, **kwargs):
        return np.ones(4, dtype=np.float32)


class OpinionCollection:
    """The passages of each opinion, as ChromaDB would return them for a parent_doc_id filter."""

    def __init__(self, passages):
        self.passages = passages

    def get(self, where=None, include=None, **kwargs):
        texts = self.passages.get((where or {}).get("parent_doc_id", {}).get("$eq"), [])
        return {"ids": [f"p{i}" for i in range(len(texts))], "documents": list(texts),
                "metadatas": [{"chunk_index": i} for i in range(len(texts))], "embeddings": [np.ones(4, dtype=np.float32) for _ in texts]}

    def query(self, **kwargs):
        return {"documents": [[]], "metadatas": [[]]}


class KeywordNLI:
    """Entailment when the passage contains the holding's first word; [contradiction, entailment, neutral]."""

    class _Config:
        label2id = {"contradiction": 0, "entailment": 1, "neutral": 2}

    class _Model:
        pass

    def __init__(self):
        self.model = self._Model()
        self.model.config = self._Config()

    def predict(self, pairs, **kwargs):
        return np.array([[0.0, 6.0, 0.0] if hypothesis.split()[0].lower() in passage.lower() else [0.0, -6.0, 6.0] for passage, hypothesis in pairs])


class VerdictCacheTests(unittest.TestCase):
    QUESTION = "May police search a cell phone seized from an arrestee without a warrant?"
    OTHER = "May police search a car incident to arrest without a warrant?"

    def memory(self, mode):
        return AdaptiveMemory(chroma_path=":memory:", namespace=f"t{mode}{id(self)}", cache_mode=mode)

    def store(self, memory):
        memory.store_verdict({"query": self.QUESTION, "final_answer": "A warrant is generally required.", "judge_confidence": 0.9, "avg_ccs": 1.0,
                              "judge_verified_citations": [CitationRecord(case_name="Riley v. California", court="", year=2014, holding="warrant required")]},
                             SameVectorEmbedder())

    def test_same_question_is_recognised_whatever_its_case_spacing_and_punctuation(self):
        self.assertEqual(normalize_question("  May POLICE search a cell-phone ,  without a warrant ?? "),
                         normalize_question("may police search a cell phone without a warrant"))
        memory = self.memory("exact")
        self.store(memory)
        hit, answer = memory.check_cache("  MAY POLICE SEARCH A CELL PHONE SEIZED FROM AN ARRESTEE WITHOUT A WARRANT ?", SameVectorEmbedder())
        self.assertTrue(hit)
        self.assertEqual(answer, "A warrant is generally required.")

    def test_a_different_question_is_never_answered_from_the_cache(self):
        memory = self.memory("exact")
        self.store(memory)
        # The embedder gives both questions the same vector: cosine similarity is 1.0.
        self.assertEqual(memory.check_cache(self.OTHER, SameVectorEmbedder()), (False, ""))

    def test_the_old_similarity_rule_does_answer_a_different_question(self):
        memory = self.memory("similarity")
        self.store(memory)
        hit, answer = memory.check_cache(self.OTHER, SameVectorEmbedder())
        self.assertTrue(hit)  # the defect the audit found, kept only to reproduce earlier runs
        self.assertEqual(answer, "A warrant is generally required.")

    def test_cache_off_never_hits(self):
        memory = self.memory("off")
        self.store(memory)
        self.assertEqual(memory.check_cache(self.QUESTION, SameVectorEmbedder()), (False, ""))

    def test_a_hit_gives_back_the_citations_and_the_confidence_not_only_the_text(self):
        memory = self.memory("exact")
        self.store(memory)
        check, _ = make_memory_nodes(memory, SameVectorEmbedder())
        update = check({"query": self.QUESTION})
        self.assertTrue(update["memory_cache_hit"])
        self.assertEqual(update["final_answer"], "A warrant is generally required.")
        self.assertAlmostEqual(update["judge_confidence"], 0.9)
        self.assertEqual([c.case_name for c in update["judge_verified_citations"]], ["Riley v. California"])

    def test_unknown_cache_mode_is_refused(self):
        with self.assertRaises(ValueError):
            AdaptiveMemory(chroma_path=":memory:", namespace="bad", cache_mode="fuzzy")


class GatedCheckerTests(unittest.TestCase):
    def checker(self, gate):
        from src.novel.cce import CitationConfidenceEngine

        tmp = tempfile.mkdtemp(prefix="lexagent_test_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        known = os.path.join(tmp, "known.pkl")
        with open(known, "wb") as handle:
            pickle.dump({r.name for r in REPORTS} | {r.cite for r in REPORTS}, handle)
        collection = OpinionCollection({"riley": ["Introductory facts about the arrest.", "A warrant is required before searching a cell phone."],
                                        "katz": ["A person in a telephone booth has a reasonable expectation of privacy."]})
        return CitationConfidenceEngine(collection, SameVectorEmbedder(), known, nli_model=KeywordNLI(), catalog=CaseCatalog(REPORTS), policy="gated", gate=gate)

    def cite(self, checker, name, holding, cite=""):
        record = CitationRecord(case_name=name, court="", year=0, holding=holding, reporter_cite=cite)
        return checker.compute_ccs(record)

    CLOSED = {"validated": False, "model": "fake", "threshold": 0.70, "ranking": "hybrid", "passages": 3, "source": "test"}
    OPEN = {**CLOSED, "validated": True}

    def test_gate_closed_an_existing_case_is_verified_on_existence_alone(self):
        checker = self.checker(self.CLOSED)
        wrong_holding = self.cite(checker, "Riley v. California", "privacy in a telephone booth")
        self.assertEqual((wrong_holding["existence_verdict"], wrong_holding["support_verdict"]), ("EXISTS", "NOT_SHOWN"))
        self.assertFalse(wrong_holding["support_validated"])
        # The unvalidated support verdict does not reach the tier or the score the system acts on.
        self.assertEqual((wrong_holding["tier"], wrong_holding["ccs"]), ("VERIFIED", 1.0))
        right_holding = self.cite(checker, "Riley v. California", "warrant required for a phone")
        self.assertEqual((right_holding["support_verdict"], right_holding["tier"], right_holding["ccs"]), ("SUPPORTED", "VERIFIED", 1.0))

    def test_gate_open_the_support_verdict_decides_the_tier(self):
        checker = self.checker(self.OPEN)
        right = self.cite(checker, "Riley v. California", "warrant required for a phone")
        wrong = self.cite(checker, "Riley v. California", "privacy in a telephone booth")
        self.assertEqual((right["tier"], right["support_validated"]), ("SUPPORTED", True))
        self.assertEqual(wrong["tier"], "NOT_SHOWN")  # not "misattributed": the model did not fire, nothing more
        self.assertGreater(right["ccs"], 0.9)
        self.assertLess(wrong["ccs"], 0.1)

    def test_the_evidence_is_the_window_that_produced_the_score(self):
        result = self.cite(self.checker(self.CLOSED), "Riley v. California", "warrant required for a phone")
        self.assertIn("warrant is required", result["evidence_window"])
        self.assertEqual(result["evidence"], result["evidence_window"])

    def test_a_case_that_does_not_exist_is_not_found_whatever_the_gate(self):
        for gate in (self.CLOSED, self.OPEN):
            result = self.cite(self.checker(gate), "Zebulon v. Quillfeather Holdings", "anything", cite="573 U.S. 999")
            self.assertEqual((result["existence_verdict"], result["support_verdict"], result["tier"], result["ccs"]),
                             ("NOT_FOUND", "NOT_CHECKED", "FABRICATED", 0.0))

    def test_a_case_outside_the_corpus_is_not_called_fabricated(self):
        result = self.cite(self.checker(self.CLOSED), "Dobbs v. Jackson Women's Health Organization", "anything", cite="597 U.S. 215")
        self.assertEqual((result["existence_verdict"], result["tier"]), ("NOT_IN_CORPUS", "UNVERIFIABLE"))

    def test_existence_can_be_checked_without_a_holding(self):
        checker = self.checker(self.CLOSED)
        found = checker.check_existence("Katz v. United States", "389 U.S. 347")
        self.assertEqual((found["existence_verdict"], found["resolved_doc_id"], found["cite_check"]), ("EXISTS", "katz", "MATCH"))
        self.assertEqual(checker.check_existence("Zebulon v. Quillfeather Holdings", "573 U.S. 999")["existence_verdict"], "NOT_FOUND")
        self.assertEqual(checker.policy, "gated")

    def test_legacy_policy_also_states_existence_and_support_separately(self):
        from src.novel.cce import CitationConfidenceEngine

        tmp = tempfile.mkdtemp(prefix="lexagent_test_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        known = os.path.join(tmp, "known.pkl")
        with open(known, "wb") as handle:
            pickle.dump({r.name for r in REPORTS}, handle)
        checker = CitationConfidenceEngine(OpinionCollection({}), SameVectorEmbedder(), known, nli_model=KeywordNLI(), catalog=CaseCatalog(REPORTS), policy="legacy")
        result = self.cite(checker, "Zebulon v. Quillfeather Holdings", "anything", cite="573 U.S. 999")
        self.assertEqual((result["policy"], result["tier"], result["existence_verdict"]), ("legacy", "FABRICATED", "NOT_FOUND"))


class MetricsTests(unittest.TestCase):
    CATALOG = {"riley v. california": ("EXISTS", "riley"), "katz v. united states": ("EXISTS", "katz"), "terry v. ohio": ("EXISTS", "terry")}

    def resolver(self, name, cite=""):
        verdict, doc_id = self.CATALOG.get(name.lower().strip(), ("NOT_FOUND", None))
        return {"existence_verdict": verdict, "resolved_doc_id": doc_id, "resolved_name": name if doc_id else None,
                "cite_check": ("MATCH" if cite else "NOT_GIVEN")}

    def test_case_names_are_found_in_text(self):
        text = "In Riley v. California (573 U.S. 373) the Court held X. See also United States v. Robinson and Katz v United States. Under Terry v. Ohio, a stop is allowed."
        self.assertEqual(metrics_v2.case_mentions(text), ["Riley v. California", "United States v. Robinson", "Katz v. United States", "Terry v. Ohio"])

    def test_citing_more_cases_does_not_lower_the_target_measure(self):
        target = {"doc_id": "riley", "name": "Riley v. California"}
        one = metrics_v2.score_case({"case_id": 1, "final_answer": "", "answer_citations": [{"case_name": "Riley v. California"}]}, self.resolver, target)
        three = metrics_v2.score_case({"case_id": 1, "final_answer": "", "answer_citations": [
            {"case_name": "Riley v. California"}, {"case_name": "Katz v. United States"}, {"case_name": "Terry v. Ohio"}]}, self.resolver, target)
        self.assertTrue(one["answer"]["target_cited"])
        self.assertTrue(three["answer"]["target_cited"])
        self.assertEqual((one["answer"]["citations"], three["answer"]["citations"]), (1, 3))
        self.assertNotIn("precision", " ".join(three["answer"].keys()))

    def test_existence_is_counted_per_citation_and_duplicates_are_merged(self):
        score = metrics_v2.score_case({"case_id": 2, "final_answer": "", "answer_citations": [
            {"case_name": "Katz v. United States"}, {"case_name": "katz v. united states", "reporter_cite": "389 U.S. 347"},
            {"case_name": "Zebulon v. Quillfeather"}]}, self.resolver, {"doc_id": "riley", "name": "Riley v. California"})
        answer = score["answer"]
        self.assertEqual((answer["citations"], answer["existing"], answer["not_found"], answer["target_cited"]), (2, 1, 1, False))
        self.assertEqual((answer["reporter_cites_given"], answer["reporter_cites_correct"]), (1, 1))

    def test_answer_and_pool_are_scored_separately(self):
        record = {"case_id": 3, "final_answer": "", "answer_citations": [],
                  "pool_citations": [{"case_name": "Riley v. California"}, {"case_name": "Terry v. Ohio"}]}
        score = metrics_v2.score_case(record, self.resolver, {"doc_id": "riley", "name": "Riley v. California"})
        self.assertFalse(score["answer"]["target_cited"])
        self.assertTrue(score["answer"]["no_existing_authority"])
        self.assertTrue(score["pool"]["target_cited"])

    def test_uncited_case_mentions_in_the_answer_are_reported(self):
        record = {"case_id": 4, "final_answer": "Riley v. California controls; Zebulon v. Quillfeather is distinguishable.",
                  "answer_citations": [{"case_name": "Riley v. California"}]}
        mentions = metrics_v2.score_case(record, self.resolver, None)["mentions"]
        self.assertEqual((mentions["case_mentions_in_answer"], mentions["uncited_mentions"], mentions["uncited_mentions_not_found"]), (2, 1, 1))

    def test_skipped_debates_are_their_own_category_and_support_carries_its_validation_status(self):
        scores = [
            metrics_v2.score_case({"case_id": 5, "memory_cache_hit": True, "final_answer": "", "answer_citations": []}, self.resolver, None),
            metrics_v2.score_case({"case_id": 6, "final_answer": "", "answer_citations": [{"case_name": "Katz v. United States"}],
                                   "final_citation_reports": [{"existence_verdict": "EXISTS", "support_verdict": "SUPPORTED", "support_validated": False},
                                                              {"existence_verdict": "EXISTS", "support_verdict": "NOT_SHOWN", "support_validated": False},
                                                              {"existence_verdict": "NOT_FOUND", "support_verdict": "NOT_CHECKED", "support_validated": False}]},
                                  self.resolver, None)]
        total = metrics_v2.aggregate(scores)
        self.assertEqual((total["cases"], total["skipped_debates"]), (2, 1))
        self.assertEqual(total["holding_support"], {"citations_checked": 2, "support_shown_rate": 0.5, "support_model_validated": False})
        self.assertEqual(total["answer"]["answers_without_existing_authority"], 1)


if __name__ == "__main__":
    unittest.main()
