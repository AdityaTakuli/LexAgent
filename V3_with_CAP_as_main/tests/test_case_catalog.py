# LexAgent v3.0 | test_case_catalog.py
"""Tests for resolving a citation to the actual opinion, and for the citation checker's order of work.

No model, GPU or database is used: the index, the embedder and the NLI model are small fakes.

    python -m unittest discover -s tests -v
"""

import logging
import os
import pickle
import shutil
import sys
import tempfile
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("LEXAGENT_DATA_DIR", tempfile.mkdtemp(prefix="lexagent_test_data_"))
logging.disable(logging.INFO)

import numpy as np  # noqa: E402

from src.data.case_catalog import (  # noqa: E402
    CITE_MATCH,
    CITE_MISMATCH,
    CITE_NOT_GIVEN,
    CITE_UNKNOWN_PAGE,
    CITE_VOLUME_NOT_IN_CORPUS,
    CaseCatalog,
    Report,
    names_match,
    names_overlap,
    parse_us_cite,
    split_cases_by_source_text,
)
from src.data.chunking import chunk_document  # noqa: E402
from src.data.corpus_schema import CitationRecord, CorpusDocument  # noqa: E402


def report(doc_id, name, volume, page, year, words):
    return Report(doc_id=doc_id, name=name, volume=str(volume), first_page=page, year=year, chunks=max(1, words // 375), words=words)


# The shape of the real corpus in miniature: each opinion shares its name with one-line order entries.
REPORTS = [
    report("roe", "Roe v. Wade", 410, 113, 1973, 12000),
    report("roe-order-1", "Roe v. Wade", 402, 941, 1971, 40),
    report("roe-order-2", "Roe v. Wade", 408, 919, 1972, 30),
    report("doe", "Doe v. Bolton", 410, 179, 1973, 9000),
    report("carpenter-2018", "Carpenter v. United States", 585, 296, 2018, 36000),
    report("carpenter-1987", "Carpenter v. United States", 484, 19, 1987, 6000),
    report("carpenter-order", "Carpenter v. United States", 358, 818, 1958, 25),
    report("casey", "Planned Parenthood v. Casey", 505, 833, 1992, 54000),
    report("casey-order", "Planned Parenthood of Southeastern Pennsylvania v. Casey", 502, 1056, 1992, 141),
    report("buckley-order-1", "Buckley v. Valeo", 423, 1040, 1975, 216),
    report("buckley-order-2", "Buckley v. Valeo", 424, 936, 1976, 106),
    report("scott", "Scott v. Sandford", 60, 393, 1857, 90000),
    report("chadha", "Immigration & Naturalization Service v. Chadha", 462, 919, 1983, 32000),
    report("smith", "Employment Division v. Smith", 494, 872, 1990, 16000),
    report("tlo", "New Jersey v. T. L. O.", 469, 325, 1985, 20000),
    report("other-at-995", "Jones v. Illinois", 410, 995, 1973, 20),
    report("postal", "Smith v. United States Postal Service", 500, 100, 1991, 8000),
    report("smith-us", "Smith v. United States", 508, 223, 1993, 5000),
]


class NameMatchingTests(unittest.TestCase):
    def test_same_case_written_differently(self):
        for a, b in (
            ("Planned Parenthood of Southeastern Pennsylvania v. Casey", "Planned Parenthood of Southeastern Pa. v. Casey"),
            ("Planned Parenthood of Southeastern Pennsylvania v. Casey", "Planned Parenthood v. Casey"),
            ("INS v. Chadha", "Immigration & Naturalization Service v. Chadha"),
            ("New Jersey v. T.L.O.", "New Jersey v. T. L. O."),
            ("Chevron U.S.A. Inc. v. Natural Resources Defense Council, Inc.", "Chevron U. S. A. Inc. v. Natural Res. Def. Council, Inc."),
            ("Bowen v. American Hospital Assn.", "Bowen v. American Hospital Ass'n"),
            ("Employment Division, Department of Human Resources of Oregon v. Smith", "Employment Division v. Smith"),
            ("Shelby County v. Holder", "Shelby Cnty. v. Holder"),
            ("Hall v. Florida", "Hall v. Fla."),
            ("Miranda v. State of Arizona", "Miranda v. Arizona"),
            ("Brown v. Board of Education", "Brown v. Board of Education of Topeka"),
            ("Cruzan v. Director, Missouri Department of Health", "Cruzan ex rel. Cruzan v. Director, Missouri Department of Health"),
        ):
            self.assertTrue(names_match(a, b), f"{a!r} vs {b!r}")
            self.assertTrue(names_match(b, a), f"{b!r} vs {a!r}")

    def test_different_cases_are_not_merged(self):
        for a, b in (
            ("Roe v. Wade", "Doe v. Wade"),
            ("Roe v. Wade", "Roe v. Bolton"),
            ("Miller v. California", "Miller v. California Pacific Railroad Co."),
            ("Consolidated Steel Logistics v. Apex Transport", "Jones v. Illinois"),
            ("United States v. Lopez", "United States v. Jacobsen"),
            ("", "Roe v. Wade"),
        ):
            self.assertFalse(names_match(a, b), f"{a!r} vs {b!r}")

    def test_overlap_is_looser_but_still_needs_both_parties(self):
        self.assertFalse(names_match("Dred Scott v. Sandford", "Scott v. Sandford"))
        self.assertTrue(names_overlap("Dred Scott v. Sandford", "Scott v. Sandford"))
        self.assertTrue(names_overlap("Barron v. Baltimore", "Barron v. Mayor of Baltimore"))
        self.assertFalse(names_overlap("Smith v. Jones", "Smith v. Brown"))
        self.assertFalse(names_overlap("Apex Transport v. Illinois", "Jones v. Illinois"))

    def test_reporter_cite_parsing(self):
        self.assertEqual(parse_us_cite("410 U.S. 113"), ("410", 113))
        self.assertEqual(parse_us_cite("see 410 U. S. 113, 153 (1973)"), ("410", 113))
        self.assertEqual(parse_us_cite("573 US 373"), ("573", 373))
        self.assertIsNone(parse_us_cite("134 S. Ct. 2473"))
        self.assertIsNone(parse_us_cite(""))


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = CaseCatalog(REPORTS)

    def test_name_alone_finds_the_opinion_not_an_order_entry(self):
        resolution = self.catalog.resolve("Roe v. Wade")
        self.assertEqual((resolution.report.doc_id, resolution.basis, resolution.cite_status), ("roe", "name", CITE_NOT_GIVEN))
        self.assertEqual(resolution.same_name_reports, 3)

    def test_reporter_cite_identifies_the_report(self):
        resolution = self.catalog.resolve("Roe v. Wade", "410 U.S. 113")
        self.assertEqual((resolution.report.doc_id, resolution.basis, resolution.cite_status), ("roe", "cite", CITE_MATCH))
        # The conventional name differs from the reported one; the cite settles it.
        resolution = self.catalog.resolve("Dred Scott v. Sandford", "60 U.S. 393")
        self.assertEqual((resolution.report.doc_id, resolution.cite_status), ("scott", CITE_MATCH))

    def test_cite_that_belongs_to_another_case_is_a_mismatch(self):
        resolution = self.catalog.resolve("Roe v. Wade", "585 U.S. 296")
        self.assertEqual((resolution.report.doc_id, resolution.basis, resolution.cite_status), ("roe", "name", CITE_MISMATCH))
        invented = self.catalog.resolve("Consolidated Steel Logistics v. Apex Transport", "410 U.S. 995")
        self.assertIsNone(invented.report)
        self.assertEqual(invented.cite_status, CITE_MISMATCH)

    def test_pin_cite_and_missing_volume(self):
        pin = self.catalog.resolve("Roe v. Wade", "410 U.S. 153")
        self.assertEqual((pin.report.doc_id, pin.cite_status), ("roe", CITE_UNKNOWN_PAGE))
        missing = self.catalog.resolve("Dobbs v. Jackson Women's Health Organization", "597 U.S. 215")
        self.assertIsNone(missing.report)
        self.assertEqual(missing.cite_status, CITE_VOLUME_NOT_IN_CORPUS)

    def test_same_name_different_cases(self):
        self.assertEqual(self.catalog.resolve("Carpenter v. United States").report.doc_id, "carpenter-2018")
        self.assertEqual(self.catalog.resolve("Carpenter v. United States", year=1987).report.doc_id, "carpenter-1987")
        self.assertEqual(self.catalog.resolve("Carpenter v. United States", "484 U.S. 19").report.doc_id, "carpenter-1987")
        # A year that matches only an order entry does not pull the citation away from the opinion.
        self.assertEqual(self.catalog.resolve("Carpenter v. United States", year=1958).report.doc_id, "carpenter-2018")

    def test_opinion_reported_under_another_spelling(self):
        self.assertEqual(self.catalog.resolve("Planned Parenthood of Southeastern Pennsylvania v. Casey").report.doc_id, "casey")
        self.assertEqual(self.catalog.resolve("INS v. Chadha").report.doc_id, "chadha")
        self.assertEqual(self.catalog.resolve("New Jersey v. T.L.O.").report.doc_id, "tlo")
        self.assertEqual(self.catalog.resolve("Employment Division, Department of Human Resources of Oregon v. Smith").report.doc_id, "smith")

    def test_exact_name_wins_over_a_longer_name_that_merely_starts_the_same(self):
        # A party name cut to its leading words is accepted as a spelling ('Planned Parenthood' for
        # 'Planned Parenthood of Southeastern Pennsylvania'), so the two names below do "match".
        # That must never pull a citation away from the opinion that carries the exact name.
        self.assertTrue(names_match("Smith v. United States", "Smith v. United States Postal Service"))
        self.assertEqual(self.catalog.resolve("Smith v. United States").report.doc_id, "smith-us")
        self.assertEqual(self.catalog.resolve("Smith v. United States Postal Service").report.doc_id, "postal")

    def test_case_with_only_order_entries_has_no_opinion(self):
        resolution = self.catalog.resolve("Buckley v. Valeo", "424 U.S. 1")
        self.assertFalse(resolution.report.is_opinion)
        self.assertEqual(resolution.cite_status, CITE_UNKNOWN_PAGE)
        self.assertIsNone(self.catalog.resolve("Marbury v. Madison").report)

    def test_test_set_keeps_only_cases_the_corpus_can_support(self):
        cases = [
            {"id": 1, "target_case": "Roe v. Wade"},
            {"id": 2, "target_case": "INS v. Chadha", "reporter_citation": "462 U.S. 919"},
            {"id": 3, "target_case": "Buckley v. Valeo", "reporter_citation": "424 U.S. 1"},
            {"id": 4, "target_case": "Dobbs v. Jackson Women's Health Organization", "reporter_citation": "597 U.S. 215"},
            {"id": 5, "target_case": "Planned Parenthood of Southeastern Pennsylvania v. Casey", "reporter_citation": "505 U.S. 833"},
        ]
        evaluable, excluded = split_cases_by_source_text(cases, self.catalog)
        self.assertEqual([case["id"] for case in evaluable], [1, 2, 5])
        self.assertEqual([case["id"] for case in excluded], [3, 4])
        self.assertIn("words", excluded[0]["excluded_reason"])
        self.assertIn("not in the corpus", excluded[1]["excluded_reason"])

    def test_built_from_chunks_and_saved(self):
        opinion = CorpusDocument(doc_id="CAP::1::abc", text=" ".join(f"w{i}" for i in range(1000)), case_name="Roe v. Wade", year=1973, volume="410", reporter="U.S.", first_page=113)
        order = CorpusDocument(doc_id="CAP::2::def", text="Certiorari denied. " * 10, case_name="Roe v. Wade", year=1971, volume="402", reporter="U.S.", first_page=941)
        catalog = CaseCatalog.from_chunks(chunk_document(opinion) + chunk_document(order))
        self.assertEqual(len(catalog), 2)
        found = catalog.resolve("Roe v. Wade").report
        self.assertEqual((found.doc_id, found.volume, found.first_page, found.year, found.chunks, found.words), ("CAP::1::abc", "410", 113, 1973, 3, 1000))

        tmp = tempfile.mkdtemp(prefix="lexagent_test_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "catalog.json")
        catalog.save(path)
        reloaded = CaseCatalog.load(path)
        self.assertEqual(sorted(reloaded.reports, key=lambda r: r.doc_id), sorted(catalog.reports, key=lambda r: r.doc_id))
        self.assertEqual(reloaded.volumes, {"410", "402"})
        self.assertIsNone(CaseCatalog.load(os.path.join(tmp, "missing.json")))


class FakeEmbedder:
    def encode(self, text, **kwargs):
        return np.zeros(4, dtype=np.float32)


class FakeCollection:
    """Returns the stored text of whichever document the query is filtered to, and records the filters."""

    def __init__(self, texts):
        self.texts = texts
        self.filters = []

    def query(self, query_embeddings=None, n_results=3, where=None, **kwargs):
        self.filters.append(where)
        doc_id = (where or {}).get("parent_doc_id", {}).get("$eq")
        text = self.texts.get(doc_id)
        return {"documents": [[text] if text else []], "metadatas": [[{}] if text else []]}


class FakeNLI:
    """Entailment when the passage contains the holding's key word; scores are [contradiction, entailment, neutral]."""

    def predict(self, pairs):
        return np.array([[0.0, 6.0, 0.0] if hypothesis.split()[0].lower() in passage.lower() else [6.0, -6.0, 0.0] for passage, hypothesis in pairs])


class CitationCheckOrderTests(unittest.TestCase):
    def setUp(self):
        from src.novel.cce import CitationConfidenceEngine

        tmp = tempfile.mkdtemp(prefix="lexagent_test_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        known_path = os.path.join(tmp, "known.pkl")
        with open(known_path, "wb") as handle:
            pickle.dump({r.name for r in REPORTS} | {r.cite for r in REPORTS}, handle)
        self.collection = FakeCollection({
            "roe": "privacy right under the Fourteenth Amendment protects the decision.",
            "roe-order-1": "Certiorari denied.",
            "doe": "abortion statute procedural requirements are invalid.",
            "casey": "undue burden standard governs.",
        })
        self.cce = CitationConfidenceEngine(self.collection, FakeEmbedder(), known_path, nli_model=FakeNLI(), policy="legacy", catalog=CaseCatalog(REPORTS))

    def check(self, name, holding, cite="", year=0):
        citation = CitationRecord(case_name=name, court="Supreme Court", year=year, holding=holding, reporter_cite=cite)
        return self.cce.compute_ccs(citation)

    def test_holding_is_checked_against_the_opinion_itself(self):
        result = self.check("Roe v. Wade", "privacy is protected")
        self.assertEqual((result["tier"], result["existence"], result["evidence_available"]), ("SUPPORTED", 1.0, True))
        self.assertEqual(self.collection.filters, [{"parent_doc_id": {"$eq": "roe"}}])  # the opinion, never an order entry
        self.assertEqual(result["evidence_source"], "Roe v. Wade, 410 U.S. 113")
        for key in ("ccs", "tier", "existence", "support", "evidence_available", "exact_score", "fuzzy_score", "semantic_score", "best_match", "evidence", "citation"):
            self.assertIn(key, result)

    def test_real_case_with_someone_elses_holding_is_misattributed(self):
        result = self.check("Roe v. Wade", "undue burden governs")
        self.assertEqual((result["tier"], result["existence"]), ("MISATTRIBUTED", 1.0))

    def test_invented_case_never_reaches_the_holding_check(self):
        result = self.check("Consolidated Steel Logistics v. Apex Transport", "privacy is protected", cite="410 U.S. 995")
        self.assertEqual((result["tier"], result["support"], result["evidence_available"]), ("FABRICATED", None, False))
        self.assertEqual(self.collection.filters, [])
        self.assertEqual(result["cite_check"], CITE_MISMATCH)
        no_cite = self.check("Nexus Robotics v. Pacific Coast Harbor Commission", "privacy is protected")
        self.assertEqual(no_cite["tier"], "UNVERIFIABLE")
        self.assertEqual(self.collection.filters, [])

    def test_case_outside_the_corpus_is_unverifiable_not_fabricated(self):
        result = self.check("Dobbs v. Jackson Women's Health Organization", "privacy is protected", cite="597 U.S. 215")
        self.assertEqual((result["tier"], result["cite_check"]), ("UNVERIFIABLE", CITE_VOLUME_NOT_IN_CORPUS))

    def test_shared_words_alone_do_not_make_a_case_exist(self):
        # Both names share every word of a real case's name, which word-overlap matching scores as 100.
        from src.novel.cce import CitationConfidenceEngine
        tmp = tempfile.mkdtemp(prefix="lexagent_test_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        known_path = os.path.join(tmp, "known.pkl")
        reports = REPORTS + [report("jackson", "Jackson v. Jackson", 300, 50, 1937, 4000), report("halas", "Halas v. Department of Energy", 450, 10, 1981, 3000)]
        with open(known_path, "wb") as handle:
            pickle.dump({r.name for r in reports}, handle)
        collection = FakeCollection({"jackson": "privacy is protected here.", "halas": "privacy is protected here."})
        cce = CitationConfidenceEngine(collection, FakeEmbedder(), known_path, nli_model=FakeNLI(), policy="legacy", catalog=CaseCatalog(reports))

        for name in ("Dobbs v. Jackson Women's Health Organization", "Dynamix Energy Systems v. Department of Energy Oversight"):
            result = cce.compute_ccs(CitationRecord(case_name=name, court="", year=0, holding="privacy is protected"))
            self.assertEqual((result["tier"], result["evidence_available"]), ("UNVERIFIABLE", False), name)
            self.assertLess(result["existence"], 0.85, name)
        self.assertEqual(collection.filters, [])
        # A different spelling of a real case, and a typo, still exist.
        for name in ("Halas v. Dept. of Energy", "Jackson v. Jacksom"):
            result = cce.compute_ccs(CitationRecord(case_name=name, court="", year=0, holding="privacy is protected"))
            self.assertEqual(result["tier"], "SUPPORTED", name)

    def test_case_with_only_order_entries_is_unverifiable(self):
        result = self.check("Buckley v. Valeo", "privacy is protected", cite="424 U.S. 1")
        self.assertEqual((result["tier"], result["existence"], result["evidence_available"]), ("UNVERIFIABLE", 1.0, False))
        self.assertEqual(self.collection.filters, [])

    def test_reporter_cite_is_verified_before_the_holding(self):
        matched = self.check("Roe v. Wade", "privacy is protected", cite="410 U.S. 113")
        self.assertEqual((matched["tier"], matched["cite_check"]), ("SUPPORTED", CITE_MATCH))
        # Right case and holding, but the reporter cite given is Doe v. Bolton's.
        wrong_cite = self.check("Roe v. Wade", "privacy is protected", cite="410 U.S. 179")
        self.assertEqual((wrong_cite["tier"], wrong_cite["cite_check"]), ("UNCERTAIN", CITE_MISMATCH))
        # A different spelling of the name is the same citation.
        variant = self.check("Planned Parenthood of Southeastern Pennsylvania v. Casey", "undue burden governs", cite="505 U.S. 833")
        self.assertEqual((variant["tier"], variant["cite_check"], variant["best_match"]), ("SUPPORTED", CITE_MATCH, "Planned Parenthood v. Casey"))


if __name__ == "__main__":
    unittest.main()
