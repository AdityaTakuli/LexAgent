# LexAgent v3.0 | test_us_reports_pdf.py
"""Tests for importing U.S. Reports volumes from the Supreme Court's official bound-volume PDFs.

A small synthetic "bound volume" is generated on the fly, so no network access is needed.
Restoring the ligatures that the real PDFs drop depends on their embedded fonts and cannot be
reproduced with a synthetic file; that part is checked against the real volumes instead.

    python -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("LEXAGENT_DATA_DIR", tempfile.mkdtemp(prefix="lexagent_test_data_"))
logging.disable(logging.INFO)

try:
    import pymupdf
except ImportError:  # the importer is optional; the rest of the pipeline works without it
    pymupdf = None

from src.data import us_reports_pdf  # noqa: E402
from src.data.cap_checkpoint import UNAVAILABLE, atomic_write_json  # noqa: E402
from src.data.cap_ingest import chunk_cap_volume, chunk_volume_path, inspect_raw_volume, raw_volume_path  # noqa: E402
from src.data.us_reports_pdf import (  # noqa: E402
    _is_case_running_head,
    _join_text,
    import_us_reports_volume,
    short_case_name,
    title_case_caption,
)

SENTENCE = "The judgment of the court of appeals is reversed and the case is remanded for further proceedings."


def body_lines(tag, count=14):
    return [f"{tag} line {n}. {SENTENCE}" for n in range(count)]


def add_page(doc, header=(), label=None, body=(), footnotes=(), title=()):
    """One page laid out like a bound volume: running head, section label, body, small-type footnotes."""
    page = doc.new_page(width=612, height=792)
    for x, text in header:
        page.insert_text((x, 128), text, fontsize=9)
    if label:
        page.insert_text((280, 152), label, fontsize=9)
    y = 176
    for text, size in title:
        page.insert_text((200, y), text, fontsize=size)
        y += 22
    for text in body:
        page.insert_text((155, y), text, fontsize=11)
        y += 13
    y += 20
    for text in footnotes:
        page.insert_text((155, y), text, fontsize=8)
        y += 10
    return page


def build_volume(path):
    """Volume 900: two reported cases, a figure page, a misprinted running head, then the orders section."""
    doc = pymupdf.open()
    # Front matter: the volume's own title page has the same heading as the first case, but no caption.
    add_page(doc, title=[("CASES ADJUDGED", 12), ("IN THE", 9), ("SUPREME COURT OF THE UNITED STATES", 14), ("AT", 9),
                         ("OCTOBER TERM, 2020", 11), ("January 5 Through April 2, 2021", 11)])
    # Case 1 starts on page 1, under the "CASES ADJUDGED" heading.
    add_page(doc, title=[("CASES ADJUDGED", 12), ("IN THE", 9), ("SUPREME COURT OF THE UNITED STATES", 14), ("AT", 9), ("OCTOBER TERM, 2020", 11)],
             body=["ALPHA et al. v. BETA", "certiorari to the united states court of appeals", "No. 20-1. Argued January 5, 2021—Decided March 1, 2021"] + body_lines("alpha-syllabus", 6))
    add_page(doc, header=[(155, "2"), (260, "ALPHA v. BETA")], label="Opinion of the Court", body=body_lines("alpha-opinion"))
    add_page(doc, header=[(250, "Cite as: 900 U. S. 1 (2021)"), (444, "3")], label="Opinion of the Court", body=body_lines("alpha-more"),
             footnotes=["1 This footnote belongs at the end of the case, not in the middle of a sentence."])
    # Case 2 starts on page 4.
    add_page(doc, header=[(155, "4"), (250, "OCTOBER TERM, 2020")], label="Syllabus",
             body=["GAMMA INT'L, INC., et al. v. McDELTA", "certiorari to the supreme court of ohio", "No. 20-2. Argued February 1, 2021—Decided April 2, 2021"] + body_lines("gamma-syllabus", 6))
    add_page(doc, body=["Figure 1. A map reproduced in the opinion."])  # no running head at all
    add_page(doc, header=[(155, "6"), (240, "GAMMA INT'L, INC. v. McDELTA")], label="Opinion of the Court", body=body_lines("gamma-opinion"))
    # The running head misprints the first page (2 instead of 4); the page still belongs to case 2.
    add_page(doc, header=[(250, "Cite as: 900 U. S. 2 (2021)"), (444, "7")], label="Thomas, J., dissenting", body=body_lines("gamma-dissent"))
    # Orders section: never imported.
    add_page(doc, header=[(260, "ORDERS"), (444, "901")], label="April 5, 2021", body=body_lines("orders-odd"))
    add_page(doc, header=[(155, "902"), (250, "OCTOBER TERM, 2020")], label="April 5, 2021 900 U. S.", body=body_lines("orders-even"))
    doc.save(path)
    doc.close()


class CaseNameTests(unittest.TestCase):
    def test_short_names(self):
        self.assertEqual(short_case_name("RILEY v. CALIFORNIA"), "Riley v. California")
        self.assertEqual(short_case_name("McCULLEN et al. v. COAKLEY"), "McCullen v. Coakley")
        self.assertEqual(short_case_name("DEPARTMENT OF HOMELAND SECURITY v. MacLEAN"), "Department of Homeland Security v. MacLean")
        self.assertEqual(short_case_name("LEXMARK INT'L, INC. v. STATIC CONTROL COMPONENTS, INC."), "Lexmark Int'l, Inc. v. Static Control Components, Inc.")
        self.assertEqual(short_case_name("O'BRIEN v. UNITED STATES"), "O'Brien v. United States")
        self.assertEqual(short_case_name("CTS CORP. v. WALDBURGER"), "CTS Corp. v. Waldburger")
        self.assertEqual(title_case_caption("POM WONDERFUL LLC v. COCA-COLA CO."), "Pom Wonderful LLC v. Coca-Cola Co.")

    def test_running_heads(self):
        self.assertTrue(_is_case_running_head("RILEY v. CALIFORNIA"))
        self.assertTrue(_is_case_running_head("McCULLEN v. COAKLEY"))
        for other in ("ORDERS", "OCTOBER TERM, 2013", "TABLE OF CASES REPORTED", "Opinion of the Court", "", "INDEX"):
            self.assertFalse(_is_case_running_head(other), other)

    def test_words_broken_across_lines_are_rejoined(self):
        self.assertEqual(_join_text(["a shoot­", "ing that had occurred"]), "a shooting that had occurred")
        self.assertEqual(_join_text(["password-protected", "phone"]), "password-protected\nphone")


@unittest.skipIf(pymupdf is None, "pymupdf is not installed")
class ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lexagent_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.raw_dir = os.path.join(self.tmp, "raw")
        self.chunks_dir = os.path.join(self.tmp, "chunks")
        self.pdf_dir = os.path.join(self.tmp, "pdf")
        os.makedirs(self.pdf_dir)
        build_volume(os.path.join(self.pdf_dir, "900BV.pdf"))
        # The PDF is already in the cache: any network use is a bug.
        patcher = mock.patch.object(us_reports_pdf.requests, "get", side_effect=AssertionError("network used"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def read_raw(self, volume="900"):
        with open(raw_volume_path(volume, self.raw_dir), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_volume_is_split_into_cases(self):
        header = import_us_reports_volume("900", raw_dir=self.raw_dir, pdf_dir=self.pdf_dir)
        self.assertEqual(header["status"], "complete", header.get("error"))
        self.assertEqual((header["metadata_count"], header["cases_count"], header["failed_count"]), (2, 2, 0))
        self.assertEqual(header["parse_stats"]["problems"], [])
        self.assertEqual(len(header["parse_stats"]["notes"]), 1)  # the misprinted running head
        self.assertTrue(inspect_raw_volume("900", self.raw_dir).ok)

        alpha, gamma = self.read_raw()["cases"]
        self.assertEqual((alpha["id"], alpha["name_abbreviation"], alpha["first_page"], alpha["last_page"]), ("usr-900-1", "Alpha v. Beta", "1", "3"))
        self.assertEqual((gamma["id"], gamma["name_abbreviation"], gamma["first_page"], gamma["last_page"]), ("usr-900-4", "Gamma Int'l, Inc. v. McDelta", "4", "7"))
        self.assertEqual((alpha["decision_date"], gamma["decision_date"]), ("2021-03-01", "2021-04-02"))
        self.assertEqual(alpha["citations"], [{"type": "official", "cite": "900 U.S. 1"}])
        self.assertEqual(alpha["authority_identity"], "US_REPORTS_OFFICIAL_PDF")
        self.assertEqual(alpha["provenance"]["pdf_pages"], [2, 4])

        alpha_text = alpha["casebody"]["opinions"][0]["text"]
        gamma_text = gamma["casebody"]["opinions"][0]["text"]
        for expected in ("alpha-syllabus line 0", "alpha-opinion line 13", "alpha-more line 13"):
            self.assertIn(expected, alpha_text)
        for expected in ("gamma-syllabus line 0", "Figure 1.", "gamma-opinion line 0", "gamma-dissent line 13"):
            self.assertIn(expected, gamma_text)
        # Running heads, page numbers and the orders section stay out of the text.
        for text in (alpha_text, gamma_text):
            for unwanted in ("Cite as:", "OCTOBER TERM", "Opinion of the Court", "orders-", "CASES ADJUDGED"):
                self.assertNotIn(unwanted, text)
        self.assertNotIn("gamma", alpha_text)
        # The footnote is moved to the end of its case.
        self.assertTrue(alpha_text.rstrip().endswith("not in the middle of a sentence."))
        self.assertLess(alpha_text.index("alpha-more line 13"), alpha_text.index("Footnotes:"))

    def test_imported_volume_flows_through_the_chunk_stage(self):
        import_us_reports_volume("900", raw_dir=self.raw_dir, pdf_dir=self.pdf_dir)
        chunk_cap_volume("900", raw_dir=self.raw_dir, chunks_dir=self.chunks_dir)
        with open(chunk_volume_path("900", self.chunks_dir), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["documents_count"], 2)
        self.assertIn("900 U.S. 4", payload["known_cases"])
        self.assertIn("Alpha v. Beta", payload["known_cases"])
        chunk = payload["chunks"][0]
        self.assertEqual((chunk["volume"], chunk["reporter"], chunk["first_page"], chunk["year"]), ("900", "U.S.", 1, 2021))
        self.assertEqual(chunk["metadata"]["authority_identity"], "US_REPORTS_OFFICIAL_PDF")
        self.assertEqual(chunk["court"], "Supreme Court of the United States")

    def test_a_verified_volume_is_never_replaced(self):
        atomic_write_json(raw_volume_path("900", self.raw_dir), {
            "schema_version": 2, "volume": "900", "status": "complete", "metadata_count": 1, "cases_count": 1,
            "empty_text_count": 0, "failed_count": 0, "failed_cases": [],
            "cases": [{"id": 1, "casebody": {"opinions": [{"text": "text from CAP " * 50}]}}],
        })
        header = import_us_reports_volume("900", raw_dir=self.raw_dir, pdf_dir=self.pdf_dir)
        self.assertTrue(header["skipped"])
        self.assertEqual(self.read_raw()["cases"][0]["id"], 1)

    def test_volume_without_an_official_pdf_is_unavailable(self):
        response = mock.Mock(status_code=404, content=b"")
        with mock.patch.object(us_reports_pdf.requests, "get", return_value=response):
            header = import_us_reports_volume("599", raw_dir=self.raw_dir, pdf_dir=self.pdf_dir)
        self.assertEqual(header["status"], "unavailable")
        self.assertEqual(inspect_raw_volume("599", self.raw_dir).state, UNAVAILABLE)

    def test_a_pdf_that_does_not_split_cleanly_is_failed_not_complete(self):
        doc = pymupdf.open()
        # A running head cites a case whose first page is nowhere in the file.
        add_page(doc, header=[(250, "Cite as: 901 U. S. 50 (2021)"), (444, "51")], label="Opinion of the Court", body=body_lines("orphan"))
        doc.save(os.path.join(self.pdf_dir, "901BV.pdf"))
        doc.close()
        header = import_us_reports_volume("901", raw_dir=self.raw_dir, pdf_dir=self.pdf_dir)
        self.assertEqual(header["status"], "failed")
        self.assertFalse(inspect_raw_volume("901", self.raw_dir).ok)

    def build_orders_only(self, name):
        doc = pymupdf.open()
        add_page(doc, header=[(260, "ORDERS"), (444, "901")], label="June 1, 2021", body=body_lines("orders-odd"))
        add_page(doc, header=[(155, "902"), (250, "OCTOBER TERM, 2020")], label="June 1, 2021 902 U. S.", body=body_lines("orders-even"))
        doc.save(os.path.join(self.pdf_dir, name))
        doc.close()

    def test_preliminary_print_in_parts_with_a_part_not_printed_yet(self):
        os.replace(os.path.join(self.pdf_dir, "900BV.pdf"), os.path.join(self.pdf_dir, "902US1PP_web.pdf"))
        self.build_orders_only("902US2PP_Ord.pdf")
        urls = ["https://example.test/opinions/preliminaryprint/902US1PP_web.pdf", "https://example.test/opinions/preliminaryprint/902US2PP_Ord.pdf"]
        header = import_us_reports_volume("902", raw_dir=self.raw_dir, pdf_dir=self.pdf_dir, urls=urls)
        self.assertEqual(header["status"], "complete", header.get("error"))
        self.assertEqual((header["edition"], header["cases_count"], header["partial"]), ("preliminary print", 2, True))
        self.assertIn("902US2PP_Ord.pdf contains orders only", header["coverage"])
        self.assertEqual([part["file"] for part in header["parse_stats"]["parts"]], ["902US1PP_web.pdf", "902US2PP_Ord.pdf"])
        self.assertEqual(self.read_raw("902")["cases"][0]["provenance"]["url"], urls[0])

    def test_volume_with_only_its_orders_published_is_unavailable(self):
        self.build_orders_only("903US1PP_Ord.pdf")
        header = import_us_reports_volume("903", raw_dir=self.raw_dir, pdf_dir=self.pdf_dir, urls=["https://example.test/opinions/preliminaryprint/903US1PP_Ord.pdf"])
        self.assertEqual(header["status"], "unavailable")
        self.assertIn("only the orders section", header["error"])
        self.assertEqual(inspect_raw_volume("903", self.raw_dir).state, UNAVAILABLE)

    def test_official_index_lists_bound_volumes_and_preliminary_print_parts(self):
        html = """
            <a href='boundvolumes/587BV.pdf'>587</a> <a href="../publicinfo/guide.pdf">guide</a>
            <a href='preliminaryprint/590US2PP_web.pdf'>590 part 2</a> <a href='preliminaryprint/590US1PP_web.pdf'>590 part 1</a>
            <a href='preliminaryprint/597US1PP_Ord.pdf'>597</a>
        """
        response = mock.Mock(status_code=200, text=html)
        with mock.patch.object(us_reports_pdf.requests, "get", return_value=response):
            index = us_reports_pdf.fetch_official_pdf_index()
        base = "https://www.supremecourt.gov/opinions/"
        self.assertEqual(index, {
            "587": [base + "boundvolumes/587BV.pdf"],
            "590": [base + "preliminaryprint/590US1PP_web.pdf", base + "preliminaryprint/590US2PP_web.pdf"],
            "597": [base + "preliminaryprint/597US1PP_Ord.pdf"],
        })

    def test_cli_flag(self):
        import master_run
        args = master_run.build_parser().parse_args(["--download-us-reports-pdf", "--volumes", "573-587"])
        self.assertTrue(args.download_us_reports_pdf)
        self.assertFalse(args.download_cap or args.chunk_cap or args.build_db)
        with mock.patch.object(master_run, "step_download_us_reports_pdf", return_value={"counts": {"FAILED": 0, "NOT_STARTED": 0, "IN_PROGRESS": 0}}) as step, \
                mock.patch.object(master_run, "step_download_cap") as cap, \
                mock.patch.object(sys, "argv", ["master_run.py", "--download-us-reports-pdf", "--volumes", "573-587"]), \
                contextlib.redirect_stdout(io.StringIO()):
            master_run.main()
        step.assert_called_once_with(volumes_mode="573-587", checkpoint_interval=64, redownload=False)
        self.assertFalse(cap.called)


if __name__ == "__main__":
    unittest.main()
