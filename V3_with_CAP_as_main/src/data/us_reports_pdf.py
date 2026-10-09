# LexAgent v3.0 | us_reports_pdf.py
"""Supplement source for U.S. Reports volumes that CAP does not publish.

CAP's static archive ends at volume 572. The Supreme Court publishes the official bound
volumes of later U.S. Reports as PDFs (https://www.supremecourt.gov/opinions/boundvolumes/).
This module converts one such PDF into the same raw volume cache the download stage writes
(cap_raw_us_<N>.json, explicit status, validated), so the chunk and index stages treat it
like any other volume. Every case it writes is labelled with its real provenance.

What is imported: the reported opinions (syllabus, opinions, concurrences, dissents) and
opinions in chambers. The orders section of each volume is not imported.

Text notes:
  - These PDFs map the "fi", "fl", "ffi" ligature glyphs to a bare "f", so plain text
    extraction yields "offcer" and "traffc". The ligatures are restored exactly, glyph by
    glyph, from the embedded fonts.
  - Footnotes are moved to the end of each case, and words broken across lines are rejoined.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests

from config import (
    CAP_CHECKPOINT_INTERVAL,
    CAP_DOWNLOAD_CHECKPOINT_PATH,
    CAP_RAW_DIR,
    US_REPORTS_INDEX_URL,
    US_REPORTS_PDF_DIR,
    US_REPORTS_PDF_URL,
)
from src.data.cap_checkpoint import COMPLETE, FAILED, UNAVAILABLE, StageCheckpoint, atomic_replace, atomic_write_json, utc_now
from src.data.cap_ingest import (
    RAW_COMPLETE,
    RAW_FAILED,
    RAW_SCHEMA_VERSION,
    RAW_UNAVAILABLE,
    CapNotFound,
    _LiveProgress,
    _raw_header,
    _run_stage,
    inspect_raw_volume,
    raw_volume_path,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

SOURCE_LABEL = "supremecourt.gov official U.S. Reports PDF"
AUTHORITY_IDENTITY = "US_REPORTS_OFFICIAL_PDF"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (LexAgent research)"}

# Page geometry of the bound volumes (points). The running head sits above the body text.
HEADER_ZONE_Y = 158.0
WATERMARK_MIN_SIZE = 15.0
FOOTNOTE_SIZE_GAP = 1.5

LIGATURES = ((0xFB03, "ffi"), (0xFB04, "ffl"), (0xFB00, "ff"), (0xFB01, "fi"), (0xFB02, "fl"))
CITE_RE = re.compile(r"Cite as: (\d+) U\. ?S\. (\d+)(?: \((\d{4})\))?")
TERM_RE = re.compile(r"OCTOBER TERM, \d{4}")
DATE_RE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}")
DECIDED_RE = re.compile(r"Decided\s+((?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4})")
HEAD_THEN_LABEL_RE = re.compile(r"([A-Z0-9 ,.&'’\-]+?)\s+((?:Opinion|Syllabus|Per Curiam|Appendix|Statement|Decree|Counsel|[A-Z][a-z]+, (?:C\. )?J\.).*)")
DOCKET_RE = re.compile(r"Nos?\. ?\d+[–\-]\d+|No\. ?\d+, Orig\.|No\. ?\d+A\d+")
NON_CASE_HEADS = ("ORDERS", "INDEX", "TABLE OF CASES", "CASES ADJUDGED", "REPORTER", "NOTES", "RULES", "AMENDMENT", "STATISTICS")
# Forms left behind when a ligature was not restored; used only to check the result.
LIGATURE_SUSPECTS = re.compile(r"\b(offcer|offcers|offce|ofcial|ofcer|traffc|frst|fnd|fnding|specifc|suffcient|confict|identifed|affrm|affrmed|fnal|fled a|benefts|signifcant|diffcult|fve)\b", re.IGNORECASE)
LIGATURE_OVERFIX = re.compile(r"fii|fll[a-z]|ffii|ffll", re.IGNORECASE)
LIGATURE_OVERFIX_LIMIT = 25  # a few real names ("Fiil-Flynn") look like this; a broken restore gives thousands
MAX_HEADLESS_PAGES = 6       # pages without a running head (figures, appendices) tolerated inside a case

SMALL_WORDS = {"v.", "et", "al.", "ux.", "vir", "of", "the", "and", "for", "in", "on", "ex", "rel.", "to", "by", "de", "la", "aka", "dba", "fka", "nka"}
KEEP_UPPER = {"LLC", "LLP", "LP", "L.P.", "USA", "U.S.", "II", "III", "IV", "N.A.", "PLC", "FCC", "EPA", "NLRB", "SEC", "FBI", "IRS", "CSX", "BNSF", "AT&T", "RJR", "OBB", "T-MOBILE"}


def bound_volume_url(volume: str) -> str:
    return US_REPORTS_PDF_URL.format(volume=volume)


def bound_volume_pdf_path(volume: str, pdf_dir: str = US_REPORTS_PDF_DIR) -> str:
    return os.path.join(pdf_dir, f"{volume}BV.pdf")


def title_case_caption(name: str) -> str:
    """'McCULLEN et al. v. COAKLEY' -> 'McCullen et al. v. Coakley'. Cosmetic: matching is case-insensitive."""
    words = []
    for index, word in enumerate(name.split()):
        bare = word.strip(",;:()")
        if bare.upper() in KEEP_UPPER:
            words.append(word.upper())
        elif word.lower() in SMALL_WORDS and index > 0:
            words.append(word.lower())
        elif bare.isalpha() and bare.isupper() and len(bare) <= 4 and not re.search(r"[AEIOUY]", bare):
            words.append(word)  # vowel-less short tokens are acronyms: CTS, NML, CLS
        elif word.isupper() or re.match(r"(Mc|Mac|De|Di|La|Le)[A-Z]{2,}", word):
            parts = re.split(r"([-'’])", word)
            fixed = []
            for position, part in enumerate(parts):
                prefix = re.match(r"(Mc|Mac|De|Di|La|Le)(?=[A-Z]{2,})", part)
                # INT'L -> Int'l and COMM'N -> Comm'n, but O'BRIEN -> O'Brien.
                contraction = position >= 2 and parts[position - 1] in "'’" and len(parts[position - 2]) > 1
                if prefix:
                    fixed.append(prefix.group(1) + part[prefix.end():].capitalize())
                elif contraction:
                    fixed.append(part.lower())
                else:
                    fixed.append(part.capitalize() if part[:1].isalpha() else part)
            words.append("".join(fixed))
        else:
            words.append(word)
    return " ".join(words)


def short_case_name(running_head: str) -> str:
    """Running head ('RILEY v. CALIFORNIA') to the usual short citation name ('Riley v. California')."""
    name = re.sub(r"\s+", " ", running_head).strip()
    name = re.sub(r",? et (al|ux|vir)\.?", "", name, flags=re.IGNORECASE)
    return title_case_caption(name).strip(" ,")


def fetch_official_pdf_index(timeout: int = 60) -> Dict[str, List[str]]:
    """Read the Court's U.S. Reports page and return, per volume, the official PDFs it links to.

    A volume has either one bound-volume PDF or, while it is still in production, one
    preliminary-print PDF per part (returned in part order).
    """
    response = requests.get(US_REPORTS_INDEX_URL, timeout=timeout, headers=HTTP_HEADERS)
    response.raise_for_status()
    bound: Dict[str, str] = {}
    preliminary: Dict[str, List[Tuple[int, str]]] = {}
    for href in re.findall(r"""href=['"]([^'"]+\.pdf)['"]""", response.text, flags=re.IGNORECASE):
        url = urljoin(US_REPORTS_INDEX_URL, href)
        bound_match = re.search(r"/boundvolumes/(\d+)BV\.pdf$", url, flags=re.IGNORECASE)
        part_match = re.search(r"/preliminaryprint/(\d+)US(\d+)PP[^/]*\.pdf$", url, flags=re.IGNORECASE)
        if bound_match:
            bound[bound_match.group(1)] = url
        elif part_match:
            preliminary.setdefault(part_match.group(1), []).append((int(part_match.group(2)), url))
    index = {volume: [url for _, url in sorted(set(parts))] for volume, parts in preliminary.items()}
    index.update({volume: [url] for volume, url in bound.items()})
    return index


def download_bound_volume_pdf(volume: str, pdf_dir: str = US_REPORTS_PDF_DIR, timeout: int = 180) -> str:
    """Fetch the official bound-volume PDF once. Raises CapNotFound if the Court does not publish it."""
    return download_official_pdf(bound_volume_url(volume), pdf_dir, timeout)


def download_official_pdf(url: str, pdf_dir: str = US_REPORTS_PDF_DIR, timeout: int = 180) -> str:
    """Fetch one official PDF into the PDF cache once. Raises CapNotFound on HTTP 404."""
    os.makedirs(pdf_dir, exist_ok=True)
    path = os.path.join(pdf_dir, os.path.basename(urlparse(url).path))
    if os.path.exists(path):
        with open(path, "rb") as handle:
            if handle.read(4) == b"%PDF":
                return path
    response = requests.get(url, timeout=timeout, headers=HTTP_HEADERS)
    if response.status_code == 404:
        raise CapNotFound(url)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError("the server did not return a PDF")
    with open(path + ".tmp", "wb") as handle:
        handle.write(response.content)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(path + ".tmp", path)
    return path


def _ligature_tables(doc, page, cache: Dict[int, Dict[int, str]]) -> Dict[str, Dict[int, str]]:
    """Per font on the page: glyph id -> the ligature that glyph really is."""
    import pymupdf

    tables: Dict[str, Dict[int, str]] = {}
    for font in page.get_fonts():
        xref, basefont = font[0], font[3]
        if xref not in cache:
            table: Dict[int, str] = {}
            try:
                buffer = doc.extract_font(xref)[3]
                if buffer:
                    embedded = pymupdf.Font(fontbuffer=buffer)
                    for codepoint, letters in LIGATURES:
                        gid = embedded.has_glyph(codepoint)
                        if gid:
                            table[gid] = letters
            except Exception:
                pass
            cache[xref] = table
        if cache[xref]:
            tables[basefont.split("+")[-1]] = cache[xref]
    return tables


def _page_lines(doc, page, font_cache: Dict[int, Dict[int, str]]) -> Tuple[List[dict], int]:
    """Text lines of a page as {y, x, size, text}, with ligature glyphs restored. Also returns the number restored."""
    tables = _ligature_tables(doc, page, font_cache)
    ligature_at: Dict[Tuple[float, float], str] = {}
    if tables:
        for span in page.get_texttrace():
            table = tables.get(str(span.get("font", "")).split("+")[-1])
            if not table:
                continue
            for ucs, gid, origin, _bbox in span["chars"]:
                # Only glyphs the PDF reports as a bare "f" are wrong; correctly mapped ones are left alone.
                if gid in table and ucs == 0x66:
                    ligature_at[(round(origin[0], 1), round(origin[1], 1))] = table[gid]

    lines: List[dict] = []
    restored = 0
    for block in page.get_text("rawdict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            pieces: List[str] = []
            size = 0.0
            for span in line["spans"]:
                size = max(size, span["size"])
                for char in span["chars"]:
                    letters = ligature_at.get((round(char["origin"][0], 1), round(char["origin"][1], 1)))
                    if letters and char["c"] == "f":
                        pieces.append(letters)
                        restored += 1
                    else:
                        pieces.append(char["c"])
            text = "".join(pieces).strip()
            if text:
                lines.append({"y": line["bbox"][1], "x": line["bbox"][0], "size": size, "text": text})
    lines.sort(key=lambda item: (round(item["y"]), item["x"]))
    return lines, restored


def _read_page(doc, index: int, font_cache: Dict[int, Dict[int, str]]) -> dict:
    """Split one page into its running head, body lines and footnote lines, and classify it."""
    page = doc[index]
    lines, restored = _page_lines(doc, page, font_cache)
    lines = [line for line in lines if line["size"] < WATERMARK_MIN_SIZE or "CASES ADJUDGED" in line["text"] or line["y"] < 300 and "SUPREME COURT" in line["text"]]
    info: Dict[str, Any] = {"index": index, "kind": "other", "printed": None, "head": "", "label": "", "body": [], "footnotes": [], "restored": restored, "cite": None}

    all_text = [line["text"] for line in lines]
    if "CASES ADJUDGED" in all_text[:3]:
        # First page of the volume's first case: a title block, then the caption.
        term_at = next((i for i, text in enumerate(all_text) if TERM_RE.fullmatch(text)), None)
        after = all_text[term_at + 1:] if term_at is not None else []
        # The volume's own title page has the same heading but no case caption under it.
        if any(re.search(r"\bv\.|^IN RE\b|^EX PARTE\b", text) for text in after[:4]):
            info.update(kind="start", printed=1, label="Syllabus", body=after)
            return info

    header = [line for line in lines if line["y"] < HEADER_ZONE_Y and line["size"] < 10.5]
    content = [line for line in lines if not (line["y"] < HEADER_ZONE_Y and line["size"] < 10.5)]
    if header:
        top = min(line["y"] for line in header)
        row1 = [line["text"] for line in header if line["y"] - top < 6]
        numbers = [text for text in row1 if text.isdigit()]
        info["printed"] = int(numbers[0]) if numbers else None
        info["head"] = " ".join(text for text in row1 if not text.isdigit()).strip()
        # Below the first row: the section label and, under a long case name, the rest of the running head.
        below = [line["text"] for line in header if line["y"] - top >= 6]
        if _is_case_running_head(info["head"]):
            continuation: List[str] = []
            label: List[str] = []
            for text in below:
                mixed = HEAD_THEN_LABEL_RE.fullmatch(text)
                if _is_case_running_head(text):
                    continuation.append(text)
                elif mixed:
                    continuation.append(mixed.group(1).strip())
                    label.append(mixed.group(2).strip())
                else:
                    label.append(text)
            info["label"] = " ".join(label).strip()
            info["head"] = " ".join([info["head"]] + continuation).strip()
        else:
            info["label"] = " ".join(below).strip()
        if info["printed"] is None:
            # A long running head is set on the same line as the page number.
            leading = re.fullmatch(r"(\d+)\s+(\S.*)", info["head"])
            trailing = re.fullmatch(r"(\S.*?)\s+(\d+)", info["head"])
            if leading:
                info["printed"], info["head"] = int(leading.group(1)), leading.group(2).strip()
            elif trailing:
                info["printed"], info["head"] = int(trailing.group(2)), trailing.group(1).strip()

    sizes = Counter()
    for line in content:
        sizes[round(line["size"], 1)] += len(line["text"])
    body_size = sizes.most_common(1)[0][0] if sizes else 0.0
    for line in content:
        (info["footnotes"] if line["size"] <= body_size - FOOTNOTE_SIZE_GAP else info["body"]).append(line["text"])

    head, label = info["head"], info["label"]
    cite = CITE_RE.fullmatch(head)
    if cite:
        info.update(kind="cite", cite=(cite.group(1), int(cite.group(2)), cite.group(3)))
    elif TERM_RE.fullmatch(head) and info["printed"] is not None and label and not DATE_RE.search(label) and "U. S." not in label and label != "ORDERS":
        info["kind"] = "start"
    elif info["printed"] is not None and _is_case_running_head(head):
        info["kind"] = "name"
    return info


def _is_case_running_head(head: str) -> bool:
    """True for a case-name running head such as 'RILEY v. CALIFORNIA' or 'McCULLEN v. COAKLEY'."""
    if not head or TERM_RE.fullmatch(head) or any(word in head for word in NON_CASE_HEADS):
        return False
    letters = [c for c in re.sub(r"\b(v|et|al|ux|vir|fka|aka|dba|nka|ex|rel|in|re)\b\.?", "", head) if c.isalpha()]
    return len(letters) >= 2 and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.8


def _join_text(lines: List[str]) -> str:
    """Join lines into running text, rejoining words that the typesetter broke across lines."""
    text = "\n".join(lines)
    text = text.replace("­\n", "").replace("­", "")
    return text.strip()


def _caption(body: List[str]) -> Tuple[str, str]:
    """Caption (the upper-case lines at the top of a case's first page) and the text after it."""
    caption: List[str] = []
    for line in body[:12]:
        letters = [c for c in line if c.isalpha()]
        if not letters:
            break
        upper_share = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_share < 0.6 and not re.fullmatch(r"(et al\.|v\.)", line.strip()):
            break
        caption.append(line.strip())
    return re.sub(r"\s+", " ", " ".join(caption)).strip(), ""


def parse_bound_volume(pdf_path: str, volume: str, source_url: Optional[str] = None) -> Tuple[List[dict], dict]:
    """Split one official U.S. Reports PDF into case records in CAP's record shape. Returns (cases, stats)."""
    import pymupdf

    volume = str(volume)
    source_url = source_url or bound_volume_url(volume)
    doc = pymupdf.open(pdf_path)
    font_cache: Dict[int, Dict[int, str]] = {}
    groups: List[dict] = []
    current: Optional[dict] = None
    problems: List[str] = []
    notes: List[str] = []
    page_proof = False
    restored = 0

    headless: List[dict] = []  # pages without a running head seen since the last page of the current case

    def last_printed() -> int:
        return next((p["printed"] for p in reversed(current["pages"]) if p["printed"] is not None), 0)

    for index in range(doc.page_count):
        if not page_proof and index < 80 and "Page Proof Pending Publication" in doc[index].get_text():
            page_proof = True
        page = _read_page(doc, index, font_cache)
        restored += page["restored"]
        kind = page["kind"]
        if kind == "start":
            current = {"first_page": page["printed"], "pages": [page], "names": Counter(), "year": None}
            groups.append(current)
            headless = []
        elif kind == "cite":
            cite_volume, cite_page, cite_year = page["cite"]
            next_page = current is not None and page["printed"] == last_printed() + len(headless) + 1
            # The cited first page identifies the case. A running head that names some other page is
            # accepted only when it is the very next page of the current case and cannot be the start
            # of a later one (the volumes contain a few misprinted running heads).
            misprint = next_page and not (current["first_page"] < cite_page <= page["printed"])
            if current is not None and (cite_page == current["first_page"] or misprint):
                current["pages"].extend(headless + [page])
                current["year"] = current["year"] or cite_year
                if cite_page != current["first_page"] or cite_volume != volume:
                    notes.append(f"pdf page {index + 1}: running head misprinted as {cite_volume} U.S. {cite_page}")
            else:
                problems.append(f"pdf page {index + 1}: running head cites {cite_volume} U.S. {cite_page} but no case starting there was found")
                current = None
            headless = []
        elif kind == "name" and current is not None and page["printed"] == last_printed() + len(headless) + 1:
            current["pages"].extend(headless + [page])
            current["names"][page["head"]] += 1
            headless = []
        elif current is not None and len(headless) < MAX_HEADLESS_PAGES:
            # A figure, map or appendix page with no usable running head. It is kept only if the
            # case demonstrably continues after it; trailing pages like this are dropped.
            headless.append(page)
        else:
            current = None
            headless = []

    cases: List[dict] = []
    seen_pages = set()
    last_year = ""
    for group in groups:
        first_page = group["first_page"]
        if first_page in seen_pages:
            problems.append(f"two cases start at page {first_page}")
            continue
        seen_pages.add(first_page)
        pages = group["pages"]
        body_lines = [line for page in pages for line in page["body"]]
        footnote_lines = [line for page in pages for line in page["footnotes"]]
        text = _join_text(body_lines)
        if footnote_lines:
            text += "\n\nFootnotes:\n" + _join_text(footnote_lines)

        caption, _ = _caption(pages[0]["body"])
        running = group["names"].most_common(1)[0][0] if group["names"] else caption
        name_abbreviation = short_case_name(running) if running else ""
        first_text = _join_text(pages[0]["body"])
        decided = DECIDED_RE.search(first_text)
        decision_date = ""
        if decided:
            try:
                decision_date = datetime.strptime(decided.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                pass
        if not decision_date:
            # Short per curiam dispositions print no date: fall back to the year in the running head,
            # then to the year of the case before it (a volume runs in date order).
            decision_date = group["year"] or last_year
        last_year = decision_date[:4] or last_year
        docket = DOCKET_RE.search(first_text)

        if not name_abbreviation or len(text) < 300:
            problems.append(f"case at page {first_page}: no name or almost no text")
            continue
        last_page = pages[-1]["printed"] or first_page
        cases.append({
            "id": f"usr-{volume}-{first_page}",
            "name": title_case_caption(caption) if caption else name_abbreviation,
            "name_abbreviation": name_abbreviation,
            "decision_date": decision_date,
            "docket_number": docket.group(0) if docket else "",
            "volume": volume,
            "first_page": str(first_page),
            "last_page": str(last_page),
            "citations": [{"type": "official", "cite": f"{volume} U.S. {first_page}"}],
            "court": {"name_abbreviation": "U.S.", "name": "Supreme Court of the United States"},
            "jurisdiction": {"name_long": "United States", "name": "U.S."},
            "authority_identity": AUTHORITY_IDENTITY,
            "provenance": {"source": "supremecourt.gov", "url": source_url, "pdf_pages": [pages[0]["index"] + 1, pages[-1]["index"] + 1]},
            "casebody": {"head_matter": "", "opinions": [{"type": "full report (syllabus and all opinions)", "author": None, "text": text}]},
        })

    all_text = "\n".join(case["casebody"]["opinions"][0]["text"] for case in cases)
    stats = {
        "pdf_pages": doc.page_count,
        "page_proof": page_proof,
        "cases": len(cases),
        "case_pages": sum(len(group["pages"]) for group in groups),
        "ligatures_restored": restored,
        "ligature_suspects_left": len(LIGATURE_SUSPECTS.findall(all_text)),
        "ligature_overfix": len(LIGATURE_OVERFIX.findall(all_text)),
        "problems": problems,
        "notes": notes,
    }
    doc.close()
    return cases, stats


def import_us_reports_volume(
    volume: str,
    raw_dir: str = CAP_RAW_DIR,
    pdf_dir: str = US_REPORTS_PDF_DIR,
    force: bool = False,
    urls: Optional[List[str]] = None,
) -> dict:
    """Write the raw cache for one volume from the Court's official U.S. Reports PDF(s).

    urls lists the official PDFs for the volume (one bound volume, or one preliminary print per
    part); without it the bound-volume address is tried. A verified COMPLETE cache (from CAP or
    from an earlier import) is left alone unless force=True.

    The result carries an explicit status, exactly like a CAP download:
      - "complete":    every PDF was parsed and every check on the split passed
      - "failed":      a PDF could not be fetched or the split did not pass its checks
      - "unavailable": the Court has not published this volume's opinions as a U.S. Reports PDF
    A volume whose later part has not been printed yet is "complete" with "partial": true.
    """
    volume = str(volume)
    existing = inspect_raw_volume(volume, raw_dir)
    if existing.ok and not force:
        return {**_raw_header(existing.payload), "skipped": True}

    listed = bool(urls)
    urls = list(urls) if urls else [bound_volume_url(volume)]
    result: Dict[str, Any] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "volume": volume,
        "reporter": "us",
        "status": RAW_FAILED,
        "source": SOURCE_LABEL,
        "source_url": urls[0],
        "source_urls": urls,
        "edition": "bound volume" if "/boundvolumes/" in urls[0].lower() else "preliminary print",
        "partial": False,
        "coverage": "Reported opinions and opinions in chambers. The orders section is not included.",
        "metadata_count": 0,
        "cases_count": 0,
        "empty_text_count": 0,
        "empty_text_case_ids": [],
        "failed_count": 0,
        "failed_cases": [],
        "error": None,
        "downloaded_at": utc_now(),
        "cases": [],
    }

    def store() -> dict:
        if existing.ok and result["status"] != RAW_COMPLETE:
            return {**_raw_header(result), "kept_previous_complete": True}
        os.makedirs(raw_dir, exist_ok=True)
        atomic_write_json(raw_volume_path(volume, raw_dir), result)
        return _raw_header(result)

    cases: List[dict] = []
    parts: List[dict] = []
    for url in urls:
        name = os.path.basename(urlparse(url).path)
        try:
            pdf_path = download_official_pdf(url, pdf_dir)
        except CapNotFound:
            if listed:
                result["error"] = f"{name} is listed by the Court but could not be downloaded (HTTP 404)"
            else:
                result.update(status=RAW_UNAVAILABLE, error="not published by CAP, and the Supreme Court has no bound-volume PDF for it (HTTP 404)")
            return store()
        except Exception as e:
            result["error"] = f"download of {name} failed: {type(e).__name__}: {e}"
            return store()
        try:
            with open(pdf_path, "rb") as handle:
                pdf_sha256 = hashlib.sha256(handle.read()).hexdigest()
            part_cases, stats = parse_bound_volume(pdf_path, volume, source_url=url)
        except Exception as e:
            result["error"] = f"{name} could not be parsed: {type(e).__name__}: {e}"
            return store()
        cases.extend(part_cases)
        parts.append({"file": name, "url": url, "pdf_sha256": pdf_sha256, **stats})

    cases.sort(key=lambda case: int(case["first_page"]))
    first_pages = [case["first_page"] for case in cases]
    problems = [problem for part in parts for problem in part["problems"]]
    problems += [f"page {page} starts a case in more than one PDF" for page in sorted(set(p for p in first_pages if first_pages.count(p) > 1), key=int)]
    stats = {
        "cases": len(cases),
        "pdf_pages": sum(part["pdf_pages"] for part in parts),
        "ligatures_restored": sum(part["ligatures_restored"] for part in parts),
        "ligature_suspects_left": sum(part["ligature_suspects_left"] for part in parts),
        "ligature_overfix": sum(part["ligature_overfix"] for part in parts),
        "problems": problems,
        "notes": [note for part in parts for note in part["notes"]],
        "parts": [{key: value for key, value in part.items() if key not in ("problems", "notes")} for part in parts],
    }
    orders_only = [part["file"] for part in parts if part["cases"] == 0 and not part["problems"]]
    result.update(
        cases=cases, cases_count=len(cases), metadata_count=len(cases), parse_stats=stats,
        page_proof=any(part["page_proof"] for part in parts),
        partial=bool(cases) and bool(orders_only),
    )
    if result["partial"]:
        result["coverage"] += f" Incomplete volume: {', '.join(orders_only)} contains orders only, so the opinions of that part are not published yet."

    if problems:
        result["error"] = f"{len(problems)} problem(s) splitting the PDF into cases (first: {problems[0]})"
    elif not cases and len(orders_only) == len(parts):
        result.update(status=RAW_UNAVAILABLE, error="the Court has published only the orders section of this volume so far; its opinions are not available as a U.S. Reports PDF")
    elif not cases:
        result["error"] = "no cases were found in the PDF"
    elif stats["ligature_overfix"] > LIGATURE_OVERFIX_LIMIT:
        result["error"] = f"ligature restoration produced {stats['ligature_overfix']} doubled letters"
    else:
        result["status"] = RAW_COMPLETE
    return store()


def run_us_reports_pdf_stage(
    volumes: List[str],
    raw_dir: str = CAP_RAW_DIR,
    pdf_dir: str = US_REPORTS_PDF_DIR,
    checkpoint_path: str = CAP_DOWNLOAD_CHECKPOINT_PATH,
    checkpoint_interval: int = CAP_CHECKPOINT_INTERVAL,
    selection: str = "custom",
    force: bool = False,
    max_workers: int = 8,
) -> dict:
    """STAGE 1b: fill volumes CAP does not publish from the Supreme Court's official U.S. Reports PDFs.

    Bound volumes are used where they exist, preliminary prints otherwise. Only raw JSON is
    written; nothing is chunked or embedded. Volumes whose raw cache is already verified
    COMPLETE are skipped, so CAP volumes are never replaced. Uses the download ledger.
    """
    volumes = [str(v) for v in volumes]
    os.makedirs(raw_dir, exist_ok=True)
    ckpt = StageCheckpoint(checkpoint_path, "download", volumes, checkpoint_interval, selection)

    def ledger_info(vol: str) -> Tuple[str, dict]:
        check = inspect_raw_volume(vol, raw_dir)
        payload = check.payload or {}
        info = {key: payload.get(key) for key in ("metadata_count", "cases_count", "empty_text_count", "failed_count", "source", "edition", "partial", "page_proof") if key in payload}
        if check.ok:
            info["sha256"] = check.sha256
        else:
            info["error"] = check.reason
        return check.state, info

    pending = [vol for vol in volumes if force or not inspect_raw_volume(vol, raw_dir).ok]
    index: Dict[str, List[str]] = {}
    if pending:
        try:
            index = fetch_official_pdf_index()
        except Exception as e:
            logger.warning("Could not read the Court's U.S. Reports page (%s). Only bound-volume PDFs will be tried.", e)

    # PDFs are parsed in parallel worker processes; the ledger is still updated one volume at a time.
    pool = ProcessPoolExecutor(max_workers=max(1, min(max_workers, len(pending) or 1)))
    futures = {vol: pool.submit(import_us_reports_volume, vol, raw_dir, pdf_dir, force, index.get(vol)) for vol in pending}

    def process(vol: str, live: _LiveProgress) -> Tuple[str, dict]:
        if vol in futures:
            futures[vol].result()
        return ledger_info(vol)

    def describe(state: str, info: dict) -> str:
        if state == COMPLETE:
            notes = [note for flag, note in ((info.get("partial"), "INCOMPLETE VOLUME: later part not printed yet"), (info.get("page_proof"), "page proof")) if flag]
            return f"{info.get('cases_count')} cases from the official {info.get('edition') or 'PDF'}" + (f" ({'; '.join(notes)})" if notes else "")
        return str(info.get("error"))

    try:
        return _run_stage(
            ckpt,
            "CAP STAGE 1b: U.S. REPORTS OFFICIAL PDF SUPPLEMENT",
            [f"Source:             {US_REPORTS_INDEX_URL}", f"PDF cache:          {pdf_dir}", f"Raw cache:          {raw_dir}"],
            ledger_info, process, describe, force=force,
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
