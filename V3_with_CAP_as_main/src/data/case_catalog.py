# LexAgent v3.0 | case_catalog.py
"""Catalog of the reports in the corpus, and resolution of a citation to the actual opinion.

A U.S. Reports volume holds full opinions and, in its orders section, thousands of short
entries (certiorari denied, rehearing denied) that carry the same case name as the opinion
they relate to. "Roe v. Wade" is the name of the opinion at 410 U.S. 113 and of several
one-line orders in other volumes. Anything that looks a case up by name alone can therefore
land on an order entry instead of the opinion.

The catalog lists every report (one per source document) with its volume, first page, year
and length, and resolve() picks the report a citation actually refers to:
  1. by reporter citation (volume and first page), when it is given and the name agrees;
  2. otherwise by name, taking the substantial report rather than an order entry.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.data.cap_checkpoint import atomic_write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

CATALOG_SCHEMA_VERSION = 1

# Below this a report is an order entry or a short notation ("certiorari denied", sometimes with a
# brief statement), not opinion text a holding can be checked against.
MIN_OPINION_WORDS = 300
# A test case needs at least this much of its target opinion in the corpus to be evaluated properly.
MIN_WORDS_FOR_EVALUATION = 500

CITE_MATCH = "MATCH"                      # the reporter cite and the case name identify the same report
CITE_MISMATCH = "MISMATCH"                # a report starts at the cited page, but it is a different case
CITE_UNKNOWN_PAGE = "UNKNOWN_PAGE"        # the volume is in the corpus but no report starts at that page (pin cite or wrong page)
CITE_VOLUME_NOT_IN_CORPUS = "VOLUME_NOT_IN_CORPUS"
CITE_NOT_GIVEN = "NOT_GIVEN"

_US_CITE_RE = re.compile(r"(\d{1,3})\s*U\.?\s*S\.?\s*(\d{1,4})\b", re.IGNORECASE)
_VERSUS_RE = re.compile(r"\s+v(?:s)?\.?\s+", re.IGNORECASE)
_STOP_WORDS = {"the", "of", "and", "for", "in", "on", "et", "al", "ex", "rel", "a", "an", "to", "by", "at", "state"}


def parse_us_cite(text: str) -> Optional[Tuple[str, int]]:
    """'410 U.S. 113 (1973)' -> ('410', 113). Returns None when there is no U.S. Reports citation."""
    match = _US_CITE_RE.search(text or "")
    return (str(int(match.group(1))), int(match.group(2))) if match else None


def _tokens(text: str) -> List[str]:
    """Comparable tokens of one party name: lower-case, no punctuation, 'U. S. A.' -> 'usa', no filler words."""
    text = re.sub(r"['’`]", "", (text or "").lower())
    words = [w for w in re.split(r"[^a-z0-9]+", text) if w]
    merged: List[str] = []
    letters = ""
    for word in words + [""]:
        if len(word) == 1 and word.isalpha():
            letters += word
            continue
        if letters:
            merged.append(letters)
            letters = ""
        if word:
            merged.append(word)
    return [w for w in merged if w not in _STOP_WORDS]


def _sides(name: str) -> Tuple[List[str], List[str]]:
    parts = _VERSUS_RE.split((name or "").strip(), maxsplit=1)
    return _tokens(parts[0]), (_tokens(parts[1]) if len(parts) > 1 else [])


def normalize_name(name: str) -> str:
    left, right = _sides(name)
    return " ".join(left) + (" v " + " ".join(right) if right else "")


def _token_equivalent(a: str, b: str) -> bool:
    """Same word, allowing reporter abbreviations: 'res' ~ 'resources', 'assn' ~ 'association'."""
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 2 or short[0] != long[0]:
        return False
    if len(short) >= 3 and long.startswith(short):
        return True
    if short[-1] == long[-1]:  # contraction: first and last letter kept, letters in order ('pa' ~ 'pennsylvania')
        position = 0
        for char in short:
            position = long.find(char, position) + 1
            if position == 0:
                return False
        return True
    return False


def _side_match(a: Sequence[str], b: Sequence[str]) -> bool:
    """Do two party names denote the same party? Handles abbreviation, truncation and acronyms."""
    if not a or not b:
        return not a and not b
    short, long = (a, b) if len(set(a)) <= len(set(b)) else (b, a)
    short_set, long_set = list(dict.fromkeys(short)), list(dict.fromkeys(long))
    if len(short_set) == 1 and len(long_set) >= 2 and short_set[0] == "".join(word[0] for word in long_set):
        return True  # acronym: 'ins' for Immigration & Naturalization Service
    matched = sum(1 for word in short_set if any(_token_equivalent(word, other) for other in long_set))
    if matched < len(short_set):
        return False
    if len(short_set) / len(long_set) >= 0.6:
        return True
    # A long party name cut down to its leading words: 'Employment Division' for
    # 'Employment Division, Department of Human Resources of Oregon'.
    return len(short_set) >= 2 and all(_token_equivalent(s, l) for s, l in zip(short_set, long_set))


def names_overlap(a: str, b: str) -> bool:
    """Looser than names_match: each party shares at least one word with its counterpart.

    Used only once a reporter cite has already pinned the report down, where the conventional name
    of a case can differ from the reported one ('Dred Scott v. Sandford' is reported as
    'Scott v. Sandford', 'Barron v. Baltimore' as 'Barron v. Mayor of Baltimore').
    """
    left_a, right_a = _sides(a)
    left_b, right_b = _sides(b)

    def share(x: Sequence[str], y: Sequence[str]) -> bool:
        if not x or not y:
            return not x and not y
        return any(_token_equivalent(i, j) for i in x for j in y)

    return bool(left_a or right_a) and share(left_a, left_b) and share(right_a, right_b)


def names_match(a: str, b: str) -> bool:
    """True when two case names are the same case written differently. Both parties must agree."""
    left_a, right_a = _sides(a)
    left_b, right_b = _sides(b)
    if not (left_a or right_a) or not (left_b or right_b):
        return False
    return _side_match(left_a, left_b) and _side_match(right_a, right_b)


@dataclass(frozen=True)
class Report:
    """One reported case in the corpus (one source document, split into chunks for retrieval)."""
    doc_id: str
    name: str
    volume: str
    first_page: int
    year: int
    chunks: int
    words: int

    @property
    def cite(self) -> str:
        return f"{self.volume} U.S. {self.first_page}" if self.volume and self.first_page else ""

    @property
    def is_opinion(self) -> bool:
        return self.words >= MIN_OPINION_WORDS


@dataclass(frozen=True)
class Resolution:
    """Which report a citation refers to, and how its reporter cite checked out."""
    report: Optional[Report]
    basis: str        # "cite" | "name" | ""
    cite_status: str
    same_name_reports: int = 0


class CaseCatalog:
    def __init__(self, reports: Iterable[Report]):
        self.reports: List[Report] = list(reports)
        self.volumes: Set[str] = {r.volume for r in self.reports if r.volume}
        self._by_cite: Dict[Tuple[str, int], List[Report]] = defaultdict(list)
        self._by_name: Dict[str, List[Report]] = defaultdict(list)
        self._by_prefix: Dict[str, Set[str]] = defaultdict(set)
        for report in self.reports:
            if report.volume and report.first_page:
                self._by_cite[(report.volume, report.first_page)].append(report)
            key = normalize_name(report.name)
            if not key:
                continue
            if key not in self._by_name:
                left, right = _sides(report.name)
                for token in set(left + right):
                    self._by_prefix[token[:3]].add(key)
            self._by_name[key].append(report)

    def __len__(self) -> int:
        return len(self.reports)

    # ── building and persistence ──────────────────────────────────────────
    @classmethod
    def from_chunks(cls, chunks: Iterable[Any]) -> "CaseCatalog":
        """Build from retrieval chunks (CorpusDocument objects): one report per parent document."""
        grouped: Dict[str, dict] = {}
        for chunk in chunks:
            meta = chunk.metadata or {}
            doc_id = meta.get("parent_doc_id") or chunk.doc_id
            entry = grouped.get(doc_id)
            if entry is None:
                entry = grouped[doc_id] = {"name": chunk.case_name, "volume": str(chunk.volume or ""),
                                           "first_page": int(chunk.first_page or 0), "year": int(chunk.year or 0), "chunks": 0, "words": 0}
            entry["chunks"] += 1
            entry["words"] = max(entry["words"], int(meta.get("word_end") or 0))
        return cls(Report(doc_id=doc_id, **entry) for doc_id, entry in grouped.items())

    @classmethod
    def from_collection(cls, collection, page_size: int = 20000) -> "CaseCatalog":
        """Build from the metadata stored in the ChromaDB collection (used when no catalog file exists)."""
        grouped: Dict[str, dict] = {}
        offset = 0
        while True:
            metadatas = collection.get(include=["metadatas"], limit=page_size, offset=offset).get("metadatas") or []
            if not metadatas:
                break
            for meta in metadatas:
                doc_id = meta.get("parent_doc_id")
                if not doc_id:
                    continue
                entry = grouped.get(doc_id)
                if entry is None:
                    entry = grouped[doc_id] = {"name": str(meta.get("case_name") or ""), "volume": str(meta.get("volume") or ""),
                                               "first_page": int(meta.get("first_page") or 0), "year": int(meta.get("year") or 0), "chunks": 0, "words": 0}
                entry["chunks"] += 1
                entry["words"] = max(entry["words"], int(meta.get("word_end") or 0))
            offset += len(metadatas)
        return cls(Report(doc_id=doc_id, **entry) for doc_id, entry in grouped.items())

    def save(self, path: str) -> None:
        rows = [[r.doc_id, r.name, r.volume, r.first_page, r.year, r.chunks, r.words] for r in self.reports]
        atomic_write_json(path, {"schema_version": CATALOG_SCHEMA_VERSION, "fields": ["doc_id", "name", "volume", "first_page", "year", "chunks", "words"], "reports": rows})

    @classmethod
    def load(cls, path: str) -> Optional["CaseCatalog"]:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("schema_version") != CATALOG_SCHEMA_VERSION:
                return None
            return cls(Report(*row) for row in data["reports"])
        except Exception as e:
            logger.warning("Could not read the case catalog at %s: %s", path, e)
            return None

    # ── lookup ────────────────────────────────────────────────────────────
    def reports_at(self, volume: str, first_page: int) -> List[Report]:
        return list(self._by_cite.get((str(volume), int(first_page)), []))

    def reports_named(self, name: str) -> List[Report]:
        """Reports carrying this case name: exactly, or under an equivalent spelling if no exact one is an opinion."""
        key = normalize_name(name)
        exact = list(self._by_name.get(key, []))
        if any(report.is_opinion for report in exact):
            return exact
        left, right = _sides(name)
        postings = sorted((self._by_prefix.get(token[:3], set()) for token in set(left + right)), key=len)
        candidate_keys: Set[str] = set()
        for index, keys in enumerate(postings):
            if index == 0 or len(keys) <= 5000:
                candidate_keys |= keys
        variants = [report for other in candidate_keys if other != key
                    for report in self._by_name[other] if names_match(name, report.name)]
        return exact + variants

    def resolve(self, case_name: str, reporter_cite: str = "", year: int = 0) -> Resolution:
        """Find the report a citation refers to. The reporter cite is checked first, then the name."""
        cite = parse_us_cite(reporter_cite) or parse_us_cite(case_name)
        cite_status = CITE_NOT_GIVEN
        cited_volume = ""
        if cite:
            cited_volume, page = cite
            if cited_volume not in self.volumes:
                cite_status = CITE_VOLUME_NOT_IN_CORPUS
            else:
                at_page = self.reports_at(cited_volume, page)
                agreeing = ([report for report in at_page if names_match(case_name, report.name)]
                            or [report for report in at_page if names_overlap(case_name, report.name)])
                if agreeing:
                    return Resolution(max(agreeing, key=lambda r: r.words), "cite", CITE_MATCH, len(agreeing))
                cite_status = CITE_MISMATCH if at_page else CITE_UNKNOWN_PAGE

        named = self.reports_named(case_name)
        if not named:
            return Resolution(None, "", cite_status, 0)
        pool = named
        # A case name can belong to several distinct cases. The cited volume, then the year, narrow it
        # down, but only when they point at real opinion text; otherwise the longest report is the opinion.
        for narrowed in ([r for r in named if cited_volume and r.volume == cited_volume and r.is_opinion],
                         [r for r in named if year and r.year == int(year) and r.is_opinion]):
            if narrowed:
                pool = narrowed
                break
        return Resolution(max(pool, key=lambda r: (r.words, r.chunks)), "name", cite_status, len(named))


def split_cases_by_source_text(cases: List[dict], catalog: CaseCatalog, min_words: int = MIN_WORDS_FOR_EVALUATION) -> Tuple[List[dict], List[dict]]:
    """Split test cases into those whose target opinion is in the corpus and those it cannot support.

    Returns (evaluable, excluded). Each excluded case gets an 'excluded_reason'.
    """
    evaluable: List[dict] = []
    excluded: List[dict] = []
    for case in cases:
        resolution = catalog.resolve(case.get("target_case", ""), case.get("reporter_citation", ""))
        report = resolution.report
        if report is not None and report.words >= min_words:
            evaluable.append(case)
        elif report is None:
            excluded.append({**case, "excluded_reason": "the target case is not in the corpus"})
        else:
            excluded.append({**case, "excluded_reason": f"only {report.words} words of the target case are in the corpus ({report.cite})"})
    return evaluable, excluded
