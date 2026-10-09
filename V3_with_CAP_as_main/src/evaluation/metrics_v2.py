# LexAgent v3.0 | metrics_v2.py
"""Evaluation measures that say what they measure (written after the forensic audit).

The earlier measures had three problems: "authority precision" was recall divided by the number of
citations, the fabricated-name rate was string similarity to a list of names, and answer similarity
could not tell a right answer from an answer to another question. This module replaces them with
separate, plainly named quantities computed from a saved per-case trace:

  existence            a citation resolves to a report in the case catalog
  cite correctness     the reporter cite given belongs to the named case
  target cited         one of the citations is the benchmark's target decision (same report)
  citation count       reported next to "target cited", never divided into it
  holding support      reported only with the validation status of the model that judged it
  uncited mentions     case names in the answer text that the answer does not cite
  answer w/o authority the answer rests on no existing citation
  skipped debate       the answer came from the verdict cache; scored as its own category

Relevance of the other citations is NOT measured: the benchmark names one target per question and
has no set of acceptable authorities. Nothing here is called "precision".

Two scopes are reported for a debate, because they answer different questions:
  "answer"  what the final answer rests on (the judge's relied-on citations; a baseline's citations)
  "pool"    every citation any agent made in any round
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, Iterable, List, Optional

from src.data.case_catalog import names_match

# "Smith v. Jones", "United States v. Windsor", "Planned Parenthood of Southeastern Pa. v. Casey"
CASE_MENTION = re.compile(
    r"\b((?:[A-Z][\w.'&-]*)(?:\s+(?:of|the|for|ex|rel\.|[A-Z][\w.'&-]*)){0,7}?)\s+v\.?\s+"
    r"((?:[A-Z][\w.'&-]*)(?:\s+(?:of|the|for|[A-Z][\w.'&-]*)){0,6})"
)
LEADING_WORDS = {"in", "see", "under", "per", "cf.", "but", "and", "the", "as", "since", "after", "following", "citing", "also", "both"}
EXISTS, NOT_FOUND, NOT_IN_CORPUS = "EXISTS", "NOT_FOUND", "NOT_IN_CORPUS"

Resolver = Callable[[str, str], dict]


def case_mentions(text: str) -> List[str]:
    """Case names written out in a piece of text ("X v. Y"), in order, without duplicates."""
    text = text or ""
    found, seen, position = [], set(), 0
    while True:
        match = CASE_MENTION.search(text, position)
        if not match:
            break
        first = match.group(1).strip().split()
        while len(first) > 1 and first[0].lower() in LEADING_WORDS:
            first = first[1:]
        # The second party ends at a sentence end: "States." closes it, "Pa." or "Inc." does not.
        kept = []
        for word in match.group(2).split():
            kept.append(word)
            if word.endswith(".") and len(word) > 5:
                break
        end = match.start(2) + match.group(2).find(kept[-1]) + len(kept[-1])
        position = max(end, match.start() + 1)
        name = f"{' '.join(first)} v. {' '.join(kept)}".rstrip(".,;:")
        key = name.lower()
        if key not in seen:
            seen.add(key)
            found.append(name)
    return found


def citations_from_baseline_output(raw: str) -> Optional[List[dict]]:
    """The citations of a single-agent answer, read from the model's raw output.

    The baseline is asked for {"final_answer": ..., "citations": [{"case_name": ...}]}. The model
    often returns the citations as plain strings instead ("citations": ["Miranda v. Arizona", ...]),
    most of the time when it is given eight passages. The parser of the runs kept only the object
    form and silently dropped the rest, which counted those answers as citing nothing. Both forms
    are read here. Returns None when the output holds no JSON object (the caller keeps what the run
    saved).
    """
    found = re.search(r"\{.*\}", str(raw or ""), re.DOTALL)
    if not found:
        return None
    try:
        parsed = json.loads(found.group(0))
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    listed = parsed.get("citations")
    if not isinstance(listed, list):
        return []
    citations = []
    for item in listed:
        if isinstance(item, dict):
            name = str(item.get("case_name", "") or "").strip()
            cite = str(item.get("reporter_cite", "") or item.get("citation", "") or "").strip()
        elif isinstance(item, str):
            text = item.strip()
            reporter = re.search(r"\b\d{1,3}\s+U\.\s?S\.\s+\d{1,4}\b", text)
            cite = reporter.group(0) if reporter else ""
            # "Miranda v. Arizona, 384 U.S. 436 (1966)" -> "Miranda v. Arizona"
            name = re.split(r",\s*\d|\s*\(\d{4}\)|\s+\d{1,3}\s+U\.\s?S\.", text)[0].strip().rstrip(",;.")
        else:
            continue
        if name:
            citations.append({"case_name": name, "reporter_cite": cite})
    return citations


def citation_key(citation: dict) -> str:
    return re.sub(r"\s+", " ", str(citation.get("case_name", "")).strip().lower())


def distinct(citations: Iterable[dict]) -> List[dict]:
    """One entry per cited name, keeping the first that carries a reporter cite."""
    by_name: Dict[str, dict] = {}
    for citation in citations or []:
        key = citation_key(citation)
        if not key:
            continue
        current = by_name.get(key)
        if current is None or (not current.get("reporter_cite") and citation.get("reporter_cite")):
            by_name[key] = citation
    return list(by_name.values())


def check_citations(citations: Iterable[dict], resolver: Resolver, target: Optional[dict] = None) -> List[dict]:
    """Existence, cite correctness and target match for each distinct citation."""
    rows = []
    for citation in distinct(citations):
        found = resolver(citation.get("case_name", ""), citation.get("reporter_cite", "") or "")
        exists = found.get("existence_verdict") == EXISTS
        is_target = False
        if target:
            same_report = bool(found.get("resolved_doc_id")) and found.get("resolved_doc_id") == target.get("doc_id")
            is_target = same_report or (exists and names_match(citation.get("case_name", ""), target.get("name", "")))
        rows.append({
            "case_name": citation.get("case_name", ""), "reporter_cite": citation.get("reporter_cite", "") or "",
            "existence": found.get("existence_verdict"), "exists": exists,
            "cite_check": found.get("cite_check"), "resolved_to": found.get("resolved_name"),
            "resolved_doc_id": found.get("resolved_doc_id"), "is_target": is_target,
        })
    return rows


def summarise_scope(rows: List[dict]) -> dict:
    """The counts for one scope of one answer."""
    with_cite = [r for r in rows if r["cite_check"] in ("MATCH", "MISMATCH", "UNKNOWN_PAGE", "VOLUME_NOT_IN_CORPUS")]
    return {
        "citations": len(rows),
        "existing": sum(1 for r in rows if r["exists"]),
        "not_found": sum(1 for r in rows if r["existence"] == NOT_FOUND),
        "not_in_corpus": sum(1 for r in rows if r["existence"] == NOT_IN_CORPUS),
        "reporter_cites_given": len(with_cite),
        "reporter_cites_correct": sum(1 for r in with_cite if r["cite_check"] == "MATCH"),
        "reporter_cites_wrong": sum(1 for r in with_cite if r["cite_check"] == "MISMATCH"),
        "target_cited": any(r["is_target"] for r in rows),
        "distinct_existing_cases": len({r["resolved_doc_id"] for r in rows if r["exists"] and r["resolved_doc_id"]}),
        "no_existing_authority": not any(r["exists"] for r in rows),
    }


def uncited_mentions(answer: str, answer_rows: List[dict], resolver: Resolver) -> dict:
    """Case names the answer text mentions without citing them, and whether those cases exist."""
    cited = [r["case_name"] for r in answer_rows]
    uncited = [m for m in case_mentions(answer) if not any(names_match(m, name) for name in cited)]
    checked = [(m, resolver(m, "").get("existence_verdict")) for m in uncited]
    return {"case_mentions_in_answer": len(case_mentions(answer)), "uncited_mentions": len(uncited),
            "uncited_mentions_not_found": sum(1 for _, verdict in checked if verdict != EXISTS),
            "uncited_mention_names": uncited}


def support_summary(citation_reports: Iterable[dict]) -> dict:
    """What the citation checker reported about holding support, with its validation status.

    Only citations that exist and whose support was actually checked are counted. If the support
    model was not validated the figures describe the checker, not the answers.
    """
    checked = [r for r in citation_reports or [] if r.get("existence_verdict") == EXISTS and r.get("support_verdict") not in (None, "NOT_CHECKED")]
    return {
        "support_checked": len(checked),
        "support_shown": sum(1 for r in checked if r.get("support_verdict") == "SUPPORTED"),
        "support_contradicted": sum(1 for r in checked if r.get("support_verdict") == "CONTRADICTED"),
        "support_model_validated": bool(checked) and all(bool(r.get("support_validated")) for r in checked),
    }


def score_case(record: dict, resolver: Resolver, target: Optional[dict]) -> dict:
    """All measures for one saved case record (debate or baseline)."""
    skipped = bool(record.get("memory_cache_hit"))
    answer_rows = check_citations(record.get("answer_citations", []), resolver, target)
    pool_rows = check_citations(record.get("pool_citations", record.get("answer_citations", [])), resolver, target)
    final_reports = record.get("final_citation_reports", [])
    return {
        "case_id": record.get("case_id"), "skipped_debate": skipped,
        "answer": summarise_scope(answer_rows), "pool": summarise_scope(pool_rows),
        "mentions": uncited_mentions(record.get("final_answer", ""), answer_rows, resolver),
        "support": support_summary(final_reports),
        "rounds": record.get("rounds"), "confidence": record.get("judge_confidence"),
        "answer_rows": answer_rows, "pool_rows": pool_rows,
    }


def aggregate(scores: List[dict]) -> dict:
    """Means and pooled rates over cases. Skipped debates are counted and kept out of nothing silently."""
    n = len(scores)
    if not n:
        return {"cases": 0}

    def rate(numerator: int, denominator: int) -> Optional[float]:
        return round(numerator / denominator, 4) if denominator else None

    out = {"cases": n, "skipped_debates": sum(1 for s in scores if s["skipped_debate"])}
    for scope in ("answer", "pool"):
        rows = [s[scope] for s in scores]
        citations = sum(r["citations"] for r in rows)
        given = sum(r["reporter_cites_given"] for r in rows)
        out[scope] = {
            "target_cited_rate": rate(sum(1 for r in rows if r["target_cited"]), n),
            "citations_per_answer": round(citations / n, 3),
            "citations": citations,
            "existence_rate": rate(sum(r["existing"] for r in rows), citations),
            "citations_not_found": sum(r["not_found"] for r in rows),
            "citations_not_in_corpus": sum(r["not_in_corpus"] for r in rows),
            "reporter_cites_given": given,
            "reporter_cite_correct_rate": rate(sum(r["reporter_cites_correct"] for r in rows), given),
            "answers_without_existing_authority": sum(1 for r in rows if r["no_existing_authority"]),
        }
    mentions = [s["mentions"] for s in scores]
    out["answers_with_uncited_case_mentions"] = sum(1 for m in mentions if m["uncited_mentions"])
    out["uncited_mentions"] = sum(m["uncited_mentions"] for m in mentions)
    out["uncited_mentions_not_found"] = sum(m["uncited_mentions_not_found"] for m in mentions)
    support = [s["support"] for s in scores]
    checked = sum(x["support_checked"] for x in support)
    out["holding_support"] = {
        "citations_checked": checked,
        "support_shown_rate": rate(sum(x["support_shown"] for x in support), checked),
        "support_model_validated": bool(checked) and all(x["support_model_validated"] for x in support if x["support_checked"]),
    }
    return out
