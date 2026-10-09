# LexAgent v3.0 | ledger.py
"""Citation ledger of one debate (repair 6 of the forensic audit).

The audit found that each round replaced the citations and the citation report of the round before
it, and that the judge read the last report only: a citation that was supported in round one could
be gone when the verdict was written. The ledger keeps, for every citation made in any round, the
best verdict it received and the evidence for it. The citation checker adds to it after each round,
the controller reads from it whether anything is supported yet, and the judge reads it instead of
the last report.

Nothing here calls a model. All functions are pure.
"""

import re
from typing import Dict, List, Optional

# A higher rank is a better verdict. A later, lower verdict never replaces an earlier, higher one.
TIER_RANK = {
    "SUPPORTED": 6,     # exists, and a passage of the opinion shows the holding
    "VERIFIED": 5,      # exists; no validated support model (gate closed), so support is not judged
    "UNCERTAIN": 4,     # supported, but the reporter cite belongs to another case
    "NOT_SHOWN": 3,     # exists; the passages read do not show the holding
    "MISATTRIBUTED": 2, # exists; a passage contradicts the holding (legacy: low support score)
    "UNVERIFIABLE": 1,  # cannot be checked (not in the corpus, or no opinion text)
    "FABRICATED": 0,    # the cited volume is in the corpus and the case is not in it
}
RELIABLE_TIERS = ("SUPPORTED", "VERIFIED")
JUDGE_LABEL = {
    "SUPPORTED": "SUPPORTED", "VERIFIED": "EXISTS (support not judged)", "UNCERTAIN": "UNCERTAIN (cite belongs to another case)",
    "NOT_SHOWN": "NOT SHOWN", "MISATTRIBUTED": "CONTRADICTED", "UNVERIFIABLE": "CANNOT BE CHECKED", "FABRICATED": "NOT FOUND",
}


def clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(name or "").lower())).strip()


def ledger_key(entry: dict) -> str:
    """One key per cited decision: the report it resolves to, else its cleaned name."""
    resolved = entry.get("resolved_doc_id")
    if resolved:
        return f"report::{resolved}"
    citation = entry.get("citation") or {}
    return f"name::{clean_name(citation.get('case_name', ''))}"


def _better(new: dict, old: dict) -> bool:
    new_rank, old_rank = TIER_RANK.get(new.get("tier"), -1), TIER_RANK.get(old.get("tier"), -1)
    if new_rank != old_rank:
        return new_rank > old_rank
    return float(new.get("ccs") or 0.0) > float(old.get("ccs") or 0.0)


def merge_round(ledger: Optional[Dict[str, dict]], report: Dict[str, dict], round_number: int) -> tuple:
    """Add one round's citation report to the ledger.

    Returns (new ledger, number of decisions that are reliable now and were not before this round).
    The input ledger is not modified.
    """
    merged = {key: dict(value) for key, value in (ledger or {}).items()}
    reliable_before = {key for key, value in merged.items() if value.get("tier") in RELIABLE_TIERS}
    for entry in (report or {}).values():
        if not isinstance(entry, dict):
            continue
        citation = entry.get("citation") or {}
        key = ledger_key(entry)
        if key == "name::":
            continue
        candidate = {
            "case_name": entry.get("resolved_name") or citation.get("case_name", ""),
            "cited_as": citation.get("case_name", ""),
            "reporter_cite": entry.get("resolved_cite") or citation.get("reporter_cite", ""),
            "holding": citation.get("holding", ""),
            "tier": entry.get("tier", ""),
            "ccs": float(entry.get("ccs") or 0.0),
            "existence_verdict": entry.get("existence_verdict"),
            "support_verdict": entry.get("support_verdict"),
            "support_score": entry.get("support_score", entry.get("support")),
            "evidence": entry.get("evidence_window") or entry.get("evidence") or "",
            "resolved_doc_id": entry.get("resolved_doc_id"),
            "agent_source": citation.get("agent_source", ""),
            "round": round_number,
            "citation": dict(citation),
        }
        previous = merged.get(key)
        if previous is None:
            candidate["rounds_cited"] = [round_number]
            merged[key] = candidate
            continue
        rounds = sorted(set(previous.get("rounds_cited", [])) | {round_number})
        if _better(candidate, previous):
            candidate["rounds_cited"] = rounds
            merged[key] = candidate
        else:
            previous["rounds_cited"] = rounds
    reliable_after = {key for key, value in merged.items() if value.get("tier") in RELIABLE_TIERS}
    return merged, len(reliable_after - reliable_before)


def reliable_entries(ledger: Optional[Dict[str, dict]]) -> List[dict]:
    """Ledger entries an answer may rest on, best first."""
    entries = [value for value in (ledger or {}).values() if value.get("tier") in RELIABLE_TIERS]
    return sorted(entries, key=lambda e: (-TIER_RANK.get(e.get("tier"), -1), -float(e.get("ccs") or 0.0)))


def has_reliable_citation(ledger: Optional[Dict[str, dict]]) -> bool:
    return any(value.get("tier") in RELIABLE_TIERS for value in (ledger or {}).values())


def format_for_judge(ledger: Optional[Dict[str, dict]], evidence_chars: int = 300) -> str:
    """The citation record the judge reads: every decision cited in the debate, best verdict first."""
    if not ledger:
        return "(No citation was checked in this debate)"
    ordered = sorted(ledger.values(), key=lambda e: (-TIER_RANK.get(e.get("tier"), -1), -float(e.get("ccs") or 0.0)))
    lines = []
    for number, entry in enumerate(ordered, start=1):
        label = JUDGE_LABEL.get(entry.get("tier"), str(entry.get("tier") or "UNKNOWN"))
        cite = f", {entry['reporter_cite']}" if entry.get("reporter_cite") else ""
        line = f"  [{number}] {entry.get('case_name') or entry.get('cited_as')}{cite} — {label}"
        holding = str(entry.get("holding") or "").strip()
        if holding:
            line += f"\n      cited for: {holding[:240]}"
        evidence = " ".join(str(entry.get("evidence") or "").split())
        if evidence and entry.get("tier") in ("SUPPORTED", "UNCERTAIN"):
            line += f"\n      passage of the opinion: \"{evidence[:evidence_chars]}\""
        lines.append(line)
    return "\n".join(lines)
