# LexAgent v3.0 | judge_agent.py
"""Judge Agent — DVA+ Protocol."""

import re
import json
import time
from typing import List, Dict, Any
from config import LEDGER_EVIDENCE_CHARS
from src.agents.state import LexAgentState, debate_protocol
from src.agents.prompts import (
    SYSTEM_JUDGE, USER_JUDGE, SYSTEM_JUDGE_LEDGER, USER_JUDGE_LEDGER, format_mistral_prompt
)
from src.agents.prosecutor_agent import (
    _invoke_llm, _safe_parse_json, _safe_year, _clean_case_name
)
from src.data.case_catalog import names_match
from src.data.corpus_schema import CitationRecord
from src.novel.ledger import RELIABLE_TIERS, clean_name, format_for_judge
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ask_judge(llm, prompt: str, fallback_prompt: str, attempts: int = 3) -> dict:
    """Ask for the ruling; retry when the reply is empty or not a readable ruling.

    A busy free-tier endpoint can return nothing, or a reply cut short. Scoring that as a ruling
    gives confidence 0 even when every citation checked out, so it is asked again instead.
    """
    parsed: dict = {}
    for attempt in range(1, attempts + 1):
        raw = _invoke_llm(llm, prompt, fallback_prompt=fallback_prompt)
        parsed = _safe_parse_json(raw, fallback_key="verdict")
        if not isinstance(parsed, dict):
            parsed = {}
        if "verified_citations" in parsed or "confidence" in parsed:
            return parsed
        logger.warning("Judge reply unusable (attempt %d/%d, %d chars): %r",
                       attempt, attempts, len(raw or ""), (raw or "")[:200])
        if attempt < attempts:
            time.sleep(8 * attempt)
    return parsed


def _named_in_ruling(parsed: dict, names: List[str]) -> List[str]:
    """Checked case names the judge wrote into its ruling text but left out of verified_citations."""
    text = " ".join(str(parsed.get(k, "") or "") for k in ("verdict", "legal_holding", "reasoning")).lower()
    found = []
    for name in names:
        tokens = set(re.findall(r"\b[a-z]{3,}\b", name.lower())) - {"the", "and", "for", "state", "court", "united", "states"}
        if tokens and all(t in text for t in tokens):
            found.append(name)
    return found


def _format_debate_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "(No prior debate rounds recorded)"
    lines = []
    for item in history:
        r = item.get("round", 0)
        role = str(item.get("role", "")).upper()
        if role == "PROSECUTOR":
            arg = str(item.get("argument", ""))[:400]
            lines.append(f"Round {r} [PROSECUTOR]: {arg}")
        elif role == "DEFENSE":
            arg = str(item.get("counter_argument", ""))[:400]
            lines.append(f"Round {r} [DEFENSE]: {arg}")
        elif role == "REFLECTION":
            gaps = item.get("gaps", [])
            lines.append(f"Round {r} [REFLECTION]: Gaps: {gaps}")
    return "\n\n".join(lines) if lines else "(No debate history)"


def _format_ccs_report(report: dict) -> str:
    if not report:
        return "CCE report not yet available"
    lines = []
    for cid, data in report.items():
        ccs = data.get("ccs", 0.0)
        tier = data.get("tier", "UNKNOWN")
        evidence = data.get("evidence", "")[:100]
        lines.append(f"  {cid}: CCS={ccs:.2f} [{tier}]"
                     + (f" — {evidence}" if evidence else ""))
    return "\n".join(lines)


def _find_in_ledger(case_name: str, ledger: dict):
    """The ledger entry a name written by the judge refers to: the same cleaned name, else the same
    two parties. (The v1 rule accepted any two shared words, so that one "United States v." case
    could stand in for another.)"""
    wanted = clean_name(case_name)
    if not wanted:
        return None
    for entry in ledger.values():
        if wanted in (clean_name(entry.get("case_name", "")), clean_name(entry.get("cited_as", ""))):
            return entry
    for entry in ledger.values():
        if names_match(case_name, entry.get("case_name", "")) or names_match(case_name, entry.get("cited_as", "")):
            return entry
    return None


def _judge_from_ledger(state: LexAgentState) -> dict:
    """Ledger protocol: the judge reads the record of all rounds and is instructed by verdict.

    The program still decides what the answer rests on: a decision the judge relies on counts only
    if the ledger holds it as SUPPORTED (or VERIFIED while no support model is validated). A decision
    the judge relied on whose holding was not shown is kept and reported as uncertain, not dropped.
    """
    ledger = state.get("citation_ledger") or {}
    user_content = USER_JUDGE_LEDGER.format(
        query=state["query"],
        jurisdiction=state.get("jurisdiction", "US_Federal"),
        debate_history=_format_debate_history(state.get("debate_history", [])),
        ccs_report=format_for_judge(ledger, LEDGER_EVIDENCE_CHARS),
        reflection_summary=f"Gaps: {state.get('reflection_gaps', [])}. Quality: {state.get('reflection_score', 0.5):.2f}",
    )
    prompt = format_mistral_prompt(SYSTEM_JUDGE_LEDGER, user_content)
    fallback_prompt = format_mistral_prompt(
        "You are the SENIOR JUDGE in a legal debate. Respond ONLY in valid JSON.",
        f"Deliver ruling on: {state['query']}.\nProvide JSON with 'verdict', 'legal_holding', 'verified_citations', 'uncertain_citations', 'rejected_citations', 'confidence', 'reasoning'."
    )
    parsed = _ask_judge(state["llm"], prompt, fallback_prompt)

    verified, uncertain, seen = [], [], set()

    def record(entry: dict, given: dict, confidence: float, note: str = "") -> CitationRecord:
        return CitationRecord(
            case_name=entry.get("case_name") or _clean_case_name(given.get("case_name", "")),
            court=str(given.get("court", "") or "Supreme Court"),
            year=_safe_year(given.get("year")),
            holding=note or str(given.get("holding", "") or entry.get("holding", "") or ""),
            self_confidence=confidence,
            agent_source="judge",
            reporter_cite=str(entry.get("reporter_cite", "") or ""),
        )

    for given in parsed.get("verified_citations", []) or []:
        if not isinstance(given, dict):
            continue
        name = _clean_case_name(given.get("case_name", ""))
        entry = _find_in_ledger(name, ledger) if name else None
        if entry is None:
            if name:
                logger.info("Judge citation '%s' is not in the citation record; not counted", name)
            continue
        key = entry.get("resolved_doc_id") or clean_name(entry.get("case_name", ""))
        if key in seen:
            continue
        seen.add(key)
        if entry.get("tier") in RELIABLE_TIERS:
            score = float(entry.get("ccs") or 0.0)
            verified.append(record(entry, given, score if score > 0 else 0.85))
        elif entry.get("tier") in ("NOT_SHOWN", "UNCERTAIN"):
            logger.info("Judge relied on '%s' (%s); reported as uncertain", name, entry.get("tier"))
            uncertain.append(record(entry, given, 0.50, note="relied on by the judge; holding not shown in the passages read"))
        else:
            logger.info("Judge citation '%s' is %s in the citation record; not counted", name, entry.get("tier"))

    if not verified:
        # The judge argued from checked decisions but did not list them: count those it named.
        reliable = [e.get("case_name", "") for e in ledger.values() if e.get("tier") in RELIABLE_TIERS]
        for name in _named_in_ruling(parsed, reliable):
            entry = _find_in_ledger(name, ledger)
            key = (entry or {}).get("resolved_doc_id") or clean_name(name)
            if entry and key not in seen:
                seen.add(key)
                score = float(entry.get("ccs") or 0.0)
                verified.append(record(entry, {"case_name": name}, score if score > 0 else 0.85))
                logger.info("Judge named checked decision '%s' in its ruling; counted as verified", name)

    for given in parsed.get("uncertain_citations", []) or []:
        if not isinstance(given, dict):
            continue
        name = _clean_case_name(given.get("case_name", ""))
        if not name:
            continue
        entry = _find_in_ledger(name, ledger) or {}
        key = entry.get("resolved_doc_id") or clean_name(entry.get("case_name", "") or name)
        if key in seen:
            continue
        seen.add(key)
        uncertain.append(record(entry, given, 0.50, note=str(given.get("note", "") or "")))

    final_answer = parsed.get("legal_holding") or parsed.get("verdict")
    if not final_answer or str(final_answer).strip() == "":
        if verified:
            final_answer = f"Based on verified precedent ({', '.join(c.case_name for c in verified)}), {parsed.get('reasoning', 'the legal proposition is resolved.')}"
        else:
            final_answer = "Insufficient evidence for a verified answer."

    try:
        self_rating = min(max(float(parsed.get("confidence", 0.0)), 0.0), 1.0)
    except (ValueError, TypeError):
        self_rating = 0.0
    if verified:
        mean_score = sum(c.self_confidence for c in verified) / len(verified)
        confidence = round(0.7 * mean_score + 0.3 * self_rating, 2) if self_rating > 0 else round(mean_score, 2)
    elif uncertain:
        confidence = round(min(self_rating, 0.35) if self_rating > 0 else 0.20, 2)
    else:
        confidence = round(min(self_rating, 0.20), 2)

    logger.info("Judge result: keys=%s self_rating=%.2f verified=%d uncertain=%d confidence=%.2f",
                sorted(parsed.keys()), self_rating, len(verified), len(uncertain), confidence)
    # The parts of the confidence, reported next to the blended number.
    existing = [e for e in ledger.values() if e.get("tier") not in ("FABRICATED", "UNVERIFIABLE")]
    reliable = [e for e in existing if e.get("tier") in RELIABLE_TIERS]
    components = {
        "decisions_cited_in_the_debate": len(ledger),
        "existence_rate": round(len(existing) / len(ledger), 3) if ledger else None,
        "support_rate_among_existing": round(len(reliable) / len(existing), 3) if existing else None,
        "judge_self_rating": self_rating,
    }
    return {
        "judge_verdict":              parsed.get("verdict", ""),
        "judge_confidence":           confidence,
        "judge_verified_citations":   verified,
        "judge_uncertain_citations":  uncertain,
        "judge_reasoning":            parsed.get("reasoning", ""),
        "final_answer":               final_answer,
        "structured_output": {
            "answer":                final_answer,
            "confidence":            confidence,
            "confidence_components": components,
            "verdict":               parsed.get("verdict", ""),
            "verified_citations":    [c.to_dict() for c in verified],
            "uncertain_citations":   [c.to_dict() for c in uncertain],
            "rejected_citations":    parsed.get("rejected_citations", []),
            "debate_rounds":         state.get("debate_round", 0) + 1,
            "reasoning":             parsed.get("reasoning", ""),
        },
    }


def judge_node(state: LexAgentState) -> dict:
    logger.info("Judge node: rendering verdict for round %d",
                state.get("debate_round", 0))

    if state.get("use_cce", True) and debate_protocol(state) == "ledger":
        return _judge_from_ledger(state)

    debate_history_str = _format_debate_history(state.get("debate_history", []))
    ccs_report = _format_ccs_report(state.get("citation_confidence_report", {}))
    reflection_summary = (
        f"Gaps: {state.get('reflection_gaps', [])}. "
        f"Quality: {state.get('reflection_score', 0.5):.2f}"
    )

    user_content = USER_JUDGE.format(
        query=state["query"],
        jurisdiction=state.get("jurisdiction", "US_Federal"),
        debate_history=debate_history_str,
        ccs_report=ccs_report,
        reflection_summary=reflection_summary,
    )
    prompt = format_mistral_prompt(SYSTEM_JUDGE, user_content)
    fallback_prompt = format_mistral_prompt(
        "You are the SENIOR JUDGE in a legal debate. Respond ONLY in valid JSON.",
        f"Deliver ruling on: {state['query']}.\nProvide JSON with 'verdict', 'legal_holding', 'verified_citations', 'uncertain_citations', 'rejected_citations', 'confidence', 'reasoning'."
    )

    parsed = _ask_judge(state["llm"], prompt, fallback_prompt)

    use_cce = state.get("use_cce", True)
    verified_ccs_map = {}
    uncertain_ccs_map = {}

    if use_cce:
        for data in state.get("citation_confidence_report", {}).values():
            c_name = _clean_case_name(data.get("citation", {}).get("case_name", "")).lower()
            if not c_name:
                continue
            ccs_score = float(data.get("ccs", 0.0) or 0.0)
            tier = data.get("tier", "")
            if tier in ("SUPPORTED", "VERIFIED"):
                verified_ccs_map[c_name] = max(verified_ccs_map.get(c_name, 0.0), ccs_score if ccs_score > 0 else 0.85)
            elif tier == "UNCERTAIN":
                uncertain_ccs_map[c_name] = max(uncertain_ccs_map.get(c_name, 0.0), ccs_score if ccs_score > 0 else 0.50)

    verified = []
    for c in parsed.get("verified_citations", []) or []:
        try:
            case_name = _clean_case_name(c.get("case_name", ""))
            if not case_name:
                continue
            case_name_lower = case_name.lower()
            actual_ccs = None

            if not use_cce:
                # In A2 ablation without CCE: accept cited precedent without CCE verification
                actual_ccs = float(c.get("confidence", 0.75) or 0.75)
            elif case_name_lower in verified_ccs_map:
                actual_ccs = verified_ccs_map[case_name_lower]
            else:
                c_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', case_name_lower)) - {'the', 'and', 'for', 'state', 'court'}
                for v_name, v_score in verified_ccs_map.items():
                    v_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', v_name)) - {'the', 'and', 'for', 'state', 'court'}
                    # Match if tokens overlap heavily or one is subset of other (e.g. Riley v. Cal. -> Riley v. California)
                    if c_tokens and (c_tokens.issubset(v_tokens) or v_tokens.issubset(c_tokens) or (len(c_tokens & v_tokens) >= 2)):
                        actual_ccs = v_score
                        break
            
            if use_cce and actual_ccs is None:
                logger.info("Rejecting judge citation '%s': not verified by CCE", case_name)
                continue
            verified.append(CitationRecord(
                case_name=case_name,
                court=str(c.get("court", "") or "Supreme Court"),
                year=_safe_year(c.get("year")),
                holding=str(c.get("holding", "") or ""),
                self_confidence=actual_ccs,
                agent_source="judge"
            ))
        except (ValueError, TypeError) as e:
            logger.warning("Skipping malformed verified citation: %s (%s)", c, e)

    if use_cce and not verified and verified_ccs_map:
        # The judge argued from checked decisions but did not list them: count those it named.
        for name in _named_in_ruling(parsed, list(verified_ccs_map)):
            verified.append(CitationRecord(case_name=name.title(), court="Supreme Court", year=0, holding="",
                                           self_confidence=verified_ccs_map[name], agent_source="judge"))
            logger.info("Judge named checked decision '%s' in its ruling; counted as verified", name)

    uncertain = []
    for c in parsed.get("uncertain_citations", []) or []:
        try:
            case_name = _clean_case_name(c.get("case_name", ""))
            if not case_name:
                continue
            case_name_lower = case_name.lower()
            actual_ccs = uncertain_ccs_map.get(case_name_lower, 0.50) if use_cce else 0.50
            uncertain.append(CitationRecord(
                case_name=case_name,
                court="",
                year=_safe_year(c.get("year")),
                holding=str(c.get("note", "") or ""),
                self_confidence=actual_ccs,
                agent_source="judge"
            ))
        except (ValueError, TypeError) as e:
            logger.warning("Skipping malformed uncertain citation: %s (%s)", c, e)

    final_answer = parsed.get("legal_holding") or parsed.get("verdict")
    if not final_answer or str(final_answer).strip() == "":
        if verified:
            final_answer = f"Based on verified precedent ({', '.join([c.case_name for c in verified])}), {parsed.get('reasoning', 'the legal proposition is resolved.')}"
        else:
            final_answer = "Insufficient evidence for a verified answer."

    raw_model_conf = 0.0
    try:
        raw_model_conf = float(parsed.get("confidence", 0.0))
        raw_model_conf = min(max(raw_model_conf, 0.0), 1.0)
    except (ValueError, TypeError):
        pass

    if not use_cce:
        # A2 ablation: Confidence reflects model's self-assessed confidence unconstrained by CCE
        confidence = round(raw_model_conf if raw_model_conf > 0 else 0.70, 2)
    elif verified:
        avg_verified_ccs = sum(c.self_confidence for c in verified) / len(verified)
        confidence = round(0.7 * avg_verified_ccs + 0.3 * raw_model_conf, 2) if raw_model_conf > 0 else round(avg_verified_ccs, 2)
    elif uncertain:
        confidence = round(min(raw_model_conf, 0.35) if raw_model_conf > 0 else 0.20, 2)
    else:
        confidence = round(min(raw_model_conf, 0.20), 2)
    logger.info("Judge result: keys=%s model_conf=%.2f verified=%d uncertain=%d confidence=%.2f",
                sorted(parsed.keys()), raw_model_conf, len(verified), len(uncertain), confidence)

    return {
        "judge_verdict":              parsed.get("verdict", ""),
        "judge_confidence":           confidence,
        "judge_verified_citations":   verified,
        "judge_uncertain_citations":  uncertain,
        "judge_reasoning":            parsed.get("reasoning", ""),
        "final_answer":               final_answer,
        "structured_output": {
            "answer":                final_answer,
            "confidence":            confidence,
            "verdict":               parsed.get("verdict", ""),
            "verified_citations":    [c.to_dict() for c in verified],
            "uncertain_citations":   [c.to_dict() for c in uncertain],
            "rejected_citations":    parsed.get("rejected_citations", []),
            "debate_rounds":         state.get("debate_round", 0) + 1,
            "reasoning":             parsed.get("reasoning", ""),
        },
    }
