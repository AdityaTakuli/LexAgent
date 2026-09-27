# LexAgent v3.0 | judge_agent.py
"""Judge Agent — DVA+ Protocol."""

import re
import json
from typing import List, Dict, Any
from src.agents.state import LexAgentState
from src.agents.prompts import SYSTEM_JUDGE, USER_JUDGE, format_mistral_prompt
from src.agents.prosecutor_agent import (
    _invoke_llm, _safe_parse_json, _safe_year, _clean_case_name
)
from src.data.corpus_schema import CitationRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


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


def judge_node(state: LexAgentState) -> dict:
    logger.info("Judge node: rendering verdict for round %d",
                state.get("debate_round", 0))

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

    raw = _invoke_llm(state["llm"], prompt, fallback_prompt=fallback_prompt)
    parsed = _safe_parse_json(raw, fallback_key="verdict")

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
    for c in parsed.get("verified_citations", []):
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

    uncertain = []
    for c in parsed.get("uncertain_citations", []):
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
