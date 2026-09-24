# LexAgent v3.0 | prosecutor_agent.py
"""Prosecutor Agent — DVA+ Protocol."""

import json
import re
import time
from typing import List, Optional
from src.agents.state import LexAgentState
from src.agents.prompts import SYSTEM_PROSECUTOR, USER_PROSECUTOR, format_mistral_prompt
from src.data.corpus_schema import CorpusDocument, CitationRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_year(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        m = re.search(r'\d{4}', str(val))
        return int(m.group(0)) if m else default


def _clean_case_name(name: str) -> str:
    if not name:
        return ""
    cleaned = str(name).strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _format_passages(passages: List[CorpusDocument], max_chars: int = 500) -> str:
    if not passages:
        return "(No passages retrieved)"
    lines = []
    for i, doc in enumerate(passages, 1):
        case_label = f" [{doc.case_name}]" if doc.case_name else ""
        vol_label = f" ({doc.volume} {doc.reporter or 'U.S.'} {doc.first_page})" if doc.volume and doc.first_page else ""
        source_label = f" [{doc.source}]" if doc.source else ""
        text_preview = doc.text[:max_chars].replace("\n", " ")
        lines.append(f"[{i}]{case_label}{vol_label}{source_label}: {text_preview}")
    return "\n".join(lines)


def _invoke_llm(llm, prompt: str, fallback_prompt: Optional[str] = None, max_retries: int = 2) -> str:
    current_prompt = prompt
    last_text = ""
    for attempt in range(max_retries + 1):
        try:
            result = llm.invoke(current_prompt)
            text = result.strip() if isinstance(result, str) else str(result).strip()
            last_text = text
            if text and "{" in text:
                return text
            if fallback_prompt and attempt == 0:
                current_prompt = fallback_prompt
            elif attempt < max_retries:
                current_prompt = f"{prompt}\n[INST] REMINDER: Return ONLY a valid JSON object starting with {{ and ending with }}. [/INST]"
            time.sleep(0.5)
        except Exception as e:
            logger.warning("LLM invocation attempt %d failed: %s", attempt + 1, e)
            if attempt < max_retries:
                time.sleep(1)
            else:
                return "{}"
    return last_text if (last_text and last_text.strip()) else "{}"


def _safe_parse_json(raw: str, fallback_key: str = "text") -> dict:
    if not raw or not raw.strip():
        return {}
    text = raw.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        extracted = json_match.group(0)
        cleaned = re.sub(r',\s*([}\]])', r'\1', extracted)
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    start_idx = text.find('{')
    if start_idx != -1:
        candidate = text[start_idx:]
        candidate = re.sub(r'```.*$', '', candidate, flags=re.DOTALL).strip()
        repaired = candidate
        unescaped_quotes = len(re.findall(r'(?<!\\)"', repaired))
        if unescaped_quotes % 2 != 0:
            repaired += '"'
        stack = []
        in_string = False
        escape = False
        for ch in repaired:
            if ch == '"' and not escape:
                in_string = not in_string
            elif not in_string:
                if ch in ('{', '['):
                    stack.append(ch)
                elif ch in ('}', ']'):
                    if stack:
                        top = stack[-1]
                        if (top == '{' and ch == '}') or (top == '[' and ch == ']'):
                            stack.pop()
            escape = (ch == '\\' and not escape)

        for op in reversed(stack):
            repaired += '}' if op == '{' else ']'
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        try:
            return json.loads(repaired)
        except Exception:
            pass

    result = {}
    for key in ["argument", "counter_argument", "verdict", "legal_holding", "reasoning", "reason"]:
        m = re.search(rf'"{key}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
        if m:
            result[key] = m.group(1).encode().decode('unicode_escape', errors='ignore')

    citations = []
    cite_matches = re.finditer(r'\{\s*"case_name"\s*:\s*"([^"]+)"(.*?)\}', text, re.DOTALL)
    for cm in cite_matches:
        full_cite = cm.group(0)
        try:
            citations.append(json.loads(full_cite))
        except Exception:
            c_data = {"case_name": cm.group(1)}
            court_m = re.search(r'"court"\s*:\s*"([^"]+)"', full_cite)
            if court_m:
                c_data["court"] = court_m.group(1)
            year_m = re.search(r'"year"\s*:\s*(\d+)', full_cite)
            if year_m:
                c_data["year"] = int(year_m.group(1))
            holding_m = re.search(r'"holding"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', full_cite)
            if holding_m:
                c_data["holding"] = holding_m.group(1)
            conf_m = re.search(r'"(?:self_confidence|ccs)"\s*:\s*([\d\.]+)', full_cite)
            if conf_m:
                c_data["self_confidence"] = float(conf_m.group(1))
            citations.append(c_data)

    if citations:
        result["citations"] = citations
        result["counter_citations"] = citations
        result["verified_citations"] = citations

    for field in ["overall_confidence", "challenge_strength", "confidence", "argument_quality"]:
        m = re.search(rf'"{field}"\s*:\s*([\d\.]+)', text)
        if m:
            try:
                result[field] = float(m.group(1))
            except Exception:
                pass

    for field in ["concede", "continue_debate"]:
        m = re.search(rf'"{field}"\s*:\s*(true|false)', text, re.IGNORECASE)
        if m:
            result[field] = m.group(1).lower() == "true"

    for field in ["challenges", "gaps", "targeted_queries"]:
        items = re.findall(rf'"{field}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if items:
            str_items = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', items[0])
            result[field] = str_items

    if result:
        return result
    return {fallback_key: raw}


def prosecutor_node(state: LexAgentState) -> dict:
    logger.info("Prosecutor node: generating argument for round %d", state.get("debate_round", 0))

    context = _format_passages(state.get("retrieved_passages", []))
    high_risk = state.get("high_risk_citations", [])

    user_content = USER_PROSECUTOR.format(
        query=state["query"],
        jurisdiction=state.get("jurisdiction", "US_Federal"),
        context=context,
        high_risk_citations=json.dumps(high_risk) if high_risk else "(none)"
    )
    prompt = format_mistral_prompt(SYSTEM_PROSECUTOR, user_content)
    fallback_prompt = format_mistral_prompt(
        "You are the PROSECUTOR in a legal debate. Respond ONLY in valid JSON.",
        f"Legal Question: {state['query']}\nProvide JSON with keys 'argument', 'citations', 'overall_confidence'."
    )

    raw = _invoke_llm(state["llm"], prompt, fallback_prompt=fallback_prompt)
    parsed = _safe_parse_json(raw, fallback_key="argument")

    citations = []
    for c in parsed.get("citations", []):
        try:
            case_name = _clean_case_name(c.get("case_name", ""))
            if not case_name:
                continue
            citations.append(CitationRecord(
                case_name=case_name,
                court=str(c.get("court", "") or ""),
                year=_safe_year(c.get("year")),
                holding=str(c.get("holding", "") or ""),
                self_confidence=float(c.get("self_confidence", 0.5) if c.get("self_confidence") is not None else 0.5),
                agent_source="prosecutor"
            ))
        except (ValueError, TypeError) as e:
            logger.warning("Skipping malformed citation: %s (%s)", c, e)

    history = list(state.get("debate_history", []))
    history.append({
        "round":     state.get("debate_round", 0),
        "role":      "prosecutor",
        "argument":  parsed.get("argument", ""),
        "citations": [c.to_dict() for c in citations],
        "overall_confidence": parsed.get("overall_confidence", 0.0)
    })

    return {
        "prosecutor_argument":  parsed.get("argument", ""),
        "prosecutor_citations": citations,
        "debate_history":       history,
    }
