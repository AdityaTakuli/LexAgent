# LexAgent v3.0 | reflection_agent.py
"""Reflection Agent: Gap identification and targeted retrieval."""

from src.agents.state import LexAgentState
from src.agents.prompts import SYSTEM_REFLECTION, USER_REFLECTION, format_mistral_prompt
from src.agents.prosecutor_agent import _invoke_llm, _safe_parse_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_cce_summary(state: LexAgentState) -> str:
    report = state.get("citation_confidence_report", {})
    if not report:
        all_cites = (state.get("prosecutor_citations", []) +
                     state.get("defense_citations", []))
        return f"{len(all_cites)} citations made. CCE scores pending."

    lines = []
    for cid, data in report.items():
        ccs = data.get("ccs", 0.0)
        tier = data.get("tier", "UNKNOWN")
        lines.append(f"  {cid}: CCS={ccs:.2f} [{tier}]")

    avg_ccs = state.get("avg_ccs", 0.5)
    header = f"Average CCS: {avg_ccs:.2f} | {len(report)} citations scored"
    return header + "\n" + "\n".join(lines)


def reflection_node(state: LexAgentState) -> dict:
    logger.info("Reflection node: analyzing debate quality for round %d",
                state.get("debate_round", 0))

    cce_summary = _build_cce_summary(state)

    user_content = USER_REFLECTION.format(
        query=state["query"],
        prosecutor_argument=state.get("prosecutor_argument", "(No argument)"),
        defense_argument=state.get("defense_argument", "(No argument)"),
        cce_report_summary=cce_summary,
    )
    prompt = format_mistral_prompt(SYSTEM_REFLECTION, user_content)
    fallback_prompt = format_mistral_prompt(
        "You are the REFLECTION AGENT in a legal debate. Respond ONLY in valid JSON.",
        f"Assess debate on question: {state['query']}.\nProvide JSON with 'gaps', 'targeted_queries', 'argument_quality', 'continue_debate', 'reason'."
    )

    raw = _invoke_llm(state["llm"], prompt, fallback_prompt=fallback_prompt)
    parsed = _safe_parse_json(raw, fallback_key="gaps")

    gaps = parsed.get("gaps", [])
    if isinstance(gaps, str):
        gaps = [gaps] if gaps.strip() else []

    queries = parsed.get("targeted_queries", [])
    if isinstance(queries, str):
        queries = [queries] if queries.strip() else []

    quality = 0.5
    try:
        quality = float(parsed.get("argument_quality", 0.5))
        quality = min(max(quality, 0.0), 1.0)
    except (ValueError, TypeError):
        pass

    history = list(state.get("debate_history", []))
    history.append({
        "round":            state.get("debate_round", 0),
        "role":             "reflection",
        "gaps":             gaps,
        "targeted_queries": queries,
        "quality":          quality,
        "continue_debate":  parsed.get("continue_debate", False),
        "reason":           parsed.get("reason", ""),
    })

    return {
        "reflection_gaps":    gaps,
        "reflection_queries": queries,
        "reflection_score":   quality,
        "debate_history":     history,
    }
