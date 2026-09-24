# LexAgent v3.0 | defense_agent.py
"""Defense Agent — DVA+ Protocol."""

from src.agents.state import LexAgentState
from src.agents.prompts import SYSTEM_DEFENSE, USER_DEFENSE, format_mistral_prompt
from src.agents.prosecutor_agent import (
    _format_passages, _invoke_llm, _safe_parse_json, _safe_year, _clean_case_name
)
from src.data.corpus_schema import CitationRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


def defense_node(state: LexAgentState) -> dict:
    logger.info("Defense node: challenging Prosecutor for round %d",
                state.get("debate_round", 0))

    context = _format_passages(state.get("retrieved_passages", []))

    pros_cites = state.get("prosecutor_citations", [])
    if pros_cites:
        pros_cites_str = "\n".join([
            f"- {c.case_name} ({c.year}): {c.holding[:200]} [conf={c.self_confidence:.2f}]"
            for c in pros_cites
        ])
    else:
        pros_cites_str = "(No citations provided by prosecutor)"

    user_content = USER_DEFENSE.format(
        query=state["query"],
        prosecutor_argument=state.get("prosecutor_argument", "(No argument provided)"),
        prosecutor_citations=pros_cites_str,
        context=context,
    )
    prompt = format_mistral_prompt(SYSTEM_DEFENSE, user_content)
    fallback_prompt = format_mistral_prompt(
        "You are the DEFENSE ATTORNEY in a legal debate. Respond ONLY in valid JSON.",
        f"Challenge Prosecutor on: {state['query']}.\nReturn JSON with 'challenges', 'counter_argument', 'counter_citations', 'challenge_strength', 'concede'."
    )

    raw = _invoke_llm(state["llm"], prompt, fallback_prompt=fallback_prompt)
    parsed = _safe_parse_json(raw, fallback_key="counter_argument")

    citations = []
    for c in parsed.get("counter_citations", []):
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
                agent_source="defense"
            ))
        except (ValueError, TypeError) as e:
            logger.warning("Skipping malformed defense citation: %s (%s)", c, e)

    challenge_strength = 0.0
    try:
        challenge_strength = float(parsed.get("challenge_strength", 0.0))
        challenge_strength = min(max(challenge_strength, 0.0), 1.0)
    except (ValueError, TypeError):
        pass

    concede = bool(parsed.get("concede", False))
    challenges = parsed.get("challenges", [])
    if isinstance(challenges, str):
        challenges = [challenges]

    counter_arg = parsed.get("counter_argument", "")

    history = list(state.get("debate_history", []))
    history.append({
        "round":              state.get("debate_round", 0),
        "role":               "defense",
        "challenges":         challenges,
        "counter_argument":   counter_arg,
        "counter_citations":  [c.to_dict() for c in citations],
        "challenge_strength": challenge_strength,
        "concede":            concede,
    })

    return {
        "defense_argument":   counter_arg,
        "defense_citations":  citations,
        "defense_challenges": challenges,
        "challenge_strength": challenge_strength,
        "defense_concede":    concede,
        "debate_history":     history,
    }
