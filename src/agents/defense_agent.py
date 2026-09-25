# LexAgent v3.0 | defense_agent.py
"""
Defense Agent — DVA+ Protocol with Evidence-Disjoint Adversarial Defense (Novelty 3).
Forces evidential independence between debating agents by executing a dedicated
counter-retrieval pass with Prosecutor's documents strictly excluded.
Computes Evidence Overlap Ratio (EOR) to monitor echo-chamber effects.
"""

from typing import List, Optional, Any, Set
from src.agents.state import LexAgentState
from src.agents.prompts import SYSTEM_DEFENSE, USER_DEFENSE, format_mistral_prompt
from src.agents.prosecutor_agent import (
    _format_passages, _invoke_llm, _safe_parse_json, _safe_year, _clean_case_name
)
from src.data.corpus_schema import CitationRecord, CorpusDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_counter_queries(query: str, prosecutor_citations: list) -> List[str]:
    """Generates deterministic counter-queries to attack Prosecutor claims."""
    counter_queries = []

    # 1. Limitation query
    counter_queries.append(f"{query} exception distinguished does not apply")

    # 2. Treatment queries (per cited case to find negative subsequent treatment / overruling)
    for c in prosecutor_citations[:3]:
        c_name = c.case_name if hasattr(c, "case_name") else c.get("case_name", "")
        if c_name:
            counter_queries.append(f"{c_name} overruled distinguished limited")

    # 3. Holding queries (per cited holding)
    for c in prosecutor_citations[:2]:
        h = c.holding if hasattr(c, "holding") else c.get("holding", "")
        if h and len(h) > 10:
            counter_queries.append(f"{h[:80]} limitation exception")

    return counter_queries


def _retrieve_disjoint_counter_evidence(
    hybrid_retriever: Any,
    queries: List[str],
    prosecutor_passages: List[CorpusDocument],
    jurisdiction: str = "US_Federal",
    top_k: int = 4
) -> List[CorpusDocument]:
    """Retrieves counter-evidence strictly excluding Prosecutor's documents."""
    if not hybrid_retriever or not queries:
        return []

    # Fingerprint Prosecutor passages to exclude
    pros_ids: Set[str] = set()
    pros_snippets: Set[str] = set()
    for doc in prosecutor_passages or []:
        d_id = getattr(doc, "doc_id", "") or (doc.get("doc_id", "") if isinstance(doc, dict) else "")
        if d_id:
            pros_ids.add(str(d_id))
        d_text = getattr(doc, "text", "") or (doc.get("text", "") if isinstance(doc, dict) else "")
        if d_text:
            pros_snippets.add(d_text[:80].strip().lower())

    disjoint_results = []
    seen_counter_ids = set()

    for q in queries:
        try:
            hits = hybrid_retriever.retrieve(q, jurisdiction)
            for hit in hits:
                h_id = getattr(hit, "doc_id", "") or (hit.get("doc_id", "") if isinstance(hit, dict) else "")
                h_text = getattr(hit, "text", "") or (hit.get("text", "") if isinstance(hit, dict) else "")
                h_snippet = h_text[:80].strip().lower()

                # Exclude if present in Prosecutor's evidence
                if h_id and h_id in pros_ids:
                    continue
                if h_snippet and h_snippet in pros_snippets:
                    continue
                if h_id and h_id in seen_counter_ids:
                    continue

                if h_id:
                    seen_counter_ids.add(h_id)
                disjoint_results.append(hit)
                if len(disjoint_results) >= top_k:
                    break
        except Exception as e:
            logger.debug("Counter-retrieval query failed for '%s': %s", q, e)

        if len(disjoint_results) >= top_k:
            break

    return disjoint_results[:top_k]


def _compute_evidence_overlap_ratio(
    defense_citations: List[CitationRecord],
    prosecutor_passages: List[CorpusDocument],
    counter_passages: List[CorpusDocument],
    prosecutor_citations: List[CitationRecord],
) -> float:
    """
    Computes Evidence Overlap Ratio (EOR):
      Share of Defense citations that come only from Prosecutor's evidence.
      1.0 = pure echo; 0.0 = complete evidential independence.
    """
    if not defense_citations:
        return 0.0

    pros_text = " ".join([
        f"{getattr(p, 'case_name', '')} {getattr(p, 'text', '')}"
        for p in (prosecutor_passages or [])
    ]).lower()
    for pc in (prosecutor_citations or []):
        c_name = pc.case_name if hasattr(pc, "case_name") else pc.get("case_name", "")
        pros_text += " " + str(c_name).lower()

    counter_text = " ".join([
        f"{getattr(p, 'case_name', '')} {getattr(p, 'text', '')}"
        for p in (counter_passages or [])
    ]).lower()

    echo_count = 0
    for dc in defense_citations:
        name = dc.case_name.lower().strip()
        in_pros = name in pros_text and len(name) > 3
        in_counter = name in counter_text and len(name) > 3

        # If cited only from Prosecutor's pool and not in counter-evidence, it's an echo
        if in_pros and not in_counter:
            echo_count += 1

    return round(echo_count / len(defense_citations), 4)


def make_defense_node(hybrid_retriever: Any = None):
    """Factory to create an evidence-disjoint Defense node with active counter-retrieval."""
    def defense_node(state: LexAgentState) -> dict:
        round_num = state.get("debate_round", 0)
        logger.info("Evidence-Disjoint Defense node: round %d", round_num)

        pros_passages = state.get("retrieved_passages", [])
        pros_cites = state.get("prosecutor_citations", [])

        # 1. Independent Counter-Evidence Retrieval (Prosecutor docs strictly excluded)
        retriever = hybrid_retriever or state.get("hybrid_retriever")
        counter_queries = _build_counter_queries(state["query"], pros_cites)
        counter_passages = _retrieve_disjoint_counter_evidence(
            retriever,
            counter_queries,
            pros_passages,
            jurisdiction=state.get("jurisdiction", "US_Federal"),
            top_k=4
        )

        shared_context = _format_passages(pros_passages)
        counter_context = _format_passages(counter_passages) if counter_passages else "(No independent counter-precedent found)"

        # Format Prosecutor citations
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
            context=shared_context,
            counter_context=counter_context,
        )
        prompt = format_mistral_prompt(SYSTEM_DEFENSE, user_content)
        fallback_prompt = format_mistral_prompt(
            "You are the DEFENSE ATTORNEY in a legal debate. Respond ONLY in valid JSON.",
            f"Challenge Prosecutor on: {state['query']} using counter-evidence.\nReturn JSON with 'challenges', 'counter_argument', 'counter_citations', 'challenge_strength', 'concede'."
        )

        raw = _invoke_llm(state["llm"], prompt, fallback_prompt=fallback_prompt)
        parsed = _safe_parse_json(raw, fallback_key="counter_argument")

        citations = []
        for c in parsed.get("counter_citations", []):
            try:
                case_name = _clean_case_name(c.get("case_name", ""))
                if not case_name:
                    continue
                rep_cite = str(c.get("reporter_cite", "") or c.get("citation", "") or "")
                citations.append(CitationRecord(
                    case_name=case_name,
                    court=str(c.get("court", "") or ""),
                    year=_safe_year(c.get("year")),
                    holding=str(c.get("holding", "") or ""),
                    self_confidence=float(c.get("self_confidence", 0.5) if c.get("self_confidence") is not None else 0.5),
                    agent_source="defense",
                    reporter_cite=rep_cite,
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

        # 2. Compute Evidence Overlap Ratio (EOR)
        eor = _compute_evidence_overlap_ratio(citations, pros_passages, counter_passages, pros_cites)
        logger.info("Defense round %d: Evidence Overlap Ratio (EOR) = %.4f", round_num, eor)

        history = list(state.get("debate_history", []))
        history.append({
            "round":                  round_num,
            "role":                   "defense",
            "challenges":             challenges,
            "counter_argument":       counter_arg,
            "counter_citations":      [c.to_dict() for c in citations],
            "challenge_strength":     challenge_strength,
            "evidence_overlap_ratio": eor,
            "concede":                concede,
        })

        return {
            "defense_argument":         counter_arg,
            "defense_citations":        citations,
            "defense_challenges":       challenges,
            "challenge_strength":       challenge_strength,
            "defense_concede":          concede,
            "defense_counter_passages": counter_passages,
            "evidence_overlap_ratio":   eor,
            "debate_history":           history,
        }
    return defense_node


# Default standalone node for backward-compatibility
defense_node = make_defense_node(hybrid_retriever=None)
