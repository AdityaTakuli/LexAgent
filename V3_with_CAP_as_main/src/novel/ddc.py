# LexAgent v3.0 | ddc.py
"""Dynamic Debate Controller (DDC) — Adaptive debate termination mechanism."""


from config import (
    DDC_MAX_ROUNDS, DDC_CCS_CONTINUE_THRESH, DDC_MIN_GAPS_TO_CONTINUE,
    DDC_CHALLENGE_STRENGTH_THRESH,
)
from src.agents.state import LexAgentState, debate_protocol
from src.novel.ledger import has_reliable_citation
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DynamicDebateController:
    """Citation-quality-driven debate termination."""

    def decide(self, state: LexAgentState) -> tuple:
        avg_ccs         = state.get("avg_ccs", 1.0)
        gaps            = state.get("reflection_gaps", [])
        challenge_str   = state.get("challenge_strength", 0.0)
        current_round   = state.get("debate_round", 0)
        defense_concede = state.get("defense_concede", False)

        use_defense     = state.get("use_defense", True)
        use_cce         = state.get("use_cce", True)

        # 1. Hard stops
        if use_defense and defense_concede:
            logger.info("DDC: Defense conceded — TERMINATE")
            return "TERMINATE", "Defense conceded — Prosecutor argument accepted"

        if current_round >= DDC_MAX_ROUNDS - 1:
            logger.info("DDC: Max rounds (%d) reached — TERMINATE", DDC_MAX_ROUNDS)
            return "TERMINATE", f"Maximum rounds ({DDC_MAX_ROUNDS}) reached"

        if use_cce and debate_protocol(state) == "ledger":
            return self._decide_from_ledger(state, gaps, current_round)

        # 2. Extract min_ccs to prevent bad citations from hiding behind averages
        report = state.get("citation_confidence_report", {})
        ccs_vals = [v.get("ccs", 1.0) for v in report.values() if isinstance(v, dict) and "ccs" in v]
        min_ccs = min(ccs_vals) if ccs_vals else avg_ccs

        # 3. Decision conditions
        weak_citations   = use_cce and ((avg_ccs < DDC_CCS_CONTINUE_THRESH) or (min_ccs < 0.40))
        has_gaps         = len(gaps) >= DDC_MIN_GAPS_TO_CONTINUE
        strong_challenge = use_defense and (challenge_str > DDC_CHALLENGE_STRENGTH_THRESH)

        if weak_citations and has_gaps:
            reason = (
                f"Round {current_round + 1}: avg_CCS={avg_ccs:.2f}, min_CCS={min_ccs:.2f} with "
                f"{len(gaps)} unresolved gap(s). Initiating next debate round."
            )
            logger.info("DDC: CONTINUE — %s", reason)
            return "CONTINUE", reason

        if strong_challenge and has_gaps:
            reason = (
                f"Defense challenge_strength={challenge_str:.2f} with {len(gaps)} gaps. Additional evidence needed."
            )
            logger.info("DDC: CONTINUE — %s", reason)
            return "CONTINUE", reason

        if weak_citations:
            reason = (
                f"Insufficient verified evidence: avg_CCS={avg_ccs:.2f}; "
                f"no usable reflection gap was produced in round {current_round}."
            )
            logger.info("DDC: TERMINATE — %s", reason)
            return "TERMINATE", reason

        reason = f"Deliberation concluded: avg_CCS={avg_ccs:.2f}, gaps={len(gaps)}, round={current_round}"
        logger.info("DDC: TERMINATE — %s", reason)
        return "TERMINATE", reason

    def _decide_from_ledger(self, state: LexAgentState, gaps: list, current_round: int) -> tuple:
        """Ledger protocol: another round is worth its cost only while the debate has nothing to
        rest an answer on, something is still open, and the last round brought something new.

        Two conditions of the earlier rule are gone, for the reasons the audit measured: the mean
        and minimum score (a supported citation next to an unsupported one no longer forces a
        round), and the defense's self-reported challenge strength (it was above the threshold in
        all 288 rounds of the 96 debates examined, 0.8 in 287 of them, so it decided nothing).
        """
        supported = has_reliable_citation(state.get("citation_ledger"))
        has_gaps = len(gaps) >= DDC_MIN_GAPS_TO_CONTINUE
        new_passages = int(state.get("round_new_passages", 0) or 0)
        new_supported = int(state.get("round_new_supported", 0) or 0)
        progress = (new_passages + new_supported) > 0
        facts = (f"round {current_round + 1}: supported citation in the ledger={'yes' if supported else 'no'}, "
                 f"gaps={len(gaps)}, new passages={new_passages}, newly supported={new_supported}")
        if supported:
            reason = f"A supported citation exists; no further round is needed ({facts})"
        elif not has_gaps:
            reason = f"Nothing supported and no open gap to search for ({facts})"
        elif not progress:
            reason = f"Nothing supported and the last round brought nothing new ({facts})"
        else:
            reason = f"No supported citation yet and a gap remains ({facts}). Initiating next debate round."
            logger.info("DDC: CONTINUE — %s", reason)
            return "CONTINUE", reason
        logger.info("DDC: TERMINATE — %s", reason)
        return "TERMINATE", reason



def make_ddc_node(ddc: DynamicDebateController):
    def ddc_node(state: LexAgentState) -> dict:
        decision, reason = ddc.decide(state)
        updates = {
            "ddc_decision": decision,
            "ddc_reason":   reason,
        }
        if decision == "CONTINUE":
            updates["debate_round"] = state.get("debate_round", 0) + 1
        return updates
    return ddc_node
