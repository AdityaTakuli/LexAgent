# LexAgent v3.0 | ddc.py
"""Dynamic Debate Controller (DDC) — Adaptive debate termination mechanism."""

import os
import sys

from config import (
    DDC_MAX_ROUNDS, DDC_CCS_CONTINUE_THRESH, DDC_MIN_GAPS_TO_CONTINUE,
    DDC_CHALLENGE_STRENGTH_THRESH,
)
from src.agents.state import LexAgentState
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

        # Hard stops
        if defense_concede:
            logger.info("DDC: Defense conceded — TERMINATE")
            return "TERMINATE", "Defense conceded — Prosecutor argument accepted"

        if current_round >= DDC_MAX_ROUNDS - 1:
            logger.info("DDC: Max rounds (%d) reached — TERMINATE", DDC_MAX_ROUNDS)
            return "TERMINATE", f"Maximum rounds ({DDC_MAX_ROUNDS}) reached"

        # CONTINUE conditions
        weak_citations   = avg_ccs < DDC_CCS_CONTINUE_THRESH
        has_gaps         = len(gaps) >= DDC_MIN_GAPS_TO_CONTINUE
        strong_challenge = challenge_str > DDC_CHALLENGE_STRENGTH_THRESH

        if weak_citations and has_gaps:
            reason = (
                f"Round {current_round + 1}: avg_CCS={avg_ccs:.2f} < "
                f"{DDC_CCS_CONTINUE_THRESH} with {len(gaps)} unresolved gap(s). Initiating Round 2."
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

        reason = f"Sufficient evidence: avg_CCS={avg_ccs:.2f}, gaps={len(gaps)}, round={current_round}"
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
