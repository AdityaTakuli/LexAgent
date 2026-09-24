# LexAgent v3.0 | hallucination_rate.py
"""Hallucination Rate (HR) Evaluation Metric."""

import os
import sys
from typing import List

try:
    from fuzzywuzzy import fuzz
except ImportError:
    try:
        from thefuzz import fuzz
    except ImportError:
        import difflib
        class FuzzFallback:
            @staticmethod
            def ratio(s1, s2):
                return int(difflib.SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio() * 100)
            @staticmethod
            def token_set_ratio(s1, s2):
                return FuzzFallback.ratio(s1, s2)
        fuzz = FuzzFallback()

from config import FUZZY_MATCH_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_hallucination_rate(
    predictions: List[dict],
    known_cases: set,
) -> dict:
    total = 0
    fake = 0
    fake_list = []

    normalized_known = {c.lower().strip() for c in known_cases if c}

    for pred in predictions:
        citations = pred.get("citations", []) or pred.get("verified_citations", []) or []
        for c in citations:
            total += 1
            if isinstance(c, dict):
                case_name = c.get("case_name", "")
            else:
                case_name = getattr(c, "case_name", "")

            name_lower = case_name.lower().strip()
            if not name_lower:
                fake += 1
                fake_list.append({"case_name": "(empty)", "best_match": "", "best_ratio": 0})
                continue

            if name_lower in normalized_known:
                continue

            best_ratio = 0
            best_match = ""
            for kc in normalized_known:
                r = fuzz.ratio(name_lower, kc)
                if r > best_ratio:
                    best_ratio = r
                    best_match = kc
                if best_ratio >= FUZZY_MATCH_THRESHOLD:
                    break

            if best_ratio < FUZZY_MATCH_THRESHOLD:
                fake += 1
                fake_list.append({
                    "case_name": case_name,
                    "best_match": best_match,
                    "best_ratio": best_ratio,
                })

    hr = round(fake / total, 4) if total > 0 else 0.0
    logger.info("HR computed: %.4f (%d/%d fake citations)", hr, fake, total)
    return {
        "hr": hr,
        "n_fake": fake,
        "n_total": total,
        "fake_citations": fake_list,
    }
