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


# Known landmark Supreme Court cases outside typical local sub-slices (for off-corpus identification)
OFF_CORPUS_SCOTUS_REGISTRY = {
    "lockett v. ohio", "argersinger v. hamlin", "milliken v. bradley",
    "moran v. burbine", "gertz v. robert welch, inc.", "arkansas v. sanders",
    "california v. acevedo", "city of arlington v. fed. commc'ns comm'n",
    "bellotti v. baird", "first national bank v. bellotti", "milkovich v. lorain journal co.",
    "g.m. leasing corp. v. united states", "kolender v. lawson",
    "marbury v. madison", "mcculloch v. maryland", "gibbons v. ogden",
    "loving v. virginia", "tinker v. des moines", "new york v. quarles",
}


def compute_hallucination_rate(
    predictions: List[dict],
    known_cases: set,
) -> dict:
    total = 0
    fake = 0
    fabricated = 0
    off_corpus = 0
    misattributed = 0
    unsupported = 0
    fake_list = []

    normalized_known = {c.lower().strip() for c in known_cases if c}

    for pred in predictions:
        citations = pred.get("citations", []) or pred.get("verified_citations", []) or []
        cce_report = pred.get("citation_confidence_report", {})

        for c in citations:
            total += 1
            if isinstance(c, dict):
                case_name = c.get("case_name", "")
                tier = c.get("tier", "")
            else:
                case_name = getattr(c, "case_name", "")
                tier = getattr(c, "tier", "")

            name_lower = case_name.lower().strip()
            if not name_lower:
                fake += 1
                fabricated += 1
                fake_list.append({"case_name": "(empty)", "best_match": "", "best_ratio": 0, "type": "FABRICATED"})
                continue

            if tier == "MISATTRIBUTED":
                misattributed += 1

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
                # Distinguish genuine off-corpus SCOTUS authority from pure invention
                is_scotus = any(fuzz.ratio(name_lower, sc) >= 80 for sc in OFF_CORPUS_SCOTUS_REGISTRY)
                c_type = "OFF_CORPUS" if is_scotus else "FABRICATED"
                if is_scotus:
                    off_corpus += 1
                else:
                    fabricated += 1

                fake_list.append({
                    "case_name": case_name,
                    "best_match": best_match,
                    "best_ratio": best_ratio,
                    "type": c_type,
                })

        # Count misattributions from CCE report if available
        if isinstance(cce_report, dict):
            for v in cce_report.values():
                if isinstance(v, dict):
                    t = v.get("tier", "")
                    if t == "MISATTRIBUTED" and misattributed == 0:
                        misattributed += 1
                    elif t == "UNVERIFIABLE" and unsupported == 0:
                        unsupported += 1

    hr = round(fake / total, 4) if total > 0 else 0.0
    cfr = round(fabricated / total, 4) if total > 0 else 0.0
    oor = round(off_corpus / total, 4) if total > 0 else 0.0
    mar = round(misattributed / total, 4) if total > 0 else 0.0
    ucr = round(unsupported / total, 4) if total > 0 else 0.0
    lhr = round((fabricated + misattributed + unsupported) / total, 4) if total > 0 else 0.0

    logger.info("Decomposed Hallucination Rate: HR=%.4f (CFR=%.4f, OOR=%.4f, MAR=%.4f, UCR=%.4f, LHR=%.4f) across %d citations",
                hr, cfr, oor, mar, ucr, lhr, total)
    return {
        "hr": hr,
        "cfr": cfr,
        "oor": oor,
        "mar": mar,
        "ucr": ucr,
        "lhr": lhr,
        "n_fake": fake,
        "n_fabricated": fabricated,
        "n_off_corpus": off_corpus,
        "n_misattributed": misattributed,
        "n_unsupported": unsupported,
        "n_total": total,
        "fake_citations": fake_list,
    }
