# LexAgent v3.0 | citation_accuracy.py
"""Citation Accuracy Score (CAS) — Semantic verification against CaseHOLD."""

import os
import sys

try:
    from sentence_transformers import util
except ImportError:
    util = None

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
        fuzz = FuzzFallback()

from config import CAS_SEMANTIC_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_cas(predictions: list, ground_truth_data: list, embedder) -> dict:
    """
    Computes Answer Semantic Similarity (ASS) [formerly CAS],
    plus Authority Recall (AR), Authority Precision (AP), and Holding Attribution Accuracy (HAA).
    """
    correct_ass = 0
    total_ass = 0
    sim_scores = []

    total_citations = 0
    matched_target_citations = 0
    supported_citations = 0

    if not util or not embedder:
        return {
            "ass": 0.0,
            "cas": 0.0,
            "ap": 0.0,
            "ar": 0.0,
            "haa": 0.0,
            "n_correct": 0,
            "n_total": len(predictions),
        }

    for pred, gt in zip(predictions, ground_truth_data):
        pred_holding = pred.get("answer", "")
        gt_holding = gt.get("answer", gt.get("ground_truth_holding", gt.get("holding_0", "")))
        if not gt_holding and "endings" in gt and "label" in gt:
            try:
                gt_holding = gt["endings"][int(gt["label"])]
            except (IndexError, TypeError, ValueError):
                gt_holding = ""

        if pred_holding and gt_holding:
            total_ass += 1
            p_vec = embedder.encode(pred_holding[:500], convert_to_tensor=True)
            g_vec = embedder.encode(gt_holding[:500], convert_to_tensor=True)
            sim = float(util.cos_sim(p_vec, g_vec))
            sim_scores.append(sim)
            if sim > CAS_SEMANTIC_THRESHOLD:
                correct_ass += 1

        # Authority Precision & Recall
        target_case = gt.get("target_case", "").lower().strip()
        cites = pred.get("citations", []) or pred.get("verified_citations", []) or []
        found_target = False
        for c in cites:
            total_citations += 1
            c_name = (c.get("case_name", "") if isinstance(c, dict) else getattr(c, "case_name", "")).lower().strip()
            if target_case and fuzz.ratio(c_name, target_case) >= 80:
                matched_target_citations += 1
                found_target = True

            tier = c.get("tier", "") if isinstance(c, dict) else getattr(c, "tier", "")
            if tier == "SUPPORTED":
                supported_citations += 1

    ass = round(sum(sim_scores) / len(sim_scores), 4) if sim_scores else 0.0
    cas = round(correct_ass / total_ass, 4) if total_ass > 0 else 0.0
    ap = round(matched_target_citations / total_citations, 4) if total_citations > 0 else 0.0
    ar = round(matched_target_citations / len(ground_truth_data), 4) if ground_truth_data else 0.0
    haa = round(supported_citations / total_citations, 4) if total_citations > 0 else 0.0

    return {
        "ass": ass,
        "cas": cas,  # Kept as alias for backward compatibility
        "ap": ap,
        "ar": ar,
        "haa": haa,
        "n_correct": correct_ass,
        "n_total": total_ass,
    }
