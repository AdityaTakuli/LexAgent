# LexAgent v3.0 | citation_accuracy.py
"""Citation Accuracy Score (CAS) — Semantic verification against CaseHOLD."""

import os
import sys

try:
    from sentence_transformers import util
except ImportError:
    util = None

from config import CAS_SEMANTIC_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_cas(predictions: list, ground_truth_data: list, embedder) -> dict:
    correct = 0
    total = 0

    if not util or not embedder:
        return {"cas": 0.0, "n_correct": 0, "n_total": len(predictions)}

    for pred, gt in zip(predictions, ground_truth_data):
        pred_holding = pred.get("answer", "")
        gt_holding = gt.get("answer", gt.get("holding_0", ""))
        if not gt_holding and "endings" in gt and "label" in gt:
            try:
                gt_holding = gt["endings"][int(gt["label"])]
            except (IndexError, TypeError, ValueError):
                gt_holding = ""

        if not pred_holding or not gt_holding:
            continue

        total += 1
        p_vec = embedder.encode(pred_holding, convert_to_tensor=True)
        g_vec = embedder.encode(gt_holding, convert_to_tensor=True)

        if float(util.cos_sim(p_vec, g_vec)) > CAS_SEMANTIC_THRESHOLD:
            correct += 1

    cas = round(correct / total, 4) if total > 0 else 0.0
    return {"cas": cas, "n_correct": correct, "n_total": total}
