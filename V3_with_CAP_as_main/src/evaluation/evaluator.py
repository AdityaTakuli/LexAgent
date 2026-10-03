# LexAgent v3.0 | evaluator.py
"""Master Evaluator for LexAgent v3.0."""

import os
import sys

from src.evaluation.hallucination_rate import compute_hallucination_rate
from src.evaluation.citation_accuracy import compute_cas
from src.evaluation.dcr_metric import compute_dcr
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LexAgentEvaluator:
    """Master evaluator orchestrating all metrics."""

    def __init__(self, known_cases: set, casehold_data: list, embedder):
        self.known_cases = known_cases
        self.casehold_data = casehold_data
        self.embedder = embedder
        logger.info("LexAgentEvaluator initialized (%d known cases, %d CaseHOLD samples)",
                    len(known_cases), len(casehold_data))

    def evaluate(self, method_name: str, predictions: list) -> dict:
        hr = compute_hallucination_rate(predictions, self.known_cases)
        cas = compute_cas(predictions, self.casehold_data[:len(predictions)], self.embedder)
        return {
            "method": method_name,
            "hr": hr["hr"],
            "cfr": hr.get("cfr", 0.0),
            "oor": hr.get("oor", 0.0),
            "mar": hr.get("mar", 0.0),
            "ucr": hr.get("ucr", 0.0),
            "lhr": hr.get("lhr", 0.0),
            "ass": cas.get("ass", 0.0),
            "cas": cas["cas"],
            "ap": cas.get("ap", 0.0),
            "ar": cas.get("ar", 0.0),
            "haa": cas.get("haa", 0.0),
            "n_fake": hr["n_fake"],
            "n_citations": hr["n_total"],
            "n_correct_holdings": cas["n_correct"],
        }

    def evaluate_lexagent(self, results: list) -> dict:
        predictions = []
        for r in results:
            predictions.append({
                "answer": r.get("final_answer", ""),
                "citations": [
                    {"case_name": c.get("case_name", ""), "tier": c.get("tier", "")}
                    for c in r.get("judge_verified_citations", [])
                    if isinstance(c, dict)
                ],
                "citation_confidence_report": r.get("citation_confidence_report", {}),
            })

        base_metrics = self.evaluate("LexAgent_v3_DVA+", predictions)
        dcr = compute_dcr(results)

        base_metrics["dcr"] = dcr["dcr"]
        base_metrics["dcr_converged"] = dcr["n_converged"]
        base_metrics["dcr_total"] = dcr["n_total"]
        return base_metrics
