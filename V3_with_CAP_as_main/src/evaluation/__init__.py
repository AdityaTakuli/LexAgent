from .hallucination_rate import compute_hallucination_rate
from .citation_accuracy import compute_cas
from .dcr_metric import compute_dcr
from .rhr_metric import compute_rhr, DEFAULT_PARAPHRASE_PAIRS
from .stress_test import run_citation_stress_test, build_stress_test_dataset
from .evaluator import LexAgentEvaluator

__all__ = [
    "compute_hallucination_rate",
    "compute_cas",
    "compute_dcr",
    "compute_rhr",
    "DEFAULT_PARAPHRASE_PAIRS",
    "run_citation_stress_test",
    "build_stress_test_dataset",
    "LexAgentEvaluator",
]
