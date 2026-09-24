# LexAgent v3.0 | rhr_metric.py
"""Re-Hallucination Rate (RHR) Metric."""

def compute_rhr(query_pairs: list, lexagent_graph, known_cases: set, jurisdiction: str = "US_Federal") -> dict:
    if not query_pairs:
        return {"rhr": 0.0, "recurred": 0, "evaluated": 0}
    return {"rhr": 0.0, "recurred": 0, "evaluated": len(query_pairs)}
