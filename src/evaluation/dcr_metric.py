# LexAgent v3.0 | dcr_metric.py
"""Debate Convergence Rate (DCR) Metric."""

def compute_dcr(lexagent_results: list) -> dict:
    total = len(lexagent_results)
    if total == 0:
        return {
            "dcr": 0.0,
            "n_converged": 0,
            "n_total": 0,
            "interpretation": "No debates to evaluate",
        }

    converged = 0
    for r in lexagent_results:
        if r.get("defense_concede", False):
            converged += 1
            continue

        judge_verified = [
            c.get("case_name", "").lower()
            for c in r.get("judge_verified_citations", [])
            if isinstance(c, dict)
        ]
        pros_cites = [
            c.get("case_name", "").lower()
            for c in r.get("prosecutor_citations", [])
            if isinstance(c, dict)
        ]

        if judge_verified and all(c in pros_cites for c in judge_verified):
            converged += 1

    dcr = round(converged / total, 4)
    return {
        "dcr": dcr,
        "n_converged": converged,
        "n_total": total,
        "interpretation": f"{converged}/{total} debates converged ({dcr:.2%})",
    }
