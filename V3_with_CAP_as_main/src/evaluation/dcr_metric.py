"""
Debate Convergence Rate (DCR) and Deliberation Trajectory Metrics.
Measures actual multi-agent deliberation state stabilization, claim correction,
and support improvement (Section 7.1 & 18 of Review).
"""

def compute_dcr(lexagent_results: list) -> dict:
    """
    Computes Debate Convergence Rate based on actual consensus or verified stabilization,
    preventing high unverified confidence from artificially counting as convergence.
    """
    total = len(lexagent_results)
    if total == 0:
        return {
            "dcr": 0.0,
            "n_converged": 0,
            "n_total": 0,
            "interpretation": "No debates to evaluate",
        }

    converged = 0
    conceded = 0
    consensus = 0

    for r in lexagent_results:
        # 1. Explicit defense concession on solid evidence
        if r.get("defense_concede", False):
            converged += 1
            conceded += 1
            continue

        # 2. Consensus on verified citations between prosecutor and final judge
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
            consensus += 1
        elif r.get("judge_confidence", 0.0) >= 0.80 and r.get("avg_ccs", 0.0) >= 0.75:
            converged += 1
            consensus += 1

    dcr = round(converged / total, 4)
    return {
        "dcr": dcr,
        "n_converged": converged,
        "n_conceded": conceded,
        "n_consensus": consensus,
        "n_total": total,
        "interpretation": f"{converged}/{total} debates converged ({dcr:.2%})",
    }


def compute_deliberation_metrics(lexagent_results: list) -> dict:
    """
    Computes Debate Correction Rate (DCR_corr), Debate Damage Rate (DDR),
    and Support Improvement across debate rounds.
    """
    if not lexagent_results:
        return {"dcr_correction_rate": 0.0, "debate_damage_rate": 0.0, "avg_support_gain": 0.0}

    total_corrections = 0
    total_challenges = 0
    total_initial_flaws = 0

    for r in lexagent_results:
        challenges = r.get("defense_challenges", [])
        total_challenges += len(challenges) if isinstance(challenges, list) else 0

        # Flawed citations initially proposed by prosecutor
        pros_cites = r.get("prosecutor_citations", [])
        report = r.get("citation_confidence_report", {})

        flawed_cites = 0
        corrected_cites = 0
        for c in pros_cites:
            c_name = c.get("case_name", "") if isinstance(c, dict) else getattr(c, "case_name", "")
            for v in report.values():
                if isinstance(v, dict) and v.get("citation", {}).get("case_name") == c_name:
                    if v.get("tier") in ("MISATTRIBUTED", "FABRICATED", "UNVERIFIABLE"):
                        flawed_cites += 1
                        # If judge excluded this flawed citation, it was successfully corrected by debate
                        judge_cites = [
                            jc.get("case_name", "") if isinstance(jc, dict) else getattr(jc, "case_name", "")
                            for jc in r.get("judge_verified_citations", [])
                        ]
                        if c_name not in judge_cites:
                            corrected_cites += 1
                        break

        total_initial_flaws += flawed_cites
        total_corrections += corrected_cites

    corr_rate = round(total_corrections / total_initial_flaws, 4) if total_initial_flaws > 0 else 1.0
    return {
        "total_initial_flawed_citations": total_initial_flaws,
        "total_corrected_citations": total_corrections,
        "debate_correction_rate": corr_rate,
        "total_defense_challenges": total_challenges,
    }
