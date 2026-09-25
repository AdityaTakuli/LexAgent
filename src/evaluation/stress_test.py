# LexAgent v3.0 | stress_test.py
"""
Citation Stress-Test Protocol (Novelty 5).
A controlled, zero-LLM diagnostic suite that feeds 6 distinct classes of labelled
citations directly into the Citation Confidence Engine (CCE) verifier:
  1. Correct: Target case + true holding (Expected: SUPPORTED)
  2. Misattributed: Target case + holding from different case (Expected: MISATTRIBUTED)
  3. CaseHOLD distractor: Target case + plausible hard negative holding (Expected: MISATTRIBUTED)
  4. Fabricated with cite: Invented case name with cite in covered volume (Expected: FABRICATED)
  5. Fabricated without cite: Invented case name with no cite (Expected: UNVERIFIABLE)
  6. Real off-corpus: Real landmark case outside covered volumes (Expected: UNVERIFIABLE)
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional

from src.data.corpus_schema import CitationRecord
from src.novel.cce import CitationConfidenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Real landmark cases that are NOT in the 27 Landmark CAP Volumes (Volumes 347, 410, 585, etc.)
OFF_CORPUS_LANDMARK_CASES = [
    {
        "case_name": "Marbury v. Madison",
        "reporter_cite": "5 U.S. 137",
        "year": 1803,
        "holding": "It is emphatically the province and duty of the judicial department to say what the law is, establishing judicial review.",
    },
    {
        "case_name": "McCulloch v. Maryland",
        "reporter_cite": "17 U.S. 316",
        "year": 1819,
        "holding": "Congress has implied powers under the Necessary and Proper Clause to create the Second Bank of the United States.",
    },
    {
        "case_name": "Gibbons v. Ogden",
        "reporter_cite": "22 U.S. 1",
        "year": 1824,
        "holding": "The Commerce Clause of the Constitution grants the federal government the power to regulate interstate commerce.",
    },
    {
        "case_name": "Loving v. Virginia",
        "reporter_cite": "388 U.S. 1",
        "year": 1967,
        "holding": "State anti-miscegenation statutes prohibiting interracial marriage violate both the Equal Protection and Due Process Clauses.",
    },
    {
        "case_name": "Tinker v. Des Moines Independent Community School District",
        "reporter_cite": "393 U.S. 503",
        "year": 1969,
        "holding": "Students do not shed their constitutional rights to freedom of speech or expression at the schoolhouse gate.",
    },
]

# Hard negative holdings (distractors)
HARD_NEGATIVE_DISTRACTORS = [
    "Police officers may conduct a full warrantless search of any arrestee's home without probable cause.",
    "The Eighth Amendment permits mandatory death sentences for all non-violent property crimes committed by minors.",
    "State courts are not required to provide legal counsel to indigent criminal defendants in capital trials.",
    "Prosecutors have no constitutional duty to disclose exculpatory evidence requested by defense counsel.",
    "Private citizens are strictly subject to Fourth Amendment warrant requirements when investigating neighbors.",
]

# Fabricated case names
INVENTED_PARTY_NAMES = [
    ("Consolidated Steel Logistics v. Apex Transport", "410 U.S. 995"),
    ("United Maritime Carriers v. Federal Commerce Commission", "347 U.S. 981"),
    ("Quantum Biometrics Corp v. State Department of Revenue", "585 U.S. 991"),
    ("Meridian National Bank v. Global Mining Enterprises", "372 U.S. 988"),
    ("Vanguard Telecommunications v. Pacific Rail Authorities", "384 U.S. 977"),
]

INVENTED_NO_CITE_NAMES = [
    "CyberCorp Dynamics v. Hyperion Aerospace",
    "OmniGlobal Media Group v. Vertex Holdings",
    "Atlas Artificial Intelligence Inc v. State of Columbia",
    "Pinnacle Pharmaceutical Consortium v. United Health Alliance",
    "Nexus Robotics v. Pacific Coast Harbor Commission",
]


def build_stress_test_dataset(benchmark_cases: list) -> List[Dict[str, Any]]:
    """Builds synthetic labelled test dataset with 6 controlled citation types."""
    dataset = []

    # 1. Correct Citations (Target case + true holding)
    for case in benchmark_cases[:10]:
        target = case.get("target_case", "")
        holding = case.get("ground_truth_holding", "")
        cite = case.get("reporter_citation", "")
        if target and holding:
            dataset.append({
                "type": "correct",
                "label": "SUPPORTED",
                "case_name": target,
                "reporter_cite": cite,
                "holding": holding,
                "description": f"True holding for {target}",
            })

    # 2. Misattributed Citations (Target case + holding from different benchmark case)
    n_cases = min(10, len(benchmark_cases))
    for i in range(n_cases):
        target = benchmark_cases[i].get("target_case", "")
        cite = benchmark_cases[i].get("reporter_citation", "")
        swapped_case = benchmark_cases[(i + 5) % n_cases]
        swapped_holding = swapped_case.get("ground_truth_holding", "")
        if target and swapped_holding:
            dataset.append({
                "type": "misattributed",
                "label": "REJECT",
                "expected_verdicts": ["MISATTRIBUTED", "UNCERTAIN"],
                "case_name": target,
                "reporter_cite": cite,
                "holding": swapped_holding,
                "description": f"{target} paired with holding of {swapped_case.get('target_case', '')}",
            })

    # 3. CaseHOLD Distractor Citations (Target case + hard negative distractor)
    for i, distractor in enumerate(HARD_NEGATIVE_DISTRACTORS):
        if i < len(benchmark_cases):
            target = benchmark_cases[i].get("target_case", "")
            cite = benchmark_cases[i].get("reporter_citation", "")
            dataset.append({
                "type": "casehold_distractor",
                "label": "REJECT",
                "expected_verdicts": ["MISATTRIBUTED", "UNCERTAIN"],
                "case_name": target,
                "reporter_cite": cite,
                "holding": distractor,
                "description": f"{target} paired with negative distractor",
            })

    # 4. Fabricated With Citation (Invented name + reporter cite in covered volume)
    for name, cite in INVENTED_PARTY_NAMES:
        dataset.append({
            "type": "fabricated_with_cite",
            "label": "FABRICATED",
            "expected_verdicts": ["FABRICATED"],
            "case_name": name,
            "reporter_cite": cite,
            "holding": "Invented proposition regarding regulatory compliance.",
            "description": f"Fabricated case in covered volume: {cite}",
        })

    # 5. Fabricated Without Citation (Invented name, no reporter cite)
    for name in INVENTED_NO_CITE_NAMES:
        dataset.append({
            "type": "fabricated_no_cite",
            "label": "UNVERIFIABLE",
            "expected_verdicts": ["UNVERIFIABLE"],
            "case_name": name,
            "reporter_cite": "",
            "holding": "Unsubstantiated statutory interpretation.",
            "description": f"Fabricated case with no reporter volume: {name}",
        })

    # 6. Real Off-Corpus Landmark Cases (Real cases outside covered volumes)
    for off in OFF_CORPUS_LANDMARK_CASES:
        dataset.append({
            "type": "real_off_corpus",
            "label": "UNVERIFIABLE",
            "expected_verdicts": ["UNVERIFIABLE"],
            "case_name": off["case_name"],
            "reporter_cite": off["reporter_cite"],
            "holding": off["holding"],
            "description": f"Real landmark case outside covered corpus: {off['case_name']}",
        })

    return dataset


def run_citation_stress_test(cce: CitationConfidenceEngine, benchmark_cases: list) -> dict:
    """
    Executes the Citation Stress-Test Protocol.
    Feeds synthetic labelled citations straight into factorized CCE and reports
    per-type catch rates, false acceptance rate, and power.
    """
    logger.info("Executing Citation Stress-Test Protocol across 6 citation classes...")
    start_time = time.time()

    dataset = build_stress_test_dataset(benchmark_cases)
    per_type_counts: Dict[str, Dict[str, int]] = {}
    detailed_results = []

    for item in dataset:
        c_type = item["type"]
        if c_type not in per_type_counts:
            per_type_counts[c_type] = {
                "total": 0, "accepted": 0, "rejected": 0,
                "SUPPORTED": 0, "UNCERTAIN": 0, "MISATTRIBUTED": 0,
                "FABRICATED": 0, "UNVERIFIABLE": 0,
            }

        rec = CitationRecord(
            case_name=item["case_name"],
            court="Supreme Court",
            year=1990,
            holding=item["holding"],
            self_confidence=0.85,
            agent_source="stress_test",
        )
        rec.reporter_cite = item.get("reporter_cite", "")

        res = cce.compute_ccs(rec)
        tier = res["tier"]
        ccs = res["ccs"]
        accepted = (tier == "SUPPORTED")

        counts = per_type_counts[c_type]
        counts["total"] += 1
        if accepted:
            counts["accepted"] += 1
        else:
            counts["rejected"] += 1

        counts[tier] = counts.get(tier, 0) + 1

        detailed_results.append({
            "type": c_type,
            "case_name": item["case_name"],
            "reporter_cite": item.get("reporter_cite", ""),
            "expected_label": item["label"],
            "verdict": tier,
            "ccs": ccs,
            "existence": res.get("existence", 0.0),
            "support": res.get("support", 0.0),
            "accepted_by_judge": accepted,
        })

    elapsed = time.time() - start_time

    # Key Metrics
    correct_counts = per_type_counts.get("correct", {"total": 0, "accepted": 0})
    power = (correct_counts["accepted"] / correct_counts["total"]) if correct_counts["total"] > 0 else 0.0

    misattributed_counts = per_type_counts.get("misattributed", {"total": 0, "accepted": 0})
    far_misattributed = (misattributed_counts["accepted"] / misattributed_counts["total"]) if misattributed_counts["total"] > 0 else 0.0

    casehold_counts = per_type_counts.get("casehold_distractor", {"total": 0, "accepted": 0})
    far_casehold = (casehold_counts["accepted"] / casehold_counts["total"]) if casehold_counts["total"] > 0 else 0.0

    fab_with_cite = per_type_counts.get("fabricated_with_cite", {"total": 0, "FABRICATED": 0})
    fab_catch_rate = (fab_with_cite.get("FABRICATED", 0) / fab_with_cite["total"]) if fab_with_cite["total"] > 0 else 0.0

    off_corpus = per_type_counts.get("real_off_corpus", {"total": 0, "FABRICATED": 0})
    false_fab_rate = (off_corpus.get("FABRICATED", 0) / off_corpus["total"]) if off_corpus["total"] > 0 else 0.0

    summary_metrics = {
        "total_evaluated": len(dataset),
        "power": round(power, 4),
        "false_acceptance_rate_misattributed": round(far_misattributed, 4),
        "false_acceptance_rate_casehold": round(far_casehold, 4),
        "fabrication_catch_rate": round(fab_catch_rate, 4),
        "false_fabrication_rate": round(false_fab_rate, 4),
        "elapsed_seconds": round(elapsed, 2),
        "per_type_counts": per_type_counts,
        "details": detailed_results,
    }

    return summary_metrics
