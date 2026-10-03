# LexAgent v3.0 | run_stress_test.py
"""
Citation Stress-Test Runner (Novelty 5).
Executes the zero-LLM diagnostic stress test over 6 controlled citation classes.
Outputs the headline paper table: per error type, what fraction the verifier accepts/rejects.
"""

import os
import sys
import json
import argparse
import pickle
import chromadb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    LOCAL_CHROMA_CORPUS_PATH,
    DRIVE_CHROMA_CORPUS_PATH,
    KNOWN_CASES_PATH,
    RESULTS_DIR,
)
from src.utils.chroma_sync import ensure_chroma_ready
from src.models.load_model import load_embedder
from src.novel.cce import CitationConfidenceEngine
from src.evaluation.stress_test import run_citation_stress_test
from experiments.benchmark_cases import BENCHMARK_30_CASES


def main(output_file: str = "citation_stress_test_results.json"):
    print("\n" + "=" * 85)
    print("  LEXAGENT v3.0: CITATION STRESS-TEST PROTOCOL (Novelty 5)")
    print("  Controlled Diagnostic Verification Across 6 Error Types")
    print("=" * 85)

    if not os.path.exists(KNOWN_CASES_PATH):
        raise FileNotFoundError(f"Catalog not found at {KNOWN_CASES_PATH}. Please build DB first.")

    embedder = load_embedder()
    active_path = ensure_chroma_ready(LOCAL_CHROMA_CORPUS_PATH, DRIVE_CHROMA_CORPUS_PATH)
    client = chromadb.PersistentClient(path=active_path)
    colls = [c.name for c in client.list_collections()]
    collection_name = "cap_authorities" if ("cap_authorities" in colls or not colls) else colls[0]
    collection = client.get_or_create_collection(collection_name)

    cce = CitationConfidenceEngine(collection, embedder, KNOWN_CASES_PATH)

    results = run_citation_stress_test(cce, BENCHMARK_30_CASES)

    print("\n" + "─" * 85)
    print(f"  {'Citation Type':<25} | {'Total':<6} | {'Accepted':<9} | {'Rejected':<9} | {'Verdict Breakdown'}")
    print("─" * 85)

    for c_type, counts in results["per_type_counts"].items():
        breakdown_str = (
            f"SUP:{counts.get('SUPPORTED', 0)} "
            f"UNC:{counts.get('UNCERTAIN', 0)} "
            f"MIS:{counts.get('MISATTRIBUTED', 0)} "
            f"FAB:{counts.get('FABRICATED', 0)} "
            f"UNV:{counts.get('UNVERIFIABLE', 0)}"
        )
        print(f"  {c_type:<25} | {counts['total']:<6} | {counts['accepted']:<9} | {counts['rejected']:<9} | {breakdown_str}")

    print("─" * 85)
    print("\n  SUMMARY DIAGNOSTIC METRICS:")
    print(f"  - Power (Correct Citations Accepted):              {results['power']:.2%}")
    print(f"  - False-Acceptance Rate on Misattributed Citations: {results['false_acceptance_rate_misattributed']:.2%}")
    print(f"  - False-Acceptance Rate on CaseHOLD Distractors:   {results['false_acceptance_rate_casehold']:.2%}")
    print(f"  - Fabrication Catch Rate (Invented Cases):         {results['fabrication_catch_rate']:.2%}")
    print(f"  - False-Fabrication Rate (Off-Corpus Cases):        {results['false_fabrication_rate']:.2%}")
    print(f"  - Execution Time:                                  {results['elapsed_seconds']}s")
    print("=" * 85)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed stress test results saved to: {out_path}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Citation Stress Test Runner")
    parser.add_argument("--output", default="citation_stress_test_results.json", help="Output JSON filename")
    args = parser.parse_args()
    main(args.output)
