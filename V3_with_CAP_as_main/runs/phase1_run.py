# LexAgent v3.0 | phase1_run.py
"""Phase 1 Smoke Test: Hybrid Retrieval verification over primary CAP database."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.bootstrap import load_lexagent_components


def run_phase1_test(query: str = "Does the Fourteenth Amendment protect abortion privacy?"):
    print("\n" + "=" * 70)
    print("  PHASE 1 SMOKE TEST: CAP HYBRID RETRIEVAL (Dense + BM25)")
    print("=" * 70)
    print(f"  Query: '{query}'")

    components = load_lexagent_components(load_llm=False)
    hybrid = components["hybrid"]

    results = hybrid.retrieve(query)
    print(f"\n  Retrieved {len(results)} authority passages from CAP:")
    for idx, doc in enumerate(results, 1):
        case = doc.case_name or "(unnamed)"
        vol = f"{doc.volume} {doc.reporter or 'U.S.'} {doc.first_page}" if doc.volume and doc.first_page else ""
        print(f"  [{idx}] {case} {vol}: {doc.text[:120]}...")

    entity_map = hybrid.get_entity_map(results)
    print(f"\n  Extracted Entity Map ({len(entity_map)} keys): {list(entity_map.keys())[:5]}")
    print("=" * 70)
    print("  PHASE 1 SMOKE TEST PASSED!")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Does the Fourteenth Amendment protect abortion privacy?")
    args = parser.parse_args()
    run_phase1_test(args.query)
