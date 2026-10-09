# LexAgent v3.0 | experiments/v2/e2b_near_miss_probe.py
"""
E2b — A harder question for the support model that passed the gate (post-hoc; not part of the gate).

The gate (E2) was decided on the labelled citations that exist: a correct holding against the
holding of an unrelated case, and against blatantly false statements. Those wrong citations are
easy: an opinion on searches says nothing about free speech. A real misattribution is a near miss:
a real case cited for a proposition from the same area of law that it did not decide.

This probe builds near misses without new labels: each benchmark case is paired with the reference
holding of the *most similar other* benchmark case (cosine of holding embeddings). Some of these are
partly true of the cited case - opinions restate the holdings of related cases - so the acceptance
rate is an upper bound on the error, not the error itself. It is reported next to the gate, and the
gate decision is not changed by it.

    python experiments/v2/e2b_near_miss_probe.py
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

from config import RESULTS_DIR  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "v2", "e2_verifier_validation", "near_miss_probe.json")


def main():
    from experiments.v2.runner import benchmark_cases, checker, load_components
    from src.data.corpus_schema import CitationRecord

    parts = load_components(load_llm=False)
    cases, digest = benchmark_cases()
    cce = checker(parts, "gated")
    if not cce.gate.get("validated"):
        print("the gate is closed; nothing to probe")
        return
    holdings = np.asarray(parts["embedder"].encode([c.get("ground_truth_holding", "") for c in cases], show_progress_bar=False), dtype="float64")
    holdings /= np.linalg.norm(holdings, axis=1, keepdims=True)
    similarity = holdings @ holdings.T
    np.fill_diagonal(similarity, -1.0)
    far = max(1, len(cases) // 2)

    def verdict(case, holding):
        record = CitationRecord(case_name=case["target_case"], court="", year=0, holding=holding, agent_source="probe")
        record.reporter_cite = case.get("reporter_citation", "") or ""
        return cce.compute_ccs(record)

    rows = []
    for index, case in enumerate(cases):
        neighbour = int(np.argmax(similarity[index]))
        correct = verdict(case, case["ground_truth_holding"])
        unrelated = verdict(case, cases[(index + far) % len(cases)]["ground_truth_holding"])
        near = verdict(case, cases[neighbour]["ground_truth_holding"])
        rows.append({"case_id": case["id"], "cited_case": case["target_case"], "near_miss_holding_from": cases[neighbour]["target_case"],
                     "holding_similarity": round(float(similarity[index, neighbour]), 4),
                     "correct": correct["support_verdict"], "correct_score": correct.get("support_score"),
                     "unrelated": unrelated["support_verdict"], "unrelated_score": unrelated.get("support_score"),
                     "near_miss": near["support_verdict"], "near_miss_score": near.get("support_score")})

    def accepted(key):
        return sum(1 for r in rows if r[key] == "SUPPORTED")

    result = {
        "benchmark_sha256": digest, "support_model": cce.support_model_name, "gate": cce.gate, "citations_per_kind": len(rows),
        "accepted": {"correct_holding": accepted("correct"), "holding_of_an_unrelated_case": accepted("unrelated"),
                     "holding_of_the_most_similar_other_case": accepted("near_miss")},
        "contradicted": {k: sum(1 for r in rows if r[k] == "CONTRADICTED") for k in ("correct", "unrelated", "near_miss")},
        "reading": "The near-miss rate is an upper bound on misattribution error: an opinion often restates the holding of a closely "
                   "related case, so some of these acceptances are true. It shows how far the verifier can be trusted to tell a case's "
                   "own holding from a neighbouring one.",
        "rows": rows,
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print("accepted (of %d each):" % len(rows), result["accepted"])
    print("contradicted:", result["contradicted"])
    examples = [r for r in rows if r["near_miss"] == "SUPPORTED"][:8]
    for r in examples:
        print(f"   near miss accepted: {r['cited_case'][:34]} <- holding of {r['near_miss_holding_from'][:34]} (score {r['near_miss_score']}, similarity {r['holding_similarity']})")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
