# LexAgent v3.0 | experiments/v2/e3_memory.py
"""
E3 — The verdict cache on a workload that has repeats (no model generates anything here).

The benchmark has no repeated question, so it cannot show what a cache is for. This experiment takes
the verdicts stored by one full pass of the repaired system (namespace v2_e3_memory_pass1, written by
the queue) and asks, for the old rule (embedding cosine > 0.92) and the new rule (same question after
normalisation), what each would return for:

  first pass        the 96 distinct questions in order, the store filling as it goes
                    -> every hit is an answer written for another question
  exact repeats     the same 96 questions asked again
  format variants   the same questions with different case, spacing and punctuation
  paraphrases       the same legal question in other words (experiments/v2/runner.write_paraphrases)

A hit is correct only if the stored answer was written for the same benchmark case. The new rule is
also exercised through AdaptiveMemory itself, to check that the code does what the simulation says.

    python experiments/v2/e3_memory.py
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

from config import MEMORY_CACHE_THRESHOLD, RESULTS_DIR  # noqa: E402

V2_DIR = os.path.join(RESULTS_DIR, "v2")
NAMESPACE = "v2_e3_memory_pass1"


def variant_of(question: str) -> str:
    """The same question with different case, spacing and punctuation."""
    return "  " + question.upper().replace(",", " ,").replace("?", " ??") + "  "


def main(root: str = V2_DIR, parts: dict = None) -> dict:
    from experiments.v2.runner import benchmark_cases, load_components
    from src.novel.adaptive_memory import AdaptiveMemory, normalize_question

    directory = os.path.join(root, "e3_memory")
    os.makedirs(directory, exist_ok=True)
    paraphrase_path = os.path.join(directory, "paraphrases.json")
    if not os.path.exists(paraphrase_path):
        print("  [e3] paraphrases.json is missing; run the queue first")
        return {}
    if parts is None:
        parts = load_components(load_llm=False)
    embedder = parts["embedder"]
    cases, digest = benchmark_cases()
    memory = AdaptiveMemory(namespace=NAMESPACE, cache_mode="exact")
    stored = memory.verdict_collection.get(include=["metadatas", "embeddings"])
    by_question = {normalize_question(c["query"]): c["id"] for c in cases}
    stored_ids = [by_question.get(normalize_question(m.get("query", ""))) for m in stored["metadatas"]]
    if not stored_ids:
        print("  [e3] no stored verdicts in the namespace yet")
        return {}
    matrix = np.asarray(stored["embeddings"], dtype="float64")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    stored_set = {i for i in stored_ids if i is not None}
    position = {c["id"]: n for n, c in enumerate(cases)}
    paraphrases = {r["case_id"]: r["paraphrase"] for r in json.load(open(paraphrase_path, encoding="utf-8"))["rows"]}

    def embed(texts):
        vectors = np.asarray(embedder.encode(texts, show_progress_bar=False), dtype="float64")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def similarity_rule(vector, allowed):
        """The old rule: nearest stored question, if its cosine exceeds the threshold."""
        candidates = [n for n, case_id in enumerate(stored_ids) if case_id in allowed]
        if not candidates:
            return None
        scores = matrix[candidates] @ vector
        best = int(np.argmax(scores))
        return stored_ids[candidates[best]] if scores[best] > MEMORY_CACHE_THRESHOLD else None

    def exact_rule(text, allowed):
        case_id = by_question.get(normalize_question(text))
        return case_id if case_id in allowed else None

    def tally(name, texts, owners, allowed_for):
        vectors = embed(texts)
        out = {"queries": len(texts), "with_a_stored_answer_for_the_same_case": sum(1 for owner in owners if owner in allowed_for(owner))}
        for rule in ("similarity_0.92", "exact"):
            correct = wrong = 0
            for text, vector, owner in zip(texts, vectors, owners):
                allowed = allowed_for(owner)
                returned = similarity_rule(vector, allowed) if rule.startswith("similarity") else exact_rule(text, allowed)
                if returned is None:
                    continue
                if returned == owner:
                    correct += 1
                else:
                    wrong += 1
            hits = correct + wrong
            out[rule] = {"hits": hits, "correct": correct, "wrong_question": wrong,
                         "precision": round(correct / hits, 4) if hits else None,
                         "recall_on_stored": round(correct / out["with_a_stored_answer_for_the_same_case"], 4) if out["with_a_stored_answer_for_the_same_case"] else None}
        return name, out

    ids = [c["id"] for c in cases]
    questions = [c["query"] for c in cases]
    earlier = lambda owner: {i for i in stored_set if position[i] < position[owner]}   # what the store held when this question arrived
    everything = lambda owner: stored_set
    workloads = dict([
        tally("first_pass_distinct_questions", questions, ids, earlier),
        tally("exact_repeats", questions, ids, everything),
        tally("format_variants", [variant_of(q) for q in questions], ids, everything),
        tally("paraphrases", [paraphrases.get(i, "") for i in ids], ids, everything),
    ])

    # The same through the real code path, new rule.
    code = {"exact_repeats": 0, "format_variants": 0, "paraphrases": 0, "wrong": 0}
    for case in cases:
        for key, text in (("exact_repeats", case["query"]), ("format_variants", variant_of(case["query"])), ("paraphrases", paraphrases.get(case["id"], ""))):
            verdict = memory.lookup_verdict(text, embedder)
            if verdict:
                code[key] += 1
                if by_question.get(normalize_question(verdict.get("query", ""))) != case["id"]:
                    code["wrong"] += 1

    # Could the earlier verdicts be reused with what they rested on?
    restored = sum(1 for m in stored["metadatas"] if m.get("citations") not in (None, "", "[]"))
    pass1_path = os.path.join(root, "e4_repaired_Full", "seed0", "cases.jsonl")
    not_found = None
    if os.path.exists(pass1_path):
        rows = [json.loads(line) for line in open(pass1_path, encoding="utf-8") if line.strip()]
        not_found = sum(1 for r in rows for rnd in (r.get("trace") or {}).get("rounds", []) for v in (rnd.get("citation_check") or {}).values()
                        if v.get("existence_verdict") == "NOT_FOUND")
    result = {
        "benchmark_sha256": digest, "namespace": NAMESPACE, "verdicts_stored_in_pass_one": len(stored_ids),
        "stored_verdicts_that_carry_their_citations": restored, "similarity_threshold": MEMORY_CACHE_THRESHOLD,
        "workloads": workloads, "new_rule_through_AdaptiveMemory": code,
        "simulation_agrees_with_code": all(code[k] == workloads[k]["exact"]["hits"] for k in ("exact_repeats", "format_variants", "paraphrases")) and code["wrong"] == 0,
        "re_hallucination": {"citations_not_found_in_pass_one": not_found,
                             "measurable": bool(not_found),
                             "note": "The re-hallucination rate needs citations that failed the existence check in the first pass. "
                                     "With none, there is nothing to repeat and the rate is undefined, not zero."},
    }
    with open(os.path.join(directory, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    lines = ["# E3 — verdict cache on a workload with repeats", "",
             f"Verdicts stored by one full pass: {len(stored_ids)} of {len(cases)} questions. "
             f"A hit is correct only if the stored answer was written for the same benchmark case.", "",
             "| Workload | Queries | Old rule (cosine > 0.92): hits / correct / wrong question | New rule (same question): hits / correct / wrong question |", "|---|---|---|---|"]
    for name, row in workloads.items():
        old, new = row["similarity_0.92"], row["exact"]
        lines.append(f"| {name.replace('_', ' ')} | {row['queries']} | {old['hits']} / {old['correct']} / {old['wrong_question']} | {new['hits']} / {new['correct']} / {new['wrong_question']} |")
    lines += ["", f"New rule through `AdaptiveMemory.lookup_verdict`: {code}; agrees with the simulation: {result['simulation_agrees_with_code']}.",
              f"Citations that failed the existence check in the first pass: {not_found}. {result['re_hallucination']['note']}", ""]
    with open(os.path.join(directory, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"  [e3] wrote {os.path.join(directory, 'results.json')}")
    return result


if __name__ == "__main__":
    main()
