# LexAgent v3.0 | run_cce_evidence_study.py
"""
Evidence-selection study for the Citation Confidence Engine (no LLM).

The citation checker accepts few correct citations. This diagnostic asks whether that is caused
by WHICH passages of the opinion the entailment model reads, or by the entailment model itself.
For every correct citation and every citation with a swapped holding from the stress test, the
citation is resolved to its opinion as usual, and the support step is repeated with different
ways of choosing the passages of that opinion:

    dense    the passages nearest to the holding in embedding space (what the checker does)
    lexical  the passages with the highest BM25 score for the holding, within the opinion
    hybrid   reciprocal-rank fusion of the two
    whole    every passage of the opinion

A good selection should raise the acceptance of correct citations without raising the acceptance
of swapped holdings.

    python experiments/run_cce_evidence_study.py
"""

import argparse
import json
import os
import re
import sys
import time

import chromadb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DRIVE_CHROMA_CORPUS_PATH,
    KNOWN_CASES_PATH,
    LOCAL_CHROMA_CORPUS_PATH,
    RESULTS_DIR,
    RRF_K,
)
from experiments.benchmark_cases import get_benchmark_cases
from src.data.corpus_schema import CitationRecord
from src.evaluation.stress_test import build_stress_test_dataset
from src.models.load_model import load_embedder
from src.novel.cce import CitationConfidenceEngine
from src.utils.chroma_sync import ensure_chroma_ready

# (name, ranking, number of passages; None = every passage of the opinion)
STRATEGIES = [
    ("dense_top3", "dense", 3),
    ("dense_top5", "dense", 5),
    ("lexical_top3", "lexical", 3),
    ("lexical_top5", "lexical", 5),
    ("hybrid_top3", "hybrid", 3),
    ("hybrid_top5", "hybrid", 5),
    ("whole_opinion", "whole", None),
]
WINDOW_CHARS = 450
WINDOW_STEP = 350


def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def opinion_passages(collection, doc_id: str) -> list:
    """Every passage of one opinion, in reading order, with its embedding."""
    got = collection.get(where={"parent_doc_id": {"$eq": doc_id}}, include=["documents", "metadatas", "embeddings"])
    rows = [
        {"id": pid, "text": doc or "", "index": int((meta or {}).get("chunk_index", 0)), "embedding": emb}
        for pid, doc, meta, emb in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])
    ]
    return sorted(rows, key=lambda row: row["index"])


def dense_order(passages: list, holding_vector) -> list:
    import numpy as np

    matrix = np.asarray([p["embedding"] for p in passages], dtype="float32")
    query = np.asarray(holding_vector, dtype="float32")
    scores = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-12)
    return list(np.argsort(-scores))


def lexical_order(passages: list, holding: str) -> list:
    from rank_bm25 import BM25Okapi

    scores = BM25Okapi([_tokens(p["text"]) for p in passages]).get_scores(_tokens(holding))
    return sorted(range(len(passages)), key=lambda i: -scores[i])


def fused_order(*orders: list) -> list:
    fused = {}
    for order in orders:
        for rank, index in enumerate(order, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(fused, key=lambda i: -fused[i])


def support_score(cce: CitationConfidenceEngine, text: str, holding: str) -> tuple:
    """Entailment of the holding by the text: the checker's own windowing, over all of the text."""
    import numpy as np

    windows = [text[i:i + WINDOW_CHARS] for i in range(0, len(text), WINDOW_STEP)] or [text]
    logits = np.asarray(cce.nli_model.predict([[w, holding] for w in windows], batch_size=128, show_progress_bar=False))
    if logits.ndim == 1:
        return float(logits.max()), len(windows)
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exps / exps.sum(axis=1, keepdims=True)
    return float(probabilities[:, cce._get_entailment_index()].max()), len(windows)


def main(output_file: str = "cce_evidence_selection_study.json", limit: int = 100) -> dict:
    print("\n" + "=" * 85)
    print("  LEXAGENT v3.0: CITATION CHECKER - EVIDENCE SELECTION STUDY")
    print("=" * 85)

    embedder = load_embedder()
    active_path = ensure_chroma_ready(LOCAL_CHROMA_CORPUS_PATH, DRIVE_CHROMA_CORPUS_PATH)
    client = chromadb.PersistentClient(path=active_path)
    names = [c.name for c in client.list_collections()]
    collection = client.get_or_create_collection("cap_authorities" if ("cap_authorities" in names or not names) else names[0])
    cce = CitationConfidenceEngine(collection, embedder, KNOWN_CASES_PATH, policy="legacy")
    if cce.catalog is None or cce.nli_model is None:
        raise RuntimeError("The study needs the case catalog and the NLI model; build the database first.")

    cases = get_benchmark_cases(limit)
    items = [item for item in build_stress_test_dataset(cases, []) if item["type"] in ("correct", "misattributed")]
    print(f"  Citations: {sum(1 for i in items if i['type'] == 'correct')} correct, "
          f"{sum(1 for i in items if i['type'] == 'misattributed')} with a swapped holding")

    started = time.time()
    counts = {name: {"correct": 0, "misattributed": 0, "windows": 0} for name, _, _ in STRATEGIES}
    deployed = {"correct": 0, "misattributed": 0}
    totals = {"correct": 0, "misattributed": 0}
    unresolved = []
    details = []

    for position, item in enumerate(items, start=1):
        record = CitationRecord(case_name=item["case_name"], court="Supreme Court", year=1990,
                                holding=item["holding"], self_confidence=0.85, agent_source="evidence_study")
        record.reporter_cite = item.get("reporter_cite", "")
        kind = item["type"]
        totals[kind] += 1

        # The checker exactly as deployed, for reference.
        if cce.compute_ccs(record)["tier"] == "SUPPORTED":
            deployed[kind] += 1

        _, matched_name, _ = cce._compute_existence(record)
        resolution = cce._resolve_opinion(record, matched_name)
        report = resolution.report if resolution is not None else None
        if report is None or not report.is_opinion:
            unresolved.append(item["case_name"])
            continue

        passages = opinion_passages(collection, report.doc_id)
        if not passages:
            unresolved.append(item["case_name"])
            continue
        dense = dense_order(passages, embedder.encode(item["holding"]))
        lexical = lexical_order(passages, item["holding"])
        orders = {"dense": dense, "lexical": lexical, "hybrid": fused_order(dense, lexical),
                  "whole": list(range(len(passages)))}

        row = {"type": kind, "case_name": item["case_name"], "opinion": f"{report.name}, {report.cite}", "passages": len(passages)}
        for name, ranking, top in STRATEGIES:
            chosen = orders[ranking] if top is None else orders[ranking][:top]
            text = " ".join(passages[i]["text"] for i in chosen)
            score, windows = support_score(cce, text, item["holding"])
            counts[name]["windows"] += windows
            if score >= cce.tau:
                counts[name][kind] += 1
            row[name] = round(score, 4)
        details.append(row)
        if position % 24 == 0 or position == len(items):
            print(f"  [{position:3d}/{len(items)}] scored")

    def summary(accepted: dict, windows: int = None) -> dict:
        out = {
            "correct_accepted": accepted["correct"], "correct_total": totals["correct"],
            "swapped_accepted": accepted["misattributed"], "swapped_total": totals["misattributed"],
        }
        if windows is not None:
            out["mean_windows_scored"] = round(windows / max(1, len(details)), 1)
        return out

    results = {
        "acceptance_threshold": cce.tau,
        "window_chars": WINDOW_CHARS,
        "window_step": WINDOW_STEP,
        "citations": len(items),
        "unresolved": sorted(set(unresolved)),
        "deployed_checker": summary(deployed),
        "strategies": {name: summary(counts[name], counts[name]["windows"]) for name, _, _ in STRATEGIES},
        "elapsed_seconds": round(time.time() - started, 1),
        "details": details,
    }

    print("\n" + "-" * 85)
    print(f"  {'Passages read by the entailment model':<40} | {'Correct accepted':<17} | {'Swapped accepted':<17} | Windows")
    print("-" * 85)
    d = results["deployed_checker"]
    print(f"  {'checker as deployed':<40} | {d['correct_accepted']:>3}/{d['correct_total']:<13} | {d['swapped_accepted']:>3}/{d['swapped_total']:<13} |")
    for name, _, _ in STRATEGIES:
        s = results["strategies"][name]
        print(f"  {name:<40} | {s['correct_accepted']:>3}/{s['correct_total']:<13} | {s['swapped_accepted']:>3}/{s['swapped_total']:<13} | {s['mean_windows_scored']}")
    print("-" * 85)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, output_file)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nEvidence study saved to: {out_path}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Citation checker: evidence selection study")
    parser.add_argument("--output", default="cce_evidence_selection_study.json", help="Output JSON filename")
    parser.add_argument("--limit", type=int, default=100, help="Benchmark cases to draw citations from (default: 100)")
    arguments = parser.parse_args()
    main(output_file=arguments.output, limit=arguments.limit)
