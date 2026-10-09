# LexAgent v3.0 | experiments/v2/e2_verifier_validation.py
"""
E2 — Can any support model tell a correct holding from a wrong one? (the gate of repair 4)

The citation checker's support verdict may steer the judge, the controller and the memory only if a
support model passes this test. The rule is fixed before the run:

  data      the labelled citations of the stress test that name a real case: 96 correct, 96 with the
            holding of another case, 20 with a hand-written false holding
  split     benchmark cases are shuffled once (seed 42) into 48 development and 48 test cases; a
            citation follows the case it names
  select    the candidate with the highest development AUC (correct against all wrong citations);
            its threshold is the one that maximises (true-accept rate - false-accept rate) on development
  gate      opens only if that one candidate reaches AUC >= config.CCE_GATE_MIN_AUC on the test split

Candidates: NLI cross-encoders of increasing size, scored on 450-character windows or on whole
passages; three ways of choosing passages from the cited opinion; and the generator itself asked a
yes/no question over each passage. No debate is run.

Writes  data/results/v2/e2_verifier_validation/results.json   every candidate, development and test
        data/cce_support_gate.json                             the decision the checker reads

    python experiments/v2/e2_verifier_validation.py
"""

import contextlib
import io
import json
import os
import random
import re
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

import config  # noqa: E402
from config import CCE_GATE_MIN_AUC, CCE_GATE_PATH, KNOWN_CASES_PATH, RESULTS_DIR, RRF_K  # noqa: E402

OUT_DIR = os.path.join(RESULTS_DIR, "v2", "e2_verifier_validation")
NLI_MODELS = [
    "cross-encoder/nli-deberta-v3-small",   # the model the checker has used so far
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-large",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "tasksource/deberta-base-long-nli",     # trained with long premises
]
GENERATOR = "generator-yes-no"
SELECTIONS = {"dense_top3": ("dense", 3), "hybrid_top3": ("hybrid", 3), "hybrid_top5": ("hybrid", 5)}
SPLIT_SEED = 42
WINDOW, STEP = 450, 350


def auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    positive, negative = np.asarray(positive), np.asarray(negative)
    wins = (positive[:, None] > negative[None, :]).sum() + 0.5 * (positive[:, None] == negative[None, :]).sum()
    return float(wins / (len(positive) * len(negative)))


def auc_interval(positive, negative, resamples=2000, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    positive, negative = np.asarray(positive), np.asarray(negative)
    values = [auc(rng.choice(positive, len(positive)), rng.choice(negative, len(negative))) for _ in range(resamples)]
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def best_threshold(positive, negative):
    """Threshold with the largest (true-accept rate - false-accept rate)."""
    candidates = sorted(set(np.round(np.concatenate([positive, negative]), 4)))
    scored = [(float((np.asarray(positive) >= t).mean() - (np.asarray(negative) >= t).mean()), float(t)) for t in candidates]
    return max(scored)[1] if scored else 0.5


def softmax(rows):
    rows = np.asarray(rows, dtype="float64")
    e = np.exp(rows - rows.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def prepare_items(parts):
    """Every labelled citation with the passages of the opinion it names and their rankings."""
    from rank_bm25 import BM25Okapi

    from experiments.v2.runner import benchmark_cases, checker
    from src.data.corpus_schema import CitationRecord
    from src.evaluation.stress_test import build_stress_test_dataset

    cases, digest = benchmark_cases()
    cce = checker(parts, "legacy")
    order = [c["id"] for c in cases]
    random.Random(SPLIT_SEED).shuffle(order)
    development = set(order[: len(order) // 2])
    dataset = [i for i in build_stress_test_dataset(cases, []) if i["type"] in ("correct", "misattributed", "casehold_distractor")]
    seen = {"correct": 0, "misattributed": 0, "casehold_distractor": 0}
    tokens = lambda text: re.findall(r"[a-z0-9]+", text.lower())
    items, cache = [], {}
    for entry in dataset:
        case = cases[seen[entry["type"]] % len(cases)]   # citations of each type are generated in benchmark order
        seen[entry["type"]] += 1
        record = CitationRecord(case_name=entry["case_name"], court="", year=0, holding=entry["holding"], agent_source="e2")
        record.reporter_cite = entry.get("reporter_cite", "")
        _, matched, _ = cce._compute_existence(record)
        report = cce._resolve_opinion(record, matched).report
        if report.doc_id not in cache:
            got = parts["collection"].get(where={"parent_doc_id": {"$eq": report.doc_id}}, include=["documents", "metadatas", "embeddings"])
            rows = sorted(zip(got["metadatas"], got["documents"], got["embeddings"]), key=lambda r: int((r[0] or {}).get("chunk_index", 0)))
            cache[report.doc_id] = ([t or "" for _, t, _ in rows], np.asarray([v for _, _, v in rows], dtype="float32"))
        texts, matrix = cache[report.doc_id]
        query = np.asarray(parts["embedder"].encode(entry["holding"]), dtype="float32")
        dense = [int(i) for i in np.argsort(-(matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-12))]
        lexical_scores = BM25Okapi([tokens(t) for t in texts]).get_scores(tokens(entry["holding"]))
        lexical = sorted(range(len(texts)), key=lambda i: -lexical_scores[i])
        fused = {}
        for ranked in (dense, lexical):
            for rank, index in enumerate(ranked, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (RRF_K + rank)
        rankings = {"dense": dense, "hybrid": sorted(fused, key=lambda i: -fused[i])}
        needed = sorted({i for name, (ranking, count) in SELECTIONS.items() for i in rankings[ranking][:count]})
        items.append({"type": entry["type"], "label": 1 if entry["type"] == "correct" else 0, "case_id": case["id"],
                      "split": "development" if case["id"] in development else "test", "case_name": entry["case_name"],
                      "holding": entry["holding"], "doc_id": report.doc_id, "texts": texts, "rankings": rankings, "needed": needed})
    return items, digest


def score_nli(model_name, items):
    """Entailment probability for every needed passage: best 450-character window, and the whole passage."""
    from sentence_transformers import CrossEncoder

    long_input = "long" in model_name
    model = CrossEncoder(model_name, max_length=1024 if long_input else 512)
    mapping = getattr(model.model.config, "label2id", {}) or {}
    entail = next(int(v) for k, v in mapping.items() if "entail" in str(k).lower())
    window_pairs, passage_pairs, owners = [], [], []
    for n, item in enumerate(items):
        for index in item["needed"]:
            text = item["texts"][index]
            passage_pairs.append([text, item["holding"]])
            starts = list(range(0, max(1, len(text)), STEP))
            for start in starts:
                window_pairs.append([text[start:start + WINDOW], item["holding"]])
            owners.append((n, index, len(starts)))
    window_probs = softmax(model.predict(window_pairs, batch_size=64, show_progress_bar=False))[:, entail]
    passage_probs = softmax(model.predict(passage_pairs, batch_size=16, show_progress_bar=False))[:, entail]
    scores, cursor = {}, 0
    for position, (n, index, count) in enumerate(owners):
        scores[(n, index)] = {"window": float(window_probs[cursor:cursor + count].max()), "passage": float(passage_probs[position])}
        cursor += count
    del model
    return scores


def score_generator(parts, items):
    """P(yes) from the generator for: does this passage show that the court held the statement?"""
    import torch

    model, tokenizer = parts["model"], parts["tokenizer"]
    yes = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes", "yes")]
    no = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ("No", " No", "no")]
    scores = {}
    for n, item in enumerate(items):
        for index in item["needed"]:
            prompt = ("[INST] Passage from the opinion in " + item["case_name"] + ":\n\"" + item["texts"][index][:2800] + "\"\n\n"
                      "Statement: " + item["holding"] + "\n\n"
                      "Does the passage show that the court in this case held what the statement says? Answer Yes or No. [/INST]")
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
            with torch.no_grad():
                logits = model(**encoded).logits[0, -1].float()
            y, nn = torch.logsumexp(logits[yes], 0), torch.logsumexp(logits[no], 0)
            p = float(torch.softmax(torch.stack([y, nn]), 0)[0])
            scores[(n, index)] = {"passage": p}
    return scores


def evaluate(items, scores, granularity, selection):
    ranking, count = SELECTIONS[selection]
    values = np.array([max(scores[(n, i)][granularity] for i in item["rankings"][ranking][:count]) for n, item in enumerate(items)])
    out = {}
    for split in ("development", "test"):
        mask = np.array([item["split"] == split for item in items])
        kind = np.array([item["type"] for item in items])
        correct = values[mask & (kind == "correct")]
        swapped = values[mask & (kind == "misattributed")]
        false_holding = values[mask & (kind == "casehold_distractor")]
        wrong = np.concatenate([swapped, false_holding])
        out[split] = {"correct": correct, "swapped": swapped, "false_holding": false_holding, "wrong": wrong,
                      "auc_correct_vs_wrong": auc(correct, wrong), "auc_correct_vs_swapped": auc(correct, swapped)}
    threshold = best_threshold(out["development"]["correct"], out["development"]["wrong"])
    summary = {"threshold_chosen_on_development": round(threshold, 4)}
    for split in ("development", "test"):
        block = out[split]
        summary[split] = {
            "auc_correct_vs_wrong": round(block["auc_correct_vs_wrong"], 4), "auc_correct_vs_swapped": round(block["auc_correct_vs_swapped"], 4),
            "correct_accepted": f"{int((block['correct'] >= threshold).sum())}/{len(block['correct'])}",
            "swapped_accepted": f"{int((block['swapped'] >= threshold).sum())}/{len(block['swapped'])}",
            "false_holding_accepted": f"{int((block['false_holding'] >= threshold).sum())}/{len(block['false_holding'])}",
        }
    summary["test"]["auc_95ci"] = auc_interval(out["test"]["correct"], out["test"]["wrong"])
    return summary


def main():
    from experiments.v2.runner import load_components

    os.makedirs(OUT_DIR, exist_ok=True)
    started = time.time()
    parts = load_components(load_llm=True)
    items, digest = prepare_items(parts)
    counts = {s: {t: sum(1 for i in items if i["split"] == s and i["type"] == t) for t in ("correct", "misattributed", "casehold_distractor")}
              for s in ("development", "test")}
    print("citations:", counts)

    candidates, failures = {}, {}
    for model_name in NLI_MODELS + [GENERATOR]:
        began = time.time()
        try:
            scores = score_generator(parts, items) if model_name == GENERATOR else score_nli(model_name, items)
        except Exception as error:  # a model that cannot be downloaded or run is reported, not hidden
            failures[model_name] = f"{type(error).__name__}: {error}"
            print(f"  FAILED {model_name}: {failures[model_name][:160]}")
            continue
        for granularity in (("passage",) if model_name == GENERATOR else ("window", "passage")):
            for selection in SELECTIONS:
                name = f"{model_name} | {'450-character windows' if granularity == 'window' else 'whole passages'} | {selection}"
                candidates[name] = {"model": model_name, "granularity": granularity, "selection": selection,
                                    **evaluate(items, scores, granularity, selection)}
        print(f"  scored {model_name} in {time.time() - began:.0f}s")

    chosen = max(candidates, key=lambda k: candidates[k]["development"]["auc_correct_vs_wrong"])
    pick = candidates[chosen]
    passed = pick["test"]["auc_correct_vs_wrong"] >= CCE_GATE_MIN_AUC
    integrated = pick["model"] != GENERATOR
    decision = {
        "validated": bool(passed and integrated),
        "model": pick["model"] if integrated else config.NLI_MODEL_NAME,
        "threshold": pick["threshold_chosen_on_development"] if (passed and integrated) else config.TAU_ACCEPTANCE_THRESHOLD,
        "ranking": SELECTIONS[pick["selection"]][0], "passages": SELECTIONS[pick["selection"]][1],
        "granularity": pick["granularity"],
        "selected_candidate": chosen,
        "development_auc": pick["development"]["auc_correct_vs_wrong"], "test_auc": pick["test"]["auc_correct_vs_wrong"],
        "test_auc_95ci": pick["test"]["auc_95ci"], "required_test_auc": CCE_GATE_MIN_AUC,
        "passed_the_test": bool(passed),
        "source": "experiments/v2/e2_verifier_validation.py, " + datetime.now().isoformat(timespec="seconds"),
        "note": ("gate open: the support verdict of this model may steer the judge, the controller and the memory" if (passed and integrated) else
                 ("the selected candidate passed but is not a cross-encoder the checker can load; gate stays closed until it is integrated" if passed else
                  "gate closed: no candidate separates correct from wrong holdings well enough; the checker verifies existence only")),
    }
    results = {
        "rule": {"split_seed": SPLIT_SEED, "selection": "highest development AUC (correct vs all wrong)",
                 "threshold": "maximises true-accept rate minus false-accept rate on development", "gate": f"test AUC >= {CCE_GATE_MIN_AUC}"},
        "benchmark_sha256": digest, "citations": counts, "candidates": candidates, "models_that_could_not_be_run": failures,
        "decision": decision, "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    with open(CCE_GATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=False)

    print("\n" + "-" * 118)
    print(f"  {'candidate':<86} | dev AUC | test AUC | test correct / swapped accepted")
    print("-" * 118)
    for name, row in sorted(candidates.items(), key=lambda kv: -kv[1]["development"]["auc_correct_vs_wrong"]):
        print(f"  {name[:86]:<86} | {row['development']['auc_correct_vs_wrong']:.3f}   | {row['test']['auc_correct_vs_wrong']:.3f}    | "
              f"{row['test']['correct_accepted']} / {row['test']['swapped_accepted']}")
    print("-" * 118)
    print(f"  selected: {chosen}\n  test AUC {decision['test_auc']} {decision['test_auc_95ci']} against the required {CCE_GATE_MIN_AUC}: "
          f"{'GATE OPEN' if decision['validated'] else 'GATE CLOSED'}")
    print(f"  wrote {os.path.join(OUT_DIR, 'results.json')} and {CCE_GATE_PATH}")


if __name__ == "__main__":
    main()
