# LexAgent v3.0 | experiments/forensics/run_cheap_diagnostics.py
"""
Cheap, deterministic diagnostics for the forensic audit. NO LLM is loaded or called, no benchmark,
ablation or stress test is re-run, and no existing result file is touched. The only file written is
data/results/forensic_audit/diagnostics.json (a new file). The memory tests use a temporary
directory, never the project's memory store (which is only listed, read-only).

  1. NLI sanity: label mapping, probability normalisation, premise/hypothesis order.
  2. Case resolution: benchmark target -> canonical case -> opinion, and name variants.
  3. Evidence coverage for 20 labelled citations: what the checker reads and where the best
     entailing text of the whole opinion lies.
  4. Memory contamination: fresh namespace is empty, namespaces are isolated, the avoid list is
     returned regardless of relevance.
  5. Cache collisions: similarity of the benchmark questions under the cache's embedding, which
     question's answer each logged hit returned, and constructed look-alike questions.
  6. Retrieval: is the target opinion among the passages retrieved for each benchmark question
     (retrieval only, recomputed deterministically; it is not saved anywhere by the runs).
  7. Answer-similarity metric: its floor and what trivial answers score.

    python experiments/forensics/run_cheap_diagnostics.py
"""

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

from config import (  # noqa: E402
    BM25_INDEX_PATH, CHROMA_MEMORY_PATH, CORPUS_JSON_PATH, DRIVE_CHROMA_CORPUS_PATH, KNOWN_CASES_PATH,
    LOCAL_CHROMA_CORPUS_PATH, MEMORY_CACHE_THRESHOLD, MEMORY_HIGH_RISK_CCS_THRESH, NLI_MODEL_NAME, RESULTS_DIR,
    TOP_K_BM25, TOP_K_DENSE,
)

OUT_DIR = os.path.join(RESULTS_DIR, "forensic_audit")
OUT_PATH = os.path.join(OUT_DIR, "diagnostics.json")
ANALYSIS_PATH = os.path.join(OUT_DIR, "analysis_existing_results.json")
FALLBACK_ANSWER = "Insufficient evidence for a verified answer."
FUZZY_SAMPLE = 6  # benchmark cases used for the slow name-variant checks


def save(report):
    """Write what has been computed so far, so that a later failure loses nothing."""
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str)


def softmax(rows):
    rows = np.asarray(rows, dtype="float64")
    e = np.exp(rows - rows.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cosine_matrix(vectors):
    v = np.asarray(vectors, dtype="float64")
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v @ v.T


def quantiles(values):
    values = np.asarray(list(values), dtype="float64")
    if not len(values):
        return {}
    return {"n": int(len(values)), "min": round(float(values.min()), 4), "p25": round(float(np.percentile(values, 25)), 4),
            "median": round(float(np.median(values)), 4), "mean": round(float(values.mean()), 4),
            "p75": round(float(np.percentile(values, 75)), 4), "max": round(float(values.max()), 4)}


# ── 1 ───────────────────────────────────────────────────────────────────────
def nli_sanity(cce):
    model = cce.nli_model
    config = getattr(getattr(model, "model", None), "config", None)
    premise = "The court held that the police must obtain a warrant before searching the digital contents of a cell phone."
    pairs = {
        "entailment": (premise, "Police need a warrant to search a cell phone."),
        "contradiction": (premise, "Police never need a warrant to search a cell phone."),
        "neutral": (premise, "The defendant was sentenced to ten years in prison."),
        "entailment_2": ("A man is playing a guitar on a stage.", "A person is playing an instrument."),
        "contradiction_2": ("A man is playing a guitar on a stage.", "Nobody is playing music."),
        "neutral_2": ("A man is playing a guitar on a stage.", "The man is famous."),
    }
    raw = np.asarray(model.predict([list(p) for p in pairs.values()], show_progress_bar=False), dtype="float64")
    probs = softmax(raw)
    entail = cce._get_entailment_index()
    id2label = {int(k): v for k, v in (getattr(config, "id2label", {}) or {}).items()}
    rows, correct = {}, 0
    for (name, _), logits, p in zip(pairs.items(), raw, probs):
        predicted = id2label.get(int(p.argmax()), str(int(p.argmax())))
        expected = name.split("_")[0]
        correct += int(predicted.lower() == expected)
        rows[name] = {"raw_output": [round(float(x), 3) for x in logits], "softmax": [round(float(x), 4) for x in p],
                      "predicted_label": predicted, "entailment_probability_used_by_cce": round(float(p[entail]), 4)}
    # The checker passes [opinion window, holding]: premise first. Swap the order and compare.
    swapped = softmax(model.predict([[h, p] for p, h in pairs.values()], show_progress_bar=False))
    return {
        "model": NLI_MODEL_NAME, "id2label": id2label, "entailment_index_used": entail,
        "raw_outputs_already_sum_to_one": bool(np.allclose(raw.sum(axis=1), 1.0, atol=1e-3)),
        "raw_outputs_are_logits": bool((raw < 0).any() or (raw > 1).any()),
        "labels_correct_on_obvious_pairs": f"{correct}/{len(pairs)}",
        "pairs": rows,
        "entailment_probability_premise_first": [round(float(x), 4) for x in probs[:, entail]],
        "entailment_probability_order_swapped": [round(float(x), 4) for x in swapped[:, entail]],
        "max_length_tokens": getattr(model, "max_length", None),
    }


# ── 2 ───────────────────────────────────────────────────────────────────────
def case_resolution(cce, cases):
    from src.data.corpus_schema import CitationRecord

    def run(name, cite=""):
        record = CitationRecord(case_name=name, court="Supreme Court", year=0, holding="x", agent_source="forensic")
        record.reporter_cite = cite
        e, matched, kind = cce._compute_existence(record)
        resolution = cce._resolve_opinion(record, matched)
        report = resolution.report if resolution is not None else None
        return {"existence": round(float(e), 3), "match_type": kind,
                "doc_id": report.doc_id if report is not None else None,
                "is_opinion": bool(report.is_opinion) if report is not None else False,
                "cite_status": resolution.cite_status if resolution is not None else None,
                "words": report.words if report is not None else 0,
                "resolved_name": report.name if report is not None else None}

    variants = {
        "name_and_cite": lambda c: (c["target_case"], c.get("reporter_citation", "")),
        "name_only": lambda c: (c["target_case"], ""),
        "vs_instead_of_v": lambda c: (c["target_case"].replace(" v. ", " vs. "), ""),
        "lower_case": lambda c: (c["target_case"].lower(), ""),
        "cite_appended_to_name": lambda c: (f"{c['target_case']}, {c.get('reporter_citation', '')}".strip(", "), ""),
        "year_appended_to_name": lambda c: (f"{c['target_case']} ({c.get('year', '') or 1990})", ""),
        "united_states_abbreviated": lambda c: (c["target_case"].replace("United States", "U.S."), ""),
        "no_trailing_period": lambda c: (c["target_case"].rstrip("."), ""),
    }
    reference = {c["id"]: run(*variants["name_and_cite"](c)) for c in cases}
    out, failures = {}, {}
    # A name that is not an exact catalog entry is compared with every one of the ~314,000 known
    # names by fuzzy matching, which takes seconds per citation; those variants use a sample.
    sample = cases[:FUZZY_SAMPLE]
    timings = []
    for label, make in variants.items():
        pool = cases if label in ("name_and_cite", "name_only") else sample
        rows = {}
        for c in pool:
            began = time.time()
            rows[c["id"]] = run(*make(c))
            if label not in ("name_and_cite", "name_only"):
                timings.append(time.time() - began)
        changed = [c for c in pool if make(c) != variants["name_only"](c) or label in ("name_only", "name_and_cite")]
        ok_exist = sum(1 for c in changed if rows[c["id"]]["existence"] >= 0.85)
        ok_opinion = sum(1 for c in changed if rows[c["id"]]["is_opinion"] and rows[c["id"]]["doc_id"] == reference[c["id"]]["doc_id"])
        out[label] = {"citations_tested": len(changed), "exists": ok_exist, "resolved_to_the_reference_opinion": ok_opinion}
        failures[label] = [{"case_id": c["id"], "given": make(c)[0], **rows[c["id"]]} for c in changed
                           if not (rows[c["id"]]["is_opinion"] and rows[c["id"]]["doc_id"] == reference[c["id"]]["doc_id"])][:6]
    return {"reference_all_resolve_to_an_opinion": sum(1 for r in reference.values() if r["is_opinion"]),
            "reference_cite_status": {k: sum(1 for r in reference.values() if r["cite_status"] == k) for k in {r["cite_status"] for r in reference.values()}},
            "variants": out, "example_failures": failures,
            "seconds_per_existence_check_of_a_non_exact_name": quantiles(timings),
            "known_names_scanned_per_non_exact_check": len(cce.known_cases)}


# ── 3 ───────────────────────────────────────────────────────────────────────
def evidence_coverage(cce, collection, embedder, cases, count=20):
    from experiments.run_cce_evidence_study import WINDOW_CHARS, WINDOW_STEP, dense_order, opinion_passages
    from src.data.corpus_schema import CitationRecord
    from src.evaluation.stress_test import build_stress_test_dataset

    items = [i for i in build_stress_test_dataset(cases, []) if i["type"] == "correct"][:count]
    entail = cce._get_entailment_index()
    scan_limit = int(os.environ.get("LEXAGENT_CCE_SCAN_LIMIT", "8000"))

    def windows_of(text):
        return [(start, text[start:start + WINDOW_CHARS]) for start in range(0, len(text), WINDOW_STEP)] or [(0, text)]

    def score(texts, holding):
        logits = np.asarray(cce.nli_model.predict([[t, holding] for t in texts], batch_size=128, show_progress_bar=False))
        return softmax(logits)[:, entail]

    rows = []
    for item in items:
        record = CitationRecord(case_name=item["case_name"], court="Supreme Court", year=1990, holding=item["holding"], agent_source="forensic")
        record.reporter_cite = item.get("reporter_cite", "")
        deployed = cce.compute_ccs(record)
        _, matched, _ = cce._compute_existence(record)
        report = cce._resolve_opinion(record, matched).report
        passages = opinion_passages(collection, report.doc_id)
        order = dense_order(passages, embedder.encode(item["holding"]))
        chosen = [int(i) for i in order[:3]]
        scanned_text = " ".join(passages[i]["text"] for i in chosen)
        scanned = windows_of(scanned_text[:scan_limit])
        scanned_scores = score([w for _, w in scanned], item["holding"])
        best = int(scanned_scores.argmax())
        # where in the three passages the best window lies
        offsets, position = [], 0
        for i in chosen:
            offsets.append((position, position + len(passages[i]["text"]), i))
            position += len(passages[i]["text"]) + 1
        best_chunk = next((i for a, b, i in offsets if a <= scanned[best][0] < b), chosen[-1])
        # the whole opinion, passage by passage
        whole_best, whole_chunk, whole_text = 0.0, None, ""
        per_chunk_best = []
        for n, passage in enumerate(passages):
            ws = windows_of(passage["text"])
            ss = score([w for _, w in ws], item["holding"])
            per_chunk_best.append(float(ss.max()))
            if float(ss.max()) > whole_best:
                whole_best, whole_chunk, whole_text = float(ss.max()), n, ws[int(ss.argmax())][1]
        tokens = set(re.findall(r"[a-z]{5,}", item["holding"].lower()))
        full_text = " ".join(p["text"] for p in passages).lower()
        rows.append({
            "case_name": item["case_name"], "resolved_opinion": f"{report.name}, {report.cite}",
            "opinion_passages": len(passages), "opinion_words": report.words,
            "passages_read": chosen, "characters_read": min(len(scanned_text), scan_limit),
            "characters_dropped_by_the_8000_limit": max(0, len(scanned_text) - scan_limit),
            "share_of_opinion_read": round(min(len(scanned_text), scan_limit) / max(1, len(full_text)), 4),
            "windows_scored": len(scanned),
            "support_deployed": deployed["support"], "tier_deployed": deployed["tier"],
            "best_window_score": round(float(scanned_scores.max()), 4), "best_window_passage": best_chunk,
            "best_window_text": scanned[best][1][:160],
            "evidence_snippet_shown_to_judge": (deployed.get("evidence") or "")[:100],
            "snippet_is_the_best_window": (deployed.get("evidence") or "")[:60] == scanned[best][1][:60],
            "whole_opinion_best_score": round(whole_best, 4), "whole_opinion_best_passage": whole_chunk,
            "whole_opinion_best_text": whole_text[:160],
            "best_passage_was_read": whole_chunk in chosen,
            "passages_scoring_at_least_0.70": sum(1 for s in per_chunk_best if s >= 0.70),
            "holding_terms_found_in_read_text": round(sum(1 for t in tokens if t in scanned_text.lower()) / max(1, len(tokens)), 3),
            "holding_terms_found_in_whole_opinion": round(sum(1 for t in tokens if t in full_text) / max(1, len(tokens)), 3),
        })
    accepted = [r for r in rows if r["tier_deployed"] == "SUPPORTED"]
    rejected = [r for r in rows if r["tier_deployed"] != "SUPPORTED"]
    return {
        "citations": len(rows), "accepted_as_deployed": len(accepted),
        "deployed_support_reproduced_by_best_window": sum(1 for r in rows if r["support_deployed"] is not None and abs(r["support_deployed"] - r["best_window_score"]) < 0.02),
        "snippet_shown_to_judge_is_the_best_window": sum(1 for r in rows if r["snippet_is_the_best_window"]),
        "share_of_opinion_read": quantiles(r["share_of_opinion_read"] for r in rows),
        "characters_dropped_by_the_8000_limit": quantiles(r["characters_dropped_by_the_8000_limit"] for r in rows),
        "rejected_but_an_entailing_passage_exists_elsewhere": sum(1 for r in rejected if r["whole_opinion_best_score"] >= 0.70),
        "rejected_and_no_passage_of_the_opinion_entails": sum(1 for r in rejected if r["whole_opinion_best_score"] < 0.70),
        "rejected": len(rejected),
        "holding_terms_in_read_text": quantiles(r["holding_terms_found_in_read_text"] for r in rows),
        "holding_terms_in_whole_opinion": quantiles(r["holding_terms_found_in_whole_opinion"] for r in rows),
        "rows": rows,
    }


# ── 4 ───────────────────────────────────────────────────────────────────────
def memory_contamination(embedder):
    import chromadb
    from src.novel.adaptive_memory import AdaptiveMemory

    tmp = tempfile.mkdtemp(prefix="lexagent_forensic_memory_")
    out = {"store_used_for_the_test": "temporary directory (deleted afterwards)"}
    try:
        client = chromadb.PersistentClient(path=tmp)
        first = AdaptiveMemory(namespace=f"forensic_{int(time.time())}_a", client=client)
        out["fresh_namespace_counts"] = {"verdicts": first.verdict_collection.count(), "risk_entries": first.risk_collection.count()}
        out["avoid_list_from_empty_memory"] = first.get_high_risk_citations("Does the Fourth Amendment require a warrant?", embedder)

        def citation(name, tier, ccs):
            return {"tier": tier, "ccs": ccs, "citation": {"case_name": name}}

        # One debate about the Fourth Amendment in which a REAL case was labelled misattributed.
        state = {"query": "May police search a cell phone seized from an arrestee without a warrant?",
                 "citation_confidence_report": {"a": citation("Riley v. California", "MISATTRIBUTED", 0.05),
                                                "b": citation("Smith v. Imaginary Party", "FABRICATED", 0.0),
                                                "c": citation("Katz v. United States", "SUPPORTED", 0.91),
                                                "d": citation("Carpenter v. United States", "UNCERTAIN", 0.52),
                                                "e": citation("Dobbs v. Jackson Women's Health Organization", "UNVERIFIABLE", 0.0)}}
        first.update_risk_index(state, embedder)
        out["risk_entries_after_one_debate"] = first.risk_collection.count()
        stored = first.risk_collection.get()
        out["risk_entries_written"] = [{"case_name": m["case_name"], "tier": m["tier"], "avg_ccs": m["avg_ccs"]} for m in stored["metadatas"]]
        out["tiers_written_as_risk"] = sorted({m["tier"] for m in stored["metadatas"]})
        # An unrelated question later in the same run.
        unrelated = "Does the Takings Clause require compensation when a regulation destroys all economic value of land?"
        out["avoid_list_for_an_unrelated_question"] = first.get_high_risk_citations(unrelated, embedder)
        result = first.risk_collection.query(query_embeddings=[embedder.encode(unrelated).tolist()], n_results=first.risk_collection.count())
        out["distance_of_those_entries_from_the_unrelated_question"] = [round(float(d), 4) for d in result["distances"][0]]
        out["lookup_applies_a_relevance_threshold"] = False
        # A second namespace in the same store.
        second = AdaptiveMemory(namespace=f"forensic_{int(time.time())}_b", client=client)
        out["second_namespace_counts"] = {"verdicts": second.verdict_collection.count(), "risk_entries": second.risk_collection.count()}
        out["second_namespace_avoid_list"] = second.get_high_risk_citations(unrelated, embedder)
        # Verdict storage rule
        first.store_verdict({"query": state["query"], "final_answer": "A warrant is generally required.", "judge_confidence": 0.9, "avg_ccs": 0.8}, embedder)
        out["verdicts_after_one_store"] = first.verdict_collection.count()
        del first, second, client
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # The project's own memory store: list only.
    try:
        real = chromadb.PersistentClient(path=CHROMA_MEMORY_PATH)
        listing = {}
        for collection in real.list_collections():
            listing[collection.name] = collection.count()
        groups = {}
        for name, count in listing.items():
            namespace = re.sub(r"_(verdict_cache|citation_risk_index)$", "", name) if re.search(r"_(verdict_cache|citation_risk_index)$", name) else "(no namespace)"
            kind = "verdicts" if name.endswith("verdict_cache") else "risk_entries"
            groups.setdefault(namespace, {})[kind] = count
        out["project_memory_store_namespaces"] = dict(sorted(groups.items()))
    except Exception as error:  # the listing is informative only
        out["project_memory_store_namespaces"] = f"could not be listed: {error}"
    return out


# ── 5 ───────────────────────────────────────────────────────────────────────
def cache_collisions(embedder, cases, analysis):
    queries = [c["query"] for c in cases]
    vectors = embedder.encode(queries, show_progress_bar=False)
    sim = cosine_matrix(vectors)
    n = len(cases)
    upper = sim[np.triu_indices(n, k=1)]
    threshold = MEMORY_CACHE_THRESHOLD
    earlier_max = [float(sim[i, :i].max()) if i else 0.0 for i in range(n)]
    out = {
        "embedding_model": "the system's embedder (Legal-BERT)", "cache_threshold_cosine": threshold,
        "pairwise_similarity_of_the_96_distinct_questions": quantiles(upper),
        "pairs_at_or_above_threshold": int((upper >= threshold).sum()), "pairs_total": int(len(upper)),
        "share_of_pairs_at_or_above_threshold": round(float((upper >= threshold).mean()), 4),
        "questions_with_an_earlier_question_above_threshold": int(sum(1 for v in earlier_max[1:] if v >= threshold)),
        "threshold_needed_for_no_collision_among_these_questions": round(float(upper.max()) + 1e-4, 4),
    }
    # Which question's answer did each logged hit return? (ablation, full variant)
    hits = []
    per_case = analysis.get("per_case", []) if analysis else []
    stored = []
    for position, row in enumerate(per_case):
        log = row.get("ablation_full_log") or {}
        if log.get("cache_hit"):
            best = max(stored, key=lambda j: sim[position, j]) if stored else None
            hits.append({
                "position": position + 1, "case_id": row["case_id"], "topic": row["topic"],
                "logged_distance": log.get("cache_distance"),
                "nearest_stored_distance": round(1 - float(sim[position, best]), 4) if best is not None else None,
                "answer_returned_was_written_for_case": per_case[best]["case_id"] if best is not None else None,
                "that_case_topic": per_case[best]["topic"] if best is not None else None,
                "same_target_decision": (per_case[best]["target_case"] == row["target_case"]) if best is not None else None,
            })
        elif log.get("verdict_stored"):
            stored.append(position)
    reproduced = sum(1 for h in hits if h["logged_distance"] is not None and h["nearest_stored_distance"] is not None
                     and abs(h["logged_distance"] - h["nearest_stored_distance"]) < 0.0015)
    out["ablation_full_run"] = {"hits_logged": len(hits), "logged_distance_reproduced_from_question_embeddings": reproduced,
                                "hits_where_both_questions_share_the_target_decision": sum(1 for h in hits if h["same_target_decision"]),
                                "hits": hits}
    # Constructed look-alikes: legally distinct questions.
    pairs = [
        ("May police search a cell phone seized from an arrestee without a warrant?",
         "May police search a car's passenger compartment incident to the arrest of an occupant without a warrant?"),
        ("Does the Sixth Amendment guarantee appointed counsel to indigent defendants in state felony trials?",
         "Does the Sixth Amendment guarantee a jury trial for petty offenses in state courts?"),
        ("Is the death penalty for a crime committed by a juvenile cruel and unusual punishment under the Eighth Amendment?",
         "Is a mandatory sentence of life without parole for a juvenile cruel and unusual punishment under the Eighth Amendment?"),
        ("Does the First Amendment protect flag burning as symbolic speech?",
         "Does the Second Amendment protect an individual right to keep a handgun in the home?"),
        ("Must a suspect be warned of the right to remain silent before custodial interrogation?",
         "Must prosecutors disclose exculpatory evidence to the defense before trial?"),
    ]
    pair_vectors = embedder.encode([q for pair in pairs for q in pair], show_progress_bar=False)
    pair_sim = cosine_matrix(pair_vectors)
    import chromadb
    from src.novel.adaptive_memory import AdaptiveMemory
    tmp = tempfile.mkdtemp(prefix="lexagent_forensic_cache_")
    constructed = []
    try:
        client = chromadb.PersistentClient(path=tmp)
        for index, (first, second) in enumerate(pairs):
            memory = AdaptiveMemory(namespace=f"pair{index}", client=client)
            memory.store_verdict({"query": first, "final_answer": f"ANSWER WRITTEN FOR: {first}", "judge_confidence": 0.9, "avg_ccs": 0.9}, embedder)
            hit, answer = memory.check_cache(second, embedder)
            constructed.append({"stored_question": first, "new_question": second,
                                "cosine_similarity": round(float(pair_sim[2 * index, 2 * index + 1]), 4),
                                "cache_returned_the_stored_answer": bool(hit)})
        del client
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out["constructed_legally_distinct_pairs"] = constructed
    out["constructed_pairs_wrongly_answered_from_cache"] = sum(1 for c in constructed if c["cache_returned_the_stored_answer"])
    return out, vectors


# ── 6 ───────────────────────────────────────────────────────────────────────
def retrieval_diagnostic(cce, collection, embedder, cases, analysis):
    from src.data.build_corpus import load_corpus
    from src.data.case_catalog import names_match
    from src.data.corpus_schema import CitationRecord
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever

    corpus = load_corpus(CORPUS_JSON_PATH)
    dense = DenseRetriever(collection, embedder)
    bm25 = BM25Retriever.load_index(BM25_INDEX_PATH, corpus)
    hybrid = HybridRetriever(dense, bm25)
    per_case = {row["case_id"]: row for row in (analysis.get("per_case", []) if analysis else [])}

    def is_target(doc, case, report):
        parent = (doc.metadata or {}).get("parent_doc_id") or str(doc.doc_id).split("::chunk::")[0]
        if report is not None and parent == report.doc_id:
            return True
        return bool(doc.case_name) and names_match(doc.case_name, case["target_case"])

    def first_rank(docs, case, report):
        return next((rank for rank, doc in enumerate(docs, 1) if is_target(doc, case, report)), None)

    rows = []
    for case in cases:
        record = CitationRecord(case_name=case["target_case"], court="", year=0, holding="x")
        record.reporter_cite = case.get("reporter_citation", "")
        _, matched, _ = cce._compute_existence(record)
        report = cce._resolve_opinion(record, matched).report
        lex_docs = hybrid.retrieve(case["query"], "US_Federal")   # what the debate's first round retrieves
        base_docs = hybrid.retrieve(case["query"])                 # what the baseline retrieves
        dense_docs = dense.retrieve(case["query"], "US_Federal")
        bm25_docs = bm25.retrieve(case["query"])
        names = [d.case_name for d in lex_docs]
        row = per_case.get(case["id"], {})
        rows.append({
            "case_id": case["id"], "target_case": case["target_case"],
            "rank_debate_first_round": first_rank(lex_docs, case, report), "passages_debate": len(lex_docs),
            "rank_baseline_list": first_rank(base_docs, case, report),
            "in_baseline_prompt_top4": (first_rank(base_docs[:4], case, report) is not None),
            "rank_dense_only": first_rank(dense_docs, case, report), "rank_bm25_only": first_rank(bm25_docs, case, report),
            "distinct_cases_in_debate_passages": len(set(names)), "passages_from_most_frequent_case": max((names.count(x) for x in set(names)), default=0),
            "same_passages_for_both_systems": [d.doc_id for d in lex_docs] == [d.doc_id for d in base_docs],
            "passages_shared_by_both_systems": len({d.doc_id for d in lex_docs} & {d.doc_id for d in base_docs}),
            "target_name_in_question": bool(case["target_case"].split(" v.")[0].lower() in case["query"].lower()),
            "baseline_cited_target": bool((row.get("baseline") or {}).get("ar")),
            "lexagent_cited_target": bool((row.get("lexagent") or {}).get("ar")),
            "lexagent_debated": not row.get("benchmark_cache_signature", False),
        })

    def share(predicate, subset=None):
        subset = rows if subset is None else subset
        return {"count": sum(1 for r in subset if predicate(r)), "of": len(subset),
                "share": round(sum(1 for r in subset if predicate(r)) / max(1, len(subset)), 4)}

    debated = [r for r in rows if r["lexagent_debated"]]
    found8 = [r for r in rows if r["rank_debate_first_round"] is not None]
    found4 = [r for r in rows if r["in_baseline_prompt_top4"]]
    return {
        "note": "Retrieval recomputed deterministically for the plain benchmark question (no LLM). The runs did not save "
                "their retrieved passages; later debate rounds retrieved with reflection queries appended, which is not reproduced here.",
        "top_k": {"dense": TOP_K_DENSE, "bm25": TOP_K_BM25, "fused_passages_to_debate": TOP_K_DENSE + TOP_K_BM25, "passages_in_baseline_prompt": 4,
                  "characters_per_passage": {"debate_prosecutor": 500, "baseline": 300}},
        "target_at_rank_1": share(lambda r: r["rank_debate_first_round"] == 1),
        "target_in_top_4": share(lambda r: r["rank_debate_first_round"] is not None and r["rank_debate_first_round"] <= 4),
        "target_in_top_8_debate_first_round": share(lambda r: r["rank_debate_first_round"] is not None),
        "target_in_baseline_prompt": share(lambda r: r["in_baseline_prompt_top4"]),
        "target_absent_from_all_retrieved_passages": share(lambda r: r["rank_debate_first_round"] is None),
        "target_found_by_dense_top4": share(lambda r: r["rank_dense_only"] is not None),
        "target_found_by_bm25_top4": share(lambda r: r["rank_bm25_only"] is not None),
        "identical_passage_lists_for_both_systems": share(lambda r: r["same_passages_for_both_systems"]),
        "passages_shared_by_both_systems": quantiles(r["passages_shared_by_both_systems"] for r in rows),
        "distinct_cases_among_debate_passages": quantiles(r["distinct_cases_in_debate_passages"] for r in rows),
        "passages_from_the_most_frequent_case": quantiles(r["passages_from_most_frequent_case"] for r in rows),
        "question_names_a_party_of_the_target": share(lambda r: r["target_name_in_question"]),
        "baseline_cited_target_when_it_was_in_its_prompt": share(lambda r: r["baseline_cited_target"], found4),
        "baseline_cited_target_when_it_was_not_in_its_prompt": share(lambda r: r["baseline_cited_target"], [r for r in rows if not r["in_baseline_prompt_top4"]]),
        "debate_cited_target_when_retrieved_in_round_one": share(lambda r: r["lexagent_cited_target"], [r for r in debated if r["rank_debate_first_round"] is not None]),
        "debate_cited_target_when_not_retrieved_in_round_one": share(lambda r: r["lexagent_cited_target"], [r for r in debated if r["rank_debate_first_round"] is None]),
        "target_in_top_8_among_debated_cases": share(lambda r: r["rank_debate_first_round"] is not None, debated),
        "target_in_top_8_among_skipped_cases": share(lambda r: r["rank_debate_first_round"] is not None, [r for r in rows if not r["lexagent_debated"]]),
        "rows": rows,
    }


# ── 7 ───────────────────────────────────────────────────────────────────────
def answer_similarity_metric(embedder, cases, analysis, question_vectors):
    holdings = [c.get("ground_truth_holding", "")[:500] for c in cases]
    hv = np.asarray(embedder.encode(holdings, show_progress_bar=False), dtype="float64")
    hv_n = hv / np.linalg.norm(hv, axis=1, keepdims=True)
    cross = hv_n @ hv_n.T
    n = len(cases)
    off = cross[~np.eye(n, dtype=bool)]
    fallback = np.asarray(embedder.encode([FALLBACK_ANSWER], show_progress_bar=False)[0], dtype="float64")
    fallback /= np.linalg.norm(fallback)
    qv = np.asarray(question_vectors, dtype="float64")
    qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    fallback_scores = hv_n @ fallback
    question_scores = (hv_n * qv).sum(axis=1)
    per_case = {row["case_id"]: row for row in (analysis.get("per_case", []) if analysis else [])}
    lex = np.array([per_case[c["id"]]["lexagent"]["ass"] for c in cases]) if per_case else np.array([])
    base = np.array([per_case[c["id"]]["baseline"]["ass"] for c in cases]) if per_case else np.array([])
    out = {
        "definition": "cosine(Legal-BERT(final_answer[:500]), Legal-BERT(reference_holding[:500]))",
        "holding_of_a_different_case": quantiles(off),
        "the_fixed_fallback_sentence": quantiles(fallback_scores),
        "the_question_itself_as_the_answer": quantiles(question_scores),
        "recorded_lexagent": quantiles(lex), "recorded_baseline": quantiles(base),
    }
    if per_case:
        out["lexagent_answers_identical_to_the_fallback_sentence"] = int(sum(1 for a, f in zip(lex, fallback_scores) if abs(a - round(float(f), 4)) < 5e-5))
        out["baseline_answers_identical_to_the_fallback_sentence"] = int(sum(1 for a, f in zip(base, fallback_scores) if abs(a - round(float(f), 4)) < 5e-5))
        skipped = np.array([per_case[c["id"]]["benchmark_cache_signature"] for c in cases])
        out["recorded_lexagent_skipped_debates"] = quantiles(lex[skipped])
        out["recorded_lexagent_debated"] = quantiles(lex[~skipped])
        out["recorded_baseline_on_skipped_cases"] = quantiles(base[skipped])
        out["recorded_baseline_on_debated_cases"] = quantiles(base[~skipped])
        out["share_of_lexagent_answers_scoring_below_a_random_other_holding_mean"] = round(float((lex < off.mean()).mean()), 4)
        out["metric_range_used"] = {"floor_random_other_holding_mean": round(float(off.mean()), 4), "ceiling": 1.0,
                                    "lexagent_minus_baseline": round(float(lex.mean() - base.mean()), 4)}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    started = time.time()
    import chromadb
    from experiments.benchmark_cases import get_benchmark_cases
    from src.models.load_model import load_embedder
    from src.novel.cce import CitationConfidenceEngine
    from src.utils.chroma_sync import ensure_chroma_ready

    with contextlib.redirect_stdout(io.StringIO()):
        cases = get_benchmark_cases(100)
    analysis = json.load(open(ANALYSIS_PATH, encoding="utf-8")) if os.path.exists(ANALYSIS_PATH) else {}
    embedder = load_embedder()
    client = chromadb.PersistentClient(path=ensure_chroma_ready(LOCAL_CHROMA_CORPUS_PATH, DRIVE_CHROMA_CORPUS_PATH))
    collection = client.get_or_create_collection("cap_authorities")
    cce = CitationConfidenceEngine(collection, embedder, KNOWN_CASES_PATH, policy="legacy")

    report = {"note": "Deterministic diagnostics. No LLM was loaded or called; no experiment was re-run; no result file was modified.",
              "llm_loaded": False, "cases": len(cases)}
    steps = [
        ("nli_sanity", lambda: nli_sanity(cce)),
        ("evidence_coverage", lambda: evidence_coverage(cce, collection, embedder, cases)),
        ("memory_contamination", lambda: memory_contamination(embedder)),
    ]
    for name, step in steps:
        try:
            report[name] = step()
        except Exception as error:  # one failed diagnostic must not hide the others
            report[name] = {"error": f"{type(error).__name__}: {error}"}
        print(f"[{time.time() - started:6.0f}s] {name} done")
        save(report)
    vectors = None
    try:
        report["cache_collisions"], vectors = cache_collisions(embedder, cases, analysis)
    except Exception as error:
        report["cache_collisions"] = {"error": f"{type(error).__name__}: {error}"}
    print(f"[{time.time() - started:6.0f}s] cache_collisions done")
    save(report)
    try:
        if vectors is None:
            vectors = embedder.encode([c["query"] for c in cases], show_progress_bar=False)
        report["answer_similarity_metric"] = answer_similarity_metric(embedder, cases, analysis, vectors)
    except Exception as error:
        report["answer_similarity_metric"] = {"error": f"{type(error).__name__}: {error}"}
    print(f"[{time.time() - started:6.0f}s] answer_similarity_metric done")
    save(report)
    try:
        report["retrieval"] = retrieval_diagnostic(cce, collection, embedder, cases, analysis)
    except Exception as error:
        report["retrieval"] = {"error": f"{type(error).__name__}: {error}"}
    print(f"[{time.time() - started:6.0f}s] retrieval done")
    try:
        report["case_resolution"] = case_resolution(cce, cases)
    except Exception as error:
        report["case_resolution"] = {"error": f"{type(error).__name__}: {error}"}
    print(f"[{time.time() - started:6.0f}s] case_resolution done")
    report["seconds"] = round(time.time() - started, 1)
    save(report)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
