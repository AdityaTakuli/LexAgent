# LexAgent v3.0 | experiments/v2/e5_retrieval.py
"""
E5 — Retrieval only: how often is the target opinion among the retrieved passages? (no LLM)

The audit found the target opinion missing from the debate's passages for one question in five, and
the dense retriever (Legal-BERT, mean pooled) finding it far less often than BM25. This experiment
measures alternatives on the same 96 questions and the same 528,283 passages. It changes nothing in
the system: the new embeddings live only in memory for the duration of the run.

  dense      the system's Legal-BERT index, with and without the jurisdiction string in the query;
             two models trained for retrieval (BAAI/bge-base-en-v1.5, all-MiniLM-L6-v2)
  lexical    the system's BM25 (whitespace tokens); BM25 on word tokens (punctuation removed)
  fusion     reciprocal-rank fusion of a dense and a lexical ranking at several depths, with and
             without a limit of two passages per case

"Found" means that a passage of the opinion the benchmark's target resolves to is in the list.

    python experiments/v2/e5_retrieval.py
"""

import json
import os
import re
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

import config  # noqa: E402
from config import RESULTS_DIR, RRF_K  # noqa: E402

OUT_DIR = os.path.join(RESULTS_DIR, "v2", "e5_retrieval")
DEPTH = 50
DENSE_CANDIDATES = [
    ("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", 512),
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "", 256),
]


def parent_of(doc_id: str) -> str:
    return str(doc_id).split("::chunk::")[0]


def words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def fuse(rankings, depth_each, top=8, per_case=None):
    """Reciprocal-rank fusion of ranked lists of passage ids; optionally at most `per_case` passages per case."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking[:depth_each], start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores, key=lambda d: -scores[d])
    if per_case is None:
        return ordered[:top]
    kept, seen = [], {}
    for doc_id in ordered:
        case = parent_of(doc_id)
        if seen.get(case, 0) < per_case:
            kept.append(doc_id)
            seen[case] = seen.get(case, 0) + 1
        if len(kept) == top:
            break
    return kept


def hit_rates(rankings, targets, depths=(1, 4, 8, 20)):
    out = {}
    for depth in depths:
        found = sum(1 for ranking, target in zip(rankings, targets) if any(parent_of(d) == target for d in ranking[:depth]))
        out[f"top{depth}"] = {"found": found, "share": round(found / len(targets), 4)}
    return out


def main():
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer

    from experiments.v2.runner import benchmark_cases
    from src.data.build_corpus import load_corpus
    from src.data.case_catalog import CaseCatalog
    from src.data.corpus_schema import CorpusDocument
    from src.models.load_model import load_embedder
    from src.novel.cce import CitationConfidenceEngine
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.utils.chroma_sync import ensure_chroma_ready

    os.makedirs(OUT_DIR, exist_ok=True)
    started = time.time()
    cases, digest = benchmark_cases()
    questions = [c["query"] for c in cases]
    embedder = load_embedder()
    client = chromadb.PersistentClient(path=ensure_chroma_ready(config.LOCAL_CHROMA_CORPUS_PATH, config.DRIVE_CHROMA_CORPUS_PATH))
    collection = client.get_or_create_collection("cap_authorities")
    cce = CitationConfidenceEngine(collection, embedder, config.KNOWN_CASES_PATH, catalog=CaseCatalog.load(config.CASE_CATALOG_PATH), policy="gated")
    targets = [cce.check_existence(c.get("target_case", ""), c.get("reporter_citation", "") or "").get("resolved_doc_id") for c in cases]
    corpus = load_corpus(config.CORPUS_JSON_PATH)
    ids = [d.doc_id for d in corpus]
    print(f"[{time.time() - started:5.0f}s] corpus loaded: {len(corpus):,} passages; targets resolved: {sum(1 for t in targets if t)}/{len(targets)}")

    rankings, timings = {}, {}

    # ── dense: the system's index ──
    for label, suffix in (("legal-bert, question + jurisdiction (the debate's query)", " US_Federal"), ("legal-bert, question only (the earlier baseline's query)", "")):
        vectors = embedder.encode([q + suffix for q in questions], show_progress_bar=False).tolist()
        got = collection.query(query_embeddings=vectors, n_results=DEPTH, include=[])
        rankings[label] = got["ids"]

    # ── lexical: the system's BM25, and BM25 on word tokens ──
    began = time.time()
    system_bm25 = BM25Retriever.load_index(config.BM25_INDEX_PATH, corpus)
    system_bm25.top_k = DEPTH
    rankings["bm25, whitespace tokens (the system's)"] = [[d.doc_id for d in system_bm25.retrieve(q)] for q in questions]
    del system_bm25
    print(f"[{time.time() - started:5.0f}s] system BM25 done")
    began = time.time()
    word_corpus = [CorpusDocument(doc_id=d.doc_id, text=words(d.text), source=d.source, case_name=d.case_name) for d in corpus]
    word_bm25 = BM25Retriever(word_corpus, top_k=DEPTH)
    rankings["bm25, word tokens"] = [[d.doc_id for d in word_bm25.retrieve(words(q))] for q in questions]
    timings["bm25_word_index_seconds"] = round(time.time() - began, 1)
    del word_bm25, word_corpus
    print(f"[{time.time() - started:5.0f}s] word-token BM25 done")

    # ── dense: retrieval-trained models (embeddings kept in memory only) ──
    failures = {}
    texts = [d.text for d in corpus]
    for label, name, instruction, max_tokens in DENSE_CANDIDATES:
        began = time.time()
        try:
            model = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
            model.max_seq_length = max_tokens
            if torch.cuda.is_available():
                model.half()
            blocks = []
            for start in range(0, len(texts), 20000):
                block = model.encode(texts[start:start + 20000], batch_size=256, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
                blocks.append(block.half() if torch.cuda.is_available() else block)
                if start % 100000 == 0:
                    print(f"    {label}: {start:,} passages embedded, {time.time() - began:.0f}s")
            matrix = torch.cat(blocks)
            queries = model.encode([instruction + q for q in questions], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).to(matrix.dtype)
            top = torch.topk(queries @ matrix.T, DEPTH, dim=1).indices.cpu().numpy()
            rankings[f"{label} (retrieval-trained)"] = [[ids[i] for i in row] for row in top]
            timings[f"{label}_embedding_seconds"] = round(time.time() - began, 1)
            del model, matrix, blocks, queries
            torch.cuda.empty_cache()
            print(f"[{time.time() - started:5.0f}s] {label} done")
        except Exception as error:  # a model that cannot be downloaded or run is reported, not hidden
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"    FAILED {name}: {failures[name][:160]}")

    single = {label: hit_rates(ranked, targets) for label, ranked in rankings.items()}
    dense_names = [n for n in rankings if not n.startswith("bm25")]
    lexical_names = [n for n in rankings if n.startswith("bm25")]
    fused = {}
    for dense in dense_names:
        for lexical in lexical_names:
            for depth in (4, 8, 20):
                for cap in (None, 2):
                    lists = [fuse([rankings[dense][q], rankings[lexical][q]], depth, top=8, per_case=cap) for q in range(len(questions))]
                    found = sum(1 for ranking, target in zip(lists, targets) if any(parent_of(d) == target for d in ranking))
                    first = sum(1 for ranking, target in zip(lists, targets) if ranking and parent_of(ranking[0]) == target)
                    in4 = sum(1 for ranking, target in zip(lists, targets) if any(parent_of(d) == target for d in ranking[:4]))
                    distinct = float(np.mean([len({parent_of(d) for d in ranking}) for ranking in lists]))
                    fused[f"{dense} + {lexical} | {depth}+{depth} | {'max 2 per case' if cap else 'no limit'}"] = {
                        "dense": dense, "lexical": lexical, "depth_each": depth, "per_case_limit": cap,
                        "target_at_rank_1": first, "target_in_top_4": in4, "target_in_top_8": found,
                        "share_top_8": round(found / len(targets), 4), "distinct_cases_in_top_8": round(distinct, 2)}
    current = next((v for k, v in fused.items() if k.startswith("legal-bert, question + jurisdiction") and "whitespace" in k and "| 4+4 | no limit" in k), None)
    best = max(fused.items(), key=lambda kv: (kv[1]["target_in_top_8"], kv[1]["target_in_top_4"]))
    result = {"benchmark_sha256": digest, "questions": len(questions), "passages": len(corpus), "depth_retrieved": DEPTH,
              "single_retrievers": single, "fusion_top_8": fused, "the_system_as_it_is": current, "best_fusion": {"name": best[0], **best[1]},
              "models_that_could_not_be_run": failures, "timings": timings, "seconds": round(time.time() - started, 1),
              "note": "Selecting the best configuration on these 96 questions and reporting it on the same questions is optimistic; "
                      "treat the ranking of configurations as the finding, not the best number."}
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    lines = ["# E5 — retrieval only", "", f"{len(questions)} questions, {len(corpus):,} passages. \"Found\" = a passage of the target opinion is in the list.", "",
             "## Single retrievers", "", "| Retriever | Top 1 | Top 4 | Top 8 | Top 20 |", "|---|---|---|---|---|"]
    for label, row in single.items():
        lines.append(f"| {label} | {row['top1']['found']} | {row['top4']['found']} | {row['top8']['found']} | {row['top20']['found']} |")
    lines += ["", "## Fused lists of 8 passages (what a debate round receives)", "",
              "| Dense | Lexical | Depth each | Per-case limit | Target rank 1 | In top 4 | In top 8 | Distinct cases |", "|---|---|---|---|---|---|---|---|"]
    for name, row in sorted(fused.items(), key=lambda kv: (-kv[1]["target_in_top_8"], -kv[1]["target_in_top_4"])):
        lines.append(f"| {row['dense']} | {row['lexical']} | {row['depth_each']} | {row['per_case_limit'] or 'none'} | {row['target_at_rank_1']} | "
                     f"{row['target_in_top_4']} | {row['target_in_top_8']} | {row['distinct_cases_in_top_8']} |")
    lines += ["", result["note"], ""]
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {os.path.join(OUT_DIR, 'results.json')}")
    print("system as it is:", current)
    print("best fusion:", best[0], best[1])


if __name__ == "__main__":
    main()
