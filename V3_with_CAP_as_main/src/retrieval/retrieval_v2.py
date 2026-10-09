# LexAgent v3.0 | retrieval_v2.py
"""Second retrieval index (repair 7 of the forensic audit).

The audit measured that the dense retriever of the system, Legal-BERT with mean pooling, places the
target opinion among its four results for 39 of 96 questions, against 74 for BM25: Legal-BERT is a
masked language model and was never trained to retrieve. This module retrieves with

  dense     an embedding model trained for retrieval (config.RETRIEVAL_V2_MODEL), on the question
            alone: the jurisdiction string is no longer appended to it;
  lexical   BM25 over word tokens (punctuation removed) instead of whitespace tokens;
  fusion    reciprocal-rank fusion as before, with at most config.RETRIEVAL_V2_PER_CASE_LIMIT
            passages of one case in the list, so that one long opinion cannot fill the context.

The Legal-BERT collection is not touched. The citation checker was validated on it and keeps using
it. The index files are built once by runs/build_retrieval_v2.py and are tied to the corpus by its
passage count and a hash of the passage identifiers.
"""

import hashlib
import json
import os
import re
from typing import Dict, List

import numpy as np

from config import (
    RETRIEVAL_V2_DIR, RETRIEVAL_V2_MAX_TOKENS, RETRIEVAL_V2_MODEL, RETRIEVAL_V2_PER_CASE_LIMIT,
    RETRIEVAL_V2_QUERY_PREFIX, RRF_K, TOP_K_BM25, TOP_K_DENSE,
)
from src.data.corpus_schema import CorpusDocument
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDINGS_FILE = "dense_embeddings.float16.npy"
BM25_FILE = "bm25_word_index.pkl"
MANIFEST_FILE = "manifest.json"
# Candidates taken from each ranking before fusion: twice the number of passages each ranking
# contributed before, because the per-case limit removes some of them.
CANDIDATE_DEPTH = 2 * max(TOP_K_DENSE, TOP_K_BM25)
# When one long opinion dominates both rankings, the per-case limit leaves fewer than eight passages
# (as few as two). The list is then filled, under the same limit, from this many candidates of each
# ranking. The passages ranked from the first CANDIDATE_DEPTH keep their places at the top.
FILL_DEPTH = 50


def word_tokens(text: str) -> str:
    """Lower-cased word characters separated by single spaces (the BM25 tokenisation of this index)."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def parent_of(doc_id: str) -> str:
    return str(doc_id).split("::chunk::")[0]


def corpus_fingerprint(corpus: List[CorpusDocument]) -> dict:
    digest = hashlib.sha256()
    for doc in corpus:
        digest.update(str(doc.doc_id).encode("utf-8"))
        digest.update(b"\n")
    return {"passages": len(corpus), "doc_ids_sha256": digest.hexdigest()}


def fuse(rankings: List[List[int]], top: int, per_case: int, parents: List[str], k: int = RRF_K) -> List[tuple]:
    """Reciprocal-rank fusion of ranked lists of passage indices, at most `per_case` passages per case.

    Returns (index, score) pairs, best first.
    """
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
    kept, per_parent = [], {}
    for index in sorted(scores, key=lambda i: (-scores[i], i)):
        parent = parents[index]
        if per_case and per_parent.get(parent, 0) >= per_case:
            continue
        per_parent[parent] = per_parent.get(parent, 0) + 1
        kept.append((index, scores[index]))
        if len(kept) == top:
            break
    return kept


def fuse_and_fill(rankings: List[List[int]], top: int, per_case: int, parents: List[str], depth: int, k: int = RRF_K) -> List[tuple]:
    """Fusion of the first `depth` results of each ranking; if the per-case limit leaves fewer than
    `top` passages, the rest is taken, under the same limit, from the fusion of the whole rankings."""
    kept = fuse([ranking[:depth] for ranking in rankings], top, per_case, parents, k)
    if len(kept) >= top:
        return kept
    chosen = {index for index, _ in kept}
    per_parent: Dict[str, int] = {}
    for index in chosen:
        per_parent[parents[index]] = per_parent.get(parents[index], 0) + 1
    for index, score in fuse(rankings, len(parents), 0, parents, k):
        if index in chosen:
            continue
        parent = parents[index]
        if per_case and per_parent.get(parent, 0) >= per_case:
            continue
        per_parent[parent] = per_parent.get(parent, 0) + 1
        kept.append((index, score))
        if len(kept) == top:
            break
    return kept


class FusedRetrieverV2(HybridRetriever):
    """Same interface as HybridRetriever: retrieve(query, jurisdiction) and get_entity_map(results)."""

    def __init__(self, corpus: List[CorpusDocument], matrix, encoder, bm25_word: BM25Retriever,
                 top: int = TOP_K_DENSE + TOP_K_BM25, depth: int = CANDIDATE_DEPTH,
                 per_case: int = RETRIEVAL_V2_PER_CASE_LIMIT, k: int = RRF_K, fill_depth: int = FILL_DEPTH):
        self.corpus = corpus
        self.matrix = matrix          # passages x dimensions, rows L2-normalised (torch tensor or numpy array)
        self.encoder = encoder        # .encode(list of str, normalize_embeddings=True) -> vectors
        self.bm25 = bm25_word
        self.fill_depth = max(fill_depth, depth)
        self.bm25.top_k = self.fill_depth
        self.top, self.depth, self.per_case, self.k = top, depth, per_case, k
        self.parents = [parent_of(doc.doc_id) for doc in corpus]
        self.index_of = {doc.doc_id: i for i, doc in enumerate(corpus)}
        self.dense = None             # there is no Legal-BERT retriever inside this one
        logger.info("FusedRetrieverV2 initialized (%d passages, depth %d each, at most %d per case, top %d)",
                    len(corpus), depth, per_case, top)

    # ── the two rankings ────────────────────────────────────────────────────
    def dense_ranking(self, query: str) -> List[int]:
        vector = self.encoder.encode([RETRIEVAL_V2_QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False)
        if hasattr(self.matrix, "device"):  # torch tensor
            import torch

            with torch.no_grad():
                q = torch.as_tensor(np.asarray(vector), device=self.matrix.device, dtype=self.matrix.dtype)
                return torch.topk((self.matrix @ q.T).squeeze(1), self.fill_depth).indices.tolist()
        scores = self.matrix @ np.asarray(vector, dtype=self.matrix.dtype)[0]
        return np.argsort(-scores)[:self.fill_depth].tolist()

    def lexical_ranking(self, query: str) -> List[int]:
        return [self.index_of[doc.doc_id] for doc in self.bm25.retrieve(word_tokens(query)) if doc.doc_id in self.index_of]

    # ── HybridRetriever interface ───────────────────────────────────────────
    def retrieve(self, query: str, jurisdiction: str = "") -> List[CorpusDocument]:
        """`jurisdiction` is accepted for compatibility and is not part of either query."""
        dense, lexical = self.dense_ranking(query), self.lexical_ranking(query)
        results = []
        for index, score in fuse_and_fill([dense, lexical], self.top, self.per_case, self.parents, self.depth, self.k):
            doc = self.corpus[index]
            results.append(CorpusDocument(
                doc_id=doc.doc_id, text=doc.text, source=doc.source, case_name=doc.case_name, year=doc.year, court=doc.court,
                volume=doc.volume, reporter=doc.reporter, first_page=doc.first_page,
                metadata={**(doc.metadata or {}), "rrf_score": score, "retrieval": "v2",
                          "in_dense": index in dense[:self.depth], "in_lexical": index in lexical[:self.depth]}))
        logger.info("Retrieval v2: query='%s...' -> %d results (%d distinct cases)",
                    query[:50], len(results), len({parent_of(d.doc_id) for d in results}))
        return results


def index_paths(directory: str = RETRIEVAL_V2_DIR) -> dict:
    return {name: os.path.join(directory, name).replace("\\", "/") for name in (EMBEDDINGS_FILE, BM25_FILE, MANIFEST_FILE)}


def load_manifest(directory: str = RETRIEVAL_V2_DIR) -> dict:
    path = index_paths(directory)[MANIFEST_FILE]
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_retriever_v2(corpus: List[CorpusDocument], directory: str = RETRIEVAL_V2_DIR, device: str = None) -> FusedRetrieverV2:
    """Load the index built by runs/build_retrieval_v2.py and check that it belongs to this corpus."""
    paths, manifest = index_paths(directory), load_manifest(directory)
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"retrieval v2 index incomplete in {directory}: {missing}. Run runs/build_retrieval_v2.py")
    fingerprint = corpus_fingerprint(corpus)
    for part in ("dense", "lexical"):
        if (manifest.get(part) or {}).get("corpus") != fingerprint:
            raise ValueError(f"the {part} part of the retrieval v2 index was built from another corpus; rebuild it")
    if manifest["dense"]["model"] != RETRIEVAL_V2_MODEL:
        raise ValueError(f"dense index was built with {manifest['dense']['model']}, configuration names {RETRIEVAL_V2_MODEL}")

    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SentenceTransformer(RETRIEVAL_V2_MODEL, device=device)
    encoder.max_seq_length = RETRIEVAL_V2_MAX_TOKENS
    stored = np.load(paths[EMBEDDINGS_FILE], mmap_mode="r")
    if stored.shape[0] != len(corpus):
        raise ValueError(f"dense index holds {stored.shape[0]} vectors, the corpus {len(corpus)} passages")
    if device == "cuda":
        matrix = torch.as_tensor(np.ascontiguousarray(stored), device="cuda")  # float16 on the GPU, about 0.8 GB
        encoder.half()
    else:
        matrix = np.asarray(stored, dtype=np.float32)
    bm25 = BM25Retriever.load_index(paths[BM25_FILE], corpus)
    return FusedRetrieverV2(corpus, matrix, encoder, bm25)
