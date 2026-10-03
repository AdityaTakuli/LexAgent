# LexAgent v3.0 | bm25_retriever.py
"""Memory-efficient, scalable BM25Okapi sparse retriever over legal corpus.

Uses a streaming inverted index to support 500k+ chunks with minimal RAM (<500MB),
preventing Linux kernel OOM (Out-Of-Memory) termination.
"""

from collections import Counter, defaultdict
import gc
import math
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import TOP_K_BM25, BM25_INDEX_PATH
from src.data.corpus_schema import CorpusDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """High-efficiency BM25Okapi sparse retriever with inverted index."""

    def __init__(
        self,
        corpus: List[CorpusDocument],
        top_k: int = TOP_K_BM25,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.corpus = corpus
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.doc_len: np.ndarray = np.empty(0, dtype=np.int32)
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        # Inverted index: term -> (doc_ids_array, tfs_array)
        self.inverted_index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        if corpus:
            self._build_index()

    def _build_index(self) -> None:
        """Build scalable inverted index without holding 200M+ tokens in RAM."""
        N = len(self.corpus)
        logger.info("Building memory-efficient BM25 index over %d documents...", N)
        if N == 0:
            return

        doc_lengths = np.zeros(N, dtype=np.int32)
        postings = defaultdict(lambda: ([], []))
        df = defaultdict(int)

        report_step = max(50000, N // 10)
        t0 = time.time()

        for idx, doc in enumerate(self.corpus):
            text = getattr(doc, "text", "") if hasattr(doc, "text") else (doc.get("text", "") if isinstance(doc, dict) else "")
            tokens = text.lower().split()
            doc_lengths[idx] = len(tokens)

            if tokens:
                counts = Counter(tokens)
                for term, count in counts.items():
                    p_docs, p_tfs = postings[term]
                    p_docs.append(idx)
                    p_tfs.append(count)
                    df[term] += 1

            if (idx + 1) % report_step == 0 or (idx + 1) == N:
                elapsed = time.time() - t0
                pct = 100.0 * (idx + 1) / N
                print(f"     BM25 Inverted Index: {idx + 1:,}/{N:,} docs ({pct:.1f}%) in {elapsed:.1f}s...")

        self.doc_len = doc_lengths
        self.avgdl = float(np.mean(doc_lengths)) if N > 0 else 0.0

        # Compute IDFs
        negative_idfs = []
        for term, freq in df.items():
            val = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = val
            if val < 0:
                negative_idfs.append(term)

        avg_idf = sum(v for v in self.idf.values() if v > 0) / len(self.idf) if self.idf else 0.0
        eps_idf = self.epsilon * avg_idf
        for term in negative_idfs:
            self.idf[term] = eps_idf

        # Convert postings lists to compact numpy arrays
        for term, (p_docs, p_tfs) in postings.items():
            self.inverted_index[term] = (
                np.array(p_docs, dtype=np.int32),
                np.array(p_tfs, dtype=np.int16),
            )

        del postings
        del df
        gc.collect()
        logger.info("BM25 index built successfully (%d unique terms, avgdl=%.1f).", len(self.inverted_index), self.avgdl)

    def retrieve(self, query: str) -> List[CorpusDocument]:
        """Retrieve top-k documents via BM25 scoring."""
        if not self.inverted_index or not self.corpus:
            return []

        tokens = query.lower().split()
        if not tokens:
            return []

        doc_scores: Dict[int, float] = defaultdict(float)

        for q in tokens:
            if q not in self.inverted_index:
                continue
            q_idf = self.idf.get(q, 0.0)
            doc_ids, tfs = self.inverted_index[q]

            dls = self.doc_len[doc_ids]
            denoms = tfs + self.k1 * (1.0 - self.b + self.b * (dls / self.avgdl))
            scores = q_idf * (tfs * (self.k1 + 1.0)) / denoms

            for doc_id, score in zip(doc_ids, scores):
                doc_scores[int(doc_id)] += float(score)

        if not doc_scores:
            return []

        sorted_pairs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]

        results: List[CorpusDocument] = []
        for doc_idx, score in sorted_pairs:
            if score > 0:
                doc = self.corpus[doc_idx]
                if isinstance(doc, dict):
                    doc = CorpusDocument.from_dict(doc)
                doc_metadata = getattr(doc, "metadata", {}) or {}
                doc_with_score = CorpusDocument(
                    doc_id=getattr(doc, "doc_id", str(doc_idx)),
                    text=getattr(doc, "text", ""),
                    source=getattr(doc, "source", "CAP"),
                    case_name=getattr(doc, "case_name", ""),
                    year=getattr(doc, "year", 0),
                    court=getattr(doc, "court", ""),
                    volume=str(getattr(doc, "volume", "")),
                    reporter=getattr(doc, "reporter", "U.S."),
                    first_page=int(getattr(doc, "first_page", 0) or 0),
                    metadata={**doc_metadata, "bm25_score": float(score)},
                )
                results.append(doc_with_score)

        logger.info(
            "BM25 retrieval: query='%s...' -> %d results",
            query[:50], len(results)
        )
        return results

    def save_index(self, path: str = BM25_INDEX_PATH) -> None:
        """Save BM25 index to pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        index_data = {
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "idf": self.idf,
            "inverted_index": self.inverted_index,
            "top_k": self.top_k,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
        }
        with open(path, 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved: %s (%.2f MB)", path, os.path.getsize(path) / 1e6)

    @classmethod
    def load_index(
        cls,
        path: str,
        corpus: List[CorpusDocument],
    ) -> 'BM25Retriever':
        """Load BM25 index from pickle file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index not found: {path}")

        with open(path, 'rb') as f:
            index_data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.corpus = corpus
        instance.top_k = index_data.get("top_k", TOP_K_BM25)
        instance.k1 = index_data.get("k1", 1.5)
        instance.b = index_data.get("b", 0.75)
        instance.epsilon = index_data.get("epsilon", 0.25)
        instance.doc_len = index_data.get("doc_len", np.empty(0, dtype=np.int32))
        instance.avgdl = index_data.get("avgdl", 0.0)
        instance.idf = index_data.get("idf", {})
        instance.inverted_index = index_data.get("inverted_index", {})

        # Backwards compatibility fallback if old format exists
        if "tokenized_corpus" in index_data and not instance.inverted_index:
            from rank_bm25 import BM25Okapi
            instance.bm25 = BM25Okapi(index_data["tokenized_corpus"])
        else:
            instance.bm25 = None

        logger.info("BM25 index loaded from: %s (%d vocabulary terms)", path, len(instance.inverted_index))
        return instance

