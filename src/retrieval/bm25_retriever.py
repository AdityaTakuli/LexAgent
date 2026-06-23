# LexAgent v2.0 | Phase 1 | bm25_retriever.py
"""BM25 sparse retrieval for lexical matching.

Provides traditional keyword-based retrieval using BM25Okapi.
Complementary to dense retrieval — captures exact keyword matches
that neural embeddings may miss (e.g., statute numbers, case IDs).
"""

import os
import pickle
import sys
from typing import Any, ClassVar, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import TOP_K_BM25, BM25_INDEX_PATH
from src.data.corpus_schema import CorpusDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """BM25-based sparse retriever over tokenized corpus.
    
    Tokenizes documents via whitespace + lowercasing and builds a
    BM25Okapi index. Supports serialization to/from pickle for
    persistence across Colab sessions.
    
    # NOVEL — BM25 results feed into HybridRetriever RRF fusion
    
    Attributes:
        corpus: List of CorpusDocument instances.
        top_k: Number of documents to retrieve.
        bm25: BM25Okapi index instance.
        tokenized_corpus: List of tokenized document texts.
    """
    
    def __init__(
        self,
        corpus: List[CorpusDocument],
        top_k: int = TOP_K_BM25,
    ):
        """Initialize BM25Retriever and build index.
        
        Args:
            corpus: Full legal corpus as List[CorpusDocument].
            top_k: Number of top results to return (default: 4).
        """
        self.corpus = corpus
        self.top_k = top_k
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        
        # Build the BM25 index on initialization
        self._build_index()
    
    def _build_index(self) -> None:
        """Build BM25Okapi index from corpus.
        
        Tokenizes all documents using whitespace splitting and
        lowercasing. This simple tokenization works well for legal
        text where exact term matching is important.
        """
        logger.info("Building BM25 index over %d documents...", len(self.corpus))
        
        self.tokenized_corpus = [
            doc.text.lower().split() for doc in self.corpus
        ]
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built successfully.")
    
    def retrieve(self, query: str) -> List[CorpusDocument]:
        """Retrieve top-k documents via BM25 scoring.
        
        Args:
            query: Natural language legal query.
        
        Returns:
            List of CorpusDocument instances ranked by BM25 score.
        """
        if self.bm25 is None:
            logger.error("BM25 index not built. Call _build_index() first.")
            return []
        
        # Tokenize query using same scheme as corpus
        tokenized_query = query.lower().split()
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        
        results: List[CorpusDocument] = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                doc = self.corpus[idx]
                # Add BM25 score to metadata for debugging
                doc_with_score = CorpusDocument(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    source=doc.source,
                    case_name=doc.case_name,
                    year=doc.year,
                    court=doc.court,
                    metadata={**doc.metadata, "bm25_score": float(scores[idx])},
                )
                results.append(doc_with_score)
        
        logger.info(
            "BM25 retrieval: query='%s...' → %d results",
            query[:50], len(results)
        )
        return results
    
    def save_index(self, path: str = BM25_INDEX_PATH) -> None:
        """Save BM25 index to pickle file on Drive.
        
        Saves the tokenized corpus and BM25 parameters for fast
        reload without re-tokenizing the entire corpus.
        
        Args:
            path: File path for the pickle file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        index_data = {
            "tokenized_corpus": self.tokenized_corpus,
            "top_k": self.top_k,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info("BM25 index saved: %s (%.2f MB)",
                     path, os.path.getsize(path) / 1e6)
    
    @classmethod
    def load_index(
        cls,
        path: str,
        corpus: List[CorpusDocument],
    ) -> 'BM25Retriever':
        """Load BM25 index from pickle file.
        
        Reconstructs the BM25Okapi index from saved tokenized corpus
        without re-processing all documents.
        
        Args:
            path: Path to the saved pickle file.
            corpus: The original corpus (needed for document lookup).
        
        Returns:
            Initialized BM25Retriever with loaded index.
        
        Raises:
            FileNotFoundError: If pickle file doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index not found: {path}")
        
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        # Create instance without rebuilding index
        instance = cls.__new__(cls)
        instance.corpus = corpus
        instance.top_k = index_data.get("top_k", TOP_K_BM25)
        instance.tokenized_corpus = index_data["tokenized_corpus"]
        instance.bm25 = BM25Okapi(instance.tokenized_corpus)
        
        logger.info("BM25 index loaded from: %s", path)
        return instance
