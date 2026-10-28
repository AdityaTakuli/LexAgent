# LexAgent v2.0 | Phase 1 | hybrid_retriever.py
"""Hybrid Retrieval Engine — NOVEL COMPONENT of LexAgent v2.0.

Combines dense (Legal-BERT + ChromaDB) and sparse (BM25) retrieval
via Reciprocal Rank Fusion (RRF). Also builds an entity map of known
case names from retrieved results for the Citation Confidence Engine (CCE).

Key innovation: The entity map bridges retrieval and citation verification,
enabling the CCE to perform exact-match lookups against retrieved cases.

RRF Formula: score(d) = Σ 1/(k + rank_i(d)) for each result list i
where k is the RRF constant (default: 60) that controls rank sensitivity.
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import RRF_K, TOP_K_DENSE, TOP_K_BM25
from src.data.corpus_schema import CorpusDocument
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Hybrid dense + sparse retriever with Reciprocal Rank Fusion.
    
    # NOVEL — Core retrieval component of the DVA+ pipeline.
    # Combines neural semantic matching (dense) with exact lexical
    # matching (BM25) to maximize both recall and precision for
    # legal document retrieval.
    
    The entity map output feeds directly into the Citation Confidence
    Engine (CCE) for exact-match citation verification.
    
    Attributes:
        dense: DenseRetriever instance (Legal-BERT + ChromaDB).
        bm25: BM25Retriever instance.
        k: RRF constant controlling rank sensitivity.
    """
    
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        k: int = RRF_K,
    ):
        """Initialize HybridRetriever.
        
        Args:
            dense: Initialized DenseRetriever with ChromaDB collection.
            bm25: Initialized BM25Retriever with built index.
            k: RRF constant (default: 60). Higher k reduces the
                influence of high-ranked documents.
        """
        self.dense = dense
        self.bm25 = bm25
        self.k = k
        logger.info("HybridRetriever initialized (RRF k=%d)", k)
    
    def retrieve(
        self,
        query: str,
        jurisdiction: str = "",
    ) -> List[CorpusDocument]:
        """Retrieve documents using hybrid dense + BM25 with RRF fusion.
        
        # NOVEL — Hybrid retrieval with RRF feeds DVA+ debate pipeline.
        
        Pipeline:
          1. Dense retrieval (Legal-BERT cosine similarity)
          2. BM25 retrieval (lexical keyword matching)
          3. Reciprocal Rank Fusion to merge and re-rank
        
        Args:
            query: Natural language legal query.
            jurisdiction: Optional jurisdiction context (e.g., "US_Federal").
        
        Returns:
            List of top-(TOP_K_DENSE + TOP_K_BM25) CorpusDocuments,
            ranked by RRF score (descending).
        """
        # STEP 1: Dense retrieval
        dense_results = self.dense.retrieve(
            query=query,
            jurisdiction=jurisdiction,
        )
        logger.info("Dense retrieval returned %d results", len(dense_results))
        
        # STEP 2: BM25 retrieval
        bm25_results = self.bm25.retrieve(query=query)
        logger.info("BM25 retrieval returned %d results", len(bm25_results))
        
        # STEP 3: RRF merge
        merged = self._reciprocal_rank_fusion(dense_results, bm25_results)
        
        # Return top-8 (TOP_K_DENSE + TOP_K_BM25)
        top_n = TOP_K_DENSE + TOP_K_BM25
        final_results = merged[:top_n]
        
        logger.info(
            "Hybrid retrieval: query='%s...' → %d results (from %d dense + %d BM25)",
            query[:50], len(final_results), len(dense_results), len(bm25_results)
        )
        return final_results
    
    def _reciprocal_rank_fusion(
        self,
        *result_lists: List[CorpusDocument],
    ) -> List[CorpusDocument]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion.
        
        RRF score for document d:
          score(d) = Σ 1/(k + rank_i(d))
        
        where rank_i(d) is the 1-based rank of d in list i,
        and k is the RRF constant (default 60).
        
        Documents appearing in multiple lists accumulate higher scores.
        Deduplication is performed by doc_id.
        
        Args:
            *result_lists: Variable number of ranked CorpusDocument lists.
        
        Returns:
            Merged list sorted by RRF score descending.
        """
        # Score accumulator: doc_id → RRF score
        rrf_scores: Dict[str, float] = defaultdict(float)
        
        # Document lookup: doc_id → CorpusDocument (keep first occurrence)
        doc_lookup: Dict[str, CorpusDocument] = {}
        
        for result_list in result_lists:
            for rank, doc in enumerate(result_list, start=1):
                # RRF formula: score += 1 / (k + rank)
                rrf_scores[doc.doc_id] += 1.0 / (self.k + rank)
                
                # Store document reference (first occurrence wins)
                if doc.doc_id not in doc_lookup:
                    doc_lookup[doc.doc_id] = doc
        
        # Sort by RRF score descending
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True,
        )
        
        # Build result list with RRF scores in metadata
        merged_results: List[CorpusDocument] = []
        for doc_id in sorted_ids:
            doc = doc_lookup[doc_id]
            # Add RRF score to metadata for debugging/analysis
            enriched_doc = CorpusDocument(
                doc_id=doc.doc_id,
                text=doc.text,
                source=doc.source,
                case_name=doc.case_name,
                year=doc.year,
                court=doc.court,
                metadata={
                    **doc.metadata,
                    "rrf_score": rrf_scores[doc_id],
                },
            )
            merged_results.append(enriched_doc)
        
        logger.info(
            "RRF fusion: %d unique documents (from %d lists)",
            len(merged_results), len(result_lists)
        )
        return merged_results
    
    def get_entity_map(
        self,
        results: List[CorpusDocument],
    ) -> Dict[str, CorpusDocument]:
        """Extract known case names from retrieval results.
        
        # NOVEL — entity map feeds Citation Confidence Engine (CCE)
        
        Builds a lookup dictionary mapping case names to their
        CorpusDocument records. This enables the CCE to perform
        exact-match verification of citations against the retrieved
        evidence base.
        
        Args:
            results: List of retrieved CorpusDocuments.
        
        Returns:
            Dict mapping case_name → CorpusDocument for all results
            that have a non-empty case_name.
        """
        entity_map: Dict[str, CorpusDocument] = {}
        
        for doc in results:
            if doc.case_name:
                # Use first occurrence if duplicate case names exist
                if doc.case_name not in entity_map:
                    entity_map[doc.case_name] = doc
        
        logger.info(
            "Entity map built: %d case names from %d documents",
            len(entity_map), len(results)
        )
        return entity_map
