# LexAgent v3.0 | hybrid_retriever.py
"""Hybrid Retrieval Engine over CAP Authority Database.

Combines dense (Legal-BERT + ChromaDB) and sparse (BM25) retrieval
via Reciprocal Rank Fusion (RRF). Builds an entity map of known case names
and official citations for the Citation Confidence Engine (CCE).
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

from config import RRF_K, TOP_K_DENSE, TOP_K_BM25
from src.data.corpus_schema import CorpusDocument
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Hybrid dense + sparse retriever with Reciprocal Rank Fusion."""
    
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        k: int = RRF_K,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.k = k
        logger.info("HybridRetriever initialized (RRF k=%d)", k)
    
    def retrieve(
        self,
        query: str,
        jurisdiction: str = "",
    ) -> List[CorpusDocument]:
        """Retrieve documents using hybrid dense + BM25 with RRF fusion."""
        dense_results = self.dense.retrieve(query=query, jurisdiction=jurisdiction)
        bm25_results = self.bm25.retrieve(query=query)
        
        merged = self._reciprocal_rank_fusion(dense_results, bm25_results)
        top_n = TOP_K_DENSE + TOP_K_BM25
        final_results = merged[:top_n]
        
        logger.info(
            "Hybrid retrieval: query='%s...' -> %d results (from %d dense + %d BM25)",
            query[:50], len(final_results), len(dense_results), len(bm25_results)
        )
        return final_results
    
    def _reciprocal_rank_fusion(
        self,
        *result_lists: List[CorpusDocument],
    ) -> List[CorpusDocument]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion."""
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_lookup: Dict[str, CorpusDocument] = {}
        
        for result_list in result_lists:
            for rank, doc in enumerate(result_list, start=1):
                if isinstance(doc, dict):
                    doc = CorpusDocument.from_dict(doc)
                rrf_scores[doc.doc_id] += 1.0 / (self.k + rank)
                if doc.doc_id not in doc_lookup:
                    doc_lookup[doc.doc_id] = doc
        
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True,
        )
        
        merged_results: List[CorpusDocument] = []
        for doc_id in sorted_ids:
            doc = doc_lookup[doc_id]
            enriched_doc = CorpusDocument(
                doc_id=doc.doc_id,
                text=doc.text,
                source=doc.source,
                case_name=doc.case_name,
                year=doc.year,
                court=doc.court,
                volume=doc.volume,
                reporter=doc.reporter,
                first_page=doc.first_page,
                metadata={
                    **doc.metadata,
                    "rrf_score": rrf_scores[doc_id],
                },
            )
            merged_results.append(enriched_doc)
        
        return merged_results
    
    def get_entity_map(
        self,
        results: List[CorpusDocument],
    ) -> Dict[str, CorpusDocument]:
        """Extract known case names and official citations from retrieval results."""
        entity_map: Dict[str, CorpusDocument] = {}
        
        for doc in results:
            if doc.case_name and doc.case_name not in entity_map:
                entity_map[doc.case_name] = doc
            
            # Map official reporter citation if present (e.g., "410 U.S. 113")
            if doc.volume and doc.first_page:
                cite_str = f"{doc.volume} {doc.reporter or 'U.S.'} {doc.first_page}".strip()
                if cite_str not in entity_map:
                    entity_map[cite_str] = doc
                    
            citations = doc.metadata.get("citations", [])
            for c in citations:
                if isinstance(c, dict) and c.get("cite"):
                    entity_map[c["cite"]] = doc
        
        logger.info(
            "Entity map built: %d lookup keys from %d documents",
            len(entity_map), len(results)
        )
        return entity_map
