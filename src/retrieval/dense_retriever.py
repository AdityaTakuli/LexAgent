# LexAgent v3.0 | dense_retriever.py
"""Dense vector retrieval using ChromaDB and Legal-BERT embeddings."""

import os
import sys
from typing import Any, List, Optional

from config import TOP_K_DENSE
from src.data.corpus_schema import CorpusDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """ChromaDB-backed dense vector retriever over CAP authorities."""
    
    def __init__(
        self,
        collection: Any,
        embedder: Any,
        top_k: int = TOP_K_DENSE,
    ):
        self.collection = collection
        self.embedder = embedder
        self.top_k = top_k
        logger.info(
            "DenseRetriever initialized (top_k=%d, collection=%s)",
            top_k, collection.name if hasattr(collection, 'name') else 'unknown'
        )
    
    def retrieve(
        self,
        query: str,
        jurisdiction: str = "",
    ) -> List[CorpusDocument]:
        """Retrieve top-k documents via dense cosine similarity."""
        combined_query = f"{query} {jurisdiction}".strip()
        
        query_embedding = self.embedder.encode(
            [combined_query],
            show_progress_bar=False,
        ).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(self.top_k, max(1, self.collection.count())),
            include=["documents", "metadatas", "distances"],
        )
        
        documents: List[CorpusDocument] = []
        if results and results.get('ids') and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                text = results['documents'][0][i] if results.get('documents') else ""
                
                doc = CorpusDocument(
                    doc_id=doc_id,
                    text=text,
                    source=metadata.get("source", "CAP"),
                    case_name=metadata.get("case_name", ""),
                    year=metadata.get("year", 0),
                    court=metadata.get("court", ""),
                    volume=str(metadata.get("volume", "")),
                    reporter=metadata.get("reporter", "U.S."),
                    first_page=int(metadata.get("first_page", 0) or 0),
                    metadata={"distance": results['distances'][0][i]}
                        if results.get('distances') else {},
                )
                documents.append(doc)
        
        logger.info(
            "Dense retrieval: query='%s...' -> %d results",
            query[:50], len(documents)
        )
        return documents
