# LexAgent v2.0 | Phase 1 | dense_retriever.py
"""Dense vector retrieval using ChromaDB and Legal-BERT embeddings.

Performs cosine-similarity search over the Legal-BERT embedded corpus
stored in ChromaDB. Supports optional jurisdiction filtering.
"""

import os
import sys
from typing import Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import TOP_K_DENSE
from src.data.corpus_schema import CorpusDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """ChromaDB-backed dense vector retriever.
    
    Embeds queries using Legal-BERT and retrieves the top-k most
    similar documents from the ChromaDB persistent index.
    
    # NOVEL — Dense results feed into HybridRetriever RRF fusion
    
    Attributes:
        collection: ChromaDB collection handle.
        embedder: SentenceTransformer model for query encoding.
        top_k: Number of documents to retrieve.
    """
    
    def __init__(
        self,
        collection: Any,
        embedder: Any,
        top_k: int = TOP_K_DENSE,
    ):
        """Initialize DenseRetriever.
        
        Args:
            collection: ChromaDB collection containing the legal corpus.
            embedder: SentenceTransformer model (Legal-BERT on CPU).
            top_k: Number of top results to return (default: 4).
        """
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
        """Retrieve top-k documents via dense cosine similarity.
        
        Concatenates the query with jurisdiction context before embedding
        to leverage Legal-BERT's understanding of jurisdictional nuance.
        
        Args:
            query: Natural language legal query.
            jurisdiction: Optional jurisdiction filter (e.g., "US_Federal").
                Concatenated with query for embedding.
        
        Returns:
            List of CorpusDocument instances ranked by cosine similarity.
        """
        # Concatenate query with jurisdiction for richer embedding
        combined_query = f"{query} {jurisdiction}".strip()
        
        # Embed the query using Legal-BERT
        query_embedding = self.embedder.encode(
            [combined_query],
            show_progress_bar=False,
        ).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        # Convert ChromaDB results to CorpusDocument list
        documents: List[CorpusDocument] = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                text = results['documents'][0][i] if results['documents'] else ""
                
                doc = CorpusDocument(
                    doc_id=doc_id,
                    text=text,
                    source=metadata.get("source", "unknown"),
                    case_name=metadata.get("case_name", ""),
                    year=metadata.get("year", 0),
                    court=metadata.get("court", ""),
                    metadata={"distance": results['distances'][0][i]}
                        if results.get('distances') else {},
                )
                documents.append(doc)
        
        logger.info(
            "Dense retrieval: query='%s...' → %d results",
            query[:50], len(documents)
        )
        return documents
