# LexAgent v3.0 | bootstrap.py
"""Shared loader for LexAgent v3.0 (CAP as Primary Authority)."""

import os
import pickle
import chromadb

from config import (
    BM25_INDEX_PATH,
    CHROMA_CORPUS_PATH,
    DRIVE_CHROMA_CORPUS_PATH,
    LOCAL_CHROMA_CORPUS_PATH,
    KNOWN_CASES_PATH,
    CORPUS_JSON_PATH,
)
from src.utils.chroma_sync import ensure_chroma_ready
from src.data.build_corpus import load_corpus
from src.models.load_model import load_embedder, load_mistral_7b
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.novel.cce import CitationConfidenceEngine
from src.novel.ddc import DynamicDebateController
from src.novel.adaptive_memory import AdaptiveMemory


def load_lexagent_components(load_llm: bool = True):
    """Load primary CAP database, retrievers, models, and novel components."""
    if not os.path.exists(CORPUS_JSON_PATH):
        raise FileNotFoundError(f"Corpus chunks not found at {CORPUS_JSON_PATH}. Run build_db.py first.")
    if not os.path.exists(BM25_INDEX_PATH):
        raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}. Run build_db.py first.")
    if not os.path.exists(KNOWN_CASES_PATH):
        raise FileNotFoundError(f"Known cases catalog not found at {KNOWN_CASES_PATH}. Run build_db.py first.")

    corpus = load_corpus(CORPUS_JSON_PATH)

    with open(KNOWN_CASES_PATH, "rb") as handle:
        known_cases = pickle.load(handle)

    embedder = load_embedder()
    active_path = ensure_chroma_ready(LOCAL_CHROMA_CORPUS_PATH, DRIVE_CHROMA_CORPUS_PATH)
    client = chromadb.PersistentClient(path=active_path)
    colls = [c.name for c in client.list_collections()]
    collection_name = "cap_authorities" if ("cap_authorities" in colls or not colls) else colls[0]
    collection = client.get_or_create_collection(collection_name)

    dense = DenseRetriever(collection, embedder)
    bm25 = BM25Retriever.load_index(BM25_INDEX_PATH, corpus)
    hybrid = HybridRetriever(dense, bm25)

    llm = load_mistral_7b()[2] if load_llm else None

    cce = CitationConfidenceEngine(collection, embedder, KNOWN_CASES_PATH)
    ddc = DynamicDebateController()
    memory = AdaptiveMemory()

    return {
        "corpus": corpus,
        "known_cases": known_cases,
        "embedder": embedder,
        "collection": collection,
        "hybrid": hybrid,
        "llm": llm,
        "cce": cce,
        "ddc": ddc,
        "memory": memory,
        "known_cases_path": KNOWN_CASES_PATH,
    }
