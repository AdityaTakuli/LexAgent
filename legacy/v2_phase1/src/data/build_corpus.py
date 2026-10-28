# LexAgent v2.0 | Phase 1 | build_corpus.py
"""Corpus construction and ChromaDB indexing for LexAgent v2.0.

Merges SCOTUS opinions and CaseHOLD holdings into a unified legal corpus,
extracts case names via regex, builds ChromaDB dense index and BM25 index,
and persists known case names for the Citation Confidence Engine (CCE).
"""

import hashlib
import json
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    CHROMA_CORPUS_PATH,
    CORPUS_JSON_PATH,
    DRIVE_BASE,
    KNOWN_CASES_PATH,
)
from src.data.corpus_schema import CorpusDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════
# REGEX PATTERNS FOR CASE NAME EXTRACTION
# ══════════════════════════════════════════════════════
# Matches patterns like "Smith v. Jones", "United States v. Nixon"
# Also handles "et al.", "Inc.", "Corp.", etc.
CASE_NAME_PATTERN = re.compile(
    r'([A-Z][A-Za-z\.\,\s]+?)\s+v\.\s+([A-Z][A-Za-z\.\,\s]+?)(?:[,\.;\)]|\d{3}|$)',
    re.MULTILINE
)


def _extract_case_name(text: str) -> str:
    """Extract the first case name (X v. Y pattern) from text.
    
    Args:
        text: Legal text that may contain case citations.
    
    Returns:
        Extracted case name string, or empty string if none found.
    """
    match = CASE_NAME_PATTERN.search(text)
    if match:
        plaintiff = match.group(1).strip().rstrip(',')
        defendant = match.group(2).strip().rstrip(',')
        return f"{plaintiff} v. {defendant}"
    return ""


def _generate_doc_id(source: str, index: int, text: str) -> str:
    """Generate a unique document ID.
    
    Uses source + index + text hash for deterministic deduplication.
    
    Args:
        source: Dataset source name.
        index: Document index within source.
        text: Document text (used for hash).
    
    Returns:
        Unique document ID string.
    """
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"{source}_{index}_{text_hash}"


def build_legal_corpus(dataset_paths: Dict[str, str]) -> List[CorpusDocument]:
    """Build unified legal corpus from downloaded datasets.
    
    Merges SCOTUS opinions and CaseHOLD holdings. LegalBench is reserved
    for evaluation only and is NOT added to the retrieval corpus.
    
    Target: ~5000 documents after deduplication.
    
    Args:
        dataset_paths: Dict from download_all_datasets() with keys:
            "scotus", "casehold", "legalbench".
    
    Returns:
        List of deduplicated CorpusDocument instances.
    """
    corpus: List[CorpusDocument] = []
    seen_ids: Set[str] = set()
    
    # ── Process SCOTUS opinions ──────────────────────────────────────
    scotus_path = dataset_paths.get("scotus", "")
    if scotus_path and os.path.exists(scotus_path):
        logger.info("Processing SCOTUS opinions...")
        with open(scotus_path, 'r', encoding='utf-8') as f:
            scotus_data = json.load(f)
        
        for i, entry in enumerate(scotus_data):
            text = entry.get("text", "")
            if not text or len(text) < 100:  # Skip very short entries
                continue
            
            # Truncate very long opinions to first 2000 chars for embedding
            truncated_text = text[:2000] if len(text) > 2000 else text
            
            doc_id = _generate_doc_id("SCOTUS", i, truncated_text)
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            
            case_name = _extract_case_name(text)
            
            doc = CorpusDocument(
                doc_id=doc_id,
                text=truncated_text,
                source="SCOTUS",
                case_name=case_name,
                year=0,  # FairLex SCOTUS doesn't include year directly
                court="Supreme Court of the United States",
                metadata={"original_length": len(text)},
            )
            corpus.append(doc)
        
        logger.info("  → SCOTUS: %d documents added", len([d for d in corpus if d.source == 'SCOTUS']))
    else:
        logger.warning("SCOTUS dataset not found at: %s", scotus_path)
    
    # ── Process CaseHOLD holdings ────────────────────────────────────
    casehold_path = dataset_paths.get("casehold", "")
    if casehold_path and os.path.exists(casehold_path):
        logger.info("Processing CaseHOLD holdings...")
        with open(casehold_path, 'r', encoding='utf-8') as f:
            casehold_data = json.load(f)
        
        # Take up to 2000 entries to reach ~5000 total corpus size
        max_casehold = 2000
        count = 0
        
        for i, entry in enumerate(casehold_data):
            if count >= max_casehold:
                break
            
            # CaseHOLD has "citing_prompt" and holding options
            text = entry.get("citing_prompt", "")
            if not text:
                continue
            
            # Append the correct holding if available
            label = entry.get("label", -1)
            holding_keys = [
                "holding_0", "holding_1", "holding_2",
                "holding_3", "holding_4"
            ]
            if isinstance(label, int) and 0 <= label < len(holding_keys):
                correct_holding = entry.get(holding_keys[label], "")
                if correct_holding:
                    text = f"{text} HOLDING: {correct_holding}"
            
            doc_id = _generate_doc_id("CaseHOLD", i, text)
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            
            case_name = _extract_case_name(text)
            
            doc = CorpusDocument(
                doc_id=doc_id,
                text=text,
                source="CaseHOLD",
                case_name=case_name,
                year=0,
                court="",
                metadata={"label": label},
            )
            corpus.append(doc)
            count += 1
        
        logger.info("  → CaseHOLD: %d documents added", count)
    else:
        logger.warning("CaseHOLD dataset not found at: %s", casehold_path)
    
    logger.info("Total corpus size: %d documents (after deduplication)", len(corpus))
    
    # ── Save corpus to Drive ─────────────────────────────────────────
    corpus_dicts = [doc.to_dict() for doc in corpus]
    os.makedirs(os.path.dirname(CORPUS_JSON_PATH), exist_ok=True)
    with open(CORPUS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(corpus_dicts, f, ensure_ascii=False, indent=2)
    logger.info("Corpus saved: %s", CORPUS_JSON_PATH)
    
    return corpus


def build_chroma_index(
    corpus: List[CorpusDocument],
    embedder: Any,
) -> Any:
    """Build ChromaDB dense vector index from corpus.
    
    Creates a persistent ChromaDB collection with Legal-BERT embeddings.
    Batch-embeds documents in groups of 64 for memory efficiency.
    
    # NOVEL — ChromaDB index feeds HybridRetriever → entity_map → CCE
    
    Args:
        corpus: List of CorpusDocument instances.
        embedder: SentenceTransformer model (Legal-BERT).
    
    Returns:
        ChromaDB Collection handle.
    
    Raises:
        RuntimeError: If ChromaDB initialization fails.
    """
    import chromadb
    
    logger.info("Building ChromaDB index at: %s", CHROMA_CORPUS_PATH)
    os.makedirs(CHROMA_CORPUS_PATH, exist_ok=True)
    
    try:
        client = chromadb.PersistentClient(path=CHROMA_CORPUS_PATH)
    except Exception as e:
        logger.error("ChromaDB initialization failed: %s", e)
        print(f"FATAL: ChromaDB could not initialize at {CHROMA_CORPUS_PATH}: {e}")
        raise RuntimeError(f"ChromaDB init failed: {e}")
    
    # Delete existing collection if rebuilding
    try:
        client.delete_collection("legal_corpus")
        logger.info("Deleted existing 'legal_corpus' collection for rebuild.")
    except Exception:
        pass  # Collection doesn't exist yet
    
    collection = client.create_collection(
        name="legal_corpus",
        metadata={"hnsw:space": "cosine"},
    )
    
    # ── Batch embed and insert ───────────────────────────────────────
    batch_size = 64
    total = len(corpus)
    known_cases: Set[str] = set()
    
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = corpus[start_idx:end_idx]
        
        # Prepare batch data
        texts = [doc.text for doc in batch]
        ids = [doc.doc_id for doc in batch]
        metadatas = [
            {
                "case_name": doc.case_name,
                "year": doc.year,
                "court": doc.court,
                "source": doc.source,
            }
            for doc in batch
        ]
        
        # Embed batch using Legal-BERT
        embeddings = embedder.encode(
            texts,
            show_progress_bar=False,
            batch_size=batch_size,
        ).tolist()
        
        # Add to ChromaDB
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        # Collect known case names for CCE
        # NOVEL — known_cases set feeds Citation Confidence Engine
        for doc in batch:
            if doc.case_name:
                known_cases.add(doc.case_name)
        
        # Progress reporting every 500 docs
        if (start_idx + batch_size) % 500 < batch_size or end_idx == total:
            logger.info(
                "  ChromaDB indexing: %d / %d documents (%.1f%%)",
                end_idx, total, 100.0 * end_idx / total
            )
    
    logger.info("ChromaDB index built: %d documents, %d known case names",
                collection.count(), len(known_cases))
    
    # ── Save known case names for CCE (Phase 3) ─────────────────────
    # NOVEL — known_cases.pkl enables exact-match lookup in CCE
    os.makedirs(os.path.dirname(KNOWN_CASES_PATH), exist_ok=True)
    with open(KNOWN_CASES_PATH, 'wb') as f:
        pickle.dump(known_cases, f)
    logger.info("Known cases saved: %d entries → %s", len(known_cases), KNOWN_CASES_PATH)
    
    return collection
