# LexAgent v2.0 | Phase 1 | data/__init__.py
"""Data pipeline: corpus schema, download, and indexing."""
from .corpus_schema import CorpusDocument, CitationRecord
from .download_datasets import download_all_datasets
from .build_corpus import build_legal_corpus, build_chroma_index
