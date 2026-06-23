# LexAgent v2.0 | Phase 1 | config.py
"""
Central configuration for LexAgent v2.0 — DVA+ Protocol.
All paths, model names, retrieval parameters, and novel component thresholds.
"""

import os

# ══════════════════════════════════════════════════════
# PATHS — All data persisted to Google Drive
# ══════════════════════════════════════════════════════
DRIVE_BASE = "/content/drive/MyDrive/LexAgent"
CHROMA_CORPUS_PATH = f"{DRIVE_BASE}/chroma_corpus"
CHROMA_MEMORY_PATH = f"{DRIVE_BASE}/chroma_memory"   # for Adaptive Memory (Phase 3)
BM25_INDEX_PATH    = f"{DRIVE_BASE}/bm25_index.pkl"
CORPUS_JSON_PATH   = f"{DRIVE_BASE}/corpus.json"
KNOWN_CASES_PATH   = f"{DRIVE_BASE}/known_cases.pkl"

# ══════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════
MODEL_NAME     = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDER_NAME  = "nlpaueb/legal-bert-base-uncased"

# ══════════════════════════════════════════════════════
# RETRIEVAL PARAMETERS
# ══════════════════════════════════════════════════════
TOP_K_DENSE    = 4
TOP_K_BM25     = 4
RRF_K          = 60               # Reciprocal Rank Fusion constant

# ══════════════════════════════════════════════════════
# CCS — Citation Confidence Score (NOVEL component)
# ══════════════════════════════════════════════════════
CCS_VERIFIED_THRESHOLD  = 0.75   # CCS >= 0.75 → VERIFIED
CCS_UNCERTAIN_THRESHOLD = 0.40   # 0.40 <= CCS < 0.75 → UNCERTAIN
# CCS < 0.40 → LIKELY_FAKE

# CCS weights (NOVEL: continuous citation confidence scoring)
CCS_EXACT_WEIGHT    = 0.50
CCS_FUZZY_WEIGHT    = 0.30
CCS_SEMANTIC_WEIGHT = 0.20

# ══════════════════════════════════════════════════════
# DDC — Dynamic Debate Controller (NOVEL component)
# ══════════════════════════════════════════════════════
DDC_MAX_ROUNDS          = 2
DDC_CCS_CONTINUE_THRESH = 0.60   # avg CCS < 0.60 triggers second round
DDC_MIN_GAPS_TO_CONTINUE = 1     # at least 1 reflection gap triggers second round

# ══════════════════════════════════════════════════════
# ADAPTIVE MEMORY (NOVEL component)
# ══════════════════════════════════════════════════════
MEMORY_CACHE_THRESHOLD      = 0.92  # cosine sim >= 0.92 → cache hit
MEMORY_HIGH_RISK_CCS_THRESH = 0.45  # citations consistently below → flagged as high-risk
