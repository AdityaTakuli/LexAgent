# LexAgent v3.0 | config.py
"""
Central configuration for LexAgent v3.0 — Caselaw Access Project (CAP) Edition.
In V3, Harvard's Caselaw Access Project (CAP) is the primary authority database.
Supports scalable multi-volume ingestion: Landmark Preset (~27 volumes),
Modern Era (Volumes 300 to 585), Full CAP (Volumes 1 to 600+), or custom ranges.
"""

import os
import sys

# ══════════════════════════════════════════════════════
# REPRODUCIBILITY & SEED CONTROL
# ══════════════════════════════════════════════════════
RANDOM_SEED = 42

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Improves run-to-run reproducibility and enables controlled repeated experiments."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_default_data_dir() -> str:
    """Determine persistent storage path.
    
    1. Highest precedence: LEXAGENT_DATA_DIR environment variable.
    2. If running in Google Colab (/content exists): /content/drive/MyDrive/LexAgentV3
    3. If running locally: <V3_dir>/data
    """
    if "LEXAGENT_DATA_DIR" in os.environ:
        return os.environ["LEXAGENT_DATA_DIR"]
    if os.path.exists("/content") or "google.colab" in sys.modules:
        return "/content/drive/MyDrive/LexAgentV3"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


DRIVE_BASE = get_default_data_dir()

# Environment detection
IS_COLAB = os.path.exists("/content") or "google.colab" in sys.modules

# Primary CAP Authority Database Paths
DRIVE_CHROMA_CORPUS_PATH = os.path.join(DRIVE_BASE, "cap_chroma_corpus").replace("\\", "/")
LOCAL_CHROMA_CORPUS_PATH = "/content/cap_chroma_local" if IS_COLAB else DRIVE_CHROMA_CORPUS_PATH
# In Google Colab, writing directly over Google Drive FUSE causes SQLite corruption (code 11)
# and severe I/O lag. CHROMA_CORPUS_PATH points to fast local NVMe SSD in Colab.
CHROMA_CORPUS_PATH   = LOCAL_CHROMA_CORPUS_PATH

BM25_INDEX_PATH      = os.path.join(DRIVE_BASE, "cap_bm25_index.pkl").replace("\\", "/")
KNOWN_CASES_PATH     = os.path.join(DRIVE_BASE, "cap_known_cases.pkl").replace("\\", "/")
CORPUS_JSON_PATH     = os.path.join(DRIVE_BASE, "cap_corpus.json").replace("\\", "/")
CORPUS_MANIFEST_PATH = os.path.join(DRIVE_BASE, "cap_corpus_manifest.json").replace("\\", "/")
CAP_RAW_DIR          = os.path.join(DRIVE_BASE, "cap_raw_volumes").replace("\\", "/")  # Checkpointed raw volume JSONs
CAP_CHUNKS_DIR       = os.path.join(DRIVE_BASE, "cap_chunks_cache").replace("\\", "/")  # Checkpointed per-volume chunks
INDEXING_CHECKPOINT_PATH = os.path.join(DRIVE_BASE, "cap_indexing_checkpoint.json").replace("\\", "/")  # Resumable indexing progress

# Adaptive Memory ChromaDB Path
DRIVE_CHROMA_MEMORY_PATH = os.path.join(DRIVE_BASE, "chroma_memory").replace("\\", "/")
LOCAL_CHROMA_MEMORY_PATH = "/content/cap_chroma_memory_local" if IS_COLAB else DRIVE_CHROMA_MEMORY_PATH
CHROMA_MEMORY_PATH   = LOCAL_CHROMA_MEMORY_PATH

# Benchmark / Evaluation Datasets
CASEHOLD_DATASET_PATH   = os.path.join(DRIVE_BASE, "casehold_dataset.json").replace("\\", "/")
LEGALBENCH_DATASET_PATH = os.path.join(DRIVE_BASE, "legalbench_dataset.json").replace("\\", "/")
RESULTS_DIR             = os.path.join(DRIVE_BASE, "results").replace("\\", "/")

# ══════════════════════════════════════════════════════
# CAP (CASELAW ACCESS PROJECT) CONFIGURATION
# ══════════════════════════════════════════════════════
CAP_BASE_URL = os.environ.get("LEXAGENT_CAP_BASE_URL", "https://static.case.law")

# Landmark 27 Volumes that directly cover all 30 constitutional & statutory benchmark cases
LANDMARK_CAP_VOLUMES = [
    "347",  # Brown v. Board of Education (1954)
    "367",  # Mapp v. Ohio (1961)
    "372",  # Gideon v. Wainwright (1963)
    "373",  # Brady v. Maryland (1963)
    "376",  # New York Times v. Sullivan (1964)
    "381",  # Griswold v. Connecticut (1965)
    "384",  # Miranda v. Arizona (1966)
    "389",  # Katz v. United States (1967)
    "392",  # Terry v. Ohio (1968)
    "410",  # Roe v. Wade, Doe v. Bolton (1973)
    "411",  # San Antonio v. Rodriguez, US v. Russell (1973)
    "413",  # Miller v. California (1973)
    "466",  # Strickland v. Washington, US v. Jacobsen (1984)
    "467",  # Chevron U.S.A. v. NRDC (1984)
    "476",  # Batson v. Kentucky (1986)
    "509",  # Daubert v. Merrell Dow (1993)
    "514",  # United States v. Lopez (1995)
    "533",  # Kyllo v. United States (2001)
    "543",  # Roper v. Simmons (2005)
    "554",  # District of Columbia v. Heller (2008)
    "558",  # Citizens United v. FEC (2010)
    "560",  # Graham v. Florida (2010)
    "567",  # Miller v. Alabama (2012)
    "573",  # Riley v. California (2014)
    "576",  # Obergefell v. Hodges (2015)
    "585",  # Carpenter v. United States (2018)
    "587",  # Gamble v. United States (2019)
]

# Mode selection: "landmark" (default), "modern" (300-585), "all" (1-600), or custom list/range
CAP_VOLUMES_MODE = os.environ.get("LEXAGENT_CAP_VOLUMES", "landmark")

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
# CCS — Citation Confidence Score (NOVEL component: Factorized CCE)
# ══════════════════════════════════════════════════════
CCS_VERIFIED_THRESHOLD  = 0.75   # CCS >= 0.75 → SUPPORTED
CCS_UNCERTAIN_THRESHOLD = 0.40   # 0.40 <= CCS < 0.75 → UNCERTAIN
TAU_ACCEPTANCE_THRESHOLD = 0.70  # Conformal / Factorized acceptance threshold

# NLI Cross-Encoder for Entailment Support (Novelty 1)
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Legacy CCS weights (for backward compatibility / ablation comparisons)
CCS_EXACT_WEIGHT    = 0.50
CCS_FUZZY_WEIGHT    = 0.30
CCS_SEMANTIC_WEIGHT = 0.20

# CCE internal thresholds
CCE_HOLDING_SEM_THRESHOLD = 0.75
CCE_HOLDING_FALLBACK_SCORE = 0.7
CCE_FUZZY_THRESHOLD_RATIO  = 0.85

# ══════════════════════════════════════════════════════
# DDC — Dynamic Debate Controller (NOVEL component)
# ══════════════════════════════════════════════════════
DDC_MAX_ROUNDS                = 2
DDC_CCS_CONTINUE_THRESH       = 0.60
DDC_MIN_GAPS_TO_CONTINUE      = 1
DDC_CHALLENGE_STRENGTH_THRESH = 0.5

# ══════════════════════════════════════════════════════
# ADAPTIVE MEMORY (NOVEL component)
# ══════════════════════════════════════════════════════
MEMORY_CACHE_THRESHOLD      = 0.92
MEMORY_HIGH_RISK_CCS_THRESH = 0.45

# ══════════════════════════════════════════════════════
# EVALUATION THRESHOLDS
# ══════════════════════════════════════════════════════
CAS_SEMANTIC_THRESHOLD  = 0.70
FUZZY_MATCH_THRESHOLD   = 85
