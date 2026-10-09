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
CAP_DOWNLOAD_CHECKPOINT_PATH = os.path.join(DRIVE_BASE, "cap_download_checkpoint.json").replace("\\", "/")  # Stage 1 ledger (per-volume state + 64-volume batches)
CAP_CHUNK_CHECKPOINT_PATH    = os.path.join(DRIVE_BASE, "cap_chunk_checkpoint.json").replace("\\", "/")     # Stage 2 ledger (per-volume state + 64-volume batches)
CAP_BUILD_STATE_PATH         = os.path.join(DRIVE_BASE, "cap_build_state.json").replace("\\", "/")          # Which corpus the BM25 / known-cases / corpus JSON were built from
CASE_CATALOG_PATH            = os.path.join(DRIVE_BASE, "cap_case_catalog.json").replace("\\", "/")         # Every report in the corpus (name, cite, year, length), used to find the actual opinion for a citation

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

# CAP ends at U.S. Reports volume 572. Later volumes can be imported from the Supreme Court's
# official bound-volume PDFs (--download-us-reports-pdf).
US_REPORTS_PDF_URL = os.environ.get("LEXAGENT_US_REPORTS_PDF_URL", "https://www.supremecourt.gov/opinions/boundvolumes/{volume}BV.pdf")
US_REPORTS_INDEX_URL = os.environ.get("LEXAGENT_US_REPORTS_INDEX_URL", "https://www.supremecourt.gov/opinions/USReports.aspx")
US_REPORTS_PDF_DIR = os.path.join(DRIVE_BASE, "us_reports_bound_volumes").replace("\\", "/")

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

# Staged pipeline: a batch checkpoint is committed after every N volumes of the download and chunk stages
CAP_CHECKPOINT_INTERVAL = int(os.environ.get("LEXAGENT_CHECKPOINT_INTERVAL", "64"))

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
# How the citation checker turns its findings into the verdict the rest of the system acts on:
#   "gated"  (default) existence and holding support are reported separately. The support verdict
#            may steer the judge, the controller and the memory only if a support model has been
#            validated on held-out labelled citations (the gate file below). Until then a citation
#            that exists is "VERIFIED" (existence only) and its support score is information.
#   "legacy" the behaviour of the runs made before the forensic audit, kept to reproduce them.
CCE_POLICY = os.environ.get("LEXAGENT_CCE_POLICY", "gated")
# Written by experiments/v2/e2_verifier_validation.py; absent or "validated": false = gate closed.
CCE_GATE_PATH = os.environ.get("LEXAGENT_CCE_GATE_FILE", os.path.join(DRIVE_BASE, "cce_support_gate.json").replace("\\", "/"))
CCE_GATE_MIN_AUC = 0.90  # a support model must reach this on the held-out split to open the gate

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
DDC_MAX_ROUNDS                = int(os.environ.get("LEXAGENT_MAX_ROUNDS", "3"))
DDC_CCS_CONTINUE_THRESH       = 0.60
DDC_MIN_GAPS_TO_CONTINUE      = 1
DDC_CHALLENGE_STRENGTH_THRESH = 0.5

# ══════════════════════════════════════════════════════
# DEBATE PROTOCOL (repair 6 of the forensic audit)
# ══════════════════════════════════════════════════════
# How a debate keeps what it verified, what the judge reads, and when the controller stops:
#   "v1"      as in every run made up to 7 October 2026: each round replaces the citations and the
#             citation report of the previous one, the judge reads the last report only, and the
#             controller continues on weak scores or on the defense's self-reported strength.
#   "ledger"  a per-debate ledger keeps the best verdict and its evidence for every citation of
#             every round; the judge reads the ledger and is instructed by tier, not by a score
#             threshold; the controller continues only while no supported citation exists, a gap
#             remains and the last round added a new passage or a new supported citation.
# A state that carries its own "protocol" key (every runner in experiments/v2 sets one) wins.
DEBATE_PROTOCOL = os.environ.get("LEXAGENT_DEBATE_PROTOCOL", "v1")
DEBATE_PROTOCOLS = ("v1", "ledger")
LEDGER_EVIDENCE_CHARS = 300  # characters of the evidence window the judge is shown per citation

# ══════════════════════════════════════════════════════
# RETRIEVAL V2 (repair 7 of the forensic audit)
# ══════════════════════════════════════════════════════
# A second, separate retrieval index. The Legal-BERT collection is kept: the citation checker was
# validated on it and continues to use it. Built by runs/build_retrieval_v2.py.
RETRIEVAL_V2_DIR            = os.path.join(DRIVE_BASE, "retrieval_v2").replace("\\", "/")
RETRIEVAL_V2_MODEL          = "BAAI/bge-base-en-v1.5"
RETRIEVAL_V2_QUERY_PREFIX   = "Represent this sentence for searching relevant passages: "
RETRIEVAL_V2_MAX_TOKENS     = 512
RETRIEVAL_V2_PER_CASE_LIMIT = 2   # at most this many passages of one case in the fused list

# ══════════════════════════════════════════════════════
# ADAPTIVE MEMORY (NOVEL component)
# ══════════════════════════════════════════════════════
MEMORY_CACHE_THRESHOLD      = 0.92
# How the verdict cache decides that a question was answered before:
#   "exact"      the same question after normalising case, whitespace and punctuation (default)
#   "similarity" embedding cosine above MEMORY_CACHE_THRESHOLD. Kept only to reproduce the runs made
#                before the forensic audit: it returns answers written for other questions.
#   "off"        never. Evaluation runners use this: a benchmark of distinct questions has nothing to cache.
MEMORY_CACHE_MODE           = os.environ.get("LEXAGENT_MEMORY_CACHE", "exact")
MEMORY_HIGH_RISK_CCS_THRESH = 0.45

# ══════════════════════════════════════════════════════
# EVALUATION THRESHOLDS
# ══════════════════════════════════════════════════════
CAS_SEMANTIC_THRESHOLD  = 0.70
FUZZY_MATCH_THRESHOLD   = 85

# ══════════════════════════════════════════════════════
# OFFLINE SERVER & HIGH-END GPU SETTINGS
# ══════════════════════════════════════════════════════
OFFLINE_MODE             = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1" or os.environ.get("HF_HUB_OFFLINE", "0") == "1"
PRECISION_MODE           = os.environ.get("LEXAGENT_PRECISION", "auto")           # "auto", "bfloat16", "float16", "4bit"
DEFAULT_GPU_BATCH_SIZE   = int(os.environ.get("LEXAGENT_BATCH_SIZE", "256"))       # Dense embedding batch size on GPU
CCE_OPINION_SCAN_LIMIT   = int(os.environ.get("LEXAGENT_CCE_SCAN_LIMIT", "8000"))  # Char window scan for CCE NLI
DEFAULT_PARALLEL_WORKERS = int(os.environ.get("LEXAGENT_WORKERS", "16"))         # Thread workers for volume ingestion
