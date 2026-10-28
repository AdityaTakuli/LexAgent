# LexAgent v2.0 | Phase 1 | download_datasets.py
"""Dataset download pipeline for LexAgent v2.0.

Downloads three legal datasets from HuggingFace:
  1. LegalBench — contract_qa, statutory_reasoning, rule_qa (200 questions)
  2. CaseHOLD — full train split for CCS calibration and CAS evaluation
  3. SCOTUS — first 3000 opinions from FairLex for RAG corpus

All datasets are saved to Google Drive for persistence across sessions.
"""

import json
import os
import sys
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import DRIVE_BASE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _download_legalbench(save_dir: str) -> str:
    """Download LegalBench subtasks: contract_qa, statutory_reasoning, rule_qa.
    
    Samples 200 questions evenly across 3 subtasks (~67 each).
    
    Args:
        save_dir: Directory to save the dataset JSON.
    
    Returns:
        Path to saved JSON file.
    """
    from datasets import load_dataset
    
    subtasks = ["contract_qa", "rule_qa"]
    samples_per_task = 200 // len(subtasks)  # ~67 each
    all_samples: List[Dict[str, Any]] = []
    
    for subtask in subtasks:
        logger.info("Downloading LegalBench/%s...", subtask)
        try:
            ds = load_dataset("nguha/legalbench", subtask, split="test",trust_remote_code=True)
        except Exception as e:
            logger.warning("First attempt failed for %s: %s. Retrying...", subtask, e)
            try:
                ds = load_dataset("nguha/legalbench", subtask, split="test",trust_remote_code=True)
            except Exception as retry_e:
                logger.error("Failed to download LegalBench/%s: %s", subtask, retry_e)
                continue
        
        # Sample up to samples_per_task entries
        n = min(samples_per_task, len(ds))
        for i in range(n):
            sample = dict(ds[i])
            sample["subtask"] = subtask
            all_samples.append(sample)
        
        logger.info("  → Collected %d samples from %s", n, subtask)
    
    # Save to Drive
    filepath = os.path.join(save_dir, "legalbench_dataset.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    logger.info("LegalBench saved: %d samples → %s", len(all_samples), filepath)
    return filepath


def _download_casehold(save_dir: str) -> str:
    """Download CaseHOLD full train split for CCS calibration (via LexGLUE).
    
    Args:
        save_dir: Directory to save the dataset JSON.
    
    Returns:
        Path to saved JSON file.
    """
    from datasets import load_dataset
    
    logger.info("Downloading CaseHOLD (LexGLUE version, full train split)...")
    try:
        # Changed from "casehold/casehold" to "lex_glue", "case_hold"
        ds = load_dataset("coastalcph/lex_glue", "case_hold", split="train")
    except Exception as e:
        logger.warning("First attempt failed for CaseHOLD: %s. Retrying...", e)
        try:
            ds = load_dataset("coastalcph/lex_glue", "case_hold", split="train")
        except Exception as retry_e:
            logger.error("Failed to download CaseHOLD: %s", retry_e)
            raise
    
    # Convert to list of dicts
    samples = [dict(ds[i]) for i in range(len(ds))]
    
    filepath = os.path.join(save_dir, "casehold_dataset.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    logger.info("CaseHOLD saved: %d samples → %s", len(samples), filepath)
    return filepath


def _download_scotus(save_dir: str) -> str:
    """Download SCOTUS opinions from FairLex (first 3000).
    
    Args:
        save_dir: Directory to save the dataset JSON.
    
    Returns:
        Path to saved JSON file.
    """
    from datasets import load_dataset
    
    logger.info("Downloading SCOTUS (coastalcph/lex_glue, first 3000 opinions)...")
    try:
        ds = load_dataset("coastalcph/lex_glue", "scotus", split="train")
    except Exception as e:
        logger.warning("First attempt failed for SCOTUS: %s. Retrying...", e)
        try:
            ds = load_dataset("coastalcph/lex_glue", "scotus", split="train")
        except Exception as retry_e:
            logger.error("Failed to download SCOTUS: %s", retry_e)
            raise
    
    # Take first 3000 opinions
    n = min(3000, len(ds))
    samples = []
    for i in range(n):
        entry = dict(ds[i])
        # Extract text field (FairLex SCOTUS uses "text" column)
        samples.append({
            "text": entry.get("text", ""),
            "label": entry.get("label", -1),
            "idx": i,
        })
    
    filepath = os.path.join(save_dir, "scotus_dataset.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    logger.info("SCOTUS saved: %d opinions → %s", n, filepath)
    return filepath


def download_all_datasets(save_dir: str = DRIVE_BASE) -> Dict[str, str]:
    """Download all three legal datasets for LexAgent v2.0.
    
    Downloads are idempotent — existing files on Drive are skipped.
    
    Args:
        save_dir: Base directory for saving datasets (default: DRIVE_BASE).
    
    Returns:
        Dictionary mapping dataset name to saved file path:
        {"legalbench": path, "casehold": path, "scotus": path}
    """
    os.makedirs(save_dir, exist_ok=True)
    result: Dict[str, str] = {}
    
    # ── LegalBench ───────────────────────────────────────────────────
    lb_path = os.path.join(save_dir, "legalbench_dataset.json")
    if os.path.exists(lb_path):
        logger.info("LegalBench already exists, skipping download.")
        result["legalbench"] = lb_path
    else:
        result["legalbench"] = _download_legalbench(save_dir)
    
    # ── CaseHOLD ─────────────────────────────────────────────────────
    ch_path = os.path.join(save_dir, "casehold_dataset.json")
    if os.path.exists(ch_path):
        logger.info("CaseHOLD already exists, skipping download.")
        result["casehold"] = ch_path
    else:
        result["casehold"] = _download_casehold(save_dir)
    
    # ── SCOTUS ───────────────────────────────────────────────────────
    sc_path = os.path.join(save_dir, "scotus_dataset.json")
    if os.path.exists(sc_path):
        logger.info("SCOTUS already exists, skipping download.")
        result["scotus"] = sc_path
    else:
        result["scotus"] = _download_scotus(save_dir)
    
    logger.info("All datasets ready: %s", list(result.keys()))
    return result
