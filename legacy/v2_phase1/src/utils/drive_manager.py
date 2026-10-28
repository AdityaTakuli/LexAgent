# LexAgent v2.0 | Phase 1 | drive_manager.py
"""Google Drive I/O manager for persistent storage on Colab.

All data is saved to DRIVE_BASE on Google Drive so that:
- ChromaDB indices persist across Colab sessions
- Corpus and BM25 indices survive runtime restarts
- Model artifacts and evaluation results are preserved
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    DRIVE_BASE,
    CHROMA_CORPUS_PATH,
    CHROMA_MEMORY_PATH,
    BM25_INDEX_PATH,
    CORPUS_JSON_PATH,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriveManager:
    """Manages all file I/O operations to Google Drive.
    
    Provides idempotent save/load methods for JSON and pickle files,
    along with Google Drive mounting for Colab environments.
    """
    
    def __init__(self, base_path: str = DRIVE_BASE):
        """Initialize DriveManager.
        
        Args:
            base_path: Root directory on Google Drive for all LexAgent data.
        """
        self.base_path = base_path
    
    def mount(self) -> None:
        """Mount Google Drive and create all required subdirectories.
        
        Creates the following directory structure on Drive:
          DRIVE_BASE/
          ├── chroma_corpus/   (ChromaDB persistent index)
          ├── chroma_memory/   (Adaptive Memory index - Phase 3)
          └── (flat files: corpus.json, bm25_index.pkl, known_cases.pkl)
        """
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("Google Drive mounted successfully.")
        except ImportError:
            logger.warning(
                "Not running in Google Colab. "
                "Using local filesystem at: %s", self.base_path
            )
        except Exception as e:
            logger.error("Failed to mount Google Drive: %s", e)
            raise
        
        # Create all required directories (idempotent)
        dirs_to_create = [
            self.base_path,
            CHROMA_CORPUS_PATH,
            CHROMA_MEMORY_PATH,
        ]
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            logger.info("Directory ensured: %s", dir_path)
    
    def save_json(self, data: Any, filename: str) -> str:
        """Save data as JSON to DRIVE_BASE.
        
        Args:
            data: JSON-serializable data.
            filename: Target filename (e.g., 'corpus.json').
        
        Returns:
            Full path of saved file.
        """
        filepath = os.path.join(self.base_path, filename)
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info("Saved JSON: %s (%.2f MB)", filepath,
                     os.path.getsize(filepath) / 1e6)
        return filepath
    
    def load_json(self, filename: str) -> Any:
        """Load JSON data from DRIVE_BASE.
        
        Args:
            filename: Source filename (e.g., 'corpus.json').
        
        Returns:
            Deserialized Python object.
        
        Raises:
            FileNotFoundError: If file does not exist.
        """
        filepath = os.path.join(self.base_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info("Loaded JSON: %s", filepath)
        return data
    
    def save_pickle(self, obj: Any, filename: str) -> str:
        """Save object as pickle to DRIVE_BASE.
        
        Args:
            obj: Any picklable Python object.
            filename: Target filename (e.g., 'bm25_index.pkl').
        
        Returns:
            Full path of saved file.
        """
        filepath = os.path.join(self.base_path, filename)
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        
        logger.info("Saved pickle: %s (%.2f MB)", filepath,
                     os.path.getsize(filepath) / 1e6)
        return filepath
    
    def load_pickle(self, filename: str) -> Any:
        """Load pickle data from DRIVE_BASE.
        
        Args:
            filename: Source filename (e.g., 'bm25_index.pkl').
        
        Returns:
            Deserialized Python object.
        
        Raises:
            FileNotFoundError: If file does not exist.
        """
        filepath = os.path.join(self.base_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pickle file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        
        logger.info("Loaded pickle: %s", filepath)
        return obj
    
    def exists(self, filename: str) -> bool:
        """Check if a file exists on Drive.
        
        Args:
            filename: Filename to check (relative to DRIVE_BASE).
        
        Returns:
            True if file exists, False otherwise.
        """
        filepath = os.path.join(self.base_path, filename)
        return os.path.exists(filepath)
