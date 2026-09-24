# LexAgent v3.0 | chroma_sync.py
"""ChromaDB Storage, Integrity & Drive Synchronization Utility.

Resolves SQLite corruption ("database disk image is malformed" code 11) caused by
running active database transactions over Google Drive network FUSE mounts.
Directs active ChromaDB operations to local NVMe SSD (/content/cap_chroma_local in Colab)
with automatic health-checks, malformed-database self-healing, and safe Drive backups.
"""

import os
import shutil
import sqlite3
import sys
import time
from typing import Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


def is_colab_environment() -> bool:
    """Detect if running inside Google Colab."""
    return os.path.exists("/content") or "google.colab" in sys.modules


def check_sqlite_health(db_dir: str) -> bool:
    """Check if chroma.sqlite3 in db_dir is valid, non-corrupt, and readable."""
    sqlite_file = os.path.join(db_dir, "chroma.sqlite3")
    if not os.path.exists(sqlite_file):
        return True  # Empty directory / fresh start
    if os.path.getsize(sqlite_file) == 0:
        return False  # Truncated 0-byte file from aborted process

    conn = None
    try:
        conn = sqlite3.connect(sqlite_file, timeout=5.0)
        cursor = conn.cursor()
        cursor.execute("PRAGMA quick_check;")
        res = cursor.fetchone()
        return bool(res and res[0] == "ok")
    except Exception as e:
        logger.warning("SQLite integrity check failed for %s: %s", db_dir, e)
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def clean_corrupted_chroma(db_dir: str, checkpoint_path: Optional[str] = None):
    """Safely wipe a corrupted ChromaDB directory and reset indexing checkpoint."""
    if os.path.exists(db_dir):
        logger.warning("Purging corrupted ChromaDB directory: %s", db_dir)
        try:
            shutil.rmtree(db_dir, ignore_errors=True)
        except Exception as e:
            logger.error("Failed to remove directory %s: %s", db_dir, e)

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info("Reset indexing checkpoint: %s", checkpoint_path)
        except Exception:
            pass


def sync_chroma_to_drive(local_path: str, drive_path: str):
    """Safely sync ChromaDB from local SSD to Google Drive."""
    if not is_colab_environment() or local_path == drive_path:
        return
    if not os.path.exists(local_path):
        return

    # Do not sync if local database is corrupt
    if not check_sqlite_health(local_path):
        logger.warning("Skipping sync: Local ChromaDB appears corrupt.")
        return

    t0 = time.time()
    try:
        os.makedirs(drive_path, exist_ok=True)
        for item in os.listdir(local_path):
            s = os.path.join(local_path, item)
            d = os.path.join(drive_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        elapsed = time.time() - t0
        logger.info("Synced ChromaDB to Drive (%s) in %.2fs", drive_path, elapsed)
    except Exception as e:
        logger.warning("Could not sync ChromaDB to Drive: %s", e)


def prepare_working_chroma(
    local_path: str,
    drive_path: str,
    checkpoint_path: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Prepare healthy ChromaDB directory.
    - If overwrite: wipes both local and drive directories and checkpoint.
    - If drive or local ChromaDB is malformed: auto-heals by wiping corrupted files.
    - In Colab: restores healthy drive copy to fast local SSD (/content/cap_chroma_local).
    - Returns the active path to use for ChromaDB client.
    """
    if overwrite:
        print("  -> Overwrite requested: Purging existing ChromaDB collections...")
        clean_corrupted_chroma(local_path, checkpoint_path)
        clean_corrupted_chroma(drive_path, checkpoint_path)
        os.makedirs(local_path, exist_ok=True)
        return local_path

    # 1. Check persistent Google Drive directory health
    if os.path.exists(drive_path) and os.listdir(drive_path):
        if not check_sqlite_health(drive_path):
            print(f"\n  [!] AUTO-HEAL: Malformed ChromaDB detected on Google Drive: {drive_path}")
            print("      (Cause: Incomplete transaction / FUSE network interruption in previous run)")
            print("      Auto-wiping corrupted database to prevent 'database disk image is malformed' crash...")
            clean_corrupted_chroma(drive_path, checkpoint_path)
            clean_corrupted_chroma(local_path, checkpoint_path)

    # 2. Check local SSD directory health
    if os.path.exists(local_path) and os.listdir(local_path):
        if not check_sqlite_health(local_path):
            print(f"\n  [!] AUTO-HEAL: Malformed ChromaDB detected on local SSD: {local_path}")
            clean_corrupted_chroma(local_path, checkpoint_path)

    # 3. If running in Colab and local SSD is empty, restore from Drive if healthy
    if is_colab_environment() and local_path != drive_path:
        local_has_files = os.path.exists(local_path) and bool(os.listdir(local_path))
        drive_has_files = os.path.exists(drive_path) and bool(os.listdir(drive_path))

        if not local_has_files and drive_has_files and check_sqlite_health(drive_path):
            print(f"  -> Restoring healthy ChromaDB from Google Drive to fast local SSD ({local_path})...")
            try:
                shutil.copytree(drive_path, local_path, dirs_exist_ok=True)
                print("  [OK] Local fast ChromaDB restored from Drive backup.")
            except Exception as e:
                logger.warning("Could not copy Drive ChromaDB to local: %s", e)

        os.makedirs(local_path, exist_ok=True)
        return local_path

    os.makedirs(local_path, exist_ok=True)
    os.makedirs(drive_path, exist_ok=True)
    target = local_path if is_colab_environment() else drive_path
    return target


def ensure_chroma_ready(local_path: str, drive_path: str) -> str:
    """Ensure ChromaDB is loaded on local fast SSD in Colab for fast inference/retrieval."""
    if is_colab_environment() and local_path != drive_path:
        local_has_files = os.path.exists(local_path) and bool(os.listdir(local_path))
        drive_has_files = os.path.exists(drive_path) and bool(os.listdir(drive_path))

        if not local_has_files and drive_has_files:
            if check_sqlite_health(drive_path):
                print(f"  -> Restoring ChromaDB to local NVMe SSD for fast retrieval ({local_path})...")
                try:
                    shutil.copytree(drive_path, local_path, dirs_exist_ok=True)
                except Exception as e:
                    logger.warning("Failed to copy to local SSD: %s", e)

        if os.path.exists(local_path) and os.listdir(local_path):
            return local_path

    return drive_path if os.path.exists(drive_path) else local_path
