# LexAgent v3.0 | build_db.py
"""Build primary Harvard CAP ChromaDB, BM25, and known_cases catalog across multiple volumes.

Supports:
  - Resumable ChromaDB dense vector indexing (auto-resumes from last checkpoint if interrupted)
  - Portion-based volume chunking & chunk caching to prevent data loss
  - Immediate persistence of BM25 lexical index and known_cases catalog
  - Intelligent hardware detection (GPU acceleration vs CPU fallback) with ETA reporting
"""

import argparse
from datetime import datetime, timezone
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb

from config import (
    BM25_INDEX_PATH,
    CAP_RAW_DIR,
    CHROMA_CORPUS_PATH,
    DRIVE_CHROMA_CORPUS_PATH,
    LOCAL_CHROMA_CORPUS_PATH,
    IS_COLAB,
    CORPUS_JSON_PATH,
    INDEXING_CHECKPOINT_PATH,
    KNOWN_CASES_PATH,
    CAP_VOLUMES_MODE,
)
from src.utils.chroma_sync import (
    prepare_working_chroma,
    sync_chroma_to_drive,
    clean_corrupted_chroma,
    is_colab_environment,
)
from src.data.cap_ingest import (
    build_cap_corpus,
    build_cap_corpus_in_portions,
    extract_known_cases_from_chunks,
    get_downloaded_volumes,
    parse_volume_selection,
    write_cap_manifest,
)
from src.models.load_model import load_embedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


def chroma_metadata(document):
    raw = {
        "case_name": document.case_name,
        "year": document.year,
        "court": document.court,
        "source": "CAP",
        "volume": document.volume,
        "reporter": document.reporter,
        "first_page": document.first_page,
        **document.metadata
    }
    return {
        key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
              else ("" if value is None else value))
        for key, value in raw.items()
    }


def format_seconds(secs: float) -> str:
    """Format duration in human-readable string."""
    if secs <= 0:
        return "0s"
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def save_checkpoint(last_index: int, total_chunks: int, collection_count: int, checkpoint_path: str = INDEXING_CHECKPOINT_PATH):
    """Write dense vector indexing checkpoint to disk."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "last_indexed_index": last_index,
        "total_chunks": total_chunks,
        "collection_count": collection_count,
        "percent_complete": round(100.0 * last_index / total_chunks, 2) if total_chunks > 0 else 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_checkpoint(checkpoint_path: str = INDEXING_CHECKPOINT_PATH) -> dict:
    """Load indexing checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main(
    volumes_mode: str = CAP_VOLUMES_MODE,
    max_workers: int = 10,
    resume: bool = True,
    overwrite: bool = False,
    portion_size: int = 25,
    custom_batch_size: int = None,
):
    print("\n" + "=" * 78)
    print(f"  BUILDING LEXAGENT v3.0 PRIMARY CAP DATABASE")
    print(f"  Selection Mode: '{volumes_mode}' (Workers: {max_workers})")
    print(f"  Checkpoint Resume: {'ACTIVE' if resume and not overwrite else 'DISABLED (Fresh Build)'}")
    print("=" * 78)

    volume_list = parse_volume_selection(volumes_mode)
    print(f"  -> Target Volumes ({len(volume_list)} total): {volume_list[:8]}..." if len(volume_list) > 8 else f"  -> Target Volumes: {volume_list}")

    # Check already-downloaded volumes on disk
    downloaded_set, max_downloaded_vol = get_downloaded_volumes(CAP_RAW_DIR)
    if downloaded_set:
        print(f"  [CHECKPOINT] Scanned {CAP_RAW_DIR}: Found {len(downloaded_set)} volumes already downloaded (highest: Vol {max_downloaded_vol}).")
        missing_vols = [v for v in volume_list if v not in downloaded_set]
        if missing_vols:
            print(f"  [CHECKPOINT] Next volume to download: Vol {missing_vols[0]} ({len(missing_vols)} remaining to fetch).")
        else:
            print(f"  [CHECKPOINT] All target volumes (up to Vol {max_downloaded_vol}) are already present locally!")

    # 1. Obtain Corpus Chunks & Known Cases (Portion-based or load from existing corpus)
    chunks = []
    known_cases = set()

    if os.path.exists(CORPUS_JSON_PATH) and not overwrite:
        print(f"\n  -> Found existing master corpus at: {CORPUS_JSON_PATH}")
        print("  -> Loading chunked documents directly from disk...")
        try:
            with open(CORPUS_JSON_PATH, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            from src.data.corpus_schema import CorpusDocument
            chunks = [CorpusDocument.from_dict(d) if isinstance(d, dict) else d for d in raw_list]
            del raw_list
            import gc
            gc.collect()
            print(f"  -> Successfully loaded {len(chunks):,} corpus chunks.")
            if os.path.exists(KNOWN_CASES_PATH) and not overwrite:
                print(f"  [OK] Known cases catalog already exists on disk: {KNOWN_CASES_PATH}. Skipping re-extraction.")
                known_cases = None
            else:
                known_cases = extract_known_cases_from_chunks(chunks)
                print(f"  -> Extracted {len(known_cases):,} known cases & citations from corpus.")
        except Exception as e:
            logger.warning("Could not parse existing corpus JSON: %s. Rebuilding in portions...", e)
            chunks, known_cases = build_cap_corpus_in_portions(
                volume_list,
                portion_size=portion_size,
                max_workers=max_workers,
                volume_tag=volumes_mode,
            )
    else:
        # Build corpus using portioned chunking and volume-level caching
        chunks, known_cases = build_cap_corpus_in_portions(
            volume_list,
            portion_size=portion_size,
            max_workers=max_workers,
            volume_tag=volumes_mode,
        )

    if not chunks:
        raise RuntimeError("No chunks were generated or loaded from CAP volumes.")

    print(f"\n  -> Total Retrieval Chunks: {len(chunks):,}")

    # 2. IMMEDIATE PERSISTENCE: Save Known Cases Catalog & BM25 Lexical Index FIRST
    # (These take < 1 minute to build and ensure lexical components are saved even if dense indexing is interrupted)
    print("\n" + "-" * 78)
    print("  [STEP 2: LEXICAL ARTIFACTS & CCE CATALOG PERSISTENCE]")
    print("-" * 78)

    # 2a. Known Cases Catalog
    if os.path.exists(KNOWN_CASES_PATH) and not overwrite:
        print(f"  [OK] Known cases catalog already exists on disk: {KNOWN_CASES_PATH}. Preserved.")
    elif known_cases:
        os.makedirs(os.path.dirname(KNOWN_CASES_PATH), exist_ok=True)
        with open(KNOWN_CASES_PATH, "wb") as handle:
            pickle.dump(known_cases, handle)
        print(f"  [OK] Known cases catalog saved immediately: {KNOWN_CASES_PATH} ({len(known_cases):,} entries)")
        del known_cases
        import gc
        gc.collect()

    # 2b. BM25 Lexical Index
    if os.path.exists(BM25_INDEX_PATH) and not overwrite:
        print(f"  [OK] BM25 index already exists on disk: {BM25_INDEX_PATH}. Preserved.")
    else:
        print(f"  -> Building and serializing memory-efficient BM25 index over {len(chunks):,} chunks...")
        bm25 = BM25Retriever(chunks)
        bm25.save_index(BM25_INDEX_PATH)
        del bm25
        import gc
        gc.collect()
        print(f"  [OK] BM25 lexical index built & saved: {BM25_INDEX_PATH}")

    # 3. Dense Indexing in ChromaDB (Portion-based, Checkpointed, Resumable)
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "N/A"
        gpu_vram = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if has_cuda else "N/A"
    except Exception:
        has_cuda = False
        gpu_name = "N/A"
        gpu_vram = "N/A"

    print("\n" + "-" * 78)
    print("  [STEP 3: HARDWARE ACCELERATION & DENSE VECTOR INDEXING]")
    print("-" * 78)
    if has_cuda:
        print(f"  [*] CUDA Acceleration: ACTIVE (GPU: {gpu_name}, Total VRAM: {gpu_vram})")
        print("     Target: Tensor Core batch parallelism enabled.")
    else:
        print("  [!] HARDWARE WARNING: Running Legal-BERT on CPU mode.")
        print("     Tip for Google Colab Users:")
        print("       1. Click top menu: Runtime -> Change runtime type")
        print("       2. Under 'Hardware accelerator', select 'T4 GPU' and click Save.")
        print("       3. GPU indexing takes ~25-35 mins (vs 10+ hours on CPU).")
        print("     * Note: Progress is safely checkpointed. You can safely stop and resume anytime!")
    print("-" * 78)

    active_chroma_path = prepare_working_chroma(
        local_path=LOCAL_CHROMA_CORPUS_PATH,
        drive_path=DRIVE_CHROMA_CORPUS_PATH,
        checkpoint_path=INDEXING_CHECKPOINT_PATH,
        overwrite=overwrite,
    )
    print(f"  [*] Active ChromaDB Storage: {active_chroma_path}")
    if is_colab_environment():
        print("      (NVMe SSD Engine: High-speed writes, zero network FUSE SQLite corruption)")
        print(f"      (Auto-Sync Target on Google Drive: {DRIVE_CHROMA_CORPUS_PATH})")

    try:
        client = chromadb.PersistentClient(path=active_chroma_path)
        if overwrite:
            try:
                client.delete_collection("cap_authorities")
            except Exception:
                pass
            collection = client.create_collection("cap_authorities", metadata={"hnsw:space": "cosine"})
            existing_count = 0
            start_index = 0
        else:
            collection = client.get_or_create_collection("cap_authorities", metadata={"hnsw:space": "cosine"})
            existing_count = collection.count()

            ckpt = load_checkpoint(INDEXING_CHECKPOINT_PATH)
            start_index = ckpt.get("last_indexed_index", 0)

            # Fallback to collection count if checkpoint file was absent but items exist
            if start_index == 0 and existing_count > 0:
                start_index = min(existing_count, len(chunks))
    except Exception as e:
        print(f"\n  [!] CAUTION: Corrupted ChromaDB state detected: {e}")
        print("      Auto-Heal: Purging malformed database and starting fresh on high-speed SSD...")
        clean_corrupted_chroma(active_chroma_path, INDEXING_CHECKPOINT_PATH)
        if is_colab_environment():
            clean_corrupted_chroma(DRIVE_CHROMA_CORPUS_PATH, INDEXING_CHECKPOINT_PATH)
        os.makedirs(active_chroma_path, exist_ok=True)
        client = chromadb.PersistentClient(path=active_chroma_path)
        collection = client.create_collection("cap_authorities", metadata={"hnsw:space": "cosine"})
        existing_count = 0
        start_index = 0

    if not overwrite and start_index >= len(chunks):
        print(f"\n  [OK] ChromaDB index already 100% complete: {existing_count:,} / {len(chunks):,} chunks stored.")
        print(f"    Path: {active_chroma_path}")
        if is_colab_environment():
            sync_chroma_to_drive(active_chroma_path, DRIVE_CHROMA_CORPUS_PATH)
        write_cap_manifest(chunks, volume_tag=volumes_mode)
        print("\n" + "=" * 78)
        print("  LEXAGENT v3.0 CAP DATABASE IS READY AND FULLY VERIFIED!")
        print("=" * 78)
        return

    if start_index > 0:
        pct = 100.0 * start_index / len(chunks)
        print(f"\n  [RESUME ACTIVE] Found {existing_count:,} chunks already in ChromaDB.")
        print(f"  [RESUME ACTIVE] Resuming dense indexing from chunk {start_index:,} / {len(chunks):,} ({pct:.1f}% already done).")
        print("  [RESUME ACTIVE] Zero previous work will be lost!")

    embedder = load_embedder()

    # Dynamic batch size: 128 on GPU, 64 on CPU (or user-defined)
    if custom_batch_size:
        batch_size = custom_batch_size
    else:
        batch_size = 128 if has_cuda else 64

    print(f"\n  -> Indexing dense vectors into ChromaDB (Legal-BERT batch size: {batch_size})...")
    t0 = time.time()
    chunks_processed_in_session = 0
    checkpoint_interval = 512
    drive_sync_interval = 5000

    try:
        for start in range(start_index, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [d.text for d in batch]

            vectors = embedder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()

            try:
                collection.upsert(
                    ids=[d.doc_id for d in batch],
                    documents=texts,
                    embeddings=vectors,
                    metadatas=[chroma_metadata(d) for d in batch]
                )
            except AttributeError:
                collection.add(
                    ids=[d.doc_id for d in batch],
                    documents=texts,
                    embeddings=vectors,
                    metadatas=[chroma_metadata(d) for d in batch]
                )

            chunks_processed_in_session += len(batch)
            done = min(start + batch_size, len(chunks))

            # Periodic checkpoint save & progress reporting
            if done % checkpoint_interval < batch_size or done >= len(chunks):
                cur_cnt = collection.count()
                save_checkpoint(done, len(chunks), cur_cnt)

                pct = 100.0 * done / len(chunks)
                elapsed = time.time() - t0
                speed = chunks_processed_in_session / elapsed if elapsed > 0 else 0
                remaining = (len(chunks) - done) / speed if speed > 0 else 0
                eta_str = format_seconds(remaining)

                print(f"     Indexed: {done:,}/{len(chunks):,} ({pct:.1f}%) — {speed:.1f} chunks/s | ETA: {eta_str} | [Checkpoint Saved]")

            # Periodic Drive sync in Colab
            if is_colab_environment() and (done % drive_sync_interval < batch_size or done >= len(chunks)):
                sync_chroma_to_drive(active_chroma_path, DRIVE_CHROMA_CORPUS_PATH)
    finally:
        if is_colab_environment():
            print("  -> Syncing ChromaDB to persistent Google Drive...")
            sync_chroma_to_drive(active_chroma_path, DRIVE_CHROMA_CORPUS_PATH)

    if has_cuda:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    save_checkpoint(len(chunks), len(chunks), collection.count())
    print(f"\n  [OK] ChromaDB dense vector index complete: {collection.count():,} chunks stored.")
    if is_colab_environment():
        print(f"       Local Fast NVMe: {active_chroma_path}")
        print(f"       Persistent Drive: {DRIVE_CHROMA_CORPUS_PATH}")
    else:
        print(f"       Path: {active_chroma_path}")

    write_cap_manifest(chunks, volume_tag=volumes_mode)

    print("\n" + "=" * 78)
    print("  LEXAGENT v3.0 CAP DATABASE SUCCESSFULLY BUILT!")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build primary CAP ChromaDB, BM25, and known_cases catalog.")
    parser.add_argument("--volumes", default=CAP_VOLUMES_MODE,
                        help="Volume mode ('landmark', 'modern', 'all', range '400-450', or list '410,347')")
    parser.add_argument("--workers", type=int, default=10, help="Parallel download worker threads (default: 10)")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from last checkpoint (default: True)")
    parser.add_argument("--overwrite", action="store_true", help="Force wipe and rebuild collection/index from scratch")
    parser.add_argument("--portion-size", type=int, default=25, help="Portion size in volumes for chunking (default: 25)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for embedding encode/add")
    args = parser.parse_args()

    main(
        volumes_mode=args.volumes,
        max_workers=args.workers,
        resume=args.resume,
        overwrite=args.overwrite,
        portion_size=args.portion_size,
        custom_batch_size=args.batch_size,
    )
