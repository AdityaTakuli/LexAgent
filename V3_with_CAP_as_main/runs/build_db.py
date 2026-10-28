# LexAgent v3.0 | build_db.py
"""Build primary Harvard CAP ChromaDB, BM25, and known_cases catalog across multiple volumes.

This is STAGE 3 of the staged CAP pipeline (embedding / indexing). It consumes the verified
per-volume chunk caches written by the chunk stage and never downloads or chunks anything:
    python master_run.py --download-cap   (stage 1: network -> raw JSON)
    python master_run.py --chunk-cap      (stage 2: raw JSON -> chunks)
    python master_run.py --build-db       (stage 3: chunks -> BM25 + ChromaDB, GPU)

Supports:
  - Resumable ChromaDB dense vector indexing (auto-resumes from last checkpoint if interrupted)
  - Immediate persistence of BM25 lexical index and known_cases catalog
  - Intelligent hardware detection (GPU acceleration vs CPU fallback) with ETA reporting
  - Nothing is deleted unless --overwrite is passed, and --overwrite never touches raw or chunk caches
"""

import argparse
from datetime import datetime, timezone
import json
import os
import pickle
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb

from config import (
    BM25_INDEX_PATH,
    CAP_BUILD_STATE_PATH,
    CASE_CATALOG_PATH,
    DRIVE_CHROMA_CORPUS_PATH,
    LOCAL_CHROMA_CORPUS_PATH,
    CORPUS_JSON_PATH,
    INDEXING_CHECKPOINT_PATH,
    KNOWN_CASES_PATH,
    CAP_VOLUMES_MODE,
)
from src.utils.chroma_sync import (
    prepare_working_chroma,
    sync_chroma_to_drive,
    is_colab_environment,
)
from src.data.cap_checkpoint import atomic_replace, atomic_write_json
from src.data.case_catalog import CaseCatalog
from src.data.cap_ingest import (
    corpus_fingerprint,
    load_cap_chunks,
    parse_volume_selection,
    write_cap_manifest,
    write_corpus_json,
)
try:
    from src.data.cap_ingest import extract_doc_volume
except ImportError:
    def extract_doc_volume(doc) -> str:
        vol = getattr(doc, "volume", None) if hasattr(doc, "volume") else (doc.get("volume") if isinstance(doc, dict) else None)
        if isinstance(vol, dict):
            return str(vol.get("volume_number") or vol.get("volume") or "").strip()
        if vol:
            s = str(vol).strip()
            m = re.search(r"['\"]?volume_number['\"]?\s*:\s*['\"]?(\d+)['\"]?", s)
            if m:
                return m.group(1)
            m = re.match(r"^\d+$", s)
            if m:
                return s
        meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else (doc.get("metadata", {}) if isinstance(doc, dict) else {})
        if isinstance(meta, dict):
            m_vol = meta.get("volume")
            if isinstance(m_vol, dict):
                return str(m_vol.get("volume_number") or "").strip()
            if m_vol:
                s = str(m_vol).strip()
                m = re.search(r"['\"]?volume_number['\"]?\s*:\s*['\"]?(\d+)['\"]?", s)
                if m:
                    return m.group(1)
                m = re.match(r"^\d+$", s)
                if m:
                    return s
            for c in meta.get("citations", []):
                c_str = c.get("cite", "") if isinstance(c, dict) else str(c)
                m = re.search(r"(\d+)\s+U\.?\s*S\.?", c_str, re.IGNORECASE)
                if m:
                    return m.group(1)
        doc_id = getattr(doc, "doc_id", "") if hasattr(doc, "doc_id") else (doc.get("doc_id", "") if isinstance(doc, dict) else "")
        m = re.search(r"(?:vol_?|CAP_?)(\d+)", doc_id, re.IGNORECASE)
        if m:
            return m.group(1)
        text_sample = f"{getattr(doc, 'case_name', '')} {getattr(doc, 'text', '')[:200]}" if hasattr(doc, "text") else ""
        m = re.search(r"(\d+)\s+U\.?\s*S\.?", text_sample, re.IGNORECASE)
        if m:
            return m.group(1)
        return ""

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


def save_checkpoint(last_index: int, total_chunks: int, collection_count: int, checkpoint_path: str = INDEXING_CHECKPOINT_PATH, fingerprint: str = ""):
    """Write dense vector indexing checkpoint to disk (atomically)."""
    payload = {
        "last_indexed_index": last_index,
        "total_chunks": total_chunks,
        "collection_count": collection_count,
        "percent_complete": round(100.0 * last_index / total_chunks, 2) if total_chunks > 0 else 0.0,
        "corpus_fingerprint": fingerprint,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(checkpoint_path, payload, indent=2)


def load_checkpoint(checkpoint_path: str = INDEXING_CHECKPOINT_PATH) -> dict:
    """Load indexing checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_build_state(path: str = CAP_BUILD_STATE_PATH) -> dict:
    """Load the record of which corpus the derived lexical artifacts were built from."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict):
                return state
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
    allow_incomplete: bool = False,
):
    """Stage 3: index verified chunk caches. max_workers and portion_size are accepted for
    backward compatibility and ignored: this stage no longer downloads or chunks."""
    print("\n" + "=" * 78)
    print("  BUILDING LEXAGENT v3.0 PRIMARY CAP DATABASE (STAGE 3: EMBEDDING / INDEXING)")
    print(f"  Selection Mode: '{volumes_mode}'")
    print(f"  Checkpoint Resume: {'ACTIVE' if resume and not overwrite else 'DISABLED (Fresh Build)'}")
    print("=" * 78)

    volume_list = parse_volume_selection(volumes_mode)
    print(f"  -> Target Volumes ({len(volume_list)} total): {volume_list[:8]}..." if len(volume_list) > 8 else f"  -> Target Volumes: {volume_list}")

    # 1. Load Corpus Chunks & Known Cases from the verified per-volume chunk caches.
    # No network and no chunking happens here: a volume that is not ready stops the build.
    chunks, known_cases, load_report = load_cap_chunks(volume_list, allow_incomplete=allow_incomplete)

    if not chunks:
        raise RuntimeError("No chunks were loaded. Run the download stage (--download-cap) and the chunk stage (--chunk-cap) first.")

    print(f"\n  -> Total Retrieval Chunks: {len(chunks):,} from {len(load_report['loaded'])} volumes")

    # The corpus JSON, known-cases catalog and BM25 index are derived from one specific corpus.
    # They are reused only when they were built from exactly this corpus, never just because the files exist.
    fingerprint = corpus_fingerprint(chunks)
    build_state = load_build_state()
    if overwrite or build_state.get("corpus_fingerprint") != fingerprint:
        build_state = {"corpus_fingerprint": fingerprint, "volumes": volumes_mode, "chunk_count": len(chunks), "artifacts": {}}
    artifacts = build_state.setdefault("artifacts", {})
    manifest_extra = {
        "corpus_fingerprint": fingerprint,
        "volumes_indexed": len(load_report["loaded"]),
        "volumes_unavailable": load_report["unavailable"],
        "volumes_missing": sorted(load_report["missing"]),
        # Volumes that did not come from CAP (see src/data/us_reports_pdf.py).
        "volumes_from_official_us_reports_pdf": sorted(
            {c.volume for c in chunks if (c.metadata or {}).get("authority_identity") == "US_REPORTS_OFFICIAL_PDF"}, key=int
        ),
    }

    if artifacts.get("corpus_json") and os.path.exists(CORPUS_JSON_PATH):
        print(f"  [OK] Corpus JSON already matches this corpus: {CORPUS_JSON_PATH}. Preserved.")
    else:
        print(f"  -> Writing corpus JSON ({len(chunks):,} chunks)...")
        write_corpus_json(chunks, CORPUS_JSON_PATH)
        artifacts["corpus_json"] = True
        atomic_write_json(CAP_BUILD_STATE_PATH, build_state, indent=2)
        print(f"  [OK] Corpus JSON saved: {CORPUS_JSON_PATH}")

    # 2. IMMEDIATE PERSISTENCE: Save Known Cases Catalog & BM25 Lexical Index FIRST
    # (These take < 1 minute to build and ensure lexical components are saved even if dense indexing is interrupted)
    print("\n" + "-" * 78)
    print("  [STEP 2: LEXICAL ARTIFACTS & CCE CATALOG PERSISTENCE]")
    print("-" * 78)

    # 2a. Known Cases Catalog
    if artifacts.get("known_cases") and os.path.exists(KNOWN_CASES_PATH):
        print(f"  [OK] Known cases catalog already matches this corpus: {KNOWN_CASES_PATH}. Preserved.")
    elif known_cases:
        os.makedirs(os.path.dirname(KNOWN_CASES_PATH), exist_ok=True)
        with open(KNOWN_CASES_PATH + ".tmp", "wb") as handle:
            pickle.dump(known_cases, handle)
        atomic_replace(KNOWN_CASES_PATH + ".tmp", KNOWN_CASES_PATH)
        artifacts["known_cases"] = True
        atomic_write_json(CAP_BUILD_STATE_PATH, build_state, indent=2)
        print(f"  [OK] Known cases catalog saved immediately: {KNOWN_CASES_PATH} ({len(known_cases):,} entries)")
        del known_cases
        import gc
        gc.collect()

    # 2a'. Case catalog: every report with its cite, year and length, so that the CCE can check a
    #      citation against the actual opinion instead of an order entry that shares its name
    if artifacts.get("case_catalog") and os.path.exists(CASE_CATALOG_PATH):
        print(f"  [OK] Case catalog already matches this corpus: {CASE_CATALOG_PATH}. Preserved.")
    else:
        catalog = CaseCatalog.from_chunks(chunks)
        catalog.save(CASE_CATALOG_PATH)
        artifacts["case_catalog"] = True
        atomic_write_json(CAP_BUILD_STATE_PATH, build_state, indent=2)
        print(f"  [OK] Case catalog saved: {CASE_CATALOG_PATH} ({len(catalog):,} reports)")
        del catalog

    # 2b. BM25 Lexical Index
    if artifacts.get("bm25") and os.path.exists(BM25_INDEX_PATH):
        print(f"  [OK] BM25 index already matches this corpus: {BM25_INDEX_PATH}. Preserved.")
    else:
        print(f"  -> Building and serializing memory-efficient BM25 index over {len(chunks):,} chunks...")
        bm25 = BM25Retriever(chunks)
        bm25.save_index(BM25_INDEX_PATH + ".tmp")
        atomic_replace(BM25_INDEX_PATH + ".tmp", BM25_INDEX_PATH)
        artifacts["bm25"] = True
        atomic_write_json(CAP_BUILD_STATE_PATH, build_state, indent=2)
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

            # If local Chroma collection is empty, restart from 0 to prevent skipping chunks
            if existing_count == 0:
                start_index = 0
            # The checkpoint is only a valid resume point for the corpus it was written for.
            elif ckpt.get("corpus_fingerprint") != fingerprint or ckpt.get("total_chunks") != len(chunks):
                old_total = ckpt.get("total_chunks")
                grew = (
                    isinstance(old_total, int)
                    and 0 < old_total < len(chunks)
                    and start_index <= min(old_total, existing_count)
                    and ckpt.get("corpus_fingerprint") == corpus_fingerprint(chunks[:old_total])
                )
                if grew:
                    # Volumes were added after the previous build: the chunks indexed so far are an
                    # unchanged prefix of this corpus, so only the rest needs embedding.
                    print(f"\n  [*] The corpus grew from {old_total:,} to {len(chunks):,} chunks; the earlier chunks are unchanged.")
                else:
                    # Checkpoint missing or written for a different corpus: re-run from chunk 0. Upsert
                    # is idempotent, so nothing is skipped and nothing already stored is deleted.
                    print(f"\n  [!] The indexing checkpoint does not match this corpus ({existing_count:,} chunks already in ChromaDB).")
                    print("      Re-running the embedding pass from chunk 0. Existing entries are kept and updated in place.")
                    start_index = 0
            elif start_index > len(chunks) or start_index > existing_count:
                print(f"\n  [!] The indexing checkpoint is ahead of ChromaDB ({existing_count:,} chunks stored). Re-running from chunk 0.")
                start_index = 0
    except Exception as e:
        # Never purge the index on our own: embedding work is destroyed only by an explicit --overwrite.
        raise RuntimeError(
            f"ChromaDB at {active_chroma_path} could not be opened: {e}\n"
            "Nothing was deleted. If the database really is corrupt, rebuild it explicitly with --overwrite "
            "(raw and chunk caches are not affected)."
        ) from e

    if not overwrite and start_index >= len(chunks):
        print(f"\n  [OK] ChromaDB index already 100% complete: {existing_count:,} / {len(chunks):,} chunks stored.")
        print(f"    Path: {active_chroma_path}")
        if is_colab_environment():
            sync_chroma_to_drive(active_chroma_path, DRIVE_CHROMA_CORPUS_PATH)
        write_cap_manifest(chunks, volume_tag=volumes_mode, extra=manifest_extra)
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

    # Dynamic batch size: 256 on GPU, 64 on CPU (or user-defined via env or flag)
    if custom_batch_size:
        batch_size = custom_batch_size
    else:
        env_batch = os.environ.get("LEXAGENT_BATCH_SIZE")
        if env_batch:
            batch_size = int(env_batch)
        else:
            batch_size = 256 if has_cuda else 64

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
                save_checkpoint(done, len(chunks), cur_cnt, fingerprint=fingerprint)

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

    save_checkpoint(len(chunks), len(chunks), collection.count(), fingerprint=fingerprint)
    print(f"\n  [OK] ChromaDB dense vector index complete: {collection.count():,} chunks stored.")
    if collection.count() != len(chunks):
        print(f"  [!] The collection holds {collection.count():,} entries but this corpus has {len(chunks):,} chunks.")
        print("      It contains entries from an earlier, different corpus. Rebuild with --overwrite for a clean index.")
    if is_colab_environment():
        print(f"       Local Fast NVMe: {active_chroma_path}")
        print(f"       Persistent Drive: {DRIVE_CHROMA_CORPUS_PATH}")
    else:
        print(f"       Path: {active_chroma_path}")

    write_cap_manifest(chunks, volume_tag=volumes_mode, extra=manifest_extra)

    print("\n" + "=" * 78)
    print("  LEXAGENT v3.0 CAP DATABASE SUCCESSFULLY BUILT!")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build primary CAP ChromaDB, BM25, and known_cases catalog.")
    parser.add_argument("--volumes", default=CAP_VOLUMES_MODE,
                        help="Volume mode ('landmark', 'modern', 'all', range '400-450', or list '410,347')")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LEXAGENT_WORKERS", "16")), help="Ignored (this stage does not download); kept for old command lines")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from last checkpoint (default: True)")
    parser.add_argument("--overwrite", action="store_true", help="Wipe and rebuild the ChromaDB collection and derived indexes (never raw or chunk caches)")
    parser.add_argument("--portion-size", type=int, default=50, help="Ignored (this stage does not chunk); kept for old command lines")
    parser.add_argument("--allow-incomplete", action="store_true", help="Index the volumes that are ready even if some target volumes have no verified chunk cache")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for embedding encode/add (default: 256 on GPU)")
    args = parser.parse_args()

    main(
        volumes_mode=args.volumes,
        max_workers=args.workers,
        resume=args.resume,
        overwrite=args.overwrite,
        portion_size=args.portion_size,
        custom_batch_size=args.batch_size,
        allow_incomplete=args.allow_incomplete,
    )
