# LexAgent v3.0 | cap_ingest.py
"""
CAP (Caselaw Access Project) Multi-Volume Ingestion Engine.

The pipeline is split into separate, crash-resumable stages:
  1. DOWNLOAD (run_download_stage): CAP network -> validated raw JSON cache. No chunking.
  2. CHUNK    (run_chunk_stage):    verified raw JSON -> chunk cache. No network.
  3. INDEX    (runs/build_db.py):   verified chunk caches -> BM25 + ChromaDB. No network.

A cache file is trusted only when it carries an explicit "complete" status AND passes
validation. The mere existence of a file never means the volume succeeded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import requests

from config import (
    CAP_BASE_URL,
    CAP_RAW_DIR,
    CAP_CHUNKS_DIR,
    CAP_CHECKPOINT_INTERVAL,
    CAP_CHUNK_CHECKPOINT_PATH,
    CAP_DOWNLOAD_CHECKPOINT_PATH,
    CORPUS_JSON_PATH,
    CORPUS_MANIFEST_PATH,
    DEFAULT_PARALLEL_WORKERS,
    LANDMARK_CAP_VOLUMES,
)
from src.data.cap_checkpoint import (
    COMPLETE,
    FAILED,
    IN_PROGRESS,
    NOT_STARTED,
    UNAVAILABLE,
    StageCheckpoint,
    atomic_replace,
    atomic_write_json,
    utc_now,
)
from src.data.chunking import chunk_corpus
from src.data.corpus_schema import CorpusDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Cache file formats. Files without these explicit fields (older caches) are never trusted.
RAW_SCHEMA_VERSION = 2
CHUNK_SCHEMA_VERSION = 2
RAW_COMPLETE = "complete"
RAW_FAILED = "failed"
RAW_UNAVAILABLE = "unavailable"
CHUNK_COMPLETE = "complete"

HTTP_MAX_ATTEMPTS = 4


class CapNotFound(Exception):
    """The CAP server answered HTTP 404 for a resource."""


class VolumeNotAvailableError(RuntimeError):
    """A raw volume is missing or unverified, so it cannot be chunked. The chunk stage never downloads."""

    def __init__(self, volume: str, state: str, reason: str):
        self.volume = str(volume)
        self.state = state
        self.reason = reason
        if state == UNAVAILABLE:
            message = f"Volume {volume} is not published by CAP ({reason}); there is nothing to chunk."
        else:
            message = f"Volume {volume} is not available/verified ({reason}). Run the download stage (--download-cap)."
        super().__init__(message)


class ChunksNotAvailableError(RuntimeError):
    """Target volumes have no verified chunk cache. The index build never downloads or chunks."""

    def __init__(self, missing: Dict[str, str]):
        self.missing = dict(missing)
        listing = "; ".join(f"Vol {vol}: {reason}" for vol, reason in list(self.missing.items())[:10])
        more = f" (+{len(self.missing) - 10} more)" if len(self.missing) > 10 else ""
        super().__init__(
            f"{len(self.missing)} target volume(s) have no verified chunk cache: {listing}{more}. "
            "Run the download stage (--download-cap) and the chunk stage (--chunk-cap) first, "
            "or pass --allow-incomplete to index only the volumes that are ready."
        )


@dataclass
class ArtifactCheck:
    """Result of validating one cache file (raw volume or chunk volume)."""
    state: str  # COMPLETE | FAILED | UNAVAILABLE | NOT_STARTED
    reason: str = ""
    payload: Optional[dict] = None
    sha256: str = ""

    @property
    def ok(self) -> bool:
        return self.state == COMPLETE


def extract_doc_volume(doc) -> str:
    """Robustly extract volume number string from a CorpusDocument or dict."""
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


def parse_volume_selection(mode_or_range: str) -> List[str]:
    """Parse a volume specification into a sorted list of volume strings.

    Supported formats:
      - "landmark": Preset of 27 landmark constitutional volumes.
      - "modern": Volumes 300 to 585 (~1940 to 2018).
      - "all": Volumes 1 to 600 inclusive (exactly 600 target volumes).
      - "400-450": Range from 400 to 450 inclusive.
      - "410,347,384": Explicit comma-separated volume list.
    """
    cleaned = str(mode_or_range).strip().lower()

    if cleaned == "landmark":
        return list(LANDMARK_CAP_VOLUMES)

    if cleaned == "modern":
        return [str(v) for v in range(300, 586)]

    if cleaned == "all":
        return [str(v) for v in range(1, 601)]

    if "-" in cleaned:
        parts = cleaned.split("-")
        try:
            start, end = int(parts[0]), int(parts[1])
            return [str(v) for v in range(start, end + 1)]
        except ValueError:
            pass

    if "," in cleaned:
        return [v.strip() for v in cleaned.split(",") if v.strip()]

    # Single volume
    return [cleaned]


def raw_volume_path(volume: str, raw_dir: str = CAP_RAW_DIR) -> str:
    return os.path.join(raw_dir, f"cap_raw_us_{volume}.json")


def chunk_volume_path(volume: str, chunks_dir: str = CAP_CHUNKS_DIR) -> str:
    return os.path.join(chunks_dir, f"cap_chunks_us_{volume}.json")


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_from_casebody(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    body = value.get("data", value)
    if isinstance(body, str):
        return body
    if not isinstance(body, dict):
        return ""
    parts = [str(body.get("head_matter", ""))]
    for opinion in body.get("opinions", []) or []:
        if isinstance(opinion, dict):
            parts.append(str(opinion.get("text", "")))
    return "\n\n".join(part for part in parts if part).strip()


def _case_url(record: dict, volume: str) -> str:
    for key in ("casebody_url", "case_url", "download_url"):
        if record.get(key):
            return record[key]
    direct_url = record.get("url")
    if isinstance(direct_url, str) and direct_url.endswith(".json"):
        return direct_url
    casebody = record.get("casebody")
    if isinstance(casebody, dict) and casebody.get("url"):
        return casebody["url"]
    for key in ("case_file_name", "file_name", "filename"):
        filename = record.get(key)
        if filename:
            filename = str(filename)
            if filename.startswith("http"):
                return filename
            if not filename.endswith(".json"):
                filename += ".json"
            return f"{CAP_BASE_URL}/us/{volume}/cases/{filename.lstrip('/')}"
    first_page = record.get("first_page")
    if first_page is not None:
        try:
            return f"{CAP_BASE_URL}/us/{volume}/cases/{int(first_page):04d}-01.json"
        except (TypeError, ValueError):
            pass
    raise ValueError(f"CAP metadata has no static case filename for record {record.get('id')}")


# ══════════════════════════════════════════════════════
# STAGE 1 — DOWNLOAD (network + raw JSON only)
# ══════════════════════════════════════════════════════
_SESSION: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()


def _session() -> requests.Session:
    """One pooled HTTP session shared by the download worker threads."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            session = requests.Session()
            # Sized so every worker thread keeps its own connection alive even at high --workers values.
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=256)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _SESSION = session
        return _SESSION


def _http_get_json(url: str, timeout: int = 45) -> Any:
    """GET a CAP JSON resource, retrying transient failures. Raises CapNotFound on HTTP 404."""
    last_error: Optional[Exception] = None
    for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
        try:
            response = _session().get(url, timeout=timeout)
            if response.status_code == 404:
                raise CapNotFound(url)
            response.raise_for_status()
            return response.json()
        except CapNotFound:
            raise
        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt < HTTP_MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"{type(last_error).__name__}: {last_error} (after {HTTP_MAX_ATTEMPTS} attempts)")


def _download_single_case(record: dict, volume: str, timeout: int = 30) -> Tuple[str, Optional[dict], str]:
    """Fetch one case body. Returns (outcome, case, error); outcome is 'ok', 'empty_text' or 'failed'."""
    try:
        url = _case_url(record, volume)
        full = _http_get_json(url, timeout)
        if not isinstance(full, dict):
            return "failed", None, "case JSON is not an object"
        merged = {**record, **full}
        casebody = merged.get("casebody", full)
        if _text_from_casebody(casebody):
            merged["casebody"] = casebody
            return "ok", merged, ""
        return "empty_text", None, ""
    except CapNotFound:
        return "failed", None, "HTTP 404 for case file"
    except Exception as e:
        return "failed", None, f"{type(e).__name__}: {e}"


def _raw_payload_problem(data: dict) -> str:
    """Return why a raw payload marked complete is not actually complete, or '' if it is consistent."""
    cases = data.get("cases")
    if not isinstance(cases, list):
        return "cases is not a list"
    metadata_count = data.get("metadata_count")
    if not isinstance(metadata_count, int) or metadata_count <= 0:
        return "metadata was not retrieved (metadata_count missing or zero)"
    if data.get("failed_count") != 0 or data.get("failed_cases"):
        return f"{data.get('failed_count')} case download(s) failed"
    if data.get("cases_count") != len(cases):
        return f"cases_count={data.get('cases_count')} but {len(cases)} cases are stored"
    empty_count = data.get("empty_text_count")
    if not isinstance(empty_count, int) or metadata_count != len(cases) + empty_count:
        return f"metadata lists {metadata_count} cases but {len(cases)} stored + {empty_count} empty are accounted for"
    seen_ids = set()
    for case in cases:
        if not isinstance(case, dict) or not _text_from_casebody(case.get("casebody")):
            return "a stored case has no text"
        seen_ids.add(str(case.get("id")))
    if len(seen_ids) != len(cases):
        return "stored cases contain duplicate ids"
    return ""


def inspect_raw_volume(volume: str, raw_dir: str = CAP_RAW_DIR) -> ArtifactCheck:
    """Validate one raw volume cache. Only an explicit 'complete' status that passes validation is COMPLETE."""
    volume = str(volume)
    path = raw_volume_path(volume, raw_dir)
    if not os.path.exists(path):
        return ArtifactCheck(NOT_STARTED, "no raw cache file")
    try:
        with open(path, "rb") as handle:
            blob = handle.read()
        data = json.loads(blob)
    except Exception as e:
        return ArtifactCheck(FAILED, f"raw cache is unreadable or corrupt: {e}")
    if not isinstance(data, dict):
        return ArtifactCheck(FAILED, "raw cache is not a JSON object")
    if str(data.get("volume")) != volume:
        return ArtifactCheck(FAILED, f"raw cache is for volume {data.get('volume')!r}, expected {volume}", data)

    status = data.get("status")
    if status == RAW_UNAVAILABLE:
        return ArtifactCheck(UNAVAILABLE, data.get("error") or "volume not published by CAP", data)
    if data.get("schema_version") != RAW_SCHEMA_VERSION or status is None:
        return ArtifactCheck(FAILED, "raw cache has no explicit completion status (older or partial cache)", data)
    if status != RAW_COMPLETE:
        return ArtifactCheck(FAILED, data.get("error") or f"raw cache status is {status!r}", data)
    problem = _raw_payload_problem(data)
    if problem:
        return ArtifactCheck(FAILED, f"raw cache failed validation: {problem}", data)
    return ArtifactCheck(COMPLETE, "", data, hashlib.sha256(blob).hexdigest())


def get_downloaded_volumes(raw_dir: str = CAP_RAW_DIR) -> Tuple[Set[str], int]:
    """Scan raw_dir for volumes whose raw cache is verified COMPLETE.

    A file that merely exists is not counted: failed, partial, corrupt and
    not-published volumes are all excluded.

    Returns:
        (set_of_verified_volume_strings, max_volume_int)
    """
    if not os.path.exists(raw_dir):
        return set(), 0

    downloaded = set()
    max_vol = 0

    for fname in os.listdir(raw_dir):
        if fname.startswith("cap_raw_us_") and fname.endswith(".json"):
            vol_str = fname[len("cap_raw_us_"):-len(".json")]
            if inspect_raw_volume(vol_str, raw_dir).ok:
                downloaded.add(vol_str)
                if vol_str.isdigit():
                    max_vol = max(max_vol, int(vol_str))

    return downloaded, max_vol


def _raw_header(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "cases"}


def download_cap_volume(
    volume: str,
    raw_dir: str = CAP_RAW_DIR,
    max_workers: int = 10,
    timeout: int = 45,
    force: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Download one CAP volume into the raw cache. Network and raw JSON only: nothing is chunked.

    A verified COMPLETE cache is left alone unless force=True. Anything else (missing, failed,
    partial, corrupt, older format) is fetched again; cases an earlier failed attempt already
    stored are reused so only the missing ones hit the network.

    The cache is always written with an explicit status:
      - "complete":    metadata retrieved and every listed case accounted for, none failed
      - "failed":      metadata or at least one case could not be downloaded
      - "unavailable": CAP answered HTTP 404 for the volume (it is not published)

    Returns the cache header (every field except the cases themselves).
    """
    volume = str(volume)
    os.makedirs(raw_dir, exist_ok=True)
    cache_file = raw_volume_path(volume, raw_dir)

    existing = inspect_raw_volume(volume, raw_dir)
    if existing.ok and not force:
        return {**_raw_header(existing.payload), "skipped": True}

    reusable: Dict[str, dict] = {}
    if not force and existing.payload and existing.payload.get("schema_version") == RAW_SCHEMA_VERSION:
        for case in existing.payload.get("cases") or []:
            if isinstance(case, dict) and _text_from_casebody(case.get("casebody")):
                reusable[str(case.get("id"))] = case

    metadata_url = f"{CAP_BASE_URL}/us/{volume}/CasesMetadata.json"
    result = {
        "schema_version": RAW_SCHEMA_VERSION,
        "volume": volume,
        "reporter": "us",
        "status": RAW_FAILED,
        "source_url": metadata_url,
        "metadata_count": 0,
        "cases_count": len(reusable),
        "empty_text_count": 0,
        "empty_text_case_ids": [],
        "failed_count": 0,
        "failed_cases": [],
        "error": None,
        "downloaded_at": utc_now(),
        "cases": list(reusable.values()),
    }

    def store() -> dict:
        # A forced refresh that fails must not replace a verified copy.
        if existing.ok and result["status"] != RAW_COMPLETE:
            return {**_raw_header(result), "kept_previous_complete": True}
        atomic_write_json(cache_file, result)
        return _raw_header(result)

    try:
        metadata = _http_get_json(metadata_url, timeout)
    except CapNotFound:
        result.update(status=RAW_UNAVAILABLE, cases=[], cases_count=0,
                      error="CasesMetadata.json returned HTTP 404 (CAP does not publish this volume)")
        return store()
    except Exception as e:
        result["error"] = f"metadata download failed: {e}"
        logger.warning("Volume %s: %s", volume, result["error"])
        return store()

    if not isinstance(metadata, list) or not metadata:
        result["error"] = "CasesMetadata.json is empty or not a list"
        return store()

    total = len(metadata)
    outcomes: List[Optional[Tuple[str, Optional[dict], str]]] = [None] * total
    pending: List[int] = []
    for idx, record in enumerate(metadata):
        if not isinstance(record, dict):
            outcomes[idx] = ("failed", None, "metadata record is not an object")
        elif str(record.get("id")) in reusable:
            outcomes[idx] = ("ok", reusable[str(record.get("id"))], "")
        else:
            pending.append(idx)

    done = total - len(pending)
    if progress:
        progress(done, total)

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_map = {executor.submit(_download_single_case, metadata[idx], volume, timeout): idx for idx in pending}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                outcomes[idx] = future.result()
            except Exception as e:
                outcomes[idx] = ("failed", None, f"{type(e).__name__}: {e}")
            done += 1
            if progress:
                progress(done, total)
    finally:
        # Do not wait for queued requests when interrupted (Ctrl+C); nothing partial is written.
        executor.shutdown(wait=False, cancel_futures=True)

    # Cases are stored in metadata order so the cache (and everything derived from it) is deterministic.
    cases: List[dict] = []
    empty_ids: List[Any] = []
    failed: List[dict] = []
    for record, (outcome, case, error) in zip(metadata, outcomes):
        if outcome == "ok":
            cases.append(case)
        elif outcome == "empty_text":
            empty_ids.append(record.get("id"))
        else:
            failed.append({
                "id": record.get("id") if isinstance(record, dict) else None,
                "file_name": record.get("file_name") if isinstance(record, dict) else None,
                "error": error,
            })

    result.update(
        metadata_count=total,
        cases=cases,
        cases_count=len(cases),
        empty_text_count=len(empty_ids),
        empty_text_case_ids=empty_ids,
        failed_count=len(failed),
        failed_cases=failed,
    )
    if failed:
        result["error"] = f"{len(failed)} of {total} case downloads failed (first: {failed[0]['error']})"
    else:
        result["status"] = RAW_COMPLETE
        problem = _raw_payload_problem(result)
        if problem:
            result.update(status=RAW_FAILED, error=f"downloaded volume failed validation: {problem}")

    header = store()
    logger.debug("Volume %s: %s (%d/%d cases).", volume, result["status"], len(cases), total)
    return header


def load_raw_volume(volume: str, raw_dir: str = CAP_RAW_DIR) -> List[dict]:
    """Return the cases of a verified raw volume. Never downloads: raises if the volume is not verified."""
    check = inspect_raw_volume(volume, raw_dir)
    if not check.ok:
        raise VolumeNotAvailableError(volume, check.state, check.reason)
    return check.payload["cases"]


# ══════════════════════════════════════════════════════
# STAGE RUNNER (shared by the download and chunk stages)
# ══════════════════════════════════════════════════════
class _LiveProgress:
    """Live progress for the volume being processed.

    On a terminal the line is redrawn in place. When output is piped to a log file,
    a line is printed every few seconds instead so the log still shows movement.
    """

    def __init__(self):
        self.is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.prefix = ""
        self._last = 0.0

    def start(self, prefix: str) -> None:
        self.prefix = prefix
        self._last = 0.0 if self.is_tty else time.monotonic()

    def update(self, done: int, total: int) -> None:
        now = time.monotonic()
        if self.is_tty:
            if now - self._last < 0.1 and done < total:
                return
            self._last = now
            sys.stdout.write(f"\r{self.prefix} {done}/{total} cases")
            sys.stdout.flush()
        elif now - self._last >= 10.0 and done < total:
            self._last = now
            print(f"{self.prefix} ... {done}/{total} cases", flush=True)

    def finish(self, line: str) -> None:
        if self.is_tty:
            sys.stdout.write("\r" + " " * 110 + "\r")
        print(line, flush=True)


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(max(seconds, 0)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_volume_ranges(volumes: List[str]) -> str:
    """Compact listing: ['573', '574', '575', '580'] -> '573-575, 580'."""
    if not volumes:
        return "none"
    if not all(str(v).isdigit() for v in volumes):
        return ", ".join(volumes)
    numbers = sorted(int(v) for v in volumes)
    ranges = []
    start = prev = numbers[0]
    for number in numbers[1:] + [None]:
        if number is not None and number == prev + 1:
            prev = number
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        if number is not None:
            start = prev = number
    return ", ".join(ranges)


def _run_stage(
    ckpt: StageCheckpoint,
    title: str,
    header_lines: List[str],
    inspect: Callable[[str], Tuple[str, dict]],
    process: Callable[[str, _LiveProgress], Tuple[str, dict]],
    describe: Callable[[str, dict], str],
    force: bool = False,
) -> dict:
    """Run one stage over ckpt.volumes: verify what is on disk, process the rest, checkpoint as it goes.

    A volume that fails is recorded as FAILED and the run moves on to the next one.
    """
    volumes = ckpt.volumes
    total = len(volumes)
    print("\n" + "=" * 78)
    print(f"  [{title}]")
    for line in header_lines:
        print(f"    {line}")
    print(f"    Target volumes:     {total} ({volumes[0]} ... {volumes[-1]})")
    print(f"    Checkpoint file:    {ckpt.path}")
    print(f"    Batch checkpoint:   every {ckpt.interval} volumes ({ckpt.batch_count()} batches)")
    print("=" * 78)
    print(f"  -> Verifying the cache files already on disk for {total} volumes...", flush=True)

    # Durable state is whatever validates on disk, not what the ledger remembers.
    to_process: List[str] = []
    carried: Counter = Counter()
    for vol in volumes:
        previous = ckpt.state(vol)
        state, info = inspect(vol)
        if previous == COMPLETE and state != COMPLETE:
            logger.warning("Volume %s was recorded COMPLETE but its cache no longer validates (%s). It will be redone.", vol, info.get("error"))
        if previous == IN_PROGRESS and state == NOT_STARTED:
            info = {**info, "error": "interrupted during a previous run"}
            carried["interrupted"] += 1
        elif state != COMPLETE or force:
            carried[state] += 1
        ckpt.set_state(vol, state, save=False, **info)
        if state != COMPLETE or force:
            to_process.append(vol)

    remaining = Counter(ckpt.batch_index(vol) for vol in to_process)
    for index in range(1, ckpt.batch_count() + 1):
        if remaining[index] == 0:
            ckpt.batch_commits.setdefault(str(index), utc_now())
    ckpt.save(run_state="RUNNING")

    print(f"    Verified COMPLETE:  {total - len(to_process)} (skipped)")
    breakdown = ", ".join(f"{label.lower().replace('_', ' ')}: {count}" for label, count in sorted(carried.items()))
    print(f"    To process now:     {len(to_process)}" + (f" ({breakdown})" if breakdown else ""), flush=True)

    live = _LiveProgress()
    started = time.monotonic()
    try:
        for position, vol in enumerate(to_process, 1):
            ckpt.set_state(vol, IN_PROGRESS)
            tag = f"  [{position}/{len(to_process)}] Vol {vol}:"
            live.start(tag)
            volume_started = time.monotonic()
            try:
                state, info = process(vol, live)
            except Exception as e:
                state, info = FAILED, {"error": f"{type(e).__name__}: {e}"}
            ckpt.set_state(vol, state, **info)

            counts = ckpt.summary()["counts"]
            elapsed = time.monotonic() - started
            eta = elapsed / position * (len(to_process) - position)
            live.finish(
                f"{tag} {state} - {describe(state, info)} | {time.monotonic() - volume_started:.1f}s"
                f" | overall {counts[COMPLETE]}/{total} complete, {counts[FAILED]} failed,"
                f" {counts[UNAVAILABLE]} unavailable | ETA {_format_duration(eta)}"
            )

            index = ckpt.batch_index(vol)
            remaining[index] -= 1
            if remaining[index] == 0:
                batch = ckpt.commit_batch(index)
                print(
                    f"  [CHECKPOINT] Batch {index}/{ckpt.batch_count()} (volumes {batch['first_volume']}-{batch['last_volume']}):"
                    f" {batch['complete']}/{batch['size']} COMPLETE, {len(batch['failed_volumes'])} failed,"
                    f" {len(batch['unavailable_volumes'])} unavailable - checkpoint saved.",
                    flush=True,
                )
    except KeyboardInterrupt:
        ckpt.save(run_state="INTERRUPTED")
        print("\n  [INTERRUPTED] Progress is saved. Run the same command again to resume.", flush=True)
        raise

    ckpt.save(run_state="FINISHED")
    summary = ckpt.summary()
    counts = summary["counts"]
    not_processed = counts[NOT_STARTED] + counts[IN_PROGRESS]
    print("\n" + "=" * 78)
    print(f"  [{title} - REPORT]  (elapsed {_format_duration(time.monotonic() - started)})")
    print(f"    Target volumes:     {total}")
    print(f"    COMPLETE:           {counts[COMPLETE]}")
    print(f"    FAILED:             {counts[FAILED]}  -> volumes {_format_volume_ranges(summary['failed_volumes'])}")
    print(f"    UNAVAILABLE:        {counts[UNAVAILABLE]}  -> volumes {_format_volume_ranges(summary['unavailable_volumes'])}")
    if not_processed:
        print(f"    NOT PROCESSED:      {not_processed}")
    for entry in summary["not_completed"][:15]:
        print(f"      Vol {entry['volume']}: {entry['status']} - {entry['reason']}")
    if len(summary["not_completed"]) > 15:
        print(f"      ... and {len(summary['not_completed']) - 15} more (full record: 'not_completed' in the checkpoint file)")
    verdict = "YES" if summary["all_target_volumes_complete"] else "NO"
    print(f"  ALL {total} TARGET VOLUMES COMPLETE: {verdict}")
    if not summary["all_target_volumes_complete"]:
        print(f"  Record of every volume not completed: {ckpt.path}")
    print("=" * 78, flush=True)
    return summary


def run_download_stage(
    volumes: List[str],
    raw_dir: str = CAP_RAW_DIR,
    checkpoint_path: str = CAP_DOWNLOAD_CHECKPOINT_PATH,
    checkpoint_interval: int = CAP_CHECKPOINT_INTERVAL,
    max_workers: int = DEFAULT_PARALLEL_WORKERS,
    selection: str = "custom",
    force: bool = False,
) -> dict:
    """STAGE 1: download raw CAP volumes. Network and raw JSON only: no chunking, no embedding, no ChromaDB.

    Resumes by default: volumes whose raw cache is verified COMPLETE are skipped, everything else
    (never started, failed, interrupted, corrupt) is downloaded. A volume that fails is recorded
    and skipped so the run continues. Returns the stage summary.
    """
    os.makedirs(raw_dir, exist_ok=True)
    ckpt = StageCheckpoint(checkpoint_path, "download", volumes, checkpoint_interval, selection)

    def ledger_info(check: ArtifactCheck) -> dict:
        payload = check.payload or {}
        info = {key: payload.get(key) for key in ("metadata_count", "cases_count", "empty_text_count", "failed_count") if key in payload}
        if check.ok:
            info["sha256"] = check.sha256
        else:
            info["error"] = check.reason
        return info

    def inspect(vol: str) -> Tuple[str, dict]:
        check = inspect_raw_volume(vol, raw_dir)
        return check.state, ledger_info(check)

    def process(vol: str, live: _LiveProgress) -> Tuple[str, dict]:
        download_cap_volume(vol, raw_dir=raw_dir, max_workers=max_workers, force=force, progress=live.update)
        # The volume is COMPLETE only if what is now on disk validates.
        return inspect(vol)

    def describe(state: str, info: dict) -> str:
        if state == COMPLETE:
            return f"{info.get('cases_count')}/{info.get('metadata_count')} cases ({info.get('empty_text_count')} without text)"
        return str(info.get("error"))

    return _run_stage(
        ckpt,
        "CAP STAGE 1: DOWNLOAD ONLY",
        [f"Source:             {CAP_BASE_URL}/us/<volume>/", f"Raw cache:          {raw_dir}", f"Worker threads:     {max_workers}"],
        inspect, process, describe, force=force,
    )


def extract_known_cases_from_chunks(chunks: Iterable[Any]) -> Set[str]:
    """Extract case names and citation references from chunks for CCE verification."""
    known_cases: Set[str] = set()
    for chunk in chunks:
        if isinstance(chunk, dict):
            c_name = chunk.get("case_name") or ""
            vol = str(chunk.get("volume") or "")
            rep = chunk.get("reporter") or "U.S."
            page = chunk.get("first_page") or 0
            citations = (chunk.get("metadata") or {}).get("citations", [])
        else:
            c_name = getattr(chunk, "case_name", "") or ""
            vol = str(getattr(chunk, "volume", "") or "")
            rep = getattr(chunk, "reporter", "U.S.") or "U.S."
            page = getattr(chunk, "first_page", 0) or 0
            citations = (getattr(chunk, "metadata", {}) or {}).get("citations", [])

        if c_name:
            known_cases.add(c_name.strip())
        if vol and page:
            known_cases.add(f"{vol} {rep} {page}".strip())
        for cite_item in citations:
            if isinstance(cite_item, dict) and cite_item.get("cite"):
                known_cases.add(cite_item["cite"].strip())
            elif isinstance(cite_item, str) and cite_item.strip():
                known_cases.add(cite_item.strip())
    return known_cases


# ══════════════════════════════════════════════════════
# STAGE 2 — CHUNK (raw JSON -> chunks only, no network)
# ══════════════════════════════════════════════════════
def _documents_from_cases(cases: Iterable[dict], default_volume: str = "") -> Tuple[List[CorpusDocument], Set[str]]:
    """Convert raw CAP case records into source documents and collect their known-case entries.

    CAP's case files do not carry the reporter volume, so default_volume (the volume the
    records were downloaded from) is used whenever a record has none of its own.
    """
    documents: list[CorpusDocument] = []
    known_cases: Set[str] = set()

    for record in cases:
        text = _text_from_casebody(record.get("casebody"))
        if len(text) < 100:
            continue
        case_id = str(record.get("id", ""))
        name = record.get("name_abbreviation") or record.get("name") or ""
        court = (record.get("court") or {}).get("name", "") if isinstance(record.get("court"), dict) else ""
        date = str(record.get("decision_date") or "")
        try:
            year = int(date[:4])
        except ValueError:
            year = 0

        first_page_int = 0
        try:
            first_page_int = int(record.get("first_page") or 0)
        except (ValueError, TypeError):
            pass

        vol_val = str(record.get("volume", "") or "")
        if not vol_val and record.get("volume_number"):
            vol_val = str(record.get("volume_number"))
        if not vol_val:
            vol_val = str(default_volume or "")

        citations = record.get("citations") or []
        digest = hashlib.sha256((case_id + text).encode("utf-8")).hexdigest()[:24]

        if name:
            known_cases.add(name.strip())
        if vol_val and first_page_int:
            known_cases.add(f"{vol_val} U.S. {first_page_int}".strip())
        for cite_item in citations:
            if isinstance(cite_item, dict) and cite_item.get("cite"):
                known_cases.add(cite_item["cite"].strip())

        documents.append(CorpusDocument(
            doc_id=f"CAP::{case_id}::{digest}",
            text=text,
            source="CAP",
            case_name=name,
            year=year,
            court=court,
            volume=vol_val,
            reporter="U.S.",
            first_page=first_page_int,
            metadata={
                "cap_id": case_id,
                "volume": vol_val,
                "decision_date": date,
                "docket": record.get("docket_number"),
                "citations": citations,
                # Records imported from another source (see us_reports_pdf.py) say so themselves.
                "authority_identity": record.get("authority_identity") or "CAP_CANONICAL"
            },
        ))

    return documents, known_cases


def _read_chunk_volume(
    volume: str,
    chunks_dir: str = CAP_CHUNKS_DIR,
    raw_sha256: Optional[str] = None,
) -> Tuple[ArtifactCheck, List[CorpusDocument]]:
    """Validate one chunk cache and return its parsed chunks. The check's payload omits the chunk list."""
    volume = str(volume)
    path = chunk_volume_path(volume, chunks_dir)
    if not os.path.exists(path):
        return ArtifactCheck(NOT_STARTED, "no chunk cache file"), []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as e:
        return ArtifactCheck(FAILED, f"chunk cache is unreadable or corrupt: {e}"), []
    if not isinstance(data, dict):
        return ArtifactCheck(FAILED, "chunk cache is not a JSON object"), []

    raw_chunks = data.pop("chunks", None)
    if str(data.get("volume")) != volume:
        return ArtifactCheck(FAILED, f"chunk cache is for volume {data.get('volume')!r}, expected {volume}", data), []
    if data.get("schema_version") != CHUNK_SCHEMA_VERSION or data.get("status") != CHUNK_COMPLETE:
        return ArtifactCheck(FAILED, "chunk cache has no explicit completion status (older or partial cache)", data), []
    if not isinstance(raw_chunks, list) or data.get("chunk_count") != len(raw_chunks):
        return ArtifactCheck(FAILED, f"chunk_count={data.get('chunk_count')} does not match the stored chunks", data), []
    if raw_sha256 is not None and data.get("raw_sha256") != raw_sha256:
        return ArtifactCheck(FAILED, "chunk cache is stale: the raw volume changed after it was chunked", data), []
    try:
        chunks = [CorpusDocument.from_dict(row) for row in raw_chunks]
    except Exception as e:
        return ArtifactCheck(FAILED, f"chunk structure cannot be parsed: {e}", data), []
    if any(not isinstance(c.doc_id, str) or not c.doc_id or not isinstance(c.text, str) or not c.text for c in chunks):
        return ArtifactCheck(FAILED, "a stored chunk has no doc_id or no text", data), []
    return ArtifactCheck(COMPLETE, "", data), chunks


def inspect_chunk_volume(volume: str, chunks_dir: str = CAP_CHUNKS_DIR, raw_sha256: Optional[str] = None) -> ArtifactCheck:
    """Validate one chunk cache. Pass raw_sha256 to also require that it was built from that exact raw file."""
    return _read_chunk_volume(volume, chunks_dir, raw_sha256)[0]


def load_chunk_volume(volume: str, chunks_dir: str = CAP_CHUNKS_DIR) -> Tuple[List[CorpusDocument], Set[str]]:
    """Return the chunks and known cases of a verified chunk cache. Never downloads or chunks."""
    check, chunks = _read_chunk_volume(volume, chunks_dir)
    if not check.ok:
        raise ChunksNotAvailableError({str(volume): check.reason})
    known_cases = set(check.payload.get("known_cases") or [])
    if not known_cases:
        known_cases = extract_known_cases_from_chunks(chunks)
    return chunks, known_cases


def chunk_cap_volume(
    volume: str,
    raw_dir: str = CAP_RAW_DIR,
    chunks_dir: str = CAP_CHUNKS_DIR,
    force: bool = False,
) -> dict:
    """Chunk one verified raw volume into the chunk cache. Raw JSON to chunks only.

    This never touches the network. If the raw volume is missing or not verified it raises
    VolumeNotAvailableError; the download stage is the only thing allowed to fetch it.
    A verified chunk cache built from the same raw file is left alone unless force=True.

    Returns the chunk cache header (every field except the chunks themselves).
    """
    volume = str(volume)
    raw = inspect_raw_volume(volume, raw_dir)
    if not raw.ok:
        raise VolumeNotAvailableError(volume, raw.state, raw.reason)

    if not force:
        existing = inspect_chunk_volume(volume, chunks_dir, raw_sha256=raw.sha256)
        if existing.ok:
            return {**existing.payload, "skipped": True}

    documents, known_cases = _documents_from_cases(raw.payload["cases"], default_volume=volume)
    chunks = chunk_corpus(documents)

    header = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "volume": volume,
        "status": CHUNK_COMPLETE,
        "raw_sha256": raw.sha256,
        "raw_cases_count": len(raw.payload["cases"]),
        "documents_count": len(documents),
        "chunk_count": len(chunks),
        "chunked_at": utc_now(),
    }
    os.makedirs(chunks_dir, exist_ok=True)
    atomic_write_json(chunk_volume_path(volume, chunks_dir), {
        **header,
        "chunks": [c.to_dict() for c in chunks],
        "known_cases": sorted(list(known_cases)),
    })
    return header


def run_chunk_stage(
    volumes: List[str],
    raw_dir: str = CAP_RAW_DIR,
    chunks_dir: str = CAP_CHUNKS_DIR,
    checkpoint_path: str = CAP_CHUNK_CHECKPOINT_PATH,
    checkpoint_interval: int = CAP_CHECKPOINT_INTERVAL,
    selection: str = "custom",
    force: bool = False,
) -> dict:
    """STAGE 2: chunk verified raw volumes. No network, no embedding, no ChromaDB.

    Resumes by default: a volume is skipped only if its chunk cache is verified COMPLETE and was
    built from the raw file currently on disk. A volume whose raw cache is missing or unverified
    is reported as FAILED with "Run the download stage" and is never downloaded from here.
    """
    os.makedirs(chunks_dir, exist_ok=True)
    ckpt = StageCheckpoint(checkpoint_path, "chunk", volumes, checkpoint_interval, selection)

    def inspect(vol: str) -> Tuple[str, dict]:
        raw_path = raw_volume_path(vol, raw_dir)
        if not os.path.exists(raw_path):
            return NOT_STARTED, {"error": "no raw cache file"}
        check = inspect_chunk_volume(vol, chunks_dir, raw_sha256=_file_sha256(raw_path))
        payload = check.payload or {}
        info = {key: payload.get(key) for key in ("raw_cases_count", "documents_count", "chunk_count", "raw_sha256") if key in payload}
        if not check.ok:
            info["error"] = check.reason
        return check.state, info

    def process(vol: str, live: _LiveProgress) -> Tuple[str, dict]:
        try:
            chunk_cap_volume(vol, raw_dir=raw_dir, chunks_dir=chunks_dir, force=force)
        except VolumeNotAvailableError as e:
            return (UNAVAILABLE if e.state == UNAVAILABLE else FAILED), {"error": str(e)}
        # The volume is COMPLETE only if what is now on disk validates.
        return inspect(vol)

    def describe(state: str, info: dict) -> str:
        if state == COMPLETE:
            return f"{info.get('chunk_count')} chunks from {info.get('documents_count')} documents"
        return str(info.get("error"))

    return _run_stage(
        ckpt,
        "CAP STAGE 2: CHUNK ONLY",
        [f"Raw cache:          {raw_dir}", f"Chunk cache:        {chunks_dir}", "Network:            never used by this stage"],
        inspect, process, describe, force=force,
    )


# ══════════════════════════════════════════════════════
# STAGE 3 INPUT — verified chunk caches -> corpus (no network, no chunking)
# ══════════════════════════════════════════════════════
def load_cap_chunks(
    volumes: List[str],
    chunks_dir: str = CAP_CHUNKS_DIR,
    raw_dir: str = CAP_RAW_DIR,
    allow_incomplete: bool = False,
) -> Tuple[List[CorpusDocument], Set[str], dict]:
    """Load the verified chunk caches of the target volumes for the index build.

    Nothing is downloaded and nothing is chunked here. Volumes that CAP does not publish
    are skipped and reported. Any other volume without a verified chunk cache raises
    ChunksNotAvailableError, unless allow_incomplete=True.

    Returns (chunks, known_cases, report) where report lists the loaded, unavailable and missing volumes.
    """
    all_chunks: List[CorpusDocument] = []
    all_known_cases: Set[str] = set()
    loaded: List[str] = []
    unavailable: List[str] = []
    missing: Dict[str, str] = {}

    print(f"\n  -> Loading verified chunk caches for {len(volumes)} volumes from {chunks_dir}...", flush=True)
    for i, vol in enumerate(volumes, 1):
        vol = str(vol)
        check, chunks = _read_chunk_volume(vol, chunks_dir)
        if check.ok:
            known_cases = set(check.payload.get("known_cases") or []) or extract_known_cases_from_chunks(chunks)
            all_chunks.extend(chunks)
            all_known_cases.update(known_cases)
            loaded.append(vol)
        elif inspect_raw_volume(vol, raw_dir).state == UNAVAILABLE:
            unavailable.append(vol)
        else:
            missing[vol] = check.reason
        if i % 50 == 0 or i == len(volumes):
            print(f"     Loaded [{i}/{len(volumes)}] volumes | Total chunks: {len(all_chunks):,}", flush=True)

    if unavailable:
        print(f"  [NOTICE] {len(unavailable)} target volume(s) are not published by CAP and are not in the corpus: {_format_volume_ranges(unavailable)}")
    if missing:
        if not allow_incomplete:
            raise ChunksNotAvailableError(missing)
        print(f"  [WARNING] --allow-incomplete: {len(missing)} target volume(s) have no verified chunk cache and are NOT indexed: {_format_volume_ranges(list(missing))}")

    return all_chunks, all_known_cases, {"loaded": loaded, "unavailable": unavailable, "missing": missing}


def corpus_fingerprint(chunks: List[CorpusDocument]) -> str:
    """Identity of a corpus: its ordered chunks. Indexes built for one corpus must not be reused for another.

    The chunk id already contains a digest of the chunk text, so hashing the id plus the
    remaining fields covers the whole chunk without re-hashing the text.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        fields = [chunk.doc_id, chunk.source, chunk.case_name, chunk.year, chunk.court,
                  chunk.volume, chunk.reporter, chunk.first_page, chunk.metadata]
        digest.update(json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_corpus_json(chunks: List[CorpusDocument], path: str = CORPUS_JSON_PATH) -> None:
    """Write the corpus as one JSON array, streamed chunk by chunk and replaced atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write("[")
        for i, chunk in enumerate(chunks):
            if i:
                handle.write(", ")
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False))
        handle.write("]")
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(tmp_path, path)


def build_cap_corpus_in_portions(
    volumes: List[str],
    raw_dir: str = CAP_RAW_DIR,
    chunks_dir: str = CAP_CHUNKS_DIR,
    portion_size: int = 25,
    max_workers: int = 10,
    volume_tag: str = "multi",
) -> Tuple[List[CorpusDocument], Set[str]]:
    """Assemble the corpus for the target volumes from verified chunk caches.

    Kept for callers of the old name. It no longer downloads or chunks anything:
    portion_size and max_workers are ignored, and a volume without a verified chunk
    cache raises ChunksNotAvailableError. Use run_download_stage and run_chunk_stage
    to prepare the data first.
    """
    chunks, known_cases, _ = load_cap_chunks(volumes, chunks_dir=chunks_dir, raw_dir=raw_dir)
    return chunks, known_cases


def build_cap_corpus(cases: Iterable[dict], volume_tag: str = "multi") -> Tuple[List[CorpusDocument], Set[str]]:
    """Convert raw CAP case bodies into retrieval chunks and extract known cases."""
    # If cases is empty but corpus JSON exists, load existing corpus directly
    if not cases and os.path.exists(CORPUS_JSON_PATH):
        try:
            with open(CORPUS_JSON_PATH, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            if isinstance(raw_list, list) and len(raw_list) > 0:
                chunks = [CorpusDocument.from_dict(d) if isinstance(d, dict) else d for d in raw_list]
                known_cases = extract_known_cases_from_chunks(chunks)
                logger.info("Loaded %d chunks from existing corpus %s", len(chunks), CORPUS_JSON_PATH)
                return chunks, known_cases
        except Exception as e:
            logger.warning("Failed to load existing %s: %s", CORPUS_JSON_PATH, e)

    documents, known_cases = _documents_from_cases(cases)

    chunks = chunk_corpus(documents)
    write_corpus_json(chunks, CORPUS_JSON_PATH)

    logger.info("CAP Corpus: %d cases -> %d chunks saved to %s", len(documents), len(chunks), CORPUS_JSON_PATH)
    return chunks, known_cases


def write_cap_manifest(chunks: list[CorpusDocument], volume_tag: str = "multi", extra: Optional[dict] = None) -> None:
    corpus_hash = _file_sha256(CORPUS_JSON_PATH) if os.path.exists(CORPUS_JSON_PATH) else ""
    manifest = {
        "source": "Caselaw Access Project (Harvard)",
        "license": "CC0-1.0",
        "volumes": volume_tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": len(chunks),
        "corpus_sha256": corpus_hash,
        "is_primary": True,
        **(extra or {}),
    }
    atomic_write_json(CORPUS_MANIFEST_PATH, manifest, indent=2)
    logger.info("Manifest written to: %s", CORPUS_MANIFEST_PATH)
