# LexAgent v3.0 | cap_checkpoint.py
"""Durable stage checkpoints for the staged CAP pipeline (download stage, chunk stage).

Two things are tracked, and they are not the same thing:
  - Per-volume state: every target volume is NOT_STARTED, IN_PROGRESS, COMPLETE, FAILED
    or UNAVAILABLE, and the ledger is flushed to disk on every transition.
  - Batch checkpoints: the target list is cut into batches of `interval` volumes
    (1-64, 65-128, ...). A batch is committed once every volume in it has been processed.

The ledger is a record, not the source of truth. A volume only counts as COMPLETE when its
cache file passes validation, so each stage re-validates the files on disk when it starts.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1

NOT_STARTED = "NOT_STARTED"
IN_PROGRESS = "IN_PROGRESS"
COMPLETE = "COMPLETE"
FAILED = "FAILED"
UNAVAILABLE = "UNAVAILABLE"  # the source does not publish this volume (HTTP 404); never counted as COMPLETE
VOLUME_STATES = (NOT_STARTED, IN_PROGRESS, COMPLETE, FAILED, UNAVAILABLE)

BATCH_PENDING = "PENDING"        # at least one volume not yet processed
BATCH_COMPLETE = "COMPLETE"      # every volume COMPLETE
BATCH_INCOMPLETE = "INCOMPLETE"  # every volume processed, some FAILED or UNAVAILABLE


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_replace(tmp_path: str, path: str) -> None:
    """os.replace with a short retry: on Windows the target can be briefly locked by a reader."""
    for attempt in range(10):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def atomic_write_json(path: str, payload: Any, indent: Optional[int] = None) -> None:
    """Write JSON so a crash mid-write can never destroy the previous valid file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(tmp_path, path)


class StageCheckpoint:
    """Per-volume state ledger plus batch checkpoints for one pipeline stage."""

    def __init__(self, path: str, stage: str, volumes: List[str], interval: int, selection: str = "custom"):
        if interval <= 0:
            raise ValueError("checkpoint interval must be a positive number of volumes")
        self.path = path
        self.stage = stage
        self.volumes = [str(v) for v in volumes]
        self.interval = interval
        self.selection = selection
        self.created_at = utc_now()
        self.run_state = "RUNNING"
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.batch_commits: Dict[str, str] = {}
        self._batch_of = {vol: idx // interval + 1 for idx, vol in enumerate(self.volumes)}
        self._load()

    def _load(self) -> None:
        """Merge a previous ledger. An unreadable one is set aside, never treated as 'start over'."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or data.get("stage") != self.stage or not isinstance(data.get("volumes"), dict):
                raise ValueError("not a checkpoint for this stage")
        except Exception as e:
            aside = f"{self.path}.unreadable-{int(time.time())}"
            logger.warning("Checkpoint %s is unreadable (%s). Moved to %s; state will be rebuilt from the cache files.", self.path, e, aside)
            try:
                os.replace(self.path, aside)
            except OSError:
                pass
            return

        self.created_at = data.get("created_at") or self.created_at
        # State for volumes outside the current selection is kept, so a subset run never erases the full ledger.
        self.entries = {str(vol): entry for vol, entry in data["volumes"].items() if isinstance(entry, dict)}
        same_batches = data.get("checkpoint_interval") == self.interval and data.get("target_volumes") == self.volumes
        if same_batches and isinstance(data.get("batch_commits"), dict):
            self.batch_commits = dict(data["batch_commits"])

    def state(self, volume: str) -> str:
        return self.entries.get(str(volume), {}).get("status", NOT_STARTED)

    def set_state(self, volume: str, status: str, save: bool = True, **info: Any) -> None:
        if status not in VOLUME_STATES:
            raise ValueError(f"unknown volume state: {status}")
        volume = str(volume)
        attempts = self.entries.get(volume, {}).get("attempts", 0)
        if status == IN_PROGRESS:
            attempts += 1
        self.entries[volume] = {"status": status, "attempts": attempts, "updated_at": utc_now(), **info}
        if save:
            self.save()

    def batch_index(self, volume: str) -> int:
        return self._batch_of[str(volume)]

    def batch_count(self) -> int:
        return (len(self.volumes) + self.interval - 1) // self.interval

    def batch(self, index: int) -> Dict[str, Any]:
        members = self.volumes[(index - 1) * self.interval:index * self.interval]
        states = [self.state(vol) for vol in members]
        if all(state == COMPLETE for state in states):
            status = BATCH_COMPLETE
        elif any(state in (NOT_STARTED, IN_PROGRESS) for state in states):
            status = BATCH_PENDING
        else:
            status = BATCH_INCOMPLETE
        return {
            "index": index,
            "first_volume": members[0],
            "last_volume": members[-1],
            "size": len(members),
            "status": status,
            "complete": states.count(COMPLETE),
            "failed_volumes": [vol for vol, state in zip(members, states) if state == FAILED],
            "unavailable_volumes": [vol for vol, state in zip(members, states) if state == UNAVAILABLE],
            "checkpointed_at": self.batch_commits.get(str(index)),
        }

    def commit_batch(self, index: int) -> Dict[str, Any]:
        """Record the batch checkpoint once every volume of the batch has been processed."""
        self.batch_commits.setdefault(str(index), utc_now())
        self.save()
        return self.batch(index)

    def summary(self) -> Dict[str, Any]:
        by_state: Dict[str, List[str]] = {state: [] for state in VOLUME_STATES}
        for vol in self.volumes:
            by_state[self.state(vol)].append(vol)
        completed = by_state[COMPLETE]
        return {
            "stage": self.stage,
            "target_volume_count": len(self.volumes),
            "counts": {state: len(vols) for state, vols in by_state.items()},
            "completed_volumes": completed,
            "failed_volumes": by_state[FAILED],
            "unavailable_volumes": by_state[UNAVAILABLE],
            "in_progress_volumes": by_state[IN_PROGRESS],
            "not_started_volumes": by_state[NOT_STARTED],
            "last_completed_volume": completed[-1] if completed else None,
            "all_target_volumes_complete": len(completed) == len(self.volumes),
            # Everything that is not verified COMPLETE, with the reason, in one place.
            "not_completed": [
                {"volume": vol, "status": self.state(vol), "reason": self.entries.get(vol, {}).get("error")}
                for vol in self.volumes if self.state(vol) != COMPLETE
            ],
        }

    def save(self, run_state: Optional[str] = None) -> None:
        if run_state:
            self.run_state = run_state
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": self.stage,
            "dataset": {
                "source": "Caselaw Access Project (static.case.law)",
                "reporter": "us",
                "selection": self.selection,
                "first_volume": self.volumes[0] if self.volumes else None,
                "last_volume": self.volumes[-1] if self.volumes else None,
            },
            "checkpoint_interval": self.interval,
            "created_at": self.created_at,
            "updated_at": utc_now(),
            "run_state": self.run_state,
            **self.summary(),
            "batches": [self.batch(index) for index in range(1, self.batch_count() + 1)],
            "batch_commits": self.batch_commits,
            "target_volumes": self.volumes,
            "volumes": self.entries,
        }
        atomic_write_json(self.path, payload, indent=1)
