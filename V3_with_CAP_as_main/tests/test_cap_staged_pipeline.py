# LexAgent v3.0 | test_cap_staged_pipeline.py
"""Local tests for the staged CAP pipeline (download -> chunk -> index).

Everything runs against a fake CAP server and temporary directories: no real
network access and no GPU. Run from the V3_with_CAP_as_main directory:

    python -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Point every configured path at a throwaway directory before config is imported.
os.environ["LEXAGENT_DATA_DIR"] = tempfile.mkdtemp(prefix="lexagent_test_data_")
os.environ["LEXAGENT_MOCK_LLM"] = "1"
logging.disable(logging.INFO)  # keep the test output to warnings and errors

import config  # noqa: E402

# If another test module imported config first, config points at that module's throwaway directory,
# not the one made above. These tests empty the data directory before each test, so they must work
# in the directory config really uses, and must never do so in a real one.
TEST_DATA_DIR = os.path.normpath(config.DRIVE_BASE)
if not os.path.basename(TEST_DATA_DIR).startswith("lexagent_test_data_"):
    raise RuntimeError(
        f"config points at {TEST_DATA_DIR}, which is not a throwaway test directory. "
        "These tests delete everything in the data directory; refusing to run."
    )
from src.data import cap_ingest  # noqa: E402
from src.data.cap_checkpoint import (  # noqa: E402
    COMPLETE,
    FAILED,
    IN_PROGRESS,
    NOT_STARTED,
    UNAVAILABLE,
    StageCheckpoint,
    atomic_write_json,
)
from src.data.cap_ingest import (  # noqa: E402
    CapNotFound,
    ChunksNotAvailableError,
    VolumeNotAvailableError,
    build_cap_corpus_in_portions,
    chunk_cap_volume,
    chunk_volume_path,
    download_cap_volume,
    get_downloaded_volumes,
    inspect_chunk_volume,
    inspect_raw_volume,
    load_cap_chunks,
    load_raw_volume,
    parse_volume_selection,
    raw_volume_path,
    run_chunk_stage,
    run_download_stage,
)


def tearDownModule():
    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class FakeCAP:
    """Stands in for cap_ingest._http_get_json: serves volumes held in memory and records every request."""

    def __init__(self):
        self.metadata = {}
        self.bodies = {}
        self.fail_urls = set()
        self.interrupt_urls = set()
        self.calls = []
        self._lock = threading.Lock()

    def metadata_url(self, volume):
        return f"{config.CAP_BASE_URL}/us/{volume}/CasesMetadata.json"

    def case_url(self, volume, index):
        return f"{config.CAP_BASE_URL}/us/{volume}/cases/{index:04d}-01.json"

    def add_volume(self, volume, n_cases=3, empty=(), words=600):
        volume = str(volume)
        records = []
        for index in range(1, n_cases + 1):
            record = {
                "id": int(volume) * 1000 + index,
                "name_abbreviation": f"Party {volume}-{index} v. State",
                "decision_date": "1950-06-01",
                "first_page": str(index),
                "file_name": f"{index:04d}-01",
                "citations": [{"type": "official", "cite": f"{volume} U.S. {index}"}],
                "court": {"name": "Supreme Court of the United States"},
            }
            records.append(record)
            text = "" if index in empty else " ".join(f"word{volume}x{index}x{n}" for n in range(words))
            self.bodies[self.case_url(volume, index)] = {
                **record,
                "casebody": {"head_matter": "", "opinions": [{"type": "majority", "text": text}]},
            }
        self.metadata[volume] = records

    def volume_calls(self, volume):
        prefix = f"{config.CAP_BASE_URL}/us/{volume}/"
        return [url for url in self.calls if url.startswith(prefix)]

    def __call__(self, url, timeout=45):
        with self._lock:
            self.calls.append(url)
        if url in self.interrupt_urls:
            raise KeyboardInterrupt()
        if url in self.fail_urls:
            raise RuntimeError("simulated network failure")
        match = re.match(rf"{re.escape(config.CAP_BASE_URL)}/us/(\d+)/CasesMetadata\.json$", url)
        if match:
            if match.group(1) not in self.metadata:
                raise CapNotFound(url)
            return [dict(record) for record in self.metadata[match.group(1)]]
        if url in self.bodies:
            return dict(self.bodies[url])
        raise CapNotFound(url)


class PipelineTestCase(unittest.TestCase):
    """Gives each test its own raw/chunk directories and a fake CAP server."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lexagent_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.raw_dir = os.path.join(self.tmp, "raw")
        self.chunks_dir = os.path.join(self.tmp, "chunks")
        self.download_ckpt = os.path.join(self.tmp, "cap_download_checkpoint.json")
        self.chunk_ckpt = os.path.join(self.tmp, "cap_chunk_checkpoint.json")
        self.cap = FakeCAP()
        patcher = mock.patch.object(cap_ingest, "_http_get_json", self.cap)
        patcher.start()
        self.addCleanup(patcher.stop)

    def download(self, volumes, interval=64, **kwargs):
        with quiet():
            return run_download_stage(
                [str(v) for v in volumes], raw_dir=self.raw_dir, checkpoint_path=self.download_ckpt,
                checkpoint_interval=interval, max_workers=4, **kwargs,
            )

    def chunk(self, volumes, interval=64, **kwargs):
        with quiet():
            return run_chunk_stage(
                [str(v) for v in volumes], raw_dir=self.raw_dir, chunks_dir=self.chunks_dir,
                checkpoint_path=self.chunk_ckpt, checkpoint_interval=interval, **kwargs,
            )


class VolumeSelectionTests(unittest.TestCase):
    def test_all_is_exactly_volumes_1_to_600(self):
        volumes = parse_volume_selection("all")
        self.assertEqual(len(volumes), 600)
        self.assertEqual(volumes, [str(v) for v in range(1, 601)])
        self.assertEqual((volumes[0], volumes[-1]), ("1", "600"))
        self.assertEqual(len(set(volumes)), 600)

    def test_other_selections(self):
        self.assertEqual(parse_volume_selection("400-403"), ["400", "401", "402", "403"])
        self.assertEqual(parse_volume_selection("410,347"), ["410", "347"])
        self.assertEqual(parse_volume_selection("410"), ["410"])
        self.assertEqual(len(parse_volume_selection("landmark")), 27)


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lexagent_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.path = os.path.join(self.tmp, "nested", "state.json")

    def test_write_creates_directories_and_leaves_no_temp_file(self):
        atomic_write_json(self.path, {"a": 1})
        self.assertEqual(read_json(self.path), {"a": 1})
        self.assertEqual(os.listdir(os.path.dirname(self.path)), ["state.json"])

    def test_failed_write_keeps_the_previous_file(self):
        atomic_write_json(self.path, {"generation": 1})
        with self.assertRaises(TypeError):
            atomic_write_json(self.path, {"generation": 2, "bad": object()})
        self.assertEqual(read_json(self.path), {"generation": 1})
        atomic_write_json(self.path, {"generation": 3})
        self.assertEqual(read_json(self.path), {"generation": 3})

    def test_crash_before_replace_keeps_the_previous_file(self):
        atomic_write_json(self.path, {"generation": 1})
        with mock.patch("src.data.cap_checkpoint.os.replace", side_effect=OSError("power loss")):
            with self.assertRaises(OSError):
                atomic_write_json(self.path, {"generation": 2})
        self.assertEqual(read_json(self.path), {"generation": 1})


class StageCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lexagent_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.path = os.path.join(self.tmp, "cap_download_checkpoint.json")
        self.all_volumes = parse_volume_selection("all")

    def test_600_volumes_make_ten_batches_of_64_with_a_final_batch_of_24(self):
        ckpt = StageCheckpoint(self.path, "download", self.all_volumes, 64, "all")
        self.assertEqual(ckpt.batch_count(), 10)
        batches = [ckpt.batch(index) for index in range(1, 11)]
        self.assertEqual([(b["first_volume"], b["last_volume"]) for b in batches], [
            ("1", "64"), ("65", "128"), ("129", "192"), ("193", "256"), ("257", "320"),
            ("321", "384"), ("385", "448"), ("449", "512"), ("513", "576"), ("577", "600"),
        ])
        self.assertEqual([b["size"] for b in batches], [64] * 9 + [24])
        self.assertEqual(ckpt.batch_index("64"), 1)
        self.assertEqual(ckpt.batch_index("65"), 2)
        self.assertEqual(ckpt.batch_index("600"), 10)

    def test_round_trip_keeps_every_volume_state(self):
        ckpt = StageCheckpoint(self.path, "download", self.all_volumes, 64, "all")
        for vol in range(1, 97):
            ckpt.set_state(str(vol), COMPLETE, save=False, cases_count=10)
        ckpt.set_state("97", FAILED, save=False, error="3 of 10 case downloads failed")
        ckpt.set_state("98", COMPLETE, save=False, cases_count=10)
        ckpt.set_state("99", IN_PROGRESS, save=False)
        ckpt.commit_batch(1)

        data = read_json(self.path)
        self.assertEqual(data["stage"], "download")
        self.assertEqual(data["target_volume_count"], 600)
        self.assertEqual(data["checkpoint_interval"], 64)
        self.assertEqual(data["dataset"]["first_volume"], "1")
        self.assertEqual(data["dataset"]["last_volume"], "600")
        self.assertEqual(data["counts"], {NOT_STARTED: 501, IN_PROGRESS: 1, COMPLETE: 97, FAILED: 1, UNAVAILABLE: 0})
        # A failure at 97 with a success at 98 is representable: no single "last volume" number.
        self.assertEqual(data["failed_volumes"], ["97"])
        self.assertIn("98", data["completed_volumes"])
        self.assertNotIn("97", data["completed_volumes"])
        self.assertEqual(data["last_completed_volume"], "98")
        self.assertFalse(data["all_target_volumes_complete"])
        self.assertEqual(data["batches"][0]["status"], "COMPLETE")
        self.assertIsNotNone(data["batches"][0]["checkpointed_at"])
        self.assertEqual(data["batches"][1]["status"], "PENDING")
        self.assertIn("updated_at", data)
        self.assertEqual(data["schema_version"], 1)

        reloaded = StageCheckpoint(self.path, "download", self.all_volumes, 64, "all")
        self.assertEqual(reloaded.state("1"), COMPLETE)
        self.assertEqual(reloaded.state("97"), FAILED)
        self.assertEqual(reloaded.state("98"), COMPLETE)
        self.assertEqual(reloaded.state("99"), IN_PROGRESS)
        self.assertEqual(reloaded.state("100"), NOT_STARTED)
        self.assertEqual(reloaded.entries["97"]["error"], "3 of 10 case downloads failed")

    def test_unreadable_checkpoint_is_set_aside_not_fatal(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write('{"stage": "download", "volumes": {"1": {"status": "COMPL')
        with quiet():
            ckpt = StageCheckpoint(self.path, "download", ["1", "2"], 64)
        self.assertEqual(ckpt.state("1"), NOT_STARTED)
        self.assertTrue(any(".unreadable-" in name for name in os.listdir(self.tmp)))

    def test_subset_run_does_not_erase_other_volumes(self):
        ckpt = StageCheckpoint(self.path, "download", self.all_volumes, 64, "all")
        ckpt.set_state("500", COMPLETE)
        subset = StageCheckpoint(self.path, "download", ["1", "2"], 64, "1-2")
        subset.set_state("1", COMPLETE)
        again = StageCheckpoint(self.path, "download", self.all_volumes, 64, "all")
        self.assertEqual(again.state("500"), COMPLETE)
        self.assertEqual(again.state("1"), COMPLETE)

    def test_rejects_unknown_state_and_bad_interval(self):
        ckpt = StageCheckpoint(self.path, "download", ["1"], 64)
        with self.assertRaises(ValueError):
            ckpt.set_state("1", "DONE")
        with self.assertRaises(ValueError):
            StageCheckpoint(self.path, "download", ["1"], 0)


class RawValidationTests(PipelineTestCase):
    def test_complete_volume_is_detected(self):
        self.cap.add_volume(5, n_cases=4, empty=(2,))
        header = download_cap_volume("5", raw_dir=self.raw_dir, max_workers=4)
        self.assertEqual(header["status"], "complete")
        self.assertEqual((header["metadata_count"], header["cases_count"], header["empty_text_count"]), (4, 3, 1))
        check = inspect_raw_volume("5", self.raw_dir)
        self.assertTrue(check.ok)
        self.assertEqual(len(check.sha256), 64)
        self.assertEqual([case["id"] for case in load_raw_volume("5", self.raw_dir)], [5001, 5003, 5004])

    def test_file_existing_is_not_success(self):
        os.makedirs(self.raw_dir)
        cases = [{"id": 1, "casebody": {"opinions": [{"text": "x" * 500}]}}]
        samples = {
            "old format without status": json.dumps({"volume": "7", "cases": cases}),
            "old not_found sentinel": json.dumps({"volume": "7", "cases": [], "status": "not_found"}),
            "truncated json": '{"schema_version": 2, "volume": "7", "status": "complete", "cases": [{"id": 1',
            "empty file": "",
            "not an object": "[1, 2, 3]",
            "explicitly failed": json.dumps({"schema_version": 2, "volume": "7", "status": "failed", "cases": cases, "error": "boom"}),
            "wrong volume": json.dumps({"schema_version": 2, "volume": "8", "status": "complete", "cases": cases,
                                        "metadata_count": 1, "cases_count": 1, "empty_text_count": 0, "failed_count": 0, "failed_cases": []}),
            "claims complete but a case is missing": json.dumps({"schema_version": 2, "volume": "7", "status": "complete", "cases": cases,
                                                                 "metadata_count": 2, "cases_count": 1, "empty_text_count": 0, "failed_count": 0, "failed_cases": []}),
            "claims complete but a case failed": json.dumps({"schema_version": 2, "volume": "7", "status": "complete", "cases": cases,
                                                             "metadata_count": 1, "cases_count": 1, "empty_text_count": 0, "failed_count": 1, "failed_cases": [{"id": 2}]}),
            "claims complete without metadata": json.dumps({"schema_version": 2, "volume": "7", "status": "complete", "cases": [],
                                                            "metadata_count": 0, "cases_count": 0, "empty_text_count": 0, "failed_count": 0, "failed_cases": []}),
        }
        for label, content in samples.items():
            with self.subTest(label):
                with open(raw_volume_path("7", self.raw_dir), "w", encoding="utf-8") as handle:
                    handle.write(content)
                check = inspect_raw_volume("7", self.raw_dir)
                self.assertFalse(check.ok)
                self.assertEqual(check.state, FAILED)
                self.assertTrue(check.reason)
                self.assertEqual(get_downloaded_volumes(self.raw_dir), (set(), 0))
                with self.assertRaises(VolumeNotAvailableError):
                    load_raw_volume("7", self.raw_dir)

    def test_missing_file_is_not_started(self):
        self.assertEqual(inspect_raw_volume("7", self.raw_dir).state, NOT_STARTED)


class DownloadVolumeTests(PipelineTestCase):
    def test_partial_download_is_failed_then_retry_fetches_only_what_is_missing(self):
        self.cap.add_volume(9, n_cases=5)
        self.cap.fail_urls.add(self.cap.case_url(9, 3))
        header = download_cap_volume("9", raw_dir=self.raw_dir, max_workers=4)
        self.assertEqual(header["status"], "failed")
        self.assertEqual((header["failed_count"], header["cases_count"]), (1, 4))
        self.assertEqual(header["failed_cases"][0]["id"], 9003)
        self.assertEqual(read_json(raw_volume_path("9", self.raw_dir))["status"], "failed")
        self.assertEqual(inspect_raw_volume("9", self.raw_dir).state, FAILED)
        self.assertEqual(get_downloaded_volumes(self.raw_dir), (set(), 0))

        self.cap.fail_urls.clear()
        self.cap.calls.clear()
        header = download_cap_volume("9", raw_dir=self.raw_dir, max_workers=4)
        self.assertEqual(header["status"], "complete")
        self.assertEqual(sorted(self.cap.calls), sorted([self.cap.metadata_url(9), self.cap.case_url(9, 3)]))
        self.assertEqual([case["id"] for case in load_raw_volume("9", self.raw_dir)], [9001, 9002, 9003, 9004, 9005])

    def test_http_404_is_unavailable_and_network_error_is_failed(self):
        header = download_cap_volume("599", raw_dir=self.raw_dir)
        self.assertEqual(header["status"], "unavailable")
        self.assertEqual(inspect_raw_volume("599", self.raw_dir).state, UNAVAILABLE)

        self.cap.add_volume(11)
        self.cap.fail_urls.add(self.cap.metadata_url(11))
        header = download_cap_volume("11", raw_dir=self.raw_dir)
        self.assertEqual(header["status"], "failed")
        self.assertIn("metadata download failed", header["error"])
        self.assertEqual(inspect_raw_volume("11", self.raw_dir).state, FAILED)

    def test_verified_volume_is_not_downloaded_again(self):
        self.cap.add_volume(12)
        download_cap_volume("12", raw_dir=self.raw_dir)
        self.cap.calls.clear()
        header = download_cap_volume("12", raw_dir=self.raw_dir)
        self.assertTrue(header["skipped"])
        self.assertEqual(self.cap.calls, [])

    def test_failed_forced_refresh_keeps_the_verified_copy(self):
        self.cap.add_volume(13)
        download_cap_volume("13", raw_dir=self.raw_dir)
        self.cap.fail_urls.add(self.cap.metadata_url(13))
        header = download_cap_volume("13", raw_dir=self.raw_dir, force=True)
        self.assertTrue(header["kept_previous_complete"])
        self.assertTrue(inspect_raw_volume("13", self.raw_dir).ok)

    def test_cases_are_stored_in_metadata_order(self):
        self.cap.add_volume(14, n_cases=40)
        download_cap_volume("14", raw_dir=self.raw_dir, max_workers=8)
        self.assertEqual([case["id"] for case in load_raw_volume("14", self.raw_dir)], [14000 + i for i in range(1, 41)])

    def test_http_retry_wrapper(self):
        class Response:
            def __init__(self, status, payload=None):
                self.status_code = status
                self._payload = payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise cap_ingest.requests.HTTPError(f"HTTP {self.status_code}")

            def json(self):
                return self._payload

        class Session:
            def __init__(self, responses):
                self.responses = list(responses)
                self.count = 0

            def get(self, url, timeout=None):
                self.count += 1
                item = self.responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        self.addCleanup(mock.patch.stopall)
        mock.patch.stopall()  # use the real _http_get_json for this test
        with mock.patch.object(cap_ingest.time, "sleep") as sleep:
            session = Session([cap_ingest.requests.ConnectionError("reset"), Response(503), Response(200, {"ok": True})])
            with mock.patch.object(cap_ingest, "_session", return_value=session):
                self.assertEqual(cap_ingest._http_get_json("http://x/y.json"), {"ok": True})
            self.assertEqual(session.count, 3)
            self.assertEqual(sleep.call_count, 2)

            session = Session([Response(404)])
            with mock.patch.object(cap_ingest, "_session", return_value=session):
                with self.assertRaises(CapNotFound):
                    cap_ingest._http_get_json("http://x/y.json")
            self.assertEqual(session.count, 1)  # a 404 is an answer, not a transient error

            session = Session([Response(500)] * cap_ingest.HTTP_MAX_ATTEMPTS)
            with mock.patch.object(cap_ingest, "_session", return_value=session):
                with self.assertRaises(RuntimeError):
                    cap_ingest._http_get_json("http://x/y.json")
            self.assertEqual(session.count, cap_ingest.HTTP_MAX_ATTEMPTS)


class DownloadStageTests(PipelineTestCase):
    def test_failures_are_skipped_recorded_and_the_run_continues(self):
        for vol in (1, 2, 3, 4, 5):
            self.cap.add_volume(vol)
        self.cap.fail_urls.add(self.cap.case_url(2, 1))       # volume 2: one case fails
        self.cap.fail_urls.add(self.cap.metadata_url(4))      # volume 4: metadata fails
        summary = self.download([1, 2, 3, 4, 5, 6], interval=2)  # volume 6 does not exist (404)

        self.assertEqual(summary["completed_volumes"], ["1", "3", "5"])
        self.assertEqual(summary["failed_volumes"], ["2", "4"])
        self.assertEqual(summary["unavailable_volumes"], ["6"])
        self.assertFalse(summary["all_target_volumes_complete"])
        self.assertEqual(summary["last_completed_volume"], "5")
        self.assertEqual([entry["volume"] for entry in summary["not_completed"]], ["2", "4", "6"])
        self.assertTrue(all(entry["reason"] for entry in summary["not_completed"]))

        data = read_json(self.download_ckpt)
        self.assertEqual(data["run_state"], "FINISHED")
        self.assertEqual(data["volumes"]["2"]["status"], FAILED)
        self.assertEqual(data["volumes"]["6"]["status"], UNAVAILABLE)
        self.assertEqual([b["status"] for b in data["batches"]], ["INCOMPLETE", "INCOMPLETE", "INCOMPLETE"])
        self.assertTrue(all(b["checkpointed_at"] for b in data["batches"]))
        self.assertEqual(data["batches"][0]["failed_volumes"], ["2"])

    def test_second_run_skips_verified_volumes_and_retries_the_rest(self):
        for vol in (1, 2, 3):
            self.cap.add_volume(vol)
        self.cap.fail_urls.add(self.cap.case_url(2, 1))
        self.download([1, 2, 3])
        self.cap.fail_urls.clear()
        self.cap.calls.clear()

        summary = self.download([1, 2, 3])
        self.assertTrue(summary["all_target_volumes_complete"])
        self.assertEqual(self.cap.volume_calls(1), [])
        self.assertEqual(self.cap.volume_calls(3), [])
        self.assertEqual(sorted(self.cap.volume_calls(2)), sorted([self.cap.metadata_url(2), self.cap.case_url(2, 1)]))
        self.assertEqual(read_json(self.download_ckpt)["volumes"]["2"]["attempts"], 2)

        self.cap.calls.clear()
        self.assertTrue(self.download([1, 2, 3])["all_target_volumes_complete"])
        self.assertEqual(self.cap.calls, [])  # fully idempotent once everything is verified

    def test_crash_mid_volume_resumes_at_that_volume(self):
        for vol in (1, 2, 3, 4, 5):
            self.cap.add_volume(vol)
        self.cap.interrupt_urls.add(self.cap.case_url(3, 2))  # the process dies while downloading volume 3
        with self.assertRaises(KeyboardInterrupt):
            self.download([1, 2, 3, 4, 5], interval=2)

        data = read_json(self.download_ckpt)
        self.assertEqual(data["run_state"], "INTERRUPTED")
        self.assertEqual(data["completed_volumes"], ["1", "2"])
        self.assertEqual(data["in_progress_volumes"], ["3"])
        self.assertEqual(data["not_started_volumes"], ["4", "5"])
        self.assertFalse(os.path.exists(raw_volume_path("3", self.raw_dir)))  # nothing partial was written
        self.assertEqual(data["batches"][0]["status"], "COMPLETE")

        self.cap.interrupt_urls.clear()
        self.cap.calls.clear()
        summary = self.download([1, 2, 3, 4, 5], interval=2)
        self.assertTrue(summary["all_target_volumes_complete"])
        self.assertEqual(self.cap.volume_calls(1), [])
        self.assertEqual(self.cap.volume_calls(2), [])
        self.assertEqual(len(self.cap.volume_calls(3)), 4)  # metadata + 3 cases: volume 3 was not assumed done

    def test_corrupted_cache_is_detected_and_only_that_volume_is_redone(self):
        for vol in (1, 2, 3):
            self.cap.add_volume(vol)
        self.download([1, 2, 3])
        path = raw_volume_path("2", self.raw_dir)
        with open(path, "r+b") as handle:
            handle.truncate(os.path.getsize(path) // 2)
        self.assertEqual(inspect_raw_volume("2", self.raw_dir).state, FAILED)

        self.cap.calls.clear()
        with self.assertLogs("src.data.cap_ingest", level="WARNING"):
            summary = self.download([1, 2, 3])
        self.assertTrue(summary["all_target_volumes_complete"])
        self.assertEqual(self.cap.volume_calls(1), [])
        self.assertEqual(self.cap.volume_calls(3), [])
        self.assertEqual(len(self.cap.volume_calls(2)), 4)

    def test_ledger_claiming_complete_without_a_valid_file_is_not_trusted(self):
        for vol in (1, 2):
            self.cap.add_volume(vol)
        self.download([1, 2])
        os.remove(raw_volume_path("1", self.raw_dir))
        self.cap.calls.clear()
        with self.assertLogs("src.data.cap_ingest", level="WARNING"):
            self.download([1, 2])
        self.assertEqual(len(self.cap.volume_calls(1)), 4)
        self.assertEqual(self.cap.volume_calls(2), [])

    def test_lost_checkpoint_never_restarts_the_download(self):
        for vol in (1, 2, 3):
            self.cap.add_volume(vol)
        self.download([1, 2, 3])
        os.remove(self.download_ckpt)
        self.cap.calls.clear()
        summary = self.download([1, 2, 3])
        self.assertTrue(summary["all_target_volumes_complete"])
        self.assertEqual(self.cap.calls, [])
        self.assertEqual(read_json(self.download_ckpt)["completed_volumes"], ["1", "2", "3"])

    def test_explicit_redownload_refetches(self):
        self.cap.add_volume(1)
        self.download([1])
        self.cap.calls.clear()
        self.download([1], force=True)
        self.assertEqual(len(self.cap.volume_calls(1)), 4)

    def test_full_600_volume_target_with_64_volume_checkpoints(self):
        for vol in range(1, 131):  # the fake source publishes volumes 1-130 only
            self.cap.add_volume(vol, n_cases=1, words=120)
        self.cap.fail_urls.add(self.cap.case_url(97, 1))
        summary = self.download(parse_volume_selection("all"), interval=64, selection="all")

        self.assertEqual(summary["target_volume_count"], 600)
        self.assertEqual(summary["counts"][COMPLETE], 129)
        self.assertEqual(summary["failed_volumes"], ["97"])
        self.assertEqual(summary["counts"][UNAVAILABLE], 470)
        self.assertFalse(summary["all_target_volumes_complete"])

        data = read_json(self.download_ckpt)
        self.assertEqual(len(data["batches"]), 10)
        self.assertEqual(data["batches"][0]["status"], "COMPLETE")           # 1-64
        self.assertEqual(data["batches"][1]["status"], "INCOMPLETE")         # 65-128, volume 97 failed
        self.assertEqual(data["batches"][1]["failed_volumes"], ["97"])
        self.assertEqual((data["batches"][9]["first_volume"], data["batches"][9]["last_volume"], data["batches"][9]["size"]), ("577", "600", 24))
        self.assertTrue(all(batch["checkpointed_at"] for batch in data["batches"]))
        self.assertEqual(data["dataset"]["selection"], "all")
        self.assertEqual(len(data["volumes"]), 600)


class ChunkStageTests(PipelineTestCase):
    def setUp(self):
        super().setUp()
        for vol in (1, 2, 3):
            self.cap.add_volume(vol)
        self.download([1, 2, 3])
        # From here on any network use is a bug: the chunk stage must never download.
        self.cap.calls.clear()

    def tearDown(self):
        self.assertEqual(self.cap.calls, [], "the chunk stage touched the network")

    def test_chunks_verified_raw_volumes(self):
        summary = self.chunk([1, 2, 3], interval=2)
        self.assertTrue(summary["all_target_volumes_complete"])
        payload = read_json(chunk_volume_path("1", self.chunks_dir))
        self.assertEqual((payload["status"], payload["volume"], payload["documents_count"]), ("complete", "1", 3))
        self.assertEqual(payload["chunk_count"], len(payload["chunks"]))
        self.assertGreater(payload["chunk_count"], 3)  # 600-word cases split into overlapping chunks
        self.assertIn("1 U.S. 1", payload["known_cases"])
        # CAP case files carry no volume number: every chunk takes the volume it was downloaded from.
        self.assertEqual({chunk["volume"] for chunk in payload["chunks"]}, {"1"})
        self.assertEqual({chunk["metadata"]["volume"] for chunk in payload["chunks"]}, {"1"})
        self.assertEqual({chunk["volume"] for chunk in read_json(chunk_volume_path("3", self.chunks_dir))["chunks"]}, {"3"})
        data = read_json(self.chunk_ckpt)
        self.assertEqual(data["stage"], "chunk")
        self.assertEqual([b["status"] for b in data["batches"]], ["COMPLETE", "COMPLETE"])

    def test_missing_raw_volume_is_refused_not_downloaded(self):
        summary = self.chunk([1, 250])
        self.assertEqual(summary["completed_volumes"], ["1"])
        self.assertEqual(summary["failed_volumes"], ["250"])
        reason = summary["not_completed"][0]["reason"]
        self.assertIn("Volume 250 is not available/verified", reason)
        self.assertIn("Run the download stage", reason)
        self.assertFalse(os.path.exists(chunk_volume_path("250", self.chunks_dir)))
        with self.assertRaises(VolumeNotAvailableError):
            chunk_cap_volume("250", raw_dir=self.raw_dir, chunks_dir=self.chunks_dir)

    def test_failed_and_corrupt_raw_volumes_are_refused(self):
        with open(raw_volume_path("2", self.raw_dir), "r+b") as handle:
            handle.truncate(200)
        payload = read_json(raw_volume_path("3", self.raw_dir))
        payload.update(status="failed", failed_count=1, error="1 of 3 case downloads failed")
        atomic_write_json(raw_volume_path("3", self.raw_dir), payload)

        summary = self.chunk([1, 2, 3])
        self.assertEqual(summary["completed_volumes"], ["1"])
        self.assertEqual(summary["failed_volumes"], ["2", "3"])
        self.assertTrue(all("Run the download stage" in entry["reason"] for entry in summary["not_completed"]))

    def test_volume_not_published_by_cap_is_unavailable(self):
        with quiet():
            run_download_stage(["599"], raw_dir=self.raw_dir, checkpoint_path=self.download_ckpt, checkpoint_interval=64)
        self.cap.calls.clear()
        summary = self.chunk([1, 599])
        self.assertEqual(summary["unavailable_volumes"], ["599"])
        self.assertEqual(summary["failed_volumes"], [])

    def test_second_run_skips_verified_chunk_caches(self):
        self.chunk([1, 2, 3])
        stamps = {vol: os.path.getmtime(chunk_volume_path(vol, self.chunks_dir)) for vol in ("1", "2", "3")}
        with mock.patch.object(cap_ingest, "chunk_corpus", side_effect=AssertionError("re-chunked a verified volume")):
            summary = self.chunk([1, 2, 3])
        self.assertTrue(summary["all_target_volumes_complete"])
        self.assertEqual(stamps, {vol: os.path.getmtime(chunk_volume_path(vol, self.chunks_dir)) for vol in ("1", "2", "3")})

    def test_corrupt_missing_and_old_format_chunk_caches_are_rebuilt(self):
        self.chunk([1, 2, 3])
        with open(chunk_volume_path("1", self.chunks_dir), "r+b") as handle:
            handle.truncate(300)                                                   # corrupt
        os.remove(chunk_volume_path("2", self.chunks_dir))                        # missing output
        old = read_json(chunk_volume_path("3", self.chunks_dir))
        atomic_write_json(chunk_volume_path("3", self.chunks_dir), {"volume": "3", "chunks": old["chunks"], "known_cases": old["known_cases"]})
        for vol in ("1", "2", "3"):
            self.assertFalse(inspect_chunk_volume(vol, self.chunks_dir).ok)

        with self.assertLogs("src.data.cap_ingest", level="WARNING"):
            summary = self.chunk([1, 2, 3])
        self.assertTrue(summary["all_target_volumes_complete"])
        for vol in ("1", "2", "3"):
            self.assertTrue(inspect_chunk_volume(vol, self.chunks_dir).ok)

    def test_chunks_built_from_an_older_raw_file_are_stale(self):
        self.chunk([1])
        payload = read_json(raw_volume_path("1", self.raw_dir))
        payload["downloaded_at"] = "re-downloaded later"
        atomic_write_json(raw_volume_path("1", self.raw_dir), payload)
        raw_hash = inspect_raw_volume("1", self.raw_dir).sha256
        self.assertIn("stale", inspect_chunk_volume("1", self.chunks_dir, raw_sha256=raw_hash).reason)
        with self.assertLogs("src.data.cap_ingest", level="WARNING"):
            self.chunk([1])
        self.assertTrue(inspect_chunk_volume("1", self.chunks_dir, raw_sha256=raw_hash).ok)

    def test_crash_resume(self):
        real = cap_ingest.chunk_cap_volume

        def crash_on_volume_2(volume, **kwargs):
            if str(volume) == "2":
                raise KeyboardInterrupt()
            return real(volume, **kwargs)

        with mock.patch.object(cap_ingest, "chunk_cap_volume", side_effect=crash_on_volume_2):
            with self.assertRaises(KeyboardInterrupt):
                self.chunk([1, 2, 3])
        data = read_json(self.chunk_ckpt)
        self.assertEqual((data["completed_volumes"], data["in_progress_volumes"], data["not_started_volumes"]), (["1"], ["2"], ["3"]))

        stamp = os.path.getmtime(chunk_volume_path("1", self.chunks_dir))
        summary = self.chunk([1, 2, 3])
        self.assertTrue(summary["all_target_volumes_complete"])
        self.assertEqual(stamp, os.path.getmtime(chunk_volume_path("1", self.chunks_dir)))


class CorpusLoadingTests(PipelineTestCase):
    def setUp(self):
        super().setUp()
        for vol in (1, 2):
            self.cap.add_volume(vol)
        self.download([1, 2, 599])
        self.chunk([1, 2, 599])
        self.cap.calls.clear()

    def tearDown(self):
        self.assertEqual(self.cap.calls, [], "corpus loading touched the network")

    def test_loads_verified_chunks_and_skips_unpublished_volumes(self):
        with quiet():
            chunks, known_cases, report = load_cap_chunks(["1", "2", "599"], chunks_dir=self.chunks_dir, raw_dir=self.raw_dir)
        self.assertEqual((report["loaded"], report["unavailable"], report["missing"]), (["1", "2"], ["599"], {}))
        self.assertTrue(chunks)
        self.assertIn("2 U.S. 3", known_cases)
        self.assertEqual(len({chunk.doc_id for chunk in chunks}), len(chunks))

    def test_missing_chunks_stop_the_build_instead_of_downloading(self):
        with quiet(), self.assertRaises(ChunksNotAvailableError) as raised:
            load_cap_chunks(["1", "2", "3"], chunks_dir=self.chunks_dir, raw_dir=self.raw_dir)
        self.assertEqual(list(raised.exception.missing), ["3"])
        self.assertIn("--chunk-cap", str(raised.exception))
        with quiet(), self.assertRaises(ChunksNotAvailableError):
            build_cap_corpus_in_portions(["1", "3"], raw_dir=self.raw_dir, chunks_dir=self.chunks_dir)

    def test_allow_incomplete_indexes_what_is_ready(self):
        with quiet():
            chunks, _, report = load_cap_chunks(["1", "3"], chunks_dir=self.chunks_dir, raw_dir=self.raw_dir, allow_incomplete=True)
        self.assertEqual(report["loaded"], ["1"])
        self.assertEqual(list(report["missing"]), ["3"])
        self.assertTrue(chunks)


class CliTests(unittest.TestCase):
    def setUp(self):
        import master_run
        self.master_run = master_run

    def test_stage_flags_parse(self):
        parser = self.master_run.build_parser()
        args = parser.parse_args(["--download-cap", "--volumes", "all", "--workers", "16", "--checkpoint-interval", "64"])
        self.assertTrue(args.download_cap)
        self.assertFalse(args.chunk_cap or args.build_db)
        self.assertEqual((args.volumes, args.workers, args.checkpoint_interval), ("all", 16, 64))

        args = parser.parse_args(["--chunk-cap", "--volumes", "all"])
        self.assertTrue(args.chunk_cap)
        self.assertEqual(args.checkpoint_interval, 64)  # 64 is the default
        self.assertFalse(args.download_cap or args.build_db or args.overwrite or args.redownload or args.rechunk)

        args = parser.parse_args(["--build-db", "--volumes", "landmark", "--workers", "16", "--portion-size", "50", "--batch-size", "256"])
        self.assertTrue(args.build_db)
        self.assertEqual(args.batch_size, 256)

    def run_main(self, argv, download_summary=None, chunk_summary=None):
        clean = {"counts": {"FAILED": 0, "NOT_STARTED": 0, "IN_PROGRESS": 0, "COMPLETE": 1, "UNAVAILABLE": 0}}
        with mock.patch.object(self.master_run, "step_download_cap", return_value=download_summary or clean) as download, \
                mock.patch.object(self.master_run, "step_chunk_cap", return_value=chunk_summary or clean) as chunk, \
                mock.patch.object(self.master_run, "step_build_database") as build, \
                mock.patch.object(sys, "argv", ["master_run.py"] + argv), quiet():
            code = 0
            try:
                self.master_run.main()
            except SystemExit as stop:
                code = stop.code
        return download, chunk, build, code

    def test_each_stage_runs_alone(self):
        download, chunk, build, code = self.run_main(["--download-cap", "--volumes", "all", "--workers", "16", "--checkpoint-interval", "64"])
        download.assert_called_once_with(volumes_mode="all", workers=16, checkpoint_interval=64, redownload=False)
        self.assertFalse(chunk.called or build.called)
        self.assertEqual(code, 0)

        download, chunk, build, _ = self.run_main(["--chunk-cap", "--volumes", "all"])
        chunk.assert_called_once_with(volumes_mode="all", checkpoint_interval=64, rechunk=False)
        self.assertFalse(download.called or build.called)

        download, chunk, build, _ = self.run_main(["--build-db", "--volumes", "all"])
        self.assertTrue(build.called)
        self.assertFalse(download.called or chunk.called)
        self.assertFalse(build.call_args.kwargs["overwrite"])

    def test_failed_download_stops_later_stages_and_exits_nonzero(self):
        failed = {"counts": {"FAILED": 1, "NOT_STARTED": 0, "IN_PROGRESS": 0, "COMPLETE": 5, "UNAVAILABLE": 0}}
        download, chunk, build, code = self.run_main(["--download-cap", "--chunk-cap", "--build-db"], download_summary=failed)
        self.assertTrue(download.called)
        self.assertFalse(chunk.called or build.called)
        self.assertEqual(code, 1)

    def test_unpublished_volumes_alone_do_not_block_later_stages(self):
        unavailable_only = {"counts": {"FAILED": 0, "NOT_STARTED": 0, "IN_PROGRESS": 0, "COMPLETE": 572, "UNAVAILABLE": 28}}
        download, chunk, build, code = self.run_main(["--download-cap", "--chunk-cap", "--build-db"], download_summary=unavailable_only, chunk_summary=unavailable_only)
        self.assertTrue(download.called and chunk.called and build.called)
        self.assertEqual(code, 0)


class FakeCollection:
    """In-memory stand-in for a ChromaDB collection."""

    def __init__(self):
        self.rows = {}
        self.upserts = 0
        self.fail_after = None

    def count(self):
        return len(self.rows)

    def upsert(self, ids, documents, embeddings, metadatas):
        if self.fail_after is not None and self.upserts >= self.fail_after:
            raise KeyboardInterrupt()
        self.upserts += 1
        for doc_id, document in zip(ids, documents):
            self.rows[doc_id] = document


class FakeChromaClient:
    collection = FakeCollection()
    open_error = None

    def __init__(self, path):
        if FakeChromaClient.open_error:
            raise FakeChromaClient.open_error

    def get_or_create_collection(self, name, metadata=None):
        return FakeChromaClient.collection

    def create_collection(self, name, metadata=None):
        FakeChromaClient.collection = FakeCollection()
        return FakeChromaClient.collection

    def delete_collection(self, name):
        pass


class FakeEmbedder:
    """Stands in for Legal-BERT so the tests never load a model or touch the GPU."""

    def encode(self, texts, **kwargs):
        import numpy as np
        return np.zeros((len(texts), 8), dtype=np.float32)


class BuildDbTests(unittest.TestCase):
    """Runs stage 3 end to end with a fake ChromaDB and a fake embedder (no GPU, no real database)."""

    @classmethod
    def setUpClass(cls):
        sys.modules.setdefault("chromadb", types.SimpleNamespace(PersistentClient=FakeChromaClient))
        import runs.build_db as build_db
        cls.build_db = build_db

    def setUp(self):
        for name in os.listdir(TEST_DATA_DIR):
            path = os.path.join(TEST_DATA_DIR, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        FakeChromaClient.collection = FakeCollection()
        FakeChromaClient.open_error = None
        self.cap = FakeCAP()
        for vol in (1, 2, 3):
            self.cap.add_volume(vol, n_cases=4)
        patcher = mock.patch.object(cap_ingest, "_http_get_json", self.cap)
        patcher.start()
        self.addCleanup(patcher.stop)
        embedder = mock.patch.object(self.build_db, "load_embedder", return_value=FakeEmbedder())
        embedder.start()
        self.addCleanup(embedder.stop)
        # If another test module imported the real chromadb first, setUpClass's setdefault did nothing:
        # replace the module build_db actually holds, so no real database is ever opened here.
        database = mock.patch.object(self.build_db, "chromadb", types.SimpleNamespace(PersistentClient=FakeChromaClient))
        database.start()
        self.addCleanup(database.stop)

    def prepare(self, volumes):
        with quiet():
            run_download_stage(volumes)
            run_chunk_stage(volumes)
        self.cap.calls.clear()

    def build(self, volumes_mode, batch_size=4, **kwargs):
        with quiet():
            self.build_db.main(volumes_mode=volumes_mode, custom_batch_size=batch_size, **kwargs)

    def test_build_refuses_unprepared_volumes_without_touching_the_network(self):
        with quiet(), self.assertRaises(ChunksNotAvailableError):
            self.build_db.main(volumes_mode="1-3")
        self.assertEqual(self.cap.calls, [])
        self.assertFalse(os.path.exists(config.CORPUS_JSON_PATH))

    def test_build_resume_and_derived_artifacts(self):
        # 3 volumes x 120 cases x 2 chunks = 720 chunks, enough to cross the 512-chunk indexing checkpoint.
        for vol in (1, 2, 3):
            self.cap.add_volume(vol, n_cases=120)
        self.prepare(["1", "2", "3"])
        with quiet():
            chunks, _, _ = load_cap_chunks(["1", "2", "3"])
        total = len(chunks)
        self.assertEqual(total, 720)

        # Crash partway through the embedding pass: 9 batches of 64 stored, checkpoint written at 512.
        FakeChromaClient.collection.fail_after = 9
        with quiet(), self.assertRaises(KeyboardInterrupt):
            self.build_db.main(volumes_mode="1-3", custom_batch_size=64)
        self.assertEqual(FakeChromaClient.collection.count(), 576)
        self.assertEqual(read_json(config.INDEXING_CHECKPOINT_PATH)["last_indexed_index"], 512)
        self.assertEqual(self.cap.calls, [])
        # Lexical artifacts were persisted before the crash and are tied to this corpus.
        state = read_json(config.CAP_BUILD_STATE_PATH)
        self.assertEqual(state["artifacts"], {"corpus_json": True, "known_cases": True, "case_catalog": True, "bm25": True})
        self.assertEqual(len(read_json(config.CASE_CATALOG_PATH)["reports"]), 360)  # one report per case
        self.assertEqual(len(read_json(config.CORPUS_JSON_PATH)), total)
        bm25_stamp = os.path.getmtime(config.BM25_INDEX_PATH)

        # Resume: nothing is deleted and embedding continues instead of starting over.
        collection = FakeChromaClient.collection
        collection.fail_after = None
        collection.upserts = 0
        self.build("1-3", batch_size=64)
        self.assertIs(FakeChromaClient.collection, collection)
        self.assertEqual(collection.count(), total)
        self.assertEqual(collection.upserts, 4)  # resumed at chunk 512: 208 chunks left, not a full 12-batch pass
        self.assertEqual(bm25_stamp, os.path.getmtime(config.BM25_INDEX_PATH))  # BM25 reused for the same corpus
        checkpoint = read_json(config.INDEXING_CHECKPOINT_PATH)
        self.assertEqual((checkpoint["last_indexed_index"], checkpoint["total_chunks"]), (total, total))
        self.assertEqual(checkpoint["corpus_fingerprint"], state["corpus_fingerprint"])
        self.assertEqual(read_json(config.CORPUS_MANIFEST_PATH)["volumes_indexed"], 3)

        # Already complete: a further run embeds nothing.
        collection.upserts = 0
        with quiet():
            self.build_db.main(volumes_mode="1-3", custom_batch_size=64)
        self.assertEqual(collection.upserts, 0)

    def test_changed_corpus_rebuilds_derived_artifacts_instead_of_reusing_files(self):
        self.prepare(["1", "2", "3"])
        self.build("1-2")
        small = read_json(config.CAP_BUILD_STATE_PATH)["corpus_fingerprint"]
        small_count = len(read_json(config.CORPUS_JSON_PATH))
        collection = FakeChromaClient.collection
        self.assertEqual(collection.upserts, 4)  # 16 chunks in batches of 4

        # Volume 3 is added afterwards: derived files are rebuilt, but only the new chunks are embedded.
        collection.upserts = 0
        self.build("1-3")
        state = read_json(config.CAP_BUILD_STATE_PATH)
        self.assertNotEqual(state["corpus_fingerprint"], small)
        self.assertGreater(len(read_json(config.CORPUS_JSON_PATH)), small_count)
        self.assertEqual(collection.count(), len(read_json(config.CORPUS_JSON_PATH)))
        self.assertEqual(collection.upserts, 2)  # the 8 chunks of volume 3 only

        # A corpus that is not an extension of the indexed one is embedded again from chunk 0.
        collection.upserts = 0
        self.build("2-3")
        self.assertEqual(collection.upserts, 4)

    def test_metadata_change_alone_changes_the_corpus_fingerprint(self):
        self.prepare(["1"])
        with quiet():
            chunks, _, _ = load_cap_chunks(["1"])
        before = cap_ingest.corpus_fingerprint(chunks)
        chunks[0].volume = "999"
        self.assertNotEqual(before, cap_ingest.corpus_fingerprint(chunks))

    def test_unopenable_chromadb_is_never_purged_without_overwrite(self):
        self.prepare(["1"])
        os.makedirs(config.CHROMA_CORPUS_PATH, exist_ok=True)
        marker = os.path.join(config.CHROMA_CORPUS_PATH, "existing_index.bin")
        with open(marker, "w") as handle:
            handle.write("hours of embeddings")
        FakeChromaClient.open_error = ValueError("database is locked")
        with quiet(), self.assertRaises(RuntimeError) as raised:
            self.build_db.main(volumes_mode="1", custom_batch_size=4)
        self.assertIn("Nothing was deleted", str(raised.exception))
        self.assertTrue(os.path.exists(marker))

    def test_overwrite_wipes_the_index_but_never_raw_or_chunk_caches(self):
        self.prepare(["1"])
        self.build("1")
        marker = os.path.join(config.CHROMA_CORPUS_PATH, "old_index.bin")
        with open(marker, "w") as handle:
            handle.write("old")
        raw_before = sorted(os.listdir(config.CAP_RAW_DIR))
        chunks_before = sorted(os.listdir(config.CAP_CHUNKS_DIR))
        self.build("1", overwrite=True)
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(raw_before, sorted(os.listdir(config.CAP_RAW_DIR)))
        self.assertEqual(chunks_before, sorted(os.listdir(config.CAP_CHUNKS_DIR)))
        self.assertTrue(inspect_raw_volume("1").ok)
        self.assertTrue(inspect_chunk_volume("1").ok)


class ChromaSyncTests(unittest.TestCase):
    def test_malformed_database_is_reported_not_deleted(self):
        from src.utils.chroma_sync import prepare_working_chroma
        tmp = tempfile.mkdtemp(prefix="lexagent_test_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db_dir = os.path.join(tmp, "chroma")
        os.makedirs(db_dir)
        sqlite_file = os.path.join(db_dir, "chroma.sqlite3")
        with open(sqlite_file, "wb") as handle:
            handle.write(b"this is not a sqlite database" * 50)
        checkpoint = os.path.join(tmp, "cap_indexing_checkpoint.json")
        atomic_write_json(checkpoint, {"last_indexed_index": 5})

        with self.assertLogs(level="WARNING"), self.assertRaises(RuntimeError):
            prepare_working_chroma(db_dir, db_dir, checkpoint_path=checkpoint, overwrite=False)
        self.assertTrue(os.path.exists(sqlite_file))
        self.assertTrue(os.path.exists(checkpoint))

        with quiet(), self.assertLogs(level="WARNING"):
            prepare_working_chroma(db_dir, db_dir, checkpoint_path=checkpoint, overwrite=True)
        self.assertFalse(os.path.exists(sqlite_file))


if __name__ == "__main__":
    unittest.main()
