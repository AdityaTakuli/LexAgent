#!/usr/bin/env python3
# LexAgent v3.0 | master_run.py
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║                   LEXAGENT v3.0 — MASTER EXECUTION RUNNER                   ║
║                 Caselaw Access Project (CAP) Primary Edition                ║
╚═════════════════════════════════════════════════════════════════════════════╝

Single unified master file capable of:
  1. Constructing the Primary CAP Database in three separate, resumable stages:
       --download-cap  (network -> raw JSON)   --chunk-cap  (raw JSON -> chunks)
       --build-db      (chunks -> BM25 + ChromaDB embeddings on the GPU)
  2. Running Individual Phases (Phase 1, Phase 2, or Phase 3) or All Phases
  3. Running 25 to 30 Cases Comprehensive Benchmark Evaluation
  4. Running Ablation Studies (4 Variants: w/o Defense, w/o CCE, w/o Memory, w/o DDC)
  5. Running the Complete End-to-End Pipeline
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import (
    CHROMA_CORPUS_PATH,
    BM25_INDEX_PATH,
    KNOWN_CASES_PATH,
    CAP_VOLUMES_MODE,
    CAP_CHECKPOINT_INTERVAL,
    DRIVE_BASE,
)

DEFAULT_TEST_QUERY = "Does the Fourteenth Amendment protect a woman's right to privacy in abortion?"


def stage_is_clean(summary: dict) -> bool:
    """True when a download/chunk stage left no volume FAILED or unprocessed.

    Volumes the source does not publish (UNAVAILABLE) are reported by the stage but are
    not an error here: there is nothing a re-run could fetch for them.
    """
    counts = summary["counts"]
    return counts["FAILED"] == 0 and counts["NOT_STARTED"] == 0 and counts["IN_PROGRESS"] == 0


def step_download_cap(
    volumes_mode: str = CAP_VOLUMES_MODE,
    workers: int = 16,
    checkpoint_interval: int = CAP_CHECKPOINT_INTERVAL,
    redownload: bool = False,
) -> dict:
    print("\n" + "█" * 78)
    print(f"  STAGE 1: DOWNLOADING RAW HARVARD CAP VOLUMES ({volumes_mode})")
    print("█" * 78)
    from src.data.cap_ingest import parse_volume_selection, run_download_stage
    return run_download_stage(
        parse_volume_selection(volumes_mode),
        max_workers=workers,
        checkpoint_interval=checkpoint_interval,
        selection=volumes_mode,
        force=redownload,
    )


def step_download_us_reports_pdf(
    volumes_mode: str = CAP_VOLUMES_MODE,
    checkpoint_interval: int = CAP_CHECKPOINT_INTERVAL,
    redownload: bool = False,
) -> dict:
    print("\n" + "█" * 78)
    print(f"  STAGE 1b: OFFICIAL U.S. REPORTS PDFs FOR VOLUMES CAP DOES NOT PUBLISH ({volumes_mode})")
    print("█" * 78)
    from src.data.cap_ingest import parse_volume_selection
    from src.data.us_reports_pdf import run_us_reports_pdf_stage
    return run_us_reports_pdf_stage(
        parse_volume_selection(volumes_mode),
        checkpoint_interval=checkpoint_interval,
        selection=volumes_mode,
        force=redownload,
    )


def step_chunk_cap(
    volumes_mode: str = CAP_VOLUMES_MODE,
    checkpoint_interval: int = CAP_CHECKPOINT_INTERVAL,
    rechunk: bool = False,
) -> dict:
    print("\n" + "█" * 78)
    print(f"  STAGE 2: CHUNKING VERIFIED RAW CAP VOLUMES ({volumes_mode})")
    print("█" * 78)
    from src.data.cap_ingest import parse_volume_selection, run_chunk_stage
    return run_chunk_stage(
        parse_volume_selection(volumes_mode),
        checkpoint_interval=checkpoint_interval,
        selection=volumes_mode,
        force=rechunk,
    )


def step_build_database(
    volumes_mode: str = CAP_VOLUMES_MODE,
    workers: int = 10,
    resume: bool = True,
    overwrite: bool = False,
    portion_size: int = 25,
    batch_size: int = None,
    allow_incomplete: bool = False,
):
    print("\n" + "█" * 78)
    print(f"  STAGE 3: BUILDING PRIMARY HARVARD CAP DATABASE ({volumes_mode})")
    print("█" * 78)
    from runs.build_db import main as build_db_main
    build_db_main(
        volumes_mode=volumes_mode,
        max_workers=workers,
        resume=resume,
        overwrite=overwrite,
        portion_size=portion_size,
        custom_batch_size=batch_size,
        allow_incomplete=allow_incomplete,
    )


def step_prepare_and_build(volumes_mode: str = CAP_VOLUMES_MODE):
    """Run the three database stages in order: download, then chunk, then index."""
    step_download_cap(volumes_mode=volumes_mode)
    step_chunk_cap(volumes_mode=volumes_mode)
    step_build_database(volumes_mode=volumes_mode)


def step_run_phase1(query: str = DEFAULT_TEST_QUERY):
    print("\n" + "█" * 78)
    print("  RUNNING PHASE 1: HYBRID RETRIEVAL OVER CAP DATABASE")
    print("█" * 78)
    from runs.phase1_run import run_phase1_test
    run_phase1_test(query=query)


def step_run_phase2(query: str = DEFAULT_TEST_QUERY):
    print("\n" + "█" * 78)
    print("  RUNNING PHASE 2: FIXED 1-ROUND ADVERSARIAL DEBATE")
    print("█" * 78)
    from runs.phase2_run import run_phase2_test
    run_phase2_test(query=query)


def step_run_phase3(query: str = DEFAULT_TEST_QUERY):
    print("\n" + "█" * 78)
    print("  RUNNING PHASE 3: FULL DVA+ DYNAMIC DEBATE (CCE + DDC + MEMORY)")
    print("█" * 78)
    from runs.phase3_run import run_phase3_test
    run_phase3_test(query=query)


def step_test_all_phases(query: str = DEFAULT_TEST_QUERY):
    print("\n" + "█" * 78)
    print("  RUNNING ALL PHASE VERIFICATION SMOKE TESTS (1 -> 2 -> 3)")
    print("█" * 78)
    step_run_phase1(query)
    step_run_phase2(query)
    step_run_phase3(query)
    print("\n" + "=" * 78)
    print("  ALL PHASE SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 78)


def step_run_benchmark(limit: int = 100):
    print("\n" + "█" * 78)
    print(f"  STEP 3: RUNNING {limit}-CASE COMPREHENSIVE BENCHMARK EVALUATION (FULL RUN)")
    print("█" * 78)
    from experiments.run_25_cases_benchmark import run_benchmark
    run_benchmark(limit=limit)


def step_run_ablation(limit: int = 100):
    print("\n" + "█" * 78)
    print(f"  STEP 4: RUNNING ABLATION STUDIES ({limit} Cases per Variant — FULL SUITE)")
    print("█" * 78)
    from experiments.run_ablation import run_ablation_study
    run_ablation_study(limit=limit)


def step_run_stress_test(output_file: str = "citation_stress_test_results.json", limit: int = 100):
    print("\n" + "█" * 78)
    print(f"  STEP 4: RUNNING CITATION STRESS-TEST PROTOCOL ({limit} Benchmark Cases Pool)")
    print("█" * 78)
    from experiments.run_stress_test import main as run_stress_test_main
    return run_stress_test_main(output_file=output_file, limit=limit)


def step_run_baseline(limit: int = 100):
    print("\n" + "█" * 78)
    print(f"  STEP: RUNNING STANDARD SINGLE-AGENT RAG BASELINE ({limit} Cases — FULL RUN)")
    print("█" * 78)
    from experiments.run_25_cases_benchmark import run_single_agent_rag_baseline
    run_single_agent_rag_baseline(limit=limit)


def step_run_all(volumes_mode: str = "landmark", benchmark_limit: int = 100, ablation_limit: int = 100, stress_limit: int = 100):
    print("\n" + "█" * 78)
    print("  EXECUTING COMPLETE END-TO-END LEXAGENT v3.0 SUITE (FULL UNLIMITED RUN)")
    print("█" * 78)
    step_prepare_and_build(volumes_mode=volumes_mode)
    step_test_all_phases()

    # Critical Research Gate: Stress test runs first to confirm verifier zero false acceptance
    print("\n>>> EXECUTING CITATION STRESS-TEST GATE PRIOR TO BENCHMARKS <<<")
    stress_results = step_run_stress_test(limit=stress_limit)
    if stress_results.get("false_acceptance_rate_misattributed", 0.0) > 0.05:
        print("\n[WARNING] Misattributed citation false-acceptance rate > 5.0%. Review CCE weights!")

    step_run_benchmark(limit=benchmark_limit)
    step_run_ablation(limit=ablation_limit)
    print("\n" + "█" * 78)
    print("  LEXAGENT v3.0 SUITE EXECUTION 100% COMPLETE!")
    print("█" * 78)


def interactive_phase_menu():
    while True:
        print("\n" + "─" * 78)
        print("  SELECT INDIVIDUAL PHASE TO RUN:")
        print("─" * 78)
        print("  [1] Phase 1: CAP Hybrid Retrieval Test (Dense Legal-BERT + BM25 RRF)")
        print("  [2] Phase 2: Fixed 1-Round Adversarial Debate (Prosecutor & Defense)")
        print("  [3] Phase 3: Full DVA+ Dynamic Debate (Active CCE + DDC + Adaptive Memory)")
        print("  [4] Run All Phases (Phase 1 -> Phase 2 -> Phase 3)")
        print("  [0] Back to Main Menu")
        print("─" * 78)

        sub_choice = input("  Select an option [0-4]: ").strip()
        if sub_choice == "0":
            break

        custom_q = input(f"  Enter question (or press Enter for default): ").strip()
        q = custom_q if custom_q else DEFAULT_TEST_QUERY

        if sub_choice == "1":
            step_run_phase1(query=q)
        elif sub_choice == "2":
            step_run_phase2(query=q)
        elif sub_choice == "3":
            step_run_phase3(query=q)
        elif sub_choice == "4":
            step_test_all_phases(query=q)
        else:
            print("  Invalid selection.")


def interactive_menu():
    while True:
        print("\n" + "╔" + "═" * 76 + "╗")
        print("║              LEXAGENT v3.0 — MASTER CONTROL & EXECUTION SUITE              ║")
        print("║                Harvard Caselaw Access Project (CAP) Edition                ║")
        print("╚" + "═" * 76 + "╝")
        print(f"  Current Storage Target: {DRIVE_BASE}")
        print(f"  Default Volume Mode:    {CAP_VOLUMES_MODE}")
        print("-" * 78)
        print("  [1] Build Primary CAP Database (Select: landmark, modern, all, or range)")
        print("  [2] Run Individual Phase Smoke Tests (Phase 1, Phase 2, or Phase 3)")
        print("  [3] Run 25 to 30 Cases Benchmark Evaluation (Full Scorecard & Metrics)")
        print("  [4] Run Ablation Studies (4 Variants w/o Defense, CCE, Memory, DDC)")
        print("  [5] Run Citation Stress-Test Protocol (Novelty 5: 6-Class Diagnostic)")
        print("  [6] Run Standard Single-Agent RAG Baseline (Retrieve -> Mistral-7B)")
        print("  [7] Run Complete End-to-End Suite (Build DB -> Tests -> Stress Test -> Benchmark -> Ablation)")
        print("  [0] Exit")
        print("-" * 78)

        choice = input("  Select an option [0-7]: ").strip()

        if choice == "1":
            print("\n  Volume Ingestion Presets:")
            print("    - 'landmark' : 27 key constitutional volumes (~10 mins, recommended)")
            print("    - 'modern'   : Volumes 300 to 585 (~1940 to 2018)")
            print("    - 'all'      : Volumes 1 to 600 (Comprehensive)")
            print("    - Or custom  : e.g. '400-420' or '410,347,384'")
            v_input = input("  Enter volume selection [default: landmark]: ").strip()
            v_mode = v_input if v_input else "landmark"
            print("\n  Database Stages:")
            print("    - 'download' : Stage 1, download raw volumes only")
            print("    - 'chunk'    : Stage 2, chunk downloaded volumes only (no network)")
            print("    - 'build'    : Stage 3, embed and index chunked volumes (GPU)")
            print("    - 'all'      : Stages 1 -> 2 -> 3 in order")
            stage = input("  Enter stage [default: all]: ").strip().lower() or "all"
            if stage == "download":
                step_download_cap(volumes_mode=v_mode)
            elif stage == "chunk":
                step_chunk_cap(volumes_mode=v_mode)
            elif stage == "build":
                step_build_database(volumes_mode=v_mode)
            elif stage == "all":
                step_prepare_and_build(volumes_mode=v_mode)
            else:
                print("  Invalid stage.")
        elif choice == "2":
            interactive_phase_menu()
        elif choice == "3":
            num_str = input("  Enter number of cases to evaluate [25-30, default 25]: ").strip()
            num = int(num_str) if num_str.isdigit() else 25
            step_run_benchmark(limit=num)
        elif choice == "4":
            num_str = input("  Enter cases per ablation variant [default 10]: ").strip()
            num = int(num_str) if num_str.isdigit() else 10
            step_run_ablation(limit=num)
        elif choice == "5":
            step_run_stress_test()
        elif choice == "6":
            num_str = input("  Enter number of cases for baseline [default 25]: ").strip()
            num = int(num_str) if num_str.isdigit() else 25
            step_run_baseline(limit=num)
        elif choice == "7":
            step_run_all()
        elif choice in ("0", "exit", "q"):
            print("\n  Exiting LexAgent Master Runner. Goodbye!\n")
            break
        else:
            print("  Invalid selection. Please enter a number between 0 and 7.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LexAgent v3.0 Master Runner (Caselaw Access Project Edition)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Database flags: three separate stages, each resumable. Given together they run in this order.
    parser.add_argument("--download-cap", action="store_true", help="Stage 1: download raw CAP volumes only (network -> raw JSON; no chunking, no embedding)")
    parser.add_argument("--download-us-reports-pdf", action="store_true", help="Stage 1b: for volumes CAP does not publish (573+), import the Supreme Court's official bound-volume PDFs into the raw cache")
    parser.add_argument("--chunk-cap", action="store_true", help="Stage 2: chunk verified raw volumes only (raw JSON -> chunks; never uses the network)")
    parser.add_argument("--build-db", action="store_true", help="Stage 3: build BM25, known_cases and the ChromaDB embeddings from verified chunk caches (GPU)")
    parser.add_argument("--volumes", default=CAP_VOLUMES_MODE, help="Volume preset ('landmark', 'modern', 'all' = 1-600, range '400-450')")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LEXAGENT_WORKERS", "16")), help="Parallel download worker threads for --download-cap (default: 16)")
    parser.add_argument("--checkpoint-interval", type=int, default=CAP_CHECKPOINT_INTERVAL, help="Volumes per batch checkpoint in the download and chunk stages (default: 64)")
    parser.add_argument("--redownload", action="store_true", help="Stage 1: re-download the selected volumes even if their raw cache is verified complete")
    parser.add_argument("--rechunk", action="store_true", help="Stage 2: re-chunk the selected volumes even if their chunk cache is verified complete")
    parser.add_argument("--allow-incomplete", action="store_true", help="Continue to later stages / index what is ready even if some target volumes failed")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from the last checkpoint (always the default)")
    parser.add_argument("--overwrite", action="store_true", help="Stage 3 only: wipe and rebuild the ChromaDB collection and derived indexes (never raw or chunk caches)")
    parser.add_argument("--portion-size", type=int, default=50, help="Ignored; kept so old command lines still parse")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for embedding encode/add (default: 256 on GPU, 64 on CPU)")

    # Phase testing flags (individual and all)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run a specific phase smoke test (1, 2, or 3)")
    parser.add_argument("--test-phases", action="store_true", help="Run Phase 1, Phase 2, and Phase 3 verification smoke tests")
    parser.add_argument("--test-phase1", action="store_true", help="Run Phase 1 retrieval test")
    parser.add_argument("--test-phase2", action="store_true", help="Run Phase 2 fixed debate test")
    parser.add_argument("--test-phase3", action="store_true", help="Run Phase 3 full DVA+ test")
    parser.add_argument("--query", default=DEFAULT_TEST_QUERY, help="Custom query string to use for phase testing")

    # Evaluation flags
    parser.add_argument("--benchmark", action="store_true", help="Run the comprehensive benchmark evaluation")
    parser.add_argument("--benchmark-limit", type=int, default=100, help="Number of benchmark cases to evaluate (default: 100 for full suite, 0 for all)")
    parser.add_argument("--baseline", action="store_true", help="Run standard single-agent RAG baseline")
    parser.add_argument("--ablation", action="store_true", help="Run the 4-variant ablation studies")
    parser.add_argument("--ablation-limit", type=int, default=100, help="Number of cases per ablation variant (default: 100 for full suite, 0 for all)")
    parser.add_argument("--stress-test", action="store_true", help="Run Citation Stress-Test Protocol (Novelty 5: 6-class controlled diagnostic)")
    parser.add_argument("--stress-limit", type=int, default=100, help="Number of benchmark cases pool for stress test (default: 100)")

    # End-to-end flag
    parser.add_argument("--all", action="store_true", help="Run complete end-to-end pipeline (Download -> Chunk -> Build DB -> Tests -> Stress Test -> Benchmark -> Ablation)")
    return parser


def main():
    # Banners use box-drawing characters; never let a console or log encoding crash the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    args = build_parser().parse_args()

    has_flags = (
        args.download_cap or args.download_us_reports_pdf or args.chunk_cap or
        args.build_db or args.test_phases or args.test_phase1 or
        args.test_phase2 or args.test_phase3 or args.phase is not None or
        args.benchmark or args.baseline or args.ablation or args.stress_test or args.all
    )

    if not has_flags:
        interactive_menu()
        return

    if args.all:
        step_run_all(
            volumes_mode=args.volumes,
            benchmark_limit=args.benchmark_limit,
            ablation_limit=args.ablation_limit,
            stress_limit=args.stress_limit,
        )
        return

    # Database stages. A stage that leaves volumes FAILED stops the stages after it,
    # unless --allow-incomplete was given, and the process exits non-zero.
    incomplete_stage = None

    if args.download_cap:
        summary = step_download_cap(
            volumes_mode=args.volumes,
            workers=args.workers,
            checkpoint_interval=args.checkpoint_interval,
            redownload=args.redownload,
        )
        if not stage_is_clean(summary):
            incomplete_stage = "download"

    if args.download_us_reports_pdf and (incomplete_stage is None or args.allow_incomplete):
        summary = step_download_us_reports_pdf(
            volumes_mode=args.volumes,
            checkpoint_interval=args.checkpoint_interval,
            redownload=args.redownload,
        )
        if not stage_is_clean(summary):
            incomplete_stage = incomplete_stage or "official PDF"

    if args.chunk_cap and (incomplete_stage is None or args.allow_incomplete):
        summary = step_chunk_cap(
            volumes_mode=args.volumes,
            checkpoint_interval=args.checkpoint_interval,
            rechunk=args.rechunk,
        )
        if not stage_is_clean(summary):
            incomplete_stage = incomplete_stage or "chunk"

    if args.build_db and (incomplete_stage is None or args.allow_incomplete):
        from src.data.cap_ingest import ChunksNotAvailableError
        try:
            step_build_database(
                volumes_mode=args.volumes,
                workers=args.workers,
                resume=not args.overwrite,
                overwrite=args.overwrite,
                portion_size=args.portion_size,
                batch_size=args.batch_size,
                allow_incomplete=args.allow_incomplete,
            )
        except ChunksNotAvailableError as e:
            print(f"\n  [STOPPED] {e}")
            sys.exit(1)

    if incomplete_stage:
        print(f"\n  [STOPPED] The {incomplete_stage} stage did not complete every target volume; see its report above.")
        print("            Run the same command again to retry only the volumes that are not complete.")
        sys.exit(1)

    # Individual Phase execution
    if args.phase == 1 or args.test_phase1:
        step_run_phase1(query=args.query)
    elif args.phase == 2 or args.test_phase2:
        step_run_phase2(query=args.query)
    elif args.phase == 3 or args.test_phase3:
        step_run_phase3(query=args.query)
    elif args.test_phases:
        step_test_all_phases(query=args.query)

    if args.benchmark:
        step_run_benchmark(limit=args.benchmark_limit)

    if args.baseline:
        step_run_baseline(limit=args.benchmark_limit)

    if args.ablation:
        step_run_ablation(limit=args.ablation_limit)

    if args.stress_test:
        step_run_stress_test(limit=args.stress_limit)


if __name__ == "__main__":
    main()
