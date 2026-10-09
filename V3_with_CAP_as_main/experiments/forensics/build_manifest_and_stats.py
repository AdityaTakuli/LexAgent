# LexAgent v3.0 | experiments/forensics/build_manifest_and_stats.py
"""
Writes two files from what already exists. Nothing is re-run and no existing file is changed.

  data/results/EXPERIMENT_MANIFEST.json                     one record per existing experiment
  data/results/forensic_audit/STATISTICAL_VALIDATION.md     tables generated from analysis_existing_results.json

    python experiments/forensics/build_manifest_and_stats.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config  # noqa: E402

RESULTS = config.RESULTS_DIR
AUDIT = os.path.join(RESULTS, "forensic_audit")
MANIFEST_PATH = os.path.join(RESULTS, "EXPERIMENT_MANIFEST.json")
STATS_PATH = os.path.join(AUDIT, "STATISTICAL_VALIDATION.md")
INVALID = "INVALID — PRESERVED FOR AUDIT ONLY"


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=BASE_DIR, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def main():
    analysis = json.load(open(os.path.join(AUDIT, "analysis_existing_results.json"), encoding="utf-8"))
    diagnostics = json.load(open(os.path.join(AUDIT, "diagnostics.json"), encoding="utf-8"))
    inventory = analysis["inventory"]
    logs = analysis["ablation_logs"]
    namespaces = diagnostics["memory_contamination"].get("project_memory_store_namespaces", {})
    namespaces = namespaces if isinstance(namespaces, dict) else {}

    def namespace_with(prefix, verdicts):
        """The run namespace whose stored-verdict count equals what that run's log recorded."""
        found = [name for name, counts in namespaces.items() if name.startswith(prefix) and counts.get("verdicts") == verdicts]
        return found[0] if len(found) == 1 else (found or None)

    commit = git("rev-parse", "HEAD")
    code_state = {
        "git_commit": commit or None,
        "working_tree": "uncommitted changes present when the experiments ran; the exact code state of each run is not pinned",
        "files_changed_since_commit": git("status", "--short").splitlines(),
    }
    shared = {
        "model_name": config.MODEL_NAME, "generation": {"temperature": 0.1, "do_sample": True, "max_new_tokens": 1024, "precision": "bfloat16"},
        "embedding_model": config.EMBEDDER_NAME, "nli_model": config.NLI_MODEL_NAME,
        "retrieval": {"dense_top_k": config.TOP_K_DENSE, "bm25_top_k": config.TOP_K_BM25, "rrf_k": config.RRF_K, "fused_passages": config.TOP_K_DENSE + config.TOP_K_BM25},
        "cce": {"acceptance_threshold_tau": config.TAU_ACCEPTANCE_THRESHOLD, "threshold_named_in_judge_prompt": config.CCS_VERIFIED_THRESHOLD,
                "uncertain_lower_bound": config.CCS_UNCERTAIN_THRESHOLD, "existence_threshold": config.CCE_FUZZY_THRESHOLD_RATIO,
                "passages_read": 3, "scan_limit_characters": config.CCE_OPINION_SCAN_LIMIT, "window_characters": 450, "window_step": 350},
        "ddc": {"max_rounds": config.DDC_MAX_ROUNDS, "continue_below_mean_ccs": config.DDC_CCS_CONTINUE_THRESH, "continue_below_min_ccs": 0.40,
                "min_gaps": config.DDC_MIN_GAPS_TO_CONTINUE, "challenge_strength_threshold": config.DDC_CHALLENGE_STRENGTH_THRESH},
        "memory": {"verdict_cache_cosine_threshold": config.MEMORY_CACHE_THRESHOLD, "high_risk_mean_ccs": config.MEMORY_HIGH_RISK_CCS_THRESH,
                   "cache_and_risk_embedding_model": config.EMBEDDER_NAME},
        "random_seed": f"config.RANDOM_SEED = {config.RANDOM_SEED}, but config.set_random_seed() is never called by any runner: runs are unseeded",
        "benchmark_sha256": analysis["benchmark"]["sha256"], "benchmark_cases": analysis["benchmark"]["cases"],
        "corpus_fingerprint": json.load(open(config.CORPUS_MANIFEST_PATH, encoding="utf-8")).get("corpus_fingerprint"),
    }

    def record(name, file, cases, status, valid, **extra):
        path = os.path.join(RESULTS, file) if file else None
        info = inventory.get(file, {}) if file else {}
        return {"experiment": name, "result_file": path.replace("\\", "/") if path else None, "sha256": info.get("sha256"),
                "result_written": info.get("modified"), "case_count": cases, "status": status, "valid_for_publication": valid, **extra}

    conf = analysis["confidence_decomposition"]
    experiments = [
        record("LexAgent full system, benchmark run", "benchmark_100_cases_results.json", 96, "COMPLETE",
               "YES, AS A MEASUREMENT OF THE SYSTEM AS BUILT — NOT as a measurement of the debate architecture: "
               f"{conf['benchmark_run']['cache_hits']} of 96 answers were returned by the verdict cache for a different question. Must be disclosed.",
               memory_namespace=(max((n for n in namespaces if n.startswith("bench_")), key=lambda n: namespaces[n].get("verdicts", 0), default="bench_<start time>") + " (fresh namespace created by run_25_cases_benchmark.py; identified as the bench_ namespace with the most stored verdicts)"),
               per_case_output="scorecards only; no answers, citations, passages, citation-check reports or judge output were saved",
               log="data/logs/benchmark_full.log (stdout only)", skipped_debates=conf["benchmark_run"]["cache_hits"]),
        record("Single-agent RAG baseline", "baseline_standard_rag_results.json", 96, "COMPLETE",
               "YES. Note: its dense query omits the jurisdiction string the debate appends, so its retrieved passages differ "
               "from the debate's in most cases; it reads 4 passages of 300 characters.",
               memory_namespace="none (no memory)", per_case_output="scorecards only", log="data/logs/baseline_full.log"),
        record("Citation stress test (256 labelled citations)", "citation_stress_test_results.json", 256, "COMPLETE", "YES",
               memory_namespace="none", per_case_output="per-citation tier, existence, support"),
        record("Citation stress test before the evidence-selection change", "citation_stress_test_results_before_cce_fix.json", 285, "SUPERSEDED",
               "ONLY as the historical 'before' column; different item counts (285) and the earlier checker",
               memory_namespace="none"),
        record("Citation checker: evidence-selection study", "cce_evidence_selection_study.json", 192, "COMPLETE", "YES", memory_namespace="none",
               per_case_output="per-citation support score for each selection"),
    ]
    variant_notes = {
        "Full": "debate graph, all components", "A1": "defense node removed", "A3": "memory nodes removed", "A4": "one round only",
    }
    run_id = "1791231448"
    for variant in ("Full", "A1", "A3", "A4"):
        log = logs[variant]
        experiments.append(record(
            f"Ablation {variant}: {variant_notes[variant]}", "ablation_summary.json", 96, "COMPLETE",
            ("YES — the only run of the debate that the verdict cache did not touch; means only" if variant == "A3" else
             f"YES, AS MEASURED, WITH DISCLOSURE: {log['cache_hits']} of 96 debates were skipped by the verdict cache; means only"),
            row=variant, memory_namespace=(f"ablation_{run_id}_{variant}" if variant != "A3" else f"ablation_{run_id}_A3 (created, never used)"),
            per_case_output="NONE: one mean per measure; per-case behaviour is recoverable only from the log",
            log="data/logs/ablation_full.log", started=log["started"], ended=log["ended"], minutes=log["minutes"],
            skipped_debates=log["cache_hits"], debated=log["debated"]))
    a2 = logs["A2"]
    experiments.append(record(
        "Ablation A2 (corrected): citation checker removed", "ablation_summary.json", 96, "COMPLETE",
        f"YES, AS MEASURED, WITH DISCLOSURE: {a2['cache_hits']} of 96 debates were skipped by the verdict cache; means only",
        row="A2", memory_namespace=namespace_with("ablation_", a2["verdicts_stored"]) or "ablation_<rerun id>_A2",
        per_case_output="NONE", log="data/logs/ablation_A2_rerun.log", started=a2["started"], ended=a2["ended"], minutes=a2["minutes"],
        skipped_debates=a2["cache_hits"], debated=a2["debated"],
        code_note="run after src/agents/state.py declared the ablation switches"))
    old = logs["A2_before_switch_fix (INVALID)"]
    experiments.append(record(
        "Ablation A2 (first run): switch did not reach the judge", "ablation_summary_before_switch_fix.json", 96, INVALID, INVALID,
        row="A2", memory_namespace=f"ablation_{run_id}_A2", per_case_output="NONE", log="data/logs/ablation_full.log",
        started=old["started"], ended=old["ended"], minutes=old["minutes"],
        reason="LangGraph dropped use_cce from the state, so the judge rejected every citation as unverified "
               f"({old['judge_rejections_total']} rejections logged) and confidence was capped at 0.20. Do not use; do not delete."))
    for file, cases in (("benchmark_2_cases_results.json", 2), ("benchmark_3_cases_results.json", 3), ("ablation_summary_2case_dryrun.json", 2)):
        if file in inventory:
            experiments.append(record(f"Smoke run ({file})", file, cases, "DRY RUN", "NO — pipeline check on 2–3 cases", memory_namespace="own"))
    experiments.append({"experiment": "Recorded UI debate", "result_file": os.path.join(BASE_DIR, "ui", "sample_transcript.json").replace("\\", "/"),
                        "case_count": 1, "status": "COMPLETE", "valid_for_publication": "ILLUSTRATION ONLY — one unseeded debate with its full trace",
                        "memory_namespace": "ui_chat"})
    experiments.append({"experiment": "Forensic audit (this audit)", "result_file": AUDIT.replace("\\", "/"), "status": "COMPLETE",
                        "valid_for_publication": "diagnostics, not experiments; no LLM was called and nothing was re-run"})

    manifest = {
        "written": datetime.now().isoformat(timespec="seconds"),
        "rule": "Existing result files are part of the research record. They are listed here with their SHA-256 and were not modified.",
        "code_state": code_state, "settings_shared_by_all_runs": shared,
        "memory_namespaces_in_store": namespaces, "experiments": experiments,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    # ── STATISTICAL_VALIDATION.md ───────────────────────────────────────────
    lines = ["# Statistical validation (from existing per-case outputs only)", "",
             "Generated by `experiments/forensics/build_manifest_and_stats.py` from `analysis_existing_results.json`. "
             "No experiment was re-run. Bootstrap: 10,000 paired resamples, seed 42. Permutation test: 10,000 sign flips, two-sided.", ""]

    def fmt(x, d=4):
        return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))

    def table(block, title):
        lines.extend([f"## {title} (n = {block['n']})", "",
                      "| Metric | LexAgent mean | Baseline mean | LexAgent median | Baseline median | SD of difference | SE of difference | Mean difference | Bootstrap 95% CI | Permutation p | Wilcoxon p | Cohen's dz | Rank-biserial | Significant at 0.05 |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
        names = {"hr": "Fabricated-name rate", "ass": "Answer similarity", "ar": "Authority recall", "ap": "Authority precision"}
        for key in ("hr", "ass", "ar", "ap"):
            m = block[key]
            ci = m["bootstrap_95ci_mean_difference"]
            significant = "yes" if (m["sign_flip_permutation_p"] < 0.05 and (ci[0] > 0 or ci[1] < 0)) else "no"
            lines.append(f"| {names[key]} | {fmt(m['a']['mean'])} | {fmt(m['b']['mean'])} | {fmt(m['a']['median'])} | {fmt(m['b']['median'])} | "
                         f"{fmt(m['difference']['sd'])} | {fmt(m['difference']['se'])} | {m['difference']['mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                         f"{fmt(m['sign_flip_permutation_p'])} | {fmt(m.get('wilcoxon_signed_rank_p'))} | {fmt(m.get('cohens_dz'))} | {fmt(m.get('rank_biserial'))} | {significant} |")
        mc = block["ar"].get("mcnemar", {})
        lines.extend(["", f"Authority recall, McNemar exact test: target cited by LexAgent only in {mc.get('a_only')} cases, by the baseline only in {mc.get('b_only')}; p = {fmt(mc.get('exact_p'), 5)}.",
                      f"Citations per answer: LexAgent {block['citations_per_answer']['lexagent']}, baseline {block['citations_per_answer']['baseline']}. "
                      f"Answers without any citation: LexAgent {block['citation_free_answers']['lexagent']}, baseline {block['citation_free_answers']['baseline']}. "
                      f"Flagged names: LexAgent {block['flagged_names']['lexagent']} of {block['flagged_names']['lexagent_citations']} citations, "
                      f"baseline {block['flagged_names']['baseline']} of {block['flagged_names']['baseline_citations']}.", ""])

    comparison = analysis["lexagent_vs_baseline"]
    table(comparison["all_96_cases"], "Full system (benchmark run) against baseline — all cases")
    table(comparison["debated_cases_only"], "Full system against baseline — cases in which a debate actually ran")
    table(comparison["cache_signature_cases_only"], "Full system against baseline — cases answered from the verdict cache")
    lines.extend(["> " + comparison["caution"], ""])

    lines.extend(["## Full system against A1, A2, A3, A4", "", analysis["ablation_paired_from_logs"]["limitation"], "",
                  "Saved means (all 96 cases) and what the logs add:", "",
                  "| Variant | Fabricated-name rate | Answer similarity | Mean confidence (saved, /96) | Debates skipped by cache | Debated | Mean confidence of debated cases (implied) | Rounds when debated | Verdicts stored |",
                  "|---|---|---|---|---|---|---|---|---|"])
    means = {r["variant"]: r for r in json.load(open(os.path.join(RESULTS, "ablation_summary.json"), encoding="utf-8"))}
    for v in ("Full", "A1", "A2", "A3", "A4"):
        log = logs[v]
        lines.append(f"| {v} | {100 * means[v]['mean_hr']:.2f}% | {means[v]['mean_cas']:.4f} | {means[v]['mean_confidence']:.4f} | {log['cache_hits']} | {log['debated']} | "
                     f"{fmt(conf[v]['implied_mean_confidence_of_debated_cases'])} | {log['debate_rounds_distribution']} | {log['verdicts_stored']} |")
    b = conf["benchmark_run"]
    lines.extend([f"| Benchmark run (same configuration as Full) | {100 * comparison['all_96_cases']['hr']['a']['mean']:.2f}% | {comparison['all_96_cases']['ass']['a']['mean']:.4f} | "
                  f"{b['saved_mean_confidence']:.4f} | {b['cache_hits']} | {b['debated']} | {b['implied_mean_confidence_of_debated_cases']:.4f} | "
                  f"{analysis['benchmark_cache_signature']['debated_rounds_distribution']} | not logged |", "",
                  "The implied value is `saved mean × 96 / debated`, exact because a skipped debate leaves confidence at 0.0. "
                  "For the benchmark run it is computed directly from the per-case file and agrees.", "",
                  "Paired comparisons of quantities the logs record per case (Full minus variant):", "",
                  "| Comparison | Quantity | Full | Variant | Mean difference | Bootstrap 95% CI | Test | p |", "|---|---|---|---|---|---|---|---|"])
    for name, entry in analysis["ablation_paired_from_logs"]["comparisons"].items():
        hit = entry["cache_hit"]
        lines.append(f"| {name} | debate skipped by cache (0/1), n=96 | {fmt(hit['a']['mean'])} | {fmt(hit['b']['mean'])} | {hit['difference']['mean']:+.4f} | "
                     f"{hit['bootstrap_95ci_mean_difference']} | McNemar exact | {fmt(hit['mcnemar']['exact_p'], 5)} |")
        if "final_round_avg_ccs" in entry:
            c = entry["final_round_avg_ccs"]
            s = entry["final_round_has_supported_citation"]
            n = entry["cases_debated_with_citation_check_in_both"]
            lines.append(f"| {name} | final-round mean CCS, n={n} | {fmt(c['a']['mean'])} | {fmt(c['b']['mean'])} | {c['difference']['mean']:+.4f} | "
                         f"{c['bootstrap_95ci_mean_difference']} | Wilcoxon | {fmt(c.get('wilcoxon_signed_rank_p'), 5)} |")
            lines.append(f"| {name} | final round has a supported citation (0/1), n={n} | {fmt(s['a']['mean'])} | {fmt(s['b']['mean'])} | {s['difference']['mean']:+.4f} | "
                         f"{s['bootstrap_95ci_mean_difference']} | McNemar exact | {fmt(s['mcnemar']['exact_p'], 5)} |")
    variation = analysis["run_to_run_variation"]
    lines.extend(["", "## Run-to-run variation of the fabricated-name rate", "",
                  f"- Benchmark run: {variation['benchmark_run']['flagged_names']} flagged names among {variation['benchmark_run']['citations']} citations, in "
                  f"{variation['benchmark_run']['cases_with_a_flagged_name']} cases (per-case rates {variation['per_case_hr_values_when_nonzero']}); mean of per-case rates {100 * variation['benchmark_run']['mean_hr']:.2f}%.",
                  f"- Exact 95% interval for the pooled rate {variation['benchmark_run']['flagged_names']}/{variation['benchmark_run']['citations']}: "
                  f"[{100 * variation['benchmark_pooled_rate_exact_95ci'][0]:.2f}%, {100 * variation['benchmark_pooled_rate_exact_95ci'][1]:.2f}%].",
                  "- Ablation runs, sum of per-case rates over 96 cases (mean × 96): "
                  + ", ".join(f"{v} {variation[f'ablation_{v}']['sum_of_per_case_hr']}" for v in ("Full", "A1", "A2", "A3", "A4")) + ".",
                  f"- A count with expectation 3 varies between {variation['poisson_95_interval_for_a_count_with_mean_3'][0]} and "
                  f"{variation['poisson_95_interval_for_a_count_with_mean_3'][1]} in 95% of repetitions (Poisson).",
                  "- Every run flagged between 0 and 3 names. The spread between the two runs of the identical full configuration "
                  "(1.39% and 0.26%) is as large as the largest difference between any two variants. **No fabricated-name difference "
                  "among the variants, or between the full system and the baseline, is distinguishable from run-to-run noise.**", ""])
    sep = analysis["support_score_separability"]
    st = sep["stress_test"]
    lines.extend(["## Citation checker: can any threshold separate correct from wrong?", "",
                  f"Support score of the {st['correct_n']} correct citations: mean {st['correct_support']['mean']:.3f}, median {st['correct_support']['median']:.3f}; "
                  f"{st['correct_support_below_0.10']} of them score below 0.10. Swapped holdings: mean {st['swapped_support']['mean']:.3f}, median {st['swapped_support']['median']:.3f}.",
                  f"Area under the ROC curve, correct against swapped: **{st['auc_correct_vs_swapped']}**; correct against hand-written false holdings: {st['auc_correct_vs_false_holding']}.", "",
                  "| Threshold on support | Correct accepted (of 96) | Swapped accepted (of 96) | False holdings accepted (of 20) |", "|---|---|---|---|"])
    for row in st["threshold_sweep"]:
        lines.append(f"| {row['threshold']:.2f} | {row['correct_accepted']} | {row['swapped_accepted']} | {row['false_holding_accepted']} |")
    lines.extend(["", "Evidence-selection study, same question per selection:", "",
                  "| Passages read | AUC correct vs swapped | Accepted at 0.70 (correct, swapped) | Best threshold | Accepted at best threshold |", "|---|---|---|---|---|"])
    for name, row in sep["evidence_study"]["strategies"].items():
        lines.append(f"| {name} | {row['auc_correct_vs_swapped']} | {row['accepted_at_0.70']} | {row['best_threshold_by_correct_minus_swapped']} | {row['accepted_at_best_threshold']} |")
    lines.extend(["", "No selection reaches an AUC above 0.74, and no threshold accepts most correct citations without accepting a comparable "
                  "share of swapped ones. The 0.70 / 0.75 thresholds are configuration constants; the data give no threshold that would justify them.", ""])
    with open(STATS_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("wrote", MANIFEST_PATH)
    print("wrote", STATS_PATH)
    print("A2 rerun namespace:", namespace_with("ablation_", a2["verdicts_stored"]))


if __name__ == "__main__":
    main()
