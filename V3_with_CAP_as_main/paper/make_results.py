# LexAgent v3.0 | paper/make_results.py
"""Write every number the paper reports to paper/generated/, computed from the saved result files.

The paper never contains a hand-typed result: lexagent_ieee.tex reads the macros and data tables
this script writes. Run it again after any experiment is re-run, then rebuild the PDF.

    .venv\\Scripts\\python.exe paper/make_results.py
"""

import ast
import json
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PAPER_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import (  # noqa: E402
    CAP_CHUNK_CHECKPOINT_PATH,
    CAP_DOWNLOAD_CHECKPOINT_PATH,
    CASE_CATALOG_PATH,
    CHROMA_CORPUS_PATH,
    CORPUS_MANIFEST_PATH,
    RESULTS_DIR,
)
from experiments.benchmark_cases import split_benchmark_by_source_text  # noqa: E402
from src.data.case_catalog import CaseCatalog, names_match  # noqa: E402

OUT_DIR = os.path.join(PAPER_DIR, "generated")
LOGS_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "logs")
AUDIT_DIR = os.path.join(RESULTS_DIR, "forensic_audit")
V2_DIR = os.path.join(RESULTS_DIR, "v2")
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
VARIANT_ORDER = ["Full", "A1", "A2", "A3", "A4"]
VERDICTS = ["SUPPORTED", "UNCERTAIN", "MISATTRIBUTED", "FABRICATED", "UNVERIFIABLE"]
STRESS_TYPES = ["correct", "misattributed", "casehold_distractor", "fabricated_with_cite", "fabricated_no_cite", "real_off_corpus"]
STRESS_MACRO = {"correct": "Correct", "misattributed": "Misattr", "casehold_distractor": "Casehold",
                "fabricated_with_cite": "FabCite", "fabricated_no_cite": "FabNoCite", "real_off_corpus": "OffCorpus"}


def load(name: str):
    with open(os.path.join(RESULTS_DIR, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def wilson(successes: int, total: int, z: float = 1.96):
    """95% Wilson score interval for a proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap(differences, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED):
    """95% percentile interval for the mean of per-case differences."""
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}"


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def variant_minutes(log_path: str) -> dict:
    """Minutes each ablation variant ran, from the first and last log timestamp inside its section."""
    if not os.path.exists(log_path):
        return {}
    stamp = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    header = re.compile(r"Evaluating Variant \[(\w+)\]")
    spans, finished, current = {}, {}, None
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            started = header.search(line)
            if started:
                current = started.group(1)
                continue
            if current is None:
                continue
            if "-> Result:" in line:
                # Only a variant that ran to its result line is timed; a run still in progress is not.
                if current in spans:
                    finished[current] = spans[current]
                current = None
                continue
            found = stamp.search(line)
            if found:
                moment = datetime.strptime(found.group(1), "%Y-%m-%d %H:%M:%S")
                first, _ = spans.get(current, (moment, moment))
                spans[current] = (first, moment)
    return {variant: (last - first).total_seconds() / 60 for variant, (first, last) in finished.items()}


def avoid_lists_in_full_run(log_path: str) -> list:
    """The lists of 'high-risk' citations memory handed to the agents, one per debate of the full variant."""
    if not os.path.exists(log_path):
        return []
    lists, inside = [], False
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "Evaluating Variant [" in line:
                inside = "Evaluating Variant [Full]" in line
            elif inside and "High-risk citations loaded:" in line:
                found = re.search(r"High-risk citations loaded: (\[.*\])", line)
                try:
                    lists.append(ast.literal_eval(found.group(1)))
                except (AttributeError, SyntaxError, ValueError):
                    pass
    return lists


def signed(value: float, digits: int) -> str:
    """A signed number for running text and tables: a true minus sign, not a hyphen."""
    text = f"{abs(value):.{digits}f}"
    if float(text) == 0:
        return text
    return (r"$-$" if value < 0 else "+") + text


def p_text(p: float) -> str:
    """A p-value with its relation sign, for running text and tables."""
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


TABLE_ROWS = {}


def write_rows(name: str, rows) -> None:
    """Rows of a table as one macro (\\ResRows<name>), so that the table follows the result file row by row."""
    TABLE_ROWS[name] = "%\n".join(" & ".join(str(cell) for cell in row) + r" \\" for row in rows)


def audit_values(put) -> None:
    """Findings of the audit of the first evaluation (data/results/forensic_audit/, read-only)."""
    analysis = load_json(os.path.join(AUDIT_DIR, "analysis_existing_results.json"))
    diagnostics = load_json(os.path.join(AUDIT_DIR, "diagnostics.json"))

    signature = analysis["benchmark_cache_signature"]
    put("AudCacheBench", signature["cases_with_signature"])
    put("AudDebated", analysis["lexagent_vs_baseline"]["debated_cases_only"]["n"])
    put("AudCacheSignatureComplete", "1" if (signature["all_one_round_cases_have_signature"] and signature["all_citation_free_cases_have_signature"]) else "0")
    logs = {name: row for name, row in analysis["ablation_logs"].items() if "INVALID" not in name}
    with_memory = [row["cache_hits"] for name, row in logs.items() if name != "A3"]
    put("AudCacheAblMin", min(with_memory))
    put("AudCacheAblMax", max(with_memory))
    for variant, word in (("Full", "Full"), ("A1", "AOne"), ("A2", "ATwo"), ("A3", "AThree"), ("A4", "AFour")):
        put(f"AudCacheAbl{word}", logs[variant]["cache_hits"])
    rounds = signature["debated_rounds_distribution"]
    put("AudDebatedAtMaxRounds", rounds.get("3", 0))
    put("AudDebatedBelowMaxRounds", sum(count for number, count in rounds.items() if number != "3"))

    debated = analysis["lexagent_vs_baseline"]["debated_cases_only"]["ar"]
    put("AudDebatedLexAR", pct(debated["a"]["mean"], 1))
    put("AudDebatedBaseAR", pct(debated["b"]["mean"], 1))
    put("AudDebatedARLow", signed(100 * debated["bootstrap_95ci_mean_difference"][0], 1))
    put("AudDebatedARHigh", signed(100 * debated["bootstrap_95ci_mean_difference"][1], 1))
    put("AudDebatedARP", p_text(debated["sign_flip_permutation_p"]))
    put("AudDebatedConf", f"{analysis['benchmark_confidence']['debated_only']['mean']:.2f}")
    put("AudAPIdentity", "1" if analysis["authority_precision_identity_check"] else "0")
    decomposition = analysis["confidence_decomposition"]
    put("AudImpliedConfFull", f"{decomposition['Full']['implied_mean_confidence_of_debated_cases']:.2f}")
    put("AudImpliedConfAThree", f"{decomposition['A3']['implied_mean_confidence_of_debated_cases']:.2f}")

    no_memory = logs["A3"]["citation_check"]
    put("AudAThreeSupportedAny", no_memory["supported_in_any_round"])
    put("AudAThreeSupportLost", no_memory["supported_earlier_but_not_in_final_round"])
    put("AudAThreeDebates", no_memory["debates_with_citation_check"])

    similarity = diagnostics["answer_similarity_metric"]
    put("AudSimFloor", f"{similarity['holding_of_a_different_case']['mean']:.3f}")
    put("AudSimQuestion", f"{similarity['the_question_itself_as_the_answer']['mean']:.3f}")

    collisions = diagnostics["cache_collisions"]
    put("AudCollisionQuestions", collisions["questions_with_an_earlier_question_above_threshold"])
    put("AudCollisionPairs", collisions["pairs_at_or_above_threshold"])
    put("AudCollisionPairsTotal", thousands(collisions["pairs_total"]))
    put("AudNoCollisionThreshold", f"{collisions['threshold_needed_for_no_collision_among_these_questions']:.4f}")
    put("AudCacheSameTarget", collisions["ablation_full_run"]["hits_where_both_questions_share_the_target_decision"])
    put("AudCacheAblFullReproduced", collisions["ablation_full_run"]["logged_distance_reproduced_from_question_embeddings"])


VERIFIER_MODELS = [
    ("cross-encoder/nli-deberta-v3-small", "DeBERTa-v3-small NLI (first evaluation)"),
    ("cross-encoder/nli-deberta-v3-base", "DeBERTa-v3-base NLI"),
    ("cross-encoder/nli-deberta-v3-large", "DeBERTa-v3-large NLI"),
    ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "DeBERTa-v3-large, five NLI corpora"),
    ("tasksource/deberta-base-long-nli", "DeBERTa-base long-input NLI"),
    ("generator-yes-no", "Generator asked yes/no"),
]
SELECTION_TEXT = {"dense_top3": "dense, 3", "hybrid_top3": "fused, 3", "hybrid_top5": "fused, 5"}
RETRIEVERS = [
    ("legal-bert, question + jurisdiction (the debate's query)", "Legal-BERT (dense retriever of the system)"),
    ("bm25, whitespace tokens (the system's)", "BM25, whitespace tokens (lexical retriever of the system)"),
    ("bm25, word tokens", "BM25, word tokens"),
    ("bge-base-en-v1.5 (retrieval-trained)", "bge-base-en-v1.5"),
    ("all-MiniLM-L6-v2 (retrieval-trained)", "all-MiniLM-L6-v2"),
]
WORKLOADS = [("first_pass_distinct_questions", "Distinct questions"), ("exact_repeats", "Exact repeats"),
             ("format_variants", "Format variants"), ("paraphrases", "Paraphrases")]


def component_values(put) -> None:
    """Post-audit experiments that need no debate: verifier validation, retrieval, verdict cache."""
    # ── Support model validation ────────────────────────────────────────────
    validation = load_json(os.path.join(V2_DIR, "e2_verifier_validation", "results.json"))
    decision = validation["decision"]
    candidates = validation["candidates"]
    chosen = candidates[decision["selected_candidate"]]
    put("GateOpen", "1" if decision["validated"] else "0")
    put("GateRequiredAUC", f"{decision['required_test_auc']:.2f}")
    put("GateThreshold", f"{decision['threshold']:.2f}")
    put("GatePassages", decision["passages"])
    put("GateDevAUC", f"{decision['development_auc']:.3f}")
    put("GateTestAUC", f"{decision['test_auc']:.3f}")
    put("GateTestAUCLow", f"{decision['test_auc_95ci'][0]:.3f}")
    put("GateTestAUCHigh", f"{decision['test_auc_95ci'][1]:.3f}")
    put("GateTestCorrect", chosen["test"]["correct_accepted"])
    put("GateTestSwapped", chosen["test"]["swapped_accepted"])
    put("GateTestFalse", chosen["test"]["false_holding_accepted"])
    put("GateCandidates", len(candidates))
    put("GateModels", len({row["model"] for row in candidates.values()}))
    for split, tag in (("development", "Dev"), ("test", "Test")):
        counts = validation["citations"][split]
        put(f"Gate{tag}SetCorrect", counts["correct"])
        put(f"Gate{tag}SetSwapped", counts["misattributed"])
        put(f"Gate{tag}SetFalse", counts["casehold_distractor"])
    rows, other_areas = [], []
    for model, label in VERIFIER_MODELS:
        own = {name: row for name, row in candidates.items() if row["model"] == model}
        if not own:
            continue
        best_name = max(own, key=lambda name: own[name]["development"]["auc_correct_vs_wrong"])
        best = own[best_name]
        selected = best_name == decision["selected_candidate"]
        if model.startswith("cross-encoder/") or model.startswith("MoritzLaurer/"):
            other_areas.append(best["test"]["auc_correct_vs_wrong"])
        unit = "passages" if best["granularity"] == "passage" else "windows"
        cells = [label, f"{unit}; {SELECTION_TEXT[best['selection']]}", f"{best['development']['auc_correct_vs_wrong']:.3f}",
                 f"{best['test']['auc_correct_vs_wrong']:.3f}", best["test"]["correct_accepted"],
                 best["test"]["swapped_accepted"], best["test"]["false_holding_accepted"]]
        rows.append([rf"\textbf{{{cell}}}" for cell in cells] if selected else cells)
    write_rows("Verifier", rows)
    put("GateOtherTestAUCMin", f"{min(other_areas):.2f}")
    put("GateOtherTestAUCMax", f"{max(other_areas):.2f}")
    first = candidates["cross-encoder/nli-deberta-v3-small | 450-character windows | dense_top3"]
    put("GateFirstModelTestAUC", f"{first['test']['auc_correct_vs_wrong']:.3f}")

    probe = load_json(os.path.join(V2_DIR, "e2_verifier_validation", "near_miss_probe.json"))
    put("ProbeN", probe["citations_per_kind"])
    put("ProbeCorrect", probe["accepted"]["correct_holding"])
    put("ProbeUnrelated", probe["accepted"]["holding_of_an_unrelated_case"])
    put("ProbeNearMiss", probe["accepted"]["holding_of_the_most_similar_other_case"])
    put("ProbeCorrectPct", pct(probe["accepted"]["correct_holding"] / probe["citations_per_kind"], 1))
    put("ProbeNearMissPct", pct(probe["accepted"]["holding_of_the_most_similar_other_case"] / probe["citations_per_kind"], 1))

    # ── Retrieval ───────────────────────────────────────────────────────────
    retrieval = load_json(os.path.join(V2_DIR, "e5_retrieval", "results.json"))
    put("RetQuestions", retrieval["questions"])
    rows = []
    for key, label in RETRIEVERS:
        entry = retrieval["single_retrievers"][key]
        rows.append([label, entry["top1"]["found"], entry["top4"]["found"], entry["top8"]["found"]])
    system, best = retrieval["the_system_as_it_is"], retrieval["best_fusion"]
    rows.append([r"\midrule Fusion as deployed (Legal-BERT + BM25, 4 + 4)", system["target_at_rank_1"], system["target_in_top_4"], system["target_in_top_8"]])
    rows.append([rf"Best fusion tested (bge-base + word-token BM25, {best['depth_each']} + {best['depth_each']})",
                 best["target_at_rank_1"], best["target_in_top_4"], best["target_in_top_8"]])
    write_rows("Retrieval", rows)
    dense = retrieval["single_retrievers"][RETRIEVERS[0][0]]
    lexical = retrieval["single_retrievers"][RETRIEVERS[1][0]]
    put("RetDenseTopFour", dense["top4"]["found"])
    put("RetDenseTopFourPct", pct(dense["top4"]["share"], 0))
    put("RetLexicalTopFour", lexical["top4"]["found"])
    put("RetLexicalTopFourPct", pct(lexical["top4"]["share"], 0))
    put("RetBgeTopFour", retrieval["single_retrievers"]["bge-base-en-v1.5 (retrieval-trained)"]["top4"]["found"])
    put("RetSystemTopEight", system["target_in_top_8"])
    put("RetSystemTopEightPct", pct(system["share_top_8"], 0))
    put("RetSystemMissEight", retrieval["questions"] - system["target_in_top_8"])
    put("RetSystemRankOne", system["target_at_rank_1"])
    put("RetBestTopEight", best["target_in_top_8"])
    put("RetBestRankOne", best["target_at_rank_1"])

    # ── Verdict cache ───────────────────────────────────────────────────────
    memory = load_json(os.path.join(V2_DIR, "e3_memory", "results.json"))
    put("MemStored", memory["verdicts_stored_in_pass_one"])
    rows = []
    for key, label in WORKLOADS:
        load_row = memory["workloads"][key]
        old, new = load_row["similarity_0.92"], load_row["exact"]
        rows.append([label, load_row["queries"], old["hits"], old["correct"], old["wrong_question"], new["hits"], new["correct"], new["wrong_question"]])
    write_rows("Memory", rows)
    first_pass = memory["workloads"]["first_pass_distinct_questions"]
    put("MemFirstPassQueries", first_pass["queries"])
    put("MemFirstPassOldWrong", first_pass["similarity_0.92"]["wrong_question"])
    put("MemFirstPassNewWrong", first_pass["exact"]["wrong_question"])
    put("MemRepeatNewCorrect", memory["workloads"]["exact_repeats"]["exact"]["correct"])
    put("MemRepeatOldWrong", memory["workloads"]["exact_repeats"]["similarity_0.92"]["wrong_question"])
    put("MemParaphraseNewHits", memory["workloads"]["paraphrases"]["exact"]["hits"])
    put("MemNewWrongTotal", sum(memory["workloads"][key]["exact"]["wrong_question"] for key, _ in WORKLOADS))
    put("MemRehallucinationMeasurable", "1" if memory["re_hallucination"]["measurable"] else "0")


def read_cases(experiment: str, run: str) -> list:
    path = os.path.join(V2_DIR, experiment, run, "cases.jsonl")
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return [row for row in rows if not row.get("error")]


def loss_values(put) -> None:
    """Where the repaired system (first seed) loses the target between the debate and the answer.

    Read from the saved traces of e4_repaired_Full/seed0. This is the evidence on which the remaining
    repairs were chosen, so it is computed here and not typed into the paper.
    """
    rows = read_cases("e4_repaired_Full", "seed0")
    put("LossCases", len(rows))

    def cited(citations, target):
        return any(names_match(c.get("case_name", ""), target) for c in citations or [])

    lost = earlier_only = judge_skipped = never = 0
    for row in rows:
        target = row["target_case"]
        if not cited(row.get("pool_citations"), target) or cited(row.get("answer_citations"), target):
            continue
        lost += 1
        tiers = [[v.get("tier") for v in (r.get("citation_check") or {}).values()
                  if names_match((v.get("citation") or {}).get("case_name", ""), target)] for r in row["trace"]["rounds"]]
        any_round = any("SUPPORTED" in t for t in tiers)
        last_round = bool(tiers) and "SUPPORTED" in tiers[-1]
        if last_round:
            judge_skipped += 1
        elif any_round:
            earlier_only += 1
        else:
            never += 1
    put("LossLost", lost)
    put("LossSupportedEarlierOnly", earlier_only)
    put("LossSupportedLastNotUsed", judge_skipped)
    put("LossNeverSupported", never)

    available_unused = 0
    for row in rows:
        last = (row["trace"]["rounds"][-1].get("citation_check") or {}) if row["trace"]["rounds"] else {}
        if any(v.get("tier") == "SUPPORTED" for v in last.values()) and not row.get("answer_citations"):
            available_unused += 1
    put("LossAnswersEmptyAlthoughSupported", available_unused)

    strengths = [float(r["defense"]["challenge_strength"]) for row in rows for r in row["trace"]["rounds"]
                 if (r.get("defense") or {}).get("challenge_strength") is not None]
    put("LossRounds", len(strengths))
    put("LossChallengeAboveThreshold", sum(1 for s in strengths if s > 0.5))
    put("LossChallengeModeShare", pct(max(Counter(round(s, 1) for s in strengths).values()) / len(strengths), 1) if strengths else "--")
    put("LossChallengeMode", f"{Counter(round(s, 1) for s in strengths).most_common(1)[0][0]:.1f}" if strengths else "--")
    scores = [float(v.get("ccs") or 0.0) for row in rows for r in row["trace"]["rounds"] for v in (r.get("citation_check") or {}).values()
              if v.get("tier") == "SUPPORTED"]
    put("LossSupportedBelowJudgeThreshold", pct(sum(1 for s in scores if s < 0.75) / len(scores), 1) if scores else "--")
    put("LossSupportedCitations", len(scores))
    avoid = [(row["trace"].get("memory") or {}).get("avoid_list") or [] for row in rows]
    put("LossAvoidDebates", sum(1 for names in avoid if names))
    put("LossAvoidNames", len({name for names in avoid for name in names}))
    put("LossAvoidOwnTarget", sum(1 for row, names in zip(rows, avoid) if any(names_match(n, row["target_case"]) for n in names)))


PA_RUNS = [("BaseFour", "baseline_4x300"), ("BaseEight", "baseline_8x500"), ("Old", "e0_legacy_checker_no_memory"),
           ("Full", "e4_repaired_Full"), ("AOne", "e4_repaired_A1"), ("ATwo", "e4_repaired_A2"),
           ("AThree", "e4_repaired_A3"), ("AFour", "e4_repaired_A4")]
PA_MEASURES = [("TargetAnswer", "target cited by the answer", "rate"), ("TargetAny", "target cited by any agent", "rate"),
               ("CitesAnswer", "citations the answer rests on", "count"), ("CitesAll", "citations made by all agents", "count"),
               ("NoAuthority", "answer rests on no existing authority", "rate"),
               ("Sim", "answer similarity (sentence embedding)", "score")]
PA_COMPARISONS = [("FullBaseFour", "e4_repaired_Full", "baseline_4x300"), ("FullBaseEight", "e4_repaired_Full", "baseline_8x500"),
                  ("FullOld", "e4_repaired_Full", "e0_legacy_checker_no_memory"), ("OldBaseFour", "e0_legacy_checker_no_memory", "baseline_4x300"),
                  ("OldBaseEight", "e0_legacy_checker_no_memory", "baseline_8x500"), ("BaseEightBaseFour", "baseline_8x500", "baseline_4x300"),
                  ("FullAOne", "e4_repaired_Full", "e4_repaired_A1"), ("FullATwo", "e4_repaired_Full", "e4_repaired_A2"),
                  ("FullAThree", "e4_repaired_Full", "e4_repaired_A3"), ("FullAFour", "e4_repaired_Full", "e4_repaired_A4")]
SEEDS_PLANNED = 3
NUMBER_WORD = {0: "no", 1: "one", 2: "two", 3: "three"}


def show(value: float, kind: str) -> str:
    return pct(value, 1) if kind == "rate" else (f"{value:.2f}" if kind == "count" else f"{value:.3f}")


def show_difference(value: float, kind: str) -> str:
    return signed(100 * value, 1) if kind == "rate" else signed(value, 2 if kind == "count" else 3)


def post_audit_values(put) -> list:
    """The seeded, traced runs of data/results/v2/. Returns the statements the paper's prose depends on."""
    scores = load_json(os.path.join(V2_DIR, "scores.json"))
    put("PaScoresWritten", scores["written"].replace("T", " ")[:16])
    put("PaSeedsPlanned", SEEDS_PLANNED)
    put("PaSimFloor", f"{scores['similarity_metric']['floor_holding_of_another_case']:.3f}")
    put("PaSimQuestion", f"{scores['similarity_metric']['the_question_itself']:.3f}")

    means, seeds_of = {}, {}
    for tag, experiment in PA_RUNS:
        complete = {seed: run for seed, run in sorted(scores["runs"].get(experiment, {}).items()) if run.get("complete")}
        seeds_of[tag] = len(complete)
        put(f"Pa{tag}Seeds", len(complete))
        put(f"Pa{tag}SeedsWord", NUMBER_WORD.get(len(complete), str(len(complete))))
        runs = list(complete.values())
        row = {}
        for name, key, kind in PA_MEASURES:
            values = [run["measures"][key] for run in runs]
            row[name] = mean(values) if values else None
            put(f"Pa{tag}{name}", show(row[name], kind) if values else "--")
            if kind == "rate":
                spread = (sum((v - mean(values)) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) >= 2 else None
                put(f"Pa{tag}{name}SD", f"{100 * spread:.1f}" if spread is not None else "--")
        means[tag] = row
        put(f"Pa{tag}Skipped", sum(run["skipped_debates"] for run in runs) if runs else "--")
        put(f"Pa{tag}NotFound", sum(run["pool"]["citations_not_found"] for run in runs) if runs else "--")
        put(f"Pa{tag}NotFoundAnswer", sum(run["answer"]["citations_not_found"] for run in runs) if runs else "--")
        put(f"Pa{tag}CitationsMade", thousands(sum(run["pool"]["citations"] for run in runs)) if runs else "--")
        confidence = [run["mean_confidence"] for run in runs if run["mean_confidence"] is not None]
        put(f"Pa{tag}Conf", f"{mean(confidence):.2f}" if confidence else "--")
        put(f"Pa{tag}Rounds", f"{mean(run['mean_rounds'] for run in runs):.2f}" if runs and any(run["mean_rounds"] for run in runs) else "--")
        put(f"Pa{tag}Minutes", f"{mean(run['minutes'] for run in runs):.0f}" if runs else "--")
        put(f"Pa{tag}SecondsPerCase", f"{mean(60 * run['minutes'] / run['cases'] for run in runs):.0f}" if runs else "--")
        shown = [run["holding_support"]["support_shown_rate"] for run in runs if run["holding_support"]["support_shown_rate"] is not None]
        put(f"Pa{tag}SupportShown", pct(mean(shown), 1) if shown else "--")
        put(f"Pa{tag}SupportNotShown", pct(1 - mean(shown), 1) if shown else "--")
        put(f"Pa{tag}SupportChecked", thousands(sum(run["holding_support"]["citations_checked"] for run in runs)) if shown else "--")
        put(f"Pa{tag}LegacyHR", pct(mean(run["measures"]["legacy: fabricated-name rate"] for run in runs)) if runs else "--")
        row["NotFound"] = sum(run["pool"]["citations_not_found"] for run in runs)
        row["NotFoundAnswer"] = sum(run["answer"]["citations_not_found"] for run in runs)
        row["CitationsMade"] = sum(run["pool"]["citations"] for run in runs)
        row["Skipped"] = sum(run["skipped_debates"] for run in runs)
        row["SecondsPerCase"] = mean(60 * run["minutes"] / run["cases"] for run in runs) if runs else None
        row["Rounds"] = mean(run["mean_rounds"] for run in runs) if runs else None
    if means["Full"]["SecondsPerCase"] and means["BaseFour"]["SecondsPerCase"]:
        put("PaFullTimeRatio", f"{means['Full']['SecondsPerCase'] / means['BaseFour']['SecondsPerCase']:.0f}")
    debate_seeds = [seeds_of[tag] for tag, _ in PA_RUNS if not tag.startswith("Base")]
    put("PaDebateSeedsMin", min(debate_seeds))
    put("PaDebateSeedsMax", max(debate_seeds))
    low_word, high_word = NUMBER_WORD.get(min(debate_seeds), str(min(debate_seeds))), NUMBER_WORD.get(max(debate_seeds), str(max(debate_seeds)))
    put("PaDebateSeedsPhrase", (low_word if low_word == high_word else f"{low_word} to {high_word}") + (" seed" if max(debate_seeds) == 1 else " seeds"))
    put("PaComplete", "1" if min(seeds_of.values()) >= SEEDS_PLANNED else "0")
    put("PaCitationsNotFoundTotal", sum(means[tag]["NotFound"] for tag, _ in PA_RUNS))
    put("PaCitationsMadeTotal", thousands(sum(means[tag]["CitationsMade"] for tag, _ in PA_RUNS)))
    put("PaMainNotFound", sum(means[tag]["NotFound"] for tag in ("BaseFour", "BaseEight", "Old", "Full")))
    ablation_seeds = [seeds_of[tag] for tag in ("Full", "AOne", "ATwo", "AThree", "AFour")]
    low_word, high_word = NUMBER_WORD.get(min(ablation_seeds), str(min(ablation_seeds))), NUMBER_WORD.get(max(ablation_seeds), str(max(ablation_seeds)))
    put("PaAblationSeedsPhrase", (low_word if low_word == high_word else f"{low_word} to {high_word}") + (" seed" if max(ablation_seeds) == 1 else " seeds"))
    if means["Full"]["TargetAny"] is not None:
        put("PaFullFoundNotUsed", f"{100 * (means['Full']['TargetAny'] - means['Full']['TargetAnswer']):.1f}")
        put("PaOldFoundNotUsed", f"{100 * (means['Old']['TargetAny'] - means['Old']['TargetAnswer']):.1f}")

    significant = {}
    for tag, a, b in PA_COMPARISONS:
        block = scores["paired_comparisons"].get(f"{a} vs {b}")
        for name, key, kind in PA_MEASURES:
            result = block.get(key) if block else None
            base = f"Cmp{tag}{name}"
            put(f"{base}Diff", show_difference(result["difference"], kind) if result else "--")
            put(f"{base}Low", show_difference(result["ci95"][0], kind) if result else "--")
            put(f"{base}High", show_difference(result["ci95"][1], kind) if result else "--")
            put(f"{base}P", p_text(result["p"]) if result else "--")
            put(f"{base}Mark", r"$^{\ast}$" if result and result["significant"] else "")
            put(f"{base}Word", ("significant" if result["significant"] else "not significant") if result else "not available")
            significant[(tag, name)] = (bool(result["significant"]), result["difference"]) if result else (None, None)

    # Every statement below is made in the running text. A change of a result file that makes one of
    # them false is reported, so that the sentence is rewritten and not left standing beside new numbers.
    def holds(tag, name, direction):
        flag, difference = significant[(tag, name)]
        if flag is None:
            return False
        if direction == "none":
            return not flag
        return flag and (difference > 0 if direction == "up" else difference < 0)

    debates = ("Old", "Full", "AOne", "ATwo", "AThree", "AFour")
    return [
        (holds("FullOld", "TargetAnswer", "up"), "the repaired system rests on the target significantly more often than the first system"),
        (holds("FullOld", "NoAuthority", "down"), "the repaired system leaves significantly fewer answers without an existing authority than the first system"),
        (holds("FullOld", "TargetAny", "none"), "the repair does not change how often the target is cited somewhere in the debate"),
        (holds("FullBaseFour", "TargetAnswer", "none"), "repaired system against the four-passage baseline: target cited by the answer, no significant difference"),
        (holds("FullBaseFour", "TargetAny", "up"), "the debate cites the target somewhere significantly more often than the four-passage baseline"),
        (holds("FullBaseFour", "Sim", "down"), "the repaired system has significantly lower answer similarity than the four-passage baseline"),
        (holds("FullBaseFour", "NoAuthority", "up"), "the repaired system leaves significantly more answers without authority than the four-passage baseline"),
        (holds("FullBaseEight", "TargetAnswer", "down"), "the repaired system rests on the target significantly less often than the eight-passage baseline"),
        (holds("OldBaseFour", "TargetAnswer", "down"), "the first system rests on the target significantly less often than the four-passage baseline"),
        (holds("BaseEightBaseFour", "TargetAnswer", "none"), "the two baselines do not differ significantly in answers resting on the target"),
        (all(holds(f"FullA{word}", "TargetAnswer", "none") for word in ("One", "Two", "Three", "Four")),
         "no ablation variant differs significantly from the full system in target cited by the answer"),
        (means["AOne"]["TargetAnswer"] is not None and means["AOne"]["TargetAnswer"] == max(means[t]["TargetAnswer"] for t in ("Full", "AOne", "ATwo", "AThree", "AFour")),
         "the variant without the defense has the highest share of answers resting on the target"),
        (all(means[t].get("NotFoundAnswer", 0) == 0 for t in debates), "no final answer of a debate of the first queue rests on a citation to a non-existent case"),
        (means["BaseEight"].get("NotFoundAnswer", 0) > 0 and means["BaseFour"].get("NotFoundAnswer", 0) == 0,
         "eight-passage baseline answers cite non-existent cases and four-passage baseline answers do not"),
        (means["Old"].get("NotFound", 0) > 0 and means["ATwo"].get("NotFound", 0) > 0,
         "agents cited a non-existent case both in the first debate (checker active) and in the variant without the checker"),
        (seeds_of["Old"] >= 2 and seeds_of["BaseFour"] >= 2, "seed-to-seed standard deviations are available for the four-passage baseline and the first debate"),
        (all(means[t].get("Skipped", 0) == 0 for t, _ in PA_RUNS), "no debate was skipped in any post-audit run"),
        (means["Full"]["Rounds"] is not None and abs(means["Full"]["Rounds"] - 3.0) < 0.005, "every debate of the repaired system ran the maximum number of rounds"),
        (all(holds("FullAThree", name, "none") for name, _, _ in PA_MEASURES), "removing the memory changes no measure significantly"),
        (holds("FullAOne", "TargetAnswer", "none"), "the variant without the defense does not differ significantly from the full system"),
        (all(means[t]["Sim"] is not None and means[t]["Sim"] < min(means["BaseFour"]["Sim"], means["BaseEight"]["Sim"]) for t in ("Old", "Full")),
         "both debate configurations have lower answer similarity than both baselines"),
    ]


FINAL_RUNS = [("Final", "f2_final_Full"), ("BaseFourR", "baseline_r7_4x300"), ("BaseEightR", "baseline_r7_8x500"),
              ("LedgerOnly", "f1_ledger_Full"), ("RetrievalOnly", "f3_retrieval_Full")]
FINAL_COMPARISONS = [("FinalBaseFourR", "f2_final_Full", "baseline_r7_4x300"), ("FinalBaseEightR", "f2_final_Full", "baseline_r7_8x500"),
                     ("FinalFull", "f2_final_Full", "e4_repaired_Full"), ("FinalBaseFour", "f2_final_Full", "baseline_4x300"),
                     ("LedgerOnlyFull", "f1_ledger_Full", "e4_repaired_Full"), ("RetrievalOnlyFull", "f3_retrieval_Full", "e4_repaired_Full"),
                     ("FinalLedgerOnly", "f2_final_Full", "f1_ledger_Full"), ("FinalRetrievalOnly", "f2_final_Full", "f3_retrieval_Full"),
                     ("BaseFourRBaseFour", "baseline_r7_4x300", "baseline_4x300"), ("BaseEightRBaseEight", "baseline_r7_8x500", "baseline_8x500"),
                     ("BaseEightRBaseFourR", "baseline_r7_8x500", "baseline_r7_4x300")]
# The standard measures (experiments/v2/standard_metrics.py): macro name, key in the report, kind.
STANDARD_MEASURES = [
    ("HitOne", "hit_at_1", "rate"), ("HitFour", "hit_at_4", "rate"), ("HitEight", "hit_at_8", "rate"), ("MRR", "reciprocal_rank", "score"),
    ("TargetInContext", "target_in_context", "rate"), ("AuthorityAccuracy", "authority_accuracy", "rate"),
    ("AnswersFabricated", "has_fabricated", "rate"), ("CitationsFabricated", "fabricated_citation_rate", "rate"),
    ("CitationPrecision", "citation_precision", "rate"), ("CitationRecall", "citation_recall", "rate"),
    ("Grounded", "grounded", "rate"), ("Misgrounded", "misgrounded", "rate"), ("NoAuthority", "no_authority", "rate"),
    ("RougeL", "rouge_l", "score"), ("BertScore", "bertscore_f1", "score"), ("TokenF", "token_f1", "score"),
    ("Seconds", "seconds", "number"), ("Calls", "llm_calls", "number"),
]
STANDARD_PAIRED = [("AuthorityAccuracy", "authority_accuracy", "rate"), ("AnswersFabricated", "has_fabricated", "rate"),
                   ("Grounded", "grounded", "rate"), ("Misgrounded", "misgrounded", "rate"),
                   ("NoAuthority", "no_authority", "rate"), ("SentenceSupport", "sentence_support", "rate"),
                   ("RougeL", "rouge_l", "score"), ("BertScore", "bertscore_f1", "score"), ("TokenF", "token_f1", "score")]
STANDARD_SYSTEMS = [("BaseFour", "baseline_4x300"), ("BaseEight", "baseline_8x500"), ("Old", "e0_legacy_checker_no_memory"),
                    ("Full", "e4_repaired_Full"), ("BaseFourR", "baseline_r7_4x300"), ("BaseEightR", "baseline_r7_8x500"),
                    ("Final", "f2_final_Full"), ("LedgerOnly", "f1_ledger_Full"), ("RetrievalOnly", "f3_retrieval_Full")]
PERCENT = chr(92) + "%"
STANDARD_TABLE = [("hit_at_1", f"Retrieval Hit@1 ({PERCENT})", "rate"), ("hit_at_8", f"Retrieval Hit@8 ({PERCENT})", "rate"),
                  ("reciprocal_rank", "Retrieval MRR@8", "score"),
                  ("authority_accuracy", f"Authority accuracy ({PERCENT})", "rate"),
                  ("fabricated_citation_rate", f"Fabricated citations ({PERCENT} of citations)", "rate"),
                  ("has_fabricated", f"Answers with a fabricated citation ({PERCENT})", "rate"),
                  ("citation_precision", f"Citation precision ({PERCENT})", "rate"), ("citation_recall", f"Citation recall ({PERCENT})", "rate"),
                  ("grounded", f"Answers grounded ({PERCENT})", "rate"), ("misgrounded", f"Answers misgrounded ({PERCENT})", "rate"),
                  ("no_authority", f"Answers without authority ({PERCENT})", "rate"),
                  ("rouge_l", "ROUGE-L F1", "score"), ("bertscore_f1", "BERTScore F1", "score"), ("token_f1", "Token F1", "score"),
                  ("seconds", "Seconds per question", "number"), ("llm_calls", "Generator calls per question", "number")]


def show_kind(value, kind: str) -> str:
    if value is None:
        return "--"
    return pct(value, 1) if kind == "rate" else (f"{value:.3f}" if kind == "score" else f"{value:.1f}")


def difference_kind(value, kind: str) -> str:
    return signed(100 * value, 1) if kind == "rate" else signed(value, 3 if kind == "score" else 1)


def comparison_macros(put, prefix: str, block, measures, significant: dict, tag: str) -> None:
    for name, key, kind in measures:
        result = block.get(key) if block else None
        base = f"{prefix}{tag}{name}"
        put(f"{base}Diff", difference_kind(result["difference"], kind) if result else "--")
        put(f"{base}Low", difference_kind(result["ci95"][0], kind) if result else "--")
        put(f"{base}High", difference_kind(result["ci95"][1], kind) if result else "--")
        put(f"{base}P", p_text(result["p"]) if result else "--")
        put(f"{base}Mark", "$^{" + chr(92) + "ast}$" if result and result["significant"] else "")
        put(f"{base}Word", ("significant" if result["significant"] else "not significant") if result else "not available")
        significant[(prefix, tag, name)] = (bool(result["significant"]), result["difference"]) if result else (None, None)


def final_values(put) -> list:
    """The system after repairs 6 and 7, its baselines and the two attribution runs; and the
    standard measures of every system. Returns the statements the prose makes about them."""
    scores = load_json(os.path.join(V2_DIR, "scores.json"))
    means, seeds_of = {}, {}
    for tag, experiment in FINAL_RUNS:
        runs = [run for _, run in sorted(scores["runs"].get(experiment, {}).items()) if run.get("complete")]
        seeds_of[tag] = len(runs)
        put(f"Pa{tag}Seeds", len(runs))
        row = {}
        for name, key, kind in PA_MEASURES:
            values = [run["measures"][key] for run in runs]
            row[name] = mean(values) if values else None
            put(f"Pa{tag}{name}", show(row[name], kind) if values else "--")
            if kind == "rate":
                spread = (sum((v - mean(values)) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) >= 2 else None
                put(f"Pa{tag}{name}SD", f"{100 * spread:.1f}" if spread is not None else "--")
                put(f"Pa{tag}{name}Min", pct(min(values), 1) if values else "--")
                put(f"Pa{tag}{name}Max", pct(max(values), 1) if values else "--")
        row["NotFound"] = sum(run["pool"]["citations_not_found"] for run in runs)
        row["NotFoundAnswer"] = sum(run["answer"]["citations_not_found"] for run in runs)
        row["CitationsMade"] = sum(run["pool"]["citations"] for run in runs)
        row["Skipped"] = sum(run["skipped_debates"] for run in runs)
        row["Rounds"] = mean(run["mean_rounds"] for run in runs) if runs else None
        row["SecondsPerCase"] = mean(60 * run["minutes"] / run["cases"] for run in runs) if runs else None
        put(f"Pa{tag}NotFound", row["NotFound"] if runs else "--")
        put(f"Pa{tag}NotFoundAnswer", row["NotFoundAnswer"] if runs else "--")
        put(f"Pa{tag}CitationsMade", thousands(row["CitationsMade"]) if runs else "--")
        confidence = [run["mean_confidence"] for run in runs if run["mean_confidence"] is not None]
        put(f"Pa{tag}Conf", f"{mean(confidence):.2f}" if confidence else "--")
        put(f"Pa{tag}Rounds", f"{row['Rounds']:.2f}" if runs and row["Rounds"] else "--")
        put(f"Pa{tag}Minutes", f"{mean(run['minutes'] for run in runs):.0f}" if runs else "--")
        put(f"Pa{tag}SecondsPerCase", f"{row['SecondsPerCase']:.0f}" if runs else "--")
        shown = [run["holding_support"]["support_shown_rate"] for run in runs if run["holding_support"]["support_shown_rate"] is not None]
        put(f"Pa{tag}SupportShown", pct(mean(shown), 1) if shown else "--")
        means[tag] = row
    put("FinalAvailable", "1" if seeds_of["Final"] >= 1 else "0")
    put("FinalComplete", "1" if min(seeds_of["Final"], seeds_of["BaseFourR"], seeds_of["BaseEightR"]) >= SEEDS_PLANNED else "0")
    put("FinalSeedsPhrase", NUMBER_WORD.get(seeds_of["Final"], str(seeds_of["Final"])) + (" seed" if seeds_of["Final"] == 1 else " seeds"))
    if means["Final"]["TargetAny"] is not None:
        put("PaFinalFoundNotUsed", f"{100 * (means['Final']['TargetAny'] - means['Final']['TargetAnswer']):.1f}")
    if means["Final"]["SecondsPerCase"] and means["BaseFourR"]["SecondsPerCase"]:
        put("PaFinalTimeRatio", f"{means['Final']['SecondsPerCase'] / means['BaseFourR']['SecondsPerCase']:.0f}")

    significant = {}
    for tag, a, b in FINAL_COMPARISONS:
        comparison_macros(put, "Cmp", scores["paired_comparisons"].get(f"{a} vs {b}"), PA_MEASURES, significant, tag)

    # ── standard measures ───────────────────────────────────────────────────
    path = os.path.join(V2_DIR, "standard_metrics.json")
    standard = load_json(path) if os.path.exists(path) else {"experiments": {}, "paired_comparisons": {}, "written": "", "support_model": {}}
    put("StdAvailable", "1" if standard["experiments"] else "0")
    put("StdWritten", standard["written"].replace("T", " ")[:16] if standard["written"] else "--")
    for tag, experiment in STANDARD_SYSTEMS:
        entry = standard["experiments"].get(experiment)
        put(f"Std{tag}Seeds", len(entry["seeds"]) if entry else 0)
        for name, key, kind in STANDARD_MEASURES:
            put(f"Std{tag}{name}", show_kind(entry["mean"].get(key), kind) if entry else "--")
        put(f"Std{tag}Citations", thousands(entry["citations"]) if entry else "--")
        put(f"Std{tag}FabricatedCount", entry["citations_fabricated"] if entry else "--")
    extra = [("FullBaseFour", "e4_repaired_Full", "baseline_4x300"), ("FullBaseEight", "e4_repaired_Full", "baseline_8x500"), ("FullOld", "e4_repaired_Full", "e0_legacy_checker_no_memory"),
             ("OldBaseFour", "e0_legacy_checker_no_memory", "baseline_4x300"), ("FinalOld", "f2_final_Full", "e0_legacy_checker_no_memory")]
    for tag, a, b in list(FINAL_COMPARISONS) + extra:
        comparison_macros(put, "StdCmp", standard["paired_comparisons"].get(f"{a} vs {b}"), STANDARD_PAIRED, significant, tag)
    # The table: one row per measure, one column per system (the first seven systems).
    columns = STANDARD_SYSTEMS[:7]
    rows = [["Seeds"] + [str(len(standard["experiments"][e]["seeds"])) if e in standard["experiments"] else "--" for _, e in columns]]
    for key, title, kind in STANDARD_TABLE:
        rows.append([title] + [show_kind((standard["experiments"].get(e) or {"mean": {}})["mean"].get(key), kind) for _, e in columns])
    write_rows("Standard", rows)

    def holds(prefix, tag, name, direction):
        flag, difference = significant.get((prefix, tag, name), (None, None))
        if flag is None:
            return False
        if direction == "none":
            return not flag
        return flag and (difference > 0 if direction == "up" else difference < 0)

    if seeds_of["Final"] < 1:
        return []
    return [
        (holds("Cmp", "FinalBaseFourR", "TargetAnswer", "down"), "the final system rests on the target significantly less often than the four-passage baseline on the same retrieval"),
        (holds("Cmp", "FinalBaseEightR", "TargetAnswer", "down"), "the final system rests on the target significantly less often than the eight-passage baseline on the same retrieval"),
        (holds("Cmp", "FinalBaseFourR", "TargetAny", "none"), "final system against the four-passage baseline: target cited by any agent, no significant difference"),
        (holds("Cmp", "FinalFull", "TargetAnswer", "none"), "the final system and the repaired system do not differ significantly in answers resting on the target"),
        (holds("Cmp", "FinalBaseFour", "TargetAnswer", "none"), "final system against the four-passage baseline on the earlier retrieval: no significant difference"),
        (holds("Cmp", "BaseFourRBaseFour", "TargetAnswer", "up"), "the new retrieval raises the four-passage baseline significantly"),
        (holds("Cmp", "LedgerOnlyFull", "TargetAnswer", "none") and holds("Cmp", "RetrievalOnlyFull", "TargetAnswer", "none"),
         "neither repair alone changes answers resting on the target significantly"),
        (holds("Cmp", "RetrievalOnlyFull", "TargetAny", "up"), "the new retrieval alone raises target cited by any agent significantly"),
        (holds("Cmp", "LedgerOnlyFull", "TargetAny", "down"), "the ledger protocol alone lowers target cited by any agent significantly"),
        (means["Final"]["NotFoundAnswer"] == 0 and means["Final"]["NotFound"] == 0, "no agent of the final system cited a non-existent case"),
        (means["BaseEightR"]["NotFoundAnswer"] > 0, "eight-passage baseline answers on the new retrieval cite non-existent cases"),
        (means["Final"]["Rounds"] is not None and means["Final"]["Rounds"] < 1.05, "debates of the final system end after one round"),
        (holds("StdCmp", "FinalBaseFourR", "Grounded", "down") and holds("StdCmp", "FinalBaseFourR", "SentenceSupport", "down"),
         "the final system has significantly fewer grounded answers and lower citation recall than the four-passage baseline on the same retrieval"),
        (holds("StdCmp", "FinalBaseFourR", "RougeL", "down") and holds("StdCmp", "FinalBaseFourR", "BertScore", "down"),
         "the final system has significantly lower ROUGE-L and BERTScore than the four-passage baseline on the same retrieval"),
        (holds("StdCmp", "FinalOld", "AuthorityAccuracy", "up") and holds("StdCmp", "FinalOld", "Grounded", "up"),
         "the final system has significantly higher authority accuracy and more grounded answers than the first debate"),
        (holds("StdCmp", "FinalFull", "AuthorityAccuracy", "none") and holds("StdCmp", "FinalFull", "Grounded", "none"),
         "the final system and the repaired system do not differ significantly on the standard measures of authority and grounding"),
    ]


BASELINE_TAGS = [("BaseFour", "baseline_4x300"), ("BaseEight", "baseline_8x500"), ("BaseFourR", "baseline_r7_4x300"), ("BaseEightR", "baseline_r7_8x500")]


def correction_values(put) -> list:
    """The correction of the baseline's citation parsing, and where the final debate loses the target."""
    scores = load_json(os.path.join(V2_DIR, "scores.json"))
    changed_total = answers_total = 0
    for tag, experiment in BASELINE_TAGS:
        runs = [run for _, run in sorted(scores["runs"].get(experiment, {}).items()) if run.get("complete")]
        parsing = [run["baseline_citation_parsing"] for run in runs if run.get("baseline_citation_parsing")]
        if not parsing:
            for name in ("Changed", "Answers", "ChangedPct", "TargetAsSaved", "NoAuthorityAsSaved", "CitationsAsSaved", "CitationsRead"):
                put(f"Parse{tag}{name}", "--")
            continue
        changed, answers = sum(p["answers_whose_citations_changed"] for p in parsing), sum(p["answers"] for p in parsing)
        changed_total, answers_total = changed_total + changed, answers_total + answers
        put(f"Parse{tag}Changed", changed)
        put(f"Parse{tag}Answers", answers)
        put(f"Parse{tag}ChangedPct", pct(changed / answers, 0))
        put(f"Parse{tag}TargetAsSaved", pct(mean(p["as_saved_by_the_run"]["target_cited_rate"] for p in parsing), 1))
        put(f"Parse{tag}NoAuthorityAsSaved", pct(mean(p["as_saved_by_the_run"]["answers_without_existing_authority"] / p["answers"] for p in parsing), 1))
        put(f"Parse{tag}CitationsAsSaved", thousands(sum(p["citations_as_saved_by_the_run"] for p in parsing)))
        put(f"Parse{tag}CitationsRead", thousands(sum(p["citations_read_from_the_raw_output"] for p in parsing)))
        put(f"Pa{tag}NotFoundAnswerCount", sum(run["answer"]["citations_not_found"] for run in runs))
        put(f"Pa{tag}AnswerCitations", thousands(sum(run["answer"]["citations"] for run in runs)))
    put("ParseChangedTotal", thousands(changed_total))
    put("ParseAnswersTotal", thousands(answers_total))

    # Where the final system loses the target (all finished seeds of f2_final_Full).
    stats = Counter()
    directory = os.path.join(V2_DIR, "f2_final_Full")
    seeds = sorted(d for d in os.listdir(directory) if os.path.exists(os.path.join(directory, d, "summary.json"))) if os.path.isdir(directory) else []
    for seed in seeds:
        for row in read_cases("f2_final_Full", seed):
            target = row["target_case"]
            ledger = row.get("citation_ledger") or []
            entry = next((e for e in ledger if names_match(e.get("case_name", ""), target) or names_match(e.get("cited_as", ""), target)), None)
            in_answer = any(names_match(c.get("case_name", ""), target) for c in row.get("answer_citations") or [])
            in_pool = any(names_match(c.get("case_name", ""), target) for c in row.get("pool_citations") or [])
            supported = sum(1 for e in ledger if e.get("tier") == "SUPPORTED")
            stats["cases"] += 1
            stats["cited"] += in_pool
            stats["in_answer"] += in_answer
            stats["supported_decisions"] += supported
            stats["empty_although_supported"] += int(supported > 0 and not row.get("answer_citations"))
            if entry is not None and entry.get("tier") == "SUPPORTED":
                stats["target_supported"] += 1
            if in_pool and not in_answer:
                stats["lost"] += 1
                tier = entry.get("tier") if entry else None
                if tier == "SUPPORTED":
                    stats["lost_supported"] += 1
                    stats["lost_supported_other_case" if row.get("answer_citations") else "lost_supported_nothing"] += 1
                else:
                    stats["lost_not_supported"] += 1
    for name, key in (("Cases", "cases"), ("Cited", "cited"), ("InAnswer", "in_answer"), ("TargetSupported", "target_supported"), ("Lost", "lost"),
                      ("LostSupported", "lost_supported"), ("LostSupportedOtherCase", "lost_supported_other_case"),
                      ("LostSupportedNothing", "lost_supported_nothing"), ("LostNotSupported", "lost_not_supported"),
                      ("EmptyAlthoughSupported", "empty_although_supported")):
        put(f"FinalLoss{name}", stats[key] if seeds else "--")
    put("FinalLossSupportedPerDebate", f"{stats['supported_decisions'] / stats['cases']:.1f}" if stats["cases"] else "--")
    put("FinalLossTargetSupportedPct", pct(stats["target_supported"] / stats["cases"], 1) if stats["cases"] else "--")
    return [(not seeds or stats["lost_supported"] + stats["lost_not_supported"] == stats["lost"], "every lost target of the final system is classified")]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    macros = {}

    def put(name: str, value) -> None:
        if name in macros:
            raise KeyError(f"macro {name} defined twice")
        macros[name] = str(value)

    # ── Corpus ──────────────────────────────────────────────────────────────
    with open(CAP_DOWNLOAD_CHECKPOINT_PATH, "r", encoding="utf-8") as handle:
        download = json.load(handle)
    with open(CAP_CHUNK_CHECKPOINT_PATH, "r", encoding="utf-8") as handle:
        chunked = json.load(handle)
    with open(CORPUS_MANIFEST_PATH, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    pdf_volumes = set(manifest.get("volumes_from_official_us_reports_pdf", []))
    complete = [v for v, entry in download["volumes"].items() if entry["status"] == "COMPLETE"]
    cap_volumes = [v for v in complete if v not in pdf_volumes]
    cap_cases = sum(download["volumes"][v]["cases_count"] for v in cap_volumes)
    pdf_cases = sum(download["volumes"][v]["cases_count"] for v in pdf_volumes)
    chunk_total = sum(entry.get("chunk_count", 0) for entry in chunked["volumes"].values())
    if chunk_total != manifest["chunks"]:
        raise SystemExit(f"chunk ledger ({chunk_total}) and manifest ({manifest['chunks']}) disagree")
    catalog = CaseCatalog.load(CASE_CATALOG_PATH)
    reports = list(catalog.reports) if hasattr(catalog, "reports") else []
    opinions = [r for r in reports if r.is_opinion]
    years = [r.year for r in reports if r.year]
    chroma_bytes = sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(CHROMA_CORPUS_PATH) for f in files)

    put("CorpusVolumesTarget", download["target_volume_count"])
    put("CorpusVolumes", len(complete))
    put("CorpusVolumesCap", len(cap_volumes))
    put("CorpusVolumesPdf", len(pdf_volumes))
    put("CorpusVolumesUnavailable", len(download["unavailable_volumes"]))
    put("CorpusCasesCap", thousands(cap_cases))
    put("CorpusCasesPdf", thousands(pdf_cases))
    put("CorpusCases", thousands(cap_cases + pdf_cases))
    put("CorpusChunks", thousands(chunk_total))
    put("CorpusReports", thousands(len(reports)))
    put("CorpusOpinions", thousands(len(opinions)))
    put("CorpusOrders", thousands(len(reports) - len(opinions)))
    put("CorpusYearFirst", min(years))
    put("CorpusYearLast", max(years))
    put("CorpusIndexGB", f"{chroma_bytes / 1e9:.1f}")
    put("CorpusCheckpointInterval", download["checkpoint_interval"])
    put("CorpusBatches", len(download["batches"]))
    put("CorpusFailedVolumes", len(download["failed_volumes"]))

    # ── Test set ────────────────────────────────────────────────────────────
    usable, excluded = split_benchmark_by_source_text()
    put("TestCasesAll", len(usable) + len(excluded))
    put("TestCases", len(usable))
    put("TestCasesExcluded", len(excluded))

    # ── Benchmark and baseline ──────────────────────────────────────────────
    bench = load("benchmark_100_cases_results.json")
    base = load("baseline_standard_rag_results.json")
    lex_cases = {c["case_id"]: c for c in bench["per_case"] if c.get("status") == "SUCCESS"}
    base_cases = {c["case_id"]: c for c in base["per_case"] if c.get("status") == "SUCCESS"}
    shared = sorted(set(lex_cases) & set(base_cases))
    if len(shared) != len(usable):
        raise SystemExit(f"benchmark/baseline cover {len(shared)} cases, the test set has {len(usable)}")

    for prefix, cases in (("Lex", lex_cases), ("Base", base_cases)):
        rows = [cases[i] for i in shared]
        put(f"{prefix}HR", pct(mean(r["hr"] for r in rows)))
        put(f"{prefix}ASS", f"{mean(r['ass'] for r in rows):.3f}")
        put(f"{prefix}ASSpct", pct(mean(r["ass"] for r in rows), 1))
        put(f"{prefix}AR", pct(mean(r["ar"] for r in rows), 1))
        put(f"{prefix}AP", pct(mean(r["ap"] for r in rows), 1))
        cites = sum(r["total_cites"] for r in rows)
        flagged = sum(r["fake_cites"] for r in rows)
        put(f"{prefix}Cites", cites)
        put(f"{prefix}CitesPerAnswer", f"{cites / len(rows):.2f}")
        put(f"{prefix}Flagged", flagged)
        put(f"{prefix}FlaggedCases", sum(1 for r in rows if r["fake_cites"]))
        put(f"{prefix}NoCiteAnswers", sum(1 for r in rows if r["total_cites"] == 0))
        put(f"{prefix}NoCitePct", pct(mean(1.0 if r["total_cites"] == 0 else 0.0 for r in rows), 1))

    for metric, digits, scale in (("hr", 2, 100), ("ass", 3, 1), ("ar", 1, 100), ("ap", 1, 100)):
        differences = [lex_cases[i][metric] - base_cases[i][metric] for i in shared]
        low, high = paired_bootstrap(differences)
        name = metric.upper()
        put(f"Diff{name}", signed(scale * mean(differences), digits))
        put(f"Diff{name}Low", signed(scale * low, digits))
        put(f"Diff{name}High", signed(scale * high, digits))
        put(f"Diff{name}Sig", "yes" if (low > 0 or high < 0) else "no")
        put(f"Diff{name}Mark", r"$^{\ast}$" if (low > 0 or high < 0) else "")

    rows = [lex_cases[i] for i in shared]
    aggregate = bench["aggregate"]
    put("LexCFR", pct(mean(r["cfr"] for r in rows)))
    put("LexOOR", pct(mean(r["oor"] for r in rows)))
    put("LexMAR", pct(mean(r["mar"] for r in rows), 1))
    put("LexUCR", pct(mean(r["ucr"] for r in rows), 1))
    put("LexHAA", pct(mean(r["haa"] for r in rows), 1))
    put("LexConf", f"{mean(r['judge_confidence'] for r in rows):.2f}")
    converged = sum(1 for r in rows if r["dcr_status"] == "CONVERGED")
    put("LexConverged", converged)
    put("LexDCR", pct(converged / len(rows), 1))
    rounds = [r["debate_rounds"] for r in rows]
    put("LexMaxRounds", max(rounds))
    put("LexRoundsOne", sum(1 for x in rounds if x == 1))
    put("LexRoundsMax", sum(1 for x in rounds if x == max(rounds)))
    put("LexRoundsMean", f"{mean(rounds):.2f}")
    put("LexSeconds", f"{aggregate['elapsed_seconds']:.0f}")
    put("LexMinutes", f"{aggregate['elapsed_seconds'] / 60:.0f}")
    put("LexSecondsPerCase", f"{aggregate['elapsed_seconds'] / len(rows):.1f}")
    low_conf = sum(1 for r in rows if r["judge_confidence"] <= 0.20)
    high_conf = sum(1 for r in rows if r["judge_confidence"] >= 0.70)
    put("LexConfLow", low_conf)
    put("LexConfHigh", high_conf)
    put("LexConfMid", len(rows) - low_conf - high_conf)

    # Judge-confidence histogram, ten bins of width 0.1.
    bins = [0] * 10
    for r in rows:
        bins[min(9, int(r["judge_confidence"] * 10))] += 1
    with open(os.path.join(OUT_DIR, "confidence_hist.dat"), "w", encoding="utf-8") as handle:
        handle.write("mid count\n")
        for index, count in enumerate(bins):
            if not count:
                continue  # an empty bin is simply no bar (and no "0" label)
            handle.write(f"{index / 10 + 0.05:.2f} {count}\n")

    # ── Citation stress test ────────────────────────────────────────────────
    for tag, filename in (("Stress", "citation_stress_test_results.json"), ("StressOld", "citation_stress_test_results_before_cce_fix.json")):
        stress = load(filename)
        counts = stress["per_type_counts"]
        put(f"{tag}Total", stress["total_evaluated"])
        for kind in STRESS_TYPES:
            entry = counts[kind]
            key = f"{tag}{STRESS_MACRO[kind]}"
            put(f"{key}N", entry["total"])
            put(f"{key}Accepted", entry["accepted"])
            put(f"{key}AcceptedPct", pct(entry["accepted"] / entry["total"], 1) if entry["total"] else "0.0")
            low, high = wilson(entry["accepted"], entry["total"])
            put(f"{key}AcceptedLow", pct(low, 1))
            put(f"{key}AcceptedHigh", pct(high, 1))
            for verdict in VERDICTS:
                put(f"{key}{verdict.title()}", entry.get(verdict, 0))
        if tag == "Stress":
            put("StressSeconds", f"{stress['elapsed_seconds']:.0f}")
            true_accepts = counts["correct"]["accepted"]
            false_accepts = sum(counts[k]["accepted"] for k in STRESS_TYPES if k != "correct")
            put("StressFalseAccepts", false_accepts)
            put("StressAcceptPrecision", pct(true_accepts / (true_accepts + false_accepts), 1))
            wrong = sum(counts[k]["total"] for k in STRESS_TYPES if k != "correct")
            put("StressWrongTotal", wrong)
            put("StressWrongRejectedPct", pct(1 - false_accepts / wrong, 1))
            # Area under the ROC curve of the support score, correct against swapped holdings: the
            # chance that a correct citation scores above a swapped one (comparable to detector AUCs
            # reported elsewhere).
            positive = [d["support"] for d in stress["details"] if d["type"] == "correct" and d.get("support") is not None]
            negative = [d["support"] for d in stress["details"] if d["type"] == "misattributed" and d.get("support") is not None]
            wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
            put("StressAUC", f"{wins / (len(positive) * len(negative)):.2f}")
            with open(os.path.join(OUT_DIR, "stress_verdicts.dat"), "w", encoding="utf-8") as handle:
                handle.write("row " + " ".join(v.lower() for v in VERDICTS) + "\n")
                for index, kind in enumerate(STRESS_TYPES):
                    entry = counts[kind]
                    handle.write(f"{index} " + " ".join(f"{100 * entry.get(v, 0) / entry['total']:.2f}" for v in VERDICTS) + "\n")

    # ── Which passages should the citation checker read? ────────────────────
    evidence_path = os.path.join(RESULTS_DIR, "cce_evidence_selection_study.json")
    evidence_names = {"dense_top3": "DenseThree", "dense_top5": "DenseFive", "lexical_top3": "LexicalThree",
                      "lexical_top5": "LexicalFive", "hybrid_top3": "HybridThree", "hybrid_top5": "HybridFive",
                      "whole_opinion": "Whole"}
    evidence = load("cce_evidence_selection_study.json") if os.path.exists(evidence_path) else None
    put("EvidenceComplete", "1" if evidence else "0")
    rows_evidence = {"Deployed": evidence["deployed_checker"]} if evidence else {"Deployed": None}
    for key, macro in evidence_names.items():
        rows_evidence[macro] = evidence["strategies"][key] if evidence else None
    for macro, row in rows_evidence.items():
        put(f"Ev{macro}Correct", row["correct_accepted"] if row else "--")
        put(f"Ev{macro}Swapped", row["swapped_accepted"] if row else "--")
        put(f"Ev{macro}Windows", f"{row['mean_windows_scored']:.0f}" if row and "mean_windows_scored" in row else "--")
    put("EvCorrectTotal", evidence["deployed_checker"]["correct_total"] if evidence else "--")
    put("EvSwappedTotal", evidence["deployed_checker"]["swapped_total"] if evidence else "--")
    if evidence:
        if evidence["strategies"]["dense_top3"] != {**evidence["deployed_checker"], "mean_windows_scored": evidence["strategies"]["dense_top3"]["mean_windows_scored"]}:
            raise SystemExit("evidence study: the dense top-3 row does not reproduce the deployed checker")
        few = [evidence["strategies"][k] for k in evidence_names if k != "whole_opinion"]
        whole = evidence["strategies"]["whole_opinion"]
        correct_total = evidence["deployed_checker"]["correct_total"]
        swapped_total = evidence["deployed_checker"]["swapped_total"]
        put("EvFewCorrectMin", min(r["correct_accepted"] for r in few))
        put("EvFewCorrectMax", max(r["correct_accepted"] for r in few))
        put("EvFewCorrectMaxPct", pct(max(r["correct_accepted"] for r in few) / correct_total, 1))
        put("EvFewSwappedMin", min(r["swapped_accepted"] for r in few))
        put("EvFewSwappedMax", max(r["swapped_accepted"] for r in few))
        put("EvWholeCorrectPct", pct(whole["correct_accepted"] / correct_total, 1))
        put("EvWholeSwappedPct", pct(whole["swapped_accepted"] / swapped_total, 1))
        put("EvWholeWindowRatio", f"{whole['mean_windows_scored'] / evidence['strategies']['dense_top3']['mean_windows_scored']:.0f}")

    # ── Ablation ────────────────────────────────────────────────────────────
    ablation = {row["variant"]: row for row in load("ablation_summary.json")}
    full_run = all(v in ablation and ablation[v]["completed_cases"] == len(usable) for v in VARIANT_ORDER)
    put("AblationComplete", "1" if full_run else "0")
    put("AblationCases", ablation["Full"]["completed_cases"] if "Full" in ablation else 0)
    word = {"Full": "Full", "A1": "AOne", "A2": "ATwo", "A3": "AThree", "A4": "AFour"}
    for variant in VARIANT_ORDER:
        row = ablation.get(variant, {"mean_hr": 0.0, "mean_cas": 0.0, "mean_confidence": 0.0, "completed_cases": 0, "crashed_cases": 0})
        put(f"Abl{word[variant]}HR", pct(row["mean_hr"]))
        put(f"Abl{word[variant]}ASS", f"{row['mean_cas']:.3f}")
        put(f"Abl{word[variant]}Conf", f"{row['mean_confidence']:.2f}")
        put(f"Abl{word[variant]}N", row["completed_cases"])
        put(f"Abl{word[variant]}Crashed", row["crashed_cases"])
    # Wall-clock time of each variant, from the timestamps in the run logs (a re-run log wins).
    minutes = {}
    for log_name in ("ablation_full.log", "ablation_A2_rerun.log"):
        minutes.update(variant_minutes(os.path.join(LOGS_DIR, log_name)))
    for variant in VARIANT_ORDER:
        put(f"Abl{word[variant]}Minutes", f"{minutes[variant]:.0f}" if variant in minutes else "--")
    values = [ablation[v] for v in VARIANT_ORDER if v in ablation]
    put("AblHRMin", pct(min(r["mean_hr"] for r in values)))
    put("AblHRMax", pct(max(r["mean_hr"] for r in values)))
    put("AblASSMin", f"{min(r['mean_cas'] for r in values):.3f}")
    put("AblASSMax", f"{max(r['mean_cas'] for r in values):.3f}")

    # What memory told the agents to avoid during the full-system run (from that run's log).
    avoid_lists = avoid_lists_in_full_run(os.path.join(LOGS_DIR, "ablation_full.log"))
    avoided = sorted({name for names in avoid_lists for name in names})
    in_corpus = [name for name in avoided if catalog.reports_named(name)]
    targets = [case["target_case"] for case in usable]
    put("MemAvoidDebates", len(avoid_lists) if avoid_lists else "--")
    put("MemAvoidNames", len(avoided) if avoid_lists else "--")
    put("MemAvoidInCorpus", len(in_corpus) if avoid_lists else "--")
    put("MemAvoidTargets", sum(1 for name in avoided if any(names_match(name, t) for t in targets)) if avoid_lists else "--")

    # The full system was measured twice (benchmark run and ablation run): the spread between the
    # two is the run-to-run variation against which ablation differences have to be read.
    if "Full" in ablation:
        put("RepeatHRGap", f"{abs(100 * ablation['Full']['mean_hr'] - 100 * mean(r['hr'] for r in rows)):.2f}")
        put("RepeatASSGap", f"{abs(ablation['Full']['mean_cas'] - mean(r['ass'] for r in rows)):.3f}")
        put("RepeatConfGap", f"{abs(ablation['Full']['mean_confidence'] - mean(r['judge_confidence'] for r in rows)):.2f}")

    # ── Audit of the first evaluation, and the experiments run after it ─────
    audit_values(put)
    component_values(put)
    loss_values(put)
    statements = post_audit_values(put)
    statements += final_values(put)
    statements += correction_values(put)

    with open(os.path.join(OUT_DIR, "results.tex"), "w", encoding="utf-8") as handle:
        handle.write("% Generated by paper/make_results.py from the saved result files. Do not edit by hand.\n")
        for name in sorted(macros):
            handle.write(f"\\newcommand{{\\Res{name}}}{{{macros[name]}}}\n")
    with open(os.path.join(OUT_DIR, "table_rows.tex"), "w", encoding="utf-8") as handle:
        handle.write("% Generated by paper/make_results.py from the saved result files. Do not edit by hand.\n")
        for name in sorted(TABLE_ROWS):
            handle.write(f"\\newcommand{{\\ResRows{name}}}{{%\n{TABLE_ROWS[name]}}}\n")
    with open(os.path.join(OUT_DIR, "claims_check.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"Statements the running text makes about the post-audit runs (scores written {macros['PaScoresWritten']}).\n")
        for ok, text in statements:
            handle.write(f"{'holds   ' if ok else 'REWRITE '} {text}\n")

    print(f"Wrote {len(macros)} values to {os.path.join(OUT_DIR, 'results.tex')}")
    for name in sorted(macros):
        print(f"  {name:28s} {macros[name]}")
    if not full_run:
        print("\nNOTE: the ablation summary does not cover the full test set yet; the paper marks that table as incomplete.")
    if macros["PaComplete"] == "0":
        print(f"\nNOTE: the post-audit runs have {macros['PaDebateSeedsMin']} to {macros['PaDebateSeedsMax']} of {SEEDS_PLANNED} seeds per debate configuration; the paper says so.")
    broken = [text for ok, text in statements if not ok]
    if broken:
        print("\nWARNING: the running text states the following, and the result files no longer support it. Rewrite the sentence:")
        for text in broken:
            print(f"  - {text}")


if __name__ == "__main__":
    main()
