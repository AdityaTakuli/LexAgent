# LexAgent v3.0 | experiments/forensics/analyze_existing_results.py
"""
Read-only forensic analysis of the EXISTING result files and run logs.

Nothing is generated, re-run or overwritten: no model is loaded, no LLM is called, and the only
file written is data/results/forensic_audit/analysis_existing_results.json (a new file).

What it does
  1. Inventories every result file (size, time, SHA-256) and checks the per-case files:
     96/96 case ids, duplicates, order, missing fields, same questions in every per-case file.
  2. Identifies verdict-cache hits: in the benchmark scorecards by their signature, and in the
     ablation logs directly (the logs record every hit).
  3. Reconstructs, per case and per ablation variant, what the logs recorded: cache hit, rounds,
     citation-check tiers per round, judge rejections, verdict stored, avoid-list size.
  4. Paired statistics, debate system against baseline, on all 96 cases and on the cases that
     were actually debated.
  5. Separability of the citation checker's support score (stress test and evidence study).
  6. Run-to-run variation of the fabricated-name rate.

    python experiments/forensics/analyze_existing_results.py
"""

import ast
import hashlib
import io
import contextlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import RESULTS_DIR  # noqa: E402

LOGS_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "logs")
OUT_DIR = os.path.join(RESULTS_DIR, "forensic_audit")
OUT_PATH = os.path.join(OUT_DIR, "analysis_existing_results.json")
SEED = 42
RESAMPLES = 10000
SCORECARD_FIELDS = ["case_id", "topic", "hr", "cfr", "oor", "mar", "ucr", "ass", "cas", "ar", "ap", "haa",
                    "dcr_status", "evidence_overlap_ratio", "judge_confidence", "debate_rounds",
                    "total_cites", "fake_cites", "fake_names", "status"]
RAW_TRACE_FIELDS = ["final_answer", "retrieved_passages", "prosecutor_citations", "defense_citations",
                    "citation_confidence_report", "judge_verdict", "judge_reasoning", "debate_history"]
TIERS = ["SUPPORTED", "UNCERTAIN", "MISATTRIBUTED", "FABRICATED", "UNVERIFIABLE"]


# ── small statistics helpers (no external state) ────────────────────────────
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return float("nan") if not n else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def sd(xs):
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def describe(xs):
    xs = list(xs)
    return {"n": len(xs), "mean": round(mean(xs), 5), "median": round(median(xs), 5),
            "sd": round(sd(xs), 5), "se": round(sd(xs) / math.sqrt(len(xs)), 5) if len(xs) > 1 else None}


def bootstrap_ci(diffs, seed=SEED, resamples=RESAMPLES):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 5), round(means[int(0.975 * resamples) - 1], 5)]


def sign_flip_p(diffs, seed=SEED, resamples=RESAMPLES):
    """Two-sided paired permutation (sign-flip) test on the mean difference."""
    rng = random.Random(seed)
    observed = abs(mean(diffs))
    hits = 0
    for _ in range(resamples):
        if abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)) >= observed - 1e-12:
            hits += 1
    return round((hits + 1) / (resamples + 1), 5)


def paired(a, b, binary=False):
    """Paired comparison a - b: descriptive statistics, bootstrap CI, tests and effect sizes."""
    from scipy import stats

    diffs = [x - y for x, y in zip(a, b)]
    out = {"a": describe(a), "b": describe(b), "difference": describe(diffs),
           "bootstrap_95ci_mean_difference": bootstrap_ci(diffs),
           "sign_flip_permutation_p": sign_flip_p(diffs)}
    nonzero = [d for d in diffs if d != 0]
    out["pairs_that_differ"] = len(nonzero)
    if len(nonzero) >= 1:
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_signed_rank_p"] = round(float(w.pvalue), 5)
        except ValueError:
            out["wilcoxon_signed_rank_p"] = None
        pos = sum(1 for d in nonzero if d > 0)
        out["sign_test_p"] = round(float(stats.binomtest(pos, len(nonzero), 0.5).pvalue), 5)
        # matched-pairs rank-biserial correlation
        ranks = stats.rankdata([abs(d) for d in nonzero])
        r_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
        r_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
        out["rank_biserial"] = round((r_plus - r_minus) / (r_plus + r_minus), 4)
    s = sd(diffs)
    out["cohens_dz"] = round(mean(diffs) / s, 4) if s and not math.isnan(s) and s > 0 else None
    if binary:
        b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        out["mcnemar"] = {"a_only": b10, "b_only": b01,
                          "exact_p": round(float(stats.binomtest(b10, b10 + b01, 0.5).pvalue), 5) if (b10 + b01) else 1.0}
    return out


def auc(pos, neg):
    """Probability that a random positive scores above a random negative (ties count half)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name):
    with open(os.path.join(RESULTS_DIR, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


# ── 1. inventory and integrity ──────────────────────────────────────────────
def inventory():
    files = {}
    for name in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, name)
        if os.path.isfile(path) and name.endswith(".json"):
            files[name] = {"bytes": os.path.getsize(path),
                           "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
                           "sha256": sha256_file(path)}
    return files


def benchmark_definition():
    from experiments.benchmark_cases import split_benchmark_by_source_text

    with contextlib.redirect_stdout(io.StringIO()):
        usable, excluded = split_benchmark_by_source_text()
    canonical = json.dumps([[c["id"], c["query"], c.get("target_case", ""), c.get("ground_truth_holding", "")] for c in usable],
                           ensure_ascii=False, sort_keys=True)
    return usable, excluded, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_per_case(name, rows, expected_ids):
    ids = [r.get("case_id") for r in rows]
    counts = Counter(ids)
    missing_fields = sorted({f for r in rows for f in SCORECARD_FIELDS if f not in r})
    return {
        "file": name,
        "rows": len(rows),
        "distinct_case_ids": len(counts),
        "duplicates": sorted(i for i, c in counts.items() if c > 1),
        "ids_missing_from_file": sorted(set(expected_ids) - set(ids)),
        "ids_not_in_benchmark": sorted(set(ids) - set(expected_ids)),
        "same_order_as_benchmark": ids == list(expected_ids),
        "missing_scorecard_fields": missing_fields,
        "statuses": dict(Counter(r.get("status") for r in rows)),
        "raw_trace_fields_present": [f for f in RAW_TRACE_FIELDS if any(f in r for r in rows)],
        "topics_match_benchmark": None,
    }


# ── 3. per-case reconstruction from the run logs ────────────────────────────
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
QUERY = re.compile(r"Dense retrieval: query='(.*?)\.\.\.' ->")
CCE = re.compile(r"Factorized CCE scored (\d+) citations: avg_CCS=([\d.]+) \| SUPPORTED=(\d+), UNCERTAIN=(\d+), "
                 r"MISATTRIBUTED=(\d+), FABRICATED=(\d+), UNVERIFIABLE=(\d+)")


def new_case():
    return {"cache_hit": False, "cache_distance": None, "rounds_cce": [], "ddc": [], "judge_rejections": [],
            "judge_ran": False, "verdict_stored": False, "risk_entries_written": 0, "avoid_list": None,
            "augmented_rounds": 0, "retrievals": 0}


def parse_variant_sections(log_path):
    """{variant: [log lines of that variant's section]} for one ablation log."""
    sections, current = {}, None
    if not os.path.exists(log_path):
        return sections
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            started = re.search(r"Evaluating Variant \[(\w+)\]", line)
            if started:
                current = started.group(1)
                sections[current] = []
                continue
            if current is None:
                continue
            if "-> Result:" in line:
                current = None
                continue
            sections[current].append(line.rstrip("\n"))
    return sections


def reconstruct_cases(lines, cases):
    """Assign the log lines of one variant to the benchmark cases, in order.

    A case begins either with a cache hit (the debate is skipped) or with its first retrieval,
    whose logged query prefix identifies it. The run order is the benchmark order.
    """
    prefixes = [c["query"][:50] for c in cases]
    records = [new_case() for _ in cases]
    index = -1
    problems = []
    pending_avoid = None
    for line in lines:
        if "Cache HIT: distance=" in line:
            index += 1
            if index >= len(records):
                problems.append("more case starts than benchmark cases")
                break
            records[index]["cache_hit"] = True
            records[index]["cache_distance"] = float(re.search(r"distance=([\d.]+)", line).group(1))
            continue
        if "High-risk citations loaded:" in line:
            found = re.search(r"High-risk citations loaded: (\[.*\])", line)
            try:
                pending_avoid = ast.literal_eval(found.group(1))
            except (AttributeError, SyntaxError, ValueError):
                pending_avoid = []
            # A hit case logs its list right after the hit; a debated case logs it before its first retrieval.
            if index >= 0 and records[index]["cache_hit"] and records[index]["avoid_list"] is None:
                records[index]["avoid_list"] = pending_avoid
                pending_avoid = None
            continue
        query = QUERY.search(line)
        if query:
            prefix = query.group(1)
            if index < 0 or prefix != prefixes[index]:
                nxt = index + 1
                if nxt < len(prefixes) and prefix == prefixes[nxt]:
                    index = nxt
                    if pending_avoid is not None:
                        records[index]["avoid_list"] = pending_avoid
                        pending_avoid = None
                elif index >= 0 and prefix == prefixes[index]:
                    pass
                else:
                    problems.append(f"unaligned query prefix after case index {index}: {prefix!r}")
                    continue
            records[index]["retrievals"] += 1
            continue
        if index < 0:
            continue
        record = records[index]
        scored = CCE.search(line)
        if scored:
            n, avg, *tiers = scored.groups()
            record["rounds_cce"].append({"citations": int(n), "avg_ccs": float(avg), **{t: int(v) for t, v in zip(TIERS, tiers)}})
        elif "src.novel.ddc" in line and "DDC:" in line:
            record["ddc"].append("CONTINUE" if "CONTINUE" in line else "TERMINATE")
        elif "Rejecting judge citation" in line:
            record["judge_rejections"].append(re.search(r"Rejecting judge citation '(.*)': not verified", line).group(1))
        elif "Judge node: rendering verdict" in line:
            record["judge_ran"] = True
        elif "Verdict cached:" in line:
            record["verdict_stored"] = True
        elif "Risk index updated:" in line:
            record["risk_entries_written"] += int(re.search(r"updated: (\d+)", line).group(1))
        elif "augmented query with reflection queries" in line:
            record["augmented_rounds"] += 1
    started = sum(1 for r in records if r["cache_hit"] or r["retrievals"])
    return records, {"cases_aligned": started, "alignment_problems": problems[:5], "alignment_problem_count": len(problems)}


def summarise_variant(records):
    debated = [r for r in records if not r["cache_hit"] and r["retrievals"]]
    hits = [r for r in records if r["cache_hit"]]
    rounds = [len(r["ddc"]) if r["ddc"] else (1 if r["judge_ran"] else 0) for r in debated]
    with_cce = [r for r in debated if r["rounds_cce"]]
    out = {
        "cases": len(records),
        "cache_hits": len(hits),
        "cache_hit_share": round(len(hits) / len(records), 4),
        "cache_hit_distance_min_max": [min((r["cache_distance"] for r in hits), default=None),
                                       max((r["cache_distance"] for r in hits), default=None)],
        "debated": len(debated),
        "debate_rounds_distribution": dict(sorted(Counter(rounds).items())),
        "mean_rounds_when_debated": round(mean(rounds), 3) if rounds else None,
        "verdicts_stored": sum(1 for r in debated if r["verdict_stored"]),
        "debates_with_judge_rejections": sum(1 for r in debated if r["judge_rejections"]),
        "judge_rejections_total": sum(len(r["judge_rejections"]) for r in debated),
        "cases_given_an_avoid_list": sum(1 for r in records if r["avoid_list"]),
    }
    if with_cce:
        first = [r["rounds_cce"][0] for r in with_cce]
        last = [r["rounds_cce"][-1] for r in with_cce]
        every = [c for r in with_cce for c in r["rounds_cce"]]
        out["citation_check"] = {
            "debates_with_citation_check": len(with_cce),
            "rounds_scored": len(every),
            "rounds_with_zero_citations": sum(1 for c in every if c["citations"] == 0),
            "citations_scored": sum(c["citations"] for c in every),
            "tier_totals": {t: sum(c[t] for c in every) for t in TIERS},
            "supported_in_first_round": sum(1 for c in first if c["SUPPORTED"] > 0),
            "supported_in_final_round": sum(1 for c in last if c["SUPPORTED"] > 0),
            "supported_in_any_round": sum(1 for r in with_cce if any(c["SUPPORTED"] > 0 for c in r["rounds_cce"])),
            "supported_earlier_but_not_in_final_round": sum(
                1 for r in with_cce if r["rounds_cce"][-1]["SUPPORTED"] == 0 and any(c["SUPPORTED"] > 0 for c in r["rounds_cce"][:-1])),
            "mean_avg_ccs_first_round": round(mean(c["avg_ccs"] for c in first), 4),
            "mean_avg_ccs_final_round": round(mean(c["avg_ccs"] for c in last), 4),
            "final_round_zero_citations": sum(1 for c in last if c["citations"] == 0),
            "continued_although_a_citation_was_supported": sum(
                1 for r in with_cce for c, d in zip(r["rounds_cce"], r["ddc"]) if d == "CONTINUE" and c["SUPPORTED"] > 0),
        }
        multi = [r for r in with_cce if len(r["rounds_cce"]) >= 2]
        if multi:
            first_m = [r["rounds_cce"][0]["avg_ccs"] for r in multi]
            last_m = [r["rounds_cce"][-1]["avg_ccs"] for r in multi]
            out["citation_check"]["multi_round_debates"] = len(multi)
            out["citation_check"]["avg_ccs_first_vs_final_in_multi_round"] = paired(last_m, first_m)
    return out


# ── main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "note": "Read-only analysis of existing files. No experiment was re-run and no result file was modified."}

    usable, excluded, bench_hash = benchmark_definition()
    expected_ids = [c["id"] for c in usable]
    by_id = {c["id"]: c for c in usable}
    report["benchmark"] = {"cases": len(usable), "excluded_ids": [c["id"] for c in excluded], "sha256": bench_hash}
    report["inventory"] = inventory()

    # ── per-case files ──
    bench = load("benchmark_100_cases_results.json")
    base = load("baseline_standard_rag_results.json")
    integrity = {}
    for name, data in (("benchmark_100_cases_results.json", bench), ("baseline_standard_rag_results.json", base)):
        check = check_per_case(name, data["per_case"], expected_ids)
        check["topics_match_benchmark"] = all(r.get("topic") == by_id[r["case_id"]]["topic"] for r in data["per_case"] if r.get("case_id") in by_id)
        check["aggregate_matches_recomputed_mean_hr"] = abs(data["aggregate"]["mean_hr"] - mean(r["hr"] for r in data["per_case"])) < 1e-9
        integrity[name] = check
    ablation = load("ablation_summary.json")
    integrity["ablation_summary.json"] = {
        "rows": len(ablation), "variants": [r["variant"] for r in ablation],
        "per_case_rows": 0,
        "note": "The ablation runner saved one mean per measure and variant. No per-case scorecards, answers, "
                "citations or citation-check reports exist for Full/A1/A2/A3/A4.",
        "completed_cases": {r["variant"]: r["completed_cases"] for r in ablation},
    }
    old_path = os.path.join(RESULTS_DIR, "ablation_summary_before_switch_fix.json")
    if os.path.exists(old_path):
        old = load("ablation_summary_before_switch_fix.json")
        integrity["ablation_summary_before_switch_fix.json"] = {
            "rows_identical_to_current_except_A2": all(o == n for o, n in zip(old, ablation) if o["variant"] != "A2"),
            "A2_before": next(r for r in old if r["variant"] == "A2"),
            "A2_after": next(r for r in ablation if r["variant"] == "A2")}
    report["integrity"] = integrity

    lex = {r["case_id"]: r for r in bench["per_case"]}
    bl = {r["case_id"]: r for r in base["per_case"]}
    ids = [i for i in expected_ids if i in lex and i in bl]

    # ── 2. cache hits in the benchmark run, by signature ──
    signature = [i for i in ids if lex[i]["judge_confidence"] == 0.0 and lex[i]["debate_rounds"] == 1 and lex[i]["total_cites"] == 0]
    one_round = [i for i in ids if lex[i]["debate_rounds"] == 1]
    no_cites = [i for i in ids if lex[i]["total_cites"] == 0]
    report["benchmark_cache_signature"] = {
        "definition": "judge_confidence == 0.0 and debate_rounds == 1 and total_cites == 0 (a skipped debate leaves the initial state)",
        "cases_with_signature": len(signature),
        "one_round_cases": len(one_round),
        "citation_free_cases": len(no_cites),
        "all_one_round_cases_have_signature": set(one_round) == set(signature),
        "all_citation_free_cases_have_signature": set(no_cites) == set(signature),
        "first_position_with_signature": (expected_ids.index(signature[0]) + 1) if signature else None,
        "signature_case_ids": signature,
        "other_fields_all_zero_in_signature_cases": all(
            lex[i]["mar"] == 0 and lex[i]["haa"] == 0 and lex[i]["ucr"] == 0 and lex[i]["hr"] == 0 for i in signature),
        "debated_rounds_distribution": dict(Counter(lex[i]["debate_rounds"] for i in ids if i not in set(signature))),
    }
    debated_ids = [i for i in ids if i not in set(signature)]

    # ── 4. paired statistics, debate system against baseline ──
    def compare(subset):
        out = {"n": len(subset)}
        for metric in ("hr", "ass", "ar", "ap"):
            out[metric] = paired([lex[i][metric] for i in subset], [bl[i][metric] for i in subset], binary=(metric == "ar"))
        out["citations_per_answer"] = {"lexagent": round(mean(lex[i]["total_cites"] for i in subset), 3),
                                       "baseline": round(mean(bl[i]["total_cites"] for i in subset), 3)}
        out["citation_free_answers"] = {"lexagent": sum(1 for i in subset if lex[i]["total_cites"] == 0),
                                        "baseline": sum(1 for i in subset if bl[i]["total_cites"] == 0)}
        out["flagged_names"] = {"lexagent": sum(lex[i]["fake_cites"] for i in subset), "baseline": sum(bl[i]["fake_cites"] for i in subset),
                                "lexagent_citations": sum(lex[i]["total_cites"] for i in subset),
                                "baseline_citations": sum(bl[i]["total_cites"] for i in subset)}
        return out

    report["lexagent_vs_baseline"] = {
        "all_96_cases": compare(ids),
        "debated_cases_only": compare(debated_ids),
        "cache_signature_cases_only": compare(signature),
        "caution": "The debated subset is not a random sample: a case is skipped when its question embedding lies within "
                   "cosine distance 0.08 of an earlier stored question. The subset comparison removes the cache from the "
                   "picture; it does not prove what the full set would show without the cache.",
    }
    conf = [lex[i]["judge_confidence"] for i in ids]
    conf_debated = [lex[i]["judge_confidence"] for i in debated_ids]
    report["benchmark_confidence"] = {"all": describe(conf), "debated_only": describe(conf_debated),
                                      "debated_at_or_below_0.20": sum(1 for c in conf_debated if c <= 0.20),
                                      "debated_at_or_above_0.70": sum(1 for c in conf_debated if c >= 0.70)}
    # Authority precision is recall divided by the number of citations: show it by citation count.
    by_count = defaultdict(list)
    for i in ids:
        for system, rows in (("lexagent", lex), ("baseline", bl)):
            by_count[(system, min(rows[i]["total_cites"], 4))].append((rows[i]["ar"], rows[i]["ap"]))
    report["authority_precision_by_citation_count"] = {
        f"{system}_cites_{'4+' if count == 4 else count}": {"answers": len(v), "mean_ar": round(mean(a for a, _ in v), 4),
                                                             "mean_ap": round(mean(p for _, p in v), 4)}
        for (system, count), v in sorted(by_count.items())}
    report["authority_precision_identity_check"] = all(
        abs(r["ap"] - (round(r["ar"] / r["total_cites"], 4) if r["total_cites"] else 0.0)) < 1e-9
        for rows in (lex, bl) for r in rows.values())

    # ── 3. logs of the ablation runs ──
    variants = {}
    sections = parse_variant_sections(os.path.join(LOGS_DIR, "ablation_full.log"))
    rerun = parse_variant_sections(os.path.join(LOGS_DIR, "ablation_A2_rerun.log"))
    per_case_logs = {}
    for name, lines in list(sections.items()) + [("A2_rerun", rerun.get("A2", []))]:
        label = {"A2": "A2_before_switch_fix (INVALID)", "A2_rerun": "A2"}.get(name, name)
        records, alignment = reconstruct_cases(lines, usable)
        stamps = [STAMP.search(l) for l in lines]
        stamps = [datetime.strptime(s.group(1), "%Y-%m-%d %H:%M:%S") for s in stamps if s]
        summary = summarise_variant(records)
        summary["alignment"] = alignment
        summary["log_lines"] = len(lines)
        summary["started"], summary["ended"] = (stamps[0].isoformat(), stamps[-1].isoformat()) if stamps else (None, None)
        summary["minutes"] = round((stamps[-1] - stamps[0]).total_seconds() / 60, 1) if stamps else None
        variants[label] = summary
        per_case_logs[label] = records
    report["ablation_logs"] = variants
    means = {r["variant"]: r for r in ablation}
    # The saved mean confidence divides by all 96 cases; a skipped debate contributes 0.0.
    report["confidence_decomposition"] = {
        v: {"saved_mean_confidence": round(means[v]["mean_confidence"], 4),
            "cache_hits": variants[v]["cache_hits"], "debated": variants[v]["debated"],
            "implied_mean_confidence_of_debated_cases": round(means[v]["mean_confidence"] * 96 / variants[v]["debated"], 4)
            if variants[v]["debated"] else None}
        for v in ("Full", "A1", "A2", "A3", "A4") if v in variants}
    report["confidence_decomposition"]["benchmark_run"] = {
        "saved_mean_confidence": round(mean(conf), 4), "cache_hits": len(signature), "debated": len(debated_ids),
        "implied_mean_confidence_of_debated_cases": round(mean(conf_debated), 4)}

    # Paired comparison of what the logs do record per case, full system against each variant.
    paired_logs = {}
    full = per_case_logs.get("Full")
    for v in ("A1", "A2", "A3", "A4"):
        other = per_case_logs.get(v)
        if not full or not other:
            continue
        entry = {"cache_hit": paired([int(r["cache_hit"]) for r in full], [int(r["cache_hit"]) for r in other], binary=True)}
        both = [(a, b) for a, b in zip(full, other) if a["rounds_cce"] and b["rounds_cce"]]
        if both:
            entry["cases_debated_with_citation_check_in_both"] = len(both)
            entry["final_round_avg_ccs"] = paired([a["rounds_cce"][-1]["avg_ccs"] for a, _ in both], [b["rounds_cce"][-1]["avg_ccs"] for _, b in both])
            entry["final_round_has_supported_citation"] = paired(
                [int(a["rounds_cce"][-1]["SUPPORTED"] > 0) for a, _ in both], [int(b["rounds_cce"][-1]["SUPPORTED"] > 0) for _, b in both], binary=True)
        paired_logs[f"Full_vs_{v}"] = entry
    report["ablation_paired_from_logs"] = {
        "limitation": "Per-case HR, answer similarity and confidence were not saved for any ablation variant, so the paired "
                      "tests the audit asks for cannot be computed for them. Only quantities the logs recorded per case are compared.",
        "comparisons": paired_logs}

    # ── 6. run-to-run variation of the fabricated-name rate ──
    from scipy import stats
    bench_flagged = sum(lex[i]["fake_cites"] for i in ids)
    bench_cites = sum(lex[i]["total_cites"] for i in ids)
    variation = {"benchmark_run": {"flagged_names": bench_flagged, "citations": bench_cites, "mean_hr": round(mean(lex[i]["hr"] for i in ids), 5),
                                   "cases_with_a_flagged_name": sum(1 for i in ids if lex[i]["fake_cites"])},
                 "per_case_hr_values_when_nonzero": sorted(lex[i]["hr"] for i in ids if lex[i]["hr"] > 0)}
    for v in ("Full", "A1", "A2", "A3", "A4"):
        total = means[v]["mean_hr"] * means[v]["completed_cases"]
        variation[f"ablation_{v}"] = {"mean_hr": round(means[v]["mean_hr"], 5), "sum_of_per_case_hr": round(total, 4),
                                      "reading": "sum of per-case rates; one flagged name among k citations of one case adds 1/k"}
    low, high = stats.binomtest(bench_flagged, bench_cites).proportion_ci(confidence_level=0.95, method="exact")
    variation["benchmark_pooled_rate_exact_95ci"] = [round(low, 5), round(high, 5)]
    variation["poisson_95_interval_for_a_count_with_mean_3"] = [int(stats.poisson.ppf(0.025, 3)), int(stats.poisson.ppf(0.975, 3))]
    report["run_to_run_variation"] = variation

    # ── 5. separability of the support score ──
    stress = load("citation_stress_test_results.json")
    details = stress["details"]
    by_type = defaultdict(list)
    for d in details:
        by_type[d["type"]].append(d)
    correct = [d["support"] for d in by_type["correct"] if d.get("support") is not None]
    swapped = [d["support"] for d in by_type["misattributed"] if d.get("support") is not None]
    distractor = [d["support"] for d in by_type["casehold_distractor"] if d.get("support") is not None]
    sweep = []
    for threshold in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90):
        sweep.append({"threshold": threshold, "correct_accepted": sum(1 for s in correct if s >= threshold),
                      "swapped_accepted": sum(1 for s in swapped if s >= threshold),
                      "false_holding_accepted": sum(1 for s in distractor if s >= threshold)})
    report["support_score_separability"] = {
        "stress_test": {"correct_n": len(correct), "swapped_n": len(swapped), "distractor_n": len(distractor),
                        "auc_correct_vs_swapped": auc(correct, swapped), "auc_correct_vs_false_holding": auc(correct, distractor),
                        "correct_support": describe(correct), "swapped_support": describe(swapped),
                        "correct_support_below_0.10": sum(1 for s in correct if s < 0.10),
                        "threshold_sweep": sweep,
                        "existence_of_correct_citations": dict(Counter(round(d["existence"], 2) for d in by_type["correct"])),
                        "tiers_of_correct_citations": dict(Counter(d["verdict"] for d in by_type["correct"]))},
    }
    evidence_path = os.path.join(RESULTS_DIR, "cce_evidence_selection_study.json")
    if os.path.exists(evidence_path):
        evidence = load("cce_evidence_selection_study.json")
        strategies = list(evidence["strategies"].keys())
        table = {}
        for s in strategies:
            pos = [row[s] for row in evidence["details"] if row["type"] == "correct"]
            neg = [row[s] for row in evidence["details"] if row["type"] == "misattributed"]
            best = max(((sum(1 for p in pos if p >= t) - sum(1 for n in neg if n >= t)), t) for t in [x / 100 for x in range(5, 100, 5)])
            table[s] = {"auc_correct_vs_swapped": auc(pos, neg),
                        "accepted_at_0.70": [sum(1 for p in pos if p >= 0.70), sum(1 for n in neg if n >= 0.70)],
                        "best_threshold_by_correct_minus_swapped": best[1],
                        "accepted_at_best_threshold": [sum(1 for p in pos if p >= best[1]), sum(1 for n in neg if n >= best[1])]}
        lengths = [row["passages"] for row in evidence["details"] if row["type"] == "correct"]
        report["support_score_separability"]["evidence_study"] = {
            "strategies": table, "unresolved": evidence.get("unresolved", []),
            "opinion_length_in_passages": describe(lengths),
            "share_of_opinion_read_by_three_passages": round(mean(min(1.0, 3 / max(1, n)) for n in lengths), 4)}

    # ── per-case table for the reports ──
    report["per_case"] = [
        {"case_id": i, "position": expected_ids.index(i) + 1, "topic": by_id[i]["topic"], "target_case": by_id[i].get("target_case", ""),
         "benchmark_cache_signature": i in set(signature),
         "lexagent": {k: lex[i][k] for k in ("hr", "ass", "ar", "ap", "judge_confidence", "debate_rounds", "total_cites", "mar", "haa", "ucr", "fake_names")},
         "baseline": {k: bl[i][k] for k in ("hr", "ass", "ar", "ap", "total_cites", "fake_names")},
         "ablation_full_log": {k: per_case_logs["Full"][n][k] for k in ("cache_hit", "cache_distance", "rounds_cce", "ddc", "judge_rejections", "verdict_stored", "avoid_list")} if "Full" in per_case_logs else None,
         "ablation_A3_log": {k: per_case_logs["A3"][n][k] for k in ("rounds_cce", "ddc", "judge_rejections")} if "A3" in per_case_logs else None}
        for n, i in enumerate(expected_ids)]

    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    # ── console summary ──
    sig = report["benchmark_cache_signature"]
    print(f"benchmark sha256 {bench_hash[:16]}…  cases {len(usable)}")
    for name, check in integrity.items():
        print(f"  {name}: {json.dumps({k: v for k, v in check.items() if k not in ('A2_before', 'A2_after', 'note')})[:330]}")
    print(f"benchmark run: cache signature in {sig['cases_with_signature']} cases; one-round {sig['one_round_cases']}; "
          f"citation-free {sig['citation_free_cases']}; identical sets: {sig['all_one_round_cases_have_signature']}, {sig['all_citation_free_cases_have_signature']}")
    for label, summary in variants.items():
        cc = summary.get("citation_check", {})
        print(f"  [{label}] hits {summary['cache_hits']}/{summary['cases']} debated {summary['debated']} rounds {summary['debate_rounds_distribution']} "
              f"stored {summary['verdicts_stored']} | supported first/final/any {cc.get('supported_in_first_round')}/{cc.get('supported_in_final_round')}/{cc.get('supported_in_any_round')} "
              f"lost {cc.get('supported_earlier_but_not_in_final_round')} | continued-with-supported {cc.get('continued_although_a_citation_was_supported')} "
              f"| align problems {summary['alignment']['alignment_problem_count']}")
    for subset in ("all_96_cases", "debated_cases_only"):
        block = report["lexagent_vs_baseline"][subset]
        line = f"  {subset} (n={block['n']}): "
        for metric in ("hr", "ass", "ar", "ap"):
            m = block[metric]
            line += f"{metric} {m['a']['mean']:.4f} vs {m['b']['mean']:.4f} CI{m['bootstrap_95ci_mean_difference']} p={m['sign_flip_permutation_p']} | "
        print(line)
    print("  confidence decomposition:", json.dumps(report["confidence_decomposition"]))
    print("  support separability:", json.dumps(report["support_score_separability"]["stress_test"]["threshold_sweep"][5:8]),
          "AUC", report["support_score_separability"]["stress_test"]["auc_correct_vs_swapped"])
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
