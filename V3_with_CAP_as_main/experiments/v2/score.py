# LexAgent v3.0 | experiments/v2/score.py
"""
Scores every saved run under data/results/v2/ with the repaired measures (repair 3) and writes

    data/results/v2/scores.json     every number
    data/results/v2/REPORT.md       the tables

It reads the saved per-case traces only; no model generates anything. It can be run at any time,
also while the queue is still producing runs: it reports what exists.

Comparisons are paired by benchmark case. With several seeds, a case's value is its mean over the
seeds. Intervals: 10,000 bootstrap resamples of cases (seed 42). Test: two-sided sign-flip
permutation. A difference is called significant only if p < 0.05 and the interval excludes zero.

    python experiments/v2/score.py
"""

import json
import math
import os
import random
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import RESULTS_DIR  # noqa: E402
from src.evaluation import metrics_v2  # noqa: E402

V2_DIR = os.path.join(RESULTS_DIR, "v2")
RESAMPLES = 10000
SIMILARITY_MODEL = "sentence-transformers/all-mpnet-base-v2"
SKIP = {"e2_verifier_validation", "e3_memory", "e5_retrieval"}
LABELS = {
    "baseline_4x300": "Baseline, 4 passages x 300 characters",
    "baseline_8x500": "Baseline, 8 passages x 500 characters",
    "e0_legacy_checker_no_memory": "E0: debate as it was, no memory (pre-audit checker)",
    "e4_repaired_Full": "Repaired debate, full",
    "e4_repaired_A1": "Repaired debate, A1 no defense",
    "e4_repaired_A2": "Repaired debate, A2 no citation checker",
    "e4_repaired_A3": "Repaired debate, A3 no memory",
    "e4_repaired_A4": "Repaired debate, A4 one round",
}
# name, how to read one case's score, larger is better (None = neither)
MEASURES = [
    ("target cited by the answer", lambda s: float(s["answer"]["target_cited"]), True),
    ("target cited by any agent", lambda s: float(s["pool"]["target_cited"]), True),
    ("citations the answer rests on", lambda s: float(s["answer"]["citations"]), None),
    ("citations made by all agents", lambda s: float(s["pool"]["citations"]), None),
    ("answer rests on no existing authority", lambda s: float(s["answer"]["no_existing_authority"]), False),
    ("cited cases not found (all agents)", lambda s: float(s["pool"]["not_found"]), False),
    ("answer similarity (sentence embedding)", lambda s: s.get("similarity"), True),
    ("legacy: fabricated-name rate", lambda s: s["legacy"].get("hr"), False),
    ("legacy: answer similarity (Legal-BERT)", lambda s: s["legacy"].get("ass"), True),
    ("legacy: authority recall", lambda s: s["legacy"].get("ar"), True),
]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def sd(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def paired(a, b):
    diffs = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    if not diffs:
        return None
    rng = random.Random(42)
    n = len(diffs)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(RESAMPLES))
    low, high = boot[int(0.025 * RESAMPLES)], boot[int(0.975 * RESAMPLES) - 1]
    observed = abs(sum(diffs) / n)
    hits = sum(1 for _ in range(RESAMPLES) if abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / n) >= observed - 1e-12)
    p = (hits + 1) / (RESAMPLES + 1)
    spread = sd(diffs)
    return {"n": n, "mean_a": round(mean(a), 4), "mean_b": round(mean(b), 4), "difference": round(sum(diffs) / n, 4),
            "ci95": [round(low, 4), round(high, 4)], "p": round(p, 4), "dz": round((sum(diffs) / n) / spread, 3) if spread else None,
            "significant": bool(p < 0.05 and (low > 0 or high < 0))}


def reparse_baseline_citations(rows):
    """Read the citations of single-agent answers again from the saved raw output.

    The baseline parser of the runs kept a citation only when the model wrote it as an object and
    dropped it when the model wrote a plain string, which it does for one answer in ten with four
    passages and for more than half with eight. The saved files are not changed: the citations as
    the run saved them stay in the row under "answer_citations_as_run", and scoring uses the
    citations read from the raw output. Debate rows are not touched (the agents always wrote
    objects). Returns None for a run that is not a baseline.
    """
    baseline = [r for r in rows if r.get("variant") == "baseline" and not r.get("error")]
    if not baseline:
        return None
    changed = before = after = 0
    for row in baseline:
        as_run = row.get("answer_citations") or []
        row["answer_citations_as_run"] = as_run
        read = metrics_v2.citations_from_baseline_output((row.get("trace") or {}).get("raw_output", ""))
        if read is None:
            read = as_run
        before += len(metrics_v2.distinct(as_run))
        after += len(metrics_v2.distinct(read))
        changed += int(sorted(metrics_v2.citation_key(c) for c in metrics_v2.distinct(as_run)) != sorted(metrics_v2.citation_key(c) for c in metrics_v2.distinct(read)))
        row["answer_citations"] = read
        row["pool_citations"] = read
    return {"answers": len(baseline), "answers_whose_citations_changed": changed, "citations_as_saved_by_the_run": before,
            "citations_read_from_the_raw_output": after}


def load_runs(root):
    runs = {}
    for experiment in sorted(os.listdir(root)):
        folder = os.path.join(root, experiment)
        if not os.path.isdir(folder) or experiment in SKIP or experiment.startswith("_"):
            continue
        for run in sorted(os.listdir(folder)):
            path = os.path.join(folder, run, "cases.jsonl")
            if not os.path.exists(path):
                continue
            rows = []
            for line in open(path, encoding="utf-8"):
                try:
                    rows.append(json.loads(line)) if line.strip() else None
                except ValueError:
                    pass  # a line cut off by a killed process; the runner removes it when it resumes
            config_path = os.path.join(folder, run, "run_config.json")
            reparse = reparse_baseline_citations(rows)
            runs.setdefault(experiment, {})[run] = {
                "baseline_reparse": reparse,
                "rows": [r for r in rows if not r.get("error")], "errors": sum(1 for r in rows if r.get("error")),
                "complete": os.path.exists(os.path.join(folder, run, "summary.json")),
                "config": json.load(open(config_path, encoding="utf-8")) if os.path.exists(config_path) else {}}
    return runs


class Existence:
    """Existence checks through the case catalog, remembered on disk (a non-exact name costs seconds)."""

    def __init__(self, cce, path):
        self.cce, self.path, self.changed = cce, path, False
        self.cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}

    def __call__(self, name, cite=""):
        key = f"{name}||{cite}"
        if key not in self.cache:
            self.cache[key] = self.cce.check_existence(name, cite)
            self.changed = True
        return self.cache[key]

    def save(self):
        if self.changed:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.cache, handle, ensure_ascii=False)
            self.changed = False


def similarity_scores(runs, cases):
    """Answer against reference holding with a sentence-embedding model, plus the metric's floors."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SIMILARITY_MODEL)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    import numpy as np

    def encode(texts):
        return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64))

    ids = [c["id"] for c in cases]
    holdings = encode([c.get("ground_truth_holding", "") for c in cases])
    questions = encode([c["query"] for c in cases])
    cross = holdings @ holdings.T
    off = cross[~np.eye(len(cases), dtype=bool)]
    index = {case_id: i for i, case_id in enumerate(ids)}
    for experiment in runs.values():
        for run in experiment.values():
            rows = [r for r in run["rows"] if r["case_id"] in index]
            if not rows:
                continue
            answers = encode([str(r.get("final_answer", "")) for r in rows])
            for row, vector in zip(rows, answers):
                row["_similarity"] = float(vector @ holdings[index[row["case_id"]]])
    return {"model": SIMILARITY_MODEL, "floor_holding_of_another_case": round(float(off.mean()), 4),
            "the_question_itself": round(float((questions * holdings).sum(axis=1).mean()), 4)}


def main(root: str = V2_DIR, parts: dict = None) -> dict:
    from experiments.v2.runner import benchmark_cases, checker, load_components

    runs = load_runs(root)
    if not runs:
        print("no runs to score yet")
        return {}
    if parts is None:
        parts = load_components(load_llm=False)
    cases, digest = benchmark_cases()
    cce = checker(parts, "gated")
    exists = Existence(cce, os.path.join(root, "_existence_cache.json"))
    targets = {}
    for case in cases:
        found = exists(case.get("target_case", ""), case.get("reporter_citation", "") or "")
        targets[case["id"]] = {"doc_id": found.get("resolved_doc_id"), "name": case.get("target_case", "")}
    floors = similarity_scores(runs, cases)

    scored, table = {}, {}
    for experiment, seeds in runs.items():
        scored[experiment] = {}
        for run, data in seeds.items():
            per_case = {}
            for row in data["rows"]:
                score = metrics_v2.score_case(row, exists, targets.get(row["case_id"]))
                score["legacy"] = row.get("legacy_scorecard") or {}
                score["similarity"] = row.get("_similarity")
                score["seconds"] = row.get("seconds")
                per_case[row["case_id"]] = score
            scored[experiment][run] = per_case
            aggregate = metrics_v2.aggregate(list(per_case.values()))
            if data.get("baseline_reparse"):
                # The same run scored with the citations as its own parser saved them, for the record.
                as_run = []
                for row in data["rows"]:
                    copy = dict(row, answer_citations=row.get("answer_citations_as_run") or [], pool_citations=row.get("answer_citations_as_run") or [])
                    as_run.append(metrics_v2.score_case(copy, exists, targets.get(row["case_id"])))
                saved = metrics_v2.aggregate(as_run)
                aggregate["baseline_citation_parsing"] = dict(data["baseline_reparse"], as_saved_by_the_run={
                    "target_cited_rate": saved["answer"]["target_cited_rate"], "citations": saved["answer"]["citations"],
                    "answers_without_existing_authority": saved["answer"]["answers_without_existing_authority"]})
            aggregate.update({"complete": data["complete"], "errors": data["errors"],
                              "mean_confidence": round(mean(s["confidence"] for s in per_case.values()) or 0.0, 4) if any(s["confidence"] is not None for s in per_case.values()) else None,
                              "mean_rounds": round(mean(float(s["rounds"]) for s in per_case.values() if s["rounds"] is not None) or 0.0, 3),
                              "minutes": round(sum(s["seconds"] or 0 for s in per_case.values()) / 60, 1),
                              "measures": {name: (round(mean(read(s) for s in per_case.values()), 4) if mean(read(s) for s in per_case.values()) is not None else None)
                                           for name, read, _ in MEASURES}})
            table.setdefault(experiment, {})[run] = aggregate
        exists.save()

    def complete_seeds(experiment):
        """Only finished runs enter a comparison: a run in progress has some cases and not others,
        and would silently reduce the comparison to the cases it happens to have."""
        return {run: cases for run, cases in scored.get(experiment, {}).items() if table[experiment][run]["complete"]}

    def case_means(experiment, read):
        """One value per case: its mean over the finished seeds."""
        seeds = complete_seeds(experiment)
        ids = sorted(set.intersection(*[set(v) for v in seeds.values()])) if seeds else []
        return {i: mean(read(seeds[run][i]) for run in seeds) for i in ids}

    comparisons = {}
    pairs = [("e0_legacy_checker_no_memory", "baseline_4x300"), ("e0_legacy_checker_no_memory", "baseline_8x500"),
             ("e4_repaired_Full", "baseline_4x300"), ("e4_repaired_Full", "baseline_8x500"),
             ("baseline_8x500", "baseline_4x300"), ("e4_repaired_Full", "e0_legacy_checker_no_memory"),
             ("e4_repaired_Full", "e4_repaired_A1"), ("e4_repaired_Full", "e4_repaired_A2"),
             ("e4_repaired_Full", "e4_repaired_A3"), ("e4_repaired_Full", "e4_repaired_A4"),
             # The system after repairs 6 (ledger protocol) and 7 (retrieval v2).
             ("f2_final_Full", "baseline_r7_4x300"), ("f2_final_Full", "baseline_r7_8x500"),
             ("f2_final_Full", "e4_repaired_Full"), ("f2_final_Full", "baseline_4x300"),
             ("f1_ledger_Full", "e4_repaired_Full"), ("f3_retrieval_Full", "e4_repaired_Full"),
             ("f2_final_Full", "f1_ledger_Full"), ("f2_final_Full", "f3_retrieval_Full"),
             ("baseline_r7_4x300", "baseline_4x300"), ("baseline_r7_8x500", "baseline_8x500"),
             ("baseline_r7_8x500", "baseline_r7_4x300")]
    for a, b in pairs:
        if not complete_seeds(a) or not complete_seeds(b):
            continue
        block = {}
        for name, read, _ in MEASURES:
            left, right = case_means(a, read), case_means(b, read)
            ids = sorted(set(left) & set(right))
            result = paired([left[i] for i in ids], [right[i] for i in ids])
            if result:
                block[name] = result
        block["_seeds"] = {a: sorted(complete_seeds(a)), b: sorted(complete_seeds(b))}
        comparisons[f"{a} vs {b}"] = block

    spread = {}
    for experiment, seeds in table.items():
        if len(seeds) >= 2:
            spread[experiment] = {name: {"values": [seeds[r]["measures"][name] for r in sorted(seeds)],
                                         "sd": round(sd([seeds[r]["measures"][name] for r in sorted(seeds)]) or 0.0, 4)}
                                  for name, _, _ in MEASURES if all(seeds[r]["measures"][name] is not None for r in seeds)}

    report = {"written": datetime.now().isoformat(timespec="seconds"), "benchmark_sha256": digest, "cases": len(cases),
              "similarity_metric": floors, "runs": table, "paired_comparisons": comparisons, "seed_to_seed": spread}
    with open(os.path.join(root, "scores.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    write_markdown(root, report)
    write_manifest(root, runs)
    print(f"  [score] {sum(len(v) for v in table.values())} runs scored -> {os.path.join(root, 'REPORT.md')}")
    return report


def write_manifest(root, runs):
    """One record per post-audit run. The pre-audit manifest (data/results/EXPERIMENT_MANIFEST.json) is not touched."""
    entries = []
    for experiment, seeds in runs.items():
        for run, data in sorted(seeds.items()):
            cfg = data["config"]
            entries.append({
                "experiment": LABELS.get(experiment, experiment), "run_directory": os.path.join(root, experiment, run).replace("\\", "/"),
                "kind": cfg.get("kind"), "variant": cfg.get("variant"), "seed": cfg.get("seed"), "case_seed_rule": cfg.get("case_seed_rule"),
                "case_count": len(data["rows"]), "errors": data["errors"],
                "status": "COMPLETE" if data["complete"] else "RUNNING OR INTERRUPTED",
                "date": cfg.get("created"), "benchmark_sha256": cfg.get("benchmark_sha256"), "git": cfg.get("git"),
                "model_name": (cfg.get("settings") or {}).get("MODEL_NAME"), "embedding_model": (cfg.get("settings") or {}).get("EMBEDDER_NAME"),
                "settings": cfg.get("settings"), "cce_policy": cfg.get("cce_policy"), "cce_gate": cfg.get("cce_gate"),
                "memory_cache_mode": cfg.get("memory_cache_mode"), "memory_namespace": cfg.get("memory_namespace"),
                "baseline_context": ({"passages": cfg.get("context_passages"), "characters": cfg.get("context_characters_per_passage"),
                                      "dense_query_jurisdiction": cfg.get("dense_query_jurisdiction")} if cfg.get("kind") == "baseline" else None),
                "per_case_output": "full trace of every case (cases.jsonl)",
                "valid_for_publication": ("YES" if data["complete"] and not data["errors"] else "NOT YET — incomplete or with errors"),
            })
    extras = []
    for name, path, note in (
        ("E2 verifier validation", os.path.join(root, "e2_verifier_validation", "results.json"), "decides the support gate; no debate is run"),
        ("E2b near-miss probe", os.path.join(root, "e2_verifier_validation", "near_miss_probe.json"), "post-hoc; does not change the gate"),
        ("E3 verdict cache on repeats", os.path.join(root, "e3_memory", "results.json"), "cache decisions only; no debate is run"),
        ("E5 retrieval only", os.path.join(root, "e5_retrieval", "results.json"), "no LLM"),
    ):
        extras.append({"experiment": name, "result_file": path.replace("\\", "/"), "status": "COMPLETE" if os.path.exists(path) else "NOT RUN YET", "note": note})
    manifest = {"written": datetime.now().isoformat(timespec="seconds"),
                "relation_to_earlier_results": "Runs made after the forensic audit with repairs 1-4. The earlier result files and "
                                               "data/results/EXPERIMENT_MANIFEST.json are unchanged and remain the record of the pre-audit runs.",
                "runs": entries, "other_experiments": extras}
    with open(os.path.join(root, "EXPERIMENT_MANIFEST_V2.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def write_markdown(root, report):
    def cell(value, digits=3):
        return "—" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))

    lines = ["# Post-audit experiments: results so far", "",
             f"Written {report['written']}. Benchmark {report['benchmark_sha256'][:16]}…, {report['cases']} cases. "
             "Generated by `experiments/v2/score.py` from the saved traces; regenerated after every run, so it shows what exists.", "",
             "Measures: *target cited* is whether a citation resolves to the benchmark's target decision. *Answer* scope is what the final "
             "answer rests on (the judge's relied-on citations; the baseline's citations); *any agent* is every citation made in the debate. "
             "Citation counts are shown next to target-cited rates and are not divided into them. Nothing here measures whether the other "
             "citations are relevant or whether the answer is legally correct.", ""]
    floors = report.get("similarity_metric") or {}
    if "model" in floors:
        lines += [f"Answer similarity uses `{floors['model']}`. On this metric the reference holding of a *different* case scores "
                  f"{floors['floor_holding_of_another_case']} and the question itself scores {floors['the_question_itself']}; read the similarity column against those.", ""]
    lines += ["## Runs", "",
              "| Run | Seed | Cases | Skipped by cache | Target cited: answer | Target cited: any agent | Citations: answer | Citations: all agents | "
              "No existing authority | Existence rate (all) | Not found | Similarity | Confidence | Rounds | Minutes | Complete |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for experiment, seeds in report["runs"].items():
        for run, row in sorted(seeds.items()):
            m = row["measures"]
            lines.append(f"| {LABELS.get(experiment, experiment)} | {run.replace('seed', '')} | {row['cases']} | {row['skipped_debates']} | "
                         f"{cell(row['answer']['target_cited_rate'])} | {cell(row['pool']['target_cited_rate'])} | {cell(row['answer']['citations_per_answer'], 2)} | "
                         f"{cell(row['pool']['citations_per_answer'], 2)} | {row['answer']['answers_without_existing_authority']} | {cell(row['pool']['existence_rate'])} | "
                         f"{row['pool']['citations_not_found']} | {cell(m.get('answer similarity (sentence embedding)'))} | {cell(row['mean_confidence'], 2)} | "
                         f"{cell(row['mean_rounds'], 2)} | {cell(row['minutes'], 0)} | {'yes' if row['complete'] else 'running'} |")
    lines += ["", "Legacy measures (pre-audit definitions, for continuity with the frozen results only):", "",
              "| Run | Seed | Fabricated-name rate | Answer similarity (Legal-BERT) | Authority recall |", "|---|---|---|---|---|"]
    for experiment, seeds in report["runs"].items():
        for run, row in sorted(seeds.items()):
            m = row["measures"]
            lines.append(f"| {LABELS.get(experiment, experiment)} | {run.replace('seed', '')} | {cell(m.get('legacy: fabricated-name rate'), 4)} | "
                         f"{cell(m.get('legacy: answer similarity (Legal-BERT)'), 4)} | {cell(m.get('legacy: authority recall'), 4)} |")
    lines += ["", "## Paired comparisons (A minus B, per benchmark case, seeds averaged within a case)", ""]
    for name, block in report["paired_comparisons"].items():
        a, b = name.split(" vs ")
        seeds = block.get("_seeds", {})
        lines += [f"### {LABELS.get(a, a)}  vs  {LABELS.get(b, b)}", "",
                  f"Seeds: A {seeds.get(a)}, B {seeds.get(b)}.", "",
                  "| Measure | A | B | A − B | 95% interval | p | Effect size dz | Significant |", "|---|---|---|---|---|---|---|---|"]
        for measure, result in block.items():
            if measure.startswith("_"):
                continue
            lines.append(f"| {measure} | {cell(result['mean_a'])} | {cell(result['mean_b'])} | {result['difference']:+.3f} | "
                         f"[{result['ci95'][0]:+.3f}, {result['ci95'][1]:+.3f}] | {cell(result['p'], 4)} | {cell(result['dz'])} | {'yes' if result['significant'] else 'no'} |")
        lines.append("")
    if report["seed_to_seed"]:
        lines += ["## Seed-to-seed variation (same configuration, different seed)", "",
                  "| Run | Measure | Value per seed | SD |", "|---|---|---|---|"]
        for experiment, block in report["seed_to_seed"].items():
            for measure, row in block.items():
                lines.append(f"| {LABELS.get(experiment, experiment)} | {measure} | {row['values']} | {row['sd']} |")
        lines.append("")
    with open(os.path.join(root, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else V2_DIR)
