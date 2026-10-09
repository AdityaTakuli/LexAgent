# LexAgent v3.0 | experiments/v2/standard_metrics.py
"""
Standard measures for every saved run, so that the results can be set beside other work.

The measures of experiments/v2/score.py were written to say exactly what this system does. The ones
here are those other papers report, computed the same way for the baseline and for the debate from
the saved cases (no generator call):

  Retrieval                Hit@k (the target opinion is among the first k passages; with one relevant
                           opinion per question this is Recall@k) and MRR, on the first retrieval of
                           each question.
  Fabricated citations     the measure of Dahl et al. for case existence: share of citations, and of
                           answers, that name a case which does not exist.
  Citation recall /        the ALCE definitions (Gao et al., EMNLP 2023) applied to case citations.
  citation precision       Recall: share of the sentences of the final answer that a cited opinion
                           supports. Precision: share of the citations that support at least one
                           sentence. "Supports" is judged by the support model that passed the
                           validation of experiments/v2/e2_verifier_validation.py, at its validated
                           threshold, on the three passages of the cited opinion nearest to the sentence.
  Grounding outcome        after Magesh et al.: each answer is grounded (cites at least one authority
                           and every citation exists and supports the answer), misgrounded (cites a
                           case that does not exist or does not support it), or without authority.
                           Automatic, and it does not judge whether the answer is legally correct.
  Authority accuracy       the answer rests on the target decision of the question.
  ROUGE-L, BERTScore-F1,   the final answer against the reference holding.
  token F1
  Cost                     seconds and generator calls per question.

Citations to real cases that are not in the corpus cannot be checked; they are counted separately
and left out of citation precision.

    python experiments/v2/standard_metrics.py
"""

import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import RESULTS_DIR  # noqa: E402
from src.evaluation import metrics_v2  # noqa: E402

V2_DIR = os.path.join(RESULTS_DIR, "v2")
BERTSCORE_MODEL = "roberta-large"
DEPTHS = (1, 4, 8)
ABBREVIATIONS = ("v.", "vs.", "U.S.", "U. S.", "No.", "Nos.", "Inc.", "Co.", "Corp.", "Ltd.", "e.g.", "i.e.", "Cf.", "cf.", "Id.",
                 "Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", "Sr.", "St.", "Art.", "Sec.", "Stat.", "Ed.", "Ct.", "Cal.", "Fla.", "Ala.")
# Rows of the summary: (key in a per-case record, title, kind). "rate" is shown as a percentage.
SUMMARY = [
    ("hit_at_1", "Retrieval Hit@1 (%)", "rate"), ("hit_at_4", "Retrieval Hit@4 (%)", "rate"), ("hit_at_8", "Retrieval Hit@8 (%)", "rate"),
    ("reciprocal_rank", "Retrieval MRR@8", "score"), ("target_in_context", "Target opinion in the generator's context (%)", "rate"),
    ("authority_accuracy", "Authority accuracy: answer rests on the target (%)", "rate"),
    ("has_fabricated", "Answers citing a non-existent case (%)", "rate"),
    ("grounded", "Answers grounded (%)", "rate"), ("misgrounded", "Answers misgrounded (%)", "rate"),
    ("no_authority", "Answers without authority (%)", "rate"),
    ("sentence_support", "Citation recall, per answer (%)", "rate"),
    ("rouge_l", "ROUGE-L F1", "score"), ("bertscore_f1", "BERTScore F1", "score"), ("token_f1", "Token F1", "score"),
    ("seconds", "Seconds per question", "number"), ("llm_calls", "Generator calls per question", "number"),
]
PAIRED = ["authority_accuracy", "grounded", "misgrounded", "no_authority", "sentence_support", "has_fabricated", "rouge_l", "bertscore_f1", "token_f1"]
COMPARISONS = [("f2_final_Full", "baseline_r7_4x300"), ("f2_final_Full", "baseline_r7_8x500"), ("f2_final_Full", "baseline_4x300"),
               ("f2_final_Full", "e4_repaired_Full"), ("f2_final_Full", "e0_legacy_checker_no_memory"),
               ("e4_repaired_Full", "baseline_4x300"), ("e4_repaired_Full", "baseline_8x500"),
               ("e4_repaired_Full", "e0_legacy_checker_no_memory"), ("e0_legacy_checker_no_memory", "baseline_4x300"),
               ("f1_ledger_Full", "e4_repaired_Full"), ("f3_retrieval_Full", "e4_repaired_Full"),
               ("baseline_r7_4x300", "baseline_4x300")]


# ── text helpers ────────────────────────────────────────────────────────────
def split_sentences(text: str) -> list:
    """Sentences of an answer. Abbreviations of case names and reporters do not end a sentence."""
    protected = " ".join(str(text or "").split())
    for index, abbreviation in enumerate(ABBREVIATIONS):
        protected = protected.replace(abbreviation, f"\x00{index}\x00")
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])", protected)
    sentences = []
    for part in parts:
        for index, abbreviation in enumerate(ABBREVIATIONS):
            part = part.replace(f"\x00{index}\x00", abbreviation)
        part = part.strip()
        if len(re.findall(r"[A-Za-z]{2,}", part)) >= 4:  # a fragment of three words asserts nothing checkable
            sentences.append(part)
    return sentences


def tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = tokens(prediction), tokens(reference)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def sha(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, path: str):
        self.path, self.changed = path, False
        self.data = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}

    def save(self) -> None:
        if self.changed:
            with open(self.path + ".tmp", "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False)
            os.replace(self.path + ".tmp", self.path)
            self.changed = False


class Support:
    """Does an opinion support a statement? The validated support model, remembered on disk."""

    def __init__(self, cce, path: str):
        self.cce, self.cache = cce, DiskCache(path)
        self.threshold = float(cce.gate.get("threshold"))
        self.validated = bool(cce.gate.get("validated"))

    def score(self, doc_id: str, statement: str):
        key = sha(doc_id, statement)
        if key not in self.cache.data:
            try:
                value = float(self.cce._gated_support(statement, SimpleNamespace(doc_id=doc_id))[0])
            except Exception:  # an opinion without passages in the collection cannot support anything
                value = None
            self.cache.data[key] = value
            self.cache.changed = True
        return self.cache.data[key]

    def supports(self, doc_id: str, statement: str) -> bool:
        value = self.score(doc_id, statement)
        return value is not None and value >= self.threshold


# ── one case ────────────────────────────────────────────────────────────────
def retrieved_of(row: dict) -> tuple:
    """(passages of the first retrieval, number of them the generator was given)."""
    trace = row.get("trace") or {}
    if "retrieved" in trace:  # baseline
        passages = trace.get("retrieved") or []
        return passages, int(trace.get("passages_in_prompt") or len(passages))
    rounds = trace.get("rounds") or []
    passages = (rounds[0].get("retrieved") if rounds else None) or []
    return passages, len(passages)


def standard_case(row: dict, case: dict, target: dict, exists, support: Support, rouge) -> dict:
    answer = str(row.get("final_answer", "") or "")
    reference = case.get("ground_truth_holding", "")
    skipped = bool(row.get("memory_cache_hit"))
    citations = metrics_v2.distinct(row.get("answer_citations") or [])
    sentences = split_sentences(answer)

    checked = []
    for citation in citations:
        found = exists(citation.get("case_name", ""), citation.get("reporter_cite", "") or "")
        verdict = found.get("existence_verdict")
        doc_id = found.get("resolved_doc_id")
        checkable = verdict == metrics_v2.EXISTS and bool(doc_id) and bool(found.get("resolved_is_opinion"))
        supported = [support.supports(doc_id, sentence) for sentence in sentences] if checkable else [False] * len(sentences)
        same_report = bool(doc_id) and doc_id == (target or {}).get("doc_id")
        checked.append({
            "case_name": citation.get("case_name", ""), "existence": verdict, "exists": verdict == metrics_v2.EXISTS,
            "fabricated": verdict == metrics_v2.NOT_FOUND, "cannot_be_checked": verdict == metrics_v2.NOT_IN_CORPUS or (verdict == metrics_v2.EXISTS and not checkable),
            "supports_a_sentence": any(supported), "supported_sentences": supported,
            "is_target": same_report or (verdict == metrics_v2.EXISTS and metrics_v2.names_match(citation.get("case_name", ""), (target or {}).get("name", ""))),
        })

    sentence_supported = [any(c["supported_sentences"][i] for c in checked) for i in range(len(sentences))]
    judged = [c for c in checked if not c["cannot_be_checked"]]
    existing = [c for c in checked if c["exists"]]
    fabricated = [c for c in checked if c["fabricated"]]
    unsupporting = [c for c in judged if c["exists"] and not c["supports_a_sentence"]]
    if not existing and not fabricated:
        outcome = "no_authority"
    elif fabricated or unsupporting:
        outcome = "misgrounded"
    elif any(c["supports_a_sentence"] for c in judged):
        outcome = "grounded"
    else:
        outcome = "no_authority"  # only citations that cannot be checked

    passages, in_context = retrieved_of(row)
    target_id = (target or {}).get("doc_id")
    parents = [p.get("parent_doc_id") or str(p.get("doc_id", "")).split("::chunk::")[0] for p in passages]
    rank = next((i for i, parent in enumerate(parents, start=1) if target_id and parent == target_id), None)

    trace = row.get("trace") or {}
    record = {
        "skipped": skipped,
        "citations": len(checked), "citations_judged": len(judged), "citations_supporting": sum(1 for c in judged if c["supports_a_sentence"]),
        "citations_fabricated": len(fabricated), "citations_cannot_be_checked": sum(1 for c in checked if c["cannot_be_checked"]),
        "sentences": len(sentences), "sentences_supported": sum(sentence_supported),
        "sentence_support": (sum(sentence_supported) / len(sentences)) if sentences else 0.0,
        "has_fabricated": float(bool(fabricated)),
        "grounded": float(outcome == "grounded"), "misgrounded": float(outcome == "misgrounded"), "no_authority": float(outcome == "no_authority"),
        "authority_accuracy": float(any(c["is_target"] for c in checked)),
        "reciprocal_rank": (1.0 / rank) if rank and rank <= max(DEPTHS) else 0.0,
        "target_in_context": float(bool(rank) and rank <= in_context),
        "rouge_l": rouge.score(reference, answer)["rougeL"].fmeasure if (answer.strip() and reference.strip()) else 0.0,
        "token_f1": token_f1(answer, reference),
        "seconds": float(row.get("seconds") or 0.0),
        "llm_calls": float(len(trace["llm_calls"])) if "llm_calls" in trace else 1.0,
        "_answer": answer, "_reference": reference,
    }
    for depth in DEPTHS:
        record[f"hit_at_{depth}"] = float(bool(rank) and rank <= depth)
    return record


def add_bertscore(records: list, cache: DiskCache) -> str:
    """BERTScore F1 of every answer against its reference; only pairs not scored before are computed."""
    missing = sorted({(r["_answer"], r["_reference"]) for r in records if sha(r["_answer"], r["_reference"]) not in cache.data})
    note = BERTSCORE_MODEL
    if missing:
        try:
            import torch
            from bert_score import BERTScorer

            try:
                scorer = BERTScorer(model_type=BERTSCORE_MODEL, lang="en", rescale_with_baseline=False,
                                    device="cuda" if torch.cuda.is_available() else "cpu")
                values = scorer.score([a if a.strip() else "." for a, _ in missing], [r for _, r in missing], batch_size=32)[2].tolist()
            except RuntimeError:  # no room on the GPU next to the generator
                scorer = BERTScorer(model_type=BERTSCORE_MODEL, lang="en", rescale_with_baseline=False, device="cpu")
                values = scorer.score([a if a.strip() else "." for a, _ in missing], [r for _, r in missing], batch_size=16)[2].tolist()
            for (answer, reference), value in zip(missing, values):
                cache.data[sha(answer, reference)] = float(value) if answer.strip() else 0.0
            cache.changed = True
            cache.save()
        except Exception as error:
            note = f"not computed: {type(error).__name__}: {error}"
    for record in records:
        record["bertscore_f1"] = cache.data.get(sha(record["_answer"], record["_reference"]))
    return note


# ── aggregation ─────────────────────────────────────────────────────────────
def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarise_run(records: list) -> dict:
    answered = [r for r in records if not r["skipped"]]
    out = {key: mean(r.get(key) for r in answered) for key, _, _ in SUMMARY}
    judged = sum(r["citations_judged"] for r in answered)
    citations = sum(r["citations"] for r in answered)
    sentences = sum(r["sentences"] for r in answered)
    out.update({
        "cases": len(records), "skipped": len(records) - len(answered),
        "citations": citations, "citations_per_answer": citations / len(answered) if answered else None,
        "citations_fabricated": sum(r["citations_fabricated"] for r in answered),
        "citations_cannot_be_checked": sum(r["citations_cannot_be_checked"] for r in answered),
        "fabricated_citation_rate": (sum(r["citations_fabricated"] for r in answered) / citations) if citations else None,
        "citation_precision": (sum(r["citations_supporting"] for r in answered) / judged) if judged else None,
        "citation_recall": (sum(r["sentences_supported"] for r in answered) / sentences) if sentences else None,
        "sentences": sentences,
    })
    return out


def main(root: str = V2_DIR, parts: dict = None, only: str = "", out_dir: str = None) -> dict:
    """Score every finished run under `root`. `only` keeps the runs whose "experiment/run" contains
    it; `out_dir` writes the report and the caches elsewhere (both are for trying the script out)."""
    import shutil

    from rouge_score import rouge_scorer

    from experiments.v2 import score
    from experiments.v2.runner import benchmark_cases, checker, load_components

    runs = score.load_runs(root)
    if only:
        runs = {e: {r: d for r, d in seeds.items() if only in f"{e}/{r}"} for e, seeds in runs.items()}
        runs = {e: seeds for e, seeds in runs.items() if seeds}
    if not runs:
        print("no runs to score yet")
        return {}
    if parts is None:
        parts = load_components(load_llm=False)
    cases, digest = benchmark_cases()
    by_id = {c["id"]: c for c in cases}
    cce = checker(parts, "gated")
    source_root, root = root, (out_dir or root)
    os.makedirs(root, exist_ok=True)
    if root != source_root and not os.path.exists(os.path.join(root, "_existence_cache.json")) and os.path.exists(os.path.join(source_root, "_existence_cache.json")):
        shutil.copy(os.path.join(source_root, "_existence_cache.json"), os.path.join(root, "_existence_cache.json"))
    exists = score.Existence(cce, os.path.join(root, "_existence_cache.json"))
    support = Support(cce, os.path.join(root, "_support_cache.json"))
    bert_cache = DiskCache(os.path.join(root, "_bertscore_cache.json"))
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    targets = {}
    for case in cases:
        found = exists(case.get("target_case", ""), case.get("reporter_citation", "") or "")
        targets[case["id"]] = {"doc_id": found.get("resolved_doc_id"), "name": case.get("target_case", "")}

    per_case, table, bert_note = {}, {}, BERTSCORE_MODEL
    for experiment, seeds in runs.items():
        for run, data in sorted(seeds.items()):
            if not data["complete"]:
                continue  # a run in progress is scored when it is finished
            records = {}
            for row in data["rows"]:
                if row["case_id"] in by_id:
                    records[row["case_id"]] = standard_case(row, by_id[row["case_id"]], targets.get(row["case_id"]), exists, support, rouge)
            bert_note = add_bertscore(list(records.values()), bert_cache)
            per_case.setdefault(experiment, {})[run] = records
            table.setdefault(experiment, {})[run] = summarise_run(list(records.values()))
            exists.save()
            support.cache.save()
            print(f"  [standard] {experiment}/{run}: {len(records)} cases")

    keys = [key for key, _, _ in SUMMARY] + ["fabricated_citation_rate", "citation_precision", "citation_recall", "citations_per_answer"]
    experiments = {}
    for experiment, seeds in table.items():
        rows = list(seeds.values())
        experiments[experiment] = {"seeds": sorted(seeds), "mean": {k: mean(r.get(k) for r in rows) for k in keys},
                                   "sd": {k: score.sd([r.get(k) for r in rows]) for k in keys},
                                   "citations": sum(r["citations"] for r in rows), "citations_fabricated": sum(r["citations_fabricated"] for r in rows),
                                   "citations_cannot_be_checked": sum(r["citations_cannot_be_checked"] for r in rows)}

    def case_means(experiment, key):
        seeds = per_case.get(experiment, {})
        ids = sorted(set.intersection(*[set(v) for v in seeds.values()])) if seeds else []
        return {i: mean(seeds[run][i].get(key) for run in seeds) for i in ids}

    comparisons = {}
    for a, b in COMPARISONS:
        if a not in per_case or b not in per_case:
            continue
        block = {}
        for key in PAIRED:
            left, right = case_means(a, key), case_means(b, key)
            ids = sorted(set(left) & set(right))
            result = score.paired([left[i] for i in ids], [right[i] for i in ids])
            if result:
                block[key] = result
        block["_seeds"] = {a: sorted(per_case[a]), b: sorted(per_case[b])}
        comparisons[f"{a} vs {b}"] = block

    report = {
        "written": datetime.now().isoformat(timespec="seconds"), "benchmark_sha256": digest, "cases": len(cases),
        "support_model": {"name": cce.support_model_name, "threshold": support.threshold, "validated": support.validated,
                          "passages_read_per_statement": cce.gate.get("passages"), "granularity": cce.gate.get("granularity")},
        "bertscore_model": bert_note, "rouge": "ROUGE-L F1, rouge-score with stemming",
        "experiments": experiments, "runs": table, "paired_comparisons": comparisons,
        "titles": {key: title for key, title, _ in SUMMARY},
    }
    with open(os.path.join(root, "standard_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    write_markdown(root, report)
    print(f"  [standard] {sum(len(v) for v in table.values())} runs -> {os.path.join(root, 'STANDARD_METRICS.md')}")
    return report


def show(value, kind: str) -> str:
    if value is None:
        return "–"
    return f"{100 * value:.1f}" if kind == "rate" else (f"{value:.3f}" if kind == "score" else f"{value:.1f}")


def write_markdown(root: str, report: dict) -> None:
    names = list(report["experiments"])
    lines = ["# Standard measures", "",
             f"Written {report['written']}. {report['cases']} questions. Mean over the finished seeds of each configuration.",
             f"Support is judged by `{report['support_model']['name']}` at threshold {report['support_model']['threshold']} "
             f"(validated: {report['support_model']['validated']}). BERTScore: {report['bertscore_model']}. {report['rouge']}.", "",
             "| Measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
             "| Seeds | " + " | ".join(str(len(report["experiments"][n]["seeds"])) for n in names) + " |"]
    extra = [("fabricated_citation_rate", "Citations to non-existent cases (% of citations)", "rate"),
             ("citation_precision", "Citation precision (%)", "rate"), ("citation_recall", "Citation recall (% of answer sentences)", "rate"),
             ("citations_per_answer", "Citations per answer", "score")]
    for key, title, kind in SUMMARY[:7] + extra + SUMMARY[7:]:
        lines.append(f"| {title} | " + " | ".join(show(report["experiments"][n]["mean"].get(key), kind) for n in names) + " |")
    lines += ["", "## Paired differences (A minus B, per question; seeds averaged within a question)", ""]
    for name, block in report["paired_comparisons"].items():
        lines += [f"### {name}", f"Seeds: {block.get('_seeds')}", "", "| Measure | A | B | Difference | 95% interval | p | Significant |", "|---|---|---|---|---|---|---|"]
        for key in PAIRED:
            result = block.get(key)
            if result:
                lines.append(f"| {report['titles'].get(key, key)} | {result['mean_a']:.4f} | {result['mean_b']:.4f} | {result['difference']:+.4f} | "
                             f"[{result['ci95'][0]:+.3f}, {result['ci95'][1]:+.3f}] | {result['p']:.4f} | {'yes' if result['significant'] else 'no'} |")
        lines.append("")
    with open(os.path.join(root, "STANDARD_METRICS.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standard measures for the saved runs")
    parser.add_argument("--only", default="", help='score only the runs whose "experiment/run" contains this text')
    parser.add_argument("--out", default=None, help="write the report and the caches to this directory instead of the results directory")
    arguments = parser.parse_args()
    main(only=arguments.only, out_dir=arguments.out)
