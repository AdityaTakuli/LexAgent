# LexAgent v3.0 | experiments/v2/runner.py
"""
Experiment runner written after the forensic audit (repair 1: reproducible and inspectable runs).

What it guarantees, and the earlier runners did not:
  - a seed is set before every case (seed * 100003 + case id), so a case can be reproduced alone;
  - every case is saved with its full trace: retrieved passages, each agent's parsed output and raw
    model output, the citation-check report of every round, every controller decision, the judge's
    output, the final answer, and whether the verdict cache answered instead of a debate;
  - each run writes its own directory under data/results/v2/ with the configuration, the benchmark
    hash, the git commit and the memory namespace; nothing existing is ever overwritten;
  - a run that stops is resumed from its saved cases.

    python experiments/v2/runner.py --queue            # run every experiment in order
    python experiments/v2/runner.py --queue --limit 2  # smoke test on two cases, separate directory
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config  # noqa: E402
from config import KNOWN_CASES_PATH, RESULTS_DIR, set_random_seed  # noqa: E402

V2_DIR = os.path.join(RESULTS_DIR, "v2")
SEEDS = (0, 1, 2)
VARIANTS = {"Full": "all components", "A1": "defense removed", "A2": "citation checker removed",
            "A3": "memory removed", "A4": "one round only"}


# ── small helpers ───────────────────────────────────────────────────────────
def case_seed(seed: int, case_id: int) -> int:
    return int(seed) * 100003 + int(case_id)


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=BASE_DIR, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD") or None, "uncommitted_files": len(run("status", "--short").splitlines())}


def benchmark_cases(limit: int = 0):
    from experiments.benchmark_cases import split_benchmark_by_source_text
    with contextlib.redirect_stdout(io.StringIO()):
        usable, _ = split_benchmark_by_source_text()
    canonical = json.dumps([[c["id"], c["query"], c.get("target_case", ""), c.get("ground_truth_holding", "")] for c in usable],
                           ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return (usable[:limit] if limit else usable), digest


def settings_snapshot() -> dict:
    names = ["MODEL_NAME", "EMBEDDER_NAME", "NLI_MODEL_NAME", "TOP_K_DENSE", "TOP_K_BM25", "RRF_K", "TAU_ACCEPTANCE_THRESHOLD",
             "CCS_VERIFIED_THRESHOLD", "CCS_UNCERTAIN_THRESHOLD", "CCE_FUZZY_THRESHOLD_RATIO", "CCE_OPINION_SCAN_LIMIT",
             "DDC_MAX_ROUNDS", "DDC_CCS_CONTINUE_THRESH", "DDC_MIN_GAPS_TO_CONTINUE", "DDC_CHALLENGE_STRENGTH_THRESH",
             "MEMORY_CACHE_THRESHOLD", "MEMORY_HIGH_RISK_CCS_THRESH", "CCE_GATE_MIN_AUC",
             "LEDGER_EVIDENCE_CHARS", "RETRIEVAL_V2_MODEL", "RETRIEVAL_V2_MAX_TOKENS", "RETRIEVAL_V2_PER_CASE_LIMIT"]
    return {name: getattr(config, name) for name in names}


class RecordingLLM:
    """Passes every call to the real model and keeps what came back."""

    ROLES = (("PROSECUTOR", "prosecutor"), ("DEFENSE ATTORNEY", "defense"), ("REFLECTION AGENT", "reflection"), ("SENIOR JUDGE", "judge"))

    def __init__(self, llm):
        self.llm = llm
        self.calls = []

    def invoke(self, prompt):
        output = self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
        head = str(prompt)[:400]
        role = next((name for marker, name in self.ROLES if marker in head), "other")
        self.calls.append({"role": role, "prompt_characters": len(str(prompt)),
                           "prompt_sha1": hashlib.sha1(str(prompt).encode("utf-8")).hexdigest()[:16], "output": str(output)})
        return output


def passage_brief(doc) -> dict:
    meta = getattr(doc, "metadata", None) or {}
    return {"doc_id": getattr(doc, "doc_id", ""), "case_name": getattr(doc, "case_name", ""), "volume": str(getattr(doc, "volume", "")),
            "first_page": getattr(doc, "first_page", 0), "parent_doc_id": meta.get("parent_doc_id") or str(getattr(doc, "doc_id", "")).split("::chunk::")[0]}


def citation_dicts(citations) -> list:
    return [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in (citations or [])]


def read_cases(path: str) -> dict:
    done, kept, damaged = {}, [], False
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except ValueError:
                    damaged = True  # a line cut off when the process was killed mid-write
                    continue
                done[row["case_id"]] = row
                kept.append(text)
        if damaged:
            # Drop the half-written line, so that the next case is not appended onto it.
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("".join(text + "\n" for text in kept))
    return done


def start_run(experiment: str, run_name: str, root: str, config_record: dict):
    """The run's directory, its saved cases, and whether it is already complete."""
    directory = os.path.join(root, experiment, run_name)
    os.makedirs(directory, exist_ok=True)
    config_path = os.path.join(directory, "run_config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config_record, handle, indent=2, ensure_ascii=False, default=str)
    cases_path = os.path.join(directory, "cases.jsonl")
    return directory, cases_path, read_cases(cases_path), os.path.exists(os.path.join(directory, "summary.json"))


def finish_run(directory: str, rows: list, started: float, extra: dict = None) -> dict:
    cards = [r["legacy_scorecard"] for r in rows if r.get("legacy_scorecard") and r["legacy_scorecard"].get("hr") is not None]

    def mean(key):
        return round(sum(c.get(key, 0.0) or 0.0 for c in cards) / len(cards), 6) if cards else None

    summary = {
        "cases": len(rows), "errors": sum(1 for r in rows if r.get("error")),
        "skipped_by_cache": sum(1 for r in rows if r.get("memory_cache_hit")),
        "seconds_this_session": round(time.time() - started, 1), "seconds_sum_of_cases": round(sum(r.get("seconds", 0.0) for r in rows), 1),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "legacy_metric_means": {k: mean(k) for k in ("hr", "ass", "ar", "ap", "judge_confidence")},
        "note": "legacy_metric_means use the pre-audit definitions and are kept only for continuity; see experiments/v2/score.py",
        **(extra or {}),
    }
    with open(os.path.join(directory, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


# ── components (loaded once for the whole queue) ────────────────────────────
def load_components(load_llm: bool = True) -> dict:
    import chromadb
    from src.data.build_corpus import load_corpus
    from src.data.case_catalog import CaseCatalog
    from src.models.load_model import load_embedder, load_mistral_7b
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.utils.chroma_sync import ensure_chroma_ready

    embedder = load_embedder()
    client = chromadb.PersistentClient(path=ensure_chroma_ready(config.LOCAL_CHROMA_CORPUS_PATH, config.DRIVE_CHROMA_CORPUS_PATH))
    collection = client.get_or_create_collection("cap_authorities")
    corpus = load_corpus(config.CORPUS_JSON_PATH)
    hybrid = HybridRetriever(DenseRetriever(collection, embedder), BM25Retriever.load_index(config.BM25_INDEX_PATH, corpus))
    with open(KNOWN_CASES_PATH, "rb") as handle:
        known_cases = pickle.load(handle)
    parts = {"embedder": embedder, "collection": collection, "hybrid": hybrid, "known_cases": known_cases,
             "catalog": CaseCatalog.load(config.CASE_CATALOG_PATH), "checkers": {}, "corpus": corpus, "retrievers": {"v1": hybrid}}
    if load_llm:
        model, tokenizer, llm = load_mistral_7b()
        parts.update({"model": model, "tokenizer": tokenizer, "llm": llm})
    return parts


RETRIEVALS = {"v1": "Legal-BERT + whitespace BM25, 4 + 4, jurisdiction appended to the dense query (every run up to 7 October 2026)",
              "v2": "retrieval-trained embedder + word-token BM25, at most two passages per case (src/retrieval/retrieval_v2.py)"}


def retriever(parts: dict, name: str):
    """The retriever a run uses. The second index is loaded the first time a run asks for it."""
    if name not in RETRIEVALS:
        raise ValueError(f"unknown retrieval {name!r}; expected one of {sorted(RETRIEVALS)}")
    if name not in parts["retrievers"]:
        from src.retrieval.retrieval_v2 import load_retriever_v2
        parts["retrievers"][name] = load_retriever_v2(parts["corpus"])
    return parts["retrievers"][name]


def retrieval_record(name: str) -> dict:
    record = {"name": name, "meaning": RETRIEVALS[name]}
    if name == "v2":
        from src.retrieval.retrieval_v2 import CANDIDATE_DEPTH, FILL_DEPTH, load_manifest
        record.update({"candidates_per_ranking": CANDIDATE_DEPTH, "fill_depth": FILL_DEPTH, "index": load_manifest()})
    return record


def checker(parts: dict, policy: str):
    """One citation checker per policy, sharing the catalog and (where possible) the NLI model."""
    from src.novel.cce import CitationConfidenceEngine, load_support_gate

    if policy not in parts["checkers"]:
        gate = load_support_gate()
        shared = next((c.nli_model for c in parts["checkers"].values() if c.support_model_name == config.NLI_MODEL_NAME), None)
        needs_other_model = policy == "gated" and gate.get("validated") and gate.get("model") != config.NLI_MODEL_NAME
        parts["checkers"][policy] = CitationConfidenceEngine(
            parts["collection"], parts["embedder"], KNOWN_CASES_PATH, nli_model=None if needs_other_model else shared,
            catalog=parts["catalog"], policy=policy, gate=gate)
    return parts["checkers"][policy]


# ── one debate run ──────────────────────────────────────────────────────────
def initial_state(case: dict, variant: str, protocol: str = "v1") -> dict:
    return {
        # Always stated, so that a run never depends on the configured default protocol.
        "protocol": protocol, "citation_ledger": {}, "seen_passage_ids": [], "round_new_passages": 0, "round_new_supported": 0,
        "query": case["query"], "jurisdiction": case.get("jurisdiction", "US_Federal"), "debate_round": 0, "debate_history": [],
        "retrieved_passages": [], "entity_map": {}, "prosecutor_argument": "", "prosecutor_citations": [],
        "defense_argument": "", "defense_citations": [], "defense_challenges": [], "defense_concede": False,
        "challenge_strength": 0.0, "defense_counter_passages": [], "evidence_overlap_ratio": 0.0,
        "reflection_gaps": [], "reflection_queries": [], "reflection_score": 0.0, "citation_confidence_report": {},
        "avg_ccs": 0.0, "high_risk_citations": [], "ddc_decision": "", "ddc_reason": "", "judge_verdict": "",
        "judge_confidence": 0.0, "judge_verified_citations": [], "judge_uncertain_citations": [], "judge_reasoning": "",
        "memory_cache_hit": False, "cached_answer": None, "final_answer": "", "structured_output": {},
        "use_defense": variant != "A1", "use_cce": variant != "A2", "use_memory": variant != "A3",
    }


def debate_one_case(graph, recorder, case: dict, variant: str, protocol: str = "v1") -> tuple:
    """Run the graph on one question and return (final state, trace of every node)."""
    final = initial_state(case, variant, protocol)
    trace = {"memory": None, "rounds": [], "judge": None}
    current = None
    for chunk in graph.stream(dict(final), stream_mode="updates"):
        for node, update in chunk.items():
            update = update or {}
            final.update(update)
            if node == "memory_check":
                trace["memory"] = {"cache_hit": bool(update.get("memory_cache_hit")), "avoid_list": list(update.get("high_risk_citations") or [])}
            elif node == "retrieve":
                current = {"round": len(trace["rounds"]) + 1, "retrieved": [passage_brief(d) for d in update.get("retrieved_passages", [])]}
                if "round_new_passages" in update:
                    current["new_passages"] = update["round_new_passages"]
                trace["rounds"].append(current)
            elif current is None:
                continue
            elif node == "prosecutor":
                current["prosecutor"] = {"argument": update.get("prosecutor_argument", ""), "citations": citation_dicts(update.get("prosecutor_citations"))}
            elif node == "defense":
                current["defense"] = {
                    "argument": update.get("defense_argument", ""), "citations": citation_dicts(update.get("defense_citations")),
                    "challenges": [str(c) for c in update.get("defense_challenges", [])], "challenge_strength": update.get("challenge_strength"),
                    "concede": update.get("defense_concede"), "evidence_overlap_ratio": update.get("evidence_overlap_ratio"),
                    "counter_passages": [passage_brief(d) for d in update.get("defense_counter_passages", [])]}
            elif node == "cce":
                current["citation_check"] = update.get("citation_confidence_report", {})
                current["avg_ccs"] = update.get("avg_ccs")
                if "citation_ledger" in update:
                    current["newly_supported"] = update.get("round_new_supported")
                    current["ledger_size"] = len(update["citation_ledger"])
            elif node == "reflection":
                current["reflection"] = {"gaps": [str(g) for g in update.get("reflection_gaps", [])],
                                         "queries": [str(q) for q in update.get("reflection_queries", [])], "score": update.get("reflection_score")}
            elif node == "ddc":
                current["controller"] = {"decision": update.get("ddc_decision"), "reason": update.get("ddc_reason")}
            elif node == "judge":
                trace["judge"] = {"verdict": update.get("judge_verdict", ""), "confidence": update.get("judge_confidence"),
                                  "reasoning": update.get("judge_reasoning", ""), "verified": citation_dicts(update.get("judge_verified_citations")),
                                  "uncertain": citation_dicts(update.get("judge_uncertain_citations")), "structured_output": update.get("structured_output", {})}
    trace["llm_calls"] = list(recorder.calls)
    return final, trace


def run_debate(parts: dict, root: str, experiment: str, run_name: str, variant: str, seed: int, cce_policy: str,
               cache_mode: str = "off", namespace: str = None, limit: int = 0, protocol: str = "v1", retrieval: str = "v1") -> dict:
    """`protocol` and `retrieval` default to what every run made before the remaining repairs used,
    so that an earlier experiment started again runs exactly as it did."""
    from experiments.run_25_cases_benchmark import evaluate_single_case
    from experiments.run_ablation import build_ablation_graph
    from src.novel.adaptive_memory import AdaptiveMemory
    from src.novel.ddc import DynamicDebateController

    if protocol not in config.DEBATE_PROTOCOLS:
        raise ValueError(f"unknown debate protocol {protocol!r}")
    cases, digest = benchmark_cases(limit)
    cce = checker(parts, cce_policy)
    hybrid = retriever(parts, retrieval)
    # A smoke test (limit > 0) must never write into the memory namespace of a real run.
    namespace = namespace or (f"v2_{experiment}_{run_name}" if not limit else f"v2smoke{limit}_{int(time.time())}_{experiment}_{run_name}")
    record = {"kind": "debate", "experiment": experiment, "run": run_name, "variant": variant, "variant_meaning": VARIANTS[variant],
              "seed": seed, "case_seed_rule": "seed * 100003 + case_id", "cce_policy": cce_policy, "cce_gate": cce.gate,
              "debate_protocol": protocol, "retrieval": retrieval_record(retrieval),
              "memory_cache_mode": cache_mode, "memory_namespace": namespace, "benchmark_sha256": digest, "cases": len(cases),
              "settings": settings_snapshot(), "git": git_state(), "created": datetime.now().isoformat(timespec="seconds")}
    directory, cases_path, done, complete = start_run(experiment, run_name, root, record)
    if complete and len(done) >= len(cases):
        print(f"  [done already] {experiment}/{run_name}")
        return json.load(open(os.path.join(directory, "summary.json"), encoding="utf-8"))

    recorder = RecordingLLM(parts["llm"])
    memory = AdaptiveMemory(namespace=namespace, cache_mode=cache_mode)
    graph = build_ablation_graph(variant, recorder, hybrid, cce, DynamicDebateController(), memory, parts["embedder"])
    started = time.time()
    print(f"  [run] {experiment}/{run_name}: variant {variant}, seed {seed}, checker {cce_policy}, cache {cache_mode}, "
          f"protocol {protocol}, retrieval {retrieval}, {len(done)}/{len(cases)} cases saved")
    with open(cases_path, "a", encoding="utf-8") as handle:
        for position, case in enumerate(cases, start=1):
            if case["id"] in done:
                continue
            set_random_seed(case_seed(seed, case["id"]))
            recorder.calls = []
            began = time.time()
            row = {"case_id": case["id"], "position": position, "topic": case.get("topic", ""), "query": case["query"],
                   "target_case": case.get("target_case", ""), "seed": seed, "variant": variant}
            try:
                final, trace = debate_one_case(graph, recorder, case, variant, protocol)
                pool = [c for r in trace["rounds"] for side in ("prosecutor", "defense") for c in (r.get(side) or {}).get("citations", [])]
                answer = citation_dicts(final.get("judge_verified_citations"))
                last_check = (trace["rounds"][-1].get("citation_check") if trace["rounds"] else None) or {}
                row.update({
                    "memory_cache_hit": bool(final.get("memory_cache_hit")), "final_answer": final.get("final_answer", "") or "",
                    "judge_confidence": final.get("judge_confidence", 0.0), "rounds": len(trace["rounds"]),
                    "answer_citations": answer, "pool_citations": pool + answer,
                    "final_citation_reports": list(last_check.values()), "trace": trace,
                    "legacy_scorecard": evaluate_single_case(final, case, parts["known_cases"], parts["embedder"]),
                })
                if protocol == "ledger":
                    row["citation_ledger"] = list((final.get("citation_ledger") or {}).values())
            except Exception as error:  # one failed case must not stop a run of many hours
                row["error"] = f"{type(error).__name__}: {error}"
            row["seconds"] = round(time.time() - began, 1)
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            done[case["id"]] = row
            if position % 12 == 0 or position == len(cases):
                print(f"    {position}/{len(cases)} cases, {time.time() - started:.0f}s")
    return finish_run(directory, [done[c["id"]] for c in cases if c["id"] in done], started, {"memory_namespace": namespace})


# ── one baseline run ────────────────────────────────────────────────────────
BASELINE_PROMPT = (
    "[INST] You are an attorney. Answer the following legal question based on the retrieved precedents.\n"
    "Question: {query}\n"
    "Precedents:\n{context}\n"
    "Provide your answer and cite relevant case names in JSON format: "
    "{{\"final_answer\": \"...\", \"citations\": [{{\"case_name\": \"...\"}}]}} [/INST]"
)


def run_baseline(parts: dict, root: str, experiment: str, run_name: str, seed: int, passages: int, characters: int,
                 jurisdiction: str = "US_Federal", limit: int = 0, retrieval: str = "v1") -> dict:
    """Single-agent retrieve-then-answer. Same prompt wording as the earlier baseline.

    The dense query now carries the same jurisdiction string as the debate's, and the context size is
    a parameter, so that evidence input and budget are not confounded with the architecture.
    """
    import re

    from experiments.run_25_cases_benchmark import evaluate_single_case

    cases, digest = benchmark_cases(limit)
    hybrid = retriever(parts, retrieval)
    record = {"kind": "baseline", "experiment": experiment, "run": run_name, "seed": seed, "case_seed_rule": "seed * 100003 + case_id",
              "context_passages": passages, "context_characters_per_passage": characters, "dense_query_jurisdiction": jurisdiction,
              "retrieval": retrieval_record(retrieval),
              "benchmark_sha256": digest, "cases": len(cases), "settings": settings_snapshot(), "git": git_state(),
              "created": datetime.now().isoformat(timespec="seconds")}
    directory, cases_path, done, complete = start_run(experiment, run_name, root, record)
    if complete and len(done) >= len(cases):
        print(f"  [done already] {experiment}/{run_name}")
        return json.load(open(os.path.join(directory, "summary.json"), encoding="utf-8"))
    started = time.time()
    print(f"  [run] {experiment}/{run_name}: baseline, seed {seed}, {passages} passages x {characters} characters")
    with open(cases_path, "a", encoding="utf-8") as handle:
        for position, case in enumerate(cases, start=1):
            if case["id"] in done:
                continue
            set_random_seed(case_seed(seed, case["id"]))
            began = time.time()
            row = {"case_id": case["id"], "position": position, "topic": case.get("topic", ""), "query": case["query"],
                   "target_case": case.get("target_case", ""), "seed": seed, "variant": "baseline"}
            try:
                docs = hybrid.retrieve(case["query"], jurisdiction)
                context = "\n".join(f"[{j + 1}] {d.case_name}: {d.text[:characters]}" for j, d in enumerate(docs[:passages]))
                prompt = BASELINE_PROMPT.format(query=case["query"], context=context)
                raw = str(parts["llm"].invoke(prompt))
                found = re.search(r"\{.*\}", raw, re.DOTALL)
                try:
                    parsed = json.loads(found.group(0)) if found else {}
                except ValueError:
                    parsed = {}
                # Citations written as plain strings are read as well; the first version of this
                # parser kept only objects (see metrics_v2.citations_from_baseline_output).
                from src.evaluation.metrics_v2 import citations_from_baseline_output
                citations = citations_from_baseline_output(raw) or []
                answer = parsed.get("final_answer", raw) if parsed else raw
                result = {"final_answer": answer, "prosecutor_citations": citations, "defense_citations": [], "judge_verified_citations": [],
                          "evidence_overlap_ratio": 0.0, "judge_confidence": 0.70, "debate_round": 0}
                row.update({
                    "memory_cache_hit": False, "final_answer": str(answer), "judge_confidence": None, "rounds": 0,
                    "answer_citations": citations, "pool_citations": citations, "final_citation_reports": [],
                    "trace": {"retrieved": [passage_brief(d) for d in docs], "passages_in_prompt": passages, "raw_output": raw,
                              "json_parsed": bool(parsed)},
                    "legacy_scorecard": evaluate_single_case(result, case, parts["known_cases"], parts["embedder"]),
                })
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["seconds"] = round(time.time() - began, 1)
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            done[case["id"]] = row
    return finish_run(directory, [done[c["id"]] for c in cases if c["id"] in done], started)


# ── paraphrases for the memory experiment ───────────────────────────────────
def write_paraphrases(parts: dict, root: str, limit: int = 0) -> str:
    """One reworded version of every benchmark question (same legal question, different words)."""
    directory = os.path.join(root, "e3_memory")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "paraphrases.json")
    if os.path.exists(path):
        return path
    cases, digest = benchmark_cases(limit)
    rows = []
    for case in cases:
        set_random_seed(case_seed(7, case["id"]))
        prompt = ("[INST] Rewrite the following legal question in different words. Keep its legal meaning exactly the same. "
                  f"Return only the rewritten question.\nQuestion: {case['query']} [/INST]")
        text = str(parts["llm"].invoke(prompt)).strip().split("\n")[0].strip().strip('"')
        rows.append({"case_id": case["id"], "question": case["query"], "paraphrase": text})
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"benchmark_sha256": digest, "seed": 7, "rows": rows}, handle, indent=2, ensure_ascii=False)
    return path


# ── the queue ───────────────────────────────────────────────────────────────
def queue(limit: int = 0) -> None:
    """Every LLM experiment, most informative first. Safe to stop and start again."""
    root = V2_DIR if not limit else os.path.join(V2_DIR, f"_smoke_{limit}_cases")
    os.makedirs(root, exist_ok=True)
    # Only one queue may run at a time (two would write the same files and share one GPU). The lock
    # is held by the operating system for as long as this process lives, so a crash releases it.
    lock = open(os.path.join(root, ".queue.lock"), "a+")
    try:
        import msvcrt
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except ImportError:
        pass  # not Windows: no lock
    except OSError:
        print("Another queue is already running on this directory; this one exits.")
        return
    print(f"Queue root: {root}  (started {datetime.now().isoformat(timespec='seconds')})")
    parts = load_components()
    seeds = SEEDS if not limit else (0,)

    def rescore():
        try:
            from experiments.v2 import score
            score.main(root, parts)
        except Exception as error:  # a reporting error must not stop the experiments
            print(f"  [score] failed: {type(error).__name__}: {error}")

    def step(function, *args, **kwargs):
        function(parts, root, *args, limit=limit, **kwargs)
        rescore()

    # 1. Baselines: same dense query as the debate, at the old and at the debate's context size.
    for seed in seeds:
        step(run_baseline, "baseline_4x300", f"seed{seed}", seed, 4, 300)
        step(run_baseline, "baseline_8x500", f"seed{seed}", seed, 8, 500)
    # 2. E0, first seed: the system as it was, minus the verdict cache (no memory nodes, pre-audit checker).
    step(run_debate, "e0_legacy_checker_no_memory", "seed0", "A3", seeds[0], "legacy")
    # 3. E1/E4, first seed: the repaired system and its ablations. The full system's first seed runs
    #    with the exact-question cache on and a kept namespace, which the memory experiment reads.
    order = ("Full", "A3", "A4", "A1", "A2")
    for variant in order:
        first = variant == "Full"
        step(run_debate, f"e4_repaired_{variant}", "seed0", variant, seeds[0], "gated",
             cache_mode="exact" if first else "off", namespace="v2_e3_memory_pass1" if (first and not limit) else None)
    step_paraphrases = write_paraphrases(parts, root, limit)
    print(f"  [paraphrases] {step_paraphrases}")
    try:
        from experiments.v2 import e3_memory
        e3_memory.main(root, parts)
    except Exception as error:
        print(f"  [e3] failed: {type(error).__name__}: {error}")
    # 4. Remaining seeds.
    for seed in seeds[1:]:
        step(run_debate, "e0_legacy_checker_no_memory", f"seed{seed}", "A3", seed, "legacy")
    for seed in seeds[1:]:
        for variant in order:
            step(run_debate, f"e4_repaired_{variant}", f"seed{seed}", variant, seed, "gated")
    rescore()
    if not limit:
        # Tells the watchdog (experiments/v2/keep_running.ps1) that there is nothing left to restart.
        with open(os.path.join(root, "QUEUE_FINISHED"), "w", encoding="utf-8") as handle:
            handle.write(datetime.now().isoformat(timespec="seconds") + "\n")
    print("Queue finished.")


# ── the final queue: the system with the remaining repairs ──────────────────
FINAL_SEEDS = (0, 1, 2)
FINAL_FLAG = "FINAL_QUEUE_FINISHED"


def final_queue(limit: int = 0) -> None:
    """The system after repairs 6 (ledger protocol) and 7 (retrieval v2), and what it is compared with.

    f2_final_Full      both repairs: the final system. Three seeds, because the share of answers
                       resting on the target varies by about seven points between seeds of a debate.
    baseline_r7_*      the single-agent baseline on the same retrieval as the final system, at both
                       context sizes, three seeds. A debate with a better retriever must be compared
                       with a baseline that has it too.
    f1_ledger_Full     repair 6 alone (earlier retrieval), one seed: what the protocol contributes.
    f3_retrieval_Full  repair 7 alone (earlier protocol), one seed: what the retrieval contributes.

    Runs in the same results directory as the earlier queue and holds the same lock, so the two can
    never run together. Safe to stop and start again.
    """
    root = V2_DIR if not limit else os.path.join(V2_DIR, f"_smoke_final_{limit}_cases")
    os.makedirs(root, exist_ok=True)
    lock = open(os.path.join(V2_DIR if not limit else root, ".queue.lock"), "a+")
    try:
        import msvcrt
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except ImportError:
        pass
    except OSError:
        print("Another queue is already running on this directory; this one exits.")
        return
    print(f"Final queue root: {root}  (started {datetime.now().isoformat(timespec='seconds')})")
    parts = load_components()
    seeds = FINAL_SEEDS if not limit else (0,)

    def rescore():
        try:
            from experiments.v2 import score
            score.main(root, parts)
        except Exception as error:
            print(f"  [score] failed: {type(error).__name__}: {error}")

    def step(function, *args, **kwargs):
        function(parts, root, *args, limit=limit, **kwargs)
        rescore()

    def baselines(seed):
        step(run_baseline, "baseline_r7_4x300", f"seed{seed}", seed, 4, 300, retrieval="v2")
        step(run_baseline, "baseline_r7_8x500", f"seed{seed}", seed, 8, 500, retrieval="v2")

    def final(seed):
        step(run_debate, "f2_final_Full", f"seed{seed}", "Full", seed, "gated", protocol="ledger", retrieval="v2")

    # First seed of everything that is needed for the headline comparison, then the attribution
    # runs, then the remaining seeds: a stop at any point leaves the most informative runs complete.
    baselines(seeds[0])
    final(seeds[0])
    step(run_debate, "f1_ledger_Full", "seed0", "Full", seeds[0], "gated", protocol="ledger", retrieval="v1")
    for seed in seeds[1:]:
        baselines(seed)
        final(seed)
    step(run_debate, "f3_retrieval_Full", "seed0", "Full", seeds[0], "gated", protocol="v1", retrieval="v2")
    rescore()
    try:
        from experiments.v2 import standard_metrics
        standard_metrics.main(root, parts)
    except Exception as error:
        print(f"  [standard metrics] failed: {type(error).__name__}: {error}")
    if not limit:
        with open(os.path.join(root, FINAL_FLAG), "w", encoding="utf-8") as handle:
            handle.write(datetime.now().isoformat(timespec="seconds") + "\n")
    print("Final queue finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-audit experiment queue")
    parser.add_argument("--queue", action="store_true", help="run every experiment of the first post-audit queue in order")
    parser.add_argument("--final", action="store_true", help="run the final system (repairs 6 and 7) and its comparisons")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N benchmark cases (smoke test; separate directory)")
    arguments = parser.parse_args()
    if arguments.queue:
        queue(arguments.limit)
    elif arguments.final:
        final_queue(arguments.limit)
    else:
        parser.print_help()
