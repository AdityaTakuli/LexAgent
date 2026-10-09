# Post-audit repairs and experiments

The forensic audit (`data/results/forensic_audit/`) found that the earlier results were shaped by a
verdict cache that answered different questions, by measures that did not measure what their names
said, and by a citation checker that could not tell a correct holding from a wrong one. Four repairs
were approved and implemented. Everything here runs on the repaired code and writes to
`data/results/v2/`; **no earlier result file is touched**.

## The four repairs

| # | Repair | Where |
|---|---|---|
| 1 | Every run is seeded per case and saves the full trace of every case (passages, each agent's parsed and raw output, the citation check of every round, controller decisions, judge output, cache-hit flag). Runs resume; nothing is overwritten. | `experiments/v2/runner.py` |
| 2 | The verdict cache answers only the **same question** (normalised text); a hit restores the citations and the confidence, not only the answer. It is **off in evaluation runs**. The old similarity rule remains selectable only to reproduce earlier runs. | `src/novel/adaptive_memory.py`, `config.MEMORY_CACHE_MODE` |
| 3 | Measures that say what they measure: existence through the case catalog, reporter-cite correctness, target cited (by the answer / by any agent), citation counts shown beside it and never divided into it, uncited case mentions, answers without an existing authority, skipped debates as their own category. Baseline uses the debate's dense query and is run at two context sizes. | `src/evaluation/metrics_v2.py`, `experiments/v2/score.py` |
| 4 | The citation checker reports **existence** and **holding support** separately. The support verdict steers the judge, the controller and the memory only if a support model has passed validation on held-out labelled citations (the gate). `config.CCE_POLICY="legacy"` reproduces the earlier checker. | `src/novel/cce.py`, `data/cce_support_gate.json` |

The runs named `e4_repaired_*` use these four repairs only. In them the judge still reads the last
round's report, citations are still replaced from round to round, the controller still runs every
debate to the maximum, and retrieval is still Legal-BERT: those defects are part of what the
`e4_repaired_*` runs measure.

## The remaining repairs (7 October 2026)

After the first results the remaining repairs of the audit were approved "if they are important".
The saved traces of `e4_repaired_Full/seed0` decided which are:

| # | Repair | Evidence from the traces | Done |
|---|---|---|---|
| 5 | Memory risk list: write only existence failures, look them up by relevance | A risk list was handed to 71 of 96 debates, but it held 6 distinct names in total and never the target of the question; removing memory (A3) changed no measure | **no** — no measurable effect, so left as it is |
| 6 | Keep what was verified, let the judge read it, let the controller stop | Of 19 debates that cited the target and did not use it, 12 had it supported in an earlier round and lost it; 15 answers rest on nothing although a supported citation was available in the last round; the defense's self-reported challenge strength was above the threshold in all 288 rounds (0.8 in 287), so every debate ran three rounds; the judge's prompt set score thresholds (0.75, 0.40) that half of the supported citations do not meet | **yes** — `config.DEBATE_PROTOCOL = "ledger"` |
| 7 | Retrieval trained for retrieval | The target opinion is not among the passages of the first round for 20 of 96 questions; Legal-BERT finds it in its top 4 for 39, BM25 for 74 | **yes** — `src/retrieval/retrieval_v2.py` |

**Repair 6, the ledger protocol** (`src/novel/ledger.py`, and the ledger branches of `cce.py`,
`ddc.py`, `judge_agent.py`, `lexagent_graph.py`):

- A per-debate ledger keeps, for every decision cited in any round, the best verdict it received and
  the passage that supported it. A later, worse verdict never replaces an earlier, better one.
- The judge reads the ledger of all rounds, with each verdict in words (SUPPORTED, NOT SHOWN, NOT
  FOUND, ...) and the supporting passage. Its instructions refer to the verdicts, not to score
  thresholds. The program still decides what the answer rests on: only decisions the ledger holds as
  supported count; a decision the judge relied on whose holding was not shown is reported as
  uncertain instead of being dropped. A name written by the judge must name the same two parties as
  the ledger entry (the earlier rule accepted any two shared words).
- The controller continues only while no supported citation exists, a gap remains, and the last
  round brought a new passage or a newly supported citation. The mean and minimum score and the
  defense's self-reported strength no longer force a round.
- Deviation from the audit's plan, on the cautious side: the plan would have let the judge rely on
  any citation that exists and merely annotate missing support. Here an answer still rests only on
  supported citations, so "target cited by the answer" keeps its meaning across all runs.

**Repair 7, retrieval v2** (`runs/build_retrieval_v2.py` builds it into `data/retrieval_v2/`; the
Legal-BERT collection is kept and the citation checker, which was validated on it, still uses it):
`BAAI/bge-base-en-v1.5` on the question alone (no jurisdiction string), BM25 over word tokens,
reciprocal-rank fusion of the first 8 results of each, at most 2 passages per case, 8 passages.
Where one long opinion dominates both rankings the limit would leave as few as 2 passages; the list
is then filled, under the same limit, from the first 50 results of each ranking, and the passages
already ranked keep their places. The deployed retriever finds the target opinion among its 8
passages for 82 of 96 questions (76 before), among the first 4 for 77 (62 before) and ranks it
first for 60 (29 before). The depth and the per-case limit were fixed on design grounds, not by
picking the best row of the retrieval-only experiment E5 (whose best row reaches 84).

Every run states its protocol and its retrieval in `run_config.json`. Both default to the earlier
behaviour (`"v1"`), so an earlier experiment started again runs exactly as it did, and
`config.DEBATE_PROTOCOL` still defaults to `"v1"`.

### Correction (8 October 2026): the baseline's citations were under-counted

Found after the final runs, from an implausible number (69% of eight-passage baseline answers "cite
nothing"). The baseline prompt asks for `"citations": [{"case_name": ...}]`; the model often returns
`"citations": ["Miranda v. Arizona", ...]` instead. The baseline parser of `runner.py`, which was
taken over from the pre-audit baseline, kept objects only and recorded such answers as citing
nothing. Share of answers affected: about 10% with four passages, 56% (earlier retrieval) and 69%
(retrieval v2) with eight.

- Every run saves the model's raw output, so nothing was re-run. `score.py`
  (`reparse_baseline_citations`) reads the citations again from the raw output with
  `metrics_v2.citations_from_baseline_output`; the citations as the run saved them stay in each row
  as `answer_citations_as_run`, and `scores.json` records both readings per baseline run under
  `baseline_citation_parsing`. The saved `cases.jsonl` files are unchanged.
- The scores as first written are kept: `v2/before_baseline_parser_fix__*.json|md`.
- `runner.run_baseline` now reads both forms, for any future run.
- Debate agents wrote objects in every call examined (290 prosecutor, 290 defense, 285 judge
  outputs of the final runs), so debate numbers did not change.
- The pre-audit baseline (`baseline_standard_rag_results.json`) had the same defect and cannot be
  corrected: its outputs were not saved.

Effect on "answer rests on the target": baseline 4 passages 60.8% → 67.7%; 8 passages 34.7% →
74.0%; on retrieval v2, 75.0% → 81.6% and 30.6% → 77.1%. Two statements made before the correction
are therefore withdrawn: that the repaired debate is level with the best baseline, and that a single
agent gets worse with more context.

### Final result

| | Baseline 4 passages | Baseline 8 passages | Final debate |
|---|---|---|---|
| Answer rests on the target (retrieval v2, 3 seeds each) | 81.6% | 77.1% | 62.5% |
| Answers citing a non-existent case | 0% | 2.8% | 0% |
| Seconds per question | 3 | 3 | 43 |

The repairs make the debate a much better debate (34.4% → 62.5% of answers on the target, 116 s →
43 s per question), and it is still significantly behind a single agent on the same retrieval
(−19.1 and −14.6 points). Full tables: `v2/REPORT.md` and `v2/STANDARD_METRICS.md`.

### Seeds: what was dropped and what was kept

The first queue was stopped on purpose after `e4_repaired_Full/seed1`. Its remaining runs (seeds 1
and 2 of the four ablations, seed 2 of the repaired system) would have described a system that is no
longer the final one. Three seeds are kept where they decide a conclusion: the share of answers
resting on the target differs by about seven points between seeds of the same debate
(`e0_legacy_checker_no_memory`: 30.2, 30.2, 42.7), so the final system and its baselines get three
seeds each, and the two attribution runs get one.

## The experiments

| | What | Needs the LLM | Script | Output |
|---|---|---|---|---|
| E2 | Which support model, if any, passes the gate? Fixed rule: pick by development AUC, open the gate only if that candidate reaches AUC ≥ 0.90 on the test split. | only for the "generator as verifier" candidate | `e2_verifier_validation.py` | `v2/e2_verifier_validation/results.json`, `data/cce_support_gate.json` |
| E2b | The validated model against near-miss holdings (post-hoc, does not change the gate) | no | `e2b_near_miss_probe.py` | `v2/e2_verifier_validation/near_miss_probe.json` |
| E5 | Retrieval only: target opinion in the top k for alternative embedders, tokenisation and fusion | no | `e5_retrieval.py` | `v2/e5_retrieval/` |
| E0 | The debate **as it was**, without memory (pre-audit checker), 3 seeds, with traces — the measurement the old data could not give | yes | `runner.py --queue` | `v2/e0_legacy_checker_no_memory/seed*/` |
| — | Baseline at 4 × 300 and 8 × 500 characters of context, same dense query as the debate, 3 seeds | yes | `runner.py --queue` | `v2/baseline_*/seed*/` |
| E1 / E4 | The repaired system: Full, A1 (no defense), A2 (no checker), A3 (no memory), A4 (one round), 3 seeds each, per-case output | yes | `runner.py --queue` | `v2/e4_repaired_*/seed*/` |
| E3 | The verdict cache on a workload with repeats: old rule against new rule on exact repeats, format variants and paraphrases | paraphrases only | `e3_memory.py` (called by the queue) | `v2/e3_memory/` |
| F2 | **The final system**: repairs 1–4, 6 and 7. 3 seeds | yes | `runner.py --final` | `v2/f2_final_Full/seed*/` |
| — | Baseline on the retrieval of the final system, 4 × 300 and 8 × 500 characters, 3 seeds | yes | `runner.py --final` | `v2/baseline_r7_*/seed*/` |
| F1 | Repair 6 alone (ledger protocol, earlier retrieval), 1 seed | yes | `runner.py --final` | `v2/f1_ledger_Full/seed0/` |
| F3 | Repair 7 alone (retrieval v2, earlier protocol), 1 seed | yes | `runner.py --final` | `v2/f3_retrieval_Full/seed0/` |

`score.py` turns every saved run into `v2/scores.json` and `v2/REPORT.md` (paired comparisons with
bootstrap intervals and permutation tests, seed-to-seed variation). The queue calls it after every
run, so the report always shows what exists so far. Only finished runs enter a comparison.

`standard_metrics.py` computes, for every finished run, the measures other papers report, the same
way for the baseline and for the debate: retrieval Hit@k and MRR; the share of citations and of
answers that name a non-existent case; citation recall and precision in the ALCE sense (share of
answer sentences supported by a cited opinion; share of citations that support a sentence), judged
by the validated support model; a grounding outcome per answer (grounded, misgrounded, without
authority); authority accuracy; ROUGE-L, BERTScore and token F1 against the reference holding;
seconds and generator calls per question. Output: `v2/standard_metrics.json`,
`v2/STANDARD_METRICS.md`. It needs no generator and runs on a CPU in minutes.

## Run

```powershell
.venv\Scripts\python.exe experiments\v2\e2_verifier_validation.py     # decides the gate (about 10 minutes)
.venv\Scripts\python.exe experiments\v2\e5_retrieval.py               # retrieval only (about 40 minutes)
.venv\Scripts\python.exe experiments\v2\runner.py --queue             # the first queue (repairs 1-4), about 40 hours on one RTX 5090
.venv\Scripts\python.exe experiments\v2\runner.py --queue --limit 1   # smoke test in a separate directory
.venv\Scripts\python.exe runs\build_retrieval_v2.py                   # the second retrieval index (once; 3 min CPU + 5 min GPU)
.venv\Scripts\python.exe experiments\v2\runner.py --final             # the final system, its baselines and the standard measures
.venv\Scripts\python.exe experiments\v2\runner.py --final --limit 2   # smoke test in a separate directory
.venv\Scripts\python.exe experiments\v2\score.py                      # re-score what exists, at any time
.venv\Scripts\python.exe experiments\v2\standard_metrics.py           # the standard measures, at any time
```

The queue can be stopped and started again: finished runs are skipped and an interrupted run
continues after its last saved case.

### Keeping it running unattended

The queue takes about two days. A Windows scheduled task keeps it alive without any terminal,
editor or assistant session:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\v2\keep_running_install.ps1            # install
powershell -ExecutionPolicy Bypass -File experiments\v2\keep_running_install.ps1 -Remove    # stop the queue and remove the task
```

- Task `LexAgentV2Queue` runs `keep_running.ps1` every 5 minutes, silently. If no queue is running
  and the queue has not finished, it starts one as a detached process; otherwise it does nothing.
- Since 7 October 2026 the queue it keeps alive is the final one (`runner.py --final`, log
  `data/logs/v2/final_queue.log`). The first queue was stopped at a run boundary with
  `stop_queue_after_run.ps1`; it can be continued by hand with `runner.py --queue`.
- Only one queue can run: both hold an operating-system lock on `data/results/v2/.queue.lock`.
- When the final queue finishes it writes `data/results/v2/FINAL_QUEUE_FINISHED`, and the task
  removes itself.
- To pause without uninstalling: create an empty file `data/results/v2/QUEUE_PAUSED` and stop the
  running queue; delete the file to continue.
- Progress: `data/logs/v2/status.txt` (starts and restarts), `data/logs/v2/queue.log`, and
  `data/results/v2/REPORT.md` (results so far).
- The task runs only while the Windows user is logged on. After a reboot it continues at the next
  logon. A crash or a restart costs at most the one case that was in progress.

## Reading a run directory

```
data/results/v2/<experiment>/seed<n>/
    run_config.json   seed, variant, checker policy and gate, cache mode, memory namespace,
                      benchmark hash, settings, git commit
    cases.jsonl       one line per case: question, final answer, citations the answer rests on,
                      every citation made, the trace of every round, raw model outputs,
                      whether the cache answered, seconds, and the pre-audit scorecard
    summary.json      written when the run is complete
```
