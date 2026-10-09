# LexAgent paper (IEEE conference format)

| File | Purpose |
|---|---|
| `lexagent_ieee.pdf` | The compiled paper. |
| `lexagent_ieee.tex` | Source (IEEEtran, two columns). |
| `figures/*.tex` | The figures, drawn in TikZ / pgfplots: architecture, pipeline state machine, citation-checker flow, and the result charts. |
| `make_results.py` | Computes every number in the paper from the saved result files and writes `generated/results.tex` and `generated/table_rows.tex`. |
| `generated/` | Output of `make_results.py` (macros, table rows, chart data, `claims_check.txt`). Do not edit by hand. |
| `build.ps1` | Runs `make_results.py`, then compiles the PDF. |
| `render_pages.py` | Renders the PDF pages to PNG for a visual check. |

## No number is typed by hand

The text, the tables and the charts read their values from `generated/results.tex`.
After re-running any experiment, rebuild and the paper follows the new result files:

```powershell
powershell -ExecutionPolicy Bypass -File paper\build.ps1
```

`make_results.py` reads, from `data/`: the download and chunk ledgers, the corpus manifest, the case
catalog, `results/benchmark_100_cases_results.json`, `results/baseline_standard_rag_results.json`,
`results/citation_stress_test_results.json` (and the one from before the citation-checker change),
`results/cce_evidence_selection_study.json` and `results/ablation_summary.json` (the first
evaluation); `results/forensic_audit/analysis_existing_results.json` and `diagnostics.json` (the
audit); and `results/v2/scores.json`, `v2/standard_metrics.json`, `v2/e2_verifier_validation/`,
`v2/e3_memory/`, `v2/e5_retrieval/` and the saved cases of `v2/e4_repaired_Full` and
`v2/f2_final_Full` (the experiments after the audit).
It also computes the paired-bootstrap intervals (10,000 resamples, seed 42) and the Wilson intervals
of the first evaluation; the intervals and permutation tests of the post-audit runs are taken from
`scores.json` as `experiments/v2/score.py` wrote them.

## The paper follows the result files

Rebuilding the paper picks up whatever `results/v2/scores.json` and `standard_metrics.json` hold:

- Every table states how many seeds each value rests on.
- `make_results.py` also writes `generated/claims_check.txt`. It lists the statements the running
  text makes about the post-audit runs ("the difference from the four-passage baseline is not
  significant", "no ablation variant differs significantly from the full system", ...) and checks
  each against the current result files. A statement the new results no longer support is printed
  as a WARNING and marked `REWRITE`; the sentence must then be rewritten by hand. The numbers
  update on their own, the interpretation does not.

## Toolchain

The paper is compiled with [Tectonic](https://tectonic-typesetting.github.io), a single
self-contained executable; no TeX distribution is installed on the machine.
`build.ps1` expects it at `paper\.tectonic\tectonic.exe` (not committed). To get it, download
`tectonic-<version>-x86_64-pc-windows-msvc.zip` from the Tectonic releases page and unzip it there.
The first build downloads the LaTeX packages it needs into `paper\.tectonic\cache`.

The source is written for XeLaTeX (it loads the TeX Gyre fonts through `fontspec`), so it should also
compile on a full TeX Live installation or on Overleaf with the compiler set to XeLaTeX. Only the
Tectonic build was tested.

## Before submission

- Replace the placeholder author block at the top of `lexagent_ieee.tex`.
- The related-work section follows the project's literature review. Every reference in the list was
  checked against its arXiv, ACL Anthology or journal page; items of the review that could not be
  verified as publications (internal notes, blog posts, model-repository pages) were left out.
