# LexAgent comparison matrix

One set of criteria applied to every system: commercial legal tools, general anti-hallucination
methods, legal AI research systems, and LexAgent's own configurations. The LaTeX version for the
paper is `paper/tables/comparison_matrix.tex`.

Compiled 28 October 2026. Sources are listed at the end. Where a paper or vendor does not report
something, the cell says `?`. It is never filled in by guess.

**Legend:** ● yes · ◐ partial · ○ no · ? not reported · — not applicable

---

## 1. Capability matrix

| # | Criterion | What it asks |
|---|---|---|
| C1 | Retrieval | What does the system read before it answers? |
| C2 | Agents | One model, cooperating roles, or roles that contest each other? |
| C3 | Independent verifier | Is the checker a different model or a symbolic check, not the generator grading itself? |
| C4 | Citation existence | Is each cited authority checked against a catalog of real authorities? |
| C5 | Claim support | Is the cited text checked to see whether it supports the claim made about it? |
| C6 | When checked | Never, after the answer (post-hoc), or inside the generation loop (in-loop)? |
| C7 | Repairs | Does a failed check lead to a corrected answer, not just a flag or a rejection? |
| C8 | Cross-query memory | Does anything learned on one query carry over to later queries? |
| C9 | Open weights | Can it run entirely on open models on your own hardware? |

| System | Domain | C1 Retrieval | C2 Agents | C3 | C4 | C5 | C6 | C7 | C8 | C9 |
|---|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Commercial legal tools** | | | | | | | | | | |
| Lexis+ AI | US law | proprietary RAG | ? | ? | ◐¹ | ? | ? | ? | ? | ○ |
| Westlaw AI-Assisted Research | US law | proprietary RAG | ? | ? | ◐¹ | ? | ? | ? | ? | ○ |
| Ask Practical Law AI | US law | proprietary RAG | ? | ? | ? | ? | ? | ? | ? | ○ |
| CoCounsel (Thomson Reuters) | US law | Westlaw-grounded | ? | ? | ◐¹ | ? | ? | ? | ? | ○ |
| Harvey | multi | Lexis content (2025 alliance) | ? | ? | ◐¹ | ? | ? | ? | ? | ○ |
| **General anti-hallucination methods** | | | | | | | | | | |
| Self-RAG | general | dense, on demand | single | ○ | ○ | ◐² | in-loop | ◐ | ○ | ● |
| CRAG | general | dense + web fallback | single | ●³ | ○ | ○³ | before generation | ◐ | ○ | ● |
| Chain-of-Verification | general | none | single | ○ | ○ | ◐² | post-hoc | ● | ○ | ● |
| SelfCheckGPT | general | none | single | ○ | ○ | ◐² | post-hoc | ○ | ○ | ● |
| Multi-agent debate (Du et al.) | general | none | cooperative | ○ | ○ | ○ | — | ● | ○ | ○ |
| Multi-agent debate (Liang et al.) | general | none | adversarial | ○ | ○ | ○ | — | ● | ○ | ○ |
| **Legal AI research systems** | | | | | | | | | | |
| DISC-LawLLM | Chinese law | retrieval module | single (fine-tuned) | ○ | ○ | ○ | — | ○ | ○ | ● |
| ChatLaw | Chinese law | KG + retrieval | cooperative | ○ | ○ | ○ | — | ○ | ○ | ● |
| LeAK-GraphRAG | Chinese legal scholarship | hybrid (semantic + graph) | single | ○ | ○ | ○ | — | ○ | ○ | ? |
| HalluGraph | legal RAG | (given context) | single | ● | ◐⁴ | ● | post-hoc | ○ | ○ | ? |
| Falkor-IRAC | Indian judgments | graph (IRAC) | verifier agent | ● | ● | ◐ | post-hoc gate | ○⁵ | ○ | ? |
| AgentCourt | Chinese civil courts | dense (BGE-M3) | adversarial | ○ | ○ | ○ | — | ◐ | ◐⁶ | ○⁷ |
| PROClaim | biomedical claims | progressive RAG | adversarial | ◐ | ○ | ◐ | in-loop | ● | ○ | ○ |
| **LexAgent (this project)** | | | | | | | | | | |
| Single-agent baseline (ours) | US Supreme Court | hybrid (BGE + BM25, RRF) | single | ○ | ○ | ○ | — | ○ | ○ | ● |
| LexAgent, first version | US Supreme Court | hybrid (Legal-BERT + BM25) | adversarial | ◐⁸ | ● | ◐⁸ | in-loop | ● | ◐⁹ | ● |
| **LexAgent, final** | US Supreme Court | hybrid (BGE + BM25, RRF) | adversarial | ● | ● | ● | in-loop | ●¹⁰ | ◐⁹ | ● |

Notes
1. Answers link into a citator-backed database (Shepard's, KeyCite), but Magesh et al. still found
   hallucinations in 17–33% of answers, so existence linking does not guarantee correct use.
2. The checker is the generator itself (reflection tokens, verification questions or resampling).
3. CRAG's T5 evaluator grades the *retrieved documents*, not the claims in the answer.
4. Checks that response entities appear in the source context, not in an authority catalog.
5. An output with no supporting graph path is rejected, not refined.
6. A knowledge base of experience, updated after each of 1,000 simulated cases. It does not track
   citation reliability.
7. Agents run on GPT-4o-mini. **This differs from Table I of our paper, which marks AgentCourt as
   open. Table I should be corrected.**
8. The first checker was a DeBERTa-v3 cross-encoder whose support score was never validated
   (stress-test AUC 0.70).
9. Same-question verdict cache plus a citation risk index. Removing memory (ablation A3) changed no
   measure significantly.
10. The controller can start further rounds with follow-up queries. In the final runs it almost
    never does (mean 1.01 rounds), because a supported citation usually exists after round one.

**Where LexAgent stands on capabilities.** It is the only system in the matrix that combines
adversarial debate, a validated independent verifier that checks both existence and support, and
an in-loop check on open models. Each piece exists elsewhere: Falkor-IRAC and HalluGraph verify,
AgentCourt and PROClaim argue, and Self-RAG checks in-loop. None of them combines all three.

---

## 2. Reported results (each on its own benchmark)

**These numbers are not comparable across rows.** Each system was measured on a different test
set, with a different definition of "correct". The table shows what each source reports, so a
reader can see which kind of evidence exists for each system.

| System | Benchmark | Headline result |
|---|---|---|
| Lexis+ AI | Magesh et al., 202 US legal queries | 65% accurate, ~17% hallucinated |
| Westlaw AI-Assisted Research | same | 42% accurate, ~33% hallucinated |
| Ask Practical Law AI | same | 19–20% accurate, ~17% hallucinated, 62% incomplete |
| CoCounsel, Harvey | — | no independent benchmark (access restricted, per Magesh et al.) |
| ChatLaw | LawBench, Chinese bar exam | +7.73% over GPT-4 on LawBench; +11 points on the exam |
| HalluGraph | legal RAG detection | AUC 0.979 (structured documents), ≈0.89 (generative tasks) |
| Falkor-IRAC | 51 Indian Supreme Court judgments | qualitative only, no numbers |
| AgentCourt | CourtBench | +12.1% after 1,000 simulated cases |
| PROClaim | Check-COVID | 81.7% accuracy (+10.0 pp over plain debate) |
| **LexAgent, final** | 96 US Supreme Court questions, 3 seeds | 62.5% of answers rest on the target decision; 0% cite a non-existent case; 43 s per question |
| LexAgent citation checker | held-out citation pairs | test AUC 0.991 (first checker: 0.70) |

---

## 3. LexAgent against its own baselines (same questions, same measures)

This is the only part of the matrix where rows are directly comparable: 96 questions, the same
corpus and the same scoring code.

| Configuration | Retrieval | Seeds | Target in answer | Target by any agent | Answers on no existing authority | Non-existent cases in answers | Answer similarity | s / question |
|---|---|:-:|--:|--:|--:|--:|--:|--:|
| Baseline, 4 passages | v2 (BGE + BM25) | 3 | **81.6%** | 81.6% | **1.0%** | 0 | 0.747 | **3** |
| Baseline, 8 passages | v2 | 3 | 77.1% | 77.1% | 10.1% | 14 | **0.750** | 3 |
| **LexAgent, final** (ledger + v2) | v2 | 3 | 62.5% | 83.0% | 11.5% | **0** | 0.689 | 43 |
| Ledger protocol only | earlier | 1 | 57.3% | 76.0% | 13.5% | 0 | 0.677 | 40 |
| Retrieval v2 only | v2 | 1 | 64.6% | **92.7%** | 10.4% | 0 | 0.666 | 115 |
| Baseline, 4 passages | earlier (Legal-BERT + BM25) | 3 | 67.7% | 67.7% | 1.7% | 0 | 0.733 | 4 |
| Baseline, 8 passages | earlier | 3 | 74.0% | 74.0% | 3.5% | 6 | 0.746 | 3 |
| Repaired debate | earlier | 2 | 63.0% | 83.9% | 14.1% | 0 | 0.668 | 115 |
| First debate, cache off | earlier | 3 | 34.4% | 84.7% | 43.1% | 0 | 0.665 | 116 |
| A1: no defense | earlier | 1 | 70.8% | 80.2% | 9.4% | 0 | 0.657 | 36 |
| A2: no checker | earlier | 1 | 65.6% | 81.2% | 7.3% | 0 | 0.666 | 113 |
| A3: no memory | earlier | 1 | 65.6% | 87.5% | 15.6% | 0 | 0.660 | 116 |
| A4: one round only | earlier | 1 | 62.5% | 77.1% | 11.5% | 0 | 0.656 | 42 |

Final debate minus the baselines on "target in answer", same retrieval: −19.1 pp against 4
passages (95% CI −28.1 to −10.1, p < 0.001) and −14.6 pp against 8 passages (−23.3 to −5.9,
p = 0.002).

**Reading this table:**
- LexAgent's agents find the target decision as often as the baselines do, or more often (83–93%
  "by any agent"). The loss happens between finding it and resting the answer on it.
- The debate's advantage is safety. No debate answer rested on a non-existent case. The 8-passage
  baseline put 14 non-existent cases into its answers.
- The single-agent baseline is ahead on accuracy and is about 14× faster.

---

## Systems considered but not included

HyPA-RAG and LRAS are cited in the paper but left out here. Their papers, as read for this matrix,
did not report enough to fill most cells without guessing.

## Sources

- Magesh et al., "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools",
  *J. Empirical Legal Stud.* 22 (2025). https://arxiv.org/abs/2405.20362
- AgentCourt, arXiv:2408.08089 (Findings of ACL 2025). https://arxiv.org/abs/2408.08089
- PROClaim, arXiv:2603.28488. https://arxiv.org/abs/2603.28488
- HalluGraph, arXiv:2512.01659. https://arxiv.org/abs/2512.01659
- Falkor-IRAC, arXiv:2605.14665. https://arxiv.org/abs/2605.14665
- ChatLaw, arXiv:2306.16092. https://arxiv.org/abs/2306.16092
- LeAK-GraphRAG, doi:10.3233/FAIA251624
- Self-RAG (ICLR 2024), CRAG (arXiv:2401.15884), Chain-of-Verification (ACL Findings 2024),
  SelfCheckGPT (EMNLP 2023), Du et al. (ICML 2024), Liang et al. (EMNLP 2024), DISC-LawLLM
  (arXiv:2309.11325): as cited in `paper/lexagent_ieee.tex`.
- Harvey and CoCounsel features: vendor material and secondary reviews. These are the weakest
  sources in the matrix and should be checked against current vendor documentation before
  publication.
- LexAgent numbers: `paper/generated/results.tex` (computed by `paper/make_results.py`) and
  `experiments/v2/README.md`.
