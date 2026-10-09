import { EmphasisBars, GroupedBars, Histogram, StackedBars } from "../components/charts";
import { Card, ChartCard, Icon, Legend, PageHeader, Skeleton, StatTile, TIERS } from "../components/ui";
import type { CaseRow, Overview, StressSummary } from "../lib/api";
import { duration, num, pct } from "../lib/format";

const STRESS_CLASSES: { key: string; label: string; expected: string }[] = [
  { key: "correct", label: "Correct citation", expected: "should be supported" },
  { key: "misattributed", label: "Right case, another case's holding", expected: "should be rejected" },
  { key: "casehold_distractor", label: "Right case, false holding", expected: "should be rejected" },
  { key: "fabricated_with_cite", label: "Invented case with a cite", expected: "should be fabricated" },
  { key: "fabricated_no_cite", label: "Invented case, no cite", expected: "should be unverifiable" },
  { key: "real_off_corpus", label: "Real case outside the corpus", expected: "should be unverifiable" },
];

function BeforeAfter({ label, before, after, note }: { label: string; before: number | null; after: number | null; note: string }) {
  return (
    <div className="rounded-xl bg-surface px-3.5 py-3">
      <div className="label-medium text-ink-2">{label}</div>
      <div className="mt-1 flex items-baseline gap-2">
        {before !== null && (
          <>
            <span className="tnum body-large text-muted line-through decoration-1">{pct(before, 0)}</span>
            <Icon name="arrow_forward" size={14} className="self-center text-muted" />
          </>
        )}
        <span className="title-large text-ink">{pct(after, 0)}</span>
      </div>
      <div className="body-small mt-0.5 text-muted">{note}</div>
    </div>
  );
}

function stressRows(stress: StressSummary) {
  return STRESS_CLASSES.filter((item) => stress.per_type_counts[item.key]).map((item) => {
    const counts = stress.per_type_counts[item.key];
    return { label: item.label, note: item.expected, total: counts.total, values: TIERS.map((tier) => counts[tier.tier] ?? 0) };
  });
}

export function AnalysisPage({ overview, cases }: { overview: Overview | null; cases: CaseRow[] | null }) {
  if (!overview) {
    return (
      <div className="mx-auto max-w-6xl p-6">
        <Skeleton className="mb-4 h-16" />
        <div className="grid gap-4 md:grid-cols-4">
          {[0, 1, 2, 3].map((item) => (
            <Skeleton key={item} className="h-28" />
          ))}
        </div>
      </div>
    );
  }
  const { benchmark, baseline, ablation, stress, stress_before_fix: before, test_set: testSet } = overview;

  const confidenceBins = Array.from({ length: 10 }, (_, index) => ({ label: `${(index / 10).toFixed(1)}–${((index + 1) / 10).toFixed(1)}`, count: 0 }));
  let rounds3 = 0;
  for (const row of cases ?? []) {
    const metrics = row.lexagent;
    if (!metrics || metrics.judge_confidence === null) continue;
    confidenceBins[Math.min(9, Math.floor(metrics.judge_confidence * 10))].count += 1;
    if ((metrics.debate_rounds ?? 1) >= 3) rounds3 += 1;
  }
  const maxBin = Math.max(...confidenceBins.map((bin) => bin.count), 1);
  const yStep = maxBin > 40 ? 20 : maxBin > 16 ? 10 : 5;
  const yTicks = Array.from({ length: Math.ceil(maxBin / yStep) + 1 }, (_, index) => index * yStep);

  const ablationRows = ablation ?? [];
  const ablationCases = ablationRows[0]?.total_cases ?? 0;

  return (
    <div className="mx-auto max-w-6xl px-5 py-7">
      <PageHeader
        title="Evaluation"
        lead={
          <>
            Results on the {testSet.evaluable} benchmark cases the corpus can support, out of {testSet.total}. The debate system is compared with a single-agent retrieve-then-answer baseline on the same cases, the same corpus and the same model.
          </>
        }
      />

      {benchmark ? (
        <div className="mb-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatTile icon="report" label="Hallucination rate" value={benchmark.hr} format={(n) => pct(n, 2)} sub={<>Baseline {pct(baseline?.hr, 2)} · lower is better</>} hint="Share of cited cases that are not in the corpus catalog" />
          <StatTile icon="compare_arrows" label="Answer similarity" value={benchmark.ass} format={(n) => n.toFixed(3)} sub={<>Baseline {num(baseline?.ass, 3)} · to the true holding</>} />
          <StatTile icon="gavel" label="Judge confidence" value={benchmark.judge_confidence} format={(n) => n.toFixed(2)} sub={<>{benchmark.converged} of {benchmark.cases} debates converged</>} />
          <StatTile icon="timer" label="Benchmark run time" value={(overview.benchmark_seconds ?? 0) / 60} format={(n) => `${n.toFixed(0)} min`} sub={<>{duration(overview.benchmark_seconds)} for {benchmark.cases} cases</>} />
        </div>
      ) : (
        <Card className="mb-5">
          <p className="body-medium text-ink-2">No benchmark results have been saved yet. Run <code>python master_run.py --benchmark</code>.</p>
        </Card>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        {benchmark && baseline && (
          <ChartCard
            title="Debate system and baseline"
            subtitle="Mean over the test set, on one 0–100% scale"
            columns={["Measure", "LexAgent", "Baseline"]}
            rows={[
              ["Answer similarity", pct(benchmark.ass), pct(baseline.ass)],
              ["Target authority cited (recall)", pct(benchmark.ar), pct(baseline.ar)],
              ["Authority precision", pct(benchmark.ap), pct(baseline.ap)],
              ["Hallucination rate", pct(benchmark.hr, 2), pct(baseline.hr, 2)],
              ["Citations per answer", num(benchmark.total_cites), num(baseline.total_cites)],
            ]}
            footnote={<>The baseline answers closer to the reference holding and cites more precisely; the debate system cites more cases per answer ({num(benchmark.total_cites)} against {num(baseline.total_cites)}). The difference in hallucination rate ({pct(benchmark.hr, 2)} against {pct(baseline.hr, 2)}) is too small to be statistically significant on this test set.</>}
          >
            <Legend items={[{ label: "LexAgent (debate)", color: "var(--series-1)" }, { label: "Single-agent RAG baseline", color: "var(--series-2)" }]} />
            <GroupedBars
              series={[{ label: "LexAgent", color: "var(--series-1)" }, { label: "Baseline", color: "var(--series-2)" }]}
              max={100}
              ticks={[0, 25, 50, 75, 100]}
              format={(n) => `${n.toFixed(0)}%`}
              rows={[
                { label: "Answer similarity", note: "to the reference holding", values: [benchmark.ass === null ? null : benchmark.ass * 100, baseline.ass === null ? null : baseline.ass * 100] },
                { label: "Authority recall", note: "target case was cited", values: [benchmark.ar === null ? null : benchmark.ar * 100, baseline.ar === null ? null : baseline.ar * 100] },
                { label: "Authority precision", note: "citations that are the target", values: [benchmark.ap === null ? null : benchmark.ap * 100, baseline.ap === null ? null : baseline.ap * 100] },
              ]}
            />
          </ChartCard>
        )}

        {cases && benchmark && (
          <ChartCard
            title="Judge confidence across cases"
            subtitle={`${benchmark.cases} debates · ${rounds3} ran the full three rounds`}
            columns={["Confidence", "Cases"]}
            rows={confidenceBins.map((bin) => [bin.label, bin.count])}
            footnote={<>Confidence is grounded in the citation checks: a debate whose citations are not verified ends with a low score, which is why the distribution has two ends rather than a middle.</>}
          >
            <Histogram bins={confidenceBins} yTicks={yTicks} unit="cases" />
          </ChartCard>
        )}

        {ablationRows.length > 0 && (
          <ChartCard
            className="lg:col-span-2"
            title="Ablation: what each component contributes"
            subtitle={`The full system against four variants with one component removed, ${ablationCases} cases each`}
            columns={["Variant", "What is removed", "Hallucination rate", "Answer similarity", "Judge confidence", "Completed"]}
            rows={ablationRows.map((row) => [row.variant, row.description, pct(row.mean_hr, 2), num(row.mean_cas, 3), num(row.mean_confidence), `${row.completed_cases}/${row.total_cases}`])}
            footnote={<>A1 removes the defense, A2 the citation check, A3 the memory, A4 the debate controller (a single fixed round).</>}
          >
            <div className="grid gap-6 md:grid-cols-3">
              {[
                { title: "Hallucination rate", unit: "Hallucination rate", key: "mean_hr" as const, format: (n: number) => `${(n * 100).toFixed(2)}%` },
                { title: "Answer similarity", unit: "Answer similarity", key: "mean_cas" as const, format: (n: number) => n.toFixed(3) },
                { title: "Judge confidence", unit: "Judge confidence", key: "mean_confidence" as const, format: (n: number) => n.toFixed(2) },
              ].map((metric) => (
                <div key={metric.key}>
                  <h3 className="title-small mb-2.5 text-ink">{metric.title}</h3>
                  <EmphasisBars
                    emphasis="Full"
                    unit={metric.unit}
                    format={metric.format}
                    max={metric.key === "mean_hr" ? Math.max(...ablationRows.map((row) => row.mean_hr), 0.001) * 1.15 : 1}
                    rows={ablationRows.map((row) => ({ label: row.variant, detail: row.description, value: row[metric.key] }))}
                  />
                </div>
              ))}
            </div>
          </ChartCard>
        )}

        {stress && (
          <ChartCard
            className="lg:col-span-2"
            title="Citation stress test"
            subtitle={`${stress.total_evaluated} labelled citations fed straight to the citation checker, with no language model involved`}
            columns={["Citation class", "Expected", ...TIERS.map((tier) => tier.label), "Total"]}
            rows={STRESS_CLASSES.filter((item) => stress.per_type_counts[item.key]).map((item) => {
              const counts = stress.per_type_counts[item.key];
              return [item.label, item.expected, ...TIERS.map((tier) => counts[tier.tier] ?? 0), counts.total];
            })}
            footnote={<>Existence and reporter-cite checks are exact on this set. The holding check accepts {pct(stress.power, 0)} of correct citations: the entailment model, not the evidence it is given, is now the limit.</>}
          >
            <div className="mb-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <BeforeAfter label="Correct citations accepted" before={before?.power ?? null} after={stress.power} note="higher is better" />
              <BeforeAfter label="Wrong holding accepted" before={before?.false_acceptance_rate_misattributed ?? null} after={stress.false_acceptance_rate_misattributed} note="lower is better" />
              <BeforeAfter label="Invented cases flagged as fabricated" before={before?.fabrication_catch_rate ?? null} after={stress.fabrication_catch_rate} note="higher is better" />
              <BeforeAfter label="Off-corpus cases called fabricated" before={null} after={stress.false_fabrication_rate} note="lower is better" />
            </div>
            <Legend items={TIERS.map((tier) => ({ label: tier.label, color: tier.color }))} />
            <StackedBars rows={stressRows(stress)} segments={TIERS.map((tier) => ({ label: tier.label, color: tier.color }))} />
            {before && <p className="body-small mt-4 text-muted">Struck-through figures are from before the citation checker was changed to read the actual opinion.</p>}
          </ChartCard>
        )}

        <Card className="lg:col-span-2" title="Test set" subtitle={`${testSet.evaluable} of ${testSet.total} benchmark cases are evaluated`}>
          <p className="body-medium mb-3 text-ink-2">A case is evaluated only when the opinion of its target case is in the corpus. These {testSet.excluded.length} are left out, because no answer to them could be grounded:</p>
          <ul className="grid gap-2 sm:grid-cols-2">
            {testSet.excluded.map((item) => (
              <li key={item.id} className="rounded-xl bg-surface-2 px-3 py-2.5">
                <div className="title-small text-ink">{item.target_case}</div>
                <div className="body-small text-ink-2">{item.reason}</div>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
