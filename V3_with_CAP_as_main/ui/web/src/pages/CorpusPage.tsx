import { motion } from "framer-motion";
import { StackedBars } from "../components/charts";
import { Card, ChartCard, Icon, Legend, PageHeader, Skeleton, StatTile } from "../components/ui";
import type { Overview } from "../lib/api";
import { compact, int, ranges } from "../lib/format";
import { riseIn } from "../lib/motion";

const STAGE_ICON: Record<string, string> = { Download: "cloud_download", Chunk: "content_cut", Index: "hub" };

export function CorpusPage({ overview }: { overview: Overview | null }) {
  if (!overview) {
    return (
      <div className="mx-auto max-w-6xl p-6">
        <Skeleton className="h-64" />
      </div>
    );
  }
  const corpus = overview.corpus;
  const unavailable = corpus.volumes_unavailable.length;
  const volumeSegments = [
    { label: "Caselaw Access Project", color: "var(--series-1)" },
    { label: "Official U.S. Reports PDFs", color: "var(--series-3)" },
    { label: "Not available", color: "var(--de-emphasis)" },
  ];
  const reportSegments = [
    { label: "Opinions (300 words or more)", color: "var(--series-1)" },
    { label: "Order entries and short notations", color: "var(--de-emphasis)" },
  ];

  return (
    <div className="mx-auto max-w-6xl px-5 py-7">
      <PageHeader title="Corpus" lead="What the system retrieves from and verifies against: the U.S. Reports, ingested in three resumable stages and indexed for hybrid search." />

      <div className="mb-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile icon="segment" label="Passages indexed" value={corpus.chunks} format={(n) => compact(Math.round(n))} sub={<>{int(corpus.chunks)} chunks of about 450 words</>} />
        <StatTile icon="menu_book" label="Volumes" value={corpus.volumes_indexed} format={(n) => n.toFixed(0)} sub={<>of {corpus.volumes_target} targeted</>} />
        <StatTile icon="description" label="Reports" value={corpus.reports} format={(n) => compact(Math.round(n))} sub={<>{int(corpus.reports)} separate case entries</>} />
        <StatTile icon="gavel" label="Opinions" value={corpus.opinions} format={(n) => compact(Math.round(n))} sub={<>{int(corpus.order_entries)} more are order entries</>} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <ChartCard
          title="Volumes by source"
          subtitle={`${corpus.volumes_target} target volumes of the U.S. Reports`}
          columns={["Source", "Volumes"]}
          rows={[
            ["Caselaw Access Project", corpus.volumes_from_cap],
            ["Official U.S. Reports PDFs", corpus.volumes_from_official_pdf],
            ["Not available", unavailable],
          ]}
          footnote={
            <>
              CAP ends at volume 572. Later volumes come from the Supreme Court's own bound volumes and preliminary prints.
              {unavailable > 0 ? ` Volumes ${ranges(corpus.volumes_unavailable)} have no published opinions in U.S. Reports form yet.` : ""}
            </>
          }
        >
          <Legend items={volumeSegments} />
          <StackedBars rowHeight={22} segments={volumeSegments} rows={[{ label: "Volumes 1–600", total: corpus.volumes_target || corpus.volumes_indexed + unavailable, values: [corpus.volumes_from_cap, corpus.volumes_from_official_pdf, unavailable] }]} />
        </ChartCard>

        <ChartCard
          title="What a report is"
          subtitle="Most entries in a volume are one-line orders, not opinions"
          columns={["Kind of report", "Count"]}
          rows={[
            ["Opinions (300 words or more)", int(corpus.opinions)],
            ["Order entries and short notations", int(corpus.order_entries)],
          ]}
          footnote={<>An order entry carries the same case name as the opinion it relates to. The citation checker therefore resolves every citation to the actual opinion before it checks a holding.</>}
        >
          <Legend items={reportSegments} />
          <StackedBars rowHeight={22} segments={reportSegments} rows={[{ label: "All reports", total: corpus.reports, values: [corpus.opinions, corpus.order_entries] }]} />
        </ChartCard>

        <Card className="lg:col-span-2" title="Pipeline" subtitle="Each stage is separate and resumes from its own checkpoint">
          <ol className="grid gap-3 md:grid-cols-3">
            {corpus.stages.map((stage, index) => {
              const counts = stage.counts ?? {};
              return (
                <motion.li key={stage.name} {...riseIn} transition={{ ...riseIn.transition, delay: index * 0.08 }} className="rounded-2xl bg-surface-2 p-4">
                  <div className="flex items-center gap-3">
                    <span className="flex h-10 w-10 items-center justify-center rounded-full bg-accent-wash text-on-accent-wash">
                      <Icon name={STAGE_ICON[stage.name] ?? "settings"} size={20} />
                    </span>
                    <div>
                      <div className="label-medium text-ink-2">Stage {index + 1}</div>
                      <div className="title-medium text-ink">{stage.name}</div>
                    </div>
                  </div>
                  <p className="body-small mt-2 text-ink-2">{stage.detail}</p>
                  <dl className="body-medium mt-3 grid grid-cols-2 gap-y-1">
                    {stage.total !== undefined ? (
                      <>
                        <dt className="text-ink-2">Chunks embedded</dt>
                        <dd className="tnum text-right text-ink">
                          {int(stage.indexed)} / {int(stage.total)}
                        </dd>
                      </>
                    ) : (
                      <>
                        <dt className="text-ink-2">Complete</dt>
                        <dd className="tnum text-right text-ink">{int(counts.COMPLETE ?? 0)}</dd>
                        <dt className="text-ink-2">Failed</dt>
                        <dd className="tnum text-right text-ink">{int(counts.FAILED ?? 0)}</dd>
                        <dt className="text-ink-2">Unavailable</dt>
                        <dd className="tnum text-right text-ink">{int(counts.UNAVAILABLE ?? 0)}</dd>
                      </>
                    )}
                  </dl>
                </motion.li>
              );
            })}
          </ol>
        </Card>

        <Card className="lg:col-span-2" title="Models and settings">
          <dl className="body-medium grid gap-x-8 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
            {[
              { icon: "smart_toy", label: "Generator", value: corpus.models.generator },
              { icon: "scatter_plot", label: "Embedder", value: corpus.models.embedder },
              { icon: "rule", label: "Entailment model", value: corpus.models.nli },
              { icon: "tune", label: "Acceptance threshold", value: String(corpus.models.tau) },
              { icon: "repeat", label: "Maximum debate rounds", value: String(corpus.models.max_rounds) },
              { icon: "event", label: "Index built", value: corpus.built_at ? new Date(corpus.built_at).toLocaleString() : "n/a" },
            ].map((item) => (
              <div key={item.label} className="flex items-start gap-3">
                <Icon name={item.icon} size={20} className="mt-0.5 text-ink-2" />
                <div className="min-w-0">
                  <dt className="label-medium text-ink-2">{item.label}</dt>
                  <dd className="break-words text-ink">{item.value}</dd>
                </div>
              </div>
            ))}
          </dl>
        </Card>
      </div>
    </div>
  );
}
