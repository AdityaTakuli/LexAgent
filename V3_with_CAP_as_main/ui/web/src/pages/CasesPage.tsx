import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { Chip, Icon, IconButton, Meter, PageHeader, Skeleton } from "../components/ui";
import type { CaseMetrics, CaseRow } from "../lib/api";
import { num, pct } from "../lib/format";
import { M3_DURATION, M3_EASE } from "../lib/motion";

type Filter = "all" | "converged" | "contested" | "target" | "flagged";
type SortKey = "id" | "confidence" | "similarity";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "converged", label: "Converged" },
  { key: "contested", label: "Contested" },
  { key: "target", label: "Target case cited" },
  { key: "flagged", label: "Citation flagged" },
];

function StatusChip({ metrics }: { metrics: CaseMetrics | null }) {
  if (!metrics) return <span className="body-small text-muted">not run</span>;
  const converged = metrics.dcr_status === "CONVERGED";
  return (
    <span className="label-medium inline-flex h-6 items-center gap-1 whitespace-nowrap rounded-lg border border-hairline px-2 text-ink">
      <Icon name={converged ? "check_circle" : "forum"} size={15} fill={converged} style={{ color: converged ? "var(--good)" : "var(--md-sys-color-outline)" }} />
      {converged ? "Converged" : "Contested"}
    </span>
  );
}

function CompareRow({ label, a, b, hint }: { label: string; a: string; b: string; hint?: string }) {
  return (
    <tr className="border-b border-line last:border-0">
      <td className="py-2 pr-3 text-ink">
        {label}
        {hint && <div className="body-small text-muted">{hint}</div>}
      </td>
      <td className="tnum py-2 pr-3 text-ink">{a}</td>
      <td className="tnum py-2 text-ink-2">{b}</td>
    </tr>
  );
}

function CaseSheet({ row, onClose }: { row: CaseRow; onClose: () => void }) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const a = row.lexagent;
  const b = row.baseline;
  const yes = (value: number | null | undefined) => (value === null || value === undefined ? "n/a" : value >= 1 ? "Yes" : "No");
  return (
    <>
      <motion.div className="fixed inset-0 z-30 bg-black/40" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: M3_DURATION.short }} onClick={onClose} aria-hidden />
      <motion.aside
        role="dialog"
        aria-modal="true"
        aria-label={`Case ${row.id}: ${row.topic}`}
        className="scroll-thin elevation-3 fixed inset-y-0 right-0 z-40 flex w-full max-w-[460px] flex-col overflow-y-auto rounded-l-[28px] bg-surface p-6"
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ duration: M3_DURATION.long, ease: M3_EASE.emphasizedDecelerate }}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="label-medium text-accent">Case {row.id}</div>
            <h2 className="title-large text-ink">{row.topic}</h2>
          </div>
          <IconButton icon="close" label="Close" onClick={onClose} />
        </div>

        <section className="mb-5">
          <h3 className="label-large mb-1 text-ink-2">Question</h3>
          <p className="body-large text-ink">{row.query}</p>
        </section>

        <section className="mb-5 rounded-2xl bg-surface-2 p-4">
          <h3 className="label-large mb-1 flex items-center gap-1.5 text-ink-2">
            <Icon name="account_balance" size={18} />
            Target authority
          </h3>
          <p className="title-medium text-ink">
            {row.target_case}
            {row.reporter_citation ? <span className="body-medium tnum text-ink-2">, {row.reporter_citation}</span> : null}
          </p>
          <p className="body-medium mt-2 text-ink-2">{row.ground_truth_holding}</p>
        </section>

        <section className="mb-5">
          <h3 className="label-large mb-2 text-ink-2">Result on this case</h3>
          <table className="body-medium w-full text-left">
            <thead>
              <tr className="border-b border-line">
                <th className="label-medium py-2 pr-3 text-ink-2">Measure</th>
                <th className="label-medium py-2 pr-3 text-ink-2">
                  <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: "var(--series-1)" }} />
                  LexAgent
                </th>
                <th className="label-medium py-2 text-ink-2">
                  <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: "var(--series-2)" }} />
                  Baseline
                </th>
              </tr>
            </thead>
            <tbody>
              <CompareRow label="Answer similarity" hint="to the reference holding" a={num(a?.ass, 3)} b={num(b?.ass, 3)} />
              <CompareRow label="Target case cited" a={yes(a?.ar)} b={yes(b?.ar)} />
              <CompareRow label="Citations made" a={num(a?.total_cites, 0)} b={num(b?.total_cites, 0)} />
              <CompareRow label="Hallucination rate" hint="cited cases not in the catalog" a={pct(a?.hr, 0)} b={pct(b?.hr, 0)} />
              <CompareRow label="Citations verified" hint="supported by the opinion" a={pct(a?.haa, 0)} b="n/a" />
              <CompareRow label="Citations misattributed" a={pct(a?.mar, 0)} b="n/a" />
              <CompareRow label="Judge confidence" a={num(a?.judge_confidence)} b="n/a" />
              <CompareRow label="Debate rounds" a={num(a?.debate_rounds, 0)} b="1" />
            </tbody>
          </table>
        </section>

        {[
          { who: "LexAgent", names: a?.fake_names ?? [] },
          { who: "Baseline", names: b?.fake_names ?? [] },
        ]
          .filter((item) => item.names.length > 0)
          .map((item) => (
            <section key={item.who} className="mb-4 rounded-2xl bg-error-container p-4 text-on-error-container">
              <h3 className="label-large mb-1 flex items-center gap-1.5">
                <Icon name="report" size={18} fill />
                {item.who}: citations not found in the catalog
              </h3>
              <ul className="body-medium list-disc pl-5">
                {item.names.map((name, index) => (
                  <li key={index}>{name}</li>
                ))}
              </ul>
              <p className="body-small mt-2">Matching is on the exact citation string, so a real case written with its cite attached can be counted here.</p>
            </section>
          ))}
      </motion.aside>
    </>
  );
}

export function CasesPage({ cases }: { cases: CaseRow[] | null }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [sort, setSort] = useState<SortKey>("id");
  const [search, setSearch] = useState("");
  const [openId, setOpenId] = useState<number | null>(null);

  const rows = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const filtered = (cases ?? []).filter((row) => {
      const m = row.lexagent;
      if (needle && !`${row.topic} ${row.target_case} ${row.query}`.toLowerCase().includes(needle)) return false;
      if (filter === "converged") return m?.dcr_status === "CONVERGED";
      if (filter === "contested") return m !== null && m.dcr_status !== "CONVERGED";
      if (filter === "target") return (m?.ar ?? 0) >= 1;
      if (filter === "flagged") return (m?.fake_cites ?? 0) > 0 || (row.baseline?.fake_cites ?? 0) > 0;
      return true;
    });
    const by: Record<SortKey, (row: CaseRow) => number> = {
      id: (row) => row.id,
      confidence: (row) => -(row.lexagent?.judge_confidence ?? -1),
      similarity: (row) => -(row.lexagent?.ass ?? -1),
    };
    return [...filtered].sort((x, y) => by[sort](x) - by[sort](y));
  }, [cases, filter, sort, search]);

  const open = cases?.find((row) => row.id === openId) ?? null;

  return (
    <div className="mx-auto max-w-6xl px-5 py-7">
      <PageHeader title="Cases" lead="Every benchmark case that was run, with the debate system's result beside the baseline's. Select a row for the full comparison." />

      {/* One filter row above everything it scopes. */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <label className="flex h-10 min-w-[220px] flex-1 items-center gap-2 rounded-full bg-surface-2 px-4 text-ink-2 sm:max-w-xs">
          <Icon name="search" size={20} />
          <span className="sr-only">Search cases</span>
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search topic or case" className="body-medium w-full bg-transparent text-ink outline-none placeholder:text-ink-2" />
        </label>
        {FILTERS.map((item) => (
          <Chip key={item.key} selected={filter === item.key} onClick={() => setFilter(item.key)}>
            {item.label}
          </Chip>
        ))}
        <span className="body-small tnum ml-auto text-muted">
          {rows.length} of {cases?.length ?? 0}
        </span>
      </div>

      {!cases ? (
        <Skeleton className="h-96" />
      ) : (
        <div className="card-outlined overflow-x-auto">
          <table className="body-medium w-full min-w-[760px] text-left">
            <thead>
              <tr className="border-b border-line">
                {[
                  { label: "#", key: "id" as SortKey },
                  { label: "Topic and target case" },
                  { label: "Outcome" },
                  { label: "Judge confidence", key: "confidence" as SortKey },
                  { label: "Answer similarity", key: "similarity" as SortKey },
                  { label: "Rounds" },
                  { label: "Target cited" },
                ].map((column) => (
                  <th key={column.label} className="label-large whitespace-nowrap px-4 py-3 text-ink-2" aria-sort={column.key && sort === column.key ? "ascending" : undefined}>
                    {column.key ? (
                      <button type="button" onClick={() => setSort(column.key!)} className="inline-flex items-center gap-1 hover:text-ink">
                        {column.label}
                        <Icon name={sort === column.key ? "arrow_downward" : "unfold_more"} size={16} />
                      </button>
                    ) : (
                      column.label
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const m = row.lexagent;
                return (
                  <tr key={row.id} tabIndex={0} onClick={() => setOpenId(row.id)} onKeyDown={(event) => event.key === "Enter" && setOpenId(row.id)} className="cursor-pointer border-b border-line outline-offset-[-2px] last:border-0 hover:bg-surface">
                    <td className="tnum px-4 py-3 text-muted">{row.id}</td>
                    <td className="px-4 py-3">
                      <div className="text-ink">{row.topic}</div>
                      <div className="body-small text-ink-2">{row.target_case}</div>
                    </td>
                    <td className="px-4 py-3">
                      <StatusChip metrics={m} />
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="tnum w-9 text-ink">{num(m?.judge_confidence)}</span>
                        <span className="w-20">{m?.judge_confidence !== null && m?.judge_confidence !== undefined && <Meter value={m.judge_confidence} color="var(--series-1)" label="Judge confidence" />}</span>
                      </div>
                    </td>
                    <td className="tnum px-4 py-3 text-ink">
                      {num(m?.ass, 3)} <span className="text-muted">/ {num(row.baseline?.ass, 3)}</span>
                    </td>
                    <td className="tnum px-4 py-3 text-ink-2">{num(m?.debate_rounds, 0)}</td>
                    <td className="px-4 py-3">{m ? <Icon name={(m.ar ?? 0) >= 1 ? "check" : "remove"} size={18} className={(m.ar ?? 0) >= 1 ? "text-ink" : "text-muted"} /> : null}</td>
                  </tr>
                );
              })}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-10 text-center text-ink-2">
                    No cases match.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
      <p className="body-small mt-3 text-muted">Answer similarity is shown as LexAgent / baseline.</p>

      <AnimatePresence>{open && <CaseSheet key={open.id} row={open} onClose={() => setOpenId(null)} />}</AnimatePresence>
    </div>
  );
}
