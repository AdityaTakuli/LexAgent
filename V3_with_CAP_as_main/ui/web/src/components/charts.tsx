import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { useTip } from "./ui";

// Hand-built charts: thin bars (at most 24px), a 4px rounded data end on a square baseline,
// a 2px surface gap between touching fills, hairline solid gridlines, values in text ink.

// Material 3 emphasized-decelerate easing, the curve M3 uses for elements entering the screen.
const EASE = [0.05, 0.7, 0.1, 1] as [number, number, number, number];

function Gridlines({ ticks, max, format }: { ticks: number[]; max: number; format: (n: number) => string }) {
  return (
    <div className="pointer-events-none absolute inset-0" aria-hidden>
      {ticks.map((tick) => (
        <div key={tick} className="absolute top-0 bottom-5 border-l border-line" style={{ left: `${(tick / max) * 100}%` }}>
          <span className="tnum absolute -bottom-5 -translate-x-1/2 text-[11px] text-muted">{format(tick)}</span>
        </div>
      ))}
    </div>
  );
}

function Bar({ share, color, tip, delay, height = 14 }: { share: number; color: string; tip: ReactNode; delay: number; height?: number }) {
  const tipProps = useTip(tip);
  return (
    <motion.div
      {...tipProps}
      className="rounded-r-[4px] outline-offset-2"
      style={{ background: color, height, transformOrigin: "left center", width: `${Math.max(share * 100, 0.6)}%` }}
      initial={{ scaleX: 0 }}
      whileInView={{ scaleX: 1 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.7, delay, ease: EASE }}
    />
  );
}

/** Two series compared on one shared axis: one group of bars per row. */
export function GroupedBars({
  rows,
  series,
  max,
  ticks,
  format,
}: {
  rows: { label: string; note?: string; values: (number | null)[] }[];
  series: { label: string; color: string }[];
  max: number;
  ticks: number[];
  format: (n: number) => string;
}) {
  const rowHeight = 56;
  return (
    <div className="grid grid-cols-[minmax(110px,170px)_1fr] gap-x-4">
      <div className="flex flex-col">
        {rows.map((row) => (
          <div key={row.label} className="flex flex-col justify-center pr-1" style={{ height: rowHeight }}>
            <span className="body-medium text-ink">{row.label}</span>
            {row.note && <span className="body-small text-muted">{row.note}</span>}
          </div>
        ))}
      </div>
      {/* The right-hand column leaves room past the longest bar for its value label. */}
      <div className="relative pb-6 pr-12">
        <div className="absolute inset-y-0 left-0 right-12">
          <Gridlines ticks={ticks} max={max} format={format} />
        </div>
        <div className="relative flex flex-col">
          {rows.map((row, rowIndex) => (
            <div key={row.label} className="flex flex-col justify-center gap-[2px]" style={{ height: rowHeight }}>
              {row.values.map((value, seriesIndex) => (
                <div key={series[seriesIndex].label} className="flex items-center gap-2">
                  {value === null ? (
                    <span className="text-[11.5px] text-muted">n/a</span>
                  ) : (
                    <>
                      <Bar share={value / max} color={series[seriesIndex].color} delay={rowIndex * 0.05 + seriesIndex * 0.04}
                        tip={<><b>{series[seriesIndex].label}</b>: {format(value)}<br /><span className="text-ink-2">{row.label}</span></>} />
                      <span className="tnum -mr-12 w-10 shrink-0 text-[12px] text-ink-2">{format(value)}</span>
                    </>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/** One measure across a few items, with a single item emphasised and the rest in de-emphasis grey. */
export function EmphasisBars({
  rows,
  emphasis,
  max,
  format,
  unit,
}: {
  rows: { label: string; detail?: string; value: number }[];
  emphasis: string;
  max: number;
  format: (n: number) => string;
  unit: string;
}) {
  return (
    <div className="flex flex-col gap-2.5">
      {rows.map((row, index) => (
        <div key={row.label} className="grid grid-cols-[38px_1fr_auto] items-center gap-2.5">
          <span className={`text-[12.5px] ${row.label === emphasis ? "font-semibold text-ink" : "text-ink-2"}`}>{row.label}</span>
          <div className="relative border-l border-axis py-0.5">
            <Bar share={max ? row.value / max : 0} color={row.label === emphasis ? "var(--series-1)" : "var(--de-emphasis)"} delay={index * 0.06}
              tip={<><b>{row.label}</b>{row.detail ? `: ${row.detail}` : ""}<br />{unit} {format(row.value)}</>} />
          </div>
          <span className="tnum w-14 text-right text-[12.5px] text-ink-2">{format(row.value)}</span>
        </div>
      ))}
    </div>
  );
}

function Segment({ share, color, tip, first, last, delay }: { share: number; color: string; tip: ReactNode; first: boolean; last: boolean; delay: number }) {
  const tipProps = useTip(tip);
  return (
    <motion.div
      {...tipProps}
      className="h-full outline-offset-2"
      style={{ background: color, flexGrow: share, flexBasis: 0, minWidth: 3, borderRadius: `${first ? 4 : 0}px ${last ? 4 : 0}px ${last ? 4 : 0}px ${first ? 4 : 0}px` }}
      initial={{ opacity: 0, scaleY: 0.4 }}
      whileInView={{ opacity: 1, scaleY: 1 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.45, delay, ease: EASE }}
    />
  );
}

/** Part-to-whole per row: stacked horizontal bars with a 2px surface gap between segments. */
export function StackedBars({
  rows,
  segments,
  rowHeight = 18,
}: {
  rows: { label: string; note?: string; total: number; values: number[] }[];
  segments: { label: string; color: string }[];
  rowHeight?: number;
}) {
  return (
    <div className="flex flex-col gap-3.5">
      {rows.map((row, rowIndex) => {
        const shown = row.values.map((value, index) => ({ value, index })).filter((item) => item.value > 0);
        return (
          <div key={row.label} className="grid grid-cols-[minmax(120px,200px)_1fr_72px] items-center gap-x-3">
            <div className="min-w-0">
              <div className="body-medium text-ink">{row.label}</div>
              {row.note && <div className="body-small text-muted">{row.note}</div>}
            </div>
            <div className="flex gap-[2px]" style={{ height: rowHeight }} role="img" aria-label={`${row.label}: ${shown.map((item) => `${segments[item.index].label} ${item.value}`).join(", ")}`}>
              {shown.map((item, position) => (
                <Segment key={segments[item.index].label} share={item.value / row.total} color={segments[item.index].color} first={position === 0} last={position === shown.length - 1}
                  delay={rowIndex * 0.06 + position * 0.04}
                  tip={<><b>{segments[item.index].label}</b>: {item.value} of {row.total} ({((item.value / row.total) * 100).toFixed(0)}%)<br /><span className="text-ink-2">{row.label}</span></>} />
              ))}
            </div>
            <span className="tnum body-small text-right text-muted">n={row.total.toLocaleString("en-US")}</span>
          </div>
        );
      })}
    </div>
  );
}

function Column({ share, tip, delay }: { share: number; tip: ReactNode; delay: number }) {
  const tipProps = useTip(tip);
  return (
    <div className="flex h-full flex-1 items-end justify-center" {...tipProps}>
      <motion.div
        className="w-full max-w-[24px] rounded-t-[4px]"
        style={{ background: "var(--series-1)", height: `${Math.max(share * 100, share > 0 ? 2 : 0)}%`, transformOrigin: "center bottom" }}
        initial={{ scaleY: 0 }}
        whileInView={{ scaleY: 1 }}
        viewport={{ once: true, margin: "-40px" }}
        transition={{ duration: 0.6, delay, ease: EASE }}
      />
    </div>
  );
}

/** Distribution of one measure: columns from a single baseline. One series, so no legend. */
export function Histogram({ bins, yTicks, unit }: { bins: { label: string; count: number }[]; yTicks: number[]; unit: string }) {
  const max = Math.max(...yTicks, 1);
  return (
    <div className="grid grid-cols-[28px_1fr] gap-x-2">
      <div className="relative h-[170px]">
        {yTicks.map((tick) => (
          <span key={tick} className="tnum absolute right-0 translate-y-1/2 text-[11px] text-muted" style={{ bottom: `${(tick / max) * 100}%` }}>
            {tick}
          </span>
        ))}
      </div>
      <div className="relative h-[170px] border-b border-axis">
        {yTicks.filter((tick) => tick > 0).map((tick) => (
          <div key={tick} className="absolute inset-x-0 border-t border-line" style={{ bottom: `${(tick / max) * 100}%` }} aria-hidden />
        ))}
        <div className="absolute inset-0 flex items-end gap-[2px] px-1">
          {bins.map((bin, index) => (
            <Column key={bin.label} share={bin.count / max} delay={index * 0.03} tip={<><b>{bin.count}</b> {unit}<br /><span className="text-ink-2">{bin.label}</span></>} />
          ))}
        </div>
      </div>
      <div />
      <div className="flex gap-[2px] px-1 pt-1.5">
        {bins.map((bin, index) => (
          <span key={bin.label} className="tnum flex-1 text-center text-[11px] text-muted">
            {index % 2 === 0 ? bin.label.split("–")[0] : ""}
          </span>
        ))}
      </div>
    </div>
  );
}
