import { AnimatePresence, motion } from "framer-motion";
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ButtonHTMLAttributes, type CSSProperties, type ReactNode } from "react";
import type { Tier } from "../lib/api";
import { CountUp, M3_DURATION, M3_EASE, riseIn, useRipple } from "../lib/motion";

// Material 3 building blocks: icons, buttons, chips, segmented buttons, cards, tooltips.

/** A Material Symbols (Rounded) icon. `fill` switches to the filled style, as M3 does for selected states. */
export function Icon({ name, size = 20, fill = false, className = "", style }: { name: string; size?: number; fill?: boolean; className?: string; style?: CSSProperties }) {
  return (
    <span
      className={`msr ${className}`}
      aria-hidden
      style={{ fontSize: size, width: size, height: size, fontVariationSettings: `'FILL' ${fill ? 1 : 0}, 'wght' 400, 'GRAD' 0, 'opsz' ${Math.min(48, Math.max(20, size))}`, ...style }}
    >
      {name}
    </span>
  );
}

// ── Buttons ─────────────────────────────────────────────────────────────────
type ButtonVariant = "filled" | "tonal" | "outlined" | "text";
const BUTTON_STYLE: Record<ButtonVariant, string> = {
  filled: "bg-accent text-accent-ink",
  tonal: "bg-secondary-container text-on-secondary-container",
  outlined: "border border-axis text-accent",
  text: "text-accent",
};

export function Button({ variant = "filled", icon, children, className = "", ...rest }: { variant?: ButtonVariant; icon?: string } & ButtonHTMLAttributes<HTMLButtonElement>) {
  const ripple = useRipple();
  return (
    <button type="button" {...rest} onPointerDown={ripple.onPointerDown} className={`state-layer label-large inline-flex h-10 items-center gap-2 rounded-full ${icon ? "pl-4 pr-6" : "px-6"} disabled:opacity-40 ${BUTTON_STYLE[variant]} ${className}`}>
      {ripple.layer}
      {icon && <Icon name={icon} size={18} />}
      {children}
    </button>
  );
}

export function IconButton({ icon, label, variant = "standard", selected = false, className = "", ...rest }: { icon: string; label: string; variant?: "standard" | "filled" | "tonal"; selected?: boolean } & ButtonHTMLAttributes<HTMLButtonElement>) {
  const ripple = useRipple();
  const tone = variant === "filled" ? "bg-accent text-accent-ink" : variant === "tonal" ? "bg-secondary-container text-on-secondary-container" : "text-ink-2";
  return (
    <button type="button" aria-label={label} title={label} {...rest} onPointerDown={ripple.onPointerDown} className={`state-layer inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-full disabled:opacity-40 ${tone} ${className}`}>
      {ripple.layer}
      <Icon name={icon} size={22} fill={selected} />
    </button>
  );
}

/** M3 segmented button: one choice of a few, the selected segment marked with a check. */
export function SegmentedButton<T extends string>({ options, value, onChange, label }: { options: { value: T; label: string; icon?: string }[]; value: T; onChange: (value: T) => void; label: string }) {
  return (
    <div className="inline-flex h-8 shrink-0 overflow-hidden rounded-full border border-axis" role="group" aria-label={label}>
      {options.map((option, index) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={selected}
            onClick={() => onChange(option.value)}
            className={`state-layer label-medium inline-flex items-center gap-1.5 px-3 ${index ? "border-l border-axis" : ""} ${selected ? "bg-secondary-container text-on-secondary-container" : "text-ink"}`}
          >
            {selected ? <Icon name="check" size={16} /> : option.icon ? <Icon name={option.icon} size={16} /> : null}
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/** M3 chip. With `onClick` and `selected` it is a filter chip; otherwise a plain label chip. */
export function Chip({ children, icon, selected = false, onClick, tone = "neutral" }: { children: ReactNode; icon?: string; selected?: boolean; onClick?: () => void; tone?: "neutral" | "primary" }) {
  const base = "label-medium inline-flex h-7 max-w-full items-center gap-1.5 rounded-lg px-2.5";
  const look = selected
    ? "bg-secondary-container text-on-secondary-container"
    : tone === "primary"
      ? "bg-accent-wash text-on-accent-wash"
      : "border border-hairline text-ink-2";
  const content = (
    <>
      {(selected || icon) && <Icon name={selected ? "check" : icon!} size={16} />}
      <span className="truncate">{children}</span>
    </>
  );
  return onClick ? (
    <button type="button" aria-pressed={selected} onClick={onClick} className={`state-layer ${base} ${look}`}>
      {content}
    </button>
  ) : (
    <span className={`${base} ${look}`}>{content}</span>
  );
}

// ── Tooltip: one floating layer for every chart mark (M3 rich tooltip) ──────
interface TipState {
  x: number;
  y: number;
  content: ReactNode;
}
const TipContext = createContext<{ show: (target: Element, content: ReactNode) => void; hide: () => void }>({ show: () => {}, hide: () => {} });

export function TooltipLayer({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const show = useCallback((target: Element, content: ReactNode) => {
    const rect = target.getBoundingClientRect();
    setTip({ x: rect.left + rect.width / 2, y: rect.top, content });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  const value = useMemo(() => ({ show, hide }), [show, hide]);
  return (
    <TipContext.Provider value={value}>
      {children}
      <AnimatePresence>
        {tip && (
          <motion.div
            role="tooltip"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.12, ease: M3_EASE.standard }}
            className="body-small elevation-2 pointer-events-none fixed z-50 max-w-xs -translate-x-1/2 -translate-y-full rounded-xl bg-surface-container px-3 py-2 text-ink-2"
            style={{ left: Math.min(Math.max(tip.x, 140), window.innerWidth - 140), top: tip.y - 8 }}
          >
            {tip.content}
          </motion.div>
        )}
      </AnimatePresence>
    </TipContext.Provider>
  );
}

/** Hover and keyboard-focus handlers that show the same tooltip for a chart mark. */
export function useTip(content: ReactNode) {
  const { show, hide } = useContext(TipContext);
  return {
    onMouseEnter: (event: React.MouseEvent) => show(event.currentTarget, content),
    onMouseLeave: hide,
    onFocus: (event: React.FocusEvent) => show(event.currentTarget, content),
    onBlur: hide,
    tabIndex: 0,
  };
}

// ── Cards and tiles ─────────────────────────────────────────────────────────
export function Card({ title, subtitle, action, children, className = "", variant = "filled" }: { title?: string; subtitle?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; variant?: "filled" | "outlined" }) {
  return (
    <motion.section {...riseIn} className={`${variant === "outlined" ? "card-outlined" : "card"} p-4 sm:p-5 ${className}`}>
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div className="min-w-0">
            {title && <h2 className="title-medium text-ink">{title}</h2>}
            {subtitle && <p className="body-medium mt-0.5 text-ink-2">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </motion.section>
  );
}

export function StatTile({ label, value, format, sub, hint, icon }: { label: string; value: number | null; format: (n: number) => string; sub?: ReactNode; hint?: string; icon?: string }) {
  return (
    <motion.div {...riseIn} className="card p-4" title={hint}>
      <div className="label-large flex items-center gap-2 text-ink-2">
        {icon && <Icon name={icon} size={18} />}
        {label}
      </div>
      <div className="headline-medium mt-2 text-ink">{value === null ? "n/a" : <CountUp value={value} format={format} />}</div>
      {sub && <div className="body-small mt-1 text-ink-2">{sub}</div>}
    </motion.div>
  );
}

/** A chart with its table-view twin: every value is reachable without hovering. */
export function ChartCard({
  title,
  subtitle,
  children,
  columns,
  rows,
  footnote,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  columns: string[];
  rows: (string | number)[][];
  footnote?: ReactNode;
  className?: string;
}) {
  const [view, setView] = useState<"chart" | "table">("chart");
  const toggle = (
    <SegmentedButton
      label={`${title}: view`}
      value={view}
      onChange={setView}
      options={[
        { value: "chart", label: "Chart" },
        { value: "table", label: "Table" },
      ]}
    />
  );
  return (
    <Card title={title} subtitle={subtitle} action={toggle} className={className} variant="outlined">
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={view} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: M3_DURATION.short, ease: M3_EASE.standard }}>
          {view === "chart" ? (
            children
          ) : (
            <div className="overflow-x-auto">
              <table className="body-medium w-full text-left">
                <thead>
                  <tr className="border-b border-line text-ink-2">
                    {columns.map((column) => (
                      <th key={column} className="label-large py-2 pr-4">
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, index) => (
                    <tr key={index} className="border-b border-line last:border-0">
                      {row.map((cell, cellIndex) => (
                        <td key={cellIndex} className={`py-2 pr-4 ${cellIndex ? "tnum text-ink-2" : "text-ink"}`}>
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </motion.div>
      </AnimatePresence>
      {footnote && <p className="body-small mt-4 border-t border-line pt-3 text-ink-2">{footnote}</p>}
    </Card>
  );
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <ul className="body-small mb-3 flex flex-wrap gap-x-4 gap-y-1 text-ink-2">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: item.color }} aria-hidden />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

// ── Citation verdicts: status colour always comes with an icon and a label ──
export const TIERS: { tier: Tier; label: string; color: string; icon: string; description: string }[] = [
  { tier: "SUPPORTED", label: "Supported", color: "var(--good)", icon: "check_circle", description: "The case exists and its opinion supports the holding." },
  { tier: "UNCERTAIN", label: "Uncertain", color: "var(--warning)", icon: "help", description: "The case exists; support for the holding is borderline." },
  { tier: "MISATTRIBUTED", label: "Misattributed", color: "var(--serious)", icon: "warning", description: "The case exists, but its opinion does not support the holding." },
  { tier: "FABRICATED", label: "Fabricated", color: "var(--critical)", icon: "dangerous", description: "No such case at that citation in a volume the corpus covers." },
  { tier: "UNVERIFIABLE", label: "Unverifiable", color: "var(--muted)", icon: "visibility_off", description: "The opinion text is not in the corpus, so it cannot be checked." },
];

/** Verdicts the checker gives while no holding-support model has passed validation (existence only). */
const EXISTENCE_ONLY: { tier: string; label: string; color: string; icon: string; description: string }[] = [
  { tier: "VERIFIED", label: "Exists", color: "var(--good)", icon: "verified", description: "The case exists in the corpus. Whether its opinion supports the holding was not validated." },
  { tier: "NOT_SHOWN", label: "Support not shown", color: "var(--warning)", icon: "help", description: "The case exists; the passages read did not show the holding." },
];

export function VerdictBadge({ tier }: { tier: Tier | string }) {
  const meta = [...TIERS, ...EXISTENCE_ONLY].find((item) => item.tier === tier) ?? TIERS[4];
  return (
    <span className="label-medium inline-flex h-7 items-center gap-1.5 whitespace-nowrap rounded-lg border border-hairline px-2 text-ink" title={meta.description}>
      <Icon name={meta.icon} size={16} fill style={{ color: meta.color }} />
      {meta.label}
    </span>
  );
}

/** M3 linear indicator used as a 0–1 meter. The track is a lighter step of the fill's own colour. */
export function Meter({ value, color = "var(--md-sys-color-primary)", label }: { value: number; color?: string; label?: string }) {
  const clamped = Math.max(0, Math.min(1, value));
  return (
    <div className="h-1 w-full overflow-hidden rounded-full" style={{ background: `color-mix(in srgb, ${color} 22%, transparent)` }} role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={clamped} aria-label={label}>
      <motion.div className="h-full rounded-full" style={{ background: color }} initial={{ width: 0 }} animate={{ width: `${clamped * 100}%` }} transition={{ duration: 0.7, ease: M3_EASE.emphasizedDecelerate }} />
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-surface-2 ${className}`} />;
}

export function PageHeader({ title, lead }: { title: string; lead: ReactNode }) {
  return (
    <motion.header {...riseIn} className="mb-6">
      <h1 className="headline-small text-ink">{title}</h1>
      <p className="body-large mt-1 max-w-3xl text-ink-2">{lead}</p>
    </motion.header>
  );
}

/** Keeps a scroll container pinned to its bottom while new content arrives, unless the reader scrolled up. */
export function useStickToBottom<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const pinned = useRef(true);
  const onScroll = useCallback(() => {
    const node = ref.current;
    if (node) pinned.current = node.scrollHeight - node.scrollTop - node.clientHeight < 120;
  }, []);
  const stick = useCallback(() => {
    const node = ref.current;
    if (node && pinned.current) node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, []);
  return { ref, onScroll, stick };
}
