import { AnimatePresence, LayoutGroup, motion, MotionConfig } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import { Icon, TooltipLayer } from "./components/ui";
import { api, type CaseRow, type Overview, type Status } from "./lib/api";
import { M3_DURATION, M3_EASE } from "./lib/motion";
import { AnalysisPage } from "./pages/AnalysisPage";
import { CasesPage } from "./pages/CasesPage";
import { ChatPage } from "./pages/ChatPage";
import { CorpusPage } from "./pages/CorpusPage";
import { LandingPage } from "./pages/LandingPage";

type View = "home" | "chat" | "analysis" | "cases" | "corpus";
type AppView = Exclude<View, "home">;
const DESTINATIONS: { view: AppView; label: string; icon: string }[] = [
  { view: "chat", label: "Ask", icon: "forum" },
  { view: "analysis", label: "Evaluation", icon: "monitoring" },
  { view: "cases", label: "Cases", icon: "gavel" },
  { view: "corpus", label: "Corpus", icon: "database" },
];

function readRoute(): { view: View; replay: boolean } {
  const [path, query = ""] = window.location.hash.replace(/^#\/?/, "").split("?");
  const view: View = DESTINATIONS.some((item) => item.view === path) ? (path as AppView) : "home";
  return { view, replay: new URLSearchParams(query).get("replay") === "1" };
}

/** Engine state in words, with a dot that is never the only signal. */
function engineMeta(status: Status | null, reachable: boolean) {
  if (!reachable) return { color: "var(--muted)", label: "Offline", detail: "The backend is not running." };
  const state = status?.engine.state ?? "cold";
  return {
    ready: { color: "var(--good)", label: "Ready", detail: "Models loaded." },
    loading: { color: "var(--warning)", label: "Starting", detail: "Loading the models…" },
    error: { color: "var(--critical)", label: "Error", detail: status?.engine.message ?? "The models failed to load." },
    cold: { color: "var(--muted)", label: "Idle", detail: "Models load with the first question." },
  }[state];
}

function SidebarItem({ item, active, onSelect }: { item: (typeof DESTINATIONS)[number]; active: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={active ? "page" : undefined}
      className={`state-layer relative flex h-10 w-full items-center gap-3 rounded-[10px] px-3 text-left label-large ${active ? "text-ink" : "text-ink-2 hover:text-ink"}`}
    >
      {active && <motion.span layoutId="sidebar-active" className="absolute inset-0 -z-10 rounded-[10px] bg-secondary-container" transition={{ duration: M3_DURATION.medium, ease: M3_EASE.emphasized }} />}
      <Icon name={item.icon} size={18} fill={active} />
      {item.label}
    </button>
  );
}

function TabItem({ item, active, onSelect }: { item: (typeof DESTINATIONS)[number]; active: boolean; onSelect: () => void }) {
  return (
    <button type="button" onClick={onSelect} aria-current={active ? "page" : undefined} className={`flex min-h-[52px] flex-1 flex-col items-center justify-center gap-0.5 ${active ? "text-ink" : "text-ink-2"}`}>
      <Icon name={item.icon} size={22} fill={active} />
      <span className="label-small">{item.label}</span>
    </button>
  );
}

export default function App() {
  const [route, setRoute] = useState(readRoute);
  const [status, setStatus] = useState<Status | null>(null);
  const [reachable, setReachable] = useState(true);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [cases, setCases] = useState<CaseRow[] | null>(null);

  useEffect(() => {
    const onHash = () => setRoute(readRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      api
        .status()
        .then((data) => {
          if (!alive) return;
          setStatus(data);
          setReachable(true);
        })
        .catch(() => alive && setReachable(false));
    poll();
    const timer = window.setInterval(poll, 5000);
    api.overview().then((data) => alive && setOverview(data)).catch(() => undefined);
    api.cases().then((data) => alive && setCases(data.cases)).catch(() => alive && setCases([]));
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [route.view]);

  const go = useCallback((view: View) => {
    window.location.hash = view === "home" ? "/" : `/${view}`;
  }, []);

  const maxRounds = overview?.corpus.models.max_rounds ?? 3;
  // Evaluation and Cases show benchmark results; until a benchmark has been run they stay out of the navigation.
  const hasResults = Boolean(overview?.benchmark) || Boolean(cases?.length);
  const destinations = DESTINATIONS.filter((item) => hasResults || (item.view !== "analysis" && item.view !== "cases"));
  const generator = overview?.corpus.models.generator;
  const engine = engineMeta(status, reachable);

  if (route.view === "home") {
    return (
      <MotionConfig reducedMotion="user">
        <LandingPage status={reachable ? status : null} overview={overview} />
      </MotionConfig>
    );
  }

  return (
    <MotionConfig reducedMotion="user">
      <TooltipLayer>
        <div className="flex h-full bg-bg text-ink">
          {/* Sidebar on regular widths */}
          <nav aria-label="Main" className="hidden w-[232px] shrink-0 flex-col gap-1 px-3 pb-4 pt-5 md:flex">
            <a href="#/" className="mb-5 flex items-center gap-2.5 rounded-lg px-3 py-1 text-ink" aria-label="LexAgent home">
              <Icon name="balance" size={20} />
              <span className="font-display text-[21px] leading-none tracking-[-0.01em]">LexAgent</span>
            </a>
            <LayoutGroup id="sidebar">
              {destinations.map((item) => (
                <SidebarItem key={item.view} item={item} active={route.view === item.view} onSelect={() => go(item.view)} />
              ))}
            </LayoutGroup>

            <div className="mt-auto flex flex-col gap-3 px-3">
              <div className="flex items-start gap-2.5" role="status" aria-live="polite">
                <span className="relative mt-[5px] flex h-2 w-2 shrink-0">
                  {engine.label === "Starting" && <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ background: engine.color }} />}
                  <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: engine.color }} />
                </span>
                <span className="min-w-0">
                  <span className="label-medium block text-ink">{engine.label}</span>
                  <span className="body-small block text-ink-2">{engine.detail}</span>
                </span>
              </div>
              <p className="body-small border-t border-line pt-3 text-muted">
                Answers are written by an AI model{generator ? ` (${generator})` : ""} and can be wrong. Not legal advice.
              </p>
            </div>
          </nav>

          <div className="flex min-w-0 flex-1 flex-col">
            {/* Compact title bar on narrow screens */}
            <header className="flex h-12 items-center justify-between px-4 md:hidden">
              <a href="#/" className="flex items-center gap-2 text-ink" aria-label="LexAgent home">
                <Icon name="balance" size={18} />
                <span className="font-display text-[19px] leading-none">LexAgent</span>
              </a>
              <span className="label-medium flex items-center gap-1.5 text-ink-2" role="status">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: engine.color }} />
                {engine.label}
              </span>
            </header>

            <main className="min-h-0 flex-1 md:py-2 md:pr-2">
              <div className={`h-full overflow-hidden ${route.view === "chat" ? "paper md:rounded-[20px] md:border md:border-[#d9cdb8] md:shadow-[var(--shadow-1)]" : ""}`}>
                <AnimatePresence mode="wait">
                  <motion.div
                    key={route.view}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: M3_DURATION.short, ease: M3_EASE.standard }}
                    className={`scroll-thin h-full ${route.view === "chat" ? "" : "overflow-y-auto"}`}
                  >
                    {route.view === "chat" && <ChatPage status={reachable ? status : null} offline={!reachable} maxRounds={maxRounds} autoReplay={route.replay} />}
                    {route.view === "analysis" && <AnalysisPage overview={overview} cases={cases} />}
                    {route.view === "cases" && <CasesPage cases={cases} />}
                    {route.view === "corpus" && <CorpusPage overview={overview} />}
                  </motion.div>
                </AnimatePresence>
              </div>
            </main>

            {/* Tab bar on narrow screens */}
            <nav aria-label="Main" className="glass flex border-x-0 border-b-0 pb-[env(safe-area-inset-bottom)] md:hidden">
              {destinations.map((item) => (
                <TabItem key={item.view} item={item} active={route.view === item.view} onSelect={() => go(item.view)} />
              ))}
            </nav>
          </div>
        </div>
      </TooltipLayer>
    </MotionConfig>
  );
}
