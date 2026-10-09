import { AnimatePresence, LayoutGroup, motion, MotionConfig } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import { Icon, IconButton, TooltipLayer } from "./components/ui";
import { api, type CaseRow, type Overview, type Status } from "./lib/api";
import { M3_DURATION, M3_EASE, useRipple } from "./lib/motion";
import { AnalysisPage } from "./pages/AnalysisPage";
import { CasesPage } from "./pages/CasesPage";
import { ChatPage } from "./pages/ChatPage";
import { CorpusPage } from "./pages/CorpusPage";

type View = "chat" | "analysis" | "cases" | "corpus";
const DESTINATIONS: { view: View; label: string; icon: string }[] = [
  { view: "chat", label: "Chat", icon: "forum" },
  { view: "analysis", label: "Evaluation", icon: "monitoring" },
  { view: "cases", label: "Cases", icon: "gavel" },
  { view: "corpus", label: "Corpus", icon: "database" },
];

function readRoute(): { view: View; replay: boolean } {
  const [path, query = ""] = window.location.hash.replace(/^#\/?/, "").split("?");
  const view = DESTINATIONS.some((item) => item.view === path) ? (path as View) : "chat";
  return { view, replay: new URLSearchParams(query).get("replay") === "1" };
}

/** One destination of the M3 navigation rail or bar: the active indicator slides between items. */
function Destination({ item, active, onSelect, layout }: { item: (typeof DESTINATIONS)[number]; active: boolean; onSelect: () => void; layout: "rail" | "bar" }) {
  const ripple = useRipple();
  return (
    <button type="button" onClick={onSelect} aria-current={active ? "page" : undefined} className={`group flex flex-col items-center gap-1 ${layout === "rail" ? "w-full py-1.5" : "flex-1 py-3"} ${active ? "text-ink" : "text-ink-2"}`}>
      <span onPointerDown={ripple.onPointerDown} className="relative flex h-8 w-14 items-center justify-center overflow-hidden rounded-full">
        {active && <motion.span layoutId={`indicator-${layout}`} className="absolute inset-0 rounded-full bg-secondary-container" transition={{ duration: M3_DURATION.medium, ease: M3_EASE.emphasized }} />}
        <span className="absolute inset-0 rounded-full bg-current opacity-0 transition-opacity group-hover:opacity-[0.08]" />
        {ripple.layer}
        <Icon name={item.icon} size={24} fill={active} className={`relative ${active ? "text-on-secondary-container" : ""}`} />
      </span>
      <span className={`label-medium ${active ? "font-semibold" : ""}`}>{item.label}</span>
    </button>
  );
}

function EngineDot({ status }: { status: Status | null }) {
  const state = status?.engine.state ?? "cold";
  const meta = {
    ready: { color: "var(--good)", label: "Model loaded" },
    loading: { color: "var(--warning)", label: "Model loading" },
    error: { color: "var(--critical)", label: "Model failed to load" },
    cold: { color: "var(--md-sys-color-outline)", label: "Model not loaded" },
  }[state];
  return (
    <div className="flex flex-col items-center gap-1 text-ink-2" title={status?.engine.message ?? meta.label}>
      <span className="relative flex h-3 w-3">
        {state === "loading" && <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ background: meta.color }} />}
        <span className="relative inline-flex h-3 w-3 rounded-full" style={{ background: meta.color }} />
      </span>
      <span className="label-small text-center leading-tight">{state === "ready" ? "Ready" : state === "loading" ? "Loading" : state === "error" ? "Error" : "Idle"}</span>
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState(readRoute);
  const [dark, setDark] = useState(() => document.documentElement.dataset.theme === "dark");
  const [status, setStatus] = useState<Status | null>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [cases, setCases] = useState<CaseRow[] | null>(null);

  useEffect(() => {
    const onHash = () => setRoute(readRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    let alive = true;
    const poll = () => api.status().then((data) => alive && setStatus(data)).catch(() => undefined);
    poll();
    const timer = window.setInterval(poll, 5000);
    api.overview().then((data) => alive && setOverview(data)).catch(() => undefined);
    api.cases().then((data) => alive && setCases(data.cases)).catch(() => alive && setCases([]));
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  const go = useCallback((view: View) => {
    window.location.hash = `/${view}`;
  }, []);

  const toggleTheme = useCallback(() => {
    setDark((current) => {
      const next = !current;
      document.documentElement.dataset.theme = next ? "dark" : "light";
      try {
        localStorage.setItem("lexagent-theme", next ? "dark" : "light");
      } catch {
        /* storage unavailable */
      }
      return next;
    });
  }, []);

  const maxRounds = overview?.corpus.models.max_rounds ?? 3;

  return (
    <MotionConfig reducedMotion="user">
      <TooltipLayer>
        <div className="flex h-full bg-bg text-ink">
          {/* M3 navigation rail on wide screens */}
          <nav aria-label="Main" className="hidden w-[88px] shrink-0 flex-col items-center gap-3 py-4 sm:flex">
            <div className="mb-3 flex flex-col items-center gap-1">
              <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-wash text-on-accent-wash">
                <Icon name="balance" size={26} fill />
              </span>
              <span className="label-small text-ink-2">LexAgent</span>
            </div>
            <LayoutGroup id="rail">
              {DESTINATIONS.map((item) => (
                <Destination key={item.view} item={item} layout="rail" active={route.view === item.view} onSelect={() => go(item.view)} />
              ))}
            </LayoutGroup>
            <div className="mt-auto flex flex-col items-center gap-3">
              <EngineDot status={status} />
              <IconButton icon={dark ? "light_mode" : "dark_mode"} label={dark ? "Switch to light theme" : "Switch to dark theme"} onClick={toggleTheme} />
            </div>
          </nav>

          <div className="flex min-w-0 flex-1 flex-col">
            {/* Compact top bar on narrow screens */}
            <header className="flex items-center justify-between px-4 py-2 sm:hidden">
              <span className="title-large flex items-center gap-2">
                <Icon name="balance" size={24} fill className="text-accent" />
                LexAgent
              </span>
              <IconButton icon={dark ? "light_mode" : "dark_mode"} label={dark ? "Switch to light theme" : "Switch to dark theme"} onClick={toggleTheme} />
            </header>

            <main className="min-h-0 flex-1 sm:py-2 sm:pr-2">
              <div className="h-full overflow-hidden bg-bg sm:rounded-[28px] sm:bg-surface-container">
                <AnimatePresence mode="wait">
                  {/* M3 "fade through": the outgoing page fades out, the incoming one fades in and scales up slightly. */}
                  <motion.div
                    key={route.view}
                    initial={{ opacity: 0, scale: 0.97 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: M3_DURATION.short, ease: M3_EASE.emphasizedDecelerate }}
                    className={`scroll-thin h-full ${route.view === "chat" ? "" : "overflow-y-auto"}`}
                  >
                    {route.view === "chat" && <ChatPage status={status} maxRounds={maxRounds} autoReplay={route.replay} />}
                    {route.view === "analysis" && <AnalysisPage overview={overview} cases={cases} />}
                    {route.view === "cases" && <CasesPage cases={cases} />}
                    {route.view === "corpus" && <CorpusPage overview={overview} />}
                  </motion.div>
                </AnimatePresence>
              </div>
            </main>

            {/* M3 navigation bar on narrow screens */}
            <nav aria-label="Main" className="flex bg-surface-container sm:hidden">
              <LayoutGroup id="bar">
                {DESTINATIONS.map((item) => (
                  <Destination key={item.view} item={item} layout="bar" active={route.view === item.view} onSelect={() => go(item.view)} />
                ))}
              </LayoutGroup>
            </nav>
          </div>
        </div>
      </TooltipLayer>
    </MotionConfig>
  );
}
