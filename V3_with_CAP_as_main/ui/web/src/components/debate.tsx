import { useGSAP } from "@gsap/react";
import { AnimatePresence, motion } from "framer-motion";
import gsap from "gsap";
import { useRef, useState, type ReactNode } from "react";
import { modelLabel, type Citation, type CitationCheck, type DebateEvent, type Passage, type StageName } from "../lib/api";
import { num } from "../lib/format";
import { CountUp, M3_EASE, prefersReducedMotion, spring } from "../lib/motion";
import { Chip, Icon, Meter, VerdictBadge } from "./ui";

// ── The state of one debate, built up from the server's events ──────────────
export interface Round {
  n: number;
  retrieval?: Extract<DebateEvent, { type: "retrieval" }>;
  prosecutor?: Extract<DebateEvent, { type: "prosecutor" }>;
  defense?: Extract<DebateEvent, { type: "defense" }>;
  cce?: Extract<DebateEvent, { type: "cce" }>;
  reflection?: Extract<DebateEvent, { type: "reflection" }>;
  ddc?: Extract<DebateEvent, { type: "ddc" }>;
}

export interface Debate {
  status: "running" | "done" | "error";
  stage: StageName;
  stageMessage?: string;
  round: number;
  rounds: Round[];
  memoryHit?: string;
  judge?: Extract<DebateEvent, { type: "judge" }>;
  done?: Extract<DebateEvent, { type: "done" }>;
  error?: string;
  replay?: boolean;
  model?: string;
}

export const newDebate = (replay = false): Debate => ({ status: "running", stage: "memory_check", round: 1, rounds: [], replay });

export function applyEvent(debate: Debate, event: DebateEvent): Debate {
  const next: Debate = { ...debate, rounds: debate.rounds.map((round) => ({ ...round })) };
  const roundOf = (n: number): Round => {
    let round = next.rounds.find((item) => item.n === n);
    if (!round) {
      round = { n };
      next.rounds.push(round);
    }
    return round;
  };
  switch (event.type) {
    case "stage":
      next.stage = event.stage;
      next.stageMessage = event.message;
      if (event.round) next.round = event.round;
      if (event.model) next.model = event.model;
      break;
    case "memory":
      if (event.hit) next.memoryHit = event.answer ?? "";
      break;
    case "retrieval":
      roundOf(event.round).retrieval = event;
      break;
    case "prosecutor":
      roundOf(event.round).prosecutor = event;
      break;
    case "defense":
      roundOf(event.round).defense = event;
      break;
    case "cce":
      roundOf(event.round).cce = event;
      break;
    case "reflection":
      roundOf(event.round).reflection = event;
      break;
    case "ddc":
      roundOf(event.round).ddc = event;
      break;
    case "judge":
      next.judge = event;
      break;
    case "done":
      next.done = event;
      next.status = "done";
      break;
    case "error":
      next.error = event.message;
      next.status = "error";
      break;
  }
  return next;
}

// ── Stage rail: where the pipeline is right now (GSAP drives the progress line and the pulse) ──
const STAGES: { key: StageName; label: string }[] = [
  { key: "memory_check", label: "Memory" },
  { key: "retrieve", label: "Retrieve" },
  { key: "prosecutor", label: "Prosecutor" },
  { key: "defense", label: "Defense" },
  { key: "cce", label: "Verify" },
  { key: "reflection", label: "Reflect" },
  { key: "ddc", label: "Decide" },
  { key: "judge", label: "Judge" },
];

export function StageRail({ debate, maxRounds }: { debate: Debate; maxRounds: number }) {
  const scope = useRef<HTMLDivElement>(null);
  const finished = debate.status === "done";
  const index = finished ? STAGES.length - 1 : Math.max(0, STAGES.findIndex((stage) => stage.key === (debate.stage === "memory_store" ? "judge" : debate.stage)));
  const progress = index / (STAGES.length - 1);

  useGSAP(
    () => {
      const reduce = prefersReducedMotion();
      gsap.to(".rail-fill", { scaleX: progress, duration: reduce ? 0 : 0.6, ease: "power2.out" });
      gsap.killTweensOf(".rail-dot");
      gsap.set(".rail-dot", { scale: 1 });
      if (!finished && debate.status === "running" && !reduce) {
        gsap.to(`.rail-dot-${index}`, { scale: 1.35, duration: 0.9, ease: "sine.inOut", yoyo: true, repeat: -1 });
      }
    },
    { scope, dependencies: [index, finished, debate.status] },
  );

  return (
    <div ref={scope} className="rounded-2xl border border-line bg-surface-container px-5 pb-3 pt-3.5 shadow-[var(--shadow-1)]">
      <div className="label-medium mb-3 flex items-center justify-between text-ink-2">
        <span className="flex items-center gap-1.5">
          <Icon name={finished ? "task_alt" : debate.status === "error" ? "error" : "hourglass_top"} size={16} />
          {debate.stage === "loading" ? debate.stageMessage ?? "Loading the models..." : finished ? "Debate complete" : debate.status === "error" ? "Stopped" : "Debate in progress"}
        </span>
        <span className="tnum">
          {debate.model ? `${modelLabel(debate.model)} · ` : ""}Round {debate.round} of up to {maxRounds}
        </span>
      </div>
      <div className="relative mx-3">
        <div className="absolute left-0 right-0 top-[4px] h-1 rounded-full bg-surface-highest" />
        <div className="rail-fill absolute left-0 right-0 top-[4px] h-1 origin-left scale-x-0 rounded-full bg-accent" />
        <ol className="relative flex justify-between">
          {STAGES.map((stage, stageIndex) => (
            <li key={stage.key} className="flex w-0 flex-col items-center">
              <span className={`rail-dot rail-dot-${stageIndex} block h-3 w-3 rounded-full ${stageIndex <= index ? "bg-accent" : "bg-axis"}`} style={{ boxShadow: "0 0 0 2px var(--md-sys-color-surface-container)" }} />
              {/* Eight labels do not fit a phone-width rail, so a narrow screen names only the current stage. */}
              <span className={`label-small mt-1.5 whitespace-nowrap ${stageIndex === index ? "" : "hidden sm:block"} ${stageIndex === index && !finished ? "text-ink" : "text-muted"}`}>{stage.label}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

// ── Chat bubbles for each participant ───────────────────────────────────────
const ROLES = {
  retriever: { name: "Retriever", icon: "menu_book", color: "var(--md-sys-color-on-surface-variant)" },
  prosecutor: { name: "Prosecutor", icon: "swords", color: "var(--role-prosecutor)" },
  defense: { name: "Defense", icon: "shield", color: "var(--role-defense)" },
  checker: { name: "Citation check", icon: "fact_check", color: "var(--md-sys-color-on-surface-variant)" },
  reflection: { name: "Reflection", icon: "psychology", color: "var(--md-sys-color-on-surface-variant)" },
  controller: { name: "Debate controller", icon: "alt_route", color: "var(--md-sys-color-on-surface-variant)" },
  judge: { name: "Judge", icon: "gavel", color: "var(--role-judge)" },
  memory: { name: "Memory", icon: "history", color: "var(--md-sys-color-on-surface-variant)" },
} as const;

function Message({ role, meta, children, emphasis = false }: { role: keyof typeof ROLES; meta?: ReactNode; children: ReactNode; emphasis?: boolean }) {
  const info = ROLES[role];
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={spring} className="flex gap-3">
      <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-line bg-surface" aria-hidden>
        <Icon name={info.icon} size={17} style={{ color: info.color }} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
          <span className="title-small" style={{ color: role === "prosecutor" || role === "defense" ? info.color : "var(--ink)" }}>{info.name}</span>
          {meta && <span className="body-small text-muted">{meta}</span>}
        </div>
        <div className={`body-medium rounded-[18px] rounded-tl-[6px] border px-4 py-3 text-ink ${emphasis ? "border-line bg-surface-container shadow-[var(--shadow-2)]" : "border-line bg-surface"}`}>{children}</div>
      </div>
    </motion.div>
  );
}

function Disclosure({ summary, children, defaultOpen = false }: { summary: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="body-medium flex w-full items-center justify-between gap-3 text-left text-ink-2 hover:text-ink">
        <span className="min-w-0">{summary}</span>
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2, ease: M3_EASE.standard }} className="flex shrink-0">
          <Icon name="expand_more" size={20} />
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.3, ease: M3_EASE.emphasized }} className="overflow-hidden">
            <div className="pt-3">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function PassageList({ passages }: { passages: Passage[] }) {
  return (
    <ul className="flex flex-col gap-2">
      {passages.map((passage, index) => (
        <motion.li key={`${passage.doc_id}-${index}`} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * 0.04, ease: M3_EASE.emphasizedDecelerate }} className="rounded-xl border border-line bg-surface-container px-3 py-2.5">
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="title-small text-ink">{passage.case_name}</span>
            {passage.cite && <span className="tnum body-small text-ink-2">{passage.cite}</span>}
            {passage.year > 0 && <span className="tnum body-small text-muted">({passage.year})</span>}
          </div>
          <p className="body-small mt-1 line-clamp-3 text-ink-2">{passage.text}</p>
        </motion.li>
      ))}
    </ul>
  );
}

function CitationChips({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <div className="mt-3 flex flex-wrap gap-1.5 border-t border-line pt-3">
      {citations.map((cite, index) => (
        <Chip key={`${cite.case_name}-${index}`} icon="format_quote">
          {cite.case_name}
          {cite.reporter_cite ? `, ${cite.reporter_cite}` : ""}
        </Chip>
      ))}
    </div>
  );
}

const CITE_CHECK_NOTE: Record<string, string> = {
  MATCH: "Reporter cite matches the case.",
  MISMATCH: "The reporter cite belongs to a different case.",
  UNKNOWN_PAGE: "No report starts at the cited page.",
  VOLUME_NOT_IN_CORPUS: "The cited volume is not in the corpus.",
};

function CheckRow({ check }: { check: CitationCheck }) {
  return (
    <div className="rounded-xl bg-surface-2 px-3 py-2.5">
      <Disclosure
        summary={
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <VerdictBadge tier={check.tier} />
            <span className="title-small text-ink">{check.case_name}</span>
            {check.reporter_cite && <span className="tnum body-small text-ink-2">{check.reporter_cite}</span>}
            <span className="body-small text-muted">cited by the {check.agent}</span>
          </span>
        }
      >
        <dl className="body-small grid grid-cols-[112px_1fr] gap-x-3 gap-y-2">
          <dt className="text-muted">Claimed holding</dt>
          <dd className="text-ink-2">{check.holding || "none given"}</dd>
          <dt className="text-muted">Exists</dt>
          <dd className="tnum text-ink-2">
            {num(check.existence)} {check.best_match && check.best_match !== check.case_name ? `(matched to ${check.best_match})` : ""}
          </dd>
          <dt className="text-muted">Holding support</dt>
          <dd className="flex items-center gap-2 text-ink-2">
            <span className="tnum w-9">{num(check.support)}</span>
            {check.support !== null && (
              <span className="w-28">
                <Meter value={check.support} label="Holding support" />
              </span>
            )}
          </dd>
          {check.evidence_source && (
            <>
              <dt className="text-muted">Checked against</dt>
              <dd className="text-ink-2">{check.evidence_source}</dd>
            </>
          )}
          {CITE_CHECK_NOTE[check.cite_check] && (
            <>
              <dt className="text-muted">Reporter cite</dt>
              <dd className="text-ink-2">{CITE_CHECK_NOTE[check.cite_check]}</dd>
            </>
          )}
          {check.evidence && (
            <>
              <dt className="text-muted">Opinion text</dt>
              <dd className="italic text-ink-2">“{check.evidence}…”</dd>
            </>
          )}
        </dl>
      </Disclosure>
    </div>
  );
}

// ── Judge confidence gauge (GSAP sweeps the arc) ────────────────────────────
export function ConfidenceGauge({ value }: { value: number }) {
  const arc = useRef<SVGPathElement>(null);
  const length = Math.PI * 52;
  const level =
    value >= 0.7
      ? { label: "High confidence", color: "var(--good)", icon: "check_circle" }
      : value >= 0.4
        ? { label: "Moderate confidence", color: "var(--warning)", icon: "help" }
        : { label: "Low confidence", color: "var(--serious)", icon: "warning" };
  useGSAP(
    () => {
      if (!arc.current) return;
      gsap.fromTo(arc.current, { strokeDashoffset: length }, { strokeDashoffset: length * (1 - Math.max(0, Math.min(1, value))), duration: prefersReducedMotion() ? 0 : 1.2, ease: "power3.out" });
    },
    { dependencies: [value] },
  );
  return (
    <div className="flex flex-col items-center rounded-2xl bg-chart px-4 pb-3 pt-4 text-ink">
      <svg viewBox="0 0 120 68" className="w-36" role="img" aria-label={`Judge confidence ${num(value)}: ${level.label}`}>
        <path d="M 8 60 A 52 52 0 0 1 112 60" fill="none" stroke="var(--md-sys-color-surface-container-highest)" strokeWidth="8" strokeLinecap="round" />
        <path ref={arc} d="M 8 60 A 52 52 0 0 1 112 60" fill="none" stroke={level.color} strokeWidth="8" strokeLinecap="round" strokeDasharray={length} strokeDashoffset={length} />
      </svg>
      <div className="headline-medium -mt-10">
        <CountUp value={value} format={(n) => n.toFixed(2)} />
      </div>
      <div className="label-medium mt-1 flex items-center gap-1 text-ink-2">
        <Icon name={level.icon} size={15} fill style={{ color: level.color }} />
        {level.label}
      </div>
    </div>
  );
}

// ── One assistant turn: the whole debate rendered as a conversation ─────────
export function DebateThread({ debate, maxRounds }: { debate: Debate; maxRounds: number }) {
  return (
    <div className="flex flex-col gap-4">
      <div className="sticky top-2 z-10">
        <StageRail debate={debate} maxRounds={maxRounds} />
      </div>

      {debate.memoryHit !== undefined && (
        <Message role="memory" meta="answered from a cached verdict" emphasis>
          {debate.memoryHit || "A matching verdict was found in memory."}
        </Message>
      )}

      {debate.rounds.map((round) => (
        <div key={round.n} className="flex flex-col gap-4">
          {debate.rounds.length > 1 || round.n > 1 ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="label-medium flex items-center gap-3 uppercase text-muted">
              <span className="h-px flex-1 bg-line" />
              Round {round.n}
              <span className="h-px flex-1 bg-line" />
            </motion.div>
          ) : null}

          {round.retrieval && (
            <Message role="retriever" meta={`${round.retrieval.passages.length} passages by hybrid search`}>
              <Disclosure defaultOpen={round.n === 1} summary={<span>Authorities retrieved: {[...new Set(round.retrieval.passages.map((p) => p.case_name))].slice(0, 4).join("; ")}</span>}>
                <PassageList passages={round.retrieval.passages} />
              </Disclosure>
            </Message>
          )}

          {round.prosecutor && (
            <Message role="prosecutor" meta="argues the position">
              {round.prosecutor.argument ? <p className="whitespace-pre-wrap">{round.prosecutor.argument}</p> : <p className="text-ink-2">No argument this round: the language model did not answer in time, usually because it was busy. The debate went on without it.</p>}
              <CitationChips citations={round.prosecutor.citations} />
            </Message>
          )}

          {round.defense && (
            <Message role="defense" meta={round.defense.concede ? "concedes the point" : `challenge strength ${num(round.defense.challenge_strength)}`}>
              {round.defense.argument ? <p className="whitespace-pre-wrap">{round.defense.argument}</p> : <p className="text-ink-2">No counter-argument this round: the language model did not answer in time, usually because it was busy. The debate went on without it.</p>}
              {round.defense.challenges.length > 0 && (
                <ul className="mt-2 list-disc pl-5 text-ink-2">
                  {round.defense.challenges.slice(0, 4).map((challenge, index) => (
                    <li key={index}>{String(challenge)}</li>
                  ))}
                </ul>
              )}
              <CitationChips citations={round.defense.citations} />
            </Message>
          )}

          {round.cce && (
            <Message role="checker" meta={`${round.cce.checks.length} citations checked · average confidence ${num(round.cce.avg_ccs)}`}>
              {round.cce.checks.length === 0 ? (
                <span className="text-ink-2">No citations were made in this round.</span>
              ) : (
                <div className="flex flex-col gap-2">
                  {round.cce.checks.map((check) => (
                    <CheckRow key={check.key} check={check} />
                  ))}
                </div>
              )}
            </Message>
          )}

          {round.reflection && (round.reflection.gaps.length > 0 || round.reflection.queries.length > 0) && (
            <Message role="reflection" meta={`gap score ${num(round.reflection.score)}`}>
              <Disclosure summary={<span>{round.reflection.gaps.length} open gaps identified</span>}>
                <ul className="list-disc pl-5 text-ink-2">
                  {round.reflection.gaps.map((gap, index) => (
                    <li key={index}>{String(gap)}</li>
                  ))}
                </ul>
              </Disclosure>
            </Message>
          )}

          {round.ddc && (
            <Message role="controller" meta={round.ddc.decision === "CONTINUE" ? "another round" : "debate closed"}>
              <span className="text-ink-2">{round.ddc.reason || round.ddc.decision}</span>
            </Message>
          )}
        </div>
      ))}

      {debate.judge && (
        <Message role="judge" meta="final verdict" emphasis>
          <div className="grid gap-5 sm:grid-cols-[1fr_auto] sm:items-center">
            <div className="min-w-0">
              <p className="font-display whitespace-pre-wrap text-[20px] leading-[30px] text-ink">{debate.judge.final_answer || debate.judge.verdict}</p>
              {debate.judge.reasoning && debate.judge.reasoning !== debate.judge.final_answer && (
                <div className="mt-3">
                  <Disclosure summary="Judge's reasoning">
                    <p className="whitespace-pre-wrap">{debate.judge.reasoning}</p>
                  </Disclosure>
                </div>
              )}
            </div>
            <ConfidenceGauge value={debate.judge.confidence} />
          </div>
          {(debate.judge.verified.length > 0 || debate.judge.uncertain.length > 0) && (
            <div className="mt-4 grid gap-3 border-t border-line pt-3 sm:grid-cols-2">
              <div>
                <div className="label-medium mb-1.5">Relied on ({debate.judge.verified.length})</div>
                <div className="flex flex-wrap gap-1.5">
                  {debate.judge.verified.map((cite, index) => (
                    <Chip key={index} icon="verified" selected>
                      {cite.case_name}
                    </Chip>
                  ))}
                  {debate.judge.verified.length === 0 && <span className="body-small">none</span>}
                </div>
              </div>
              <div>
                <div className="label-medium mb-1.5">Set aside as uncertain ({debate.judge.uncertain.length})</div>
                <div className="flex flex-wrap gap-1.5">
                  {debate.judge.uncertain.map((cite, index) => (
                    <Chip key={index}>{cite.case_name}</Chip>
                  ))}
                  {debate.judge.uncertain.length === 0 && <span className="body-small">none</span>}
                </div>
              </div>
            </div>
          )}
        </Message>
      )}

      {debate.done && (
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="body-small pl-[52px] text-muted">
          {debate.done.rounds} {debate.done.rounds === 1 ? "round" : "rounds"} · {debate.done.seconds.toFixed(0)} s{debate.replay ? " · replay of a recorded debate" : ""}
        </motion.p>
      )}

      {debate.error && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="body-medium ml-[52px] flex items-start gap-2 rounded-xl bg-error-container px-4 py-3 text-on-error-container">
          <Icon name="error" size={20} fill />
          <span>
            <b>The debate stopped.</b> {debate.error}
          </span>
        </motion.div>
      )}

      {debate.status === "running" && <TypingDots stage={debate.stage} />}
    </div>
  );
}

const WORKING: Partial<Record<StageName, string>> = {
  loading: "Loading the corpus and the models",
  memory_check: "Checking memory for a matching verdict",
  retrieve: "Searching the corpus",
  prosecutor: "The prosecutor is writing",
  defense: "The defense is writing",
  cce: "Checking every citation against its opinion",
  reflection: "Looking for gaps in the debate",
  ddc: "Deciding whether another round is needed",
  judge: "The judge is deliberating",
  memory_store: "Saving the verdict",
};

function TypingDots({ stage }: { stage: StageName }) {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="body-medium flex items-center gap-3 pl-12 text-ink-2" aria-live="polite">
      <span className="flex gap-1" aria-hidden>
        {[0, 1, 2].map((dot) => (
          <motion.span key={dot} className="block h-1.5 w-1.5 rounded-full bg-accent" animate={{ y: [0, -4, 0], opacity: [0.4, 1, 0.4] }} transition={{ duration: 0.9, repeat: Infinity, delay: dot * 0.15 }} />
        ))}
      </span>
      {WORKING[stage] ?? "Working"}…
    </motion.div>
  );
}

export function UserBubble({ text }: { text: string }) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={spring} className="flex justify-end">
      <div className="body-large max-w-[85%] rounded-[18px] rounded-tr-[6px] bg-accent px-4 py-2.5 text-accent-ink sm:max-w-[75%]">{text}</div>
    </motion.div>
  );
}
