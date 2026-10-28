import { useGSAP } from "@gsap/react";
import { AnimatePresence, motion } from "framer-motion";
import gsap from "gsap";
import { useCallback, useEffect, useRef, useState } from "react";
import { applyEvent, DebateThread, newDebate, UserBubble, type Debate } from "../components/debate";
import { Button, Icon, IconButton, useStickToBottom } from "../components/ui";
import { api, modelLabel, replayTranscript, streamDebate, type DebateEvent, type Status } from "../lib/api";
import { compact } from "../lib/format";
import { M3_EASE, prefersReducedMotion } from "../lib/motion";

interface Turn {
  id: number;
  question: string;
  debate: Debate;
}

const HEADLINE = "Ask a question of Supreme Court law.";
const MODEL_KEY = "lexagent.model";

function readModel(): string {
  try {
    return localStorage.getItem(MODEL_KEY) ?? "";
  } catch {
    return "";
  }
}

/** Which language model writes the arguments and the ruling. Search and citation checks stay the same. */
function ModelSwitch({ options, value, onChange, disabled }: { options: string[]; value: string; onChange: (model: string) => void; disabled: boolean }) {
  return (
    <div className="mx-auto mb-2 flex w-full max-w-3xl items-center gap-2" role="radiogroup" aria-label="Language model">
      <span className="label-medium text-ink-2">Model</span>
      <div className="flex rounded-full border border-line bg-surface p-0.5">
        {options.map((option) => {
          const selected = option === value;
          return (
            <button
              key={option}
              type="button"
              role="radio"
              aria-checked={selected}
              disabled={disabled}
              onClick={() => onChange(option)}
              className={`label-medium h-8 rounded-full px-3.5 transition-colors disabled:cursor-not-allowed ${selected ? "bg-inverse-surface text-inverse-on-surface" : "text-ink-2 hover:text-ink disabled:opacity-60"}`}
            >
              {modelLabel(option)}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function SuggestionCard({ topic, query, index, onAsk, disabled }: { topic: string; query: string; index: number; onAsk: (q: string) => void; disabled: boolean }) {
  return (
    <motion.button
      type="button"
      onClick={() => onAsk(query)}
      disabled={disabled}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.35 + index * 0.05, duration: 0.35, ease: M3_EASE.emphasizedDecelerate }}
      className="state-layer group card-outlined cursor-pointer p-4 text-left text-ink transition-shadow hover:shadow-[var(--shadow-2)] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:shadow-none"
    >
      <div className="label-medium text-ink-2">{topic}</div>
      <div className="body-medium mt-1.5 line-clamp-3 text-ink">{query}</div>
      <div className="label-medium mt-3 flex items-center gap-1 text-ink-2 transition-colors group-hover:text-ink">
        Ask this <Icon name="arrow_forward" size={14} />
      </div>
    </motion.button>
  );
}

function Welcome({ status, offline, suggestions, onAsk, onReplay }: { status: Status | null; offline: boolean; suggestions: { topic: string; query: string }[]; onAsk: (q: string) => void; onReplay: () => void }) {
  const scope = useRef<HTMLDivElement>(null);
  useGSAP(
    () => {
      if (prefersReducedMotion()) return;
      gsap.from(".hero-word", { yPercent: 100, opacity: 0, duration: 0.6, ease: "power3.out", stagger: 0.035 });
      gsap.from(".hero-fade", { opacity: 0, y: 8, duration: 0.5, delay: 0.3, stagger: 0.06, ease: "power2.out" });
    },
    { scope },
  );

  const engine = status?.engine;
  return (
    <div ref={scope} className="flex min-h-full flex-col items-center justify-center px-5 py-12">
      <div className="w-full max-w-3xl">
        <p className="hero-fade eyebrow text-ink-2">U.S. Supreme Court · {status ? `${compact(status.corpus.chunks)} passages, ${status.corpus.volumes_indexed} volumes` : "U.S. Reports"}</p>
        <h1 className="display-small mt-3 text-ink">
          {HEADLINE.split(" ").map((word, index) => (
            <span key={index} className="inline-block overflow-hidden pb-1 align-bottom">
              <span className="hero-word inline-block">{word}&nbsp;</span>
            </span>
          ))}
        </h1>
        <p className="hero-fade body-large mt-3 max-w-2xl text-ink-2">
          A <span className="font-medium text-prosecutor">prosecutor</span> and a <span className="font-medium text-defense">defense</span> argue from the opinions, every case they cite is checked against the opinion it names, and a judge rules on what survives.
        </p>

        {offline && (
          <div className="hero-fade body-medium mt-6 flex items-start gap-3 rounded-2xl border border-line bg-surface px-4 py-3 text-ink-2" role="status">
            <Icon name="visibility_off" size={18} className="mt-0.5 text-muted" />
            <span>
              <span className="font-medium text-ink">Live questions are paused.</span> The LexAgent backend is offline right now. You can still watch a recorded debate.
            </span>
          </div>
        )}

        <div className="mt-8 grid gap-3 sm:grid-cols-2">
          {suggestions.slice(0, 4).map((item, index) => (
            <SuggestionCard key={item.query} topic={item.topic} query={item.query} index={index} onAsk={onAsk} disabled={offline} />
          ))}
        </div>

        <div className="hero-fade mt-6 flex flex-wrap items-center gap-x-4 gap-y-2">
          <Button variant={offline ? "filled" : "tonal"} icon="play_arrow" onClick={onReplay}>
            Watch a recorded debate
          </Button>
          {!offline && (
            <span className="body-small text-muted">
              {engine?.state === "ready" ? "Ready. A debate takes about a minute." : engine?.state === "loading" ? "Loading the models…" : "The first question also loads the models, so it takes a little longer."}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

const FALLBACK_SUGGESTIONS = [
  { topic: "Fourth Amendment", query: "May law enforcement search the digital contents of a cell phone seized from an arrestee without a warrant?" },
  { topic: "Fifth Amendment", query: "Are statements made during custodial interrogation admissible if the suspect was not warned of the right to remain silent?" },
  { topic: "Sixth Amendment", query: "Must a state provide a lawyer to a defendant charged with a felony who cannot afford one?" },
  { topic: "First Amendment", query: "What must a public official prove to recover damages for a defamatory falsehood about official conduct?" },
];

export function ChatPage({ status, offline, maxRounds, autoReplay }: { status: Status | null; offline: boolean; maxRounds: number; autoReplay: boolean }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [chosenModel, setChosenModel] = useState(readModel);
  const modelOptions = status?.models?.options ?? [];
  const model = modelOptions.includes(chosenModel) ? chosenModel : (status?.models?.default ?? "");
  const chooseModel = useCallback((next: string) => {
    setChosenModel(next);
    try {
      localStorage.setItem(MODEL_KEY, next);
    } catch {
      /* the choice just isn't remembered */
    }
  }, []);
  const [suggestions, setSuggestions] = useState<{ topic: string; query: string }[]>([]);
  const { ref: scroller, onScroll, stick } = useStickToBottom<HTMLDivElement>();
  const nextId = useRef(1);
  const cancelReplay = useRef<(() => void) | null>(null);

  useEffect(() => {
    api.suggestions().then((data) => setSuggestions(data.suggestions.length ? data.suggestions : FALLBACK_SUGGESTIONS)).catch(() => setSuggestions(FALLBACK_SUGGESTIONS));
    return () => cancelReplay.current?.();
  }, []);

  useEffect(() => {
    stick();
  }, [turns, stick]);

  const update = useCallback((id: number, event: DebateEvent) => {
    setTurns((current) => current.map((turn) => (turn.id === id ? { ...turn, debate: applyEvent(turn.debate, event) } : turn)));
    if (event.type === "done" || event.type === "error") setRunning(false);
  }, []);

  const ask = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || running) return;
      const id = nextId.current++;
      setTurns((current) => [...current, { id, question: text, debate: newDebate() }]);
      setInput("");
      setRunning(true);
      try {
        await streamDebate(text, (event) => update(id, event), undefined, model || undefined);
      } catch (error) {
        update(id, { type: "error", message: error instanceof Error ? error.message : String(error) });
      } finally {
        setRunning(false);
      }
    },
    [running, update, model],
  );

  const replay = useCallback(async () => {
    if (running) return;
    try {
      const transcript = await api.sampleTranscript();
      const id = nextId.current++;
      setTurns((current) => [...current, { id, question: transcript.query, debate: newDebate(true) }]);
      setRunning(true);
      cancelReplay.current = replayTranscript(transcript, (event) => update(id, event));
    } catch {
      /* no recorded debate available */
    }
  }, [running, update]);

  const replayed = useRef(false);
  useEffect(() => {
    if (autoReplay && !replayed.current) {
      replayed.current = true;
      void replay();
    }
  }, [autoReplay, replay]);

  return (
    <div className="flex h-full flex-col">
      <div ref={scroller} onScroll={onScroll} className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <AnimatePresence mode="wait">
          {turns.length === 0 ? (
            <motion.div key="welcome" exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.2, ease: M3_EASE.emphasizedAccelerate }} className="h-full">
              <Welcome status={status} offline={offline} suggestions={suggestions} onAsk={ask} onReplay={replay} />
            </motion.div>
          ) : (
            <motion.div key="thread" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-6">
              {turns.map((turn) => (
                <div key={turn.id} className="flex flex-col gap-5">
                  <UserBubble text={turn.question} />
                  <DebateThread debate={turn.debate} maxRounds={maxRounds} />
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="px-4 pb-4 pt-2">
        {modelOptions.length > 1 && !offline && <ModelSwitch options={modelOptions} value={model} onChange={chooseModel} disabled={running} />}
        <form
          className="glass mx-auto flex w-full max-w-3xl items-end gap-2 rounded-[22px] py-1.5 pl-4 pr-1.5 shadow-[var(--shadow-2)] focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[var(--focus)]"
          onSubmit={(event) => {
            event.preventDefault();
            void ask(input);
          }}
        >
          <label htmlFor="question" className="sr-only">
            Legal question
          </label>
          <textarea
            id="question"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void ask(input);
              }
            }}
            rows={1}
            placeholder={offline ? "Live questions are paused while the backend is offline" : running ? "The debate is in progress…" : "Ask a constitutional or federal question"}
            disabled={running || offline}
            className="body-large max-h-40 min-h-[44px] flex-1 resize-none bg-transparent py-2.5 text-ink outline-none placeholder:text-muted disabled:cursor-not-allowed"
          />
          <IconButton type="submit" icon="arrow_upward" label="Ask" variant="filled" disabled={running || offline || !input.trim()} className="!h-10 !w-10" />
        </form>
        <p className="body-small mx-auto mt-2 max-w-3xl text-center text-muted">A research prototype. Answers are AI-generated and are not legal advice; check every citation against the source.</p>
      </div>
    </div>
  );
}
