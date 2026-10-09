import { useGSAP } from "@gsap/react";
import { AnimatePresence, motion } from "framer-motion";
import gsap from "gsap";
import { useCallback, useEffect, useRef, useState } from "react";
import { applyEvent, DebateThread, newDebate, UserBubble, type Debate } from "../components/debate";
import { Button, Icon, IconButton, useStickToBottom } from "../components/ui";
import { api, replayTranscript, streamDebate, type DebateEvent, type Status } from "../lib/api";
import { compact } from "../lib/format";
import { M3_EASE, prefersReducedMotion, useRipple } from "../lib/motion";

interface Turn {
  id: number;
  question: string;
  debate: Debate;
}

const HEADLINE = "Ask a question of U.S. Supreme Court law";

function SuggestionCard({ topic, query, index, onAsk }: { topic: string; query: string; index: number; onAsk: (q: string) => void }) {
  const ripple = useRipple();
  return (
    <motion.button
      type="button"
      onClick={() => onAsk(query)}
      onPointerDown={ripple.onPointerDown}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.55 + index * 0.07, duration: 0.4, ease: M3_EASE.emphasizedDecelerate }}
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.98 }}
      className="state-layer card-outlined cursor-pointer p-4 text-left text-ink"
    >
      {ripple.layer}
      <div className="label-medium flex items-center gap-1.5 text-accent">
        <Icon name="lightbulb" size={16} />
        {topic}
      </div>
      <div className="body-medium mt-1.5 line-clamp-3 text-ink">{query}</div>
    </motion.button>
  );
}

function Welcome({ status, suggestions, onAsk, onReplay }: { status: Status | null; suggestions: { topic: string; query: string }[]; onAsk: (q: string) => void; onReplay: () => void }) {
  const scope = useRef<HTMLDivElement>(null);
  useGSAP(
    () => {
      if (prefersReducedMotion()) return;
      gsap.from(".hero-word", { yPercent: 110, opacity: 0, duration: 0.7, ease: "power4.out", stagger: 0.045 });
      gsap.from(".hero-mark", { scale: 0.6, opacity: 0, rotate: -12, duration: 0.8, ease: "back.out(1.8)" });
      gsap.from(".hero-fade", { opacity: 0, y: 10, duration: 0.6, delay: 0.45, stagger: 0.08, ease: "power2.out" });
    },
    { scope },
  );

  const engine = status?.engine;
  const gpu = status?.gpu;
  const gpuBusy = engine?.state === "cold" && gpu ? gpu.memory_used_mb / gpu.memory_total_mb > 0.5 : false;

  return (
    <div ref={scope} className="relative flex min-h-full flex-col items-center justify-center px-4 py-10">
      <div className="hero-grid pointer-events-none absolute inset-0" aria-hidden />
      <div className="relative w-full max-w-3xl text-center">
        <div className="hero-mark elevation-2 mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-[20px] bg-accent-wash text-on-accent-wash">
          <Icon name="balance" size={32} fill />
        </div>
        <h1 className="display-small text-ink">
          {HEADLINE.split(" ").map((word, index) => (
            <span key={index} className="inline-block overflow-hidden pb-1 align-bottom">
              <span className="hero-word inline-block">{word}&nbsp;</span>
            </span>
          ))}
        </h1>
        <p className="hero-fade body-large mx-auto mt-3 max-w-xl text-ink-2">
          A prosecutor and a defense argue from retrieved opinions, every citation is checked against the opinion it names, and a judge gives the verdict.
          {status ? ` The corpus holds ${compact(status.corpus.chunks)} passages from ${status.corpus.volumes_indexed} volumes of the U.S. Reports.` : ""}
        </p>

        {gpuBusy && gpu && (
          <div className="hero-fade body-medium mx-auto mt-5 flex max-w-xl items-start gap-3 rounded-2xl bg-tertiary-container px-4 py-3 text-left text-on-tertiary-container">
            <Icon name="memory" size={20} className="mt-0.5" />
            <span>
              The GPU is busy with another run ({(gpu.memory_used_mb / 1024).toFixed(0)} of {(gpu.memory_total_mb / 1024).toFixed(0)} GB in use). A live question needs it; wait for that run to finish, or replay a recorded debate below.
            </span>
          </div>
        )}

        <div className="mt-8 grid gap-3 text-left sm:grid-cols-2">
          {suggestions.slice(0, 4).map((item, index) => (
            <SuggestionCard key={item.query} topic={item.topic} query={item.query} index={index} onAsk={onAsk} />
          ))}
        </div>

        {status?.sample_transcript && (
          <div className="hero-fade mt-6">
            <Button variant="tonal" icon="play_arrow" onClick={onReplay}>
              Replay a recorded debate
            </Button>
          </div>
        )}
        <p className="hero-fade body-small mt-4 text-muted">
          {engine?.state === "ready" ? "The model is loaded." : engine?.state === "loading" ? "The model is loading…" : "The model loads on the first question, which takes about two minutes."}
        </p>
      </div>
    </div>
  );
}

export function ChatPage({ status, maxRounds, autoReplay }: { status: Status | null; maxRounds: number; autoReplay: boolean }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [suggestions, setSuggestions] = useState<{ topic: string; query: string }[]>([]);
  const { ref: scroller, onScroll, stick } = useStickToBottom<HTMLDivElement>();
  const nextId = useRef(1);
  const cancelReplay = useRef<(() => void) | null>(null);

  useEffect(() => {
    api.suggestions().then((data) => setSuggestions(data.suggestions)).catch(() => setSuggestions([]));
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
        await streamDebate(text, (event) => update(id, event));
      } catch (error) {
        update(id, { type: "error", message: error instanceof Error ? error.message : String(error) });
      } finally {
        setRunning(false);
      }
    },
    [running, update],
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
    if (autoReplay && !replayed.current && status?.sample_transcript) {
      replayed.current = true;
      void replay();
    }
  }, [autoReplay, status, replay]);

  return (
    <div className="flex h-full flex-col">
      <div ref={scroller} onScroll={onScroll} className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <AnimatePresence mode="wait">
          {turns.length === 0 ? (
            <motion.div key="welcome" exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.2, ease: M3_EASE.emphasizedAccelerate }} className="h-full">
              <Welcome status={status} suggestions={suggestions} onAsk={ask} onReplay={replay} />
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
        <form
          className="mx-auto flex w-full max-w-3xl items-end gap-2 rounded-[28px] bg-surface-2 py-2 pl-5 pr-2 focus-within:outline-2 focus-within:outline-accent"
          onSubmit={(event) => {
            event.preventDefault();
            void ask(input);
          }}
        >
          <label htmlFor="question" className="sr-only">
            Your legal question
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
            placeholder={running ? "The debate is in progress…" : "Ask about a constitutional or federal question"}
            disabled={running}
            className="body-large max-h-40 min-h-[40px] flex-1 resize-none bg-transparent py-2 text-ink outline-none placeholder:text-ink-2 disabled:opacity-60"
          />
          <IconButton type="submit" icon="arrow_upward" label="Send question" variant="filled" disabled={running || !input.trim()} />
        </form>
        <p className="body-small mx-auto mt-2 max-w-3xl text-center text-muted">A research prototype. Its answers are not legal advice; check every citation against the source.</p>
      </div>
    </div>
  );
}
