// Types and calls for the Python server (ui/server.py).

export type Tier = "SUPPORTED" | "UNCERTAIN" | "MISATTRIBUTED" | "FABRICATED" | "UNVERIFIABLE";

export interface Passage {
  doc_id: string;
  case_name: string;
  cite: string;
  year: number;
  court: string;
  text: string;
  score: number | null;
}

export interface Citation {
  case_name: string;
  court?: string;
  year?: number;
  holding?: string;
  self_confidence?: number;
  agent_source?: string;
  reporter_cite?: string;
}

export interface CitationCheck {
  key: string;
  agent: string;
  case_name: string;
  reporter_cite: string;
  holding: string;
  tier: Tier;
  ccs: number | null;
  existence: number | null;
  support: number | null;
  evidence: string;
  evidence_source: string;
  cite_check: string;
  best_match: string;
}

export type StageName = "loading" | "memory_check" | "retrieve" | "prosecutor" | "defense" | "cce" | "reflection" | "ddc" | "judge" | "memory_store";

export type DebateEvent =
  | { type: "stage"; stage: StageName; round?: number; message?: string; t?: number }
  | { type: "memory"; hit: boolean; answer?: string; t?: number }
  | { type: "retrieval"; round: number; passages: Passage[]; t?: number }
  | { type: "prosecutor"; round: number; argument: string; citations: Citation[]; t?: number }
  | {
      type: "defense";
      round: number;
      argument: string;
      citations: Citation[];
      challenges: string[];
      concede: boolean;
      challenge_strength: number;
      evidence_overlap_ratio: number;
      counter_passages: Passage[];
      t?: number;
    }
  | { type: "cce"; round: number; avg_ccs: number; checks: CitationCheck[]; t?: number }
  | { type: "reflection"; round: number; gaps: string[]; queries: string[]; score: number; t?: number }
  | { type: "ddc"; round: number; decision: string; reason: string; t?: number }
  | {
      type: "judge";
      verdict: string;
      confidence: number;
      reasoning: string;
      final_answer: string;
      verified: Citation[];
      uncertain: Citation[];
      rejected: unknown[];
      t?: number;
    }
  | { type: "done"; rounds: number; seconds: number; t?: number }
  | { type: "error"; message: string; t?: number };

export interface EngineStatus {
  state: "cold" | "loading" | "ready" | "error";
  message: string;
  loaded_seconds: number | null;
  busy: boolean;
}

export interface Status {
  engine: EngineStatus;
  gpu: { name: string; memory_total_mb: number; memory_used_mb: number; utilization: number } | null;
  corpus: { chunks: number; volumes_indexed: number; reports: number };
  sample_transcript: boolean;
}

export interface RunSummary {
  cases: number;
  hr: number | null;
  cfr: number | null;
  oor: number | null;
  mar: number | null;
  ucr: number | null;
  ass: number | null;
  ar: number | null;
  ap: number | null;
  haa: number | null;
  judge_confidence: number | null;
  debate_rounds: number | null;
  total_cites: number | null;
  converged: number;
}

export interface AblationRow {
  variant: string;
  description: string;
  total_cases: number;
  completed_cases: number;
  crashed_cases: number;
  mean_hr: number;
  mean_cas: number;
  mean_confidence: number;
}

export type VerdictCounts = { total: number; accepted: number; rejected: number } & Record<Tier, number>;

export interface StressSummary {
  total_evaluated: number;
  power: number;
  false_acceptance_rate_misattributed: number;
  false_acceptance_rate_casehold: number;
  fabrication_catch_rate: number;
  false_fabrication_rate: number;
  elapsed_seconds: number;
  per_type_counts: Record<string, VerdictCounts>;
}

export interface CorpusStats {
  chunks: number;
  reports: number;
  opinions: number;
  order_entries: number;
  volumes_target: number;
  volumes_indexed: number;
  volumes_from_cap: number;
  volumes_from_official_pdf: number;
  volumes_unavailable: string[];
  built_at: string | null;
  stages: { name: string; detail: string; counts?: Record<string, number>; indexed?: number; total?: number; updated_at?: string | null }[];
  models: { generator: string; embedder: string; nli: string; tau: number; max_rounds: number };
}

export interface Overview {
  benchmark: RunSummary | null;
  benchmark_seconds: number | null;
  baseline: RunSummary | null;
  ablation: AblationRow[] | null;
  stress: StressSummary | null;
  stress_before_fix: StressSummary | null;
  test_set: { total: number; evaluable: number; excluded: { id: number; target_case: string; topic: string; reason: string }[] };
  corpus: CorpusStats;
}

export interface CaseMetrics {
  hr: number | null;
  cfr: number | null;
  oor: number | null;
  mar: number | null;
  ucr: number | null;
  ass: number | null;
  ar: number | null;
  ap: number | null;
  haa: number | null;
  dcr_status: string | null;
  judge_confidence: number | null;
  debate_rounds: number | null;
  total_cites: number | null;
  fake_cites: number | null;
  fake_names: string[] | null;
  status: string | null;
}

export interface CaseRow {
  id: number;
  topic: string;
  query: string;
  target_case: string;
  reporter_citation: string;
  ground_truth_holding: string;
  lexagent: CaseMetrics | null;
  baseline: CaseMetrics | null;
}

export interface Transcript {
  query: string;
  recorded_at: string;
  events: DebateEvent[];
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json() as Promise<T>;
}

export const api = {
  status: () => getJson<Status>("/api/status"),
  overview: () => getJson<Overview>("/api/overview"),
  cases: () => getJson<{ cases: CaseRow[] }>("/api/cases"),
  suggestions: () => getJson<{ suggestions: { topic: string; query: string }[] }>("/api/suggestions"),
  sampleTranscript: () => getJson<Transcript>("/api/transcripts/sample"),
  loadEngine: () => fetch("/api/engine/load", { method: "POST" }),
};

/** Ask a question and receive the debate stage by stage (Server-Sent Events over a POST). */
export async function streamDebate(query: string, onEvent: (event: DebateEvent) => void, signal?: AbortSignal): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });
  if (!response.ok || !response.body) {
    let detail = `The server answered ${response.status}.`;
    try {
      detail = ((await response.json()) as { detail?: string }).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of block.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as DebateEvent);
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}

/** Replay a recorded debate with its original pacing, compressed so no pause exceeds maxGapMs. */
export function replayTranscript(transcript: Transcript, onEvent: (event: DebateEvent) => void, maxGapMs = 1400): () => void {
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let index = 0;
  const step = () => {
    if (cancelled || index >= transcript.events.length) return;
    const event = transcript.events[index];
    onEvent(event);
    index += 1;
    if (index < transcript.events.length) {
      const gap = ((transcript.events[index].t ?? 0) - (event.t ?? 0)) * 1000;
      timer = setTimeout(step, Math.max(250, Math.min(maxGapMs, gap)));
    }
  };
  timer = setTimeout(step, 300);
  return () => {
    cancelled = true;
    if (timer) clearTimeout(timer);
  };
}
