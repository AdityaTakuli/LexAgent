# LexAgent web UI

A chat-style front end for the debate, with the evaluation results and the corpus as charts.

| Page | What it shows |
|---|---|
| **Chat** | Ask a question and watch the debate arrive stage by stage: retrieved authorities, prosecutor, defense, the citation check for every citation, reflection, the controller's decision, and the judge's verdict with a confidence gauge. |
| **Evaluation** | Debate system against the single-agent baseline, the ablation, the judge-confidence distribution and the citation stress test. Every chart has a table view. |
| **Cases** | The benchmark questions with their per-case scores, searchable and sortable. |
| **Corpus** | What is indexed: volumes, sources, opinions against order entries, and the state of the three pipeline stages. |

Built with React, TypeScript, Vite and Tailwind. Components follow Material 3 (colour roles generated
from one seed colour, navigation rail / bar, state layers, type scale). Framer Motion drives the
component transitions; GSAP drives the stage rail, the gauge and the number count-ups.

## Run it

Everything is run from `V3_with_CAP_as_main`.

```powershell
# 1. Build the front end once (needs Node 20 or newer)
cd ui\web
npm install
npm run build
cd ..\..

# 2. Start the server (uses the project's virtual environment)
.venv\Scripts\python.exe ui\server.py --port 8000
```

Open <http://127.0.0.1:8000>.

- The **Evaluation**, **Cases** and **Corpus** pages read the saved files in `data/` and work at once.
- The **Chat** page loads the model stack (Mistral-7B, Legal-BERT, the NLI model, BM25, the corpus)
  on the first question, which takes about a minute, or at start-up with `--preload`.
  It needs the GPU to itself: do not start a chat while a benchmark or an ablation is running.
- `http://127.0.0.1:8000/#/chat?replay=1` replays the recorded debate in `ui/sample_transcript.json`
  without loading any model.
- Add `?theme=dark` or `?theme=light` to force a theme; otherwise the system setting is followed.

The server needs two packages that the research code does not (`fastapi`, `uvicorn`), listed in
`ui/requirements.txt`. Install them into the virtual environment only:

```powershell
.venv\Scripts\python.exe -m pip install -r ui\requirements.txt
```

## Develop

```powershell
.venv\Scripts\python.exe ui\server.py --port 8000     # API
cd ui\web; npm run dev                                 # front end with hot reload, proxies /api to :8000
```

- `ui/server.py` — FastAPI app. `POST /api/chat` streams the LangGraph run as Server-Sent Events,
  one event per node (`retrieval`, `prosecutor`, `defense`, `cce`, `reflection`, `ddc`, `judge`, `done`).
  Each finished debate is saved to `data/results/ui_transcripts/`.
- `ui/web/src/pages/` — the four pages. `ui/web/src/components/debate.tsx` renders a debate from the event stream.
- `ui/web/src/styles.css` — the Material 3 colour roles for light and dark. Regenerate them from a
  different seed colour (run in `ui/web`, then paste the output into `styles.css`):
  `node --import ./scripts/register-loader.mjs scripts/generate-m3-tokens.mjs "#2a78d6"`

The UI is a research prototype. Its answers are not legal advice.
