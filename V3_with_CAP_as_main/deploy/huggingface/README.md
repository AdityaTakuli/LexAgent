---
title: LexAgent
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Multi-agent legal debate with verified citations
---

# LexAgent API

Backend for <https://lexagent.velocitytechnologies.in>: hybrid retrieval over U.S. Reports,
a prosecutor/defense debate, a citation checker and a judge. The language model is served by
FreeLLMAPI; this Space runs everything else on CPU.

Secret to set in the Space settings: `LEXAGENT_LLM_API_KEY` (the FreeLLMAPI unified key).

A research prototype. Its answers are not legal advice.
