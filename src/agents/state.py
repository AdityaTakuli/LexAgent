# LexAgent v3.0 | state.py
"""LangGraph TypedDict state definition for LexAgent v3.0 DVA+ Protocol."""

from typing import TypedDict, List, Optional, Any
from src.data.corpus_schema import CorpusDocument, CitationRecord


class LexAgentState(TypedDict):
    """Complete state schema for LexAgent v3.0 DVA+ pipeline."""

    # ── LLM PIPELINE ──
    llm:                    Any

    # ── INPUT ──
    query:                  str
    jurisdiction:           str

    # ── RETRIEVAL ──
    retrieved_passages:     List[CorpusDocument]
    entity_map:             dict

    # ── DEBATE ROUND TRACKING ──
    debate_round:           int
    debate_history:         List[dict]

    # ── PROSECUTOR ──
    prosecutor_argument:    str
    prosecutor_citations:   List[CitationRecord]

    # ── DEFENSE ──
    defense_argument:       str
    defense_citations:      List[CitationRecord]
    defense_challenges:     List[str]
    defense_concede:        bool
    challenge_strength:     float

    # ── REFLECTION AGENT ──
    reflection_gaps:        List[str]
    reflection_queries:     List[str]
    reflection_score:       float

    # ── CCE OUTPUT ──
    citation_confidence_report: dict
    avg_ccs:                float
    high_risk_citations:    List[str]

    # ── DDC DECISION ──
    ddc_decision:           str
    ddc_reason:             str

    # ── JUDGE ──
    judge_verdict:          str
    judge_confidence:       float
    judge_verified_citations:   List[CitationRecord]
    judge_uncertain_citations:  List[CitationRecord]
    judge_reasoning:        str

    # ── ADAPTIVE MEMORY ──
    memory_cache_hit:       bool
    cached_answer:          Optional[str]

    # ── FINAL OUTPUT ──
    final_answer:           str
    structured_output:      dict
