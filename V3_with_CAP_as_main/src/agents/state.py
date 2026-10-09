# LexAgent v3.0 | state.py
"""LangGraph TypedDict state definition for LexAgent v3.0 DVA+ Protocol."""

from typing import TypedDict, List, Optional, Any
from src.data.corpus_schema import CorpusDocument, CitationRecord


class LexAgentState(TypedDict, total=False):
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

    # ── DEFENSE (Novelty 3: Evidence-Disjoint Defense) ──
    defense_argument:         str
    defense_citations:        List[CitationRecord]
    defense_challenges:       List[str]
    defense_concede:          bool
    challenge_strength:       float
    defense_counter_passages: List[CorpusDocument]
    evidence_overlap_ratio:   float

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

    # ── ABLATION SWITCHES ──
    # LangGraph drops any input key this schema does not name, so a switch left out of here
    # never reaches the nodes and they fall back to "component is on".
    use_defense:            bool
    use_cce:                bool
    use_memory:             bool

    # ── DEBATE PROTOCOL ("v1" or "ledger", see config.DEBATE_PROTOCOL) ──
    protocol:               str
    # Ledger protocol only. Best verdict and evidence for every citation of every round, keyed by
    # the report the citation resolves to (or by its cleaned name when it resolves to none).
    citation_ledger:        dict
    seen_passage_ids:       List[str]
    round_new_passages:     int
    round_new_supported:    int


def debate_protocol(state: dict) -> str:
    """The protocol a debate runs under: the state's own setting, else the configured default."""
    from config import DEBATE_PROTOCOL, DEBATE_PROTOCOLS

    protocol = state.get("protocol") or DEBATE_PROTOCOL
    if protocol not in DEBATE_PROTOCOLS:
        raise ValueError(f"unknown debate protocol {protocol!r}; expected one of {DEBATE_PROTOCOLS}")
    return protocol
