# LexAgent v2.0 | Phase 1 | corpus_schema.py
"""Data models for LexAgent v2.0 legal corpus.

Defines the canonical document and citation record schemas used throughout
the DVA+ pipeline: retrieval, debate agents, CCE, and Adaptive Memory.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class CorpusDocument:
    """A single document in the LexAgent legal corpus.
    
    Used as the canonical document representation across all pipeline stages:
    - Hybrid Retrieval returns List[CorpusDocument]
    - Prosecutor/Defense cite CorpusDocuments
    - CCE verifies citations against CorpusDocuments
    - Adaptive Memory caches CorpusDocument references
    
    Attributes:
        doc_id: Unique identifier (format: "{source}_{index}").
        text: Full document text (opinion, holding, or question).
        source: Origin dataset — "SCOTUS" | "CaseHOLD" | "LegalBench".
        case_name: Extracted case name if available, else empty string.
        year: Year of the case/opinion. 0 if unknown.
        court: Court name if available (e.g., "Supreme Court of the United States").
        metadata: Additional fields for extensibility.
    """
    doc_id: str
    text: str
    source: str            # "SCOTUS" | "CaseHOLD" | "LegalBench"
    case_name: str = ""    # extracted case name if available
    year: int = 0          # 0 if unknown
    court: str = ""        # court name if available
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorpusDocument':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class CitationRecord:
    """A citation extracted from a debate agent's response.
    
    Used by the Citation Confidence Engine (CCE) to compute CCS scores.
    # NOVEL — CitationRecord carries self_confidence for CCE scoring.
    
    Attributes:
        case_name: Cited case name (e.g., "Smith v. Jones").
        court: Court where the case was decided.
        year: Year of the decision.
        holding: The claimed holding or ruling.
        self_confidence: LLM's own confidence in this citation (0.0-1.0).
            Used by CCE as one signal in the CCS computation.
        agent_source: Which debate agent produced this citation:
            "prosecutor" | "defense".
    """
    case_name: str
    court: str
    year: int
    holding: str
    self_confidence: float     # LLM's own confidence 0-1 (used by CCE)
    agent_source: str          # "prosecutor" | "defense"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CitationRecord':
        """Deserialize from dictionary."""
        return cls(**data)
