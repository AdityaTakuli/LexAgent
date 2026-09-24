# LexAgent v3.0 | cce.py
"""
Citation Confidence Engine (CCE) — Grounded in CAP Primary Authority.
Produces a continuous Citation Confidence Score (CCS) ∈ [0, 1] for every citation
made during multi-agent debate.

Verification Tiers:
  Tier 1 — Exact match against CAP canonical case names and reporter citations
  Tier 2 — Fuzzy string match (Levenshtein token set ratio)
  Tier 3 — Semantic holding similarity (Legal-BERT cosine)
"""

import os
import sys
import pickle
import re

try:
    from fuzzywuzzy import fuzz
except ImportError:
    try:
        from thefuzz import fuzz
    except ImportError:
        import difflib
        class FuzzFallback:
            @staticmethod
            def ratio(s1, s2):
                return int(difflib.SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio() * 100)
            @staticmethod
            def token_set_ratio(s1, s2):
                return FuzzFallback.ratio(s1, s2)
        fuzz = FuzzFallback()

try:
    from sentence_transformers import util
except ImportError:
    util = None

from config import (
    CCS_EXACT_WEIGHT, CCS_FUZZY_WEIGHT, CCS_SEMANTIC_WEIGHT,
    CCS_VERIFIED_THRESHOLD, CCS_UNCERTAIN_THRESHOLD,
    CCE_HOLDING_SEM_THRESHOLD, CCE_HOLDING_FALLBACK_SCORE,
    CCE_FUZZY_THRESHOLD_RATIO,
)
from src.agents.state import LexAgentState
from src.data.corpus_schema import CitationRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CitationConfidenceEngine:
    """Continuous Citation Confidence Scoring with CAP Authority Catalog."""

    def __init__(self, chroma_collection, embedder, known_cases_path: str):
        self.collection = chroma_collection
        self.embedder = embedder
        self.known_cases: set = self._load_known_cases(known_cases_path)
        logger.info("CCE initialized: %d known cases/citations loaded from %s",
                    len(self.known_cases), known_cases_path)

    def _load_known_cases(self, path: str) -> set:
        try:
            with open(path, "rb") as f:
                cases = pickle.load(f)
            logger.info("Loaded %d known cases from %s", len(cases), path)
            return cases
        except Exception as e:
            logger.warning("Could not load known cases from %s: %s", path, e)
            return set()

    def compute_ccs(self, citation: CitationRecord) -> dict:
        case_name = citation.case_name.strip().lower()
        reporter_cite = getattr(citation, "reporter_cite", "").strip().lower()

        # ─── Tier 1: Exact match against CAP catalog ───
        exact_score = 0.0
        exact_evidence = ""
        normalized_corpus = {c.lower().strip() for c in self.known_cases}

        # Check name or reporter citation in CAP normalized catalog
        matched_catalog = (case_name in normalized_corpus) or (reporter_cite and reporter_cite in normalized_corpus)

        if matched_catalog:
            exact_score = 1.0
            try:
                chroma_result = self.collection.query(
                    query_embeddings=[self.embedder.encode(citation.case_name).tolist()],
                    n_results=1,
                    where={"case_name": {"$eq": citation.case_name}}
                )
                if chroma_result.get("documents") and chroma_result["documents"][0]:
                    retrieved_text = chroma_result["documents"][0][0]
                    if util and citation.holding:
                        holding_vec = self.embedder.encode(citation.holding, convert_to_tensor=True)
                        doc_vec = self.embedder.encode(retrieved_text[:500], convert_to_tensor=True)
                        sem_sim = float(util.cos_sim(holding_vec, doc_vec))
                        exact_score = 1.0 if sem_sim > CCE_HOLDING_SEM_THRESHOLD else CCE_HOLDING_FALLBACK_SCORE
                    exact_evidence = retrieved_text[:200]
            except Exception as e:
                logger.debug("Exact match ChromaDB lookup fallback: %s", e)
                exact_score = 0.90

        # ─── Tier 2: Fuzzy match with token-set overlap ──
        fuzzy_score = 0.0
        best_match = ""
        c_tokens = set(case_name.replace(',', ' ').replace('.', ' ').split()) - {'v', 'the', 'of', 'in', 're', 'state'}

        for real_case in self.known_cases:
            rc_lower = real_case.lower().strip()
            r1 = fuzz.ratio(case_name, rc_lower) / 100.0
            r2 = (fuzz.token_set_ratio(case_name, rc_lower) / 100.0) if hasattr(fuzz, "token_set_ratio") else 0.0
            ratio = max(r1, r2)
            if ratio < 0.85 and c_tokens:
                rc_tokens = set(rc_lower.replace(',', ' ').replace('.', ' ').split()) - {'v', 'the', 'of', 'in', 're', 'state'}
                if c_tokens and c_tokens.issubset(rc_tokens):
                    ratio = 0.95
            if ratio > fuzzy_score:
                fuzzy_score = ratio
                best_match = real_case

        fuzzy_threshold = CCS_VERIFIED_THRESHOLD * CCE_FUZZY_THRESHOLD_RATIO
        fuzzy_score = fuzzy_score if fuzzy_score >= fuzzy_threshold else 0.0

        if fuzzy_score >= 0.90 and exact_score == 0.0:
            exact_score = 0.85

        # ─── Tier 3: Semantic similarity ────────────────
        semantic_score = 0.0
        if citation.holding and util:
            try:
                results = self.collection.query(
                    query_embeddings=[self.embedder.encode(citation.case_name + " " + citation.holding).tolist()],
                    n_results=1,
                )
                if results.get("documents") and results["documents"][0]:
                    retrieved = results["documents"][0][0]
                    h_vec = self.embedder.encode(citation.holding, convert_to_tensor=True)
                    r_vec = self.embedder.encode(retrieved[:500], convert_to_tensor=True)
                    semantic_score = float(util.cos_sim(h_vec, r_vec))
            except Exception as e:
                logger.debug("Semantic similarity lookup failed: %s", e)

        # ─── CCS Weighted Aggregation ───────────────────
        ccs = (CCS_EXACT_WEIGHT    * exact_score +
               CCS_FUZZY_WEIGHT    * fuzzy_score +
               CCS_SEMANTIC_WEIGHT * semantic_score)
        ccs = round(min(max(ccs, 0.0), 1.0), 4)

        # ─── Tier label ─────────────────────────────────
        if ccs >= CCS_VERIFIED_THRESHOLD:
            tier = "VERIFIED"
        elif ccs >= CCS_UNCERTAIN_THRESHOLD:
            tier = "UNCERTAIN"
        else:
            tier = "LIKELY_FAKE"

        return {
            "ccs":            ccs,
            "tier":           tier,
            "exact_score":    round(exact_score, 3),
            "fuzzy_score":    round(fuzzy_score, 3),
            "semantic_score": round(semantic_score, 3),
            "best_match":     best_match,
            "evidence":       exact_evidence,
            "citation":       citation.to_dict(),
        }

    def run_on_state(self, state: LexAgentState) -> tuple:
        all_citations = (state.get("prosecutor_citations", []) +
                         state.get("defense_citations", []))

        report = {}
        ccs_values = []

        for cite in all_citations:
            c_record = cite if isinstance(cite, CitationRecord) else CitationRecord.from_dict(cite)
            key = f"{c_record.agent_source}::{len(report)}::{c_record.case_name}"
            result = self.compute_ccs(c_record)
            report[key] = result
            ccs_values.append(result["ccs"])

        avg_ccs = round(sum(ccs_values) / len(ccs_values), 4) if ccs_values else 0.0

        logger.info(
            "CCE scored %d citations: avg_CCS=%.4f | VERIFIED=%d, UNCERTAIN=%d, LIKELY_FAKE=%d",
            len(ccs_values), avg_ccs,
            sum(1 for v in report.values() if v["tier"] == "VERIFIED"),
            sum(1 for v in report.values() if v["tier"] == "UNCERTAIN"),
            sum(1 for v in report.values() if v["tier"] == "LIKELY_FAKE"),
        )
        return report, avg_ccs


def make_cce_node(cce: CitationConfidenceEngine):
    def cce_node(state: LexAgentState) -> dict:
        report, avg_ccs = cce.run_on_state(state)
        return {
            "citation_confidence_report": report,
            "avg_ccs": avg_ccs,
        }
    return cce_node
