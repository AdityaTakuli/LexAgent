# LexAgent v3.0 | cce.py
"""
Citation Confidence Engine (CCE) — Grounded in CAP Primary Authority.
Factorized Verification (Novelty 1):
  CCS = Existence (E) × Support (S)

5-Way Verdict Classification:
  1. SUPPORTED:     Case exists in CAP and text entails holding (CCS >= tau)
  2. UNCERTAIN:     Case exists, support is borderline (0.40 <= CCS < tau)
  3. MISATTRIBUTED: Case exists, but text does NOT entail holding (CCS < 0.40)
  4. FABRICATED:    Volume is covered in corpus, but case does not exist at citation
  5. UNVERIFIABLE:  Case not in corpus, and corpus does not cover where it would be
"""

import os
import sys
import pickle
import re
from typing import Optional, Tuple, List, Dict, Any

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
    CCS_VERIFIED_THRESHOLD,
    CCS_UNCERTAIN_THRESHOLD,
    TAU_ACCEPTANCE_THRESHOLD,
    CCE_HOLDING_SEM_THRESHOLD,
    CCE_FUZZY_THRESHOLD_RATIO,
    LANDMARK_CAP_VOLUMES,
)
from src.agents.state import LexAgentState
from src.data.corpus_schema import CitationRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CitationConfidenceEngine:
    """Factorized Citation Confidence Scoring with CAP Authority Catalog."""

    def __init__(
        self,
        chroma_collection,
        embedder,
        known_cases_path: str,
        nli_model=None,
        tau: float = TAU_ACCEPTANCE_THRESHOLD,
    ):
        self.collection = chroma_collection
        self.embedder = embedder
        self.known_cases: set = self._load_known_cases(known_cases_path)
        self.tau = tau if tau is not None else TAU_ACCEPTANCE_THRESHOLD
        self.nli_model = nli_model
        if self.nli_model is None:
            self.nli_model = self._init_nli_model()

        logger.info(
            "CCE (Factorized) initialized: %d known cases, tau=%.2f, NLI active=%s",
            len(self.known_cases),
            self.tau,
            self.nli_model is not None,
        )

    def _init_nli_model(self):
        """Attempts to load a fast NLI cross-encoder model for entailment scoring."""
        try:
            from sentence_transformers import CrossEncoder
            # Lightweight NLI cross-encoder (~0.4 GB)
            model = CrossEncoder("cross-encoder/nli-deberta-v3-small", max_length=512)
            logger.info("Loaded NLI CrossEncoder: cross-encoder/nli-deberta-v3-small")
            return model
        except Exception as e:
            logger.info("NLI CrossEncoder not initialized; using Legal-BERT semantic support fallback: %s", e)
            return None

    def _load_known_cases(self, path: str) -> set:
        try:
            with open(path, "rb") as f:
                cases = pickle.load(f)
            logger.info("Loaded %d known cases from %s", len(cases), path)
            return cases
        except Exception as e:
            logger.warning("Could not load known cases from %s: %s", path, e)
            return set()

    def _is_volume_covered(self, citation: CitationRecord) -> bool:
        """Determines if the cited reporter volume is covered in our authority corpus."""
        text_to_check = f"{citation.case_name} {getattr(citation, 'reporter_cite', '')} {getattr(citation, 'holding', '')}"
        match = re.search(r'(\d+)\s+U\.?\s*S\.?', text_to_check, re.IGNORECASE)
        if match:
            vol = str(match.group(1))
            if vol in LANDMARK_CAP_VOLUMES:
                return True
        if hasattr(citation, "volume") and str(citation.volume) in LANDMARK_CAP_VOLUMES:
            return True
        return False

    def _compute_existence(self, citation: CitationRecord) -> Tuple[float, str, str]:
        """
        Existence (E) check: Verifies if the case exists in the CAP catalog.
        Returns: (E_score, matched_canonical_name, match_type)
        """
        case_name = citation.case_name.strip().lower()
        reporter_cite = getattr(citation, "reporter_cite", "").strip().lower()
        normalized_corpus = {c.lower().strip() for c in self.known_cases if c}

        # 1. Exact canonical catalog match
        exact_match = (case_name in normalized_corpus) or (reporter_cite and reporter_cite in normalized_corpus)
        if exact_match:
            return 1.0, case_name, "EXACT"

        # 2. Fuzzy string and token-set matching
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
        if fuzzy_score >= fuzzy_threshold:
            return round(fuzzy_score, 4), best_match, "FUZZY"

        return 0.0, best_match, "NOT_FOUND"

    def _compute_support(self, citation: CitationRecord, matched_name: str) -> Tuple[float, str]:
        """
        Support (S) check: Verifies if the case opinion text entails the claimed holding.
        Returns: (S_score, evidence_snippet)
        """
        if not citation.holding or not citation.holding.strip():
            # If no holding is asserted, assume neutral default support
            return 0.50, ""

        query_name = matched_name if matched_name else citation.case_name
        opinion_text = ""

        # Retrieve opinion text from ChromaDB
        if self.collection is not None and self.embedder is not None:
            try:
                results = self.collection.query(
                    query_embeddings=[self.embedder.encode(query_name).tolist()],
                    n_results=2,
                    where={"case_name": {"$eq": query_name}} if matched_name else None
                )
                if results.get("documents") and results["documents"][0]:
                    opinion_text = " ".join(results["documents"][0])
            except Exception as e:
                logger.debug("Chroma lookup with where filter fallback: %s", e)

            if not opinion_text:
                try:
                    results = self.collection.query(
                        query_embeddings=[self.embedder.encode(f"{query_name} {citation.holding}").tolist()],
                        n_results=1,
                    )
                    if results.get("documents") and results["documents"][0]:
                        opinion_text = results["documents"][0][0]
                except Exception as e:
                    logger.debug("Chroma lookup fallback failed: %s", e)

        # 1. NLI Cross-Encoder Entailment Scoring
        support_score = 0.0
        if opinion_text and self.nli_model is not None:
            try:
                # Segment text into premise windows
                step = 350
                windows = [opinion_text[i:i+450] for i in range(0, min(len(opinion_text), 1800), step)]
                pairs = [[w, citation.holding] for w in windows]
                preds = self.nli_model.predict(pairs)

                import numpy as np
                probs = []
                for p in preds:
                    if isinstance(p, (list, np.ndarray)) and len(p) == 3:
                        # Softmax: class 2 is entailment (contradiction=0, neutral=1, entailment=2)
                        exps = np.exp(p - np.max(p))
                        prob = exps / np.sum(exps)
                        probs.append(float(prob[2]))
                    elif isinstance(p, (list, np.ndarray)) and len(p) == 1:
                        probs.append(float(p[0]))
                    else:
                        probs.append(float(p))

                support_score = max(probs) if probs else 0.0
            except Exception as e:
                logger.debug("NLI prediction failed: %s", e)
                support_score = 0.0

        # 2. Semantic Fallback (Legal-BERT Cosine Similarity)
        if support_score == 0.0 and opinion_text and self.embedder is not None and util:
            try:
                h_vec = self.embedder.encode(citation.holding, convert_to_tensor=True)
                doc_vec = self.embedder.encode(opinion_text[:500], convert_to_tensor=True)
                sem_sim = float(util.cos_sim(h_vec, doc_vec))
                support_score = min(max(sem_sim, 0.0), 1.0)
            except Exception as e:
                logger.debug("Semantic similarity fallback failed: %s", e)
                support_score = 0.50

        if not opinion_text:
            # If opinion text wasn't found in Chroma, provide fallback
            support_score = 0.50

        evidence_snippet = opinion_text[:200] if opinion_text else ""
        return round(support_score, 4), evidence_snippet

    def compute_ccs(self, citation: CitationRecord) -> dict:
        """
        Factorized Citation Confidence Scoring:
          CCS = Existence (E) × Support (S)
        Assigns one of the 5 verdicts:
          SUPPORTED, UNCERTAIN, MISATTRIBUTED, FABRICATED, UNVERIFIABLE
        """
        E, matched_name, match_type = self._compute_existence(citation)
        S, evidence = self._compute_support(citation, matched_name)

        # Factorized score
        ccs = round(E * S, 4)

        # 5-Way Verdict Classification
        if E >= 0.85:
            # Case exists in authority corpus
            if ccs >= self.tau:
                tier = "SUPPORTED"
            elif ccs >= CCS_UNCERTAIN_THRESHOLD:
                tier = "UNCERTAIN"
            else:
                # Real case, but holding is not supported by opinion text
                tier = "MISATTRIBUTED"
        else:
            # Case does NOT exist in authority corpus
            if self._is_volume_covered(citation):
                # Cited volume was indexed, but no such case exists there
                tier = "FABRICATED"
            else:
                # Volume was not ingested/covered; cannot prove it does not exist
                tier = "UNVERIFIABLE"

        return {
            "ccs":            ccs,
            "tier":           tier,
            "existence":      round(E, 3),
            "support":        round(S, 3),
            "exact_score":    1.0 if match_type == "EXACT" else 0.0,
            "fuzzy_score":    round(E, 3) if match_type == "FUZZY" else 0.0,
            "semantic_score": round(S, 3),
            "best_match":     matched_name,
            "evidence":       evidence,
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

        n_supported = sum(1 for v in report.values() if v["tier"] == "SUPPORTED")
        n_uncertain = sum(1 for v in report.values() if v["tier"] == "UNCERTAIN")
        n_misattributed = sum(1 for v in report.values() if v["tier"] == "MISATTRIBUTED")
        n_fabricated = sum(1 for v in report.values() if v["tier"] == "FABRICATED")
        n_unverifiable = sum(1 for v in report.values() if v["tier"] == "UNVERIFIABLE")

        logger.info(
            "Factorized CCE scored %d citations: avg_CCS=%.4f | SUPPORTED=%d, UNCERTAIN=%d, MISATTRIBUTED=%d, FABRICATED=%d, UNVERIFIABLE=%d",
            len(ccs_values), avg_ccs,
            n_supported, n_uncertain, n_misattributed, n_fabricated, n_unverifiable,
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
