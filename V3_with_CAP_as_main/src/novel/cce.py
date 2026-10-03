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
    NLI_MODEL_NAME,
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
            model = CrossEncoder(NLI_MODEL_NAME, max_length=512)
            logger.info("Loaded NLI CrossEncoder: %s", NLI_MODEL_NAME)
            return model
        except Exception as e:
            logger.info("NLI CrossEncoder not initialized; using Legal-BERT semantic support fallback: %s", e)
            return None

    def _get_entailment_index(self) -> int:
        """Dynamically resolves the entailment class index from NLI model config."""
        if self.nli_model is not None:
            for obj in [self.nli_model, getattr(self.nli_model, "model", None)]:
                cfg = getattr(obj, "config", None)
                if cfg is not None:
                    label2id = getattr(cfg, "label2id", None)
                    if isinstance(label2id, dict):
                        for k, v in label2id.items():
                            if "entail" in str(k).lower():
                                return int(v)
                    id2label = getattr(cfg, "id2label", None)
                    if isinstance(id2label, dict):
                        for k, v in id2label.items():
                            if "entail" in str(v).lower():
                                return int(k)
        # Default for cross-encoder/nli-deberta-v3-* (0: contradiction, 1: entailment, 2: neutral)
        return 1

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

        name_to_canonical = {c.lower().strip(): c for c in self.known_cases if c}

        # 1. Exact canonical catalog match by case name
        if case_name in name_to_canonical:
            return 1.0, name_to_canonical[case_name], "EXACT"

        # 2. Match by reporter citation ONLY if party names share material tokens
        c_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', case_name)) - {'the', 'and', 'for', 'state', 'court', 'united', 'states'}
        if reporter_cite:
            for kc in self.known_cases:
                kc_lower = kc.lower().strip()
                if reporter_cite in kc_lower:
                    kc_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', kc_lower))
                    if c_tokens and (c_tokens & kc_tokens):
                        return 1.0, kc, "EXACT"

        # 3. Fuzzy string and token-set matching against known case catalog
        fuzzy_score = 0.0
        best_match = ""

        for real_case in self.known_cases:
            rc_lower = real_case.lower().strip()
            r1 = fuzz.ratio(case_name, rc_lower) / 100.0
            r2 = (fuzz.token_set_ratio(case_name, rc_lower) / 100.0) if hasattr(fuzz, "token_set_ratio") else 0.0
            ratio = max(r1, r2)
            if ratio < 0.85 and c_tokens:
                rc_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', rc_lower))
                if c_tokens and c_tokens.issubset(rc_tokens):
                    ratio = 0.90
            if ratio > fuzzy_score:
                fuzzy_score = ratio
                best_match = real_case

        fuzzy_threshold = CCS_VERIFIED_THRESHOLD * CCE_FUZZY_THRESHOLD_RATIO
        if fuzzy_score >= fuzzy_threshold and best_match:
            return round(fuzzy_score, 4), best_match, "FUZZY"

        return 0.0, "", "NOT_FOUND"

    def _compute_support(self, citation: CitationRecord, matched_canonical_name: str) -> Tuple[Optional[float], str, bool]:
        """
        Support (S) check: Verifies if the case opinion text entails the claimed holding.
        Only retrieves chunks strictly belonging to the canonical case. Zero cross-case drift.
        Returns: (S_score, evidence_snippet, evidence_available)
        """
        if not citation.holding or not citation.holding.strip():
            return None, "", False

        query_name = matched_canonical_name if matched_canonical_name else citation.case_name
        if not query_name:
            return None, "", False

        opinion_text = ""

        # Retrieve opinion text strictly filtered by the resolved canonical case
        if self.collection is not None and self.embedder is not None:
            try:
                # 1. Exact metadata equality filter
                results = self.collection.query(
                    query_embeddings=[self.embedder.encode(query_name).tolist()],
                    n_results=3,
                    where={"case_name": {"$eq": query_name}}
                )
                if results.get("documents") and results["documents"][0]:
                    opinion_text = " ".join(results["documents"][0])
            except Exception as e:
                logger.debug("Chroma exact metadata query exception: %s", e)

            # 2. If metadata filter found nothing, query by embedding of case title and verify document identity
            if not opinion_text:
                try:
                    cand_results = self.collection.query(
                        query_embeddings=[self.embedder.encode(query_name).tolist()],
                        n_results=3,
                    )
                    docs = cand_results.get("documents", [[]])[0]
                    metas = cand_results.get("metadatas", [[]])[0]
                    matched_docs = []
                    q_lower = query_name.lower().strip()
                    for doc_str, m in zip(docs, metas):
                        meta_case = str(m.get("case_name", "")).lower().strip() if isinstance(m, dict) else ""
                        if meta_case and (meta_case in q_lower or q_lower in meta_case or fuzz.ratio(meta_case, q_lower) >= 80):
                            matched_docs.append(doc_str)
                    if matched_docs:
                        opinion_text = " ".join(matched_docs)
                except Exception as e:
                    logger.debug("Chroma canonical candidate query exception: %s", e)

        # If opinion text is not available in the database, return UNVERIFIABLE without arbitrary 0.50 score
        if not opinion_text or not opinion_text.strip():
            return None, "", False

        # 1. NLI Cross-Encoder Entailment Scoring strictly against this case's text
        support_score = 0.0
        if self.nli_model is not None:
            try:
                step = 350
                scan_limit = int(os.environ.get("LEXAGENT_CCE_SCAN_LIMIT", "8000"))
                windows = [opinion_text[i:i+450] for i in range(0, min(len(opinion_text), scan_limit), step)]
                pairs = [[w, citation.holding] for w in windows]
                preds = self.nli_model.predict(pairs)

                entail_idx = self._get_entailment_index()
                import numpy as np
                probs = []
                for p in preds:
                    if isinstance(p, (list, np.ndarray)) and len(p) >= 3:
                        exps = np.exp(p - np.max(p))
                        prob = exps / np.sum(exps)
                        probs.append(float(prob[entail_idx]))  # Index 1 = Entailment probability (not neutral)
                    elif isinstance(p, (list, np.ndarray)) and len(p) == 1:
                        probs.append(float(p[0]))
                    else:
                        probs.append(float(p))

                support_score = max(probs) if probs else 0.0
            except Exception as e:
                logger.debug("NLI prediction failed: %s", e)
                support_score = 0.0
        elif self.embedder is not None and util:
            try:
                h_vec = self.embedder.encode(citation.holding, convert_to_tensor=True)
                doc_vec = self.embedder.encode(opinion_text[:500], convert_to_tensor=True)
                sem_sim = float(util.cos_sim(h_vec, doc_vec))
                support_score = min(max(sem_sim, 0.0), 1.0)
            except Exception as e:
                logger.debug("Semantic similarity fallback failed: %s", e)
                support_score = 0.0

        evidence_snippet = opinion_text[:200].replace("\n", " ")
        return round(support_score, 4), evidence_snippet, True

    def compute_ccs(self, citation: CitationRecord) -> dict:
        """
        Factorized Citation Confidence Scoring:
          CCS = Existence (E) × Support (S)
        Assigns one of the 5 verdicts:
          SUPPORTED, UNCERTAIN, MISATTRIBUTED, FABRICATED, UNVERIFIABLE
        """
        E, matched_name, match_type = self._compute_existence(citation)
        S, evidence, evidence_available = self._compute_support(citation, matched_name)

        if not evidence_available:
            # Evidence was not found in the local index -> cannot verify holding
            ccs = 0.0
            if E >= 0.85:
                tier = "UNVERIFIABLE"  # Real case, but text is unindexed / unavailable
            else:
                if self._is_volume_covered(citation):
                    tier = "FABRICATED"
                else:
                    tier = "UNVERIFIABLE"

            return {
                "ccs":                ccs,
                "tier":               tier,
                "existence":          round(E, 3),
                "support":            None,
                "evidence_available": False,
                "exact_score":        1.0 if match_type == "EXACT" else 0.0,
                "fuzzy_score":        round(E, 3) if match_type == "FUZZY" else 0.0,
                "semantic_score":     None,
                "best_match":         matched_name,
                "evidence":           "",
                "citation":           citation.to_dict(),
            }

        # Evidence IS available in the corpus: evaluate entailment
        s_float = S if S is not None else 0.0
        ccs = round(E * s_float, 4)

        if E >= 0.85:
            # Case exists in authority corpus
            if s_float >= self.tau:
                tier = "SUPPORTED"
            elif s_float >= CCS_UNCERTAIN_THRESHOLD:
                tier = "UNCERTAIN"
            else:
                # Real case, but holding is not supported by opinion text
                tier = "MISATTRIBUTED"
        else:
            # Case does NOT exist in authority corpus
            if self._is_volume_covered(citation):
                tier = "FABRICATED"
            else:
                tier = "UNVERIFIABLE"

        return {
            "ccs":                ccs,
            "tier":               tier,
            "existence":          round(E, 3),
            "support":            round(s_float, 3),
            "evidence_available": True,
            "exact_score":        1.0 if match_type == "EXACT" else 0.0,
            "fuzzy_score":        round(E, 3) if match_type == "FUZZY" else 0.0,
            "semantic_score":     round(s_float, 3),
            "best_match":         matched_name,
            "evidence":           evidence,
            "citation":           citation.to_dict(),
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
