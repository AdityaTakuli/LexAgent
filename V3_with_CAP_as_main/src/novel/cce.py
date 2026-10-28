# LexAgent v3.0 | cce.py
"""
Citation Confidence Engine (CCE) — Grounded in CAP Primary Authority.
Factorized Verification (Novelty 1):
  CCS = Existence (E) × Support (S)

Verification order:
  1. The citation. Does the named case exist, and which report in the corpus is it? A case name
     is shared by the opinion and by one-line order entries in other volumes, so the citation is
     resolved to the actual opinion (by reporter cite when given, otherwise the substantial
     report with that name). A reporter cite that belongs to a different case is recorded.
  2. The holding. Only for a citation that exists: is the claimed holding entailed by that
     opinion's own text? The passages of that opinion most relevant to the holding are checked.

Two policies (config.CCE_POLICY):
  "legacy"  the five verdicts below, as in the runs made before the forensic audit.
  "gated"   existence and support are reported as separate fields (existence_verdict,
            support_verdict). The single "tier" that the judge, the controller and the memory act
            on uses the support verdict only when a support model has passed validation (the gate);
            with the gate closed an existing case is VERIFIED on existence alone.

5-Way Verdict Classification:
  1. SUPPORTED:     Case exists in CAP and text entails holding (CCS >= tau)
  2. UNCERTAIN:     Case exists, support is borderline (0.40 <= CCS < tau)
  3. MISATTRIBUTED: Case exists, but text does NOT entail holding (CCS < 0.40)
  4. FABRICATED:    Volume is covered in corpus, but case does not exist at citation
  5. UNVERIFIABLE:  Case not in corpus, and corpus does not cover where it would be
"""

import os
import json
import pickle
import re
from typing import Optional, Tuple, Dict

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
    CCE_FUZZY_THRESHOLD_RATIO,
    CASE_CATALOG_PATH,
    CCE_GATE_PATH,
    CCE_POLICY,
    LANDMARK_CAP_VOLUMES,
    NLI_MODEL_NAME,
    RRF_K,
)
from src.agents.state import LexAgentState
from src.data.case_catalog import CITE_MATCH, CITE_MISMATCH, CaseCatalog, Report, Resolution, names_match, parse_us_cite
from src.data.corpus_schema import CitationRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)

# How many passages of the cited opinion are checked against the claimed holding.
EVIDENCE_CHUNKS = 3


class CitationConfidenceEngine:
    """Factorized Citation Confidence Scoring with CAP Authority Catalog."""

    def __init__(
        self,
        chroma_collection,
        embedder,
        known_cases_path: str,
        nli_model=None,
        tau: float = TAU_ACCEPTANCE_THRESHOLD,
        catalog: Optional[CaseCatalog] = None,
        policy: Optional[str] = None,
        gate: Optional[dict] = None,
    ):
        self.policy = policy or CCE_POLICY
        if self.policy not in ("gated", "legacy"):
            raise ValueError(f"CCE policy must be 'gated' or 'legacy', not {self.policy!r}")
        self.gate = gate if gate is not None else load_support_gate()
        self.support_model_name = NLI_MODEL_NAME
        self.collection = chroma_collection
        self.embedder = embedder
        self.known_cases: set = self._load_known_cases(known_cases_path)
        self._name_to_canonical: Optional[Dict[str, str]] = None
        self.catalog = catalog if catalog is not None else self._load_catalog()
        self.tau = tau if tau is not None else TAU_ACCEPTANCE_THRESHOLD
        self.nli_model = nli_model
        if self.nli_model is None:
            wanted = str(self.gate.get("model") or NLI_MODEL_NAME)
            if self.policy == "gated" and self.gate.get("validated") and wanted != NLI_MODEL_NAME:
                self.nli_model = self._init_nli_model(wanted)
                self.support_model_name = wanted if self.nli_model is not None else NLI_MODEL_NAME
            if self.nli_model is None:
                self.nli_model = self._init_nli_model()
        logger.info("CCE policy=%s | support gate %s (%s)", self.policy,
                    "OPEN" if self.gate.get("validated") else "CLOSED", self.gate.get("source", ""))

        logger.info(
            "CCE (Factorized) initialized: %d known cases, %s reports in the case catalog, tau=%.2f, NLI active=%s",
            len(self.known_cases),
            f"{len(self.catalog):,}" if self.catalog is not None else "no",
            self.tau,
            self.nli_model is not None,
        )

    def _load_catalog(self) -> Optional[CaseCatalog]:
        """Load the catalog of reports written by the database build, or derive it once from the index."""
        catalog = CaseCatalog.load(CASE_CATALOG_PATH)
        if catalog is not None or self.collection is None:
            return catalog
        try:
            logger.warning("No case catalog at %s; building it from the index (one-off)...", CASE_CATALOG_PATH)
            catalog = CaseCatalog.from_collection(self.collection)
            if len(catalog):
                catalog.save(CASE_CATALOG_PATH)
                return catalog
        except Exception as e:
            logger.warning("Could not build the case catalog from the index: %s", e)
        return None

    def _init_nli_model(self, name: str = NLI_MODEL_NAME):
        """Attempts to load a fast NLI cross-encoder model for entailment scoring."""
        try:
            from sentence_transformers import CrossEncoder
            # Lightweight NLI cross-encoder (~0.4 GB)
            model = CrossEncoder(name, max_length=1024 if "long" in name else 512)
            logger.info("Loaded NLI CrossEncoder: %s", name)
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
        # The volumes actually in the corpus; the landmark preset only when there is no catalog.
        covered = self.catalog.volumes if self.catalog is not None else set(LANDMARK_CAP_VOLUMES)
        text_to_check = f"{citation.case_name} {getattr(citation, 'reporter_cite', '')} {getattr(citation, 'holding', '')}"
        match = re.search(r'(\d+)\s+U\.?\s*S\.?', text_to_check, re.IGNORECASE)
        if match:
            vol = str(match.group(1))
            if vol in covered:
                return True
        if hasattr(citation, "volume") and str(citation.volume) in covered:
            return True
        return False

    def _compute_existence(self, citation: CitationRecord) -> Tuple[float, str, str]:
        """
        Existence (E) check: Verifies if the case exists in the CAP catalog.
        Returns: (E_score, matched_canonical_name, match_type)
        """
        case_name = citation.case_name.strip().lower()
        reporter_cite = getattr(citation, "reporter_cite", "").strip().lower()

        if self._name_to_canonical is None:
            self._name_to_canonical = {c.lower().strip(): c for c in self.known_cases if c}
        name_to_canonical = self._name_to_canonical

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

    def _resolve_opinion(self, citation: CitationRecord, matched_canonical_name: str) -> Optional[Resolution]:
        """Which report in the corpus does this citation refer to? None when there is no case catalog."""
        if self.catalog is None:
            return None
        reporter_cite = getattr(citation, "reporter_cite", "") or ""
        try:
            year = int(getattr(citation, "year", 0) or 0)
        except (TypeError, ValueError):
            year = 0
        resolution = self.catalog.resolve(citation.case_name, reporter_cite, year)
        if resolution.report is None and matched_canonical_name and not parse_us_cite(matched_canonical_name):
            # The name as written resolves to nothing; try the catalog name the existence check matched it to.
            by_matched = self.catalog.resolve(matched_canonical_name, reporter_cite, year)
            if by_matched.report is not None:
                # The cite verdict belongs to the name the citation actually used.
                status = by_matched.cite_status if by_matched.basis == "cite" else resolution.cite_status
                return Resolution(by_matched.report, by_matched.basis, status, by_matched.same_name_reports)
        return resolution

    def _opinion_passages(self, holding: str, report: Report) -> str:
        """The passages of one opinion that are most relevant to the claimed holding.

        The number of passages follows LEXAGENT_CCE_SCAN_LIMIT (characters of opinion text scanned):
        the default of 8,000 covers three passages, and raising it checks more of the opinion.
        """
        scan_limit = int(os.environ.get("LEXAGENT_CCE_SCAN_LIMIT", "8000"))
        passages = max(EVIDENCE_CHUNKS, round(scan_limit / 2700))
        results = self.collection.query(
            query_embeddings=[self.embedder.encode(holding).tolist()],
            n_results=min(passages, max(1, report.chunks)),
            where={"parent_doc_id": {"$eq": report.doc_id}},
        )
        documents = (results.get("documents") or [[]])[0]
        return " ".join(doc for doc in documents if doc)

    def _compute_support(
        self,
        citation: CitationRecord,
        matched_canonical_name: str,
        report: Optional[Report] = None,
        resolved: bool = False,
    ) -> Tuple[Optional[float], str, bool]:
        """
        Support (S) check: Verifies if the case opinion text entails the claimed holding.
        The evidence is the text of the opinion the citation was resolved to (report). A citation
        that resolves only to an order entry, or to nothing, has no opinion text to check.
        Returns: (S_score, evidence_snippet, evidence_available)
        """
        if not citation.holding or not citation.holding.strip():
            return None, "", False

        query_name = matched_canonical_name if matched_canonical_name else citation.case_name
        if not query_name:
            return None, "", False

        opinion_text = ""

        if resolved:
            if report is None or not report.is_opinion or self.collection is None or self.embedder is None:
                return None, "", False
            try:
                opinion_text = self._opinion_passages(citation.holding, report)
            except Exception as e:
                logger.debug("Chroma opinion passage query exception: %s", e)

        # Without a case catalog: retrieve by case name, as before
        elif self.collection is not None and self.embedder is not None:
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
        # ── Step 1: the citation. Does the case exist, and which report is it? ──
        E, matched_name, match_type = self._compute_existence(citation)
        if match_type == "FUZZY" and E >= 0.85 and self.catalog is not None:
            # Word-overlap matching finds a "match" for almost any name in a catalog of this size
            # ("Dobbs v. Jackson Women's Health Organization" overlaps "Jackson v. Jackson"). A loose
            # match counts as the same case only if both parties agree, or the names differ by a typo.
            typo_level = fuzz.ratio(citation.case_name.strip().lower(), matched_name.strip().lower()) / 100.0
            if not names_match(citation.case_name, matched_name) and typo_level < 0.85:
                E, matched_name = round(typo_level, 4), ""
        resolution = self._resolve_opinion(citation, matched_name)
        report = resolution.report if resolution is not None else None
        cite_check = resolution.cite_status if resolution is not None else "NOT_CHECKED"

        if resolution is not None:
            if resolution.basis == "cite" and cite_check == CITE_MATCH:
                # The reporter cite and the name identify one report in the corpus: the citation exists.
                E, matched_name = 1.0, report.name
                if match_type != "EXACT":
                    match_type = "CITE"
            elif match_type == "FUZZY" and cite_check == CITE_MISMATCH:
                # The name only resembles a real case, and its reporter cite belongs to a different
                # case: the citation as given does not exist.
                E, report = 0.0, None

        def verdict(tier: str, support: Optional[float], evidence: str, evidence_available: bool) -> dict:
            return {
                "ccs":                round(E * support, 4) if support is not None else 0.0,
                "tier":               tier,
                "existence":          round(E, 3),
                "support":            round(support, 3) if support is not None else None,
                "evidence_available": evidence_available,
                "exact_score":        1.0 if match_type in ("EXACT", "CITE") else 0.0,
                "fuzzy_score":        round(E, 3) if match_type == "FUZZY" else 0.0,
                "semantic_score":     round(support, 3) if support is not None else None,
                "best_match":         matched_name,
                "evidence":           evidence,
                "evidence_source":    f"{report.name}, {report.cite}".strip(", ") if (report is not None and evidence_available) else "",
                "cite_check":         cite_check,
                "resolved_doc_id":    report.doc_id if report is not None else None,
                "resolved_name":      report.name if report is not None else None,
                "resolved_cite":      report.cite if report is not None else None,
                "resolved_is_opinion": bool(report.is_opinion) if report is not None else False,
                "citation":           citation.to_dict(),
            }

        if self.policy == "gated":
            return self._gated_verdict(citation, E, report, cite_check, verdict)

        def legacy(out: dict) -> dict:
            # The same result, with existence and support also stated separately.
            exists = out["tier"] not in ("FABRICATED",) and out["existence"] >= 0.85
            out.update({
                "policy": "legacy",
                "existence_verdict": "EXISTS" if exists else ("NOT_FOUND" if out["tier"] == "FABRICATED" else "NOT_IN_CORPUS"),
                "support_verdict": ("NOT_CHECKED" if out["support"] is None else
                                    ("SUPPORTED" if out["support"] >= self.tau else "NOT_SHOWN")),
                "support_validated": False,
            })
            return out

        if E < 0.85:
            # Case does NOT exist in authority corpus: there is no opinion to check a holding against
            tier = "FABRICATED" if self._is_volume_covered(citation) else "UNVERIFIABLE"
            return legacy(verdict(tier, None, "", False))

        # ── Step 2: the holding, against the text of that opinion ──
        S, evidence, evidence_available = self._compute_support(
            citation, matched_name, report=report, resolved=resolution is not None
        )

        if not evidence_available:
            # Real case, but its opinion text is not in the corpus -> cannot verify holding
            return legacy(verdict("UNVERIFIABLE", None, "", False))

        s_float = S if S is not None else 0.0
        if s_float >= self.tau:
            tier = "SUPPORTED"
            if cite_check == CITE_MISMATCH:
                # The holding fits the named case, but the reporter cite given belongs to another case.
                tier = "UNCERTAIN"
        elif s_float >= CCS_UNCERTAIN_THRESHOLD:
            tier = "UNCERTAIN"
        else:
            # Real case, but holding is not supported by opinion text
            tier = "MISATTRIBUTED"

        return legacy(verdict(tier, s_float, evidence, True))

    def check_existence(self, case_name: str, reporter_cite: str = "", year: int = 0) -> dict:
        """Does this citation exist, and which report is it? No holding is involved and no model is run.

        Used by the evaluation, so that "the cited case exists" is decided by the case catalog and
        not by string similarity to a list of names.
        """
        record = CitationRecord(case_name=str(case_name or ""), court="", year=int(year or 0), holding="", agent_source="evaluation")
        record.reporter_cite = str(reporter_cite or "")
        policy, self.policy = self.policy, "gated"
        try:
            out = self.compute_ccs(record)
        finally:
            self.policy = policy
        keys = ("existence_verdict", "existence", "cite_check", "resolved_doc_id", "resolved_name", "resolved_cite", "resolved_is_opinion")
        return {key: out.get(key) for key in keys}

    # ── gated policy ────────────────────────────────────────────────────────
    def _label_index(self, wanted: str) -> Optional[int]:
        """Index of an NLI label ("entailment", "contradiction") in the support model's output."""
        for obj in [self.nli_model, getattr(self.nli_model, "model", None)]:
            mapping = getattr(getattr(obj, "config", None), "label2id", None)
            if isinstance(mapping, dict):
                for label, index in mapping.items():
                    if wanted[:6] in str(label).lower():
                        return int(index)
        return None

    def _gated_support(self, holding: str, report: Report) -> Tuple[float, Optional[float], str]:
        """Support for a holding from the opinion it is cited to: (entailment, contradiction, window).

        Passages of that opinion are ranked by a fusion of embedding and lexical similarity to the
        holding (or by one of the two, as the gate file says), read whole, and scored in windows.
        The window returned is the one that produced the score.
        """
        import numpy as np

        got = self.collection.get(where={"parent_doc_id": {"$eq": report.doc_id}}, include=["documents", "metadatas", "embeddings"])
        rows = sorted(zip(got["metadatas"], got["documents"], got["embeddings"]), key=lambda row: int((row[0] or {}).get("chunk_index", 0)))
        texts = [text or "" for _, text, _ in rows]
        if not texts:
            raise LookupError("the opinion has no passages in the collection")
        ranking = str(self.gate.get("ranking", "hybrid"))
        count = int(self.gate.get("passages", EVIDENCE_CHUNKS))
        matrix = np.asarray([vector for _, _, vector in rows], dtype="float32")
        query = np.asarray(self.embedder.encode(holding), dtype="float32")
        dense = list(np.argsort(-(matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-12)))
        order = dense
        if ranking in ("hybrid", "lexical"):
            from rank_bm25 import BM25Okapi
            tokenize = lambda text: re.findall(r"[a-z0-9]+", text.lower())
            scores = BM25Okapi([tokenize(t) for t in texts]).get_scores(tokenize(holding))
            lexical = sorted(range(len(texts)), key=lambda i: -scores[i])
            order = lexical
            if ranking == "hybrid":
                fused: Dict[int, float] = {}
                for ranked in (dense, lexical):
                    for rank, index in enumerate(ranked, start=1):
                        fused[int(index)] = fused.get(int(index), 0.0) + 1.0 / (RRF_K + rank)
                order = sorted(fused, key=lambda i: -fused[i])
        if str(self.gate.get("granularity", "window")) == "passage":
            windows = [texts[int(i)] for i in order[:count]]
        else:
            windows = [texts[int(i)][start:start + 450] for i in order[:count] for start in range(0, max(1, len(texts[int(i)])), 350)]
        logits = np.asarray(self.nli_model.predict([[w, holding] for w in windows], batch_size=64, show_progress_bar=False), dtype="float64")
        if logits.ndim == 1:
            best = int(logits.argmax())
            return float(logits[best]), None, windows[best]
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exps / exps.sum(axis=1, keepdims=True)
        entail = probabilities[:, self._get_entailment_index()]
        contra_index = self._label_index("contradiction")
        best = int(entail.argmax())
        contradiction = float(probabilities[:, contra_index].max()) if contra_index is not None else None
        return float(entail[best]), contradiction, windows[best]

    def _gated_verdict(self, citation: CitationRecord, E: float, report: Optional[Report], cite_check: str, verdict) -> dict:
        """Existence and support stated separately; the support verdict acts only through an open gate."""
        gate_open = bool(self.gate.get("validated"))

        def finish(out: dict, existence: str, support: str, score=None, contradiction=None, window: str = "") -> dict:
            out.update({
                "policy": "gated", "existence_verdict": existence, "support_verdict": support,
                "support_validated": gate_open, "support_score": round(score, 4) if score is not None else None,
                "contradiction_score": round(contradiction, 4) if contradiction is not None else None,
                "evidence_window": window, "support_model": self.support_model_name,
            })
            return out

        if E < 0.85:
            if self._is_volume_covered(citation):
                return finish(verdict("FABRICATED", None, "", False), "NOT_FOUND", "NOT_CHECKED")
            return finish(verdict("UNVERIFIABLE", None, "", False), "NOT_IN_CORPUS", "NOT_CHECKED")

        score = contradiction = None
        window, support = "", "NOT_CHECKED"
        checkable = (report is not None and report.is_opinion and bool((citation.holding or "").strip())
                     and self.nli_model is not None and self.collection is not None and self.embedder is not None)
        if checkable:
            try:
                score, contradiction, window = self._gated_support(citation.holding, report)
                threshold = float(self.gate.get("threshold", self.tau))
                if score >= threshold:
                    support = "SUPPORTED"
                elif contradiction is not None and contradiction >= 0.90:
                    support = "CONTRADICTED"
                else:
                    support = "NOT_SHOWN"
            except Exception as e:
                logger.debug("Support check failed for %s: %s", citation.case_name, e)

        if gate_open:
            tier = {"SUPPORTED": "SUPPORTED", "CONTRADICTED": "MISATTRIBUTED", "NOT_SHOWN": "NOT_SHOWN"}.get(support, "UNVERIFIABLE")
            if tier == "SUPPORTED" and cite_check == CITE_MISMATCH:
                tier = "UNCERTAIN"
            out = verdict(tier, score, window, support != "NOT_CHECKED")
        else:
            # No validated support model: the citation is verified to exist, nothing more is claimed.
            # A reporter cite that belongs to another case keeps the citation out of the verified set.
            tier = "UNCERTAIN" if cite_check == CITE_MISMATCH else "VERIFIED"
            out = verdict(tier, None, window, support != "NOT_CHECKED")
            out["ccs"] = round(E * (0.5 if tier == "UNCERTAIN" else 1.0), 4)
            out["support"] = round(score, 3) if score is not None else None
            out["semantic_score"] = out["support"]
        return finish(out, "EXISTS", support, score, contradiction, window)


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

        n_supported = sum(1 for v in report.values() if v["tier"] in ("SUPPORTED", "VERIFIED"))
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


def load_support_gate(path: str = CCE_GATE_PATH) -> dict:
    """The record of whether a support model has been validated. Missing or unreadable = closed."""
    gate = {"validated": False, "model": NLI_MODEL_NAME, "threshold": TAU_ACCEPTANCE_THRESHOLD, "ranking": "hybrid",
            "passages": EVIDENCE_CHUNKS, "source": "no validation on record"}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            gate.update(json.load(handle))
        gate["source"] = gate.get("source") or path
    except (OSError, ValueError):
        pass
    return gate


def make_cce_node(cce: CitationConfidenceEngine):
    def cce_node(state: LexAgentState) -> dict:
        from src.agents.state import debate_protocol
        from src.novel.ledger import merge_round

        report, avg_ccs = cce.run_on_state(state)
        updates = {
            "citation_confidence_report": report,
            "avg_ccs": avg_ccs,
        }
        if debate_protocol(state) == "ledger":
            # Keep what this round verified next to what earlier rounds verified.
            ledger, newly_reliable = merge_round(state.get("citation_ledger"), report, state.get("debate_round", 0) + 1)
            updates["citation_ledger"] = ledger
            updates["round_new_supported"] = newly_reliable
        return updates
    return cce_node
