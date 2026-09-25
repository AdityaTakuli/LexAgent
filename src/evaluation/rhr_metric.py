# LexAgent v3.0 | rhr_metric.py
"""
Re-Hallucination Rate (RHR) Metric.
Evaluates the persistence and recurrence of hallucinated legal citations
across semantically paraphrased query pairs.
"""

import os
import sys
import pickle
from typing import List, Dict, Any, Optional

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

from config import FUZZY_MATCH_THRESHOLD, KNOWN_CASES_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard Curated Benchmark Paraphrase Pairs
DEFAULT_PARAPHRASE_PAIRS = [
    {
        "id": 1,
        "topic": "Fourth Amendment — Private Searches",
        "q1": "Does the Fourth Amendment protect against warrantless searches conducted by private individuals or carriers acting without government involvement?",
        "q2": "Is evidence discovered during an unreasonable search conducted solely by a private freight carrier prohibited by the Fourth Amendment?",
    },
    {
        "id": 2,
        "topic": "Eighth Amendment — Juvenile Death Penalty",
        "q1": "Can a state constitutionally impose the death penalty on an offender who was under the age of 18 at the time the crime was committed?",
        "q2": "Does executing an individual for capital murder committed when they were 17 years old violate the Eighth Amendment?",
    },
    {
        "id": 3,
        "topic": "Fourth Amendment — Stop and Frisk",
        "q1": "May police officers stop and briefly detain a suspect on the street without probable cause if they possess reasonable articulable suspicion of criminal activity?",
        "q2": "Does the Fourth Amendment allow a brief investigatory stop and pat-down of an individual based on reasonable suspicion rather than probable cause?",
    },
    {
        "id": 4,
        "topic": "Fifth Amendment — Custodial Interrogation",
        "q1": "Are statements made by a suspect during custodial police interrogation admissible if the suspect was not advised of their right to remain silent and right to counsel?",
        "q2": "Must police warn a person in custody of their Fifth Amendment rights before interrogation for their confessions to be admissible at trial?",
    },
    {
        "id": 5,
        "topic": "Sixth Amendment — Right to Appointed Counsel",
        "q1": "Does an indigent criminal defendant charged with a felony in state court have a constitutional right to court-appointed counsel?",
        "q2": "Are state courts obligated under the Sixth and Fourteenth Amendments to appoint free legal defense for poor individuals accused of felonies?",
    },
    {
        "id": 6,
        "topic": "Fourteenth Amendment — Due Process Brady Evidence",
        "q1": "Does the prosecution violate due process when it suppresses evidence favorable to an accused upon request?",
        "q2": "Is a criminal conviction invalid under the Due Process Clause if the state fails to disclose material exculpatory evidence to defense counsel?",
    },
    {
        "id": 7,
        "topic": "Fourth Amendment — Exclusionary Rule in State Courts",
        "q1": "Is evidence obtained through an unconstitutional search and seizure under the Fourth Amendment admissible in state criminal trials?",
        "q2": "Does the Fourth Amendment exclusionary rule prohibit state prosecutors from introducing unconstitutionally seized evidence in state court proceedings?",
    },
    {
        "id": 8,
        "topic": "Sixth Amendment — Ineffective Assistance of Counsel",
        "q1": "What legal standard must a convicted defendant satisfy to prove their legal representation violated the Sixth Amendment right to counsel?",
        "q2": "To overturn a conviction based on inadequate legal defense, what deficient performance and prejudice must be demonstrated under the Sixth Amendment?",
    },
    {
        "id": 9,
        "topic": "Fourth Amendment — Cell Phone Search Incident to Arrest",
        "q1": "May law enforcement search the digital contents of a cell phone seized from an arrestee without obtaining a search warrant?",
        "q2": "Does police searching through an arrestee's smartphone without a warrant or exigent circumstances breach Fourth Amendment privacy protections?",
    },
    {
        "id": 10,
        "topic": "Fourteenth Amendment — Due Process Abortion Rights",
        "q1": "Does the Due Process Clause of the Fourteenth Amendment encompass a woman's qualified right to terminate her pregnancy?",
        "q2": "Does the constitutional right of privacy under the Fourteenth Amendment protect a woman's choice to have an abortion?",
    },
]


def _build_initial_state(query: str, jurisdiction: str = "US_Federal") -> dict:
    return {
        "query": query,
        "jurisdiction": jurisdiction,
        "debate_round": 0,
        "debate_history": [],
        "retrieved_passages": [],
        "entity_map": {},
        "prosecutor_argument": "",
        "prosecutor_citations": [],
        "defense_argument": "",
        "defense_citations": [],
        "defense_challenges": [],
        "defense_concede": False,
        "challenge_strength": 0.0,
        "defense_counter_passages": [],
        "evidence_overlap_ratio": 0.0,
        "reflection_gaps": [],
        "reflection_queries": [],
        "reflection_score": 0.0,
        "citation_confidence_report": {},
        "avg_ccs": 0.0,
        "high_risk_citations": [],
        "ddc_decision": "",
        "ddc_reason": "",
        "judge_verdict": "",
        "judge_confidence": 0.0,
        "judge_verified_citations": [],
        "judge_uncertain_citations": [],
        "judge_reasoning": "",
        "memory_cache_hit": False,
        "cached_answer": None,
        "final_answer": "",
        "structured_output": {},
    }


def _extract_all_citations(result: dict) -> List[str]:
    names = []
    groups = [
        result.get("prosecutor_citations", []),
        result.get("defense_citations", []),
        result.get("judge_verified_citations", []),
    ]
    for group in groups:
        for c in group or []:
            name = c.get("case_name", "") if isinstance(c, dict) else getattr(c, "case_name", "")
            if name and str(name).strip():
                clean = str(name).strip()
                if clean not in names:
                    names.append(clean)
    return names


def _identify_fake_citations(citations: List[str], normalized_known: set, fuzzy_threshold: int) -> List[str]:
    fakes = []
    for cite in citations:
        cite_lower = cite.lower().strip()
        if not cite_lower or cite_lower in normalized_known:
            continue
        best_r = max((fuzz.ratio(cite_lower, kc) for kc in normalized_known), default=0)
        if best_r < fuzzy_threshold:
            fakes.append(cite)
    return fakes


def _check_recurrence(fake_cites_q1: List[str], cites_q2: List[str], fuzzy_threshold: int) -> List[str]:
    recurred = []
    for f in fake_cites_q1:
        f_lower = f.lower().strip()
        matched = False
        for c in cites_q2:
            c_lower = c.lower().strip()
            if f_lower == c_lower or fuzz.ratio(f_lower, c_lower) >= fuzzy_threshold:
                matched = True
                break
        if matched and f not in recurred:
            recurred.append(f)
    return recurred


def compute_rhr(
    query_pairs: Optional[list] = None,
    lexagent_graph = None,
    known_cases: Optional[set] = None,
    jurisdiction: str = "US_Federal",
    fuzzy_threshold: int = FUZZY_MATCH_THRESHOLD,
) -> dict:
    """
    Computes Re-Hallucination Rate (RHR) across paraphrase query pairs (Q1, Q2).
    
    For each pair:
      1. Evaluates Q1 through lexagent_graph, identifying hallucinated citations H1.
      2. Evaluates Q2 through lexagent_graph (testing memory suppression/retention).
      3. Determines if any citation in H1 recurred in Q2.
      
    RHR = sum(|H1 ∩ C2|) / sum(|H1|) for pairs with initial hallucinations.
    """
    if query_pairs is None:
        query_pairs = DEFAULT_PARAPHRASE_PAIRS

    if not query_pairs:
        return {
            "rhr": 0.0,
            "pair_rhr": 0.0,
            "recurred": 0,
            "initial_hallucinations": 0,
            "evaluated_pairs": 0,
            "hallucinating_pairs": 0,
            "recurred_pairs": 0,
            "details": [],
        }

    if lexagent_graph is None:
        raise ValueError("A compiled lexagent_graph must be provided to compute RHR.")

    if known_cases is None:
        if os.path.exists(KNOWN_CASES_PATH):
            with open(KNOWN_CASES_PATH, "rb") as f:
                known_cases = pickle.load(f)
        else:
            known_cases = set()

    normalized_known = {c.lower().strip() for c in known_cases if c}

    total_initial_fake = 0
    total_recurred_fake = 0
    hallucinating_pairs = 0
    recurred_pairs = 0
    details = []

    for pair in query_pairs:
        if isinstance(pair, dict):
            q1 = pair.get("q1") or pair.get("query_a", "")
            q2 = pair.get("q2") or pair.get("query_b", "")
            topic = pair.get("topic", "")
            pair_id = pair.get("id", len(details) + 1)
        elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
            q1, q2 = pair[0], pair[1]
            topic = ""
            pair_id = len(details) + 1
        else:
            continue

        try:
            # 1. Execute Q1
            s1 = _build_initial_state(q1, jurisdiction)
            res1 = lexagent_graph.invoke(s1)
            cites_q1 = _extract_all_citations(res1)
            fakes_q1 = _identify_fake_citations(cites_q1, normalized_known, fuzzy_threshold)

            total_initial_fake += len(fakes_q1)
            if fakes_q1:
                hallucinating_pairs += 1

            # 2. Execute Q2 (tests if AdaptiveMemory suppressed or re-hallucinated fakes_q1)
            s2 = _build_initial_state(q2, jurisdiction)
            res2 = lexagent_graph.invoke(s2)
            cites_q2 = _extract_all_citations(res2)
            fakes_q2 = _identify_fake_citations(cites_q2, normalized_known, fuzzy_threshold)

            # 3. Check if fake citations from Q1 recurred in Q2
            recurred_in_q2 = _check_recurrence(fakes_q1, cites_q2, fuzzy_threshold)
            total_recurred_fake += len(recurred_in_q2)
            if recurred_in_q2:
                recurred_pairs += 1

            details.append({
                "pair_id": pair_id,
                "topic": topic,
                "q1": q1,
                "q2": q2,
                "q1_fake_citations": fakes_q1,
                "q2_fake_citations": fakes_q2,
                "recurred_citations": recurred_in_q2,
                "recurred_count": len(recurred_in_q2),
                "status": "EVALUATED",
            })
        except Exception as e:
            logger.error("RHR evaluation failed on pair %s: %s", pair_id, e)
            details.append({
                "pair_id": pair_id,
                "topic": topic,
                "status": "ERROR",
                "error": str(e),
            })

    evaluated_pairs = len([d for d in details if d.get("status") == "EVALUATED"])
    citation_rhr = round(total_recurred_fake / total_initial_fake, 4) if total_initial_fake > 0 else 0.0
    pair_rhr = round(recurred_pairs / hallucinating_pairs, 4) if hallucinating_pairs > 0 else 0.0

    logger.info(
        "RHR evaluated %d pairs: Citation RHR=%.4f (%d/%d recurred), Pair RHR=%.4f (%d/%d pairs)",
        evaluated_pairs, citation_rhr, total_recurred_fake, total_initial_fake,
        pair_rhr, recurred_pairs, hallucinating_pairs
    )

    return {
        "rhr": citation_rhr,
        "pair_rhr": pair_rhr,
        "recurred": total_recurred_fake,
        "initial_hallucinations": total_initial_fake,
        "evaluated_pairs": evaluated_pairs,
        "hallucinating_pairs": hallucinating_pairs,
        "recurred_pairs": recurred_pairs,
        "details": details,
    }
