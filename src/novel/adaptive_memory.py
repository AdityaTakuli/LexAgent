# LexAgent v3.0 | adaptive_memory.py
"""Adaptive Memory: Cross-session citation risk tracking + verdict caching."""

import os
import sys
import hashlib
from datetime import datetime
import chromadb

from config import (
    CHROMA_MEMORY_PATH,
    MEMORY_CACHE_THRESHOLD,
    MEMORY_HIGH_RISK_CCS_THRESH,
)
from src.agents.state import LexAgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _stable_hash(text: str) -> str:
    """Stable, deterministic SHA-1 hash for IDs across Python processes."""
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()[:16]


class AdaptiveMemory:
    """Cross-session citation risk tracking + verdict caching."""

    def __init__(self, chroma_path: str = CHROMA_MEMORY_PATH, namespace: str = "", client=None):
        self.namespace = namespace
        if client is not None:
            self.client = client
        elif chroma_path == ":memory:":
            self.client = chromadb.EphemeralClient()
        else:
            self.client = chromadb.PersistentClient(path=chroma_path)

        prefix = f"{namespace}_" if namespace else ""
        verdict_coll_name = f"{prefix}verdict_cache"
        risk_coll_name = f"{prefix}citation_risk_index"

        self.verdict_collection = self.client.get_or_create_collection(
            verdict_coll_name, metadata={"hnsw:space": "cosine"}
        )
        self.risk_collection = self.client.get_or_create_collection(
            risk_coll_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            "AdaptiveMemory initialized (namespace='%s'): %d cached verdicts, %d risk entries",
            namespace,
            self.verdict_collection.count(),
            self.risk_collection.count(),
        )

    def check_cache(self, query: str, embedder) -> tuple:
        q_vec = embedder.encode(query).tolist()
        try:
            results = self.verdict_collection.query(
                query_embeddings=[q_vec], n_results=1
            )
            if (results.get("distances")
                    and results["distances"][0]
                    and results["distances"][0][0] < (1 - MEMORY_CACHE_THRESHOLD)):
                cached = results["metadatas"][0][0]
                answer = cached.get("answer", "")
                logger.info("Cache HIT: distance=%.4f", results["distances"][0][0])
                return True, answer
        except Exception as e:
            logger.debug("Cache check failed: %s", e)

        return False, ""

    def get_high_risk_citations(self, query: str, embedder) -> list:
        q_vec = embedder.encode(query).tolist()
        try:
            if self.risk_collection.count() == 0:
                return []
            results = self.risk_collection.query(
                query_embeddings=[q_vec], n_results=min(5, self.risk_collection.count())
            )
            high_risk = [
                m.get("case_name", "")
                for m in results.get("metadatas", [[]])[0]
                if m.get("avg_ccs", 1.0) < MEMORY_HIGH_RISK_CCS_THRESH
            ]
            if high_risk:
                logger.info("High-risk citations loaded: %s", high_risk)
            return high_risk
        except Exception as e:
            logger.debug("High-risk citation lookup failed: %s", e)
            return []

    def clear_verdict_cache(self):
        """Clears cached verdicts for this memory namespace."""
        try:
            name = self.verdict_collection.name
            self.client.delete_collection(name)
            self.verdict_collection = self.client.get_or_create_collection(
                name, metadata={"hnsw:space": "cosine"}
            )
            logger.info("Verdict cache cleared: %s", name)
        except Exception as e:
            logger.warning("Failed to clear verdict cache %s: %s", self.verdict_collection.name, e)

    def clear_all(self):
        """Clears both verdict cache and citation risk index."""
        self.clear_verdict_cache()
        try:
            name = self.risk_collection.name
            self.client.delete_collection(name)
            self.risk_collection = self.client.get_or_create_collection(
                name, metadata={"hnsw:space": "cosine"}
            )
            logger.info("Risk collection cleared: %s", name)
        except Exception as e:
            logger.warning("Failed to clear risk index %s: %s", self.risk_collection.name, e)

    def store_verdict(self, state: LexAgentState, embedder):
        q_vec = embedder.encode(state["query"]).tolist()
        doc_id = f"verdict_{_stable_hash(state['query'])}"

        self.verdict_collection.upsert(
            ids=[doc_id],
            embeddings=[q_vec],
            documents=[state.get("final_answer", "")],
            metadatas=[{
                "query":      state["query"],
                "answer":     state.get("final_answer", ""),
                "confidence": float(state.get("judge_confidence", 0.0)),
                "timestamp":  datetime.now().isoformat(),
                "avg_ccs":    float(state.get("avg_ccs", 0.0)),
            }],
        )
        logger.info("Verdict cached: id=%s", doc_id)

    def update_risk_index(self, state: LexAgentState, embedder):
        report = state.get("citation_confidence_report", {})
        fake_count = 0

        for cid, data in report.items():
            if data.get("tier") != "LIKELY_FAKE":
                continue

            case_name = data.get("citation", {}).get("case_name", "")
            if not case_name:
                continue

            risk_id = f"risk_{_stable_hash(case_name)}"
            risk_vec = embedder.encode(case_name).tolist()

            try:
                existing = self.risk_collection.get(ids=[risk_id])
                if existing and existing.get("ids"):
                    prev_ccs = existing["metadatas"][0].get("avg_ccs", data["ccs"])
                    prev_count = existing["metadatas"][0].get("count", 1)
                    new_count = prev_count + 1
                    new_avg_ccs = prev_ccs + (data["ccs"] - prev_ccs) / new_count
                else:
                    new_avg_ccs = data["ccs"]
                    new_count = 1
            except Exception:
                new_avg_ccs = data["ccs"]
                new_count = 1

            self.risk_collection.upsert(
                ids=[risk_id],
                embeddings=[risk_vec],
                documents=[case_name],
                metadatas=[{
                    "case_name": case_name,
                    "avg_ccs":   new_avg_ccs,
                    "count":     new_count,
                }],
            )
            fake_count += 1

        if fake_count:
            logger.info("Risk index updated: %d LIKELY_FAKE citations stored", fake_count)


def make_memory_nodes(memory: AdaptiveMemory, embedder):
    def memory_check_node(state: LexAgentState) -> dict:
        hit, cached = memory.check_cache(state["query"], embedder)
        high_risk = memory.get_high_risk_citations(state["query"], embedder)

        out = {
            "memory_cache_hit":    hit,
            "high_risk_citations": high_risk,
        }
        if hit:
            out["cached_answer"] = cached
            out["final_answer"] = cached
            logger.info("Memory cache hit — returning cached verdict")

        return out

    def memory_store_node(state: LexAgentState) -> dict:
        if state.get("judge_verified_citations") and state.get("judge_confidence", 0.0) > 0:
            memory.store_verdict(state, embedder)
        memory.update_risk_index(state, embedder)
        return {}

    return memory_check_node, memory_store_node
