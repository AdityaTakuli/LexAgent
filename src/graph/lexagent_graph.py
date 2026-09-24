# LexAgent v3.0 | lexagent_graph.py
"""LexAgent v3.0 — LangGraph StateGraph (DVA+ Protocol pipeline)."""

import os
import sys
from typing import Any

from langgraph.graph import StateGraph, END

from src.agents.state import LexAgentState
from src.agents.prosecutor_agent import prosecutor_node
from src.agents.defense_agent import defense_node
from src.agents.reflection_agent import reflection_node
from src.agents.judge_agent import judge_node
from src.novel.cce import CitationConfidenceEngine, make_cce_node
from src.novel.ddc import DynamicDebateController, make_ddc_node
from src.novel.adaptive_memory import AdaptiveMemory, make_memory_nodes
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_lexagent_graph(
    llm: Any,
    hybrid_retriever: Any,
    cce: CitationConfidenceEngine,
    ddc: DynamicDebateController,
    memory: AdaptiveMemory,
    embedder: Any,
):
    memory_check_node, memory_store_node = make_memory_nodes(memory, embedder)
    cce_node_fn = make_cce_node(cce)
    ddc_node_fn = make_ddc_node(ddc)

    builder = StateGraph(LexAgentState)

    builder.add_node("memory_check",  memory_check_node)
    builder.add_node("retrieve",      _make_retrieve_node(hybrid_retriever))
    builder.add_node("inject_llm",    _make_inject_llm_node(llm))
    builder.add_node("prosecutor",    prosecutor_node)
    builder.add_node("defense",       defense_node)
    builder.add_node("cce",           cce_node_fn)
    builder.add_node("reflection",    reflection_node)
    builder.add_node("ddc",           ddc_node_fn)
    builder.add_node("judge",         judge_node)
    builder.add_node("memory_store",  memory_store_node)

    builder.set_entry_point("memory_check")

    builder.add_conditional_edges(
        "memory_check",
        _route_after_memory_check,
        {"cache_hit": END, "proceed": "retrieve"}
    )

    builder.add_edge("retrieve",    "inject_llm")
    builder.add_edge("inject_llm",  "prosecutor")
    builder.add_edge("prosecutor",  "defense")
    builder.add_edge("defense",     "cce")
    builder.add_edge("cce",         "reflection")
    builder.add_edge("reflection",  "ddc")

    builder.add_conditional_edges(
        "ddc",
        _route_after_ddc,
        {"CONTINUE": "retrieve", "TERMINATE": "judge"}
    )

    builder.add_edge("judge",        "memory_store")
    builder.add_edge("memory_store", END)

    compiled = builder.compile()
    logger.info("LexAgent v3.0 DVA+ graph compiled successfully.")
    return compiled


def _route_after_memory_check(state: LexAgentState) -> str:
    if state.get("memory_cache_hit", False):
        logger.info("Memory cache hit — skipping debate pipeline")
        return "cache_hit"
    return "proceed"


def _route_after_ddc(state: LexAgentState) -> str:
    decision = state.get("ddc_decision", "TERMINATE")
    logger.info("DDC decision: %s — %s", decision, state.get("ddc_reason", ""))
    return decision


def _make_inject_llm_node(llm: Any):
    def inject_llm_node(state: LexAgentState) -> dict:
        return {"llm": llm}
    return inject_llm_node


def _make_retrieve_node(hybrid_retriever: Any):
    def retrieve_node(state: LexAgentState) -> dict:
        query = state["query"]
        jurisdiction = state.get("jurisdiction", "")
        current_round = state.get("debate_round", 0)

        if current_round > 0 and state.get("reflection_queries"):
            augmented = query + " " + " ".join(state["reflection_queries"][:2])
            logger.info("Round %d: augmented query with reflection queries", current_round)
            results = hybrid_retriever.retrieve(augmented, jurisdiction)
        else:
            results = hybrid_retriever.retrieve(query, jurisdiction)

        entity_map = hybrid_retriever.get_entity_map(results)
        return {
            "retrieved_passages": results,
            "entity_map":         entity_map,
            "debate_round":       current_round,
        }
    return retrieve_node
