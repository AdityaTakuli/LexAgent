# LexAgent v2.0 | Phase 1 | load_model.py
"""Model loading for Mistral-7B (NF4 4-bit) and Legal-BERT embedder.

Mistral-7B is loaded in 4-bit NF4 quantization to fit within T4 16GB VRAM.
Legal-BERT embedder runs on CPU to preserve GPU VRAM for the LLM.
"""

import os
import sys
import torch
from typing import Tuple, Any

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import MODEL_NAME, EMBEDDER_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_mistral_7b() -> Tuple[Any, Any, Any]:
    """Load Mistral-7B-Instruct-v0.3 in NF4 4-bit quantization.
    
    Uses BitsAndBytes NF4 quantization with double quantization for
    optimal memory efficiency on T4 16GB. Expected VRAM: ~5.2GB.
    
    Returns:
        Tuple of (model, tokenizer, llm_pipeline):
            - model: The quantized HuggingFace model
            - tokenizer: The associated tokenizer
            - llm_pipeline: LangChain-compatible HuggingFacePipeline wrapper
    
    Raises:
        RuntimeError: If CUDA is not available.
    """
    # ── Check CUDA availability ──────────────────────────────────────
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. Mistral-7B requires a GPU. "
            "Please enable GPU runtime: Runtime → Change runtime type → T4 GPU"
        )
        raise RuntimeError(
            "CUDA not available. LexAgent requires a T4 GPU (Colab free tier)."
        )
    
    logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
    logger.info("Loading %s with NF4 4-bit quantization...", MODEL_NAME)
    
    # ── Import heavy dependencies (deferred to reduce startup time) ──
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )
    from langchain_community.llms import HuggingFacePipeline
    
    # ── BitsAndBytes NF4 config ──────────────────────────────────────
    # NF4 quantization preserves model quality better than INT4
    # Double quantization further reduces memory overhead
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # ── Load tokenizer ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    # Ensure pad token is set (Mistral uses EOS as pad by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ── Load model with quantization ─────────────────────────────────
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load model %s: %s", MODEL_NAME, e)
        logger.info("Retrying model download...")
        # Retry once on HuggingFace download failure
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as retry_e:
            logger.error("Retry failed: %s", retry_e)
            raise
    
    # ── Print VRAM usage ─────────────────────────────────────────────
    vram_gb = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Model loaded. VRAM used: {vram_gb:.2f} GB")
    print(f"Model loaded. VRAM used: {vram_gb:.2f} GB")
    
    # ── Build HuggingFace text-generation pipeline ───────────────────
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
    )
    
    # ── Wrap in LangChain HuggingFacePipeline ────────────────────────
    llm_pipeline = HuggingFacePipeline(pipeline=text_pipeline)
    
    logger.info("LLM pipeline ready (max_new_tokens=512, temp=0.1).")
    return model, tokenizer, llm_pipeline


def load_embedder() -> Any:
    """Load Legal-BERT embedder for dense retrieval.
    
    Runs on CPU to preserve T4 VRAM for the LLM. Legal-BERT is a
    domain-specific BERT model pre-trained on legal corpora, providing
    superior embedding quality for legal text retrieval.
    
    Returns:
        SentenceTransformer model instance (on CPU).
    """
    from sentence_transformers import SentenceTransformer
    
    logger.info("Loading embedder: %s (on CPU)...", EMBEDDER_NAME)
    
    try:
        embedder = SentenceTransformer(EMBEDDER_NAME, device="cpu")
    except Exception as e:
        logger.error("Failed to load embedder %s: %s", EMBEDDER_NAME, e)
        logger.info("Retrying embedder download...")
        # Retry once on HuggingFace download failure
        try:
            embedder = SentenceTransformer(EMBEDDER_NAME, device="cpu")
        except Exception as retry_e:
            logger.error("Retry failed: %s", retry_e)
            raise
    
    logger.info("Embedder loaded: %s (dim=%d)", EMBEDDER_NAME, 
                embedder.get_sentence_embedding_dimension())
    return embedder
