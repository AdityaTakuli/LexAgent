# LexAgent v3.0 | load_model.py
"""Model loading for Mistral-7B (NF4 4-bit) and Legal-BERT embedder.

Mistral-7B is loaded in 4-bit NF4 quantization to fit within T4 16GB VRAM.
Legal-BERT embedder runs on CPU or GPU.
"""

import os
import sys
try:
    import torch
except ImportError:
    torch = None
from typing import Tuple, Any

from config import MODEL_NAME, EMBEDDER_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_mistral_7b() -> Tuple[Any, Any, Any]:
    """Load Mistral-7B-Instruct-v0.3 in NF4 4-bit quantization.
    
    Uses BitsAndBytes NF4 quantization with double quantization for
    optimal memory efficiency on T4 16GB. Expected VRAM: ~5.2GB.
    
    Returns:
        Tuple of (model, tokenizer, llm_pipeline).
    """
    use_mock = os.environ.get("LEXAGENT_MOCK_LLM", "0") == "1"

    cuda_available = torch is not None and torch.cuda.is_available()
    if not cuda_available and not use_mock:
        logger.error(
            "CUDA is not available. Mistral-7B requires a GPU. "
            "If testing locally without GPU, set LEXAGENT_MOCK_LLM=1."
        )
        raise RuntimeError(
            "CUDA not available. LexAgent requires a T4 GPU (Colab free tier) "
            "or set LEXAGENT_MOCK_LLM=1 for CPU mock testing."
        )

    if use_mock or not cuda_available:
        logger.warning("Initializing Mock LLM pipeline for CPU/testing environment...")
        class MockPipeline:
            def __call__(self, prompt, **kwargs):
                return [{"generated_text": (
                    "Based on Supreme Court and CAP precedent (e.g. 410 U.S. 113), "
                    "the applicable legal standard confirms that constitutional protections apply. "
                    "[CITATION: Roe v. Wade, 410 U.S. 113 (1973)]"
                )}]
            def invoke(self, prompt, **kwargs):
                return self(prompt, **kwargs)
        mock_pipe = MockPipeline()
        return None, None, mock_pipe

    precision = os.environ.get("LEXAGENT_PRECISION", "auto").lower()
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda_available else 0
    is_offline = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1" or os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )
    from langchain_community.llms import HuggingFacePipeline

    # High-End GPU Support: if VRAM >= 23GB (e.g. A100, H100, RTX 4090, A6000), load in native bfloat16
    if precision in ("bfloat16", "bf16") or (precision == "auto" and total_vram_gb >= 23.0):
        target_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        logger.info(
            "High-VRAM GPU detected (%.1f GB). Loading %s in unquantized %s for maximum generation quality & speed...",
            total_vram_gb, MODEL_NAME, target_dtype
        )
        model_kwargs = {"torch_dtype": target_dtype}
    elif precision in ("float16", "fp16"):
        logger.info("Loading %s in float16 precision...", MODEL_NAME)
        model_kwargs = {"torch_dtype": torch.float16}
    else:
        logger.info("Loading %s with NF4 4-bit quantization (VRAM target: ~5.2 GB)...", MODEL_NAME)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs = {"quantization_config": bnb_config}

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=is_offline,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=is_offline,
        **model_kwargs,
    )

    vram_gb = torch.cuda.memory_allocated() / 1e9
    logger.info("Model loaded. VRAM used: %.2f GB", vram_gb)

    max_new_tokens = int(os.environ.get("LEXAGENT_MAX_NEW_TOKENS", "1024"))

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = max_new_tokens
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "config") and model.config is not None:
        model.config.max_length = None

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
    )

    llm_pipeline = HuggingFacePipeline(pipeline=text_pipeline)
    logger.info("LLM pipeline ready.")
    return model, tokenizer, llm_pipeline


def load_embedder() -> Any:
    """Load Legal-BERT embedder for dense retrieval."""
    use_mock = os.environ.get("LEXAGENT_MOCK_LLM", "0") == "1"

    try:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        logger.info("Loading embedder: %s (on %s)...", EMBEDDER_NAME, device)
        embedder = SentenceTransformer(EMBEDDER_NAME, device=device)
        logger.info("Embedder loaded: %s (dim=%d)", EMBEDDER_NAME,
                    embedder.get_sentence_embedding_dimension())
        return embedder
    except Exception as e:
        if use_mock:
            logger.warning("Using MockEmbedder for testing environment (%s)", e)
            import numpy as np
            class MockEmbedder:
                def encode(self, texts, **kwargs):
                    if isinstance(texts, str):
                        return np.zeros(768, dtype=np.float32)
                    return np.zeros((len(texts), 768), dtype=np.float32)
                def get_sentence_embedding_dimension(self):
                    return 768
            return MockEmbedder()
        raise e
