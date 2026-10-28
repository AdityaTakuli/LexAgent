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

import re
import time

from config import MODEL_NAME, EMBEDDER_NAME, LLM_BACKEND, LLM_API_BASE_URL, LLM_API_KEY, LLM_API_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

_INST_TAGS = re.compile(r"\s*\[/?INST\]\s*")


class OpenAICompatibleLLM:
    """A chat model behind an OpenAI-compatible API (FreeLLMAPI, OpenRouter, vLLM, ...).

    Drop-in for the local pipeline: invoke(prompt) returns the reply text. The prompts are
    written in Mistral's [INST] format, so the tags are stripped before sending.
    """

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.1,
                 max_tokens: int = 1536, timeout: float = 180.0, max_retries: int = 4) -> None:
        import requests

        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = os.environ.get("LEXAGENT_LLM_REASONING_EFFORT", "low").strip()

    def _body(self, text: str, with_reasoning: bool) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Follow the instructions exactly. When asked for JSON, reply with one JSON object and nothing else."},
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if with_reasoning and self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def invoke(self, prompt: str, **_: Any) -> str:
        text = _INST_TAGS.sub("\n", str(prompt)).strip()
        with_reasoning = True
        last_error = ""
        rate_waits = [3, 6, 12, 20]  # a free tier's per-minute limit usually resets within this
        attempt = 0
        while True:
            wait = None
            try:
                response = self._session.post(self.url, json=self._body(text, with_reasoning), timeout=self.timeout)
            except Exception as e:  # network error: retry
                last_error = f"{type(e).__name__}: {e}"
            else:
                if response.status_code == 200:
                    data = response.json()
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    content = (message.get("content") or "").strip()
                    if not content or choice.get("finish_reason") == "length":
                        logger.warning("LLM reply %s (finish_reason=%s, usage=%s, %d chars)",
                                       "empty" if not content else "cut off at max_tokens",
                                       choice.get("finish_reason"), data.get("usage"), len(content))
                    return content
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code == 400 and with_reasoning:
                    with_reasoning = False  # some providers reject reasoning_effort; retry without it
                    continue
                if response.status_code in (401, 403, 404):
                    raise RuntimeError(f"LLM API refused the request ({last_error}). Check LEXAGENT_LLM_API_KEY / LEXAGENT_LLM_MODEL.")
                if response.status_code == 429:
                    if not rate_waits:
                        break
                    try:
                        wait = min(float(response.headers.get("Retry-After", "")), 30.0)
                    except ValueError:
                        wait = None
                    planned = rate_waits.pop(0)
                    wait = wait if wait and wait > 0 else planned
                    logger.info("LLM rate-limited; waiting %.0fs before retrying", wait)
            attempt += 1
            if wait is None:
                if attempt > self.max_retries:
                    break
                wait = min(2 ** (attempt - 1), 20)
            time.sleep(wait)
        raise RuntimeError(f"LLM API call failed: {last_error}")

    __call__ = invoke


def load_api_llm() -> OpenAICompatibleLLM:
    """The language model behind an OpenAI-compatible endpoint (see LEXAGENT_LLM_* in config.py)."""
    if not LLM_API_KEY:
        raise RuntimeError("LEXAGENT_LLM_API_KEY is not set. Put your FreeLLMAPI unified key in V3_with_CAP_as_main/.env.")
    llm = OpenAICompatibleLLM(
        LLM_API_BASE_URL, LLM_API_KEY, LLM_API_MODEL,
        max_tokens=int(os.environ.get("LEXAGENT_LLM_MAX_TOKENS", "1536")),
    )
    logger.info("Using API model %s at %s", LLM_API_MODEL, LLM_API_BASE_URL)
    return llm


def load_mistral_7b() -> Tuple[Any, Any, Any]:
    """Load the generator. With LLM_BACKEND="api" this returns the API client instead of Mistral.

    Local mode loads Mistral-7B-Instruct-v0.3 in NF4 4-bit quantization
    (BitsAndBytes, double quantization). Expected VRAM: ~5.2GB.

    Returns:
        Tuple of (model, tokenizer, llm_pipeline).
    """
    use_mock = os.environ.get("LEXAGENT_MOCK_LLM", "0") == "1"
    if LLM_BACKEND == "api" and not use_mock:
        return None, None, load_api_llm()

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
