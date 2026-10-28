# LexAgent API on a Hugging Face Docker Space (CPU).
# The language model is remote (FreeLLMAPI); this container runs retrieval, the citation
# checker (Legal-BERT + NLI) and the debate graph, and serves the web app as a fallback.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt huggingface_hub

ENV HF_HOME=/app/.cache/huggingface \
    LEXAGENT_DATA_DIR=/app/data \
    LEXAGENT_LLM_BACKEND=api \
    LEXAGENT_LLM_BASE_URL=https://llm.velocitytechnologies.in/v1 \
    LEXAGENT_LLM_MODEL=gpt-oss-120b \
    LEXAGENT_ALLOWED_ORIGINS=https://lexagent.velocitytechnologies.in \
    PYTHONUNBUFFERED=1

# Bake the two local models into the image so a cold start does not download them.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('nlpaueb/legal-bert-base-uncased'); CrossEncoder('cross-encoder/nli-deberta-v3-small')"

# The prebuilt corpus (BM25, ChromaDB, case catalog). Bump DATA_REVISION to pull a new build.
ARG DATA_REPO=iamadityatakuli/lexagent-corpus
ARG DATA_REVISION=main
RUN hf download "$DATA_REPO" --repo-type dataset --revision "$DATA_REVISION" --local-dir /app/data \
    && rm -rf /app/data/.cache

COPY . .
RUN chown -R user:user /app
USER user

EXPOSE 7860
CMD ["python", "ui/server.py", "--host", "0.0.0.0", "--port", "7860", "--preload"]
