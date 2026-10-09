# LexAgent v3.0 | runs/build_retrieval_v2.py
"""Build the second retrieval index (see src/retrieval/retrieval_v2.py). Nothing existing is changed.

    python runs/build_retrieval_v2.py --lexical     # BM25 over word tokens (CPU, about 3 minutes)
    python runs/build_retrieval_v2.py --dense       # passage embeddings (GPU, about 5 minutes)
    python runs/build_retrieval_v2.py               # both

Each part is skipped when its file exists and was built from the current corpus; --overwrite
rebuilds it. Files are written to a temporary name and renamed, so a stopped build leaves no
half-written index. The Legal-BERT collection, the existing BM25 index and the corpus are only read.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

import config  # noqa: E402
from src.retrieval.retrieval_v2 import (  # noqa: E402
    BM25_FILE, EMBEDDINGS_FILE, MANIFEST_FILE, corpus_fingerprint, index_paths, load_manifest, word_tokens,
)


def write_manifest(directory: str, part: str, record: dict) -> None:
    manifest = load_manifest(directory)
    manifest[part] = record
    path = index_paths(directory)[MANIFEST_FILE]
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def up_to_date(directory: str, part: str, file_name: str, fingerprint: dict, extra: dict = None) -> bool:
    record = load_manifest(directory).get(part) or {}
    same = record.get("corpus") == fingerprint and all(record.get(k) == v for k, v in (extra or {}).items())
    return same and os.path.exists(index_paths(directory)[file_name])


def build_lexical(corpus, directory: str, fingerprint: dict) -> None:
    from src.data.corpus_schema import CorpusDocument
    from src.retrieval.bm25_retriever import BM25Retriever

    began = time.time()
    word_corpus = [CorpusDocument(doc_id=d.doc_id, text=word_tokens(d.text), source=d.source, case_name=d.case_name) for d in corpus]
    index = BM25Retriever(word_corpus)
    target = index_paths(directory)[BM25_FILE]
    index.save_index(target + ".tmp")
    os.replace(target + ".tmp", target)
    write_manifest(directory, "lexical", {"tokenisation": "lower-cased runs of [a-z0-9]", "corpus": fingerprint,
                                          "vocabulary": len(index.inverted_index), "seconds": round(time.time() - began, 1),
                                          "built": datetime.now().isoformat(timespec="seconds")})
    print(f"  lexical index written: {target} ({os.path.getsize(target) / 1e6:.0f} MB, {time.time() - began:.0f}s)")


def build_dense(corpus, directory: str, fingerprint: dict) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    began = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.RETRIEVAL_V2_MODEL, device=device)
    model.max_seq_length = config.RETRIEVAL_V2_MAX_TOKENS
    if device == "cuda":
        model.half()
    texts = [d.text for d in corpus]
    target = index_paths(directory)[EMBEDDINGS_FILE]
    matrix = None
    for start in range(0, len(texts), 20000):
        block = model.encode(texts[start:start + 20000], batch_size=256 if device == "cuda" else 32, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        if matrix is None:
            matrix = np.lib.format.open_memmap(target + ".tmp", mode="w+", dtype=np.float16, shape=(len(texts), block.shape[1]))
        matrix[start:start + len(block)] = block
        if start % 100000 == 0:
            print(f"    {start:,} of {len(texts):,} passages embedded, {time.time() - began:.0f}s")
    matrix.flush()
    dimensions = int(matrix.shape[1])
    del matrix
    os.replace(target + ".tmp", target)
    write_manifest(directory, "dense", {"model": config.RETRIEVAL_V2_MODEL, "max_tokens": config.RETRIEVAL_V2_MAX_TOKENS,
                                        "query_prefix": config.RETRIEVAL_V2_QUERY_PREFIX, "dimensions": dimensions,
                                        "normalised": True, "dtype": "float16", "corpus": fingerprint, "device": device,
                                        "seconds": round(time.time() - began, 1), "built": datetime.now().isoformat(timespec="seconds")})
    print(f"  dense index written: {target} ({os.path.getsize(target) / 1e6:.0f} MB, {time.time() - began:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the second retrieval index")
    parser.add_argument("--lexical", action="store_true", help="build the word-token BM25 index")
    parser.add_argument("--dense", action="store_true", help="build the passage embeddings")
    parser.add_argument("--overwrite", action="store_true", help="rebuild a part that is already up to date")
    arguments = parser.parse_args()
    both = not (arguments.lexical or arguments.dense)

    from src.data.build_corpus import load_corpus

    directory = config.RETRIEVAL_V2_DIR
    os.makedirs(directory, exist_ok=True)
    corpus = load_corpus(config.CORPUS_JSON_PATH)
    fingerprint = corpus_fingerprint(corpus)
    print(f"corpus: {fingerprint['passages']:,} passages; index directory: {directory}")
    if arguments.lexical or both:
        if up_to_date(directory, "lexical", BM25_FILE, fingerprint) and not arguments.overwrite:
            print("  lexical index is up to date; skipped")
        else:
            build_lexical(corpus, directory, fingerprint)
    if arguments.dense or both:
        if up_to_date(directory, "dense", EMBEDDINGS_FILE, fingerprint, {"model": config.RETRIEVAL_V2_MODEL}) and not arguments.overwrite:
            print("  dense index is up to date; skipped")
        else:
            build_dense(corpus, directory, fingerprint)


if __name__ == "__main__":
    main()
