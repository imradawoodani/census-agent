"""
Cohere embedding client.
Handles batch embedding and cosine similarity utilities.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cohere
import numpy as np

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)
_client: Optional[cohere.Client] = None

EMBED_MODEL = "embed-english-v3.0"
BATCH_SIZE = 96  # Cohere max per call


def _get_client() -> cohere.Client:
    global _client
    if _client is None:
        _client = cohere.Client(api_key=settings.cohere_api_key)
    return _client


def _embed_batch_sync(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed in batches synchronously."""
    client = _get_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embed(
            texts=batch,
            model=EMBED_MODEL,
            input_type=input_type,
        )
        all_embeddings.extend(resp.embeddings)
    return all_embeddings


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of documents (for indexing)."""
    if not texts:
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _embed_batch_sync, texts, "search_document"
    )


async def embed_query(text: str) -> list[float]:
    """Embed a single query (for retrieval)."""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        _executor, _embed_batch_sync, [text], "search_query"
    )
    return results[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr)) / (norm_a * norm_b)


def top_k(
    query_embedding: list[float],
    candidates: list[dict],
    k: int,
    min_score: float = 0.0,
) -> list[dict]:
    """Return the top-k candidates by cosine similarity."""
    scored = [
        {**item, "_score": cosine_similarity(query_embedding, item["embedding"])}
        for item in candidates
        if cosine_similarity(query_embedding, item["embedding"]) >= min_score
    ]
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:k]
