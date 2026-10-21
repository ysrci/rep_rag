import bm25s
import faiss
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from sentence_transformers import SentenceTransformer
from .models import Chunk
from .settings import Settings


@lru_cache
def bm25_index_loader(index_path: str) -> Any:
    """Load and cache a BM25 index from disk.

    Args:
        index_path (str): File path to the saved BM25 index.

    Returns:
        Any: Loaded BM25 index instance.
    """
    return bm25s.BM25.load(index_path)


@lru_cache
def faiss_index_loader(index_path: str) -> Any:
    """Load and cache a FAISS index from disk.

    Args:
        index_path (str): File path to the saved FAISS index.

    Returns:
        Any: Loaded FAISS index instance.
    """
    return faiss.read_index(index_path)


@lru_cache
def chunks_loader(chunks_dir: str | Path) -> Any:
    """Load and cache chunk metadata and contents from JSON.

    Args:
        chunks_dir (str | Path): Directory containing the chunks JSON file.

    Returns:
        Any: Deserialized list of chunk dictionaries.
    """
    with open(f'{chunks_dir}/{Settings.CHUNKS_NAME}') as file:
        return json.load(file)


@lru_cache
def get_embedder() -> SentenceTransformer:
    """Instantiate and cache the SentenceTransformer embedding model.

    Returns:
        SentenceTransformer: Loaded sentence embedding model.
    """
    return SentenceTransformer(Settings.EMBEDDING_MODEL)


def _txt_processor(text: str) -> str:
    """Normalize text for BM25 query tokenization.

    Args:
        text (str): Raw input text to normalize.

    Returns:
        str: Processed and normalized text string.
    """
    text = text.replace('_', ' ')
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return text.lower()


def bm25_retriever(query: str, k: int, path: str | Path) -> list[int]:
    """Retrieve top-k chunk indices using BM25 sparse search.

    Args:
        query (str): Search query text.
        k (int): Number of top results to retrieve.
        path (str | Path): Directory containing the index files.

    Returns:
        list[int]: Ranked list of chunk indices matching the query.
    """
    bm25_index = bm25_index_loader(f'{path}/{Settings.BM25_NAME}')

    processed_query = _txt_processor(query)
    tokenized_query = bm25s.tokenize([processed_query])
    docs_i, _scores = bm25_index.retrieve(tokenized_query, k=k)

    return [int(x) for x in docs_i[0]]


def faiss_retriever(query: str, k: int, path: str | Path) -> list[int]:
    """Retrieve top-k chunk indices using dense FAISS vector search.

    Args:
        query (str): Search query text.
        k (int): Number of top results to retrieve.
        path (str | Path): Directory containing the index files.

    Returns:
        list[int]: Ranked list of valid chunk indices matching the query.
    """
    embedder = get_embedder()
    embedded_query = embedder.encode([query])

    index = faiss_index_loader(f'{path}/{Settings.FAISS_NAME}')
    _scores, docs_i = index.search(embedded_query, k)

    return [int(x) for x in docs_i[0] if x >= 0]


def hybrid_search_retriever(
    query: str,
    k: int,
    path: str | Path,
    rrf_k: int = Settings.RRF_K
) -> list[int]:
    """Combine BM25 and FAISS results using Reciprocal Rank Fusion (RRF).

    Args:
        query (str): Search query text.
        k (int): Number of top fused results to return.
        path (str | Path): Directory containing index files.
        rrf_k (int): Smoothing constant for reciprocal rank scoring.

    Returns:
        list[int]: Top-k chunk indices ranked by combined RRF score.
    """
    top_k_bm25 = bm25_retriever(query, k, path)
    top_k_faiss = faiss_retriever(query, k, path)

    rrf_scores: dict[int, float] = {}
    for rank, chunk_index in enumerate(top_k_bm25, start=1):
        rrf_scores[chunk_index] = (
            rrf_scores.get(chunk_index, 0) + 1 / (rrf_k + rank)
        )
    for rank, chunk_index in enumerate(top_k_faiss, start=1):
        rrf_scores[chunk_index] = (
            rrf_scores.get(chunk_index, 0) + 1 / (rrf_k + rank)
        )

    ranked_keys = sorted(
        rrf_scores,
        key=lambda k: rrf_scores[k],
        reverse=True
    )

    return ranked_keys[:k]


def searcher(query: str, k: int, path: str | Path) -> list[Chunk]:
    """Perform hybrid search and retrieve matching Chunk objects.

    Args:
        query (str): Search query text.
        k (int): Number of top chunks to return.
        path (str | Path): Directory containing the index and chunks.json.

    Returns:
        list[Chunk]: List of retrieved Chunk models.
    """
    indexes = hybrid_search_retriever(query, k, path)
    chunks = chunks_loader(path)
    res: list[Chunk] = []

    for index in indexes:
        chunk = chunks[index]
        res.append(
            Chunk(
                file_path=chunk['file_path'],
                first_character_index=chunk['first_character_index'],
                last_character_index=chunk['last_character_index'],
                text=chunk['text']
                )
            )
    return res
