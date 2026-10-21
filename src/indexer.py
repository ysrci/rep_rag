import bm25s
import pathlib
import re
import faiss
import json
from typing import Any
from sentence_transformers import SentenceTransformer
from .settings import Settings
from .chunker import loader
from .models import Chunk


def _txt_processor(text: str) -> str:
    """Normalize text for BM25 tokenization.

    Replaces underscores with spaces, separates camelCase words, and converts
    characters to lowercase.

    Args:
        text (str): Raw input text to normalize.

    Returns:
        str: Processed and normalized text string.
    """
    text = text.replace('_', ' ')
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return text.lower()


def _embedder(corpus: list[str]) -> Any:
    """Generate dense vector embeddings for a corpus of text chunks.

    Args:
        corpus (list[str]): List of chunk text strings to embed.

    Returns:
        Any: Array of dense vector embeddings for the corpus.
    """
    embedder = SentenceTransformer(Settings.EMBEDDING_MODEL)
    embeddings = embedder.encode(
        corpus,
        show_progress_bar=True,
        batch_size=Settings.EMBEDDING_BATCH_SIZE
    )
    return embeddings


def indexer(
    repo: str,
    save_path: str | pathlib.Path,
    max_chunk_size: int
) -> None:
    """Build and persist BM25 and FAISS indices from repository documents.

    Args:
        repo (str): Path to the directory containing raw source files.
        save_path (str | pathlib.Path): Directory path where index artifacts
            will be saved.
        max_chunk_size (int): Maximum character length for each document chunk.

    Returns:
        None: Does not return a value.

    Raises:
        ValueError: If no indexable chunks are found in the repository.
    """
    chunks: list[Chunk] = loader(repo, max_chunk_size)
    if not chunks:
        raise ValueError("No indexable chunks found.")
    faiss_corpus = [chunk.text for chunk in chunks]
    bm25_corpus = [_txt_processor(chunk.text) for chunk in chunks]

    corpus_tokens = bm25s.tokenize(bm25_corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    retriever.save(f'{save_path}/{Settings.BM25_NAME}')

    embeddings = _embedder(faiss_corpus)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f'{save_path}/{Settings.FAISS_NAME}')

    with open(f"{save_path}/{Settings.CHUNKS_NAME}", "w") as file:
        json.dump([d.model_dump() for d in chunks], file, indent=2)
