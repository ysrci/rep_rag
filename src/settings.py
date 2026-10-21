"""All project settings live in one class."""

from pathlib import Path


class Settings:
    """Defaults and configuration settings for the RAG pipeline.

    This class is not intended to be instantiated; all members are class-level
    constants or static helper methods.
    """
    RAW_DIR: str = "data/raw"
    PROCESSED_DIR: str = "data/processed"
    SEARCH_OUTPUT_DIR: str = "data/output/search_results"
    ANSWER_OUTPUT_DIR: str = "data/output/search_results_and_answer"

    BM25_NAME: str = "bm25_index"
    FAISS_NAME: str = "faiss_index"
    CHUNKS_NAME: str = "chunks.json"

    MAX_CHUNK_SIZE: int = 2000
    DEFAULT_CHUNK_SIZE: int = 2000
    EXTENSIONS: tuple[str, ...] = (
        ".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml"
    )
    IGNORED_DIRS: tuple[str, ...] = (
        ".git", "__pycache__", ".mypy_cache", ".venv"
    )

    DEFAULT_K: int = 5
    RRF_K: int = 60
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64

    LLM_MODEL: str = "Qwen/Qwen3-0.6B"
    MAX_NEW_TOKENS: int = 256

    @staticmethod
    def index_files(directory: str | Path) -> list[Path]:
        """Return the file paths that make up a complete index.

        Args:
            directory (str | Path): Base directory where index files reside.

        Returns:
            list[Path]: List of expected index file paths.
        """
        base = Path(directory)
        return [
            base / Settings.BM25_NAME,
            base / Settings.FAISS_NAME,
            base / Settings.CHUNKS_NAME,
        ]

    @staticmethod
    def missing_index_files(directory: str | Path) -> list[Path]:
        """Return the index file paths that do not exist yet.

        Args:
            directory (str | Path): Base directory to inspect for index files.

        Returns:
            list[Path]: List of non-existent index file paths.
        """
        return [p for p in Settings.index_files(directory) if not p.exists()]

    @staticmethod
    def index_exists(directory: str | Path) -> bool:
        """Check whether all index files exist in the specified directory.

        Args:
            directory (str | Path): Base directory to check.

        Returns:
            bool: True if all index files exist, False otherwise.
        """
        return not Settings.missing_index_files(directory)

    @staticmethod
    def is_indexable(path: Path) -> bool:
        """Determine whether a file has a supported indexable extension.

        Args:
            path (Path): File path to verify.

        Returns:
            bool: True if file extension is in EXTENSIONS, False otherwise.
        """
        return path.suffix.lower() in Settings.EXTENSIONS

    @staticmethod
    def is_ignored(path: Path) -> bool:
        """Determine whether any directory in the path is ignored.

        Args:
            path (Path): Path to evaluate.

        Returns:
            bool: True if any component is in IGNORED_DIRS, False otherwise.
        """
        return any(part in Settings.IGNORED_DIRS for part in path.parts)

    @staticmethod
    def validate_chunk_size(size: int) -> int:
        """Validate that chunk size is a positive integer within bounds.

        Args:
            size (int): Chunk size in characters to validate.

        Returns:
            int: The validated chunk size.

        Raises:
            ValueError: If size is not an integer, is non-positive, or exceeds
                MAX_CHUNK_SIZE.
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("max_chunk_size must be a positive integer")
        if size > Settings.MAX_CHUNK_SIZE:
            raise ValueError(
                f"max_chunk_size must be <= {Settings.MAX_CHUNK_SIZE}"
            )
        return size

    @staticmethod
    def ensure_raw_not_empty(raw: str) -> None:
        """Validate that the raw directory exists.

        Args:
            raw (str): Path to the raw data directory.

        Returns:
            None: Does not return a value.

        Raises:
            FileNotFoundError: If the raw directory does not exist.
        """
        raw_dir = Path(raw)

        if not raw_dir.is_dir():
            raise FileNotFoundError(
                f"Raw directory '{raw_dir}' does not exist."
            )

    @staticmethod
    def validate_k(k: int) -> int:
        """Validate that retrieval count k is a positive integer.

        Args:
            k (int): Number of top results to retrieve.

        Returns:
            int: The validated k value.

        Raises:
            ValueError: If k is not an integer or is non-positive.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        return k
