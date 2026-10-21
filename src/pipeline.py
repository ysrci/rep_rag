import json
import os
from pathlib import Path
from typing import Any
from .settings import Settings
from .aug_generator import augmenter
from .models import (
    Chunk,
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from rich import print
from tqdm import tqdm
from .indexer import indexer
from .retriever import searcher


class Pipeline:
    """Command-line interface and orchestration pipeline for RAG operations."""

    def _ensure_index(self) -> Path:
        """Verify index existence or trigger indexing if missing.

        Returns:
            Path: Directory path containing the verified index files.

        Raises:
            RuntimeError: If indexing does not produce a usable index.
        """
        processed = Path(Settings.PROCESSED_DIR)

        if not Settings.index_exists(processed):
            print("[yellow]Indices not found. Starting indexing...[/]")
            indexer(
                Settings.RAW_DIR,
                processed,
                Settings.DEFAULT_CHUNK_SIZE,
            )
        else:
            print(
                "[dim]Using existing indices. "
                "Re-index if the source documents or configuration changed.[/]"
            )

        if not Settings.index_exists(processed):
            raise RuntimeError(
                f"Indexing did not produce a usable index at {processed}"
            )

        return processed

    def index(self, max_chunk_size: int = Settings.DEFAULT_CHUNK_SIZE) -> None:
        """Build BM25 and FAISS search indices from raw documents.

        Args:
            max_chunk_size (int): Maximum character size for each chunk.

        Returns:
            None: Does not return a value.
        """
        Settings.validate_chunk_size(max_chunk_size)
        Settings.ensure_raw_not_empty(Settings.RAW_DIR)
        processed = Settings.PROCESSED_DIR
        indexer(Settings.RAW_DIR, processed, max_chunk_size)
        print(
            f"[green]Ingestion complete! Indices saved under "
            f"[blue]{processed}[/]"
        )

    def search(self, query: str, k: int = Settings.DEFAULT_K) -> None:
        """Search the index for relevant chunks and print their locations.

        Args:
            query (str): Search query text.
            k (int): Number of top results to retrieve.

        Returns:
            None: Does not return a value.
        """
        Settings.validate_k(k)

        if not query.strip():
            print("[yellow]Query cannot be empty.[/]")
            return
        processed = self._ensure_index()
        chunks = searcher(query, k, processed)
        if not chunks:
            print("[yellow]No relevant sources found.[/]")
            return
        for chunk in chunks:
            print(
                f"[italic #a7de50]{chunk.file_path} "
                f"[[#5d95e3 bold]{chunk.first_character_index}:"
                f"{chunk.last_character_index}[/]]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = Settings.DEFAULT_K,
        save_directory: str = Settings.SEARCH_OUTPUT_DIR
    ) -> None:
        """Search across all dataset questions and save results to a JSON file.

        Args:
            dataset_path (str): Path to the input dataset JSON file.
            k (int): Number of top results to retrieve per question.
            save_directory (str): Destination directory for the output file.

        Returns:
            None: Does not return a value.
        """
        Settings.validate_k(k)
        processed = self._ensure_index()
        with open(dataset_path) as file:
            dataset = json.load(file)
        rag_questions = RagDataset.model_validate(dataset)

        search_res: list[MinimalSearchResults] = []
        for q in tqdm(rag_questions.rag_questions, desc="Searching dataset"):
            search_res.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=searcher(q.question, k, processed)
                )
            )
        studen_search_res = StudentSearchResults(
            search_results=search_res,
            k=k
        )

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        output_filename = Path(dataset_path).name

        with open(save_dir / output_filename, "w") as f:
            f.write(studen_search_res.model_dump_json(indent=4))
        print(
            f'[green]Saved student_search_results to '
            f'[blue]{save_dir}/{output_filename}[/]'
        )

    def answer(self, query: str, k: int = Settings.DEFAULT_K) -> None:
        """Retrieve relevant chunks and print an LLM-generated answer.

        Args:
            query (str): Question to answer.
            k (int): Number of top context chunks to retrieve.

        Returns:
            None: Does not return a value.
        """
        Settings.validate_k(k)
        if not query.strip():
            print("[yellow]Query cannot be empty.[/]")
            return
        processed = self._ensure_index()
        top_k_chunks = searcher(query, k, processed)
        answer = augmenter(query, top_k_chunks)
        print('\n', answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = Settings.ANSWER_OUTPUT_DIR
    ) -> None:
        """Generate answers for retrieved search results and save to JSON.

        Args:
            student_search_results_path (str): Path to search results JSON
                file.
            save_directory (str): Destination directory where answers are
                saved.

        Returns:
            None: Does not return a value.
        """
        with open(student_search_results_path) as file:
            ssr_data = json.load(file)
        ssr = StudentSearchResults.model_validate(ssr_data)

        lst_min_answers: list[MinimalAnswer] = []
        os.makedirs(save_directory, exist_ok=True)
        file_cache: dict[str, str] = {}
        for ss in tqdm(ssr.search_results, desc="Generating answers"):
            context: list[Chunk] = []

            for src in ss.retrieved_sources:
                if src.file_path not in file_cache:
                    try:
                        with open(src.file_path) as f:
                            file_cache[src.file_path] = f.read()
                    except OSError:
                        continue
                text = file_cache[src.file_path]
                start = src.first_character_index
                end = src.last_character_index

                if start < 0 or end <= start or end > len(text):
                    continue

                chunk_text = text[start:end]
                if not chunk_text.strip():
                    continue

                context.append(
                    Chunk(
                        **src.model_dump(),
                        text=chunk_text,
                    )
                )

            if context:
                answer = augmenter(ss.question, context)
            else:
                answer = (
                    "I couldn't generate an answer because no valid source "
                    "context was available."
                )

            lst_min_answers.append(
                MinimalAnswer(
                    question_id=ss.question_id,
                    question=ss.question,
                    retrieved_sources=ss.retrieved_sources,
                    answer=answer,
                )
            )
        out = StudentSearchResultsAndAnswer(
            search_results=lst_min_answers,
            k=ssr.k
        )
        output_path = Path(save_directory) / Path(
            student_search_results_path
        ).name
        with open(output_path, "w") as f:
            f.write(out.model_dump_json(indent=4))
        print(f"[green]Saved answers to [blue]{output_path}[/]")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        """Evaluate retrieval quality against ground truth using Recall@k.

        Args:
            student_search_results_path (str): Path to search results JSON
                file.
            dataset_path (str): Path to ground-truth dataset JSON file.

        Returns:
            None: Does not return a value.
        """
        with open(student_search_results_path) as f:
            student = json.load(f)["search_results"]
        with open(dataset_path) as f:
            truth = {
                q["question_id"]: q["sources"]
                for q in json.load(f)["rag_questions"]
                }

        def hit(a: dict[str, Any], b: dict[str, Any]) -> bool:
            """Evaluate whether candidate chunk overlaps ground truth by >= 5%.

            Args:
                a (dict[str, Any]): Candidate retrieved chunk dictionary.
                b (dict[str, Any]): Ground-truth source chunk dictionary.

            Returns:
                bool: True if chunks match and overlap >= 5%, False otherwise.
            """
            if a["file_path"] != b["file_path"]:
                return False
            inter = min(a["last_character_index"], b["last_character_index"]) \
                - max(a["first_character_index"], b["first_character_index"])
            if inter <= 0:
                return False
            la = a["last_character_index"] - a["first_character_index"]
            lb = b["last_character_index"] - b["first_character_index"]
            return bool(inter / (la + lb - inter) >= 0.05)

        print('[#66ccff]Evaluation Results[/]')
        for k in (1, 3, 5, 10):
            scores = []
            for r in student:
                gt = truth[r["question_id"]]
                if not gt:
                    continue
                top = r["retrieved_sources"][:k]
                scores.append(
                    sum(any(hit(t, s) for t in top) for s in gt) / len(gt)
                )
            if not scores:
                print("No questions to evaluate.")
                return
            res = sum(scores) / len(scores)
            print(
                f"[#00ffcc]Recall@[bold]{k}[/#00ffcc][/bold]: "
                f"[#00cccc]{res:.3f}[/#00cccc] ({res*100:.1f}%)"
            )
