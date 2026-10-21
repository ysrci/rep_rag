import uuid
from pydantic import BaseModel, Field
from typing import List


class MinimalSource(BaseModel):
    """Source text span in a document.

    Attributes:
        file_path (str): Relative or absolute path to the source file.
        first_character_index (int): Starting character offset of the span.
        last_character_index (int): Ending character offset of the span.
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(MinimalSource):
    """Indexed text chunk with content and document location.

    Attributes:
        file_path (str): Relative or absolute path to the source file.
        first_character_index (int): Starting character offset of the chunk.
        last_character_index (int): Ending character offset of the chunk.
        text (str): Extracted text content of the chunk.
    """
    text: str


class UnansweredQuestion(BaseModel):
    """Query without associated answers or ground truth.

    Attributes:
        question_id (str): Unique identifier for the question.
        question (str): Text of the question.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question paired with ground-truth sources and reference answer.

    Attributes:
        question_id (str): Unique identifier for the question.
        question (str): Text of the question.
        sources (List[MinimalSource]): Ground-truth source references.
        answer (str): Ground-truth answer text.
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Container holding a collection of evaluation questions.

    Attributes:
        rag_questions (List[AnsweredQuestion | UnansweredQuestion]): List of
            benchmark questions.
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results containing retrieved source spans for a question.

    Attributes:
        question_id (str): Unique identifier of the question.
        question (str): Text of the question.
        retrieved_sources (List[MinimalSource]): Retrieved source locations.
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Generated answer bundled with question and retrieved sources.

    Attributes:
        question_id (str): Unique identifier of the question.
        question (str): Text of the question.
        retrieved_sources (List[MinimalSource]): Retrieved source locations.
        answer (str): Generated answer text.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """Submission format containing search results for evaluation.

    Attributes:
        search_results (List[MinimalSearchResults]): Search results per
            question.
        k (int): Number of top results retrieved per question.
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Submission format containing search results and generated answers.

    Attributes:
        search_results (List[MinimalAnswer]): List of answers and sources
            per question.
        k (int): Number of top results retrieved per question.
    """
    search_results: List[MinimalAnswer]
    k: int
