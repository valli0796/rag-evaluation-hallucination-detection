
from types import SimpleNamespace

import pytest

from src.rag_pipeline import RAGPipeline


class FakeVectorStore:
    def __init__(self):
        self.search_called = False
        self.query = None
        self.k = None

    def search(self, query, k):
        self.search_called = True
        self.query = query
        self.k = k

        return [
            {
                "content": "RAG combines retrieval with language models.",
                "metadata": {"source": "sample.pdf", "page": 1},
                "distance": 0.5
            }
        ]


class FakeLLM:
    def __init__(self, model, temperature, num_predict):
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return SimpleNamespace(
            content="RAG uses retrieved information to generate answers."
        )


def create_pipeline(monkeypatch):
    monkeypatch.setattr(
        "src.rag_pipeline.ChatOllama",
        FakeLLM
    )

    vector_store = FakeVectorStore()

    pipeline = RAGPipeline(
        k=3,
        vector_store_manager=vector_store
    )

    return pipeline, vector_store


def test_retrieve_context(monkeypatch):
    pipeline, vector_store = create_pipeline(monkeypatch)

    results = pipeline.retrieve_context("What is RAG?")

    assert vector_store.search_called
    assert vector_store.query == "What is RAG?"
    assert vector_store.k == 3
    assert len(results) == 1


def test_generate_answer(monkeypatch):
    pipeline, vector_store = create_pipeline(monkeypatch)

    retrieved_documents = vector_store.search(
        "What is RAG?",
        k=3
    )

    answer = pipeline.generate_answer(
        "What is RAG?",
        retrieved_documents
    )

    assert answer == (
        "RAG uses retrieved information to generate answers."
    )
    assert "RAG combines retrieval" in pipeline.llm.prompt


def test_generate_answer_without_context(monkeypatch):
    pipeline, vector_store = create_pipeline(monkeypatch)

    answer = pipeline.generate_answer(
        "What is RAG?",
        []
    )

    assert "could not find relevant information" in answer
    assert pipeline.llm.prompt is None


def test_ask_pipeline(monkeypatch):
    pipeline, vector_store = create_pipeline(monkeypatch)

    result = pipeline.ask("What is RAG?")

    assert result["question"] == "What is RAG?"
    assert result["answer"] == (
        "RAG uses retrieved information to generate answers."
    )
    assert len(result["retrieved_documents"]) == 1


def test_empty_question(monkeypatch):
    pipeline, vector_store = create_pipeline(monkeypatch)

    with pytest.raises(ValueError):
        pipeline.ask("")


def test_whitespace_question(monkeypatch):
    pipeline, vector_store = create_pipeline(monkeypatch)

    with pytest.raises(ValueError):
        pipeline.ask("   ")
