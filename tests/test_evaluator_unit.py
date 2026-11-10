
import pytest

from src.evaluator import RAGEvaluator


@pytest.fixture
def evaluator():
    return RAGEvaluator()


@pytest.fixture
def evidence():
    return [
        {
            "content": (
                "RAG combines information retrieval with large language models. "
                "RAG can use external documents to answer questions. "
                "RAG cannot guarantee 100% answer accuracy."
            ),
            "metadata": {"source": "sample.pdf", "page": 1},
            "distance": 0.5
        }
    ]


def test_extract_statements(evaluator):
    answer = (
        "RAG combines retrieval with language models. "
        "RAG uses external documents."
    )

    statements = evaluator.extract_statements(answer)

    assert len(statements) == 2


def test_supported_statement(evaluator, evidence):
    results = evaluator.verify_statements(
        ["RAG combines information retrieval with language models."],
        evidence
    )

    assert results[0]["supported"] is True


def test_unsupported_statement(evaluator, evidence):
    results = evaluator.verify_statements(
        ["RAG systems use blockchain technology for storage."],
        evidence
    )

    assert results[0]["supported"] is False


def test_contradictory_statement(evaluator, evidence):
    results = evaluator.verify_statements(
        ["RAG cannot use external documents."],
        evidence
    )

    assert results[0]["supported"] is False
    assert results[0]["contradiction_detected"] is True


def test_empty_answer(evaluator):
    with pytest.raises(ValueError):
        evaluator.extract_statements("")


def test_empty_question(evaluator, evidence):
    with pytest.raises(ValueError):
        evaluator.evaluate_faithfulness(
            question="",
            answer="RAG uses external documents.",
            retrieved_documents=evidence
        )


def test_empty_evidence(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate_faithfulness(
            question="What is RAG?",
            answer="RAG uses external documents.",
            retrieved_documents=[]
        )
