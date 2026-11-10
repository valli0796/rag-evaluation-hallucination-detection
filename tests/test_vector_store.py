
from src.vector_store import VectorStoreManager
from langchain_core.documents import Document


def create_manager():
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.collection_name = "test_collection"
    return manager


def test_generate_unique_ids():
    manager = create_manager()

    documents = [
        Document(
            page_content="RAG combines retrieval and generation.",
            metadata={
                "source": "sample.pdf",
                "page": 1,
                "chunk_id": 0
            }
        ),
        Document(
            page_content="Embeddings represent text numerically.",
            metadata={
                "source": "sample.pdf",
                "page": 1,
                "chunk_id": 1
            }
        )
    ]

    ids = manager._generate_ids(documents)

    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_generate_consistent_ids():
    manager = create_manager()

    documents = [
        Document(
            page_content="RAG uses external documents.",
            metadata={
                "source": "sample.pdf",
                "page": 1,
                "chunk_id": 0
            }
        )
    ]

    first_ids = manager._generate_ids(documents)
    second_ids = manager._generate_ids(documents)

    assert first_ids == second_ids


def test_add_documents():
    manager = create_manager()

    class FakeVectorStore:
        def add_documents(self, documents, ids):
            self.documents = documents
            self.ids = ids

    manager.vector_store = FakeVectorStore()

    documents = [
        Document(
            page_content="RAG uses retrieved context.",
            metadata={
                "source": "sample.pdf",
                "page": 1,
                "chunk_id": 0
            }
        )
    ]

    result = manager.add_documents(documents)

    assert result["documents_added"] == 1
    assert result["collection_name"] == "test_collection"
    assert len(manager.vector_store.documents) == 1
    assert len(manager.vector_store.ids) == 1


def test_add_empty_documents():
    manager = create_manager()

    try:
        manager.add_documents([])
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_search_empty_query():
    manager = create_manager()

    try:
        manager.search("")
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_search_invalid_k():
    manager = create_manager()

    try:
        manager.search("What is RAG?", k=0)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
