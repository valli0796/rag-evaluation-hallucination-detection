
from src.vector_store import VectorStoreManager
from langchain_core.documents import Document


def test_chromadb_add_and_search(tmp_path):
    manager = VectorStoreManager(
        persist_directory=str(tmp_path / "chroma_db"),
        collection_name="integration_test"
    )

    documents = [
        Document(
            page_content=(
                "Retrieval-Augmented Generation combines "
                "information retrieval with large language models."
            ),
            metadata={
                "source": "test.pdf",
                "page": 1,
                "chunk_id": 0
            }
        ),
        Document(
            page_content=(
                "Vector databases store numerical embeddings "
                "and support similarity search."
            ),
            metadata={
                "source": "test.pdf",
                "page": 1,
                "chunk_id": 1
            }
        )
    ]

    result = manager.add_documents(documents)

    assert result["documents_added"] == 2
    assert manager.get_collection_count() == 2

    search_results = manager.search(
        query="What is Retrieval-Augmented Generation?",
        k=2
    )

    assert len(search_results) > 0
    assert any(
        "Retrieval-Augmented Generation" in item["content"]
        for item in search_results
    )

    assert all("distance" in item for item in search_results)
    assert all("metadata" in item for item in search_results)
