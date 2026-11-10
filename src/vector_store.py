
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    def __init__(
        self,
        persist_directory="chroma_db",
        collection_name="rag_evaluation",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_directory = str(Path(persist_directory))
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 16
            }
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def _generate_ids(self, documents):
        ids = []

        for document in documents:
            content = document.page_content
            metadata = document.metadata

            unique_value = (
                f"{metadata.get('source', '')}_"
                f"{metadata.get('page', '')}_"
                f"{metadata.get('chunk_id', '')}_"
                f"{content}"
            )

            document_id = hashlib.sha256(
                unique_value.encode("utf-8")
            ).hexdigest()

            ids.append(document_id)

        return ids

    def add_documents(self, documents):
        if not documents:
            raise ValueError("No document chunks provided.")

        ids = self._generate_ids(documents)

        self.vector_store.add_documents(
            documents=documents,
            ids=ids
        )

        return {
            "documents_added": len(documents),
            "collection_name": self.collection_name
        }

    def search(self, query, k=3):
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        results = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )

        return [
            {
                "content": document.page_content,
                "metadata": document.metadata,
                "distance": float(score)
            }
            for document, score in results
        ]

    def get_collection_count(self):
        return self.vector_store._collection.count()
