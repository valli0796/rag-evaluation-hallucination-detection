
from langchain_ollama import ChatOllama
from src.vector_store import VectorStoreManager


class RAGPipeline:
    def __init__(
        self,
        model_name="llama3.2:3b",
        k=3,
        vector_store_manager=None
    ):
        self.k = k

        self.vector_store = (
            vector_store_manager
            if vector_store_manager is not None
            else VectorStoreManager()
        )

        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
            num_predict=300
        )

    def retrieve_context(self, question):
        results = self.vector_store.search(
            query=question,
            k=self.k
        )

        return results

    def generate_answer(self, question, retrieved_documents):
        if not retrieved_documents:
            return (
                "I could not find relevant information "
                "in the documents."
            )

        context = "\n\n".join(
            item["content"]
            for item in retrieved_documents
        )

        prompt = f"""
You are a document-based question answering assistant.

Answer the question using ONLY the context provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not available in the context, clearly say:
   "The provided documents do not contain this information."
4. Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content

    def ask(self, question):
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        retrieved_documents = self.retrieve_context(question)

        answer = self.generate_answer(
            question,
            retrieved_documents
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_documents
        }
