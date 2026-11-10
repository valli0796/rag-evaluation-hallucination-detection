
import csv
import time
from pathlib import Path

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.evaluator import RAGEvaluator


class RAGExperimentRunner:
    def __init__(self, pdf_path="data/documents/sample.pdf"):
        self.pdf_path = Path(pdf_path)
        self.report_path = Path("reports/experiment_results.csv")
        self.evaluator = RAGEvaluator()

        self.questions = [
            "What is Retrieval-Augmented Generation?",
            "How does a RAG system work?",
            "What are the benefits of RAG?",
            "What challenges can affect RAG performance?",
            "What is faithfulness in RAG evaluation?"
        ]

        self.experiments = [
            {
                "name": "Experiment_A",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "top_k": 2
            },
            {
                "name": "Experiment_B",
                "chunk_size": 800,
                "chunk_overlap": 120,
                "top_k": 2
            }
        ]

    def run_experiment(self, config):
        print("\n" + "=" * 55)
        print("Running:", config["name"])
        print("Chunk size:", config["chunk_size"])

        processor = DocumentProcessor(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )

        processed = processor.process_pdf(self.pdf_path)

        collection_name = (
            f"experiment_chunk_{config['chunk_size']}"
        )

        vector_store = VectorStoreManager(
            persist_directory="chroma_db/experiments",
            collection_name=collection_name
        )

        vector_store.add_documents(processed["chunks"])

        pipeline = RAGPipeline(
            k=config["top_k"],
            vector_store_manager=vector_store
        )

        experiment_results = []

        for question_number, question in enumerate(
            self.questions, start=1
        ):
            print(
                f"\nQuestion {question_number}/{len(self.questions)}:"
            )
            print(question)

            start_time = time.perf_counter()
            result = pipeline.ask(question)
            query_time = time.perf_counter() - start_time

            evaluation = self.evaluator.evaluate_faithfulness(
                question=question,
                answer=result["answer"],
                retrieved_documents=result["retrieved_documents"]
            )

            distances = [
                item["distance"]
                for item in result["retrieved_documents"]
            ]

            average_distance = (
                sum(distances) / len(distances)
                if distances else 0
            )

            row = {
                "experiment": config["name"],
                "question_number": question_number,
                "question": question,
                "chunk_size": config["chunk_size"],
                "chunk_overlap": config["chunk_overlap"],
                "top_k": config["top_k"],
                "total_chunks": processed["total_chunks"],
                "retrieved_chunks": len(
                    result["retrieved_documents"]
                ),
                "query_time_seconds": round(query_time, 2),
                "average_retrieval_distance": round(
                    average_distance, 4
                ),
                "total_statements": evaluation["total_statements"],
                "supported_statements": evaluation["supported_count"],
                "unsupported_statements": evaluation["unsupported_count"],
                "evidence_support_percentage": (
                    evaluation["faithfulness_percentage"]
                )
            }

            experiment_results.append(row)

            print("Generated answer:")
            print(result["answer"])

            print(
                "Evidence support:",
                row["evidence_support_percentage"],
                "%"
            )
            print("Query time:", row["query_time_seconds"], "seconds")

        return experiment_results

    def save_results(self, results):
        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not results:
            raise ValueError("No experiment results to save.")

        with open(
            self.report_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=results[0].keys()
            )

            writer.writeheader()
            writer.writerows(results)

        print("\nCSV report saved to:", self.report_path)

    def run_all(self):
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.pdf_path}"
            )

        all_results = []

        for config in self.experiments:
            results = self.run_experiment(config)
            all_results.extend(results)

        self.save_results(all_results)

        print("\nEXPERIMENT SUMMARY")
        print("=" * 55)

        for config in self.experiments:
            rows = [
                row for row in all_results
                if row["experiment"] == config["name"]
            ]

            average_support = sum(
                row["evidence_support_percentage"]
                for row in rows
            ) / len(rows)

            average_time = sum(
                row["query_time_seconds"]
                for row in rows
            ) / len(rows)

            print("\n", config["name"])
            print("Questions evaluated:", len(rows))
            print(
                "Average evidence support:",
                round(average_support, 2), "%"
            )
            print(
                "Average query time:",
                round(average_time, 2), "seconds"
            )

        return all_results


if __name__ == "__main__":
    runner = RAGExperimentRunner()
    runner.run_all()

    print("\nAll experiments completed.")
